// src/nemotron/nemo_serve.cuh - strata-nemo --serve: the Strata engine protocol (serve/server.py's StrataEngine) on
// stdin / stdout, with sampling, conversation reuse and image embeddings.
//
//   -> READY <max_context> stop
//   <- GEN <max_new> [key=value ...] <id,id,...>          (GENI <max_new> [key=value ...] <embeddings file> <ids>)
//   -> RESUME <n>, PP <position> <prompt tokens> <ms> <tok/s> after each prompt chunk, T <id> per token,
//      DONE <generated> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> <drafts accepted> <drafts offered>
//           <reused> 0 0 0 0 0 <prompt tokens read> 0
//   <- STOP (ends the running request), QUIT
//
// Keys: temperature, top_p, top_k (0 = off), min_p, seed; absent temperature (or 0) is greedy. Other keys are read and
// ignored. A request continues from the live sequence when its prompt starts with it, else from the checkpoint taken at
// the end of the last prompt when its prompt starts with that, else from an empty sequence.
#pragma once
#include <poll.h>
#include <unistd.h>

// ---------------------------------------------------------------------------------------------------- sampling
struct SampleParams {
    float inv_temp = 1.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0;
    unsigned long long seed = 0;
};
__device__ __forceinline__ uint64_t sp_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float sp_uniform(uint64_t seed, uint64_t step, int row, int what) {
    const uint64_t h = sp_mix(seed ^ sp_mix(step * 0x100000001b3ull + uint64_t(row) * 4 + what));
    return float(h >> 40) * (1.f / 16777216.f);                    // [0, 1)
}
constexpr int kSpChunk = (kVocab + 1023) / 1024;                    // values a thread (contiguous)
// a block sum / max over 1024 threads, the result for every thread (sh: 33 floats)
__device__ __forceinline__ float sp_bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = sh[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o, 32);
        if (threadIdx.x == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}
__device__ __forceinline__ float sp_bmax(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = sh[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o, 32));
        if (threadIdx.x == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}
// Sampling with a drafter that proposes one token a position (rows 0..n of logits: n drafts, then the bonus row).
// Each row's distribution: temperature, then the tokens above one cut (the largest of: top-k's k-th value, the
// smallest set reaching top_p of the mass, min_p of the best). Draft r is accepted with its probability under row r;
// at the first rejection the token is drawn from row r with the draft taken out, else from row n. With a point-mass
// drafter this is exact speculative sampling: the tokens have the distribution plain sampling would give them.
// out[0] = drafts accepted, out[1] = the next token.
__global__ void __launch_bounds__(1024) k_sample_spec(const float* logits, int n, const int32_t* drafts, SampleParams sp,
                                                      unsigned long long step, int32_t* out) {
    __shared__ float sh[33];
    __shared__ float part[1024];
    __shared__ int pick;
    __shared__ float base, tgt;
    const int tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocab, j0 + kSpChunk);
    for (int r = 0; r <= n; ++r) {
        const float* l = logits + size_t(r) * kVocab;
        float mx = -INFINITY;
        for (int j = j0; j < j1; ++j) mx = fmaxf(mx, l[j] * sp.inv_temp);
        const float M = sp_bmax(mx, sh);
        // x_j = z_j - M <= 0; a token is kept when x_j >= s
        auto mass = [&](float s, float& cnt) {
            float m = 0.f, c = 0.f;
            for (int j = j0; j < j1; ++j) { const float x = l[j] * sp.inv_temp - M; if (x >= s) { m += __expf(x); c += 1.f; } }
            const float tm = sp_bsum(m, sh);
            cnt = sp_bsum(c, sh);
            return tm;
        };
        float cnt = 0.f;
        const float Z = mass(-INFINITY, cnt);
        float s = sp.min_p > 0.f ? __logf(sp.min_p) : -INFINITY;
        if (sp.top_k > 0 && sp.top_k < kVocab) {                    // the largest s with at least k tokens kept
            float lo = -90.f, hi = 0.f;
            for (int it = 0; it < 26; ++it) {
                const float mid = 0.5f * (lo + hi);
                mass(mid, cnt);
                if (cnt >= float(sp.top_k)) lo = mid; else hi = mid;
            }
            s = fmaxf(s, lo);
        }
        if (sp.top_p < 1.f) {                                       // the largest s keeping at least top_p of the mass
            float lo = -90.f, hi = 0.f;
            for (int it = 0; it < 26; ++it) {
                const float mid = 0.5f * (lo + hi);
                if (mass(mid, cnt) >= sp.top_p * Z) lo = mid; else hi = mid;
            }
            s = fmaxf(s, lo);
        }
        const float K = mass(s, cnt);                               // the kept mass
        int skip = -1;
        float target;
        if (r < n) {
            const int d = drafts[r];
            const float xd = l[d] * sp.inv_temp - M;
            const float pd = xd >= s ? __expf(xd) / K : 0.f;
            if (sp_uniform(sp.seed, step, r, 0) < pd) continue;    // accepted: the next row
            skip = d;
            target = sp_uniform(sp.seed, step, r, 1) * (K - (xd >= s ? __expf(xd) : 0.f));
        } else {
            target = sp_uniform(sp.seed, step, r, 1) * K;
        }
        // the token: this thread's kept mass, the running sums, then a walk inside the chunk that holds the target
        // (rounding can put the target past the last kept token: then it is the last kept token)
        float m = 0.f;
        for (int j = j0; j < j1; ++j) { const float x = l[j] * sp.inv_temp - M; if (x >= s && j != skip) m += __expf(x); }
        part[tid] = m;
        __syncthreads();
        if (tid == 0) {
            float acc = 0.f, before = 0.f;
            int t = -1, lastnz = 0;
            for (int u = 0; u < 1024; ++u) {
                if (part[u] <= 0.f) continue;
                lastnz = u;
                if (t < 0 && acc + part[u] > target) { t = u; before = acc; }
                acc += part[u];
            }
            if (t < 0) { t = lastnz; before = acc - part[lastnz]; tgt = 1e30f; } else tgt = target;
            base = before;
            pick = t;
        }
        __syncthreads();
        if (tid == pick) {
            float acc = base;
            int best = -1;
            for (int j = j0; j < j1; ++j) {
                const float x = l[j] * sp.inv_temp - M;
                if (x < s || j == skip) continue;
                best = j;
                acc += __expf(x);
                if (acc > tgt) break;
            }
            out[0] = r;
            out[1] = best;
        }
        return;
    }
}

// ---------------------------------------------------------------------------------------------------- the server
// stdin lines, blocking or not (a STOP may arrive while a request runs)
struct LineReader {
    std::string buf;
    bool eof = false;
    bool fill(int timeout_ms) {
        pollfd p{0, POLLIN, 0};
        if (poll(&p, 1, timeout_ms) <= 0) return false;
        char tmp[65536];
        const ssize_t n = read(0, tmp, sizeof tmp);
        if (n <= 0) { eof = true; return false; }
        buf.append(tmp, size_t(n));
        return true;
    }
    bool take(std::string& line) {
        const size_t e = buf.find('\n');
        if (e == std::string::npos) return false;
        line = buf.substr(0, e);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        buf.erase(0, e + 1);
        return true;
    }
    bool get(std::string& line) {                                  // blocking; false at the end of input
        while (!take(line)) { if (eof) return false; fill(-1); if (eof && buf.find('\n') == std::string::npos) return false; }
        return true;
    }
    bool poll_line(std::string& line) { if (take(line)) return true; fill(0); return take(line); }
};

// one conversation point: every Mamba state, the drafter's context rings and anchor, the last position's logits
struct Snap {
    std::vector<float*> ssm, conv;
    float *dk[kDLayers] = {}, *dv[kDLayers] = {}, *anchor = nullptr;
    std::vector<int> toks;
    std::vector<float> logits;
    bool ok = false;
};
static void snap_copy(Model& m, Drafter* D, Snap& s, bool save) {
    const size_t ns = size_t(kMHeads) * kMHeadDim * kMState * 4, nc = size_t(kConvDim) * (kConvK - 1) * 4;
    const size_t nr = sizeof(float) * kAKv * size_t(kDRing) * 128;
    size_t k = 0;
    for (auto& b : m.blocks) {
        if (b.kind != MAMBA) continue;
        if (s.ssm.size() <= k) {
            s.ssm.push_back(nullptr); s.conv.push_back(nullptr);
            CK(cudaMalloc(&s.ssm[k], ns)); CK(cudaMalloc(&s.conv[k], nc));
        }
        if (save) { CK(cudaMemcpyAsync(s.ssm[k], b.ssm, ns, cudaMemcpyDeviceToDevice, m.stream)); CK(cudaMemcpyAsync(s.conv[k], b.conv_state, nc, cudaMemcpyDeviceToDevice, m.stream)); }
        else { CK(cudaMemcpyAsync(b.ssm, s.ssm[k], ns, cudaMemcpyDeviceToDevice, m.stream)); CK(cudaMemcpyAsync(b.conv_state, s.conv[k], nc, cudaMemcpyDeviceToDevice, m.stream)); }
        ++k;
    }
    if (D) {
        for (int l = 0; l < kDLayers; ++l) {
            if (!s.dk[l]) { CK(cudaMalloc(&s.dk[l], nr)); CK(cudaMalloc(&s.dv[l], nr)); }
            if (save) { CK(cudaMemcpyAsync(s.dk[l], D->L[l].K, nr, cudaMemcpyDeviceToDevice, m.stream)); CK(cudaMemcpyAsync(s.dv[l], D->L[l].V, nr, cudaMemcpyDeviceToDevice, m.stream)); }
            else { CK(cudaMemcpyAsync(D->L[l].K, s.dk[l], nr, cudaMemcpyDeviceToDevice, m.stream)); CK(cudaMemcpyAsync(D->L[l].V, s.dv[l], nr, cudaMemcpyDeviceToDevice, m.stream)); }
        }
        if (!s.anchor) CK(cudaMalloc(&s.anchor, kHidden * 4));
        if (save) CK(cudaMemcpyAsync(s.anchor, D->anchor, kHidden * 4, cudaMemcpyDeviceToDevice, m.stream));
        else CK(cudaMemcpyAsync(D->anchor, s.anchor, kHidden * 4, cudaMemcpyDeviceToDevice, m.stream));
    }
    CK(cudaStreamSynchronize(m.stream));
}

// a small step with the logits left on the card (all_logits: every row's)
static void forward_dev(Model& m, const std::vector<int>& toks, int write_after, bool all_logits) {
    const int T = int(toks.size());
    for (int t = 0; t < T; ++t) m.h_tp[t] = toks[t];
    m.h_tp[kTMax] = m.pos;
    CK(cudaMemcpyAsync(m.d_toks, m.h_tp, (kTMax + 1) * sizeof(int), cudaMemcpyHostToDevice, m.stream));
    if (!m.use_graph) { m.body(T, write_after, all_logits, nullptr); return; }
    const int key = T * 4 + (write_after == T) * 2 + (all_logits ? 1 : 0);
    auto it = m.graphs.find(key);
    if (it == m.graphs.end()) {
        cudaGraph_t g; cudaGraphExec_t e;
        CK(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeThreadLocal));
        m.body(T, write_after, all_logits, nullptr);
        CK(cudaStreamEndCapture(m.stream, &g));
        CK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        CK(cudaGraphDestroy(g));
        it = m.graphs.emplace(key, e).first;
    }
    CK(cudaGraphLaunch(it->second, m.stream));
}

// strata-vision's records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, one record per image
static bool read_embeddings(const std::string& path, std::vector<float>& rows, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    rows.clear();
    for (;;) {
        int32_t h[5];
        const size_t got = std::fread(h, 4, 5, f);
        if (got == 0) break;
        if (got != 5 || h[0] != 0x31455653 || h[4] != kHidden || h[1] < 0) { err = "not an embeddings file for this model: " + path; std::fclose(f); return false; }
        const size_t at = rows.size();
        rows.resize(at + size_t(h[1]) * kHidden);
        if (std::fread(rows.data() + at, 4, size_t(h[1]) * kHidden, f) != size_t(h[1]) * kHidden) { err = "short embeddings file"; std::fclose(f); return false; }
    }
    std::fclose(f);
    return true;
}

constexpr int kImageToken = 18;                                     // <image>, the model's img_context_token_id
constexpr int kImStart = 10;                                        // <|im_start|>

static int serve(Model& m, Drafter* D, const std::vector<int>& eos_ids) {
    LineReader in;
    Snap ck;
    std::vector<int> live;                                         // the tokens whose state the model holds
    std::vector<float> img_rows;
    float* d_rows = nullptr; size_t d_rows_cap = 0;
    int32_t* d_map = nullptr;
    CK(cudaMalloc(&d_map, size_t(kTBig) * 4));
    int32_t* d_drafts = nullptr; int32_t* d_out = nullptr;
    CK(cudaMalloc(&d_drafts, 16 * 4)); CK(cudaMalloc(&d_out, 2 * 4));
    if (D) m.warm_checks(kDSlots + 1);
    unsigned long long step_no = 0;
    std::printf("INFO engine=strata-nemo drafter=%d\n", D ? 1 : 0);
    std::printf("READY %d stop\n", m.max_ctx);
    std::fflush(stdout);
    auto is_eos = [&](int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); };
    std::string line;
    while (in.get(line)) {
        if (line == "QUIT") break;
        if (line == "STOP" || line.empty()) continue;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n"); std::fflush(stdout); continue; }
        // GEN <max_new> [key=value ...] [embeddings file] <ids>
        std::vector<std::string> f;
        { std::stringstream ss(line); for (std::string w; ss >> w;) f.push_back(w); }
        if (f.size() < 3) { std::printf("ERR a request needs max_new and ids\n"); std::fflush(stdout); continue; }
        const int max_new = std::max(1, std::atoi(f[1].c_str()));
        float temperature = 0.f;
        SampleParams sp;
        bool seeded = false;
        std::string emb_file;
        size_t w = 2;
        for (; w + 1 < f.size(); ++w) {
            const size_t eq = f[w].find('=');
            if (eq == std::string::npos) break;
            const std::string k = f[w].substr(0, eq), v = f[w].substr(eq + 1);
            if (k == "temperature") temperature = std::strtof(v.c_str(), nullptr);
            else if (k == "top_p") sp.top_p = std::strtof(v.c_str(), nullptr);
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = std::strtof(v.c_str(), nullptr);
            else if (k == "seed") { sp.seed = std::strtoull(v.c_str(), nullptr, 10); seeded = true; }
        }
        if (geni) emb_file = f[w++];
        std::vector<int> ids;
        { std::stringstream ss(f.back()); for (std::string v; std::getline(ss, v, ',');) if (!v.empty()) ids.push_back(std::atoi(v.c_str())); }
        if (ids.empty() || w + 1 != f.size()) { std::printf("ERR the ids are missing\n"); std::fflush(stdout); continue; }
        if (int(ids.size()) + 2 > m.max_ctx) { std::printf("ERR the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), m.max_ctx); std::fflush(stdout); continue; }
        const bool sample = temperature > 0.f;
        sp.inv_temp = sample ? 1.f / temperature : 1.f;
        if (!seeded) sp.seed = (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
        step_no = 0;                                               // (a seed gives the same answer every time)
        // images: one row per <image> token, in order
        std::vector<int> map_all(ids.size(), -1);
        int n_img = 0;
        if (geni) {
            std::string err;
            if (!read_embeddings(emb_file, img_rows, err)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); continue; }
            for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == kImageToken) map_all[i] = n_img++;
            if (size_t(n_img) * kHidden != img_rows.size()) {
                std::printf("ERR %d image tokens in the prompt, %zu embedding rows in the file\n", n_img, img_rows.size() / kHidden);
                std::fflush(stdout);
                continue;
            }
            if (img_rows.size() > d_rows_cap) { if (d_rows) CK(cudaFree(d_rows)); d_rows_cap = img_rows.size(); CK(cudaMalloc(&d_rows, d_rows_cap * 4)); }
            CK(cudaMemcpy(d_rows, img_rows.data(), img_rows.size() * 4, cudaMemcpyHostToDevice));
        }
        const auto t0 = std::chrono::steady_clock::now();
        // where to start: the live sequence, the checkpoint, or nothing (images: never mid-image, so the same rule)
        auto starts_with = [&](const std::vector<int>& a, const std::vector<int>& p) {
            return p.size() <= a.size() && std::equal(p.begin(), p.end(), a.begin());
        };
        int reuse = 0;
        bool from_ck_logits = false;
        if (!geni && !live.empty() && ids.size() > live.size() && starts_with(ids, live)) reuse = int(live.size());
        else if (!geni && ck.ok && starts_with(ids, ck.toks)) {
            snap_copy(m, D, ck, false);
            m.pos = int(ck.toks.size());
            live = ck.toks;
            reuse = m.pos;
            from_ck_logits = ids.size() == ck.toks.size();
            if (from_ck_logits && ck.logits.empty()) { m.reset(); live.clear(); reuse = 0; from_ck_logits = false; }
        } else {
            m.reset();
            live.clear();
        }
        if (reuse == 0 && m.pos != 0) m.reset();
        std::printf("RESUME %d\n", reuse);
        std::fflush(stdout);
        // the rest of the prompt, in chunks; the last chunk's logits stay on the card
        bool cancelled = false;
        int read_n = 0;
        if (from_ck_logits) CK(cudaMemcpy(m.logits, ck.logits.data(), kVocab * 4, cudaMemcpyHostToDevice));
        // the checkpoint goes in front of the last <|im_start|> (the answer's turn): the next request renders this
        // answer differently (its thinking dropped), so that is the last token the two prompts share
        size_t cut = 0;
        for (size_t i = ids.size(); i-- > size_t(reuse) + 1;) if (ids[i] == kImStart) { cut = i; break; }
        bool saved = false;
        auto save_ck = [&](size_t upto, bool with_logits) {
            snap_copy(m, D, ck, true);
            ck.toks.assign(ids.begin(), ids.begin() + upto);
            ck.logits.resize(with_logits ? kVocab : 0);
            if (with_logits) CK(cudaMemcpy(ck.logits.data(), m.logits, kVocab * 4, cudaMemcpyDeviceToHost));
            ck.ok = true;
            saved = true;
        };
        for (size_t i = size_t(reuse); i < ids.size();) {
            if (!geni && cut && i == cut && !saved) save_ck(cut, false);
            const size_t e = std::min({ids.size(), i + size_t(kTBig), (!geni && cut > i) ? cut : ids.size()});
            const std::vector<int> c(ids.begin() + i, ids.begin() + e);
            const int p0 = m.pos;
            bool has_img = false;
            for (size_t t = i; t < e; ++t) has_img |= map_all[t] >= 0;
            if (c.size() > size_t(kTMax) || has_img) {
                if (has_img) {
                    CK(cudaMemcpy(d_map, map_all.data() + i, c.size() * 4, cudaMemcpyHostToDevice));
                    m.inj_rows = d_rows; m.inj_map = d_map;
                }
                m.forward_big(c);
                m.inj_rows = nullptr; m.inj_map = nullptr;
                if (D) D->add_context(m.aux_big, int(c.size()), p0);
            } else {
                forward_dev(m, c, int(c.size()), false);
                if (D) D->add_context(m.aux, int(c.size()), p0);
            }
            m.pos += int(c.size());
            read_n += int(c.size());
            i = e;
            CK(cudaStreamSynchronize(m.stream));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("PP %d %zu %.1f %.1f\n", m.pos, ids.size(), ms, read_n / std::max(ms, 1e-3) * 1e3);
            std::fflush(stdout);
            std::string l2;
            if (i < ids.size() && in.poll_line(l2) && l2 == "STOP") { cancelled = true; break; }
        }
        const double prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        live.assign(ids.begin(), ids.begin() + m.pos);
        if (cancelled) {
            ck.ok = false;
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %d 0 0 0 0 0 %d 0\n", ids.size(), prompt_ms, reuse, read_n);
            std::fflush(stdout);
            continue;
        }
        // (a text prompt only: an image prompt's ids alone do not name its state; a prompt with no turn to cut at is
        // checkpointed at its end)
        if (geni) ck.ok = false;
        else if (!saved) save_ck(ids.size(), true);
        // writing
        const auto t1 = std::chrono::steady_clock::now();
        int produced = 0, drafts_ok = 0, drafts_offered = 0;
        const char* finish = nullptr;
        auto emit = [&](int t) {
            std::printf("T %d\n", t);
            ++produced;
            if (is_eos(t)) finish = "stop";
            else if (produced >= max_new) finish = "length";
            else if (m.pos + 2 + kTMax > m.max_ctx) finish = "length";
        };
        auto next_token = [&](int rows, int n_drafts, const std::vector<int>& drafts) -> std::pair<int, int> {
            if (!sample) {
                k_argmax_rows<<<rows, 1024, 0, m.stream>>>(m.logits, kVocab, m.best_dev);
                std::vector<int> best(rows);
                CK(cudaMemcpyAsync(best.data(), m.best_dev, rows * 4, cudaMemcpyDeviceToHost, m.stream));
                CK(cudaStreamSynchronize(m.stream));
                int a = 0;
                while (a < n_drafts && best[a] == drafts[a]) ++a;
                return {a, best[a]};
            }
            if (n_drafts) CK(cudaMemcpyAsync(d_drafts, drafts.data(), n_drafts * 4, cudaMemcpyHostToDevice, m.stream));
            k_sample_spec<<<1, 1024, 0, m.stream>>>(m.logits, n_drafts, d_drafts, sp, step_no++, d_out);
            int o2[2];
            CK(cudaMemcpyAsync(o2, d_out, 8, cudaMemcpyDeviceToHost, m.stream));
            CK(cudaStreamSynchronize(m.stream));
            return {o2[0], o2[1]};
        };
        int tok = next_token(1, 0, {}).second;
        emit(tok);
        std::fflush(stdout);
        while (!finish) {
            std::string l2;
            if (in.poll_line(l2) && l2 == "STOP") { finish = "cancel"; break; }
            if (!D) {
                forward_dev(m, {tok}, 1, false);
                m.pos += 1;
                live.push_back(tok);
                tok = next_token(1, 0, {}).second;
                emit(tok);
            } else {
                std::vector<int> dr = D->draft(tok, m.pos);
                const int room = std::max(0, std::min({kDSlots, max_new - produced - 1, m.max_ctx - m.pos - 2 - kTMax}));
                if ((int) dr.size() > room) dr.resize(room);
                std::vector<int> win = {tok};
                win.insert(win.end(), dr.begin(), dr.end());
                forward_dev(m, win, 0, true);
                const auto [a, nb] = next_token(int(win.size()), int(dr.size()), dr);
                const int p0 = m.pos;
                m.commit(a + 1);
                D->add_context(m.aux, a + 1, p0);
                for (int j = 0; j <= a; ++j) live.push_back(win[j]);
                drafts_offered += int(dr.size());
                drafts_ok += a;
                for (int j = 0; j < a && !finish; ++j) emit(dr[j]);
                if (!finish) emit(nb);
                tok = nb;
            }
            std::fflush(stdout);
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("DONE %d %zu %.1f %.1f %s %d %d %d 0 0 0 0 0 %d 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    drafts_ok, drafts_offered, reuse, read_n);
        std::fflush(stdout);
    }
    return 0;
}
