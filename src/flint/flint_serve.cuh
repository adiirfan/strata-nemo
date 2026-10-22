// src/flint/flint_serve.cuh - strata-flint --serve: the Strata engine protocol (serve/server.py's StrataEngine) on
// stdin / stdout, with sampling and conversation reuse.
//
//   -> READY <max_context> stop
//   <- GEN <max_new> [key=value ...] <id,id,...>
//   -> RESUME <n>, PP <position> <prompt tokens> <ms> <tok/s> after each prompt chunk, T <id> per token,
//      DONE <generated> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> 0 0 0 0 0 <read> 0
//   <- STOP (ends the running request), QUIT
//
// Keys: temperature, top_p, top_k (0 = off), min_p, seed; absent temperature (or 0) is greedy. Other keys are read and
// ignored. Flint keeps no running state besides its K/V cache, so a request continues from the longest prefix it shares
// with the tokens the cache holds (the sliding layers' ring allowing).
#pragma once
#include <poll.h>
#include <unistd.h>

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
__device__ __forceinline__ float sp_uniform(uint64_t seed, uint64_t step, int what) {
    const uint64_t h = sp_mix(seed ^ sp_mix(step * 0x100000001b3ull + what));
    return float(h >> 40) * (1.f / 16777216.f);
}
constexpr int kVocabF = 150144, kSpChunk = (kVocabF + 1023) / 1024;
__device__ __forceinline__ float sp_bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += fl_shfl_xor(v, o, 32);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = sh[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t += fl_shfl_xor(t, o, 32);
        if (threadIdx.x == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}
__device__ __forceinline__ float sp_bmax(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, fl_shfl_xor(v, o, 32));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = sh[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, fl_shfl_xor(t, o, 32));
        if (threadIdx.x == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}
// one token from the logits: temperature, then the tokens above one cut (the largest of: top-k's k-th value, the
// smallest set reaching top_p of the mass, min_p of the best), drawn by inverse CDF; into state[1]
__global__ void __launch_bounds__(1024) k_sample(const float* l, SampleParams sp, unsigned long long step, int* state) {
    __shared__ float sh[33];
    __shared__ float part[1024];
    __shared__ int pick;
    __shared__ float base;
    const int tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocabF, j0 + kSpChunk);
    float mx = -INFINITY;
    for (int j = j0; j < j1; ++j) mx = fmaxf(mx, l[j] * sp.inv_temp);
    const float M = sp_bmax(mx, sh);
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
    if (sp.top_k > 0 && sp.top_k < kVocabF) {
        float lo = -90.f, hi = 0.f;
        for (int it = 0; it < 26; ++it) {
            const float mid = 0.5f * (lo + hi);
            mass(mid, cnt);
            if (cnt >= float(sp.top_k)) lo = mid; else hi = mid;
        }
        s = fmaxf(s, lo);
    }
    if (sp.top_p < 1.f) {
        float lo = -90.f, hi = 0.f;
        for (int it = 0; it < 26; ++it) {
            const float mid = 0.5f * (lo + hi);
            if (mass(mid, cnt) >= sp.top_p * Z) lo = mid; else hi = mid;
        }
        s = fmaxf(s, lo);
    }
    const float K = mass(s, cnt);
    const float target = sp_uniform(sp.seed, step, 1) * K;
    float m = 0.f;
    for (int j = j0; j < j1; ++j) { const float x = l[j] * sp.inv_temp - M; if (x >= s) m += __expf(x); }
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
        if (t < 0) { t = lastnz; before = acc - part[lastnz]; }
        base = before;
        pick = t;
    }
    __syncthreads();
    if (tid == pick) {
        float acc = base;
        int best = -1;
        for (int j = j0; j < j1; ++j) {
            const float x = l[j] * sp.inv_temp - M;
            if (x < s) continue;
            best = j;
            acc += __expf(x);
            if (acc > target) break;
        }
        state[1] = best;
    }
}

// The common case, top_k in 1..64: the k best logits are found once (each thread keeps its own best k, then k rounds
// of a block maximum over the threads' heads), and temperature, top_p and min_p act on those k only. The same cut and
// draw as k_sample, without its bisection passes over the whole vocabulary (about 1.6 ms a token on the R9700).
constexpr int kTopKMax = 64;
// the k best of a row into cv / ci (sorted, best first), the whole block of 1024
__device__ void sp_topk(const float* l, int k, float* cv, int* ci) {
    __shared__ float wv[32];
    __shared__ int wi[32], wt[32];
    const int tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocabF, j0 + kSpChunk);
    // this thread's best k, sorted (k <= 64, kSpChunk values each)
    float bv[kTopKMax];
    int bi[kTopKMax];
    int nb = 0;
    for (int j = j0; j < j1; ++j) {
        const float v = l[j];
        if (nb == k && v <= bv[k - 1]) continue;
        int p = nb < k ? nb++ : k - 1;
        while (p > 0 && bv[p - 1] < v) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; --p; }
        bv[p] = v; bi[p] = j;
    }
    int head = 0;
    for (int r = 0; r < k; ++r) {
        float v = head < nb ? bv[head] : -INFINITY;
        int i = head < nb ? bi[head] : 0x7fffffff, t = tid;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = fl_shfl_xor(v, o, 32);
            const int oi = fl_shfl_xor(i, o, 32), ot = fl_shfl_xor(t, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; t = ot; }
        }
        if ((tid & 31) == 0) { wv[tid >> 5] = v; wi[tid >> 5] = i; wt[tid >> 5] = t; }
        __syncthreads();
        if (tid == 0) {
            float b = wv[0]; int bi_ = wi[0], bt = wt[0];
            for (int w = 1; w < 32; ++w) if (wv[w] > b || (wv[w] == b && wi[w] < bi_)) { b = wv[w]; bi_ = wi[w]; bt = wt[w]; }
            cv[r] = b; ci[r] = bi_; wt[0] = bt;
        }
        __syncthreads();
        if (tid == wt[0]) ++head;
        __syncthreads();
    }
}
// the kept tokens of a sorted top-k row: probabilities at the temperature, and how many survive top_p / min_p
__device__ inline int sp_keep(const float* cv, int k, const SampleParams& sp, float* p, float& kept) {
    float z = 0.f;
    for (int r = 0; r < k; ++r) { p[r] = cv[r] == -INFINITY ? 0.f : __expf((cv[r] - cv[0]) * sp.inv_temp); z += p[r]; }
    int keep = k;
    float acc = 0.f;
    for (int r = 0; r < k; ++r) {
        acc += p[r];
        if (sp.top_p < 1.f && acc >= sp.top_p * z) { keep = r + 1; break; }
    }
    if (sp.min_p > 0.f) for (int r = 1; r < keep; ++r) if (p[r] < sp.min_p * p[0]) { keep = r; break; }
    kept = 0.f;
    for (int r = 0; r < keep; ++r) kept += p[r];
    return keep;
}
__global__ void __launch_bounds__(1024) k_sample_topk(const float* l, SampleParams sp, unsigned long long step, int* state) {
    __shared__ float cv[kTopKMax];
    __shared__ int ci[kTopKMax];
    const int k = sp.top_k;
    sp_topk(l, k, cv, ci);
    if (threadIdx.x != 0) return;
    // cv is sorted, best first: softmax at the temperature, then the top_p and min_p cuts
    float p[kTopKMax], kept;
    const int keep = sp_keep(cv, k, sp, p, kept);
    const float target = sp_uniform(sp.seed, step, 1) * kept;
    float run = 0.f;
    int pick = ci[keep - 1];
    for (int r = 0; r < keep; ++r) { run += p[r]; if (run > target) { pick = ci[r]; break; } }
    state[1] = pick;
}

// Drafted tokens against the checking pass's rows (row r: the distribution after the r-th checked token): draft r is
// kept with the probability the sampler gives it; the first one refused is replaced by a draw from the kept tokens
// without it (so every token is still distributed exactly as plain sampling would give), and if all are kept the last
// row draws one more. out[0] = drafts kept, out[1] = the token after them. top_k in 1..64 only.
__global__ void __launch_bounds__(1024) k_spec_pick(const float* LG, int nd, const int* drafts, SampleParams sp,
                                                     unsigned long long step, int* out) {
    __shared__ float cv[kTopKMax];
    __shared__ int ci[kTopKMax];
    __shared__ int done;
    for (int r = 0; r <= nd; ++r) {
        sp_topk(LG + size_t(r) * kVocabF, sp.top_k, cv, ci);
        if (threadIdx.x == 0) {
            float p[kTopKMax], kept;
            const int keep = sp_keep(cv, sp.top_k, sp, p, kept);
            const int d = r < nd ? drafts[r] : -1;
            float pd = 0.f;
            int rd = -1;
            for (int u = 0; u < keep; ++u) if (ci[u] == d) { pd = p[u]; rd = u; }
            done = 0;
            if (rd >= 0 && sp_uniform(sp.seed, step + r, 2) * kept < pd) done = 0;   // kept: on to the next row
            else {
                const float rest = kept - pd, target = sp_uniform(sp.seed, step + r, 1) * rest;
                float run = 0.f;
                int pick = -1;
                for (int u = 0; u < keep; ++u) {
                    if (u == rd) continue;
                    pick = ci[u];
                    run += p[u];
                    if (run > target) break;
                }
                out[0] = r;
                out[1] = pick;
                done = 1;
            }
        }
        __syncthreads();
        if (done) return;
        __syncthreads();
    }
}

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
    bool get(std::string& line) {
        while (!take(line)) { if (eof) return false; fill(-1); if (eof && buf.find('\n') == std::string::npos) return false; }
        return true;
    }
    bool poll_line(std::string& line) { if (take(line)) return true; fill(0); return take(line); }
};

// Word-copy guessing: the latest earlier place the last 3 (else 2) tokens appeared, and the tokens that followed it
struct NgramIndex {
    std::unordered_map<uint64_t, int> g3, g2;   // n-gram -> the position just after its latest occurrence
    size_t upto = 0;                             // seq[0 .. upto) indexed
    static uint64_t key(const int* t, int n) { uint64_t h = uint64_t(n); for (int i = 0; i < n; ++i) h = h * 0x9E3779B97F4A7C15ull ^ uint64_t(uint32_t(t[i])); return h; }
    void reset() { g3.clear(); g2.clear(); upto = 0; }
    // index the n-grams ending before the last token of seq (a match must have a token after it)
    void add(const std::vector<int>& seq) {
        for (; upto + 1 < seq.size(); ++upto) {
            const size_t e = upto + 1;   // n-grams ending at upto, followed by seq[e]
            if (e >= 2) g2[key(&seq[e - 2], 2)] = int(e);
            if (e >= 3) g3[key(&seq[e - 3], 3)] = int(e);
        }
    }
    // up to kmax drafted tokens following the latest earlier match of seq's tail; returns the n-gram length used (0: none)
    int draft(const std::vector<int>& seq, int kmax, std::vector<int>& out) const {
        out.clear();
        const size_t n = seq.size();
        for (int g = 3; g >= 2; --g) {
            if (n < size_t(g) + 1) continue;
            const auto& map = g == 3 ? g3 : g2;
            const auto it = map.find(key(&seq[n - g], g));
            if (it == map.end()) continue;
            for (int i = it->second; i < int(n) && int(out.size()) < kmax; ++i) out.push_back(seq[i]);
            if (!out.empty()) return g;
        }
        return 0;
    }
};

static int serve(Model& m, const std::vector<int>& eos_ids, int chunk) {
    LineReader in;
    std::vector<int> live;                                         // the tokens whose K/V the cache holds, in order
    unsigned long long step_no = 0;
    // guessing: FLINT_SPEC=0 turns it off, FLINT_SPEC_K the most drafted tokens a check (1..3)
    const bool spec_on = !(std::getenv("FLINT_SPEC") && std::atoi(std::getenv("FLINT_SPEC")) == 0);
    const int spec_k = std::max(1, std::min(3, std::getenv("FLINT_SPEC_K") ? std::atoi(std::getenv("FLINT_SPEC_K")) : 3));
    NgramIndex ngi;
    int *sp_out = nullptr, *sp_drafts = nullptr, *sp_host = nullptr;
    CK(cudaMalloc(&sp_out, 8));
    CK(cudaMalloc(&sp_drafts, 16 * 4));
    CK(cudaMallocHost(&sp_host, 32 * 4));
    // what a plain step and a check of T tokens cost (ms), and how many drafts a check keeps, by n-gram length: running
    // averages; a check is made only when it should give more tokens a millisecond than plain steps
    double plain_ms = 0, check_ms[kVMax + 1] = {}, kept_avg[4] = {0, 0, 1.5, 2.0};
    long spec_checks = 0, spec_kept = 0;
    std::printf("INFO engine=strata-flint\n");
    std::printf("READY %d stop\n", m.max_ctx);
    std::fflush(stdout);
    auto is_eos = [&](int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); };
    std::string line;
    while (in.get(line)) {
        if (line == "QUIT") break;
        if (line == "STOP" || line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> <id,id,...>\n"); std::fflush(stdout); continue; }
        std::vector<std::string> f;
        { std::stringstream ss(line); for (std::string w; ss >> w;) f.push_back(w); }
        if (f.size() < 3) { std::printf("ERR a request needs max_new and ids\n"); std::fflush(stdout); continue; }
        const int max_new = std::max(1, std::atoi(f[1].c_str()));
        float temperature = 0.f;
        SampleParams sp;
        bool seeded = false;
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
        std::vector<int> ids;
        { std::stringstream ss(f.back()); for (std::string v; std::getline(ss, v, ',');) if (!v.empty()) ids.push_back(std::atoi(v.c_str())); }
        if (ids.empty() || w + 1 != f.size()) { std::printf("ERR the ids are missing\n"); std::fflush(stdout); continue; }
        bool bad = false;
        for (int t : ids) bad |= t < 0 || t >= m.vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        if (int(ids.size()) + 2 > m.max_ctx) { std::printf("ERR the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), m.max_ctx); std::fflush(stdout); continue; }
        const bool sample = temperature > 0.f;
        sp.inv_temp = sample ? 1.f / temperature : 1.f;
        if (!seeded) sp.seed = (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
        step_no = 0;
        const auto t0 = std::chrono::steady_clock::now();
        // the shared prefix (at least one token is read again: its logits are the first token's); the sliding layers'
        // ring must still hold the window before it
        size_t reuse = 0;
        while (reuse < live.size() && reuse < ids.size() && live[reuse] == ids[reuse]) ++reuse;
        reuse = std::min(reuse, ids.size() - 1);
        if (live.size() - reuse + kWindow > size_t(m.ring_rows())) reuse = 0;
        m.set_pos(int(reuse));
        live.resize(reuse);
        if (ngi.upto > live.size()) ngi.reset();
        std::printf("RESUME %zu\n", reuse);
        std::fflush(stdout);
        bool cancelled = false;
        int read_n = 0;
        for (size_t i = reuse; i < ids.size();) {
            const size_t e = std::min(ids.size(), i + size_t(chunk));
            const std::vector<int> c(ids.begin() + i, ids.begin() + e);
            // (hybrid: a prompt piece copies every layer's experts to the card - ~1.7 s at PCIe 4.0 x8 - so short
            // pieces go through the writing path instead, whose CPU computes the experts the card does not hold)
            static const size_t feed_max = m.hybrid ? size_t(std::getenv("FLINT_HY_FEED") ? std::atoi(std::getenv("FLINT_HY_FEED")) : 128) : 15;
            if (c.size() > feed_max) m.prefill_chunk(c.data(), int(c.size()));
            else m.feed(c);
            live.insert(live.end(), c.begin(), c.end());
            read_n += int(c.size());
            i = e;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("PP %d %zu %.1f %.1f\n", m.pos, ids.size(), ms, read_n / std::max(ms, 1e-3) * 1e3);
            std::fflush(stdout);
            std::string l2;
            if (i < ids.size() && in.poll_line(l2) && l2 == "STOP") { cancelled = true; break; }
        }
        const double prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0 %d 0\n", ids.size(), prompt_ms, reuse, read_n);
            std::fflush(stdout);
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        int produced = 0;
        const char* finish = nullptr;
        auto pick = [&]() {
            if (sample && sp.top_k > 0 && sp.top_k <= kTopKMax) k_sample_topk<<<1, 1024, 0, m.st>>>(m.logits, sp, step_no++, m.state);
            else if (sample) k_sample<<<1, 1024, 0, m.st>>>(m.logits, sp, step_no++, m.state);
            return m.next_token();
        };
        int tok = pick();
        const bool can_spec = spec_on && !m.hybrid && (!sample || (sp.top_k > 0 && sp.top_k <= kTopKMax));
        std::vector<int> seq, drafts;
        auto emit = [&](int t) {
            std::printf("T %d\n", t);
            ++produced;
            if (is_eos(t)) finish = "stop";
            else if (produced >= max_new || m.pos + 2 > m.max_ctx) finish = "length";
        };
        for (;;) {
            emit(tok);
            std::fflush(stdout);
            if (finish) break;
            std::string l2;
            if (in.poll_line(l2) && l2 == "STOP") { finish = "cancel"; break; }
            int g = 0;
            if (can_spec) {
                seq = live;
                seq.push_back(tok);
                ngi.add(live);
                const int room = std::min({spec_k, max_new - produced - 1, m.max_ctx - m.pos - 3});
                g = room > 0 ? ngi.draft(seq, room, drafts) : 0;
                // check only if expected tokens a millisecond beat a plain step (untimed costs: try it)
                if (g) {
                    const int T = 1 + int(drafts.size());
                    const double c = check_ms[T];
                    if (plain_ms > 0 && c > 0 && (1.0 + std::min(kept_avg[g], double(drafts.size()))) / c < 1.0 / plain_ms) g = 0;
                }
            }
            const auto s0 = std::chrono::steady_clock::now();
            if (!g) {
                m.step(-1);                                        // (state[1] is tok: the step reads it on the card)
                live.push_back(tok);
                tok = pick();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
                plain_ms = plain_ms > 0 ? 0.9 * plain_ms + 0.1 * ms : ms;
                continue;
            }
            const int nd = int(drafts.size()), T = 1 + nd;
            int chk[kVMax];
            chk[0] = tok;
            for (int i = 0; i < nd; ++i) chk[i + 1] = drafts[i];
            m.verify(chk, T);
            int kept = 0, next = 0;
            if (sample) {
                std::memcpy(sp_host, drafts.data(), nd * 4);
                CK(cudaMemcpyAsync(sp_drafts, sp_host, nd * 4, cudaMemcpyHostToDevice, m.st));
                k_spec_pick<<<1, 1024, 0, m.st>>>(m.vLG, nd, sp_drafts, sp, step_no, sp_out);
                CK(cudaMemcpyAsync(sp_host + 16, sp_out, 8, cudaMemcpyDeviceToHost, m.st));
                CK(cudaStreamSynchronize(m.st));
                kept = sp_host[16];
                next = sp_host[17];
            } else {
                while (kept < nd && m.vBestHost[kept] == drafts[kept]) ++kept;
                next = m.vBestHost[kept];
            }
            step_no += kept + 1;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            check_ms[T] = check_ms[T] > 0 ? 0.9 * check_ms[T] + 0.1 * ms : ms;
            kept_avg[g] = 0.9 * kept_avg[g] + 0.1 * kept;
            ++spec_checks;
            spec_kept += kept;
            m.v_commit(kept + 1, next);
            live.push_back(tok);
            for (int i = 0; i < kept && !finish; ++i) { emit(drafts[i]); live.push_back(drafts[i]); }
            if (finish) { std::fflush(stdout); break; }
            tok = next;
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        if (std::getenv("FLINT_SPEC_LOG"))
            std::fprintf(stderr, "strata-flint: guessing: %ld checks kept %ld drafts; plain %.2f ms, check 2-4: %.2f %.2f %.2f ms, kept a check by 3/2-gram %.2f %.2f\n",
                         spec_checks, spec_kept, plain_ms, check_ms[2], check_ms[3], check_ms[4], kept_avg[3], kept_avg[2]);
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0 %d 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    reuse, read_n);
        std::fflush(stdout);
    }
    return 0;
}
