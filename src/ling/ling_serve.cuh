// src/ling/ling_serve.cuh - strata-ling --serve: the Strata engine protocol (serve/server.py's StrataEngine) on stdin /
// stdout, with sampling and conversation reuse. The same lines as strata-flint's (flint_serve.cuh):
//
//   -> READY <max_context> stop
//   <- GEN <max_new> [key=value ...] <id,id,...>
//   -> RESUME <n>, PP <position> <prompt tokens> <ms> <tok/s> after each prompt piece, T <id> per token,
//      DONE <generated> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> 0 0 0 0 0 <read> 0
//   <- STOP (ends the running request), QUIT
//
// Keys: temperature, top_p, top_k (0 = off), min_p, seed; absent temperature (or 0) is greedy. ckpt=0: no saved state
// for this request. Others are read and ignored.
//
// Reuse: the KDA layers carry a running state, so a request cannot simply rewind to a shared prefix as Flint does.
// A request continues from where the last one stopped when its prompt extends everything read so far; otherwise from
// the latest saved state at or below the shared prefix. A state is saved one token before each prompt's end (the next
// turn's prompt holds this one's, and an identical prompt still has a token to read), at most LING_SNAPS (4) of them.
#pragma once   // (inside ling_main.cu's namespace; poll.h / unistd.h come from there)

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
constexpr int kVocabL = 157184, kSpChunk = (kVocabL + 1023) / 1024;
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
    const int tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocabL, j0 + kSpChunk);
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
    if (sp.top_k > 0 && sp.top_k < kVocabL) {
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

// top_k in 1..64: the k best logits found once (each thread its own best k, then k rounds of a block maximum), and
// temperature, top_p and min_p act on those k only
constexpr int kTopKMax = 64;
__global__ void __launch_bounds__(1024) k_sample_topk(const float* l, SampleParams sp, unsigned long long step, int* state) {
    __shared__ float cv[kTopKMax];
    __shared__ int ci[kTopKMax];
    __shared__ float wv[32];
    __shared__ int wi[32], wt[32];
    const int k = sp.top_k, tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocabL, j0 + kSpChunk);
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
    if (tid != 0) return;
    float p[kTopKMax], z = 0.f;
    for (int r = 0; r < k; ++r) { p[r] = cv[r] == -INFINITY ? 0.f : __expf((cv[r] - cv[0]) * sp.inv_temp); z += p[r]; }
    int keep = k;
    float acc = 0.f;
    for (int r = 0; r < k; ++r) {
        acc += p[r];
        if (sp.top_p < 1.f && acc >= sp.top_p * z) { keep = r + 1; break; }
    }
    if (sp.min_p > 0.f) for (int r = 1; r < keep; ++r) if (p[r] < sp.min_p * p[0]) { keep = r; break; }
    float kept = 0.f;
    for (int r = 0; r < keep; ++r) kept += p[r];
    const float target = sp_uniform(sp.seed, step, 1) * kept;
    float run = 0.f;
    int pick = ci[keep - 1];
    for (int r = 0; r < keep; ++r) { run += p[r]; if (run > target) { pick = ci[r]; break; } }
    state[1] = pick;
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

// a saved KDA state: the model after the first `pos` tokens of `live` (so valid while live keeps them)
struct Snap {
    float* buf = nullptr;
    int pos = -1;
    uint64_t used = 0;
};

static int serve(Model& m, const std::vector<int>& eos_ids, int chunk) {
    LineReader in;
    std::vector<int> live;          // the tokens the model has read, in order (its state is after all of them)
    unsigned long long step_no = 0;
    const int n_snaps = std::max(1, std::getenv("LING_SNAPS") ? std::atoi(std::getenv("LING_SNAPS")) : 4);
    // a prompt piece this short or shorter goes through the writing path (no expert copies): LING_FEED
    const size_t feed_max = size_t(std::getenv("LING_FEED") ? std::atoi(std::getenv("LING_FEED")) : 32);
    std::vector<Snap> snaps(n_snaps);
    for (auto& s : snaps) CK(cudaMalloc(&s.buf, Model::snap_floats() * 4));
    m.pf_alloc(chunk);   // (the prompt path's memory up front: a card too full says so at start, not mid-request)
    m.stage_alloc();
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    std::fprintf(stderr, "strata-ling: serving; %.2f GB of the card left\n", free_b / 1e9);
    uint64_t clock = 0;
    std::printf("INFO engine=strata-ling\n");
    std::printf("READY %d stop\n", m.max_ctx);
    std::fflush(stdout);
    auto is_eos = [&](int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); };
    // read ids[a, b) on top of the current state, a piece at a time; false when a STOP came in between
    int read_n = 0;
    auto read_span = [&](const std::vector<int>& ids, size_t a, size_t b, size_t total, std::chrono::steady_clock::time_point t0) {
        for (size_t i = a; i < b;) {
            const size_t e = std::min(b, i + size_t(chunk));
            if (e - i > feed_max) m.prefill_chunk(ids.data() + i, int(e - i));
            else for (size_t j = i; j < e; ++j) m.feed_token(ids[j]);
            live.insert(live.end(), ids.begin() + i, ids.begin() + e);
            read_n += int(e - i);
            i = e;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("PP %d %zu %.1f %.1f\n", m.pos, total, ms, read_n / std::max(ms, 1e-3) * 1e3);
            std::fflush(stdout);
            std::string l2;
            if (i < b && in.poll_line(l2) && l2 == "STOP") return false;
        }
        return true;
    };
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
        bool seeded = false, ckpt = true;
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
            else if (k == "ckpt") ckpt = v != "0";
        }
        std::vector<int> ids;
        { std::stringstream ss(f.back()); for (std::string v; std::getline(ss, v, ',');) if (!v.empty()) ids.push_back(std::atoi(v.c_str())); }
        if (ids.empty() || w + 1 != f.size()) { std::printf("ERR the ids are missing\n"); std::fflush(stdout); continue; }
        bool bad = false;
        for (int t : ids) bad |= t < 0 || t >= kVocabL;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        if (int(ids.size()) + 2 > m.max_ctx) { std::printf("ERR the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), m.max_ctx); std::fflush(stdout); continue; }
        const bool sample = temperature > 0.f;
        sp.inv_temp = sample ? 1.f / temperature : 1.f;
        if (!seeded) sp.seed = (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
        step_no = 0;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = ids.size();
        size_t common = 0;
        while (common < live.size() && common < n && live[common] == ids[common]) ++common;
        // where to start: on from the live state if this prompt extends it (with a token left to read), else the
        // latest saved state inside the shared part (a token left to read), else from nothing
        size_t reuse = 0;
        if (common == live.size() && common < n) reuse = common;
        else {
            Snap* best = nullptr;
            for (auto& s : snaps) if (s.pos >= 0 && size_t(s.pos) <= std::min(common, n - 1) && (!best || s.pos > best->pos)) best = &s;
            if (best) {
                m.snap_copy(best->buf, false);
                m.set_pos(best->pos);
                best->used = ++clock;
                reuse = size_t(best->pos);
            } else m.clear_state();
            live.resize(reuse);
            for (auto& s : snaps) if (s.pos > int(reuse)) s.pos = -1;   // (their MLA rows are about to be overwritten)
        }
        std::printf("RESUME %zu\n", reuse);
        std::fflush(stdout);
        read_n = 0;
        // read up to one token before the end, save the state there, then the last token
        bool ok = true;
        if (reuse + 1 < n) ok = read_span(ids, reuse, n - 1, n, t0);
        if (ok && ckpt && reuse + 1 < n) {
            Snap* s = &snaps[0];
            for (auto& c : snaps) { if (c.pos == int(n - 1)) { s = &c; break; } if (c.pos < 0 || (s->pos >= 0 && c.used < s->used)) s = &c; }
            m.snap_copy(s->buf, true);
            s->pos = int(n - 1);
            s->used = ++clock;
        }
        if (ok) ok = read_span(ids, std::max(reuse, n - 1), n, n, t0);
        const double prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0 %d 0\n", n, prompt_ms, reuse, read_n);
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
        for (;;) {
            std::printf("T %d\n", tok);
            ++produced;
            if (is_eos(tok)) finish = "stop";
            else if (produced >= max_new || m.pos + 2 > m.max_ctx) finish = "length";
            std::fflush(stdout);
            if (finish) break;
            std::string l2;
            if (in.poll_line(l2) && l2 == "STOP") { finish = "cancel"; break; }
            m.step();                                              // (state[1] is tok: the step reads it on the card)
            live.push_back(tok);
            tok = pick();
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0 %d 0\n", produced, n, prompt_ms, decode_ms, finish, reuse, read_n);
        std::fflush(stdout);
    }
    return 0;
}
