// src/ling/ling_main.cu - strata-ling: Ling-3.0-flash-VL (GGUF "bailingmoe3", 124B MoE, 5.5B active) on one card,
// the routed experts the card does not hold computed by the CPU (src/flint/flint_cpu.cpp), as strata-flint's hybrid mode.
//
//   strata-ling --model PART1.gguf [--ids 1,2,3|@FILE] [--new N] [--max-ctx N] [--logits FILE]
//
// 42 layers: 35 KDA (Kimi delta attention: depthwise conv4 + silu on q/k/v, per-channel decay, delta rule on a 128 x 128
// state a head) and 7 gated MLA (latent 512 + rope 64, absorbed); 2 dense FFN layers, then 512 routed experts (sigmoid +
// bias, 8 groups / 4 used, top 8, x 2.5, swiglu clamp in the last layers) + a shared expert. Weights stay in the GGUF's
// own formats (Q4_0, Q4_1, Q5_0, Q8_0, Q6_K, F32), read by one kernel that unpacks each per row; activations fp32.
// Reference: llama.cpp src/models/bailingmoe3.cpp + delta-net-base.cpp; ~/ling-ref/ref_ling.py (float64 numpy).
#include "strata/artifact/gguf_reader.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/mman.h>

#include "../flint/flint_cpu.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "strata-ling: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {

#if defined(__HIPCC__)
template <class T> __device__ __forceinline__ T fl_shfl_xor(T v, int m, int w = 32) { return __shfl_xor(v, m, w); }
#else
template <class T> __device__ __forceinline__ T fl_shfl_xor(T v, int m, int w = 32) { return __shfl_xor_sync(0xffffffffu, v, m, w); }
#endif

constexpr int kHidden = 2560, kLayers = 42, kDense = 2, kHeads = 32, kHd = 128, kInner = kHeads * kHd;   // 4096
constexpr int kExperts = 512, kTopK = 8, kGroups = 8, kGroupsUsed = 4, kFfExp = 768, kFfShared = 768, kFfDense = 6144;
constexpr int kLat = 512, kRope = 64, kQkHd = 192, kQkNope = 128, kKvA = kLat + kRope;   // 576
constexpr float kEps = 1e-6f, kRopeBase = 6000000.f, kGateLow = -5.f, kExpScale = 2.5f;

// GGUF tensor types used by this file, and their 32- (or 256-) value block sizes in bytes
constexpr int F32 = 0, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q8_0 = 8, Q6_K = 14;
__host__ __device__ inline size_t row_bytes(int fmt, int n) {
    switch (fmt) {
        case F32: return size_t(n) * 4;
        case Q4_0: return size_t(n / 32) * 18;
        case Q4_1: return size_t(n / 32) * 20;
        case Q5_0: return size_t(n / 32) * 22;
        case Q8_0: return size_t(n / 32) * 34;
        case Q6_K: return size_t(n / 256) * 210;
    }
    return 0;
}

__device__ __forceinline__ float h2f(const uint8_t* p) {
    uint16_t u;
    std::memcpy(&u, p, 2);
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += fl_shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf(float x) { return x / (1.f + expf(-x)); }

// values 8 g .. 8 g + 7 of a row of format fmt
__device__ __forceinline__ void deq8(int fmt, const uint8_t* row, int g, float* v) {
    if (fmt == F32) {
        const float* f = (const float*) row + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = f[i];
        return;
    }
    if (fmt == Q6_K) {
        const int e0 = g * 8, sb = e0 >> 8, r = e0 & 255, n = r >> 7, rr = r & 127, qq = rr >> 5, l0 = rr & 31;
        const uint8_t* B = row + size_t(sb) * 210;
        const uint8_t* ql = B + 64 * n + ((qq & 1) ? 32 : 0);
        const uint8_t* qh = B + 128 + 32 * n;
        const int8_t* sc = (const int8_t*) (B + 192) + 8 * n;
        const float d = h2f(B + 208) * float(sc[(l0 >> 4) + 2 * qq]);
        const int shl = (qq >> 1) ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int l = l0 + i;
            const int q = ((ql[l] >> shl) & 15) | (((qh[l] >> (2 * qq)) & 3) << 4);
            v[i] = d * float(q - 32);
        }
        return;
    }
    const int b = g >> 2, j0 = (g & 3) * 8;   // block, first value
    switch (fmt) {
        case Q4_0: {
            const uint8_t* B = row + size_t(b) * 18;
            const float d = h2f(B);
            const int sh = j0 >= 16 ? 4 : 0, o = j0 & 15;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float(((B[2 + o + i] >> sh) & 15) - 8);
            return;
        }
        case Q4_1: {
            const uint8_t* B = row + size_t(b) * 20;
            const float d = h2f(B), m = h2f(B + 2);
            const int sh = j0 >= 16 ? 4 : 0, o = j0 & 15;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float((B[4 + o + i] >> sh) & 15) + m;
            return;
        }
        case Q5_0: {
            const uint8_t* B = row + size_t(b) * 22;
            const float d = h2f(B);
            uint32_t qh;
            std::memcpy(&qh, B + 2, 4);
            const int sh = j0 >= 16 ? 4 : 0, o = j0 & 15;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = j0 + i;
                const int q = ((B[6 + o + i] >> sh) & 15) | (((qh >> j) & 1) << 4);
                v[i] = d * float(q - 16);
            }
            return;
        }
        case Q8_0: {
            const uint8_t* B = row + size_t(b) * 34;
            const float d = h2f(B);
            const int8_t* q = (const int8_t*) (B + 2) + j0;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float(q[i]);
            return;
        }
    }
}
// a row . x (K values, x in shared memory), the warp's lanes over the 8-value groups, summed over the warp
__device__ __forceinline__ float row_dot(int fmt, const uint8_t* row, const float* x, int K, int lane) {
    float acc = 0.f;
    for (int g = lane; g < K / 8; g += 32) {
        float v[8];
        deq8(fmt, row, g, v);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc += v[i] * x[g * 8 + i];
    }
    return warp_sum(acc);
}

// ---- y_j (= or +=) W_j . x for up to 6 matrices on one input: x, or rmsnorm(x) * nw. 8 warps a block, 4 rows a warp
struct Mv {
    const uint8_t* w[6];
    float* y[6];
    int rows[6], fmt[6], nmat;
    const float *x, *nw;
    int K;
    bool acc;
};
constexpr int kMvRows = 32;
__global__ void __launch_bounds__(256) k_mv(Mv J) {
    extern __shared__ float xs[];
    __shared__ float red[8];
    for (int i = threadIdx.x; i < J.K; i += 256) xs[i] = J.x[i];
    __syncthreads();
    if (J.nw) {
        float ss = 0;
        for (int i = threadIdx.x; i < J.K; i += 256) ss += xs[i] * xs[i];
        ss = warp_sum(ss);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        float t = 0;
        for (int i = 0; i < 8; ++i) t += red[i];
        const float r = rsqrtf(t / J.K + kEps);
        __syncthreads();
        for (int i = threadIdx.x; i < J.K; i += 256) xs[i] = xs[i] * r * J.nw[i];
        __syncthreads();
    }
    int rb = blockIdx.x * kMvRows, j = 0;
    while (j < J.nmat - 1 && rb >= J.rows[j]) rb -= J.rows[j++];
    const int lane = threadIdx.x & 31, r0 = rb + (threadIdx.x >> 5) * 4;
    const size_t rbytes = row_bytes(J.fmt[j], J.K);
    for (int r = r0; r < r0 + 4 && r < J.rows[j]; ++r) {
        const float v = row_dot(J.fmt[j], J.w[j] + size_t(r) * rbytes, xs, J.K, lane);
        if (lane == 0) J.y[j][r] = J.acc ? J.y[j][r] + v : v;
    }
}

// ---- KDA (one block a head, 128 threads; thread c: channel c of the head, as key i and as value j)
struct Kda {
    const float *q, *k, *v, *f, *g, *beta;     // q / k / v / decay / out-gate projections (4096), beta (32): this token's
    const float *cq, *ck, *cv, *dtb, *a, *onorm;   // conv weights (4096 x 4), dt bias, a (32), the head's out norm (128)
    float *cs, *S, *out;                       // conv state (3 x 3 x 4096: q, k, v; the 3 previous inputs), state, output
};
__global__ void __launch_bounds__(128) k_kda(Kda J) {
    __shared__ float qs[kHd], ks[kHd], red[4];
    const int h = blockIdx.x, c = threadIdx.x, ch = h * kHd + c;
    auto conv = [&](const float* x, const float* w, float* st) {   // st: [3][4096], oldest first
        const float o = w[ch * 4 + 0] * st[ch] + w[ch * 4 + 1] * st[kInner + ch] + w[ch * 4 + 2] * st[2 * kInner + ch] + w[ch * 4 + 3] * x[ch];
        st[ch] = st[kInner + ch];
        st[kInner + ch] = st[2 * kInner + ch];
        st[2 * kInner + ch] = x[ch];
        return siluf(o);
    };
    auto bsum = [&](float v) {
        v = warp_sum(v);
        __syncthreads();
        if ((c & 31) == 0) red[c >> 5] = v;
        __syncthreads();
        return red[0] + red[1] + red[2] + red[3];
    };
    float qv = conv(J.q, J.cq, J.cs), kv = conv(J.k, J.ck, J.cs + 3 * kInner), vv = conv(J.v, J.cv, J.cs + 6 * kInner);
    qv *= rsqrtf(bsum(qv * qv) + kEps);
    kv *= rsqrtf(bsum(kv * kv) + kEps);
    qs[c] = qv;
    ks[c] = kv;
    const float decay = expf(kGateLow * sigmoidf(J.a[h] * (J.f[ch] + J.dtb[ch])));   // key channel c's
    const float beta = sigmoidf(J.beta[h]);
    __shared__ float dec[kHd];
    dec[c] = decay;
    __syncthreads();
    // thread c as value j: S[i][j] *= dec[i]; sk = sum_i S[i][j] k[i]; d = (v_j - sk) beta; S[i][j] += k[i] d; o = S^T q
    float* S = J.S + size_t(h) * kHd * kHd;
    float sk = 0;
    for (int i = 0; i < kHd; ++i) {
        const float s = S[i * kHd + c] * dec[i];
        S[i * kHd + c] = s;
        sk += s * ks[i];
    }
    const float d = (vv - sk) * beta;
    float o = 0;
    for (int i = 0; i < kHd; ++i) {
        const float s = S[i * kHd + c] + ks[i] * d;
        S[i * kHd + c] = s;
        o += s * qs[i];
    }
    o *= rsqrtf(float(kHd));
    o *= rsqrtf(bsum(o * o) / kHd + kEps);
    J.out[ch] = o * J.onorm[c] * sigmoidf(J.g[ch]);
}

// ---- gated MLA (absorbed): the cache row of this position: lat (rms-normed) | k_rope (roped), fp32
__device__ __forceinline__ void rope_neox(float* x, int pos, int t) {   // 64 values, thread t < 32 rotates (t, t + 32)
    const float ang = float(pos) * powf(kRopeBase, -2.f * t / kRope);
    float s, c;
    sincosf(ang, &s, &c);
    const float a = x[t], b = x[t + 32];
    x[t] = a * c - b * s;
    x[t + 32] = a * s + b * c;
}
__global__ void __launch_bounds__(512) k_mla_kv(const float* kva, const float* norm, float* cache, const int* state) {
    __shared__ float red[16], kr[kRope];
    const int t = threadIdx.x, pos = state[0];
    const float x = kva[t];
    float v = warp_sum(x * x);
    if ((t & 31) == 0) red[t >> 5] = v;
    if (t < kRope) kr[t] = kva[kLat + t];
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 16; ++i) s += red[i];
    float* row = cache + size_t(pos) * kKvA;
    row[t] = x * rsqrtf(s / kLat + kEps) * norm[t];
    __syncthreads();
    if (t < 32) rope_neox(kr, pos, t);
    __syncthreads();
    if (t < kRope) row[kLat + t] = kr[t];
}
// per head: the absorbed query (wk_b_h q_nope_h, 512) | roped q_rope (64), into qa (32 x 576). Block a head, 256 threads
__global__ void __launch_bounds__(256) k_mla_q(const float* q, const uint8_t* wkb, int fmt, float* qa, const int* state) {
    __shared__ float qn[kQkNope], qr[kRope];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
    if (t < kQkNope) qn[t] = q[h * kQkHd + t];
    if (t < kRope) qr[t] = q[h * kQkHd + kQkNope + t];
    __syncthreads();
    if (t < 32) rope_neox(qr, state[0], t);
    __syncthreads();
    const size_t rb = row_bytes(fmt, kQkNope);
    const uint8_t* W = wkb + size_t(h) * kLat * rb;
    for (int r = w; r < kLat; r += 8) {
        const float v = row_dot(fmt, W + size_t(r) * rb, qn, kQkNope, lane);
        if (lane == 0) qa[h * kKvA + r] = v;
    }
    if (t < kRope) qa[h * kKvA + kLat + t] = qr[t];
}
// per head: softmax over the cache (positions 0 .. pos) of qa . row / sqrt(192), lat-weighted sum (512) into ol
__global__ void __launch_bounds__(256) k_mla_attn(const float* qa, const float* cache, float* ol, const int* state) {
    __shared__ float qh[kKvA], red[8], sc_max, sc_sum;
    extern __shared__ float p[];   // the scores of every position (pos + 1)
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5, n = state[0] + 1;
    for (int i = t; i < kKvA; i += 256) qh[i] = qa[h * kKvA + i];
    __syncthreads();
    const float scale = rsqrtf(float(kQkHd));
    for (int s = w; s < n; s += 8) {
        const float* row = cache + size_t(s) * kKvA;
        float a = 0;
        for (int i = lane; i < kKvA; i += 32) a += qh[i] * row[i];
        a = warp_sum(a);
        if (lane == 0) p[s] = a * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int s = t; s < n; s += 256) m = fmaxf(m, p[s]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, fl_shfl_xor(m, o, 32));
    if (lane == 0) red[w] = m;
    __syncthreads();
    if (t == 0) { float mm = red[0]; for (int i = 1; i < 8; ++i) mm = fmaxf(mm, red[i]); sc_max = mm; }
    __syncthreads();
    float z = 0;
    for (int s = t; s < n; s += 256) { const float e = expf(p[s] - sc_max); p[s] = e; z += e; }
    z = warp_sum(z);
    __syncthreads();
    if (lane == 0) red[w] = z;
    __syncthreads();
    if (t == 0) { float zz = 0; for (int i = 0; i < 8; ++i) zz += red[i]; sc_sum = zz; }
    __syncthreads();
    for (int i = t; i < kLat; i += 256) {
        float a = 0;
        for (int s = 0; s < n; ++s) a += p[s] * cache[size_t(s) * kKvA + i];
        ol[h * kLat + i] = a / sc_sum;
    }
}
// per head: out_h = (wv_b_h ol_h) * sigmoid(gate_h), 128 values. Block a head, 256 threads
__global__ void __launch_bounds__(256) k_mla_out(const float* ol, const uint8_t* wvb, int fmt, const float* gate, float* out) {
    __shared__ float l[kLat];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
    for (int i = t; i < kLat; i += 256) l[i] = ol[h * kLat + i];
    __syncthreads();
    const size_t rb = row_bytes(fmt, kLat);
    const uint8_t* W = wvb + size_t(h) * kHd * rb;
    const float gs = sigmoidf(gate[h]);
    for (int r = w; r < kHd; r += 8) {
        const float v = row_dot(fmt, W + size_t(r) * rb, l, kLat, lane);
        if (lane == 0) out[h * kHd + r] = v * gs;
    }
}

// ---- routing: p = sigmoid(logits); pick on p + bias (8 groups of 64 scored by their top-2 sum, best 4, then top 8);
// weights p normalized x 2.5. One block of 512 threads (thread = expert)
__global__ void __launch_bounds__(512) k_route(const float* logits, const float* bias, int* idx, float* wts) {
    __shared__ float sel[kExperts], prob[kExperts], gsc[kGroups];
    __shared__ int keep[kGroups];
    const int e = threadIdx.x;
    prob[e] = sigmoidf(logits[e]);
    sel[e] = prob[e] + bias[e];
    __syncthreads();
    if (e < kGroups) {   // a group's top-2 sum
        float a = -INFINITY, b = -INFINITY;
        for (int i = e * 64; i < e * 64 + 64; ++i) {
            const float v = sel[i];
            if (v > a) { b = a; a = v; } else if (v > b) b = v;
        }
        gsc[e] = a + b;
    }
    __syncthreads();
    if (e == 0) {
        bool used[kGroups] = {};
        for (int k = 0; k < kGroupsUsed; ++k) {   // the best groups (lowest index on ties, as a stable sort)
            int bi = -1;
            for (int gi = 0; gi < kGroups; ++gi) if (!used[gi] && (bi < 0 || gsc[gi] > gsc[bi])) bi = gi;
            used[bi] = true;
        }
        for (int gi = 0; gi < kGroups; ++gi) keep[gi] = used[gi];
        int top[kTopK];
        bool taken[kExperts] = {};
        float tot = 0;
        for (int k = 0; k < kTopK; ++k) {
            int bi = -1;
            for (int i = 0; i < kExperts; ++i)
                if (keep[i / 64] && !taken[i] && (bi < 0 || sel[i] > sel[bi])) bi = i;
            taken[bi] = true;
            top[k] = bi;
            tot += prob[bi];
        }
        for (int k = 0; k < kTopK; ++k) { idx[k] = top[k]; wts[k] = prob[top[k]] / tot * kExpScale; }
    }
}

// ---- experts on the card (the plan's: slot >= 0) and the shared expert. gate/up: act rows [k * 768] for the routed,
// [8 * 768] for the shared; one warp a row
constexpr int kActRows = kTopK * kFfExp + kFfShared;
struct MoeGu {
    const float *h, *nw;
    const uint8_t *ge, *ue, *gs, *us;     // the card's experts (packed by slot), the shared expert
    int fmt_e, fmt_s;
    const int *idx, *slot;
    float clamp_e, clamp_s;
    float* act;
};
__global__ void __launch_bounds__(256) k_moe_gu(MoeGu J) {
    __shared__ float xs[kHidden], red[8];
    for (int i = threadIdx.x; i < kHidden; i += 256) xs[i] = J.h[i];
    __syncthreads();
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) ss += xs[i] * xs[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < 8; ++i) t += red[i];
    const float r = rsqrtf(t / kHidden + kEps);
    __syncthreads();
    for (int i = threadIdx.x; i < kHidden; i += 256) xs[i] = xs[i] * r * J.nw[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t *G, *U;
    int fmt;
    float cl;
    if (row < kTopK * kFfExp) {
        const int sl = J.slot[J.idx[row / kFfExp]];
        if (sl < 0) { if (lane == 0) J.act[row] = 0.f; return; }
        const size_t rb = row_bytes(J.fmt_e, kHidden);
        G = J.ge + (size_t(sl) * kFfExp + row % kFfExp) * rb;
        U = J.ue + (size_t(sl) * kFfExp + row % kFfExp) * rb;
        fmt = J.fmt_e;
        cl = J.clamp_e;
    } else {
        const size_t rb = row_bytes(J.fmt_s, kHidden);
        G = J.gs + size_t(row - kTopK * kFfExp) * rb;
        U = J.us + size_t(row - kTopK * kFfExp) * rb;
        fmt = J.fmt_s;
        cl = J.clamp_s;
    }
    float g = row_dot(fmt, G, xs, kHidden, lane), u = row_dot(fmt, U, xs, kHidden, lane);
    if (cl > 0.f) { g = fminf(g, cl); u = fmaxf(-cl, fminf(u, cl)); }
    if (lane == 0) J.act[row] = siluf(g) * u;
}
struct MoeDown {
    float* h;
    const float* act;
    const uint8_t *de, *ds;
    int fmt_e, fmt_s;
    const int *idx, *slot;
    const float* wts;
};
// h[r] += sum_k w_k down_k(act_k)[r] (the card's experts) + down_shared(act_s)[r]; one warp a row
__global__ void __launch_bounds__(256) k_moe_down(MoeDown J) {
    __shared__ float xs[kActRows];
    for (int i = threadIdx.x; i < kActRows; i += 256) xs[i] = J.act[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    float tot = 0;
    const size_t rbe = row_bytes(J.fmt_e, kFfExp), rbs = row_bytes(J.fmt_s, kFfShared);
    for (int k = 0; k < kTopK; ++k) {
        const int sl = J.slot[J.idx[k]];
        if (sl < 0) continue;
        tot += J.wts[k] * row_dot(J.fmt_e, J.de + (size_t(sl) * kHidden + row) * rbe, xs + k * kFfExp, kFfExp, lane);
    }
    tot += row_dot(J.fmt_s, J.ds + size_t(row) * rbs, xs + kTopK * kFfExp, kFfShared, lane);
    if (lane == 0) J.h[row] += tot;
}

__global__ void k_embed(const uint8_t* table, int fmt, const int* state, float* h) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;   // 8-value group
    if (g >= kHidden / 8) return;
    float v[8];
    deq8(fmt, table + size_t(state[1]) * row_bytes(fmt, kHidden), g, v);
    for (int i = 0; i < 8; ++i) h[g * 8 + i] = v[i];
}
__global__ void k_swiglu(float* g, const float* u, int n) {   // g = silu(g) * u
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = siluf(g[i]) * u[i];
}
__global__ void k_add(float* h, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += y[i];
}
// argmax of the logits into state[1], and the position on
__global__ void __launch_bounds__(1024) k_argmax(const float* lg, int n, int* state) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int i = 0;
    for (int j = threadIdx.x; j < n; j += 1024) if (lg[j] > v) { v = lg[j]; i = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = fl_shfl_xor(v, o, 32);
        const int oi = fl_shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = bv[0];
        int k = bi[0];
        for (int w = 1; w < 32; ++w) if (bv[w] > b || (bv[w] == b && bi[w] < k)) { b = bv[w]; k = bi[w]; }
        state[1] = k;
        state[0] += 1;
    }
}

struct Tensor { const uint8_t* d = nullptr; int fmt = -1; };
struct Layer {
    bool kda = false, dense = false;
    const float *attn_norm, *ffn_norm;
    // KDA
    Tensor wq, wk, wv, wf, wg, wo;
    const float *beta_w, *cq, *ck, *cv, *dtb, *a, *onorm;
    float *S = nullptr, *cs = nullptr;
    // MLA
    Tensor q, kva, gate, kb, vb;
    const float* kv_norm;
    float* cache = nullptr;
    // FFN
    Tensor fg, fu, fd;                    // dense
    const float *router, *bias;
    Tensor gs, us, ds;                    // shared expert
    int fmt_eg = -1, fmt_ed = -1;         // routed experts' formats (gate/up, down)
    float clamp_e = 0, clamp_s = 0;
    const uint8_t *ge_h = nullptr, *ue_h = nullptr, *de_h = nullptr;   // all the routed experts, host
    const uint8_t *ge = nullptr, *ue = nullptr, *de = nullptr;         // the card's (by slot)
    int* slot = nullptr;
    std::vector<int> slot_h;
    std::vector<float> ffn_norm_h;
};

struct Model {
    std::vector<std::unique_ptr<strata::GgufFile>> parts;
    Layer L[kLayers];
    Tensor embed, head;
    const float* out_norm;
    int max_ctx;
    cudaStream_t st;
    float *h, *x1, *qb, *kb, *vb, *fb, *gb, *betab, *ob, *qa, *ol, *act, *logits, *rl, *wts, *hy;
    int *idx, *state, *host_state;
    int pos = 0;
    size_t loaded = 0;
    std::vector<int> layer_kvh;
    std::vector<float> clamp_exp, clamp_shexp;
    float* hy_h = nullptr;   // pinned: the layer's MoE input (host side of the CPU's experts)
    int* hy_idx = nullptr;
    float* hy_w = nullptr;
    float* hy_y = nullptr;

    const strata::TensorInfo* find(const std::string& n, const strata::GgufFile** owner) const {
        for (const auto& p : parts) if (const auto* t = p->find(n)) { if (owner) *owner = p.get(); return t; }
        return nullptr;
    }
    const uint8_t* host(const std::string& n, int* fmt = nullptr) const {
        const strata::GgufFile* o = nullptr;
        const auto* t = find(n, &o);
        if (!t) { std::fprintf(stderr, "strata-ling: %s is missing\n", n.c_str()); std::exit(1); }
        if (fmt) *fmt = int(t->type);
        return o->tensor_data(*t);
    }
    Tensor up(const std::string& n) {
        const strata::GgufFile* o = nullptr;
        const auto* t = find(n, &o);
        if (!t) { std::fprintf(stderr, "strata-ling: %s is missing\n", n.c_str()); std::exit(1); }
        const int fmt = int(t->type);
        if (fmt != F32 && fmt != Q4_0 && fmt != Q4_1 && fmt != Q5_0 && fmt != Q8_0 && fmt != Q6_K) {
            std::fprintf(stderr, "strata-ling: %s has type %s, not handled\n", n.c_str(), t->type_name());
            std::exit(1);
        }
        const size_t bytes = strata::tensor_payload_bytes(*t);
        void* d = nullptr;
        CK(cudaMalloc(&d, bytes));
        CK(cudaMemcpy(d, o->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
        loaded += bytes;
        return {(const uint8_t*) d, fmt};
    }
    const float* upf(const std::string& n) {
        const Tensor t = up(n);
        if (t.fmt != F32) { std::fprintf(stderr, "strata-ling: %s is not F32\n", n.c_str()); std::exit(1); }
        return (const float*) t.d;
    }
    std::vector<float> f32s(const strata::MetaValue* v) {
        std::vector<float> out;
        if (v) for (const auto& e : v->items) out.push_back(float(e.num()));
        return out;
    }

    Model(const std::string& path, int ctx) : max_ctx(ctx) {
        parts.push_back(std::make_unique<strata::GgufFile>(path));
        {   // the other parts of a split GGUF (NAME-00001-of-0000N.gguf)
            const auto p = path.find("-00001-of-");
            if (p != std::string::npos) {
                const int nparts = std::atoi(path.c_str() + p + 10);
                for (int i = 2; i <= nparts; ++i) {
                    char tag[16];
                    std::snprintf(tag, sizeof tag, "-%05d-of-", i);
                    std::string q = path;
                    q.replace(p, 10, tag);
                    parts.push_back(std::make_unique<strata::GgufFile>(q));
                }
            }
        }
        const auto* arch = parts[0]->get("general.architecture");
        if (!arch || arch->s != "bailingmoe3") { std::fprintf(stderr, "strata-ling: %s is not a bailingmoe3 GGUF\n", path.c_str()); std::exit(1); }
        if (const auto* v = parts[0]->get("bailingmoe3.attention.head_count_kv")) for (const auto& e : v->items) layer_kvh.push_back(int(e.num()));
        clamp_exp = f32s(parts[0]->get("bailingmoe3.swiglu_clamp_exp"));
        clamp_shexp = f32s(parts[0]->get("bailingmoe3.swiglu_clamp_shexp"));
        if (int(layer_kvh.size()) != kLayers) { std::fprintf(stderr, "strata-ling: head_count_kv has %zu entries\n", layer_kvh.size()); std::exit(1); }
        CK(cudaStreamCreate(&st));
        embed = up("token_embd.weight");
        head = up("output.weight");
        out_norm = upf("output_norm.weight");
        flint_cpu::init(std::getenv("LING_CPU_THREADS") ? std::atoi(std::getenv("LING_CPU_THREADS")) : 0);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.kda = layer_kvh[i] == 0;
            l.dense = i < kDense;
            l.attn_norm = upf(b + "attn_norm.weight");
            l.ffn_norm = upf(b + "ffn_norm.weight");
            if (l.kda) {
                l.wq = up(b + "attn_q.weight"); l.wk = up(b + "attn_k.weight"); l.wv = up(b + "attn_v.weight");
                l.wf = up(b + "ssm_f_a.weight"); l.wg = up(b + "ssm_g_a.weight"); l.wo = up(b + "attn_output.weight");
                l.beta_w = upf(b + "ssm_beta.weight");
                l.cq = upf(b + "ssm_conv1d_q.weight"); l.ck = upf(b + "ssm_conv1d_k.weight"); l.cv = upf(b + "ssm_conv1d_v.weight");
                l.dtb = upf(b + "ssm_dt.bias"); l.a = upf(b + "ssm_a"); l.onorm = upf(b + "ssm_norm.weight");
                CK(cudaMalloc(&l.S, size_t(kHeads) * kHd * kHd * 4)); CK(cudaMemset(l.S, 0, size_t(kHeads) * kHd * kHd * 4));
                CK(cudaMalloc(&l.cs, size_t(9) * kInner * 4)); CK(cudaMemset(l.cs, 0, size_t(9) * kInner * 4));
            } else {
                l.q = up(b + "attn_q.weight"); l.kva = up(b + "attn_kv_a_mqa.weight"); l.gate = up(b + "attn_gate.weight");
                l.kb = up(b + "attn_k_b.weight"); l.vb = up(b + "attn_v_b.weight"); l.wo = up(b + "attn_output.weight");
                l.kv_norm = upf(b + "attn_kv_a_norm.weight");
                CK(cudaMalloc(&l.cache, size_t(max_ctx) * kKvA * 4));
            }
            if (l.dense) {
                l.fg = up(b + "ffn_gate.weight"); l.fu = up(b + "ffn_up.weight"); l.fd = up(b + "ffn_down.weight");
            } else {
                l.router = upf(b + "ffn_gate_inp.weight");
                l.bias = upf(b + "exp_probs_b.bias");
                l.gs = up(b + "ffn_gate_shexp.weight"); l.us = up(b + "ffn_up_shexp.weight"); l.ds = up(b + "ffn_down_shexp.weight");
                l.ge_h = host(b + "ffn_gate_exps.weight", &l.fmt_eg);
                int fu = -1;
                l.ue_h = host(b + "ffn_up_exps.weight", &fu);
                l.de_h = host(b + "ffn_down_exps.weight", &l.fmt_ed);
                if (fu != l.fmt_eg || (l.fmt_eg != Q4_0 && l.fmt_eg != Q4_1) || (l.fmt_ed != Q4_0 && l.fmt_ed != Q4_1)) {
                    std::fprintf(stderr, "strata-ling: layer %d's experts are not Q4_0 / Q4_1\n", i);
                    std::exit(1);
                }
                l.clamp_e = i < int(clamp_exp.size()) ? clamp_exp[i] : 0.f;
                l.clamp_s = i < int(clamp_shexp.size()) ? clamp_shexp[i] : 0.f;
                l.slot_h.assign(kExperts, -1);   // (no plan yet: every routed expert on the CPU)
                CK(cudaMalloc(&l.slot, kExperts * 4));
                CK(cudaMemcpy(l.slot, l.slot_h.data(), kExperts * 4, cudaMemcpyHostToDevice));
                l.ge = l.ue = l.de = nullptr;
                const float* nf = (const float*) host(b + "ffn_norm.weight");
                l.ffn_norm_h.assign(nf, nf + kHidden);
            }
        }
        auto f = [&](float*& p, size_t n) { CK(cudaMalloc(&p, n * 4)); };
        f(h, kHidden); f(x1, kHidden); f(qb, kTopK * kFfExp > kHeads * kQkHd ? kTopK * kFfExp : kHeads * kQkHd); f(kb, kInner); f(vb, kInner);
        f(fb, kInner); f(gb, kInner); f(betab, kHeads); f(ob, kInner > kFfDense ? kInner : kFfDense); f(qa, kHeads * kKvA);
        f(ol, kHeads * kLat); f(act, kActRows > kFfDense ? kActRows : kFfDense); f(logits, 157184); f(rl, kExperts); f(wts, kTopK);
        f(hy, kHidden);
        CK(cudaMalloc(&idx, kTopK * 4));
        CK(cudaMalloc(&state, 8)); CK(cudaMemset(state, 0, 8));
        CK(cudaMallocHost(&host_state, 8));
        CK(cudaMallocHost(&hy_h, kHidden * 4)); CK(cudaMallocHost(&hy_y, kHidden * 4));
        CK(cudaMallocHost(&hy_idx, kTopK * 4)); CK(cudaMallocHost(&hy_w, kTopK * 4));
    }

    void mv(std::initializer_list<std::pair<Tensor, float*>> mats, std::initializer_list<int> rows, const float* x, int K,
            const float* nw = nullptr, bool acc = false) {
        Mv J{};
        int j = 0, total = 0;
        for (auto& m : mats) { J.w[j] = m.first.d; J.fmt[j] = m.first.fmt; J.y[j] = m.second; ++j; }
        j = 0;
        for (int r : rows) {
            if (r % kMvRows) { std::fprintf(stderr, "strata-ling: mv rows %d not a multiple of %d\n", r, kMvRows); std::exit(1); }
            J.rows[j++] = r;
            total += r;
        }
        J.nmat = j; J.x = x; J.nw = nw; J.K = K; J.acc = acc;
        k_mv<<<total / kMvRows, 256, K * 4, st>>>(J);
    }

    // one token (state[1]) through the model at state[0]; logits, then argmax into state[1] and state[0] + 1
    void step() {
        k_embed<<<(kHidden / 8 + 63) / 64, 64, 0, st>>>(embed.d, embed.fmt, state, h);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            if (l.kda) {
                mv({{l.wq, qb}, {l.wk, kb}, {l.wv, vb}, {l.wf, fb}, {l.wg, gb}, {Tensor{(const uint8_t*) l.beta_w, F32}, betab}},
                   {kInner, kInner, kInner, kInner, kInner, kHeads}, h, kHidden, l.attn_norm);
                k_kda<<<kHeads, kHd, 0, st>>>(Kda{qb, kb, vb, fb, gb, betab, l.cq, l.ck, l.cv, l.dtb, l.a, l.onorm, l.cs, l.S, ob});
                mv({{l.wo, h}}, {kHidden}, ob, kInner, nullptr, true);
            } else {
                mv({{l.q, qb}, {l.kva, kb}, {l.gate, betab}}, {kHeads * kQkHd, kKvA, kHeads}, h, kHidden, l.attn_norm);
                k_mla_kv<<<1, kLat, 0, st>>>(kb, l.kv_norm, l.cache, state);
                k_mla_q<<<kHeads, 256, 0, st>>>(qb, l.kb.d, l.kb.fmt, qa, state);
                k_mla_attn<<<kHeads, 256, size_t(max_ctx) * 4, st>>>(qa, l.cache, ol, state);
                k_mla_out<<<kHeads, 256, 0, st>>>(ol, l.vb.d, l.vb.fmt, betab, ob);
                mv({{l.wo, h}}, {kHidden}, ob, kInner, nullptr, true);
            }
            if (l.dense) {
                mv({{l.fg, act}, {l.fu, ob}}, {kFfDense, kFfDense}, h, kHidden, l.ffn_norm);
                k_swiglu<<<(kFfDense + 255) / 256, 256, 0, st>>>(act, ob, kFfDense);
                mv({{l.fd, h}}, {kHidden}, act, kFfDense, nullptr, true);
                continue;
            }
            mv({{Tensor{(const uint8_t*) l.router, F32}, rl}}, {kExperts}, h, kHidden, l.ffn_norm);
            k_route<<<1, kExperts, 0, st>>>(rl, l.bias, idx, wts);
            CK(cudaMemcpyAsync(hy_h, h, kHidden * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hy_idx, idx, kTopK * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hy_w, wts, kTopK * 4, cudaMemcpyDeviceToHost, st));
            k_moe_gu<<<kActRows / 8, 256, 0, st>>>(MoeGu{h, l.ffn_norm, l.ge, l.ue, l.gs.d, l.us.d, l.fmt_eg, l.gs.fmt, idx, l.slot,
                                                         l.clamp_e, l.clamp_s, act});
            k_moe_down<<<kHidden / 8, 256, 0, st>>>(MoeDown{h, act, l.de, l.ds.d, l.fmt_ed, l.ds.fmt, idx, l.slot, wts});
            CK(cudaStreamSynchronize(st));
            flint_cpu::Expert ex[kTopK];
            float w[kTopK];
            int n = 0;
            const size_t eb_gu = size_t(kFfExp) * row_bytes(l.fmt_eg, kHidden), eb_d = size_t(kHidden) * row_bytes(l.fmt_ed, kFfExp);
            for (int k = 0; k < kTopK; ++k) {
                const int e = hy_idx[k];
                if (l.slot_h[e] >= 0) continue;
                ex[n] = {l.ge_h + size_t(e) * eb_gu, l.ue_h + size_t(e) * eb_gu, l.de_h + size_t(e) * eb_d};
                w[n++] = hy_w[k];
            }
            if (n) {
                flint_cpu::moe_q(hy_h, l.ffn_norm_h.data(), kHidden, kFfExp, n, ex, w, hy_y, l.fmt_eg, l.fmt_ed, l.clamp_e);
                CK(cudaMemcpyAsync(hy, hy_y, kHidden * 4, cudaMemcpyHostToDevice, st));
                k_add<<<(kHidden + 255) / 256, 256, 0, st>>>(h, hy, kHidden);
            }
            if (std::getenv("LING_DUMP_H")) dump_h();
        }
        mv({{head, logits}}, {157184}, h, kHidden, out_norm);
        k_argmax<<<1, 1024, 0, st>>>(logits, 157184, state);
        ++pos;
    }
    std::vector<float> dumped;
    void dump_h() {
        std::vector<float> v(kHidden);
        CK(cudaMemcpyAsync(v.data(), h, kHidden * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        dumped.insert(dumped.end(), v.begin(), v.end());
    }
    void feed_token(int tok) {
        const int v = tok;
        CK(cudaMemcpyAsync(state + 1, &v, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
        step();
    }
    int next_token() {
        CK(cudaMemcpyAsync(host_state, state, 8, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return host_state[1];
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string path, ids_s, logits_path;
    int n_new = 0, ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--model") path = next();
        else if (a == "--ids") {
            ids_s = next();
            if (!ids_s.empty() && ids_s[0] == '@') {
                FILE* f = std::fopen(ids_s.c_str() + 1, "r");
                if (!f) { std::fprintf(stderr, "strata-ling: cannot read %s\n", ids_s.c_str() + 1); return 1; }
                std::string s;
                char buf[65536];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
                std::fclose(f);
                ids_s = s;
            }
        }
        else if (a == "--new") n_new = std::stoi(next());
        else if (a == "--max-ctx" || a == "--max-context") ctx = std::stoi(next());
        else if (a == "--logits") logits_path = next();
        else { std::fprintf(stderr, "usage: strata-ling --model PART1.gguf --ids 1,2,3|@FILE [--new N] [--max-ctx N] [--logits FILE]\n"); return 2; }
    }
    std::vector<int> ids;
    { std::stringstream ss(ids_s); for (std::string v; std::getline(ss, v, ',');) if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids.push_back(std::stoi(v)); }
    if (path.empty() || ids.empty()) { std::fprintf(stderr, "strata-ling: --model and --ids are needed\n"); return 2; }
    const auto t0 = std::chrono::steady_clock::now();
    Model m(path, ctx);
    std::fprintf(stderr, "strata-ling: %.1f GB on the card in %.1f s\n", m.loaded / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    std::vector<float> lg(157184);
    const auto t1 = std::chrono::steady_clock::now();
    for (int t : ids) {
        m.feed_token(t);
        if (lf) {
            CK(cudaMemcpyAsync(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost, m.st));
            CK(cudaStreamSynchronize(m.st));
            std::fwrite(lg.data(), 4, lg.size(), lf);
        }
    }
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (lf) std::fclose(lf);
    if (const char* dp = std::getenv("LING_DUMP_H")) {
        FILE* f = std::fopen(dp, "wb");
        std::fwrite(m.dumped.data(), 4, m.dumped.size(), f);
        std::fclose(f);
    }
    const auto t2 = std::chrono::steady_clock::now();
    for (int n = 0; n < n_new; ++n) {
        const int t = m.next_token();
        std::printf("%d%s", t, n + 1 < n_new ? "," : "\n");
        std::fflush(stdout);
        m.step();
    }
    CK(cudaStreamSynchronize(m.st));
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    std::fprintf(stderr, "strata-ling: read %zu in %.3f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n", ids.size(), read_s,
                 ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    return 0;
}
