// src/flint/flint_main.cu - strata-flint: Recursive Flint (40B-A2.5B MoE) on one card, from its Q4_1 GGUF.
//
// v1, the correctness build: one position at a time through all 49 layers, every weight resident in VRAM in its stored
// format, simple kernels. Checked against llama.cpp's own Flint (greedy tokens and top-5 log-probabilities) before
// anything was made fast.
//
//   strata-flint --model FILE.gguf --ids 630,6963,297 [--new N] [--top FILE] [--max-ctx N]
//
// --top writes, per written token, the five most likely next tokens and their log-probabilities ("id:logprob ...").
//
// The model: 49 layers, hidden 2048. Attention: 32 query heads and 4 K/V heads of 128, RMSNorm on each head's q and k,
// a learned "sink" logit per head added to the softmax, 35 sliding-window layers (513 positions, NEOX RoPE base 1e4)
// and 14 full-attention layers without RoPE. Feed-forward: layers 0-1 dense SwiGLU 6144; the other 47 have 256
// experts (SwiGLU 512, top 6 by sigmoid score, weights normalized to sum 1) plus a shared SwiGLU of 1024.
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
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "strata-flint: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kHidden = 2048, kLayers = 49, kDense = 2;
constexpr int kHeads = 32, kKv = 4, kHd = 128, kQDim = kHeads * kHd, kKvDim = kKv * kHd;
constexpr int kFfDense = 6144, kExperts = 256, kTopK = 6, kFfExp = 512, kFfShared = 1024;
constexpr int kWindow = 513;
constexpr float kEps = 1e-6f, kRopeBase = 10000.f;
constexpr int kQ4_1 = 3, kF32 = 0;

struct block_q4_1 { __half d, m; uint8_t qs[16]; };     // value = d * q + m, q in 0..15 (low nibbles: 0-15, high: 16-31)
static_assert(sizeof(block_q4_1) == 20, "block layout");

// ---------------------------------------------------------------------------------------------------- kernels
__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ inline float silu(float x) { return x / (1.f + expf(-x)); }

template <int N>
__device__ inline float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0;
    for (int i = 0; i < N / 32; ++i) r += sh[i];
    __syncthreads();
    return r;
}

constexpr int kMvRows = 32, kMvRpw = 4;
// Stage n_in inputs into LDS: x, or rmsnorm(x) * nw when nw, or silu(x) * x2 when x2. 256 threads.
__device__ inline void stage_in(float* xs, const float* x, const float* nw, const float* x2, int n_in) {
    for (int i = threadIdx.x * 4; i < n_in; i += 1024) {
        float4 v = *(const float4*) (x + i);
        if (x2) {
            const float4 u = *(const float4*) (x2 + i);
            v = make_float4(silu(v.x) * u.x, silu(v.y) * u.y, silu(v.z) * u.z, silu(v.w) * u.w);
        }
        *(float4*) (xs + i) = v;
    }
    __syncthreads();
    if (!nw) return;
    __shared__ float red[8];
    float ss = 0;
    for (int i = threadIdx.x; i < n_in; i += 256) ss += xs[i] * xs[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < 8; ++i) t += red[i];
    const float r = rsqrtf(t / n_in + kEps);
    __syncthreads();
    for (int i = threadIdx.x; i < n_in; i += 256) xs[i] = xs[i] * r * nw[i];
    __syncthreads();
}

// one lane's share of R rows' dot products with the staged input: lane reads word qd of every 8th Q4_1 block
template <int R>
__device__ inline void q4_rows(const uint8_t* W, size_t rowb, int nb, const float* xs, int lane, float* acc) {
    const int qd = lane & 3;
    for (int b = lane >> 2; b < nb; b += 8) {
        const float4 xa = *(const float4*) (xs + b * 32 + qd * 4);
        const float4 xb = *(const float4*) (xs + b * 32 + 16 + qd * 4);
        const float sx = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* B = W + r * rowb + size_t(b) * 20;
            const uint32_t dm = *(const uint32_t*) B;
            const uint32_t q = *(const uint32_t*) (B + 4 + qd * 4);
            const float d = __half2float(__ushort_as_half(uint16_t(dm))), m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
            const float t = float(q & 15) * xa.x + float((q >> 8) & 15) * xa.y + float((q >> 16) & 15) * xa.z + float((q >> 24) & 15) * xa.w
                          + float((q >> 4) & 15) * xb.x + float((q >> 12) & 15) * xb.y + float((q >> 20) & 15) * xb.z + float(q >> 28) * xb.w;
            acc[r] += d * t + m * sx;
        }
    }
}

// Up to three plain matrices on one staged input (stage_in's modes), 32 rows a block; y[j][r] = (or +=, when acc)
struct MvIn {
    const uint8_t* w[3];
    float* y[3];
    int rows[3], nmat;
    const float *x, *nw, *x2;
    int n_in;
    bool acc;
};
__global__ void __launch_bounds__(256) k_mv_in(MvIn J) {
    extern __shared__ float xs[];
    stage_in(xs, J.x, J.nw, J.x2, J.n_in);
    int rb = blockIdx.x * kMvRows, j = 0;
    while (j < J.nmat - 1 && rb >= J.rows[j]) rb -= J.rows[j++];
    const int lane = threadIdx.x & 31, r0 = rb + (threadIdx.x >> 5) * kMvRpw, nb = J.n_in / 32;
    const size_t rowb = size_t(nb) * 20;
    float acc[kMvRpw] = {};
    q4_rows<kMvRpw>(J.w[j] + size_t(r0) * rowb, rowb, nb, xs, lane, acc);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) { float* y = J.y[j] + r0 + r; *y = J.acc ? *y + v : v; }
    }
}

// MoE gate/up: the 6 chosen experts (512 rows each) and the shared expert (1024 rows), act = silu(gate.x) * up.x with
// x = rmsnorm(h) * nw; act rows: expert k at k * 512, the shared expert at 3072. 32 rows (of gate and of up) a block.
constexpr int kActRows = kTopK * kFfExp + kFfShared;   // 4096
__global__ void __launch_bounds__(256) k_moe_gu(const float* h, const float* nw, const uint8_t* ge, const uint8_t* ue,
                                                const uint8_t* gs, const uint8_t* us, const int* idx, float* act) {
    extern __shared__ float xs[];
    stage_in(xs, h, nw, nullptr, kHidden);
    const int row = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw, lane = threadIdx.x & 31;
    constexpr int nb = kHidden / 32;
    constexpr size_t rowb = size_t(nb) * 20;
    const uint8_t *G, *U;
    if (row < kTopK * kFfExp) {
        const size_t e = size_t(idx[row / kFfExp]) * kFfExp + row % kFfExp;
        G = ge + e * rowb; U = ue + e * rowb;
    } else {
        G = gs + size_t(row - kTopK * kFfExp) * rowb; U = us + size_t(row - kTopK * kFfExp) * rowb;
    }
    float g[kMvRpw] = {}, u[kMvRpw] = {};
    q4_rows<kMvRpw>(G, rowb, nb, xs, lane, g);
    q4_rows<kMvRpw>(U, rowb, nb, xs, lane, u);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float gv = warp_sum(g[r]), uv = warp_sum(u[r]);
        if (lane == 0) act[row + r] = silu(gv) * uv;
    }
}

// MoE down: h[r] += sum_k w[k] * (down_{idx[k]} . act_k)[r] + (down_shared . act_shared)[r]; 32 rows a block
__global__ void __launch_bounds__(256) k_moe_down(float* h, const float* act, const uint8_t* de, const uint8_t* ds,
                                                  const int* idx, const float* wts) {
    __shared__ float xs[kActRows];
    for (int i = threadIdx.x * 4; i < kActRows; i += 1024) *(float4*) (xs + i) = *(const float4*) (act + i);
    __syncthreads();
    const int r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw, lane = threadIdx.x & 31;
    float tot[kMvRpw] = {};
    constexpr size_t rowe = size_t(kFfExp / 32) * 20, rows_ = size_t(kFfShared / 32) * 20;
    for (int k = 0; k < kTopK; ++k) {
        float a[kMvRpw] = {};
        q4_rows<kMvRpw>(de + (size_t(idx[k]) * kHidden + r0) * rowe, rowe, kFfExp / 32, xs + k * kFfExp, lane, a);
        const float wk = wts[k];
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) tot[r] += wk * a[r];
    }
    q4_rows<kMvRpw>(ds + size_t(r0) * rows_, rows_, kFfShared / 32, xs + kTopK * kFfExp, lane, tot);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(tot[r]);
        if (lane == 0) h[r0 + r] += v;
    }
}

// router: logits = W . (rmsnorm(h) * nw) (fp32, 256 rows), one warp a row, 4 rows a block; the last block to finish
// picks the top 6 by sigmoid score and their normalized weights (done: a counter it resets)
__global__ void __launch_bounds__(128) k_router(const float* h, const float* nw, const uint16_t* W, float* logits,
                                                int* idx, float* wts, unsigned* done) {
    __shared__ float xs[kHidden];
    __shared__ float red[4];
    __shared__ bool last;
    for (int i = threadIdx.x * 4; i < kHidden; i += 512) *(float4*) (xs + i) = *(const float4*) (h + i);
    __syncthreads();
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 128) ss += xs[i] * xs[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    const float rr = rsqrtf((red[0] + red[1] + red[2] + red[3]) / kHidden + kEps);
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    float acc = 0;
    for (int i = lane * 8; i < kHidden; i += 256) {
        const uint4 w = *(const uint4*) (W + size_t(r) * kHidden + i);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc += __uint_as_float(ww[k] << 16) * xs[i + 2 * k] * nw[i + 2 * k]
                 + __uint_as_float(ww[k] & 0xffff0000u) * xs[i + 2 * k + 1] * nw[i + 2 * k + 1];
        }
    }
    acc = warp_sum(acc) * rr;
    if (lane == 0) logits[r] = acc;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    // top 6 over the 256 sigmoid scores, 128 threads with 2 each
    __shared__ float sc[kExperts];
    __shared__ float bv[4], picked[kTopK];
    __shared__ int bi[4];
    for (int i = threadIdx.x; i < kExperts; i += 128) sc[i] = 1.f / (1.f + expf(-((volatile float*) logits)[i]));
    __syncthreads();
    for (int k = 0; k < kTopK; ++k) {
        float v = sc[threadIdx.x];
        int i = threadIdx.x;
        if (sc[threadIdx.x + 128] > v) { v = sc[threadIdx.x + 128]; i = threadIdx.x + 128; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int j = bi[0];
            for (int w = 1; w < 4; ++w) if (bv[w] > b || (bv[w] == b && bi[w] < j)) { b = bv[w]; j = bi[w]; }
            idx[k] = j;
            picked[k] = b;
            sc[j] = -1.f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float t = 0;
        for (int k = 0; k < kTopK; ++k) t += picked[k];
        for (int k = 0; k < kTopK; ++k) wts[k] = picked[k] / t;
        *done = 0;
    }
}

// after the q/k/v products: q and k heads RMSNorm (+ RoPE on sliding layers), k and v into the cache. Blocks 0-31 the
// query heads, 32-35 the K/V heads; 128 threads.
__global__ void k_qkv_post(float* q, const float* k, const float* v, const float* qn, const float* kn, float* K,
                           float* V, const int* state, bool rope) {
    __shared__ float sh[4];
    __shared__ float t[kHd];
    const int pos = state[0], hb = blockIdx.x, i = threadIdx.x;
    const bool isq = hb < kHeads;
    const float x = isq ? q[hb * kHd + i] : k[(hb - kHeads) * kHd + i];
    const float s = block_sum<128>(x * x, sh);
    float y = x * rsqrtf(s / kHd + kEps) * (isq ? qn[i] : kn[i]);
    if (rope) {
        t[i] = y;
        __syncthreads();
        const int j = i & 63;
        const float ang = float(pos) * powf(kRopeBase, -2.f * j / kHd);
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float x0 = t[j], x1 = t[j + 64];
        y = i < 64 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    if (isq) q[hb * kHd + i] = y;
    else {
        const int kh = hb - kHeads;
        K[size_t(pos) * kKvDim + kh * kHd + i] = y;
        V[size_t(pos) * kKvDim + kh * kHd + i] = v[kh * kHd + i];
    }
}

// Split attention for one query position: block (h, c) takes chunk c of the keys lo..pos (kAttnChunks equal parts)
// for query head h; each warp walks its keys with an online softmax (lane = 4 of the 128 dims), the 8 warps merge in
// LDS, and the chunk's (max, sum, unnormalized output) goes to part. k_attn_merge adds the chunks and the sink.
constexpr int kAttnChunks = 16, kPart = kHd + 4;   // a chunk's record: max, sum, 2 spare, 128 outputs
__global__ void __launch_bounds__(256) k_attn_part(const float* q, const float* K, const float* V, float* part,
                                                   const int* state, bool swa) {
    const int h = blockIdx.x, c = blockIdx.y, kh = h / (kHeads / kKv);
    const int pos = state[0], lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    const int per = (n + kAttnChunks - 1) / kAttnChunks, a = lo + c * per, b = min(pos + 1, a + per);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float4 qv = *(const float4*) (q + h * kHd + lane * 4);
    const float scale = rsqrtf(float(kHd));
    float m = -INFINITY, l = 0;
    float4 o = make_float4(0, 0, 0, 0);
    for (int p = a + w; p < b; p += 8) {
        const float4 kv = *(const float4*) (K + size_t(p) * kKvDim + kh * kHd + lane * 4);
        const float d = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w) * scale;
        const float mn = fmaxf(m, d), cf = expf(m - mn), e = expf(d - mn);
        const float4 vv = *(const float4*) (V + size_t(p) * kKvDim + kh * kHd + lane * 4);
        o.x = o.x * cf + e * vv.x; o.y = o.y * cf + e * vv.y; o.z = o.z * cf + e * vv.z; o.w = o.w * cf + e * vv.w;
        l = l * cf + e;
        m = mn;
    }
    __shared__ float sm[8], sl[8];
    __shared__ float4 so[8][32];
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    so[w][lane] = o;
    __syncthreads();
    if (w == 0) {
        float M = -INFINITY;
        for (int i = 0; i < 8; ++i) M = fmaxf(M, sm[i]);
        float L = 0;
        float4 O = make_float4(0, 0, 0, 0);
        for (int i = 0; i < 8; ++i) {
            if (sm[i] == -INFINITY) continue;
            const float f = expf(sm[i] - M);
            L += sl[i] * f;
            O.x += so[i][lane].x * f; O.y += so[i][lane].y * f; O.z += so[i][lane].z * f; O.w += so[i][lane].w * f;
        }
        float* P = part + (size_t(h) * kAttnChunks + c) * kPart;
        if (lane == 0) { P[0] = M; P[1] = L; }
        *(float4*) (P + 4 + lane * 4) = O;
    }
}
__global__ void k_attn_merge(const float* part, const float* sinks, float* out) {
    const int h = blockIdx.x, t = threadIdx.x;   // 128 threads
    const float* P = part + size_t(h) * kAttnChunks * kPart;
    float M = sinks[h];
    for (int c = 0; c < kAttnChunks; ++c) M = fmaxf(M, P[c * kPart]);
    float L = expf(sinks[h] - M), O = 0;
    for (int c = 0; c < kAttnChunks; ++c) {
        const float mc = P[c * kPart];
        if (mc == -INFINITY) continue;
        const float f = expf(mc - M);
        L += P[c * kPart + 1] * f;
        O += P[c * kPart + 4 + t] * f;
    }
    out[h * kHd + t] = O / L;
}

// the most likely next token into state[1], and the position moves on (state[0]): 64 blocks each take a slice, the
// last one to finish picks among their 64 bests (done: a counter it resets)
__global__ void __launch_bounds__(1024) k_argmax64(const float* lg, int n, int* state, float2* best, unsigned* done) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ bool last;
    const int per = (n + gridDim.x - 1) / gridDim.x, a = blockIdx.x * per, e = min(n, a + per);
    float v = -INFINITY;
    int i = 0;
    for (int j = a + threadIdx.x; j < e; j += 1024) if (lg[j] > v) { v = lg[j]; i = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o, 32);
        const int oi = __shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w) if (bv[w] > v || (bv[w] == v && bi[w] < i)) { v = bv[w]; i = bi[w]; }
        best[blockIdx.x] = make_float2(v, __int_as_float(i));
        __threadfence();
        last = atomicAdd(done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float bvv = -INFINITY;
    int bii = 0;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const float cv = ((volatile float*) best)[2 * b];
        const int ci = __float_as_int(((volatile float*) best)[2 * b + 1]);
        if (cv > bvv || (cv == bvv && ci < bii)) { bvv = cv; bii = ci; }
    }
    state[1] = bii;
    state[0] += 1;
    *done = 0;
}

// read a set of buffers so they sit in the card's memory-side cache (64 MB on the R9700) when the main stream gets to
// them; the values are not used
struct Touch { const uint4* p[12]; uint32_t n16[12]; int n; };
__global__ void __launch_bounds__(256) k_touch(Touch T, unsigned* sink) {
    uint32_t a = 0;
    for (int j = 0; j < T.n; ++j)
        for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < T.n16[j]; i += gridDim.x * 256) {
            const uint4 v = T.p[j][i];
            a ^= v.x ^ v.w;
        }
    if (a == 0x9e3779b9u) *sink = a;
}

__global__ void k_embed(const float* table, const int* state, float* h) {
    const int i = blockIdx.x * 256 + threadIdx.x, tok = state[1];
    if (i < kHidden) h[i] = table[size_t(tok) * kHidden + i];
}


// ---------------------------------------------------------------------------------------------------- prompt path (many tokens)
// Inputs go to the matrix units as two fp16 numbers a value, hi + lo (hi = fp16(x * s), lo = fp16(x * s - hi), s a
// power of two putting the row's largest magnitude in [2^14, 2^15)): hi alone cost KL 0.006 on Flint, hi + lo is near
// fp32. Weights are unpacked from Q4_1 to fp16 (d * q + m, one rounding: KL 0.0015).
typedef _Float16 pf_v8h __attribute__((ext_vector_type(8)));
typedef float pf_v8f __attribute__((ext_vector_type(8)));
typedef _Float16 pf_v2h __attribute__((ext_vector_type(2)));
__device__ __forceinline__ pf_v8f pf_wmma(const pf_v8h& a, const pf_v8h& b, const pf_v8f& c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __trap();
    return c;
#endif
}
// LDS-only barrier (a __syncthreads' fence also covers global memory and invalidates the vector caches on gfx12)
__device__ __forceinline__ void pf_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// rows of x (T x K fp32; rmsnorm * nw first when nw; silu(x) * x2 when x2) to hi / lo fp16 and the row's scale back
__global__ void __launch_bounds__(256) k_split16(const float* x, const float* x2, const float* nw, int K, uint16_t* xh,
                                                 uint16_t* xl, float* xs) {
    const int t = blockIdx.x;
    const float* r = x + size_t(t) * K;
    const float* r2 = x2 ? x2 + size_t(t) * K : nullptr;
    __shared__ float red[8], redm[8];
    float ss = 0, mx = 0;
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = r2 ? silu(r[i]) * r2[i] : r[i];
        ss += v * v;
        mx = fmaxf(mx, fabsf(nw ? v * nw[i] : v));
    }
    ss = warp_sum(ss);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = ss; redm[threadIdx.x >> 5] = mx; }
    __syncthreads();
    float tot = 0, m = 0;
    for (int i = 0; i < 8; ++i) { tot += red[i]; m = fmaxf(m, redm[i]); }
    const float rn = nw ? rsqrtf(tot / K + kEps) : 1.f;
    m *= rn;
    const int e = m > 0.f ? ilogbf(m) : 0;
    const float up = ldexpf(1.f, 14 - e);
    for (int i = threadIdx.x; i < K; i += 256) {
        float v = r2 ? silu(r[i]) * r2[i] : r[i];
        if (nw) v = v * rn * nw[i];
        const float a = v * up;
        const __half hi = __float2half(a);
        const __half lo = __float2half(a - __half2float(hi));
        xh[size_t(t) * K + i] = __half_as_ushort(hi);
        xl[size_t(t) * K + i] = __half_as_ushort(lo);
    }
    if (threadIdx.x == 0) xs[t] = ldexpf(1.f, e - 14);
}
// the 8 fp16 weights of a lane's A operand: row W (Q4_1, GGUF layout), values k .. k + 7 (k a multiple of 8)
__device__ __forceinline__ pf_v8h pf_q41_a8(const uint8_t* W, int k) {
    const uint8_t* B = W + (k >> 5) * 20;
    const int j0 = k & 31;
    const uint32_t dm = *(const uint32_t*) B;
    const uint2 qq = *(const uint2*) (B + 4 + (j0 & 15));
    const int sh = j0 >= 16 ? 4 : 0;
    const uint32_t u0 = (qq.x >> sh) & 0x0F0F0F0Fu, u1 = (qq.y >> sh) & 0x0F0F0F0Fu;
    // bytes q0 q1 q2 q3 -> halves (1024 + q0, 1024 + q1), (1024 + q2, 1024 + q3)
    const uint32_t c = 0x64646464u;
    const uint32_t h0 = __builtin_amdgcn_perm(c, u0, 0x07010700u), h1 = __builtin_amdgcn_perm(c, u0, 0x07030702u);
    const uint32_t h2 = __builtin_amdgcn_perm(c, u1, 0x07010700u), h3 = __builtin_amdgcn_perm(c, u1, 0x07030702u);
    const pf_v2h off = {(_Float16) -1024.f, (_Float16) -1024.f};
    const _Float16 d = __builtin_bit_cast(_Float16, uint16_t(dm)), m = __builtin_bit_cast(_Float16, uint16_t(dm >> 16));
    const pf_v2h d2 = {d, d}, m2 = {m, m};
    pf_v2h w0 = (__builtin_bit_cast(pf_v2h, h0) + off), w1 = (__builtin_bit_cast(pf_v2h, h1) + off);
    pf_v2h w2 = (__builtin_bit_cast(pf_v2h, h2) + off), w3 = (__builtin_bit_cast(pf_v2h, h3) + off);
    w0 = __builtin_elementwise_fma(w0, d2, m2); w1 = __builtin_elementwise_fma(w1, d2, m2);
    w2 = __builtin_elementwise_fma(w2, d2, m2); w3 = __builtin_elementwise_fma(w3, d2, m2);
    pf_v8h r;
    r[0] = w0[0]; r[1] = w0[1]; r[2] = w1[0]; r[3] = w1[1]; r[4] = w2[0]; r[5] = w2[1]; r[6] = w3[0]; r[7] = w3[1];
    return r;
}
// Y[t][o] (= or +=) sum_k W[o][k] x[t][k] for T tokens: W Q4_1 rows (N x K), x as hi / lo fp16 (T x K) with a scale a
// row. A block: 8 warps side by side over 256 outputs (2 tiles of 16 each), 64 tokens (4 tiles), k 64 a step; the
// weights go straight from memory into the operand registers, the inputs through LDS (double-buffered).
// LO: also the lo halves (two products a pair of operands).
template <bool LO>
__global__ void __launch_bounds__(256) k_gemm_q41(const uint8_t* __restrict__ W, int N, int K, const uint16_t* __restrict__ xh,
                                                  const uint16_t* __restrict__ xl, const float* __restrict__ xs, int T,
                                                  float* __restrict__ Y, int ldy, int add) {
    constexpr int TA = 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];   // [buffer][hi, lo][token][k]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int o0 = blockIdx.x * 256 + warp * TA * 16, t0 = blockIdx.y * BN;
    const size_t rowb = size_t(K / 32) * 20;
    const uint8_t* wr[TA];
    // (rows past N - the head's 150144 is not a multiple of 256 - read row N - 1 and are not stored)
#pragma unroll
    for (int a = 0; a < TA; ++a) wr[a] = W + size_t(min(o0 + a * 16 + l16, N - 1)) * rowb;
    // B loads: 64 tokens x 64 k x 2 = 1024 uint4; thread takes token tid / 4 (+ 0) ... 4 of them
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            const uint16_t* src = hl ? xl : xh;
            r[j] = (t0 + t < T && (LO || !hl)) ? *(const uint4*) (src + size_t(t0 + t) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            *(uint4*) &sx[buf][hl][t][c] = r[j];
        }
    };
    pf_v8f acc[TA][TB];
#pragma unroll
    for (int a = 0; a < TA; ++a)
#pragma unroll
        for (int b = 0; b < TB; ++b) acc[a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[4];
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) fetch(k0 + BK, rb);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[TA];
#pragma unroll
            for (int a = 0; a < TA; ++a) fa[a] = pf_q41_a8(wr[a], k0 + kk * 16 + g * 8);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int a = 0; a < TA; ++a) acc[a][b] = pf_wmma(fa[a], bh, acc[a][b]);
                if (LO) {
                    const pf_v8h bl = *(const pf_v8h*) &sx[buf][1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                    for (int a = 0; a < TA; ++a) acc[a][b] = pf_wmma(fa[a], bl, acc[a][b]);
                }
            }
        }
        if (k0 + BK < K) put(rb, buf ^ 1);
        pf_sync();
        buf ^= 1;
    }
    // lane: outputs o0 + a * 16 + g * 8 + i, token t0 + b * 16 + l16
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = t0 + b * 16 + l16;
        if (t >= T) continue;
        const float sc = xs[t];
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (o0 + a * 16 + g * 8 + i >= N) break;
                float* y = Y + size_t(t) * ldy + o0 + a * 16 + g * 8 + i;
                const float v = acc[a][b][i] * sc;
                *y = add ? *y + v : v;
            }
    }
}


// The routed experts' grouped product: rows of the grouped order (expert e's rows bounds[e] .. bounds[e + 1]) in tiles of
// 64 (tiles: (expert, first row) pairs, *n_tiles of them). Input row of grouped row r: src[r] (or r without src).
// GU: gate (W1) and up (W2) both, Y = silu(gate) * up; otherwise Y = W1 . x. W_e = W + e * mat_bytes.
template <bool LO, bool GU>
__global__ void __launch_bounds__(256) k_gemm_q41_grp(const uint8_t* __restrict__ W1, const uint8_t* __restrict__ W2, size_t mat_bytes,
                                                      int N, int K, const uint16_t* __restrict__ xh, const uint16_t* __restrict__ xl,
                                                      const float* __restrict__ xs, const int* __restrict__ src,
                                                      const int* __restrict__ bounds, const int* __restrict__ tiles,
                                                      const int* __restrict__ n_tiles, float* __restrict__ Y, int ldy) {
    constexpr int TA = 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8, NM = GU ? 2 : 1;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];
    __shared__ int srow[BN];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int e = tiles[2 * tile], r0 = bounds[e] + tiles[2 * tile + 1], rn = min(BN, bounds[e + 1] - r0);
    const int o0 = blockIdx.x * 256 + warp * TA * 16;
    if (tid < BN) srow[tid] = tid < rn ? (src ? src[r0 + tid] : r0 + tid) : -1;
    pf_sync();
    const size_t rowb = size_t(K / 32) * 20;
    const uint8_t* wr[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) wr[mm][a] = (mm ? W2 : W1) + size_t(e) * mat_bytes + size_t(o0 + a * 16 + l16) * rowb;
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            const uint16_t* sp = hl ? xl : xh;
            const int row = srow[t];
            r[j] = (row >= 0 && (LO || !hl)) ? *(const uint4*) (sp + size_t(row) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            *(uint4*) &sx[buf][hl][t][c] = r[j];
        }
    };
    pf_v8f acc[NM][TA][TB];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int b = 0; b < TB; ++b) acc[mm][a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[4];
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) fetch(k0 + BK, rb);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[NM][TA];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) fa[mm][a] = pf_q41_a8(wr[mm][a], k0 + kk * 16 + g * 8);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                    for (int a = 0; a < TA; ++a) acc[mm][a][b] = pf_wmma(fa[mm][a], bh, acc[mm][a][b]);
                if (LO) {
                    const pf_v8h bl = *(const pf_v8h*) &sx[buf][1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                        for (int a = 0; a < TA; ++a) acc[mm][a][b] = pf_wmma(fa[mm][a], bl, acc[mm][a][b]);
                }
            }
        }
        if (k0 + BK < K) put(rb, buf ^ 1);
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = b * 16 + l16;
        if (t >= rn) continue;
        const float sc = xs[srow[t]];
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float v = GU ? silu(acc[0][a][b][i] * sc) * (acc[GU ? 1 : 0][a][b][i] * sc) : acc[0][a][b][i] * sc;
                Y[size_t(r0 + t) * ldy + o0 + a * 16 + g * 8 + i] = v;
            }
    }
}

// router logits for 16 tokens a block: thread = expert, rows rmsnorm(h_t) * nw (rn: each token's 1 / rms)
__global__ void __launch_bounds__(256) k_pf_router(const float* H, const float* rn, const float* nw, const uint16_t* W, int T,
                                                   float* logits) {
    __shared__ float xs[16][65];
    const int t0 = blockIdx.x * 16, e = threadIdx.x;
    float acc[16] = {};
    for (int k0 = 0; k0 < kHidden; k0 += 64) {
        for (int i = threadIdx.x; i < 16 * 64; i += 256) {
            const int t = i / 64, k = i % 64;
            xs[t][k] = t0 + t < T ? H[size_t(t0 + t) * kHidden + k0 + k] * rn[t0 + t] * nw[k0 + k] : 0.f;
        }
        __syncthreads();
        const uint16_t* w = W + size_t(e) * kHidden + k0;
        for (int k = 0; k < 64; k += 8) {
            const uint4 wv = *(const uint4*) (w + k);
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float w0 = __uint_as_float(ww[j] << 16), w1 = __uint_as_float(ww[j] & 0xffff0000u);
#pragma unroll
                for (int t = 0; t < 16; ++t) acc[t] += w0 * xs[t][k + 2 * j] + w1 * xs[t][k + 2 * j + 1];
            }
        }
        __syncthreads();
    }
    for (int t = 0; t < 16; ++t) if (t0 + t < T) logits[size_t(t0 + t) * kExperts + e] = acc[t];
}
// each token's 1 / rms over the hidden size (one warp a token)
__global__ void k_pf_rms(const float* H, int T, float* rn) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    float ss = 0;
    for (int i = lane; i < kHidden; i += 32) ss += H[size_t(t) * kHidden + i] * H[size_t(t) * kHidden + i];
    ss = warp_sum(ss);
    if (lane == 0) rn[t] = rsqrtf(ss / kHidden + kEps);
}
// top 6 by sigmoid score per token (one warp a token, the decode path's tie rule), weights normalized; counts per expert
__global__ void k_pf_topk(const float* logits, int T, int* idx, float* wts, int* count) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    float sc[8];
    for (int j = 0; j < 8; ++j) sc[j] = 1.f / (1.f + expf(-logits[size_t(t) * kExperts + lane * 8 + j]));
    float picked[kTopK];
    int pi[kTopK];
    float tot = 0;
    for (int k = 0; k < kTopK; ++k) {
        float v = -2.f;
        int i = 0;
        for (int j = 0; j < 8; ++j) if (sc[j] > v) { v = sc[j]; i = lane * 8 + j; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == i / 8) sc[i % 8] = -1.f;
        picked[k] = v; pi[k] = i; tot += v;
    }
    if (lane < kTopK) {
        idx[t * kTopK + lane] = pi[lane];
        wts[t * kTopK + lane] = picked[lane] / tot;
        atomicAdd(count + pi[lane], 1);
    }
}
// bounds[e] = sum of counts before e (bounds[256] = all); fill reset; tiles of 64 rows (one block of 256)
__global__ void k_pf_bounds(const int* count, int* bounds, int* fill, int* tiles, int* n_tiles) {
    __shared__ int sc[kExperts], st[kExperts];
    const int e = threadIdx.x;
    sc[e] = count[e];
    st[e] = (count[e] + 63) / 64;
    __syncthreads();
    for (int off = 1; off < kExperts; off <<= 1) {
        const int a = e >= off ? sc[e - off] : 0, b = e >= off ? st[e - off] : 0;
        __syncthreads();
        sc[e] += a; st[e] += b;
        __syncthreads();
    }
    const int nt = (count[e] + 63) / 64, base = st[e] - nt;
    bounds[e + 1] = sc[e];
    if (e == 0) bounds[0] = 0;
    fill[e] = 0;
    for (int i = 0; i < nt; ++i) { tiles[2 * (base + i)] = e; tiles[2 * (base + i) + 1] = i * 64; }
    if (e == kExperts - 1) *n_tiles = st[e];
}
// each (token, choice) its row in the grouped order
__global__ void k_pf_scatter(const int* idx, int T, const int* bounds, int* fill, int* src, int* slot) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= T * kTopK) return;
    const int e = idx[i], p = bounds[e] + atomicAdd(fill + e, 1);
    src[p] = i / kTopK;
    slot[i] = p;
}
// H[t] += sum_k w[t][k] * Y[slot[t][k]] (the routed experts' outputs, grouped order)
__global__ void k_pf_combine(float* H, const float* Y, const int* slot, const float* wts, int T) {
    const int t = blockIdx.x, i = threadIdx.x;   // 256 threads, 8 values each
    float s[8] = {};
    for (int k = 0; k < kTopK; ++k) {
        const float w = wts[t * kTopK + k];
        const float* y = Y + size_t(slot[t * kTopK + k]) * kHidden;
        for (int j = 0; j < 8; ++j) s[j] += w * y[i + 256 * j];
    }
    for (int j = 0; j < 8; ++j) H[size_t(t) * kHidden + i + 256 * j] += s[j];
}
__global__ void k_pf_embed(const float* table, const int* ids, float* H) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < kHidden; i += 256) H[size_t(t) * kHidden + i] = table[size_t(ids[t]) * kHidden + i];
}
// q / k heads: RMSNorm (+ NEOX RoPE on sliding layers) at position p0 + t; q back in place, k and v into the cache.
// grid (36 heads, T), 128 threads
__global__ void k_pf_qkpost(float* QKV, const float* qn, const float* kn, float* K, float* V, int p0, bool rope) {
    __shared__ float sh[4];
    __shared__ float tb[kHd];
    const int hb = blockIdx.x, t = blockIdx.y, i = threadIdx.x, pos = p0 + t;
    float* row = QKV + size_t(t) * (kQDim + 2 * kKvDim);
    const bool isq = hb < kHeads;
    float* src = isq ? row + hb * kHd : row + kQDim + (hb - kHeads) * kHd;
    const float x = src[i];
    const float s = block_sum<128>(x * x, sh);
    float y = x * rsqrtf(s / kHd + kEps) * (isq ? qn[i] : kn[i]);
    if (rope) {
        tb[i] = y;
        __syncthreads();
        const int j = i & 63;
        const float ang = float(pos) * powf(kRopeBase, -2.f * j / kHd);
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float x0 = tb[j], x1 = tb[j + 64];
        y = i < 64 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    if (isq) src[i] = y;
    else {
        const int kh = hb - kHeads;
        K[size_t(pos) * kKvDim + kh * kHd + i] = y;
        V[size_t(pos) * kKvDim + kh * kHd + i] = row[kQDim + kKvDim + kh * kHd + i];
    }
}
// Attention for T queries (positions p0 ..) against the cache, fp32: block = (head, 64 queries), 4 threads a query,
// each 32 of the 128 dims; keys in tiles of 32 through LDS; online softmax; the sink joins the denominator at the end.
__global__ void __launch_bounds__(256) k_pf_attn(const float* QKV, const float* K, const float* V, const float* sinks,
                                                 float* ATT, int p0, int T, bool swa) {
    __shared__ float ks[32][kHd + 4], vs[32][kHd + 4];
    const int h = blockIdx.x, kh = h / (kHeads / kKv), q0 = blockIdx.y * 64;
    const int tid = threadIdx.x, ql = tid >> 2, part = tid & 3, t = q0 + ql;
    const bool live = t < T;
    const int pos = p0 + min(t, T - 1);
    const int lo = swa ? max(0, pos - (kWindow - 1)) : 0;
    float qv[32];
    const float scale = rsqrtf(float(kHd));
    for (int j = 0; j < 32; ++j) qv[j] = live ? QKV[size_t(t) * (kQDim + 2 * kKvDim) + h * kHd + part * 32 + j] * scale : 0.f;
    float m = -INFINITY, l = 0, o[32] = {};
    // the block's key range: from the first query's window start to the last query's position
    const int last = p0 + min(q0 + 63, T - 1);
    const int first_lo = swa ? max(0, p0 + q0 - (kWindow - 1)) : 0;
    for (int k0 = first_lo; k0 <= last; k0 += 32) {
        for (int i = tid; i < 32 * 32; i += 256) {   // 32 keys x 32 float4 (k and v)
            const int key = i >> 5, c = (i & 31) * 4, p = k0 + key;
            const float4 kv = p <= last ? *(const float4*) (K + size_t(p) * kKvDim + kh * kHd + c) : make_float4(0, 0, 0, 0);
            const float4 vv = p <= last ? *(const float4*) (V + size_t(p) * kKvDim + kh * kHd + c) : make_float4(0, 0, 0, 0);
            *(float4*) &ks[key][c] = kv;
            *(float4*) &vs[key][c] = vv;
        }
        __syncthreads();
        for (int key = 0; key < 32; ++key) {
            const int p = k0 + key;
            float d = 0;
#pragma unroll
            for (int j = 0; j < 32; ++j) d += qv[j] * ks[key][part * 32 + j];
            d += __shfl_xor(d, 1, 32);
            d += __shfl_xor(d, 2, 32);
            if (p < lo || p > pos || !live) continue;
            const float mn = fmaxf(m, d), cf = expf(m - mn), e = expf(d - mn);
#pragma unroll
            for (int j = 0; j < 32; ++j) o[j] = o[j] * cf + e * vs[key][part * 32 + j];
            l = l * cf + e;
            m = mn;
        }
        __syncthreads();
    }
    if (!live) return;
    const float sk = sinks[h], M = fmaxf(m, sk);
    const float f = expf(m - M), L = l * f + expf(sk - M);
    for (int j = 0; j < 32; ++j) ATT[size_t(t) * kQDim + h * kHd + part * 32 + j] = o[j] * f / L;
}

// ---------------------------------------------------------------------------------------------------- the model
struct Layer {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *sinks;
    const uint16_t* router = nullptr;   // bf16
    const uint8_t *wq, *wk, *wv, *wo;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;              // dense (layers 0-1)
    const uint8_t *gate_e = nullptr, *up_e = nullptr, *down_e = nullptr;        // experts
    const uint8_t *gate_s = nullptr, *up_s = nullptr, *down_s = nullptr;        // shared expert
    bool swa = false;
    float *K = nullptr, *V = nullptr;
};

struct Model {
    std::unique_ptr<strata::GgufFile> g;
    std::vector<Layer> L;
    const float *embed, *out_norm;
    const uint8_t* head;
    int vocab = 0, max_ctx;
    size_t loaded = 0;
    cudaStream_t st, side;
    std::vector<cudaEvent_t> ev;
    unsigned* sink;
    // scratch
    float *act, *part, *h, *q, *k, *v, *att, *f1, *f2, *logits, *rl, *wts;
    float2* best;
    unsigned *done, *done2;
    int* idx;
    int pos = 0;
    int* state;          // on the card: [0] the position, [1] the token going in next
    int* host_state;     // pinned copy
    cudaGraphExec_t graph = nullptr;

    const void* up_tensor(const std::string& name, int type) {
        const strata::TensorInfo* t = g->find(name);
        if (!t) { std::fprintf(stderr, "strata-flint: %s is missing\n", name.c_str()); std::exit(1); }
        if (int(t->type) != type) { std::fprintf(stderr, "strata-flint: %s is %s\n", name.c_str(), t->type_name()); std::exit(1); }
        const size_t bytes = strata::tensor_payload_bytes(*t);
        void* d = nullptr;
        CK(cudaMalloc(&d, bytes));
        if (name.find("ffn_gate_inp") != std::string::npos) {
            // the router: its fp32 values are all bf16-exact (a bf16 checkpoint), so bf16 halves its reads losslessly
            const float* f = (const float*) g->tensor_data(*t);
            std::vector<uint16_t> v(bytes / 4);
            for (size_t i = 0; i < v.size(); ++i) {
                uint32_t u;
                std::memcpy(&u, f + i, 4);
                if (u & 0xffff) { std::fprintf(stderr, "strata-flint: %s is not bf16-exact\n", name.c_str()); std::exit(1); }
                v[i] = uint16_t(u >> 16);
            }
            CK(cudaMemcpy(d, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
            loaded += v.size() * 2;
            return d;
        }
        CK(cudaMemcpy(d, g->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
        loaded += bytes;
        return d;
    }

    Model(const std::string& path, int ctx) : max_ctx(ctx) {
        g = std::make_unique<strata::GgufFile>(path);
        const auto* arch = g->get("general.architecture");
        if (!arch || arch->s != "flint") { std::fprintf(stderr, "strata-flint: %s is not a Flint GGUF\n", path.c_str()); std::exit(1); }
        const auto* pat = g->get("flint.attention.sliding_window_pattern");
        vocab = int(g->find("output.weight")->shape[1]);
        CK(cudaStreamCreate(&st));
        CK(cudaStreamCreate(&side));
        ev.resize(kLayers + 2);
        for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        CK(cudaMalloc(&sink, 4));
        L.resize(kLayers);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.swa = pat && pat->items.size() == size_t(kLayers) && pat->items[i].u != 0;
            l.attn_norm = (const float*) up_tensor(b + "attn_norm.weight", kF32);
            l.ffn_norm = (const float*) up_tensor(b + "ffn_norm.weight", kF32);
            l.q_norm = (const float*) up_tensor(b + "attn_q_norm.weight", kF32);
            l.k_norm = (const float*) up_tensor(b + "attn_k_norm.weight", kF32);
            l.sinks = (const float*) up_tensor(b + "attn_sinks.weight", kF32);
            l.wq = (const uint8_t*) up_tensor(b + "attn_q.weight", kQ4_1);
            l.wk = (const uint8_t*) up_tensor(b + "attn_k.weight", kQ4_1);
            l.wv = (const uint8_t*) up_tensor(b + "attn_v.weight", kQ4_1);
            l.wo = (const uint8_t*) up_tensor(b + "attn_output.weight", kQ4_1);
            if (i < kDense) {
                l.gate = (const uint8_t*) up_tensor(b + "ffn_gate.weight", kQ4_1);
                l.up = (const uint8_t*) up_tensor(b + "ffn_up.weight", kQ4_1);
                l.down = (const uint8_t*) up_tensor(b + "ffn_down.weight", kQ4_1);
            } else {
                l.router = (const uint16_t*) up_tensor(b + "ffn_gate_inp.weight", kF32);
                l.gate_e = (const uint8_t*) up_tensor(b + "ffn_gate_exps.weight", kQ4_1);
                l.up_e = (const uint8_t*) up_tensor(b + "ffn_up_exps.weight", kQ4_1);
                l.down_e = (const uint8_t*) up_tensor(b + "ffn_down_exps.weight", kQ4_1);
                l.gate_s = (const uint8_t*) up_tensor(b + "ffn_gate_shexp.weight", kQ4_1);
                l.up_s = (const uint8_t*) up_tensor(b + "ffn_up_shexp.weight", kQ4_1);
                l.down_s = (const uint8_t*) up_tensor(b + "ffn_down_shexp.weight", kQ4_1);
            }
            CK(cudaMalloc(&l.K, size_t(max_ctx) * kKvDim * 4));
            CK(cudaMalloc(&l.V, size_t(max_ctx) * kKvDim * 4));
        }
        embed = (const float*) up_tensor("token_embd.weight", kF32);
        out_norm = (const float*) up_tensor("output_norm.weight", kF32);
        head = (const uint8_t*) up_tensor("output.weight", kQ4_1);
        auto f = [&](float*& p, size_t n) { CK(cudaMalloc(&p, n * 4)); };
        f(h, kHidden); f(q, kQDim); f(k, kKvDim); f(v, kKvDim); f(att, kQDim); f(f1, kFfDense); f(f2, kFfDense);
        f(logits, vocab); f(rl, kExperts); f(wts, kTopK);
        CK(cudaMalloc(&idx, kTopK * 4));
        CK(cudaMalloc(&part, size_t(kHeads) * kAttnChunks * kPart * 4));
        CK(cudaMalloc(&act, kActRows * 4));
        CK(cudaMalloc(&done, 4));
        CK(cudaMemset(done, 0, 4));
        CK(cudaMalloc(&state, 8));
        CK(cudaMemset(state, 0, 8));
        CK(cudaMallocHost(&host_state, 8));
        CK(cudaMalloc(&done2, 4));
        CK(cudaMemset(done2, 0, 4));
        CK(cudaMalloc(&best, 64 * sizeof(float2)));
    }

    // one token through the model (tok, or the last step's own choice when tok < 0); logits on the card, the most
    // likely next token in state[1]. Recorded as a graph on the first call and replayed after that.
    // the prompt's tokens one step each; each copy reads its own pinned slot (one shared slot was overwritten by the
    // next token before the card had read it), and the slots are kept until the stream has passed them
    void feed(const std::vector<int>& toks, const std::function<void()>& after_each = nullptr) {
        int* pin = nullptr;
        CK(cudaMallocHost(&pin, std::max<size_t>(toks.size(), 1) * 4));
        std::memcpy(pin, toks.data(), toks.size() * 4);
        for (size_t i = 0; i < toks.size(); ++i) {
            CK(cudaMemcpyAsync(state + 1, pin + i, 4, cudaMemcpyHostToDevice, st));
            step(-1);
            if (after_each) after_each();
        }
        CK(cudaStreamSynchronize(st));
        CK(cudaFreeHost(pin));
    }
    void step(int tok) {
        if (pos >= max_ctx) { std::fprintf(stderr, "strata-flint: context full (%d)\n", max_ctx); std::exit(1); }
        if (tok >= 0) { feed({tok}); return; }
        static const bool no_graph = std::getenv("FLINT_NO_GRAPH") != nullptr;
        if (no_graph) { record(); ++pos; return; }
        if (!graph) {
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            record();
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&graph, g, nullptr, nullptr, 0));
            CK(cudaGraphDestroy(g));
        }
        CK(cudaGraphLaunch(graph, st));
        static const bool check = std::getenv("FLINT_CHECK") != nullptr;
        if (check) { CK(cudaStreamSynchronize(st)); CK(cudaGetLastError()); }
        ++pos;
    }
    // the next token as the last step chose it
    int next_token() {
        CK(cudaMemcpyAsync(host_state, state, 8, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return host_state[1];
    }
    void mv_in(std::initializer_list<std::pair<const uint8_t*, float*>> mats, std::initializer_list<int> rows,
               const float* x_, int n_in, const float* nw = nullptr, const float* x2 = nullptr, bool accum = false) {
        MvIn J{};
        int j = 0, total = 0;
        for (auto& m : mats) { J.w[j] = m.first; J.y[j] = m.second; ++j; }
        j = 0;
        for (int r : rows) { J.rows[j++] = r; total += r; }
        J.nmat = j; J.x = x_; J.nw = nw; J.x2 = x2; J.n_in = n_in; J.acc = accum;
        k_mv_in<<<total / kMvRows, 256, n_in * 4, st>>>(J);
    }
    // ------------------------------------------------ the prompt path: chunks of up to tc tokens through all layers
    int tc = 0;
    float *pH = nullptr, *pQKV, *pATT, *pG, *pU, *pACT, *pY, *pXS, *pRN, *pRL, *pW;
    uint16_t *pXH, *pXL;
    int *pIds, *pIdx, *pCount, *pBounds, *pFill, *pTiles, *pNT, *pSrc, *pSlot;
    void pf_alloc(int T) {
        if (T <= tc) return;
        tc = T;
        auto f = [&](float*& p_, size_t n) { if (p_) CK(cudaFree(p_)); CK(cudaMalloc(&p_, n * 4)); };
        auto fi = [&](int*& p_, size_t n) { CK(cudaMalloc(&p_, n * 4)); };
        const size_t S = size_t(T) * kTopK;
        f(pH, size_t(T) * kHidden); pQKV = nullptr; f(pQKV, size_t(T) * (kQDim + 2 * kKvDim)); pATT = nullptr; f(pATT, size_t(T) * kQDim);
        pG = nullptr; f(pG, size_t(T) * kFfDense); pU = nullptr; f(pU, size_t(T) * kFfDense); pACT = nullptr; f(pACT, S * kFfExp);
        pY = nullptr; f(pY, S * kHidden); pXS = nullptr; f(pXS, S); pRN = nullptr; f(pRN, T); pRL = nullptr; f(pRL, size_t(T) * kExperts);
        pW = nullptr; f(pW, S);
        CK(cudaMalloc(&pXH, std::max(size_t(T) * kFfDense, S * kFfExp) * 2));
        CK(cudaMalloc(&pXL, std::max(size_t(T) * kFfDense, S * kFfExp) * 2));
        fi(pIds, T); fi(pIdx, S); fi(pCount, kExperts); fi(pBounds, kExperts + 1); fi(pFill, kExperts);
        fi(pTiles, 2 * (S / 64 + kExperts + 1)); fi(pNT, 1); fi(pSrc, S); fi(pSlot, S);
    }
    // (debugging) only the first FLINT_NLAYERS layers, then the head
    static int n_layers() { static const int n = std::getenv("FLINT_NLAYERS") ? std::atoi(std::getenv("FLINT_NLAYERS")) : kLayers; return n; }
    static bool pf_lo() { static const bool v = !std::getenv("FLINT_PF_HI") ; return v; }
    // y (T x N, row stride ldy; += when add) = W . x for the split rows (xh, xl, xs) of T tokens
    void pf_gemm(const uint8_t* W, int N, int K, int T, float* Y, int ldy, int add) {
        const dim3 gr((N + 255) / 256, (T + 63) / 64);
        if (pf_lo()) k_gemm_q41<true><<<gr, 256, 0, st>>>(W, N, K, pXH, pXL, pXS, T, Y, ldy, add);
        else k_gemm_q41<false><<<gr, 256, 0, st>>>(W, N, K, pXH, pXL, pXS, T, Y, ldy, add);
    }
    // (debugging) FLINT_DUMP_H=FILE: h after every layer, every row, appended as fp32
    void dump_rows(const float* d, int rows) {
        static const char* fn = std::getenv("FLINT_DUMP_H");
        if (!fn) return;
        static FILE* f = std::fopen(fn, "wb");
        std::vector<float> v(size_t(rows) * kHidden);
        CK(cudaMemcpyAsync(v.data(), d, v.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        std::fwrite(v.data(), 4, v.size(), f);
        std::fflush(f);
    }
    void pf_split(const float* x, const float* x2, const float* nw, int K, int rows) {
        k_split16<<<rows, 256, 0, st>>>(x, x2, nw, K, pXH, pXL, pXS);
    }
    // T tokens at positions pos .. pos + T - 1; afterwards the decode state is the next position and the most likely
    // next token. all_logits: every row's logits (into lg_all, T x vocab on the card), else only the last row's.
    void prefill_chunk(const int* ids, int T, float* lg_all = nullptr) {
        if (pos + T > max_ctx) { std::fprintf(stderr, "strata-flint: context full (%d)\n", max_ctx); std::exit(1); }
        pf_alloc(T);
        const int p0 = pos;
        CK(cudaMemcpyAsync(pIds, ids, T * 4, cudaMemcpyHostToDevice, st));
        k_pf_embed<<<T, 256, 0, st>>>(embed, pIds, pH);
        const int NQKV = kQDim + 2 * kKvDim;
        const size_t gu_bytes = size_t(kFfExp) * (kHidden / 32) * 20, d_bytes = size_t(kHidden) * (kFfExp / 32) * 20;
        const int S = T * kTopK, max_tiles = S / 64 + kExperts + 1;
        for (int i = 0; i < n_layers(); ++i) {
            Layer& l = L[i];
            pf_split(pH, nullptr, l.attn_norm, kHidden, T);
            pf_gemm(l.wq, kQDim, kHidden, T, pQKV, NQKV, 0);
            pf_gemm(l.wk, kKvDim, kHidden, T, pQKV + kQDim, NQKV, 0);
            pf_gemm(l.wv, kKvDim, kHidden, T, pQKV + kQDim + kKvDim, NQKV, 0);
            k_pf_qkpost<<<dim3(kHeads + kKv, T), kHd, 0, st>>>(pQKV, l.q_norm, l.k_norm, l.K, l.V, p0, l.swa);
            k_pf_attn<<<dim3(kHeads, (T + 63) / 64), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa);
            pf_split(pATT, nullptr, nullptr, kQDim, T);
            pf_gemm(l.wo, kHidden, kQDim, T, pH, kHidden, 1);
            pf_split(pH, nullptr, l.ffn_norm, kHidden, T);
            if (i < kDense) {
                pf_gemm(l.gate, kFfDense, kHidden, T, pG, kFfDense, 0);
                pf_gemm(l.up, kFfDense, kHidden, T, pU, kFfDense, 0);
                pf_split(pG, pU, nullptr, kFfDense, T);
                pf_gemm(l.down, kHidden, kFfDense, T, pH, kHidden, 1);
                dump_rows(pH, T);
                continue;
            }
            // shared expert first (its input is the split rows just made)
            pf_gemm(l.gate_s, kFfShared, kHidden, T, pG, kFfShared, 0);
            pf_gemm(l.up_s, kFfShared, kHidden, T, pU, kFfShared, 0);
            // routing (fp32 router on rmsnorm(h))
            k_pf_rms<<<(T + 7) / 8, 256, 0, st>>>(pH, T, pRN);
            k_pf_router<<<(T + 15) / 16, 256, 0, st>>>(pH, pRN, l.ffn_norm, l.router, T, pRL);
            CK(cudaMemsetAsync(pCount, 0, kExperts * 4, st));
            k_pf_topk<<<(T + 7) / 8, 256, 0, st>>>(pRL, T, pIdx, pW, pCount);
            k_pf_bounds<<<1, kExperts, 0, st>>>(pCount, pBounds, pFill, pTiles, pNT);
            k_pf_scatter<<<(S + 255) / 256, 256, 0, st>>>(pIdx, T, pBounds, pFill, pSrc, pSlot);
            // experts' gate/up on the split rows of each routed token
            if (pf_lo()) k_gemm_q41_grp<true, true><<<dim3(kFfExp / 256, max_tiles), 256, 0, st>>>(l.gate_e, l.up_e, gu_bytes, kFfExp, kHidden, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT, kFfExp);
            else k_gemm_q41_grp<false, true><<<dim3(kFfExp / 256, max_tiles), 256, 0, st>>>(l.gate_e, l.up_e, gu_bytes, kFfExp, kHidden, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT, kFfExp);
            // shared expert's down, added into h
            pf_split(pG, pU, nullptr, kFfShared, T);
            pf_gemm(l.down_s, kHidden, kFfShared, T, pH, kHidden, 1);
            // experts' down on their act rows (grouped order), then the weighted sum into h
            pf_split(pACT, nullptr, nullptr, kFfExp, S);
            if (pf_lo()) k_gemm_q41_grp<true, false><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down_e, nullptr, d_bytes, kHidden, kFfExp, pXH, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY, kHidden);
            else k_gemm_q41_grp<false, false><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down_e, nullptr, d_bytes, kHidden, kFfExp, pXH, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY, kHidden);
            k_pf_combine<<<T, 256, 0, st>>>(pH, pY, pSlot, pW, T);
            dump_rows(pH, T);
        }
        if (lg_all) {
            pf_split(pH, nullptr, out_norm, kHidden, T);
            pf_gemm(head, vocab, kHidden, T, lg_all, vocab, 0);
        }
        // the last row through the head with the writing path's kernel, and the next token into the decode state
        mv_in({{head, logits}}, {vocab}, pH + size_t(T - 1) * kHidden, kHidden, out_norm);
        const int st0[1] = {p0 + T - 1};
        CK(cudaMemcpyAsync(state, st0, 4, cudaMemcpyHostToDevice, st));
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done2);
        CK(cudaStreamSynchronize(st));
        pos = p0 + T;
    }
    void prefill(const std::vector<int>& ids, int chunk) {
        for (size_t i = 0; i < ids.size(); i += chunk) prefill_chunk(ids.data() + i, int(std::min<size_t>(chunk, ids.size() - i)));
    }

    // the prompt GEMM against mv_in on layer 5's q/k/v rows for T random-ish rows (normed inputs)
    void gemm_test() {
        const int T = 100, K = kHidden, N = kQDim + 2 * kKvDim;
        std::vector<float> hx(size_t(T) * K);
        uint32_t seed = 1;
        for (auto& v : hx) { seed = seed * 1664525u + 1013904223u; v = (float(seed >> 8) / 16777216.f - 0.5f) * 4.f; }
        for (int i = 0; i < 40; ++i) hx[size_t(i * 7 % T) * K + (i * 131) % K] = 3000.f;   // a few big values, as Flint has
        float *X, *Y, *Y2, *xs_;
        uint16_t *xh, *xl;
        CK(cudaMalloc(&X, hx.size() * 4)); CK(cudaMalloc(&Y, size_t(T) * N * 4)); CK(cudaMalloc(&Y2, size_t(T) * N * 4));
        CK(cudaMalloc(&xh, hx.size() * 2)); CK(cudaMalloc(&xl, hx.size() * 2)); CK(cudaMalloc(&xs_, T * 4));
        CK(cudaMemcpy(X, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        const Layer& l = L[5];
        // reference: mat-vec per row on rmsnorm(x) * attn_norm (mv_in does the norm)
        for (int t = 0; t < T; ++t)
            mv_in({{l.wq, Y2 + size_t(t) * N}, {l.wk, Y2 + size_t(t) * N + kQDim}, {l.wv, Y2 + size_t(t) * N + kQDim + kKvDim}},
                  {kQDim, kKvDim, kKvDim}, X + size_t(t) * K, K, l.attn_norm);
        k_split16<<<T, 256, 0, st>>>(X, nullptr, l.attn_norm, K, xh, xl, xs_);
        for (int lo = 0; lo < 2; ++lo) {
            // q, k, v are separate tensors: three GEMMs into the columns of Y
            const uint8_t* ws[3] = {l.wq, l.wk, l.wv};
            const int ns[3] = {kQDim, kKvDim, kKvDim}, offs[3] = {0, kQDim, kQDim + kKvDim};
            for (int j = 0; j < 3; ++j) {
                const dim3 gr(ns[j] / 256, (T + 63) / 64);
                if (lo) k_gemm_q41<true><<<gr, 256, 0, st>>>(ws[j], ns[j], K, xh, xl, xs_, T, Y + offs[j], N, 0);
                else k_gemm_q41<false><<<gr, 256, 0, st>>>(ws[j], ns[j], K, xh, xl, xs_, T, Y + offs[j], N, 0);
            }
            CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
            std::vector<float> a(size_t(T) * N), b(size_t(T) * N);
            CK(cudaMemcpy(a.data(), Y, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), Y2, b.size() * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0, mx = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                num += double(a[i] - b[i]) * (a[i] - b[i]); den += double(b[i]) * b[i];
                mx = std::max(mx, (double) std::fabs(a[i] - b[i]) / (std::fabs(b[i]) + 1e-2));
            }
            std::printf("GEMM %s vs mat-vec: relative RMS error %.3e, max %.3e\n", lo ? "hi+lo" : "hi only", std::sqrt(num / den), mx);
        }
    }
    // one step: 8 launches a MoE layer
    // the next layer's fixed weights (attention, shared expert, router; its experts are not known yet) read on the side
    // stream while this layer runs, so they come from the memory-side cache. FLINT_PREFETCH=0 off, =N blocks.
    void prefetch(int i) {
        const Layer& l = L[i];
        Touch T{};
        auto add = [&](const void* p, size_t bytes) { T.p[T.n] = (const uint4*) p; T.n16[T.n] = uint32_t(bytes / 16); ++T.n; };
        const size_t r2k = size_t(kHidden / 32) * 20;
        add(l.wq, kQDim * r2k); add(l.wk, kKvDim * r2k); add(l.wv, kKvDim * r2k);
        add(l.wo, size_t(kHidden) * (kQDim / 32) * 20);
        if (i < kDense) {
            add(l.gate, kFfDense * r2k); add(l.up, kFfDense * r2k); add(l.down, size_t(kHidden) * (kFfDense / 32) * 20);
        } else {
            add(l.router, size_t(kExperts) * kHidden * 2);
            add(l.gate_s, kFfShared * r2k); add(l.up_s, kFfShared * r2k); add(l.down_s, size_t(kHidden) * (kFfShared / 32) * 20);
        }
        static const int nb = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : 16;
        k_touch<<<nb, 256, 0, side>>>(T, sink);
    }
    void record() {
        static const int pf = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : 16;
        k_embed<<<kHidden / 256, 256, 0, st>>>(embed, state, h);
        for (int i = 0; i < n_layers(); ++i) {
            Layer& l = L[i];
            if (pf && i + 1 < kLayers) {   // layer i + 1's weights, from the start of layer i
                CK(cudaEventRecord(ev[i], st));
                CK(cudaStreamWaitEvent(side, ev[i], 0));
                prefetch(i + 1);
            }
            mv_in({{l.wq, q}, {l.wk, k}, {l.wv, v}}, {kQDim, kKvDim, kKvDim}, h, kHidden, l.attn_norm);
            k_qkv_post<<<kHeads + kKv, kHd, 0, st>>>(q, k, v, l.q_norm, l.k_norm, l.K, l.V, state, l.swa);
            k_attn_part<<<dim3(kHeads, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa);
            k_attn_merge<<<kHeads, kHd, 0, st>>>(part, l.sinks, att);
            mv_in({{l.wo, h}}, {kHidden}, att, kQDim, nullptr, nullptr, true);
            if (i < kDense) {
                mv_in({{l.gate, f1}, {l.up, f2}}, {kFfDense, kFfDense}, h, kHidden, l.ffn_norm);
                mv_in({{l.down, h}}, {kHidden}, f1, kFfDense, nullptr, f2, true);
                if (std::getenv("FLINT_DUMP_H")) dump_rows(h, 1);
                continue;
            }
            k_router<<<kExperts / 4, 128, 0, st>>>(h, l.ffn_norm, l.router, rl, idx, wts, done);
            k_moe_gu<<<kActRows / kMvRows, 256, kHidden * 4, st>>>(h, l.ffn_norm, l.gate_e, l.up_e, l.gate_s, l.up_s, idx, act);
            k_moe_down<<<kHidden / kMvRows, 256, 0, st>>>(h, act, l.down_e, l.down_s, idx, wts);
            if (std::getenv("FLINT_DUMP_H")) dump_rows(h, 1);
        }
        if (pf) {   // join the side stream back (the capture must end on one stream)
            CK(cudaEventRecord(ev[kLayers], side));
            CK(cudaStreamWaitEvent(st, ev[kLayers], 0));
        }
        mv_in({{head, logits}}, {vocab}, h, kHidden, out_norm);
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done2);
    }
};


}  // namespace

int main(int argc, char** argv) {
    // two hardware queues unless set: with the runtime's default four, graph launches leave gaps between kernels
    setenv("GPU_MAX_HW_QUEUES", "2", 0);
    std::string path, ids_s, top_path, logits_path;
    int n_new = 0, ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--model") path = next();
        else if (a == "--ids") {
            ids_s = next();
            if (!ids_s.empty() && ids_s[0] == '@') {
                std::ifstream f(ids_s.substr(1));
                ids_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }
        }
        else if (a == "--new") n_new = std::stoi(next());
        else if (a == "--top") top_path = next();
        else if (a == "--logits") logits_path = next();   // every prompt position's logits, fp32
        else if (a == "--max-ctx" || a == "--max-context") ctx = std::stoi(next());
        else { std::fprintf(stderr, "usage: strata-flint --model FILE.gguf --ids 1,2,3|@FILE [--new N] [--top FILE] [--max-ctx N]\n"); return 2; }
    }
    if (path.empty() || ids_s.empty()) { std::fprintf(stderr, "strata-flint: --model and --ids are needed\n"); return 2; }
    std::vector<int> ids;
    std::stringstream ss(ids_s);
    for (std::string v; std::getline(ss, v, ',');) ids.push_back(std::stoi(v));

    const auto t0 = std::chrono::steady_clock::now();
    Model m(path, ctx);
    if (std::getenv("FLINT_GEMM_TEST")) { m.gemm_test(); return 0; }
    std::fprintf(stderr, "strata-flint: %.1f GB on the card in %.1f s\n", m.loaded / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    FILE* top = top_path.empty() ? nullptr : std::fopen(top_path.c_str(), "w");
    std::vector<float> lg(m.vocab);
    const auto t1 = std::chrono::steady_clock::now();
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    static const int chunk = std::getenv("FLINT_CHUNK") ? std::atoi(std::getenv("FLINT_CHUNK")) : 2048;
    const bool tokwise = std::getenv("FLINT_TOKWISE") != nullptr;
    if (lf && !tokwise) {   // the prompt path with every row's logits (small prompts: T x vocab floats on the card)
        float* lga = nullptr;
        CK(cudaMalloc(&lga, ids.size() * size_t(m.vocab) * 4));
        m.prefill_chunk(ids.data(), int(ids.size()), lga);
        std::vector<float> all(ids.size() * size_t(m.vocab));
        CK(cudaMemcpy(all.data(), lga, all.size() * 4, cudaMemcpyDeviceToHost));
        std::fwrite(all.data(), 4, all.size(), lf);
    } else if (tokwise) m.feed(ids, [&]() {
        if (lf) { CK(cudaMemcpy(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost)); std::fwrite(lg.data(), 4, lg.size(), lf); }
    });
    else m.prefill(ids, chunk);
    if (lf) std::fclose(lf);
    CK(cudaStreamSynchronize(m.st));
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto t2 = std::chrono::steady_clock::now();
    for (int n = 0; n < n_new; ++n) {
        const int nxt = m.next_token();
        if (top) {
            CK(cudaMemcpy(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
            double mx = lg[nxt], s = 0;
            for (float v : lg) s += std::exp(double(v) - mx);
            const double lse = mx + std::log(s);
            std::vector<int> o(lg.size());
            for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
            std::partial_sort(o.begin(), o.begin() + 5, o.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            for (int j = 0; j < 5; ++j) std::fprintf(top, "%d:%.5f%s", o[j], lg[o[j]] - lse, j < 4 ? " " : "\n");
        }
        std::printf("%d%s", nxt, n + 1 < n_new ? "," : "\n");
        std::fflush(stdout);
        m.step(-1);
    }
    CK(cudaStreamSynchronize(m.st));
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    if (top) std::fclose(top);
    std::fprintf(stderr, "strata-flint: read %zu in %.3f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n",
                 ids.size(), read_s, ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    return 0;
}
