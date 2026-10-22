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
#include <type_traits>
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
#ifndef FLINT_MINCHUNK
#define FLINT_MINCHUNK 64   // attention: at least this many keys a chunk (short contexts: fewer chunks to merge)
#endif
constexpr int kSwaRing = 16384;
// K and V cache element types (FLINT_K32 / FLINT_V32 at build time: fp32, for measuring what fp16 costs)
#ifdef FLINT_K32
typedef float kc_t;
#else
typedef __half kc_t;
#endif
#ifdef FLINT_V32
typedef float vc_t;
#else
typedef __half vc_t;
#endif
__device__ inline float c2f(__half v) { return __half2float(v); }
__device__ inline float c2f(float v) { return v; }
template <class T> __device__ inline T f2c(float v);
template <> __device__ inline __half f2c<__half>(float v) { return __float2half(v); }
template <> __device__ inline float f2c<float>(float v) { return v; }   // sliding layers' K/V slots: the window plus the largest prompt chunk (8192)
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

// the matrix units (gfx12 WMMA 16x16x16, fp16 in, fp32 sums)
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
// (blockIdx.z: the z-th of several tokens at positions state[0] + z, its q / k / v zs floats further on)
__global__ void k_qkv_post(float* q, const float* k, const float* v, const float* qn, const float* kn, kc_t* K,
                           vc_t* V, const int* state, bool rope, int kv_mask, int zs = 0) {
    __shared__ float sh[4];
    __shared__ float t[kHd];
    q += blockIdx.z * zs; k += blockIdx.z * zs; v += blockIdx.z * zs;
    const int pos = state[0] + blockIdx.z, hb = blockIdx.x, i = threadIdx.x;
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
        const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd + i;
        K[r] = f2c<kc_t>(y);
        V[r] = f2c<vc_t>(v[kh * kHd + i]);
    }
}

// Split attention for one query position: block (h, c) takes chunk c of the keys lo..pos (kAttnChunks equal parts)
// for query head h; each warp walks its keys with an online softmax (lane = 4 of the 128 dims), the 8 warps merge in
// LDS, and the chunk's (max, sum, unnormalized output) goes to part. k_attn_merge adds the chunks and the sink.
constexpr int kAttnChunks = 16, kPart = kHd + 4, kMinChunk = FLINT_MINCHUNK;   // a chunk's record: max, sum, 2 spare, 128 outputs
__device__ inline float4 h4(const float* p) { return *(const float4*) p; }
__device__ inline float4 h4(const __half* p) {
    const uint2 u = *(const uint2*) p;
    const float2 a = __half22float2(__builtin_bit_cast(__half2, u.x)), b = __half22float2(__builtin_bit_cast(__half2, u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__global__ void __launch_bounds__(256) k_attn_part(const float* q, const kc_t* K, const vc_t* V, float* part,
                                                   const int* state, bool swa, int kv_mask, int zs = 0) {
    q += blockIdx.z * zs;
    part += size_t(blockIdx.z) * kHeads * kAttnChunks * kPart;
    const int h = blockIdx.x, c = blockIdx.y, kh = h / (kHeads / kKv);
    const int pos = state[0] + blockIdx.z, lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks), a = lo + c * per, b = min(pos + 1, a + per);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float4 qv = *(const float4*) (q + h * kHd + lane * 4);
    const float scale = rsqrtf(float(kHd));
    float m = -INFINITY, l = 0;
    float4 o = make_float4(0, 0, 0, 0);
    for (int p = a + w; p < b; p += 8) {
        const float4 kv = h4(K + size_t(p & kv_mask) * kKvDim + kh * kHd + lane * 4);
        const float d = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w) * scale;
        const float mn = fmaxf(m, d), cf = expf(m - mn), e = expf(d - mn);
        const float4 vv = h4(V + size_t(p & kv_mask) * kKvDim + kh * kHd + lane * 4);
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

// Writing-path attention on the matrix units: the 8 query heads of a K/V head stand in for the "queries" of the prompt
// kernel (columns 0-7 of the 16), so each key and value is read once for all 8. Block (K/V head, chunk c of
// kAttnChunks): its 8 warps take the chunk's 16-key tiles in turn, each keeping its own online softmax; they merge in
// LDS and the chunk's record per query head goes to part (k_attn_merge adds the chunks and the sink, as before).
__global__ void __launch_bounds__(256) k_attn_dec_w(const float* __restrict__ q, const kc_t* __restrict__ K,
                                                    const vc_t* __restrict__ V, float* __restrict__ part,
                                                    const int* __restrict__ state, int swa, int kv_mask, int zs = 0) {
    q += blockIdx.z * zs;
    part += size_t(blockIdx.z) * kHeads * kAttnChunks * kPart;
    // a warp's V tile (transposed) during the loop; the same LDS holds the warps' outputs for the merge after it
    __shared__ __align__(16) char lds[8 * kHd * (16 + 8) * 2];
    _Float16 (*vt)[kHd][16 + 8] = (_Float16 (*)[kHd][16 + 8]) lds;
    float (*wo)[8][kHd] = (float (*)[8][kHd]) lds;
    __shared__ float wm[8][8], wl[8][8];
    const int kh = blockIdx.x, c = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int pos = state[0] + blockIdx.z, lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks), a0 = lo + c * per, b0 = min(pos + 1, a0 + per);
    const int hq = kh * (kHeads / kKv) + l16;                // this lane's query head (columns 8-15: none)
    pf_v8h qb[8];
#pragma unroll
    for (int s8 = 0; s8 < 8; ++s8)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s8][i] = (_Float16) (l16 < 8 ? q[hq * kHd + s8 * 16 + g * 8 + i] : 0.f);
    const float scale = rsqrtf(float(kHd));
    float m = -INFINITY, l = 0;
    pf_v8f o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    for (int t0 = a0 + warp * 16; t0 < b0; t0 += 8 * 16) {
        // V tile: lane takes key t0 + l16, dims g * 64 .. + 63, into vt transposed
        {
            const int p = t0 + l16;
            const bool ok = p < b0;
            const vc_t* vr = V + size_t(p & kv_mask) * kKvDim + kh * kHd + g * 64;
            if constexpr (sizeof(vc_t) == 2) {
#pragma unroll
                for (int d8 = 0; d8 < 8; ++d8) {
                    const pf_v8h vv = ok ? __builtin_bit_cast(pf_v8h, *(const uint4*) (vr + d8 * 8)) : pf_v8h{};
#pragma unroll
                    for (int i = 0; i < 8; ++i) vt[warp][g * 64 + d8 * 8 + i][l16] = vv[i];
                }
            } else
                for (int d = 0; d < 64; ++d) vt[warp][g * 64 + d][l16] = (_Float16) (ok ? c2f(vr[d]) : 0.f);
        }
        // S^T: A = K rows straight from the cache (lane: key t0 + l16, dims s * 16 + g * 8 ..)
        pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
        {
            const int p = t0 + l16;
            const bool ok = p < b0;
            const kc_t* kr = K + size_t(p & kv_mask) * kKvDim + kh * kHd;
#pragma unroll
            for (int s8 = 0; s8 < 8; ++s8) {
                pf_v8h ka;
                if constexpr (sizeof(kc_t) == 2) ka = ok ? __builtin_bit_cast(pf_v8h, *(const uint4*) (kr + s8 * 16 + g * 8)) : pf_v8h{};
                else
#pragma unroll
                    for (int i = 0; i < 8; ++i) ka[i] = (_Float16) (ok ? c2f(kr[s8 * 16 + g * 8 + i]) : 0.f);
                sc = pf_wmma(ka, qb[s8], sc);
            }
        }
        // lane: head column l16, keys t0 + g * 8 + i
        float mx = -INFINITY, v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int p = t0 + g * 8 + i;
            v[i] = (l16 < 8 && p < b0) ? sc[i] * scale : -INFINITY;
            mx = fmaxf(mx, v[i]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx), ref = mn == -INFINITY ? 0.f : mn;
        const float cf = expf(m - ref);
        m = mn;
        pf_v8h pb;
        float ps = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) { const float e = expf(v[i] - ref); ps += e; pb[i] = (_Float16) e; }
        l = l * cf + ps;
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j] = o[j] * cf;
            const pf_v8h va = *(const pf_v8h*) &vt[warp][j * 16 + l16][g * 8];
            o[j] = pf_wmma(va, pb, o[j]);
        }
        __builtin_amdgcn_wave_barrier();
    }
    l += __shfl_xor(l, 16, 32);
    __syncthreads();   // (every warp done with its V tile before the outputs overwrite them)
    // merge the 8 warps: lane (head l16 < 8, half g) has dims j * 16 + g * 8 + i
    if (l16 < 8) {
        if (g == 0) { wm[warp][l16] = m; wl[warp][l16] = l; }
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) wo[warp][l16][j * 16 + g * 8 + i] = o[j][i];
    }
    __syncthreads();
    // thread t: head t / 32 (0-7), dims (t % 32) * 4 .. + 3
    const int hh = tid >> 5, d0 = (tid & 31) * 4;
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w][hh]);
    float Ls = 0, O[4] = {};
    for (int w = 0; w < 8; ++w) {
        if (wm[w][hh] == -INFINITY) continue;
        const float f = expf(wm[w][hh] - M);
        Ls += wl[w][hh] * f;
        for (int i = 0; i < 4; ++i) O[i] += wo[w][hh][d0 + i] * f;
    }
    float* P = part + (size_t(kh * (kHeads / kKv) + hh) * kAttnChunks + c) * kPart;
    if (d0 == 0) { P[0] = M; P[1] = Ls; }
    for (int i = 0; i < 4; ++i) P[4 + d0 + i] = O[i];
}

__global__ void k_attn_merge(const float* part, const float* sinks, float* out) {
    part += size_t(blockIdx.z) * kHeads * kAttnChunks * kPart;
    out += blockIdx.z * kQDim;
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

// the experts the router picks from the layer's input (before attention: 84-87% of the real ones) read into the
// memory-side cache while attention runs; idx from a router launched on the side stream
__global__ void __launch_bounds__(256) k_touch_pred(const uint8_t* ge, const uint8_t* ue, const uint8_t* de, const int* idx,
                                                    unsigned* sink) {
    constexpr size_t mb = size_t(kFfExp) * (kHidden / 32) * 20;   // a matrix of one expert (gate, up and down alike)
    uint32_t a = 0;
    for (int k = 0; k < kTopK; ++k) {
        const size_t e = size_t(idx[k]);
        const uint4* p[3] = {(const uint4*) (ge + e * mb), (const uint4*) (ue + e * mb), (const uint4*) (de + e * mb)};
        for (int j = 0; j < 3; ++j)
            for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < mb / 16; i += gridDim.x * 256) { const uint4 v = p[j][i]; a ^= v.x ^ v.w; }
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
// rows of x (T x K fp32; rmsnorm * nw first when nw; silu(x) * x2 when x2) to hi / lo fp16 and the row's scale back
// (input row t at (t / per) * gstride + (t % per) * K: rows packed in groups of per, e.g. a token's 6 expert slots
// inside its 4096-value act row; the defaults read rows back to back)
__global__ void __launch_bounds__(256) k_split16(const float* x, const float* x2, const float* nw, int K, uint16_t* xh,
                                                 uint16_t* xl, float* xs, int per = 1, int gstride = 0) {
    const int t = blockIdx.x;
    const size_t off = size_t(t / per) * (gstride ? gstride : K) + size_t(t % per) * K;
    const float* r = x + off;
    const float* r2 = x2 ? x2 + off : nullptr;
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
__global__ void k_pf_qkpost(float* QKV, const float* qn, const float* kn, kc_t* K, vc_t* V, int p0, bool rope, int kv_mask) {
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
        const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd + i;
        K[r] = f2c<kc_t>(y);
        V[r] = f2c<vc_t>(row[kQDim + kKvDim + kh * kHd + i]);
    }
}


// Attention for T queries on the matrix units: block = (K/V head, 16 queries), warp w = query head kh * 8 + w, so the 8
// heads sharing a K/V head read each key tile once. Per 16 keys a warp forms S^T = K Q^T (keys x queries: its C layout
// gives each lane one query's 8 keys, which is already the B operand of the next product), takes the online softmax,
// and adds O^T += V^T P^T (dims x queries). K and V go through LDS as fp16 (V transposed), Q and P are fp16, sums fp32.
__global__ void __launch_bounds__(256) k_pf_attn_w(const float* __restrict__ QKV, const kc_t* __restrict__ K,
                                                   const vc_t* __restrict__ V, const float* __restrict__ sinks,
                                                   float* __restrict__ ATT, int p0, int T, int swa, int kv_mask) {
    constexpr int KT = 64;
    __shared__ __align__(16) _Float16 ks[KT][kHd + 8];
    __shared__ __align__(16) _Float16 vt[kHd][KT + 8];
    const int kh = blockIdx.x, q0 = blockIdx.y * 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int h = kh * (kHeads / kKv) + warp;
    const int tq = q0 + l16;                                  // this lane's query
    const bool live = tq < T;
    const int qpos = p0 + min(tq, T - 1);
    const int qlo = swa ? max(0, qpos - (kWindow - 1)) : 0;
    // Q^T as the B operand: lane (query l16, half g) holds dims s * 16 + g * 8 .. + 7 for the 8 steps s
    pf_v8h qb[8];
    {
        const float* qr = QKV + size_t(min(tq, T - 1)) * (kQDim + 2 * kKvDim) + h * kHd;
#pragma unroll
        for (int s8 = 0; s8 < 8; ++s8)
#pragma unroll
            for (int i = 0; i < 8; ++i) qb[s8][i] = (_Float16) qr[s8 * 16 + g * 8 + i];
    }
    const float scale = rsqrtf(float(kHd));
    float m = -INFINITY, l = 0;
    pf_v8f o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const int last = p0 + min(q0 + 15, T - 1);
    const int first = swa ? max(0, p0 + q0 - (kWindow - 1)) : 0;
    for (int k0 = first; k0 <= last; k0 += KT) {
        // K tile (keys x dims) and V^T (dims x keys) into LDS as fp16: 64 keys x 128 dims, 32 values a thread
        {
            const int key = tid >> 2, c0 = (tid & 3) * 32, p = k0 + key;
            const bool ok = p <= last;
            const kc_t* kr = K + size_t(p & kv_mask) * kKvDim + kh * kHd + c0;
            const vc_t* vr = V + size_t(p & kv_mask) * kKvDim + kh * kHd + c0;
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                ks[key][c0 + c] = (_Float16) (ok ? c2f(kr[c]) : 0.f);
                vt[c0 + c][key] = (_Float16) (ok ? c2f(vr[c]) : 0.f);
            }
        }
        pf_sync();
#pragma unroll
        for (int sub = 0; sub < KT / 16; ++sub) {
            // S^T for keys k0 + sub * 16 .. + 15: A = K rows (lane l16 = key, half g = dims), B = Q^T
            pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s8 = 0; s8 < 8; ++s8) {
                const pf_v8h ka = *(const pf_v8h*) &ks[sub * 16 + l16][s8 * 16 + g * 8];
                sc = pf_wmma(ka, qb[s8], sc);
            }
            // lane: query l16, keys sub * 16 + g * 8 + i
            float mx = -INFINITY;
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int p = k0 + sub * 16 + g * 8 + i;
                v[i] = (live && p >= qlo && p <= qpos) ? sc[i] * scale : -INFINITY;
                mx = fmaxf(mx, v[i]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            // (no branch around the products: a matrix instruction needs every lane; a query with nothing visible yet
            // keeps m = -inf and adds zeros)
            const float mn = fmaxf(m, mx), ref = mn == -INFINITY ? 0.f : mn;
            const float cf = expf(m - ref);
            m = mn;
            pf_v8h pb;
            float ps = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { const float e = expf(v[i] - ref); ps += e; pb[i] = (_Float16) e; }
            l = l * cf + ps;
            // O^T += V^T P^T: A = V^T rows (lane l16 = dim within tile j, half g = keys), B = P^T (this lane's 8 keys)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                o[j] = o[j] * cf;
                const pf_v8h va = *(const pf_v8h*) &vt[j * 16 + l16][sub * 16 + g * 8];
                o[j] = pf_wmma(va, pb, o[j]);
            }
        }
        pf_sync();
    }
    // the two halves' sums, the sink, and the output: lane (query l16, half g) has dims j * 16 + g * 8 + i
    l += __shfl_xor(l, 16, 32);
    if (!live) return;
    const float sk = sinks[h], M = fmaxf(m, sk);
    const float f = expf(m - M), L = l * f + expf(sk - M);
    float* out = ATT + size_t(tq) * kQDim + h * kHd;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) out[j * 16 + g * 8 + i] = o[j][i] * f / L;
}


// ---------------------------------------------------------------------------------------------------- checking drafts
// Up to kVMax tokens at positions state[0] .. state[0] + T - 1 in one pass (the current token and the drafted ones):
// every weight is read once and used for all T, each routed expert once for the tokens that chose it.
constexpr int kVMax = 8, kVUniq = kVMax * kTopK;
// X[t] = rmsnorm(H[t]) * nw over 2048 (block t)
__global__ void k_v_norm(const float* H, const float* nw, float* X) {
    __shared__ float red[8];
    const float* h = H + size_t(blockIdx.x) * kHidden;
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) ss += h[i] * h[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < 8; ++i) t += red[i];
    const float r = rsqrtf(t / kHidden + kEps);
    for (int i = threadIdx.x; i < kHidden; i += 256) X[size_t(blockIdx.x) * kHidden + i] = h[i] * r * nw[i];
}
// out[t][i] = silu(g[t][i]) * u[t][i] (n a token, rows at stride ld)
__global__ void k_v_swiglu(const float* g, const float* u, float* out, int n, int ld) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[size_t(t) * ld + i] = silu(g[size_t(t) * ld + i]) * u[size_t(t) * ld + i];
}
// R rows of a Q4_1 matrix against T inputs x_t = X + t * ldx (global, L2-cached); lane 0 of each row's warp-sum
template <int R>
__device__ inline void q4_rows_t(const uint8_t* W, size_t rowb, int nb, const float* X, int ldx, int T, int lane,
                                 float (*acc)[kVMax]) {
    const int qd = lane & 3;
    for (int b = lane >> 2; b < nb; b += 8) {
        uint32_t dmv[R], qv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* B = W + r * rowb + size_t(b) * 20;
            dmv[r] = *(const uint32_t*) B;
            qv[r] = *(const uint32_t*) (B + 4 + qd * 4);
        }
        for (int t = 0; t < T; ++t) {
            const float* xs = X + size_t(t) * ldx;
            const float4 xa = *(const float4*) (xs + b * 32 + qd * 4);
            const float4 xb = *(const float4*) (xs + b * 32 + 16 + qd * 4);
            const float sx = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t dm = dmv[r], q = qv[r];
                const float d = __half2float(__ushort_as_half(uint16_t(dm))), m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
                const float tt = float(q & 15) * xa.x + float((q >> 8) & 15) * xa.y + float((q >> 16) & 15) * xa.z + float((q >> 24) & 15) * xa.w
                               + float((q >> 4) & 15) * xb.x + float((q >> 12) & 15) * xb.y + float((q >> 20) & 15) * xb.z + float(q >> 28) * xb.w;
                acc[r][t] += d * tt + m * sx;
            }
        }
    }
}
// The same with the T inputs staged through LDS 256 values at a time (sb: T x 256 floats), T a compile-time count so
// the per-token loops unroll and every load of a step is in flight together. Each lane takes one Q4_1 block a slice
// (4 lanes a block, 8 blocks a slice). xp[t]: token t's input row (nullptr: zeros). All threads of the block call it.
template <int R, int T>
__device__ inline void q4_rows_s(const uint8_t* W, size_t rowb, int nb, const float* const* xp, float* sb, int lane,
                                 float (*acc)[kVMax]) {
    const int qd = lane & 3, bl = lane >> 2;
    for (int kc = 0; kc < nb * 32; kc += 256) {
        uint32_t dmv[R], qv[R];
        const int b = kc / 32 + bl;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* B = W + r * rowb + size_t(b) * 20;
            dmv[r] = *(const uint32_t*) B;
            qv[r] = *(const uint32_t*) (B + 4 + qd * 4);
        }
        for (int i = threadIdx.x; i < T * 64; i += 256) {
            const int t = i >> 6, c = (i & 63) * 4;
            *(float4*) (sb + t * 256 + c) = xp[t] ? *(const float4*) (xp[t] + kc + c) : make_float4(0, 0, 0, 0);
        }
        __syncthreads();
        // each row's 8 weights unpacked once (d * q + m, fp32), then 8 products a token
        float wv[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t dm = dmv[r], q = qv[r];
            const float d = __half2float(__ushort_as_half(uint16_t(dm))), m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
#pragma unroll
            for (int i = 0; i < 4; ++i) { wv[r][i] = d * float((q >> (8 * i)) & 15) + m; wv[r][4 + i] = d * float((q >> (8 * i + 4)) & 15) + m; }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float* xs = sb + t * 256 + bl * 32;
            const float4 xa = *(const float4*) (xs + qd * 4), xb = *(const float4*) (xs + 16 + qd * 4);
#pragma unroll
            for (int r = 0; r < R; ++r)
                acc[r][t] += wv[r][0] * xa.x + wv[r][1] * xa.y + wv[r][2] * xa.z + wv[r][3] * xa.w
                           + wv[r][4] * xb.x + wv[r][5] * xb.y + wv[r][6] * xb.z + wv[r][7] * xb.w;
        }
        __syncthreads();
    }
}

// R rows (each its own pointer: gate and up rows can share a pass) against nt <= T inputs xp[0..nt), staged through LDS
// 1024 values at a time (sb: T x 1024 floats); each weight is unpacked once for all nt. acc[r][i]: input i's partial.
constexpr int kVSl = 1024;
template <int R, int T>
__device__ inline void q4_rows_p(const uint8_t* const* W, int nb, const float* const* xp, int nt, float* sb, int lane,
                                 float (*acc)[T]) {
    const int qd = lane & 3, bl = lane >> 2;
    for (int kc = 0; kc < nb * 32; kc += kVSl) {
        const int kn = min(kVSl, nb * 32 - kc);
        for (int i = threadIdx.x; i < nt * (kn / 4); i += 256) {
            const int t = i / (kn / 4), c = (i % (kn / 4)) * 4;
            *(float4*) (sb + t * kVSl + c) = *(const float4*) (xp[t] + kc + c);
        }
        __syncthreads();
        for (int b = kc / 32 + bl; b < kc / 32 + kn / 32; b += 8) {
            uint32_t dmv[R], qv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t* B = W[r] + size_t(b) * 20;
                dmv[r] = *(const uint32_t*) B;
                qv[r] = *(const uint32_t*) (B + 4 + qd * 4);
            }
            const int off = b * 32 - kc;
#pragma unroll
            for (int r0 = 0; r0 < R; r0 += 4) {   // (4 rows unpacked at a time: registers)
                float wv[4][8];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const uint32_t dm = dmv[r0 + r], q = qv[r0 + r];
                    const float d = __half2float(__ushort_as_half(uint16_t(dm))), m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
#pragma unroll
                    for (int i = 0; i < 4; ++i) { wv[r][i] = d * float((q >> (8 * i)) & 15) + m; wv[r][4 + i] = d * float((q >> (8 * i + 4)) & 15) + m; }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (t >= nt) break;
                    const float* xs = sb + t * kVSl + off;
                    const float4 xa = *(const float4*) (xs + qd * 4), xb = *(const float4*) (xs + 16 + qd * 4);
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        acc[r0 + r][t] += wv[r][0] * xa.x + wv[r][1] * xa.y + wv[r][2] * xa.z + wv[r][3] * xa.w
                                        + wv[r][4] * xb.x + wv[r][5] * xb.y + wv[r][6] * xb.z + wv[r][7] * xb.w;
                }
            }
        }
        __syncthreads();
    }
}

// Up to three matrices on T inputs, 32 rows a block: y[j] + t * ldy + r (= or += when acc)
struct MvV {
    const uint8_t* w[3];
    float* y[3];
    int rows[3], nmat;
    const float* X;
    int ldx, n_in, T, ldy;
    bool acc;
};
template <int T>
__global__ void __launch_bounds__(256) k_v_mv(MvV J) {
    __shared__ __align__(16) float sb[T * kVSl];
    int rb = blockIdx.x * kMvRows, j = 0;
    while (j < J.nmat - 1 && rb >= J.rows[j]) rb -= J.rows[j++];
    const int lane = threadIdx.x & 31, r0 = rb + (threadIdx.x >> 5) * kMvRpw, nb = J.n_in / 32;
    const size_t rowb = size_t(nb) * 20;
    float acc[kMvRpw][T] = {};
    const float* xp[T];
#pragma unroll
    for (int t = 0; t < T; ++t) xp[t] = J.X + size_t(t) * J.ldx;
    const uint8_t* wr[kMvRpw];
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) wr[r] = J.w[j] + size_t(r0 + r) * rowb;
    q4_rows_p<kMvRpw, T>(wr, nb, xp, T, sb, lane, acc);
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) { float* y = J.y[j] + size_t(t) * J.ldy + r0 + r; *y = J.acc ? *y + v : v; }
        }
}
// Router logits for T tokens (one warp an expert row, 4 a block), then in the last block: each token's top 6 (the
// writing path's rule) and the distinct experts among them: un of them, ue[u] the expert, um[u] a bitmask of the tokens
// that chose it, vu[t][k] the distinct index of token t's k-th choice.
__global__ void __launch_bounds__(128) k_v_router(const float* X, const uint16_t* W, int T, float* logits, int* idx,
                                                  float* wts, int* ue, int* um, int* vu, int* un, unsigned* done) {
    __shared__ bool last;
    const int lane = threadIdx.x & 31, e = blockIdx.x * 4 + (threadIdx.x >> 5);
    for (int t = 0; t < T; ++t) {
        const float* x = X + size_t(t) * kHidden;
        float acc = 0;
        for (int i = lane * 8; i < kHidden; i += 256) {
            const uint4 w = *(const uint4*) (W + size_t(e) * kHidden + i);
            const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) acc += __uint_as_float(ww[k] << 16) * x[i + 2 * k] + __uint_as_float(ww[k] & 0xffff0000u) * x[i + 2 * k + 1];
        }
        acc = warp_sum(acc);
        if (lane == 0) logits[t * kExperts + e] = acc;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int w = threadIdx.x >> 5;
    for (int t = w; t < T; t += 4) {
        float sc[8];
        for (int j = 0; j < 8; ++j) sc[j] = 1.f / (1.f + expf(-((volatile float*) logits)[t * kExperts + lane * 8 + j]));
        float tot = 0, pv[kTopK];
        int pi[kTopK];
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
            pv[k] = v; pi[k] = i; tot += v;
        }
        if (lane == 0)
            for (int k = 0; k < kTopK; ++k) { idx[t * kTopK + k] = pi[k]; wts[t * kTopK + k] = pv[k] / tot; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < kTopK; ++k) {
                const int x = ((volatile int*) idx)[t * kTopK + k];
                int u = 0;
                while (u < n && ue[u] != x) ++u;
                if (u == n) { ue[n] = x; um[n] = 0; ++n; }
                um[u] |= 1 << t;
                vu[t * kTopK + k] = u;
            }
        *un = n;
        *done = 0;
    }
}
// MoE gate/up for T tokens: grid.y = distinct expert u (or kVUniq: the shared expert), 32 rows a block; act[t] holds
// the token's 6 experts at k * 512 (k its choice) and the shared expert at 3072
template <int T>
__global__ void __launch_bounds__(256) k_v_moe_gu(const float* X, const uint8_t* ge, const uint8_t* ue_, const uint8_t* gs,
                                                  const uint8_t* us, const int* ue, const int* um, const int* vu,
                                                  const int* un, float* act) {
    __shared__ __align__(16) float sb[T * kVSl];
    const int u = blockIdx.y;
    const bool shared = u == kVUniq;
    if (!shared && (u >= *un || blockIdx.x * kMvRows >= kFfExp)) return;
    const int lane = threadIdx.x & 31, row = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    constexpr size_t rowb = size_t(kHidden / 32) * 20;
    const uint8_t *G, *U;
    if (shared) { G = gs + size_t(row) * rowb; U = us + size_t(row) * rowb; }
    else { const size_t e = size_t(ue[u]) * kFfExp + row; G = ge + e * rowb; U = ue_ + e * rowb; }
    // the tokens using this expert, packed (the shared one: all)
    int tl[kVMax], nt = 0;
    const int mask = shared ? (1 << T) - 1 : um[u];
    for (int t = 0; t < T; ++t) if (mask >> t & 1) tl[nt++] = t;
    // gate rows 0-3 and up rows 4-7 in one pass over the masked tokens' inputs only (packed: input i is token tl[i])
    float acc[2 * kMvRpw][T] = {};
    const float* xp[T];
#pragma unroll
    for (int i = 0; i < T; ++i) xp[i] = i < nt ? X + size_t(tl[i]) * kHidden : nullptr;
    const uint8_t* wr[2 * kMvRpw];
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) { wr[r] = G + size_t(r) * rowb; wr[kMvRpw + r] = U + size_t(r) * rowb; }
    q4_rows_p<2 * kMvRpw, T>(wr, kHidden / 32, xp, nt, sb, lane, acc);
    for (int i = 0; i < nt; ++i) {
        const int t = tl[i];
        int k = 0;
        if (!shared) while (vu[t * kTopK + k] != u) ++k;
        float* a = act + size_t(t) * kActRows + (shared ? kTopK * kFfExp : k * kFfExp);
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) {
            const float gv = warp_sum(acc[r][i]), uv = warp_sum(acc[kMvRpw + r][i]);
            if (lane == 0) a[row + r] = silu(gv) * uv;
        }
    }
}
// MoE down for T tokens: 32 output rows a block; each distinct expert's rows read once for the tokens that chose it
template <int T>
__global__ void __launch_bounds__(256) k_v_moe_down(float* H, const float* act, const uint8_t* de, const uint8_t* ds,
                                                    const int* ue, const int* um, const int* vu, const float* wts,
                                                    const int* un) {
    __shared__ __align__(16) float sb[T * kVSl];
    const int lane = threadIdx.x & 31, r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    constexpr size_t rowe = size_t(kFfExp / 32) * 20, rows_ = size_t(kFfShared / 32) * 20;
    float tot[kMvRpw][T] = {};
    const int n = *un;
    for (int u = 0; u < n; ++u) {
        const int e = ue[u], mask = um[u];
        const float* xp[T];
        float wk[T];
        int tl[T], nt = 0;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            xp[t] = nullptr; wk[t] = 0.f; tl[t] = 0;
            if (mask >> t & 1) {
                int k = 0;
                while (vu[t * kTopK + k] != u) ++k;
                xp[nt] = act + size_t(t) * kActRows + k * kFfExp;
                wk[nt] = wts[t * kTopK + k];
                tl[nt++] = t;
            }
        }
        float a[kMvRpw][T] = {};
        const uint8_t* wr[kMvRpw];
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) wr[r] = de + (size_t(e) * kHidden + r0 + r) * rowe;
        q4_rows_p<kMvRpw, T>(wr, kFfExp / 32, xp, nt, sb, lane, a);
        for (int i = 0; i < nt; ++i)
#pragma unroll
            for (int r = 0; r < kMvRpw; ++r) tot[r][tl[i]] += wk[i] * a[r][i];
    }
    const float* xs_[T];
#pragma unroll
    for (int t = 0; t < T; ++t) xs_[t] = act + size_t(t) * kActRows + kTopK * kFfExp;
    const uint8_t* wsr[kMvRpw];
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) wsr[r] = ds + size_t(r0 + r) * rows_;
    q4_rows_p<kMvRpw, T>(wsr, kFfShared / 32, xs_, T, sb, lane, tot);
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) {
            const float v = warp_sum(tot[r][t]);
            if (lane == 0) H[size_t(t) * kHidden + r0 + r] += v;
        }
}
__global__ void k_v_embed(const float* table, const int* toks, float* H) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < kHidden; i += 256) H[size_t(t) * kHidden + i] = table[size_t(toks[t]) * kHidden + i];
}
// each row's most likely token (block t over row t of the T x n logits)
__global__ void __launch_bounds__(1024) k_argmax_rows(const float* lg, int n, int* best) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* l = lg + size_t(blockIdx.x) * n;
    float v = -INFINITY;
    int i = 0;
    for (int j = threadIdx.x; j < n; j += 1024) if (l[j] > v) { v = l[j]; i = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o, 32);
        const int oi = __shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w) if (bv[w] > v || (bv[w] == v && bi[w] < i)) { v = bv[w]; i = bi[w]; }
        best[blockIdx.x] = i;
    }
}


// ------------------------------------------------ the checking pass on the matrix units
// T <= 16 tokens as the 16 columns of a WMMA tile: a warp takes 16 weight rows, the inputs (fp16 hi and lo, a scale a
// token, as on the prompt path) go through LDS 256 values at a time. 4 warps a block, 64 rows.
constexpr int kVgRows = 64, kVgSlice = 256, kVgLd = kVgSlice + 8;
struct VgIn { const uint16_t *xh, *xl; const float* xs; int ld; };   // a token's input row t: xh + t * ld ...
// 16 weight rows x 16 token columns, K split over the block's 4 warps (warp w: values w * K / 4 ..): A from the
// Q4_1 rows, B straight from the inputs' hi / lo rows (small, cached), no LDS until the 4 partial tiles are added.
// xrow[t]: input row of column t (-1: none). Returns the sum in warp 0 (C layout: lane l16 = token, rows g * 8 + i).
__device__ inline pf_v8f vg_tile(const uint8_t* Wrow, int K, const VgIn& in, const int* xrow, float (*red)[32][8]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, l16 = lane & 15, g = lane >> 4;
    const int row = xrow[l16];
    const uint16_t* bh = row >= 0 ? in.xh + size_t(row) * in.ld : nullptr;
    const uint16_t* bl = row >= 0 ? in.xl + size_t(row) * in.ld : nullptr;
    pf_v8f acc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const int k0 = warp * (K / 4), k1 = k0 + K / 4;
#pragma unroll 4
    for (int k = k0; k < k1; k += 16) {
        const pf_v8h a = pf_q41_a8(Wrow, k + g * 8);
        const pf_v8h h = bh ? *(const pf_v8h*) (bh + k + g * 8) : pf_v8h{};
        const pf_v8h l = bl ? *(const pf_v8h*) (bl + k + g * 8) : pf_v8h{};
        acc = pf_wmma(a, h, acc);
        acc = pf_wmma(a, l, acc);
    }
    if (warp) for (int i = 0; i < 8; ++i) red[warp][lane][i] = acc[i];
    pf_sync();
    if (!warp) for (int w = 1; w < 4; ++w) for (int i = 0; i < 8; ++i) acc[i] += red[w][lane][i];
    pf_sync();
    return acc;
}
// dense: up to three matrices, 64 rows a block; y[j][t * ldy + r] (= or += when add) for t < T
struct VgJob { const uint8_t* w[3]; float* y[3]; int rows[3], nmat, K, T, ldy, add; VgIn in; };
__global__ void __launch_bounds__(128) k_vg(VgJob J) {
    __shared__ float red[4][32][8];
    __shared__ int xrow[16];
    int rb = blockIdx.x * 16, j = 0;
    while (j < J.nmat - 1 && rb >= J.rows[j]) rb -= J.rows[j++];
    const int lane = threadIdx.x & 31, l16 = lane & 15, g = lane >> 4, r0 = rb;
    if (threadIdx.x < 16) xrow[threadIdx.x] = threadIdx.x < J.T ? int(threadIdx.x) : -1;
    __syncthreads();
    const size_t rowb = size_t(J.K / 32) * 20;
    const pf_v8f acc = vg_tile(J.w[j] + size_t(r0 + l16) * rowb, J.K, J.in, xrow, red);
    if ((threadIdx.x >> 5) || l16 >= J.T) return;
    const float sc = J.in.xs[l16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r0 + g * 8 + i;
        if (r >= J.rows[j]) break;
        float* y = J.y[j] + size_t(l16) * J.ldy + r;
        const float v = acc[i] * sc;
        *y = J.add ? *y + v : v;
    }
}
// experts' gate/up: grid (row block of 64, distinct expert u or kVUniq = shared); act[t][k * 512 + r] = silu(g) * u for
// the tokens that chose u (k: their choice index), act[t][3072 + r] for the shared expert
__global__ void __launch_bounds__(128) k_vg_gu(VgIn in, const uint8_t* ge, const uint8_t* ue_, const uint8_t* gs,
                                               const uint8_t* us, const int* ue, const int* um, const int* vu,
                                               const int* un, int T, float* act) {
    __shared__ float red[4][32][8];
    __shared__ int xrow[16];
    const int u = blockIdx.y;
    const bool shared = u == kVUniq;
    if (!shared && (u >= *un || blockIdx.x * 16 >= kFfExp)) return;
    const int lane = threadIdx.x & 31, l16 = lane & 15, g = lane >> 4, r0 = blockIdx.x * 16;
    const int mask = shared ? (1 << T) - 1 : um[u];
    if (threadIdx.x < 16) xrow[threadIdx.x] = (threadIdx.x < T && (mask >> threadIdx.x & 1)) ? int(threadIdx.x) : -1;
    __syncthreads();
    constexpr size_t rowb = size_t(kHidden / 32) * 20;
    const uint8_t *G, *U;
    if (shared) { G = gs + size_t(r0 + l16) * rowb; U = us + size_t(r0 + l16) * rowb; }
    else { const size_t e = size_t(ue[u]) * kFfExp + r0 + l16; G = ge + e * rowb; U = ue_ + e * rowb; }
    const pf_v8f ag = vg_tile(G, kHidden, in, xrow, red);
    const pf_v8f au = vg_tile(U, kHidden, in, xrow, red);
    if ((threadIdx.x >> 5) || xrow[l16] < 0) return;
    const int t = l16;
    int k = 0;
    if (!shared) while (vu[t * kTopK + k] != u) ++k;
    const float sc = in.xs[t];
    float* a = act + size_t(t) * kActRows + (shared ? kTopK * kFfExp : k * kFfExp) + r0 + g * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = silu(ag[i] * sc) * (au[i] * sc);
}
// experts' down: grid (row block of 64, distinct expert u or kVUniq = shared); Y[t][k][r] (k = 6: shared). The inputs
// are the act rows split per 512 (expert k of token t: row t * 6 + k) and per 1024 (the shared one: row t), in two sets
__global__ void __launch_bounds__(128) k_vg_down(VgIn ine, VgIn ins, const uint8_t* de, const uint8_t* ds, const int* ue,
                                                 const int* um, const int* vu, const int* un, int T, float* Y) {
    __shared__ float red[4][32][8];
    __shared__ int xrow[16], kof[16];
    const int u = blockIdx.y;
    const bool shared = u == kVUniq;
    if (!shared && u >= *un) return;
    const int lane = threadIdx.x & 31, l16 = lane & 15, g = lane >> 4, r0 = blockIdx.x * 16;
    if (threadIdx.x < 16) {
        const int t = threadIdx.x;
        int row = -1, k = kTopK;
        if (t < T && shared) row = t;
        else if (t < T && (um[u] >> t & 1)) { k = 0; while (vu[t * kTopK + k] != u) ++k; row = t * kTopK + k; }
        xrow[t] = row; kof[t] = k;
    }
    __syncthreads();
    const pf_v8f acc = shared ? vg_tile(ds + size_t(r0 + l16) * (kFfShared / 32) * 20, kFfShared, ins, xrow, red)
                              : vg_tile(de + (size_t(ue[u]) * kHidden + r0 + l16) * (kFfExp / 32) * 20, kFfExp, ine, xrow, red);
    if ((threadIdx.x >> 5) || xrow[l16] < 0) return;
    const float sc = (shared ? ins.xs : ine.xs)[xrow[l16]];
    float* y = Y + (size_t(l16) * (kTopK + 1) + kof[l16]) * kHidden + r0 + g * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = acc[i] * sc;
}
// H[t] += sum_k w[t][k] * Y[t][k] + Y[t][6] (block t)
__global__ void k_vg_combine(float* H, const float* Y, const float* wts) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        float s = Y[(size_t(t) * (kTopK + 1) + kTopK) * kHidden + i];
        for (int k = 0; k < kTopK; ++k) s += wts[t * kTopK + k] * Y[(size_t(t) * (kTopK + 1) + k) * kHidden + i];
        H[size_t(t) * kHidden + i] += s;
    }
}
// router logits for T tokens, fp32: a warp an expert row (4 a block), the T normed rows staged 256 values at a time
template <int T>
__global__ void __launch_bounds__(128) k_vr(const float* X, const uint16_t* W, float* logits) {
    __shared__ __align__(16) float sb[T][256];
    const int lane = threadIdx.x & 31, e = blockIdx.x * 4 + (threadIdx.x >> 5);
    float acc[T] = {};
    for (int kc = 0; kc < kHidden; kc += 256) {
        for (int i = threadIdx.x; i < T * 64; i += 128) { const int t = i >> 6, c = (i & 63) * 4; *(float4*) &sb[t][c] = *(const float4*) (X + size_t(t) * kHidden + kc + c); }
        __syncthreads();
        const uint4 w = *(const uint4*) (W + size_t(e) * kHidden + kc + lane * 8);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[t] += __uint_as_float(ww[k] << 16) * sb[t][lane * 8 + 2 * k] + __uint_as_float(ww[k] & 0xffff0000u) * sb[t][lane * 8 + 2 * k + 1];
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < T; ++t) { const float v = warp_sum(acc[t]); if (lane == 0) logits[t * kExperts + e] = v; }
}
// each token's top 6 and the distinct experts (one block of 128: k_v_router's last-block part)
__global__ void __launch_bounds__(128) k_vr_pick(const float* logits, int T, int* idx, float* wts, int* ue, int* um, int* vu, int* un) {
    __shared__ int sidx[kVMax * kTopK], sue[kVUniq], sum_[kVUniq];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int t = w; t < T; t += 4) {
        float sc[8];
        for (int j = 0; j < 8; ++j) sc[j] = 1.f / (1.f + expf(-logits[t * kExperts + lane * 8 + j]));
        float tot = 0, pv[kTopK];
        int pi[kTopK];
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
            pv[k] = v; pi[k] = i; tot += v;
        }
        if (lane == 0) for (int k = 0; k < kTopK; ++k) { sidx[t * kTopK + k] = pi[k]; idx[t * kTopK + k] = pi[k]; wts[t * kTopK + k] = pv[k] / tot; }
    }
    // the distinct experts in parallel (a serial loop here took 60 us): each expert's token mask, then their order
    __shared__ int emask[kExperts], epos[kExperts], cnt;
    for (int e = threadIdx.x; e < kExperts; e += 128) emask[e] = 0;
    __syncthreads();
    if (threadIdx.x < T * kTopK) atomicOr(&emask[sidx[threadIdx.x]], 1 << (threadIdx.x / kTopK));
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int e = 0; e < kExperts; ++e) if (emask[e]) { epos[e] = n; sue[n] = e; sum_[n] = emask[e]; ++n; }
        cnt = n;
    }
    __syncthreads();
    const int n = cnt;
    for (int u = threadIdx.x; u < n; u += 128) { ue[u] = sue[u]; um[u] = sum_[u]; }
    if (threadIdx.x < T * kTopK) vu[threadIdx.x] = epos[sidx[threadIdx.x]];
    if (threadIdx.x == 0) *un = n;
}

// (measuring) how many of the experts chosen from the layer's input (before attention) are among the real ones
__global__ void k_pred_overlap(const int* pred, const int* real, unsigned* cnt) {
    if (threadIdx.x) return;
    int hit = 0;
    for (int a = 0; a < kTopK; ++a) for (int b = 0; b < kTopK; ++b) hit += pred[a] == real[b];
    atomicAdd(cnt, unsigned(hit));
    atomicAdd(cnt + 1, unsigned(kTopK));
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
    kc_t* K = nullptr;   // [slot][K/V head][128]
    vc_t* V = nullptr;
    int kv_mask = 0x7fffffff;              // slot = position & kv_mask (a ring on the sliding layers)
};

struct Model {
    std::unique_ptr<strata::GgufFile> g;
    std::vector<Layer> L;
    const float *embed, *out_norm;
    const uint8_t* head;
    int vocab = 0, max_ctx;
    size_t loaded = 0, kv_bytes = 0;
    cudaStream_t st, side;
    std::vector<cudaEvent_t> ev;
    unsigned* sink;
    // scratch
    int* pidx = nullptr;
    float *pw = nullptr, *prl = nullptr;
    unsigned* pcnt = nullptr;
    unsigned* pdone = nullptr;
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
            // the sliding layers look back 513 positions: a ring of kSwaRing slots (a prompt chunk plus the window fits)
            static const int ring = std::getenv("FLINT_SWA_RING") ? std::atoi(std::getenv("FLINT_SWA_RING")) : kSwaRing;   // (a power of two)
            const size_t rows = l.swa && max_ctx > ring ? ring : max_ctx;
            if (l.swa && rows == size_t(ring)) l.kv_mask = ring - 1;
            CK(cudaMalloc(&l.K, rows * kKvDim * sizeof(kc_t)));
            CK(cudaMalloc(&l.V, rows * kKvDim * sizeof(vc_t)));
            kv_bytes += rows * kKvDim * (sizeof(kc_t) + sizeof(vc_t));
        }
        {   // the token table (1.2 GB fp32) stays in pinned host memory: the card reads one 8 KB row a token over the bus
            const strata::TensorInfo* t = g->find("token_embd.weight");
            const size_t bytes = strata::tensor_payload_bytes(*t);
            void* hp = nullptr;
            CK(cudaHostAlloc(&hp, bytes, cudaHostAllocMapped));
            std::memcpy(hp, g->tensor_data(*t), bytes);
            void* dp = nullptr;
            CK(cudaHostGetDevicePointer(&dp, hp, 0));
            embed = (const float*) dp;
        }
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
        CK(cudaMalloc(&pidx, kTopK * 4)); CK(cudaMalloc(&pw, kTopK * 4)); CK(cudaMalloc(&prl, kExperts * 4));
        CK(cudaMalloc(&pdone, 4)); CK(cudaMemset(pdone, 0, 4));
        if (std::getenv("FLINT_PRED")) { CK(cudaMalloc(&pcnt, 8)); CK(cudaMemset(pcnt, 0, 8)); }
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

    // ------------------------------------------------ checking drafts: T tokens (the current one and T - 1 drafted)
    float *vH = nullptr, *vX, *vQKV, *vATT, *vG, *vU, *vACT, *vLG, *vRL, *vW, *vPart, *vY, *vXS, *vXS2;
    uint16_t *vXH, *vXL, *vXH2, *vXL2;
    int *vToks, *vIdx, *vUe, *vUm, *vVu, *vUn, *vBest;
    int* vToksHost = nullptr;
    int* vBestHost = nullptr;
    unsigned* vDone;
    cudaGraphExec_t vgraph[kVMax + 1] = {};
    void v_alloc() {
        if (vH) return;
        auto f = [&](float*& p_, size_t n) { CK(cudaMalloc(&p_, n * 4)); };
        auto fi = [&](int*& p_, size_t n) { CK(cudaMalloc(&p_, n * 4)); };
        f(vH, kVMax * kHidden); f(vX, kVMax * kFfDense); f(vQKV, kVMax * (kQDim + 2 * kKvDim)); f(vATT, kVMax * kQDim);
        f(vG, kVMax * kFfDense); f(vU, kVMax * kFfDense); f(vACT, kVMax * kActRows); f(vLG, size_t(kVMax) * vocab);
        f(vRL, kVMax * kExperts); f(vW, kVMax * kTopK); f(vPart, size_t(kVMax) * kHeads * kAttnChunks * kPart);
        fi(vToks, kVMax); fi(vIdx, kVMax * kTopK); fi(vUe, kVUniq); fi(vUm, kVUniq); fi(vVu, kVMax * kTopK); fi(vUn, 1);
        fi(vBest, kVMax);
        f(vY, size_t(kVMax) * (kTopK + 1) * kHidden); f(vXS, kVMax * kTopK); f(vXS2, kVMax);
        CK(cudaMalloc(&vXH, size_t(kVMax) * kFfDense * 2)); CK(cudaMalloc(&vXL, size_t(kVMax) * kFfDense * 2));
        CK(cudaMalloc(&vXH2, size_t(kVMax) * kFfShared * 2)); CK(cudaMalloc(&vXL2, size_t(kVMax) * kFfShared * 2));
        CK(cudaMalloc(&vDone, 4));
        CK(cudaMemset(vDone, 0, 4));
        CK(cudaMallocHost(&vToksHost, kVMax * 4));
        CK(cudaMallocHost(&vBestHost, kVMax * 4));
    }
    template <class F> static void vdispatch(int T, F&& f) {
        switch (T) {
            case 1: f(std::integral_constant<int, 1>{}); break;
            case 2: f(std::integral_constant<int, 2>{}); break;
            case 3: f(std::integral_constant<int, 3>{}); break;
            case 4: f(std::integral_constant<int, 4>{}); break;
            case 5: f(std::integral_constant<int, 5>{}); break;
            case 6: f(std::integral_constant<int, 6>{}); break;
            case 7: f(std::integral_constant<int, 7>{}); break;
            default: f(std::integral_constant<int, 8>{}); break;
        }
    }
    void v_mv(std::initializer_list<std::pair<const uint8_t*, float*>> mats, std::initializer_list<int> rows,
              const float* X, int ldx, int n_in, int T, int ldy, bool accum) {
        MvV J{};
        int j = 0, total = 0;
        for (auto& m_ : mats) { J.w[j] = m_.first; J.y[j] = m_.second; ++j; }
        j = 0;
        for (int r : rows) { J.rows[j++] = r; total += r; }
        J.nmat = j; J.X = X; J.ldx = ldx; J.n_in = n_in; J.T = T; J.ldy = ldy; J.acc = accum;
        vdispatch(T, [&](auto tc) { k_v_mv<decltype(tc)::value><<<total / kMvRows, 256, 0, st>>>(J); });
    }
    // the matrix-unit version (FLINT_VSIMT=1: the first one, below)
    void vg(std::initializer_list<std::pair<const uint8_t*, float*>> mats, std::initializer_list<int> rows, int K, int T,
            int ldy, int add, VgIn in) {
        VgJob J{};
        int j = 0, total = 0;
        for (auto& m_ : mats) { J.w[j] = m_.first; J.y[j] = m_.second; ++j; }
        j = 0;
        for (int r : rows) { J.rows[j++] = r; total += r; }
        J.nmat = j; J.K = K; J.T = T; J.ldy = ldy; J.add = add; J.in = in;
        k_vg<<<total / 16, 128, 0, st>>>(J);
    }
    void v_record_mu(int T) {
        const int NQKV = kQDim + 2 * kKvDim;
        k_v_embed<<<T, 256, 0, st>>>(embed, vToks, vH);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            k_split16<<<T, 256, 0, st>>>(vH, nullptr, l.attn_norm, kHidden, vXH, vXL, vXS);
            // (k_split16 writes rows of K = 2048 at stride K: the VgIn stride is the split's K)
            vg({{l.wq, vQKV}, {l.wk, vQKV + kQDim}, {l.wv, vQKV + kQDim + kKvDim}}, {kQDim, kKvDim, kKvDim}, kHidden, T, NQKV, 0, VgIn{vXH, vXL, vXS, kHidden});
            k_qkv_post<<<dim3(kHeads + kKv, 1, T), kHd, 0, st>>>(vQKV, vQKV + kQDim, vQKV + kQDim + kKvDim, l.q_norm, l.k_norm, l.K, l.V, state, l.swa, l.kv_mask, NQKV);
            if (l.swa) k_attn_part<<<dim3(kHeads, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV);
            else k_attn_dec_w<<<dim3(kKv, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV);
            k_attn_merge<<<dim3(kHeads, 1, T), kHd, 0, st>>>(vPart, l.sinks, vATT);
            k_split16<<<T, 256, 0, st>>>(vATT, nullptr, nullptr, kQDim, vXH, vXL, vXS);
            vg({{l.wo, vH}}, {kHidden}, kQDim, T, kHidden, 1, VgIn{vXH, vXL, vXS, kQDim});
            if (i < kDense) {
                k_split16<<<T, 256, 0, st>>>(vH, nullptr, l.ffn_norm, kHidden, vXH, vXL, vXS);
                vg({{l.gate, vG}, {l.up, vU}}, {kFfDense, kFfDense}, kHidden, T, kFfDense, 0, VgIn{vXH, vXL, vXS, kHidden});
                k_split16<<<T, 256, 0, st>>>(vG, vU, nullptr, kFfDense, vXH, vXL, vXS);
                vg({{l.down, vH}}, {kHidden}, kFfDense, T, kHidden, 1, VgIn{vXH, vXL, vXS, kFfDense});
                continue;
            }
            k_v_norm<<<T, 256, 0, st>>>(vH, l.ffn_norm, vX);      // fp32 normed rows for the router
            vdispatch(T, [&](auto tc) { k_vr<decltype(tc)::value><<<kExperts / 4, 128, 0, st>>>(vX, l.router, vRL); });
            k_vr_pick<<<1, 128, 0, st>>>(vRL, T, vIdx, vW, vUe, vUm, vVu, vUn);
            k_split16<<<T, 256, 0, st>>>(vH, nullptr, l.ffn_norm, kHidden, vXH, vXL, vXS);
            k_vg_gu<<<dim3(kFfShared / 16, kVUniq + 1), 128, 0, st>>>(VgIn{vXH, vXL, vXS, kHidden}, l.gate_e, l.up_e, l.gate_s, l.up_s, vUe, vUm, vVu, vUn, T, vACT);
            // the act rows split: each expert slot (512) a row, and each token's shared part (1024) a row
            k_split16<<<T * kTopK, 256, 0, st>>>(vACT, nullptr, nullptr, kFfExp, vXH, vXL, vXS, kTopK, kActRows);
            k_split16<<<T, 256, 0, st>>>(vACT + kTopK * kFfExp, nullptr, nullptr, kFfShared, vXH2, vXL2, vXS2, 1, kActRows);
            k_vg_down<<<dim3(kHidden / 16, kVUniq + 1), 128, 0, st>>>(VgIn{vXH, vXL, vXS, kFfExp}, VgIn{vXH2, vXL2, vXS2, kFfShared}, l.down_e, l.down_s, vUe, vUm, vVu, vUn, T, vY);
            k_vg_combine<<<T, 256, 0, st>>>(vH, vY, vW);
        }
        k_split16<<<T, 256, 0, st>>>(vH, nullptr, out_norm, kHidden, vXH, vXL, vXS);
        vg({{head, vLG}}, {vocab}, kHidden, T, vocab, 0, VgIn{vXH, vXL, vXS, kHidden});
        k_argmax_rows<<<T, 1024, 0, st>>>(vLG, vocab, vBest);
    }
    void v_record(int T) {
        static const bool mu = std::getenv("FLINT_VMU") != nullptr;
        if (mu) { v_record_mu(T); return; }
        const int NQKV = kQDim + 2 * kKvDim;
        k_v_embed<<<T, 256, 0, st>>>(embed, vToks, vH);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            k_v_norm<<<T, 256, 0, st>>>(vH, l.attn_norm, vX);
            v_mv({{l.wq, vQKV}, {l.wk, vQKV + kQDim}, {l.wv, vQKV + kQDim + kKvDim}}, {kQDim, kKvDim, kKvDim}, vX, kHidden, kHidden, T, NQKV, false);
            k_qkv_post<<<dim3(kHeads + kKv, 1, T), kHd, 0, st>>>(vQKV, vQKV + kQDim, vQKV + kQDim + kKvDim, l.q_norm, l.k_norm, l.K, l.V, state, l.swa, l.kv_mask, NQKV);
            if (l.swa) k_attn_part<<<dim3(kHeads, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV);
            else k_attn_dec_w<<<dim3(kKv, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV);
            k_attn_merge<<<dim3(kHeads, 1, T), kHd, 0, st>>>(vPart, l.sinks, vATT);
            v_mv({{l.wo, vH}}, {kHidden}, vATT, kQDim, kQDim, T, kHidden, true);
            k_v_norm<<<T, 256, 0, st>>>(vH, l.ffn_norm, vX);
            if (i < kDense) {
                v_mv({{l.gate, vG}, {l.up, vU}}, {kFfDense, kFfDense}, vX, kHidden, kHidden, T, kFfDense, false);
                k_v_swiglu<<<dim3(kFfDense / 256, T), 256, 0, st>>>(vG, vU, vG, kFfDense, kFfDense);
                v_mv({{l.down, vH}}, {kHidden}, vG, kFfDense, kFfDense, T, kHidden, true);
                continue;
            }
            vdispatch(T, [&](auto tc) { k_vr<decltype(tc)::value><<<kExperts / 4, 128, 0, st>>>(vX, l.router, vRL); });
            k_vr_pick<<<1, 128, 0, st>>>(vRL, T, vIdx, vW, vUe, vUm, vVu, vUn);
            vdispatch(T, [&](auto tc) {
                constexpr int TT = decltype(tc)::value;
                k_v_moe_gu<TT><<<dim3(kFfShared / kMvRows, kVUniq + 1), 256, 0, st>>>(vX, l.gate_e, l.up_e, l.gate_s, l.up_s, vUe, vUm, vVu, vUn, vACT);
                k_v_moe_down<TT><<<kHidden / kMvRows, 256, 0, st>>>(vH, vACT, l.down_e, l.down_s, vUe, vUm, vVu, vW, vUn);
            });
        }
        k_v_norm<<<T, 256, 0, st>>>(vH, out_norm, vX);
        v_mv({{head, vLG}}, {vocab}, vX, kHidden, kHidden, T, vocab, false);
        k_argmax_rows<<<T, 1024, 0, st>>>(vLG, vocab, vBest);
    }
    // the T tokens through the model at pos .. pos + T - 1 (their K/V written); afterwards each row's most likely next
    // token in vBestHost and the logits in vLG. pos is not moved: v_commit does that.
    void verify(const int* toks, int T) {
        v_alloc();
        std::memcpy(vToksHost, toks, T * 4);
        CK(cudaMemcpyAsync(vToks, vToksHost, T * 4, cudaMemcpyHostToDevice, st));
        static const bool no_graph = std::getenv("FLINT_NO_GRAPH") != nullptr;
        if (no_graph) v_record(T);
        else {
            if (!vgraph[T]) {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                v_record(T);
                CK(cudaStreamEndCapture(st, &g));
                CK(cudaGraphInstantiate(&vgraph[T], g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
            }
            CK(cudaGraphLaunch(vgraph[T], st));
        }
        CK(cudaMemcpyAsync(vBestHost, vBest, T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    // keep n of the checked tokens: the next position is pos + n, the next token goes into state[1]
    void v_commit(int n, int next) {
        pos += n;
        const int v[2] = {pos, next};
        CK(cudaMemcpyAsync(state, v, 8, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
    }

    // the position the next token goes to (a request reusing a shorter prefix than the cache holds)
    void set_pos(int p) {
        pos = p;
        const int v[1] = {p};
        CK(cudaMemcpyAsync(state, v, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
    }
    int ring_rows() const {
        for (const Layer& l : L) if (l.swa && l.kv_mask != 0x7fffffff) return l.kv_mask + 1;
        return max_ctx;
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
            k_pf_qkpost<<<dim3(kHeads + kKv, T), kHd, 0, st>>>(pQKV, l.q_norm, l.k_norm, l.K, l.V, p0, l.swa, l.kv_mask);
            k_pf_attn_w<<<dim3(kKv, (T + 15) / 16), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa, l.kv_mask);
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
        static const int nb = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : 32;
        k_touch<<<nb, 256, 0, side>>>(T, sink);
    }
    void record() {
        static const int pf = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : 32;
        k_embed<<<kHidden / 256, 256, 0, st>>>(embed, state, h);
        for (int i = 0; i < n_layers(); ++i) {
            Layer& l = L[i];
            static const int pe = std::getenv("FLINT_PREDPF") ? std::atoi(std::getenv("FLINT_PREDPF")) : 64;
            if ((pf || pe) && i + 1 < kLayers) {   // from the start of layer i, on the side stream:
                CK(cudaEventRecord(ev[i], st));
                CK(cudaStreamWaitEvent(side, ev[i], 0));
                if (pe && i >= kDense) {   // this layer's likely experts, picked from its input
                    k_router<<<kExperts / 4, 128, 0, side>>>(h, l.ffn_norm, l.router, prl, pidx, pw, pdone);
                    k_touch_pred<<<pe, 256, 0, side>>>(l.gate_e, l.up_e, l.down_e, pidx, sink);
                }
                if (pf) prefetch(i + 1);   // layer i + 1's fixed weights
            }
            static const bool pred = std::getenv("FLINT_PRED") != nullptr;
            if (pred && i >= kDense) {   // (measuring) the router on the layer's input
                k_router<<<kExperts / 4, 128, 0, st>>>(h, l.ffn_norm, l.router, prl, pidx, pw, pdone);
            }
            mv_in({{l.wq, q}, {l.wk, k}, {l.wv, v}}, {kQDim, kKvDim, kKvDim}, h, kHidden, l.attn_norm);
            k_qkv_post<<<kHeads + kKv, kHd, 0, st>>>(q, k, v, l.q_norm, l.k_norm, l.K, l.V, state, l.swa, l.kv_mask);
            // the sliding layers (at most 513 keys) keep the per-head kernel; the full layers, whose keys grow with the
            // context, read each key once for the 8 heads sharing it
            static const bool attn_old = std::getenv("FLINT_ATTN_OLD") != nullptr;
            if (attn_old || l.swa) k_attn_part<<<dim3(kHeads, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa, l.kv_mask);
            else k_attn_dec_w<<<dim3(kKv, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa, l.kv_mask);
            k_attn_merge<<<kHeads, kHd, 0, st>>>(part, l.sinks, att);
            mv_in({{l.wo, h}}, {kHidden}, att, kQDim, nullptr, nullptr, true);
            if (i < kDense) {
                mv_in({{l.gate, f1}, {l.up, f2}}, {kFfDense, kFfDense}, h, kHidden, l.ffn_norm);
                mv_in({{l.down, h}}, {kHidden}, f1, kFfDense, nullptr, f2, true);
                if (std::getenv("FLINT_DUMP_H")) dump_rows(h, 1);
                continue;
            }
            k_router<<<kExperts / 4, 128, 0, st>>>(h, l.ffn_norm, l.router, rl, idx, wts, done);
            if (pred) k_pred_overlap<<<1, 32, 0, st>>>(pidx, idx, pcnt);
            k_moe_gu<<<kActRows / kMvRows, 256, kHidden * 4, st>>>(h, l.ffn_norm, l.gate_e, l.up_e, l.gate_s, l.up_s, idx, act);
            k_moe_down<<<kHidden / kMvRows, 256, 0, st>>>(h, act, l.down_e, l.down_s, idx, wts);
            if (std::getenv("FLINT_DUMP_H")) dump_rows(h, 1);
        }
        static const int pe2 = std::getenv("FLINT_PREDPF") ? std::atoi(std::getenv("FLINT_PREDPF")) : 64;
        if (pf || pe2) {   // join the side stream back (the capture must end on one stream)
            CK(cudaEventRecord(ev[kLayers], side));
            CK(cudaStreamWaitEvent(st, ev[kLayers], 0));
        }
        mv_in({{head, logits}}, {vocab}, h, kHidden, out_norm);
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done2);
    }
};


#include "flint_serve.cuh"

}  // namespace

int main(int argc, char** argv) {
    // two hardware queues unless set: with the runtime's default four, graph launches leave gaps between kernels
    setenv("GPU_MAX_HW_QUEUES", "2", 0);
    std::string path, ids_s, top_path, logits_path;
    int n_new = 0, ctx = 4096;
    bool serve_mode = false;
    std::vector<int> eos_ids = {150001, 150003, 150005, 150018};   // <|end_of_text|>, <|user|>, <|observation|>, <|developer|>
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
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos-ids") {
            eos_ids.clear();
            std::stringstream es(next());
            for (std::string v; std::getline(es, v, ',');) eos_ids.push_back(std::stoi(v));
        }
        // flags Strata's server adds for its own engine (no images; the cache needs no reserve)
        else if (a == "--vision") {}
        else if (a == "--vram-reserve-mib") next();
        else { std::fprintf(stderr, "usage: strata-flint --model FILE.gguf --ids 1,2,3|@FILE [--new N] [--top FILE] [--max-ctx N]\n"); return 2; }
    }
    if (path.empty() || (ids_s.empty() && !serve_mode)) { std::fprintf(stderr, "strata-flint: --model and --ids (or --serve) are needed\n"); return 2; }
    std::vector<int> ids;
    std::stringstream ss(ids_s);
    for (std::string v; std::getline(ss, v, ',');) ids.push_back(std::stoi(v));

    const auto t0 = std::chrono::steady_clock::now();
    Model m(path, ctx);
    if (std::getenv("FLINT_GEMM_TEST")) { m.gemm_test(); return 0; }
    if (std::getenv("FLINT_VTEST")) {   // (checking) the draft-checking pass against plain writing
        m.prefill(ids, 8192);
        const int p0 = m.pos;
        std::vector<int> seq;
        for (int i = 0; i < 40; ++i) { seq.push_back(m.next_token()); m.step(-1); }
        int same = 0, total = 0;
        for (int start = 0; start + 9 <= 40; start += 8) {
            m.set_pos(p0 + start);
            m.verify(seq.data() + start, 8);
            for (int i = 0; i < 8; ++i) { same += m.vBestHost[i] == seq[start + i + 1]; ++total; }
        }
        std::printf("checking pass agrees with plain writing on %d of %d next tokens\n", same, total);
        {   // how many distinct experts T tokens use in a layer (the last layer's count after a pass)
            for (int T : {1, 2, 4, 8}) {
                double sum = 0;
                int n = 0;
                for (int start = 0; start + T <= 40; start += T) {
                    m.set_pos(p0 + start);
                    m.verify(seq.data() + start, T);
                    int un = 0;
                    CK(cudaMemcpy(&un, m.vUn, 4, cudaMemcpyDeviceToHost));
                    sum += un; ++n;
                }
                std::printf("%d tokens: %.1f distinct experts in the last layer (of %d chosen)\n", T, sum / n, 6 * T);
            }
        }
        for (int T = 1; T <= kVMax; ++T) {
            m.set_pos(p0);
            for (int r = 0; r < 3; ++r) m.verify(seq.data(), T);
            const auto a0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 20; ++r) m.verify(seq.data(), T);
            std::printf("checking %d tokens: %.2f ms\n", T, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count() / 20);
        }
        return 0;
    }
    if (std::getenv("FLINT_TTEST")) {   // (measuring) small batches on the prompt path after a prompt
        m.prefill(ids, 8192);
        std::vector<int> few(16, 13);
        for (int T : {1, 2, 3, 4, 6, 9, 16}) {
            for (int r = 0; r < 3; ++r) m.prefill_chunk(few.data(), T);
            const auto a0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 10; ++r) m.prefill_chunk(few.data(), T);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count() / 10;
            std::printf("prompt path, %2d tokens: %.2f ms\n", T, ms);
        }
        return 0;
    }
    if (serve_mode) {
        std::fprintf(stderr, "strata-flint: K/V for %d positions: %.2f GB\n", ctx, m.kv_bytes / 1e9);
        const int ch = std::min(8192, std::getenv("FLINT_CHUNK") ? std::atoi(std::getenv("FLINT_CHUNK")) : 8192);
        m.pf_alloc(ch);                                   // the prompt path's buffers before READY
        return serve(m, eos_ids, ch);
    }
    std::fprintf(stderr, "strata-flint: K/V for %d positions: %.2f GB\n", ctx, m.kv_bytes / 1e9);
    std::fprintf(stderr, "strata-flint: %.1f GB on the card in %.1f s\n", m.loaded / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    FILE* top = top_path.empty() ? nullptr : std::fopen(top_path.c_str(), "w");
    std::vector<float> lg(m.vocab);
    const auto t1 = std::chrono::steady_clock::now();
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    static const int chunk = std::min(8192, std::getenv("FLINT_CHUNK") ? std::atoi(std::getenv("FLINT_CHUNK")) : 8192);
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
    if (m.pcnt) {
        unsigned c[2];
        CK(cudaMemcpy(c, m.pcnt, 8, cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "strata-flint: experts chosen from the layer input: %.1f%% of the real ones\n", 100.0 * c[0] / std::max(1u, c[1]));
    }
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    if (top) std::fclose(top);
    std::fprintf(stderr, "strata-flint: read %zu in %.3f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n",
                 ids.size(), read_s, ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    return 0;
}
