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
// value = d * q; s = the sum of the 32 original values. fp32 scale and sum: in the experts the sum of 32 values passes
// fp16's 65504 (llama.cpp guards its fp16 inputs there with an extra rescale)
struct block_q8_1 { float d, s; int8_t qs[32]; };
static_assert(sizeof(block_q4_1) == 20 && sizeof(block_q8_1) == 40, "block layouts");

// ---------------------------------------------------------------------------------------------------- kernels
__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ inline float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
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

// y = w * x / rms(x) over n values (one block of 256)
__global__ void k_rmsnorm(const float* x, const float* w, float* y, int n) {
    __shared__ float sh[8];
    float s = 0;
    for (int i = threadIdx.x; i < n; i += 256) s += x[i] * x[i];
    s = block_sum<256>(s, sh);
    const float r = rsqrtf(s / n + kEps);
    for (int i = threadIdx.x; i < n; i += 256) y[i] = x[i] * r * w[i];
}

// in place, per head of 128 (block per head, 128 threads): RMSNorm with w, then NEOX RoPE at pos when rope
__global__ void k_head_norm_rope(float* v, const float* w, const int* state, bool rope) {
    const int pos = state[0];
    __shared__ float sh[4];
    float* h = v + blockIdx.x * kHd;
    const int i = threadIdx.x;
    const float x = h[i];
    const float s = block_sum<128>(x * x, sh);
    const float y = x * rsqrtf(s / kHd + kEps) * w[i];
    if (!rope) { h[i] = y; return; }
    __shared__ float t[kHd];
    t[i] = y;
    __syncthreads();
    const int j = i & 63;
    const float ang = float(pos) * powf(kRopeBase, -2.f * j / kHd);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    const float x0 = t[j], x1 = t[j + 64];
    h[i] = i < 64 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
}

// q8_1 blocks of x (one warp per 32 values)
__global__ void k_quant_q8_1(const float* x, block_q8_1* y, int nblocks) {
    const int b = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (b >= nblocks) return;
    const float v = x[b * 32 + l];
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = amax / 127.f;
    const int q = amax == 0.f ? 0 : int(roundf(v / d));
    y[b].qs[l] = int8_t(q);
    if (l == 0) { y[b].d = d; y[b].s = sum; }
}

// y[s][r] = W_{e(s)}[r] . x_s for s < gridDim.y: W_e at w + e * mat_bytes (e = idx[s], or 0 without idx), rows of
// n_in Q4_1 values; x_s = x + s * x_stride q8_1 blocks. One warp a row, 8 rows a block.
__global__ void k_q4_1_mv(const uint8_t* w, size_t mat_bytes, const int* idx, const block_q8_1* x, int x_stride,
                          float* y, int n_in, int n_out) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31, s = blockIdx.y;
    if (r >= n_out) return;
    const int nb = n_in / 32;
    const block_q4_1* W = (const block_q4_1*) (w + (idx ? size_t(idx[s]) : 0) * mat_bytes) + size_t(r) * nb;
    const block_q8_1* X = x + size_t(s) * x_stride;
    float acc = 0;
    for (int b = l; b < nb; b += 32) {
        const block_q4_1& a = W[b];
        const block_q8_1& q = X[b];
        int si = 0;
        for (int k = 0; k < 16; ++k) {
            const int v = a.qs[k];
            si += (v & 15) * q.qs[k] + (v >> 4) * q.qs[k + 16];
        }
        acc += __half2float(a.d) * q.d * si + __half2float(a.m) * q.s;
    }
    acc = warp_sum(acc);
    if (l == 0) y[size_t(s) * n_out + r] = acc;
}

// the same with fp32 inputs (x_s = x + s * x_stride floats): the precise path, FLINT_EXACT=1
// (experiment) xround: 1 = inputs rounded to bf16, 2 = to fp16 (after scaling by 2^-sc, sc = xsc)
__device__ inline float xr(float v, int mode, float sc) {
    if (mode == 1) return __uint_as_float((__float_as_uint(v) + 0x7fff + ((__float_as_uint(v) >> 16) & 1)) & 0xffff0000u);
    if (mode == 2) return __half2float(__float2half(v * sc)) / sc;
    return v;
}
__global__ void k_q4_1_mv_f(const uint8_t* w, size_t mat_bytes, const int* idx, const float* x, int x_stride,
                            float* y, int n_in, int n_out, int xround, float sc) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31, s = blockIdx.y;
    if (r >= n_out) return;
    const int nb = n_in / 32;
    const block_q4_1* W = (const block_q4_1*) (w + (idx ? size_t(idx[s]) : 0) * mat_bytes) + size_t(r) * nb;
    const float* X = x + size_t(s) * x_stride;
    float acc = 0;
    for (int b = l; b < nb; b += 32) {
        const block_q4_1& a = W[b];
        const float d = __half2float(a.d), m = __half2float(a.m);
        for (int k = 0; k < 16; ++k) {
            const int v = a.qs[k];
            acc += (d * (v & 15) + m) * xr(X[b * 32 + k], xround, sc) + (d * (v >> 4) + m) * xr(X[b * 32 + k + 16], xround, sc);
        }
    }
    acc = warp_sum(acc);
    if (l == 0) y[size_t(s) * n_out + r] = acc;
}

// Up to three Q4_1 matrices that read the same input in one launch (q/k/v, gate/up), for `slots` inputs (gridDim.y):
// matrix j's rows go to y[j] + s * y_stride, its weights are w[j] (+ idx[s] * mat_bytes for experts), input s is
// x + s * x_stride floats. Each block takes 32 rows of one matrix: the input is staged in LDS, each warp does 4 rows,
// and each lane reads one 4-byte word of a Q4_1 block (8 values: 4 low nibbles and the 4 high ones 16 further on).
struct MvJob {
    const uint8_t* w[3];
    float* y[3];
    int rows[3];     // multiples of 32
    int nmat;
    size_t mat_bytes;
    const int* idx;
    const float* x;
    int x_stride, y_stride, n_in;
};
constexpr int kMvRows = 32, kMvRpw = 4;
__global__ void __launch_bounds__(256) k_q4_1_mv2(MvJob J) {
    extern __shared__ float xs[];
    const int s = blockIdx.y;
    const float* X = J.x + size_t(s) * J.x_stride;
    for (int i = threadIdx.x * 4; i < J.n_in; i += 256 * 4) *(float4*) (xs + i) = *(const float4*) (X + i);
    int rb = blockIdx.x * kMvRows, j = 0;
    while (j < J.nmat - 1 && rb >= J.rows[j]) rb -= J.rows[j++];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = rb + warp * kMvRpw;
    const int nb = J.n_in / 32, qd = lane & 3;
    const size_t rowb = size_t(nb) * 20;
    const uint8_t* W = J.w[j] + (J.idx ? size_t(J.idx[s]) * J.mat_bytes : 0) + size_t(r0) * rowb;
    float acc[kMvRpw] = {};
    for (int b = lane >> 2; b < nb; b += 8) {
        const float4 xa = *(const float4*) (xs + b * 32 + qd * 4);
        const float4 xb = *(const float4*) (xs + b * 32 + 16 + qd * 4);
        const float sx = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) {
            const uint8_t* B = W + r * rowb + size_t(b) * 20;
            const uint32_t dm = *(const uint32_t*) B;
            const uint32_t q = *(const uint32_t*) (B + 4 + qd * 4);
            const float d = __half2float(__ushort_as_half(uint16_t(dm))), m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
            const float t = float(q & 15) * xa.x + float((q >> 8) & 15) * xa.y + float((q >> 16) & 15) * xa.z + float((q >> 24) & 15) * xa.w
                          + float((q >> 4) & 15) * xb.x + float((q >> 12) & 15) * xb.y + float((q >> 20) & 15) * xb.z + float(q >> 28) * xb.w;
            acc[r] += d * t + m * sx;
        }
    }
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) J.y[j][size_t(s) * J.y_stride + r0 + r] = v;
    }
}

// ------------------------------------------------ the fused layer (v3)
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
__global__ void __launch_bounds__(128) k_router(const float* h, const float* nw, const float* W, float* logits,
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
    for (int i = lane * 4; i < kHidden; i += 128) {
        const float4 w = *(const float4*) (W + size_t(r) * kHidden + i);
        const float4 nn = *(const float4*) (nw + i);
        acc += w.x * xs[i] * nn.x + w.y * xs[i + 1] * nn.y + w.z * xs[i + 2] * nn.z + w.w * xs[i + 3] * nn.w;
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

// y[r] = W[r] . x for an fp32 matrix (the router); one warp a row
__global__ void k_f32_mv(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (r >= n_out) return;
    float acc = 0;
    for (int i = l; i < n_in; i += 32) acc += W[size_t(r) * n_in + i] * x[i];
    acc = warp_sum(acc);
    if (l == 0) y[r] = acc;
}

// out[i] = silu(g[i]) * u[i]
__global__ void k_swiglu(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = silu(g[i]) * u[i];
}

// the 6 experts with the highest sigmoid scores and their weights (normalized to sum 1); one block of 256
__global__ void k_route(const float* logits, int* idx, float* wts) {
    __shared__ float sc[kExperts];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ float picked[kTopK];
    const int t = threadIdx.x;
    sc[t] = 1.f / (1.f + expf(-logits[t]));
    __syncthreads();
    for (int k = 0; k < kTopK; ++k) {
        float v = sc[t];
        int i = t;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if ((t & 31) == 0) { bv[t >> 5] = v; bi[t >> 5] = i; }
        __syncthreads();
        if (t == 0) {
            float b = bv[0];
            int j = bi[0];
            for (int w = 1; w < 8; ++w) if (bv[w] > b || (bv[w] == b && bi[w] < j)) { b = bv[w]; j = bi[w]; }
            idx[k] = j;
            picked[k] = b;
            sc[j] = -1.f;
        }
        __syncthreads();
    }
    if (t == 0) {
        float s = 0;
        for (int k = 0; k < kTopK; ++k) s += picked[k];
        for (int k = 0; k < kTopK; ++k) wts[k] = picked[k] / s;
    }
}

// h += o (the attention output); one thread a value
__global__ void k_add(float* h, const float* o, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) h[i] += o[i];
}

// h += shared + sum_k w[k] * y[k] (the MoE layer's output)
__global__ void k_moe_sum(float* h, const float* shared, const float* y, const float* w) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= kHidden) return;
    float s = shared[i];
    for (int k = 0; k < kTopK; ++k) s += w[k] * y[k * kHidden + i];
    h[i] += s;
}

// k, v of this position into the cache (fp32, [pos][kv head][128])
__global__ void k_kv_put(const float* k, const float* v, float* K, float* V, const int* state) {
    const int pos = state[0];
    const int i = threadIdx.x + blockIdx.x * 256;
    if (i < kKvDim) { K[size_t(pos) * kKvDim + i] = k[i]; V[size_t(pos) * kKvDim + i] = v[i]; }
}

// one query position against keys lo..pos, one block (256) a query head; the head's sink logit joins the softmax's
// denominator only. sc: scratch of max_ctx floats a head.
__global__ void k_attn(const float* q, const float* K, const float* V, const float* sinks, float* out, float* sc,
                       const int* state, bool swa, int max_ctx) {
    const int pos = state[0], lo = swa ? max(0, pos - (kWindow - 1)) : 0;
    __shared__ float sh[8];
    __shared__ float qs[kHd];
    const int h = blockIdx.x, kh = h / (kHeads / kKv), t = threadIdx.x;
    if (t < kHd) qs[t] = q[h * kHd + t];
    __syncthreads();
    float* s = sc + size_t(h) * max_ctx;
    const float scale = rsqrtf(float(kHd));
    float mx = sinks[h];
    for (int p = lo + t; p <= pos; p += 256) {
        const float* kr = K + size_t(p) * kKvDim + kh * kHd;
        float d = 0;
        for (int i = 0; i < kHd; ++i) d += qs[i] * kr[i];
        d *= scale;
        s[p] = d;
        mx = fmaxf(mx, d);
    }
    // block max
    mx = warp_max(mx);
    if ((t & 31) == 0) sh[t >> 5] = mx;
    __syncthreads();
    mx = sh[0];
    for (int i = 1; i < 8; ++i) mx = fmaxf(mx, sh[i]);
    __syncthreads();
    float sum = 0;
    for (int p = lo + t; p <= pos; p += 256) { const float e = expf(s[p] - mx); s[p] = e; sum += e; }
    sum = block_sum<256>(sum, sh) + expf(sinks[h] - mx);
    __syncthreads();
    if (t < kHd) {
        float o = 0;
        for (int p = lo; p <= pos; ++p) o += s[p] * V[size_t(p) * kKvDim + kh * kHd + t];
        out[h * kHd + t] = o / sum;
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

__global__ void k_embed(const float* table, const int* state, float* h) {
    const int i = blockIdx.x * 256 + threadIdx.x, tok = state[1];
    if (i < kHidden) h[i] = table[size_t(tok) * kHidden + i];
}

// the most likely next token into state[1], and the position moves on (state[0]); one block of 1024
__global__ void k_argmax_next(const float* lg, int n, int* state) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int i = 0;
    for (int j = threadIdx.x; j < n; j += 1024) if (lg[j] > v) { v = lg[j]; i = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o, 32);
        const int oi = __shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w) if (bv[w] > v || (bv[w] == v && bi[w] < i)) { v = bv[w]; i = bi[w]; }
        state[1] = i;
        state[0] += 1;
    }
}

// ---------------------------------------------------------------------------------------------------- the model
struct Layer {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *sinks, *router = nullptr;
    const uint8_t *wq, *wk, *wv, *wo;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;              // dense (layers 0-1)
    const uint8_t *gate_e = nullptr, *up_e = nullptr, *down_e = nullptr;        // experts
    const uint8_t *gate_s = nullptr, *up_s = nullptr, *down_s = nullptr;        // shared expert
    bool swa = false;
    float *K = nullptr, *V = nullptr;
};

size_t q4_row(int n) { return size_t(n / 32) * sizeof(block_q4_1); }

struct Model {
    std::unique_ptr<strata::GgufFile> g;
    std::vector<Layer> L;
    const float *embed, *out_norm;
    const uint8_t* head;
    int vocab = 0, max_ctx;
    size_t loaded = 0;
    cudaStream_t st;
    // scratch
    float *act;
    unsigned* done;
    float *part, *h, *x, *q, *k, *v, *att, *o, *f1, *f2, *f3, *ye, *ysh, *logits, *sc, *rl, *wts;
    int* idx;
    block_q8_1 *xq, *aq, *eq;
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
                l.router = (const float*) up_tensor(b + "ffn_gate_inp.weight", kF32);
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
        f(h, kHidden); f(x, kHidden); f(q, kQDim); f(k, kKvDim); f(v, kKvDim); f(att, kQDim); f(o, kHidden);
        f(f1, kFfDense); f(f2, kFfDense); f(f3, kTopK * kFfExp); f(ye, kTopK * kHidden); f(ysh, kHidden);
        f(logits, vocab); f(sc, size_t(kHeads) * max_ctx); f(rl, kExperts); f(wts, kTopK);
        CK(cudaMalloc(&idx, kTopK * 4));
        CK(cudaMalloc(&part, size_t(kHeads) * kAttnChunks * kPart * 4));
        CK(cudaMalloc(&act, kActRows * 4));
        CK(cudaMalloc(&done, 4));
        CK(cudaMemset(done, 0, 4));
        CK(cudaMalloc(&state, 8));
        CK(cudaMemset(state, 0, 8));
        CK(cudaMallocHost(&host_state, 8));
        CK(cudaMalloc(&xq, (kFfDense / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&aq, (kQDim / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&eq, (kTopK * kFfExp / 32) * sizeof(block_q8_1)));
    }

    void quant(const float* a, block_q8_1* y, int n) {
        if (!q8mask) return;   // (only the q8_1 experiments read these)
        const int nb = n / 32;
        k_quant_q8_1<<<(nb + 7) / 8, 256, 0, st>>>(a, y, nb);
    }
    // which products read q8_1 inputs instead of fp32 (bits: 1 q/k/v, 2 o, 4 gate/up, 8 down, 16 head); default none
    const int q8mask = std::getenv("FLINT_Q8") ? std::atoi(std::getenv("FLINT_Q8")) : 0;
    int cls = 0;   // the class of the next product
    // y_s = W_{e(s)} . x_s for `slots` inputs: the fp32 inputs xf (stride n_in floats when strided) on the precise
    // path, their q8_1 blocks xq otherwise
    void mvx(const uint8_t* W, size_t mat_bytes, const int* ix, const float* xf, const block_q8_1* xq_, bool strided,
             float* y, int n_in, int n_out, int slots) {
        const dim3 gr((n_out + 7) / 8, slots);
        static const int xround = std::getenv("FLINT_XROUND") ? std::atoi(std::getenv("FLINT_XROUND")) : 0;
        static const float xsc = std::getenv("FLINT_XSC") ? std::ldexp(1.f, -std::atoi(std::getenv("FLINT_XSC"))) : 1.f;
        if (!(q8mask & cls)) k_q4_1_mv_f<<<gr, 256, 0, st>>>(W, mat_bytes, ix, xf, strided ? n_in : 0, y, n_in, n_out, cls == 16 ? 0 : xround, xsc);
        else k_q4_1_mv<<<gr, 256, 0, st>>>(W, mat_bytes, ix, xq_, strided ? n_in / 32 : 0, y, n_in, n_out);
    }
    void mv(const uint8_t* W, const float* xf, const block_q8_1* xq_, float* y, int n_in, int n_out) {
        mvx(W, 0, nullptr, xf, xq_, false, y, n_in, n_out, 1);
    }
    // the fast fp32-input products (FLINT_MV1=1: the simple kernel above instead)
    void mv2(std::initializer_list<std::pair<const uint8_t*, float*>> mats, std::initializer_list<int> rows, int n_in,
             const float* xf, int x_stride = 0, int slots = 1, const int* ix = nullptr, size_t mat_bytes = 0, int y_stride = 0) {
        static const bool old = std::getenv("FLINT_MV1") != nullptr;
        MvJob J{};
        int total = 0, j = 0;
        for (auto& m : mats) { J.w[j] = m.first; J.y[j] = m.second; ++j; }
        j = 0;
        for (int r : rows) { J.rows[j++] = r; total += r; }
        J.nmat = j; J.mat_bytes = mat_bytes; J.idx = ix; J.x = xf; J.x_stride = x_stride; J.y_stride = y_stride; J.n_in = n_in;
        if (old) {
            for (int k = 0; k < J.nmat; ++k)
                k_q4_1_mv_f<<<dim3((J.rows[k] + 7) / 8, slots), 256, 0, st>>>(J.w[k], mat_bytes, ix, xf, x_stride, J.y[k], n_in, J.rows[k], 0, 1.f);
            // (y_stride is n_out for every caller here)
            return;
        }
        k_q4_1_mv2<<<dim3(total / kMvRows, slots), 256, n_in * 4, st>>>(J);
    }

    // one token through the model (tok, or the last step's own choice when tok < 0); logits on the card, the most
    // likely next token in state[1]. Recorded as a graph on the first call and replayed after that.
    void step(int tok) {
        if (pos >= max_ctx) { std::fprintf(stderr, "strata-flint: context full (%d)\n", max_ctx); std::exit(1); }
        if (tok >= 0) {
            host_state[1] = tok;
            CK(cudaMemcpyAsync(state + 1, host_state + 1, 4, cudaMemcpyHostToDevice, st));
        }
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
    // the fused layer path: 8 launches a MoE layer
    void record_v3() {
        k_embed<<<kHidden / 256, 256, 0, st>>>(embed, state, h);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            mv_in({{l.wq, q}, {l.wk, k}, {l.wv, v}}, {kQDim, kKvDim, kKvDim}, h, kHidden, l.attn_norm);
            k_qkv_post<<<kHeads + kKv, kHd, 0, st>>>(q, k, v, l.q_norm, l.k_norm, l.K, l.V, state, l.swa);
            k_attn_part<<<dim3(kHeads, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa);
            k_attn_merge<<<kHeads, kHd, 0, st>>>(part, l.sinks, att);
            mv_in({{l.wo, h}}, {kHidden}, att, kQDim, nullptr, nullptr, true);
            if (i < kDense) {
                mv_in({{l.gate, f1}, {l.up, f2}}, {kFfDense, kFfDense}, h, kHidden, l.ffn_norm);
                mv_in({{l.down, h}}, {kHidden}, f1, kFfDense, nullptr, f2, true);
                continue;
            }
            k_router<<<kExperts / 4, 128, 0, st>>>(h, l.ffn_norm, l.router, rl, idx, wts, done);
            k_moe_gu<<<kActRows / kMvRows, 256, kHidden * 4, st>>>(h, l.ffn_norm, l.gate_e, l.up_e, l.gate_s, l.up_s, idx, act);
            k_moe_down<<<kHidden / kMvRows, 256, 0, st>>>(h, act, l.down_e, l.down_s, idx, wts);
        }
        mv_in({{head, logits}}, {vocab}, h, kHidden, out_norm);
        k_argmax_next<<<1, 1024, 0, st>>>(logits, vocab, state);
    }
    void record() {
        static const bool v2 = std::getenv("FLINT_V2") != nullptr;
        if (!v2 && !q8mask) { record_v3(); return; }
        k_embed<<<kHidden / 256, 256, 0, st>>>(embed, state, h);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            k_rmsnorm<<<1, 256, 0, st>>>(h, l.attn_norm, x, kHidden);
            quant(x, xq, kHidden);
            if (q8mask) {
                cls = 1; mv(l.wq, x, xq, q, kHidden, kQDim);
                mv(l.wk, x, xq, k, kHidden, kKvDim);
                mv(l.wv, x, xq, v, kHidden, kKvDim);
            } else mv2({{l.wq, q}, {l.wk, k}, {l.wv, v}}, {kQDim, kKvDim, kKvDim}, kHidden, x);
            k_head_norm_rope<<<kHeads, kHd, 0, st>>>(q, l.q_norm, state, l.swa);
            k_head_norm_rope<<<kKv, kHd, 0, st>>>(k, l.k_norm, state, l.swa);
            k_kv_put<<<(kKvDim + 255) / 256, 256, 0, st>>>(k, v, l.K, l.V, state);
            static const bool old_attn = std::getenv("FLINT_ATTN1") != nullptr;
            if (old_attn) k_attn<<<kHeads, 256, 0, st>>>(q, l.K, l.V, l.sinks, att, sc, state, l.swa, max_ctx);
            else {
                k_attn_part<<<dim3(kHeads, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa);
                k_attn_merge<<<kHeads, kHd, 0, st>>>(part, l.sinks, att);
            }
            quant(att, aq, kQDim);
            if (q8mask) { cls = 2; mv(l.wo, att, aq, o, kQDim, kHidden); }
            else mv2({{l.wo, o}}, {kHidden}, kQDim, att);
            k_add<<<kHidden / 256, 256, 0, st>>>(h, o, kHidden);
            k_rmsnorm<<<1, 256, 0, st>>>(h, l.ffn_norm, x, kHidden);
            quant(x, xq, kHidden);
            if (i < kDense) {
                if (q8mask) {
                    cls = 4; mv(l.gate, x, xq, f1, kHidden, kFfDense);
                    mv(l.up, x, xq, f2, kHidden, kFfDense);
                } else mv2({{l.gate, f1}, {l.up, f2}}, {kFfDense, kFfDense}, kHidden, x);
                k_swiglu<<<kFfDense / 256, 256, 0, st>>>(f1, f2, f1, kFfDense);
                quant(f1, xq, kFfDense);
                if (q8mask) { cls = 8; mv(l.down, f1, xq, o, kFfDense, kHidden); }
                else mv2({{l.down, o}}, {kHidden}, kFfDense, f1);
                k_add<<<kHidden / 256, 256, 0, st>>>(h, o, kHidden);
                continue;
            }
            k_f32_mv<<<kExperts / 8, 256, 0, st>>>(l.router, x, rl, kHidden, kExperts);
            k_route<<<1, 256, 0, st>>>(rl, idx, wts);
            const size_t gu_bytes = size_t(kFfExp) * q4_row(kHidden), d_bytes = size_t(kHidden) * q4_row(kFfExp);
            if (q8mask) {
                cls = 4; mvx(l.gate_e, gu_bytes, idx, x, xq, false, f1, kHidden, kFfExp, kTopK);
                mvx(l.up_e, gu_bytes, idx, x, xq, false, f2, kHidden, kFfExp, kTopK);
            } else mv2({{l.gate_e, f1}, {l.up_e, f2}}, {kFfExp, kFfExp}, kHidden, x, 0, kTopK, idx, gu_bytes, kFfExp);
            k_swiglu<<<kTopK * kFfExp / 256, 256, 0, st>>>(f1, f2, f3, kTopK * kFfExp);
            quant(f3, eq, kTopK * kFfExp);
            if (q8mask) { cls = 8; mvx(l.down_e, d_bytes, idx, f3, eq, true, ye, kFfExp, kHidden, kTopK); }
            else mv2({{l.down_e, ye}}, {kHidden}, kFfExp, f3, kFfExp, kTopK, idx, d_bytes, kHidden);
            // shared expert (its input xq is still the normed x)
            if (q8mask) {
                cls = 4; mv(l.gate_s, x, xq, f1, kHidden, kFfShared);
                mv(l.up_s, x, xq, f2, kHidden, kFfShared);
            } else mv2({{l.gate_s, f1}, {l.up_s, f2}}, {kFfShared, kFfShared}, kHidden, x);
            k_swiglu<<<kFfShared / 256, 256, 0, st>>>(f1, f2, f1, kFfShared);
            quant(f1, aq, kFfShared);
            if (q8mask) { cls = 8; mv(l.down_s, f1, aq, ysh, kFfShared, kHidden); }
            else mv2({{l.down_s, ysh}}, {kHidden}, kFfShared, f1);
            k_moe_sum<<<kHidden / 256, 256, 0, st>>>(h, ysh, ye, wts);
        }
        k_rmsnorm<<<1, 256, 0, st>>>(h, out_norm, x, kHidden);
        quant(x, xq, kHidden);
        if (q8mask) { cls = 16; mv(head, x, xq, logits, kHidden, vocab); }
        else mv2({{head, logits}}, {vocab}, kHidden, x);
        k_argmax_next<<<1, 1024, 0, st>>>(logits, vocab, state);
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
    std::fprintf(stderr, "strata-flint: %.1f GB on the card in %.1f s\n", m.loaded / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    FILE* top = top_path.empty() ? nullptr : std::fopen(top_path.c_str(), "w");
    std::vector<float> lg(m.vocab);
    const auto t1 = std::chrono::steady_clock::now();
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    for (int t : ids) {
        m.step(t);
        if (lf) { CK(cudaMemcpy(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost)); std::fwrite(lg.data(), 4, lg.size(), lf); }
    }
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
