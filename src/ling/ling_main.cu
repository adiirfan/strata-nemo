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
#include <atomic>
#include <sys/mman.h>
#include <poll.h>
#include <unistd.h>

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
// the same, rearranged on the card for aligned vector loads (k_repack): a row holds every block's quants first, then
// their scales (Q4_0: 16-byte nibbles | fp16 d; Q4_1: nibbles | d | m; Q5_0: nibbles | 4-byte high bits | d; Q8_0:
// 32 int8 | d), the same bytes a row; Q6_K's 210-byte blocks padded to 224 so each starts 16-aligned
constexpr int AQ4_0 = 102, AQ4_1 = 103, AQ5_0 = 106, AQ8_0 = 108, AQ6_K = 114;
__host__ __device__ inline int aligned_fmt(int fmt) { return fmt == F32 ? F32 : fmt + 100; }
__host__ __device__ inline size_t row_bytes(int fmt, int n) {
    switch (fmt) {
        case AQ4_0: return size_t(n / 32) * 18;
        case AQ4_1: return size_t(n / 32) * 20;
        case AQ5_0: return size_t(n / 32) * 22;
        case AQ8_0: return size_t(n / 32) * 34;
        case AQ6_K: return size_t(n / 256) * 224;
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
__device__ __forceinline__ void deq8(int fmt, const uint8_t* row, int g, float* v, int K) {
    if (fmt == F32) {
        const float* f = (const float*) row + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = f[i];
        return;
    }
    if (fmt >= 100) {   // the card's aligned forms (see AQ4_0)
        if (fmt == AQ6_K) {
            const int e0 = g * 8, sb = e0 >> 8, r = e0 & 255, n = r >> 7, rr = r & 127, qq = rr >> 5, l0 = rr & 31;
            const uint8_t* B = row + size_t(sb) * 224;
            const uint2 ql = *(const uint2*) (B + 64 * n + ((qq & 1) ? 32 : 0) + l0);
            const uint2 qh = *(const uint2*) (B + 128 + 32 * n + l0);
            const int8_t sc = ((const int8_t*) (B + 192))[8 * n + (l0 >> 4) + 2 * qq];
            const float d = h2f(B + 208) * float(sc);
            const int shl = (qq >> 1) ? 4 : 0;
            const uint8_t* qlb = (const uint8_t*) &ql;
            const uint8_t* qhb = (const uint8_t*) &qh;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float((((qlb[i] >> shl) & 15) | (((qhb[i] >> (2 * qq)) & 3) << 4)) - 32);
            return;
        }
        const int b = g >> 2, j0 = (g & 3) * 8, sh = j0 >= 16 ? 4 : 0, o = j0 & 15, nb = K / 32;
        if (fmt == AQ8_0) {
            const uint2 q = *(const uint2*) (row + size_t(b) * 32 + j0);
            const int8_t* qb = (const int8_t*) &q;
            const float d = h2f(row + size_t(nb) * 32 + b * 2);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float(qb[i]);
            return;
        }
        const uint2 q = *(const uint2*) (row + size_t(b) * 16 + o);
        const uint8_t* qb = (const uint8_t*) &q;
        const uint8_t* sc = row + size_t(nb) * 16;
        if (fmt == AQ4_0) {
            const float d = h2f(sc + b * 2);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float(((qb[i] >> sh) & 15) - 8);
        } else if (fmt == AQ4_1) {
            const float d = h2f(sc + b * 4), m = h2f(sc + b * 4 + 2);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float((qb[i] >> sh) & 15) + m;
        } else {   // AQ5_0: nibbles | the blocks' 4-byte high bits | d
            const uint32_t qh = *(const uint32_t*) (sc + b * 4);
            const float d = h2f(sc + size_t(nb) * 4 + b * 2);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = d * float(int(((qb[i] >> sh) & 15) | (((qh >> (j0 + i)) & 1u) << 4)) - 16);   // (int: qh is unsigned)
        }
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
        deq8(fmt, row, g, v, K);
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
// one 32-value block of an aligned row: its quants loaded (blk_load, ahead of use), then its dot with x's block (xv)
// with the scale applied once to the block's sum
struct Blk { uint4 a, b; };
template <int F>
__device__ __forceinline__ Blk blk_load(const uint8_t* row, int b) {
    Blk q;
    if constexpr (F == AQ8_0) { q.a = *(const uint4*) (row + size_t(b) * 32); q.b = *(const uint4*) (row + size_t(b) * 32 + 16); }
    else q.a = *(const uint4*) (row + size_t(b) * 16);
    return q;
}
template <int F>
__device__ __forceinline__ float blk_dot(const Blk& q, const uint8_t* row, int b, int nb, const float* xv) {
    float s = 0.f;
    if constexpr (F == AQ8_0) {
        const uint32_t w[8] = {q.a.x, q.a.y, q.a.z, q.a.w, q.b.x, q.b.y, q.b.z, q.b.w};
#pragma unroll
        for (int k = 0; k < 8; ++k)
#pragma unroll
            for (int t = 0; t < 4; ++t) s += float(int8_t(w[k] >> (8 * t))) * xv[4 * k + t];
        return h2f(row + size_t(nb) * 32 + b * 2) * s;
    } else {
        const uint32_t w[4] = {q.a.x, q.a.y, q.a.z, q.a.w};
        const uint8_t* sc = row + size_t(nb) * 16;
        if constexpr (F == AQ4_0) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t y = w[k] >> (8 * t);
                    s += float(int(y & 15) - 8) * xv[4 * k + t] + float(int((y >> 4) & 15) - 8) * xv[16 + 4 * k + t];
                }
            return h2f(sc + b * 2) * s;
        } else if constexpr (F == AQ4_1) {
            float xs_ = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t y = w[k] >> (8 * t);
                    s += float(y & 15) * xv[4 * k + t] + float((y >> 4) & 15) * xv[16 + 4 * k + t];
                    xs_ += xv[4 * k + t] + xv[16 + 4 * k + t];
                }
            return h2f(sc + b * 4) * s + h2f(sc + b * 4 + 2) * xs_;
        } else {   // AQ5_0
            const uint32_t qh = *(const uint32_t*) (sc + b * 4);
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t y = w[k] >> (8 * t);
                    const int j = 4 * k + t;
                    s += float(int((y & 15) | (((qh >> j) & 1u) << 4)) - 16) * xv[j] +
                         float(int(((y >> 4) & 15) | (((qh >> (j + 16)) & 1u) << 4)) - 16) * xv[16 + j];
                }
            return h2f(sc + size_t(nb) * 4 + b * 2) * s;
        }
    }
}
// one lane's blocks of R rows (a lane a block: 16 or 32 bytes of a row in one load, the next block's loads issued before
// this one's sums; x's block from xt without bank conflicts); q: each row's first block, loaded already
template <int F, int R>
__device__ __forceinline__ void blk_rows(const uint8_t* W, size_t rbytes, int nb, int ld, const float* xt, int lane, Blk* q, float* acc) {
    for (int b = lane; b < nb; b += 32) {
        Blk n[R];
        if (b + 32 < nb)
#pragma unroll
            for (int r = 0; r < R; ++r) n[r] = blk_load<F>(W + size_t(r) * rbytes, b + 32);
        float xv[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) xv[i] = xt[i * ld + b];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += blk_dot<F>(q[r], W + size_t(r) * rbytes, b, nb, xv);
#pragma unroll
        for (int r = 0; r < R; ++r) q[r] = n[r];
    }
}
// k_mvt: up to 6 matrices of any formats on one input (x, or rmsnorm(x) * nw), 4 rows a warp, 8 warps a block; each
// warp decodes its own matrix's format (the 32-value block formats a block a lane, the others 8 values a lane)
__global__ void __launch_bounds__(256) k_mvt(Mv J) {
    constexpr int R = 4;
    extern __shared__ float xs[];                 // x (normed), then x by block: xt[i * ld + b] = x[32 b + i]
    __shared__ float red[8];
    const int nb = J.K / 32, ld = nb | 1;         // (ld odd: the transposing writes and the reads in different banks)
    int rb = blockIdx.x * 8 * R, j = 0;
    while (j < J.nmat - 1 && rb >= J.rows[j]) rb -= J.rows[j++];
    const int lane = threadIdx.x & 31, r0 = rb + (threadIdx.x >> 5) * R, fmt = J.fmt[j];
    const size_t rbytes = row_bytes(fmt, J.K);
    const uint8_t* W = J.w[j] + size_t(r0) * rbytes;
    const bool blk = (fmt == AQ4_0 || fmt == AQ8_0 || fmt == AQ4_1 || fmt == AQ5_0) && (rbytes & 15) == 0;
    // the first block of each row on its way while x is staged (AQ4_0 / 4_1 / 5_0: the same 16 bytes of quants)
    Blk q[R];
    if (blk && lane < nb) {
        if (fmt == AQ8_0)
#pragma unroll
            for (int r = 0; r < R; ++r) q[r] = blk_load<AQ8_0>(W + size_t(r) * rbytes, lane);
        else
#pragma unroll
            for (int r = 0; r < R; ++r) q[r] = blk_load<AQ4_0>(W + size_t(r) * rbytes, lane);
    }
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
    float* xt = xs + J.K;
    for (int i = threadIdx.x; i < J.K; i += 256) xt[(i & 31) * ld + (i >> 5)] = xs[i];
    __syncthreads();
    float acc[R] = {};
    if (blk) {
        if (fmt == AQ4_0) blk_rows<AQ4_0, R>(W, rbytes, nb, ld, xt, lane, q, acc);
        else if (fmt == AQ8_0) blk_rows<AQ8_0, R>(W, rbytes, nb, ld, xt, lane, q, acc);
        else if (fmt == AQ4_1) blk_rows<AQ4_1, R>(W, rbytes, nb, ld, xt, lane, q, acc);
        else blk_rows<AQ5_0, R>(W, rbytes, nb, ld, xt, lane, q, acc);
    } else
    for (int g = lane; g < J.K / 8; g += 32) {
        float v[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) deq8(fmt, W + size_t(r) * rbytes, g, v[r], J.K);
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r] += v[r][i] * xs[g * 8 + i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) J.y[j][r0 + r] = J.acc ? J.y[j][r0 + r] + v : v;
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
// the writing step's attention with the cache read once for all 32 heads: block c takes positions c * per .. (per: the
// chunk), warp w heads w, w + 8, w + 16, w + 24 (lane: 18 of the 576 key values, 16 of the 512 latent ones), online
// softmax; each (chunk, head)'s max, sum and latent sum into part (kMlaChunks x 32 x 514), k_mla_merge adds them up
constexpr int kMlaChunks = 32, kMlaPart = 2 + kLat;
__global__ void __launch_bounds__(256) k_mla_attn_part(const float* qa, const float* cache, float* part, const int* state) {
    const int c = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = state[0] + 1;
    const int per = max(64, (n + kMlaChunks - 1) / kMlaChunks), a0 = c * per, b0 = min(n, a0 + per);
    const float scale = rsqrtf(float(kQkHd));
    float qv[4][18], acc[4][16], m[4], l[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float* qh = qa + (w + 8 * a) * kKvA;
#pragma unroll
        for (int i = 0; i < 18; ++i) qv[a][i] = qh[lane + 32 * i] * scale;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[a][i] = 0.f;
        m[a] = -INFINITY;
        l[a] = 0.f;
    }
    for (int s = a0; s < b0; ++s) {
        const float* row = cache + size_t(s) * kKvA;
        float kv[18];
#pragma unroll
        for (int i = 0; i < 18; ++i) kv[i] = row[lane + 32 * i];
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            float d = 0;
#pragma unroll
            for (int i = 0; i < 18; ++i) d += qv[a][i] * kv[i];
            d = warp_sum(d);
            const float mn = fmaxf(m[a], d), cf = expf(m[a] - mn), p = expf(d - mn);
            m[a] = mn;
            l[a] = l[a] * cf + p;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[a][i] = acc[a][i] * cf + p * kv[i];
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        float* P = part + (size_t(c) * kHeads + w + 8 * a) * kMlaPart;
        if (lane == 0) { P[0] = m[a]; P[1] = l[a]; }
#pragma unroll
        for (int i = 0; i < 16; ++i) P[2 + lane + 32 * i] = acc[a][i];
    }
}
// per head (block, 256 threads): the chunks' records into ol_h (512)
__global__ void __launch_bounds__(256) k_mla_merge(const float* part, float* ol) {
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < kMlaChunks; ++c) M = fmaxf(M, part[(size_t(c) * kHeads + h) * kMlaPart]);
    float L = 0;
    for (int c = 0; c < kMlaChunks; ++c) {
        const float mc = part[(size_t(c) * kHeads + h) * kMlaPart];
        if (mc != -INFINITY) L += part[(size_t(c) * kHeads + h) * kMlaPart + 1] * expf(mc - M);
    }
    for (int i = threadIdx.x; i < kLat; i += 256) {
        float o = 0;
        for (int c = 0; c < kMlaChunks; ++c) {
            const float* P = part + (size_t(c) * kHeads + h) * kMlaPart;
            if (P[0] != -INFINITY) o += P[2 + i] * expf(P[0] - M);
        }
        ol[h * kLat + i] = o / L;
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
// weights p normalized x 2.5. A block of 512 threads (thread = expert): each group's top 2 by one warp, the 8 picks by
// 8 rounds of a block-wide maximum (lowest index on ties, as a stable sort)
__device__ void route_block(const float* logits, const float* bias, int* idx, float* wts, int* count) {
    __shared__ float sel[kExperts], prob[kExperts], gsc[kGroups], wv[16];
    __shared__ int wi[16], top[kTopK], keep[kGroups];
    const int e = threadIdx.x, lane = e & 31, w = e >> 5;
    prob[e] = sigmoidf(logits[e]);
    sel[e] = prob[e] + bias[e];
    __syncthreads();
    if (w < kGroups) {   // warp w: group w's top-2 sum (lane: experts lane, lane + 32 of the group)
        const float a = sel[w * 64 + lane], b = sel[w * 64 + 32 + lane];
        float m1 = fmaxf(a, b), m2 = fminf(a, b);   // this lane's best two
        for (int o = 16; o > 0; o >>= 1) {
            const float o1 = fl_shfl_xor(m1, o, 32), o2 = fl_shfl_xor(m2, o, 32);
            const float n1 = fmaxf(m1, o1), n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
            m1 = n1; m2 = n2;
        }
        if (lane == 0) gsc[w] = m1 + m2;
    }
    __syncthreads();
    if (e == 0) {
        int used = 0;
        for (int k = 0; k < kGroupsUsed; ++k) {
            int bi = -1;
            for (int gi = 0; gi < kGroups; ++gi) if (!(used >> gi & 1) && (bi < 0 || gsc[gi] > gsc[bi])) bi = gi;
            used |= 1 << bi;
        }
        for (int gi = 0; gi < kGroups; ++gi) keep[gi] = used >> gi & 1;
    }
    __syncthreads();
    float v = keep[e >> 6] ? sel[e] : -INFINITY;
    for (int k = 0; k < kTopK; ++k) {
        float bv = v;
        int bi = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = fl_shfl_xor(bv, o, 32);
            const int oi = fl_shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { wv[w] = bv; wi[w] = bi; }
        __syncthreads();
        if (e == 0) {
            float b = wv[0];
            int ib = wi[0];
            for (int j = 1; j < 16; ++j) if (wv[j] > b || (wv[j] == b && wi[j] < ib)) { b = wv[j]; ib = wi[j]; }
            top[k] = ib;
        }
        __syncthreads();
        if (e == top[k]) v = -INFINITY;
    }
    if (e == 0) {
        float tot = 0;
        for (int k = 0; k < kTopK; ++k) tot += prob[top[k]];
        for (int k = 0; k < kTopK; ++k) {
            idx[k] = top[k];
            wts[k] = prob[top[k]] / tot * kExpScale;
            if (count) atomicAdd(&count[top[k]], 1);
        }
    }
}
__global__ void __launch_bounds__(512) k_route(const float* logits, const float* bias, int* idx, float* wts) {
    route_block(logits, bias, idx, wts, nullptr);
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
// a block's quants and dot for a format known only at run time (the four 32-value block formats)
__device__ __forceinline__ float blk_any(int fmt, const uint8_t* row, int b, int nb, const float* xv) {
    if (fmt == AQ4_0) return blk_dot<AQ4_0>(blk_load<AQ4_0>(row, b), row, b, nb, xv);
    if (fmt == AQ8_0) return blk_dot<AQ8_0>(blk_load<AQ8_0>(row, b), row, b, nb, xv);
    if (fmt == AQ4_1) return blk_dot<AQ4_1>(blk_load<AQ4_1>(row, b), row, b, nb, xv);
    return blk_dot<AQ5_0>(blk_load<AQ5_0>(row, b), row, b, nb, xv);
}
__device__ __forceinline__ bool is_blk(int fmt) { return fmt == AQ4_0 || fmt == AQ8_0 || fmt == AQ4_1 || fmt == AQ5_0; }
// R rows of one block format against xt (rearranged by block, stride ld), the rows' first blocks loaded first
template <int F, int R>
__device__ __forceinline__ void blk_rows_from(const uint8_t* W, size_t rbytes, int nb, int ld, const float* xt, int lane, float* acc) {
    Blk q[R];
    if (lane < nb)
#pragma unroll
        for (int r = 0; r < R; ++r) q[r] = blk_load<F>(W + size_t(r) * rbytes, lane);
    blk_rows<F, R>(W, rbytes, nb, ld, xt, lane, q, acc);
}
template <int R>
__device__ __forceinline__ void any_rows(int fmt, const uint8_t* W, size_t rbytes, int nb, int ld, const float* xt, int lane, float* acc) {
    if (fmt == AQ4_0) blk_rows_from<AQ4_0, R>(W, rbytes, nb, ld, xt, lane, acc);
    else if (fmt == AQ8_0) blk_rows_from<AQ8_0, R>(W, rbytes, nb, ld, xt, lane, acc);
    else if (fmt == AQ4_1) blk_rows_from<AQ4_1, R>(W, rbytes, nb, ld, xt, lane, acc);
    else blk_rows_from<AQ5_0, R>(W, rbytes, nb, ld, xt, lane, acc);
}
// act = silu(gate . x) * (up . x) with x = rmsnorm(h) * nw, for the chosen experts the card holds (an expert's 768 rows
// at k * 768) and the shared expert (at 6144); kGuR rows of gate and of up a warp, 8 warps a block
constexpr int kGuR = 2;
__global__ void __launch_bounds__(256) k_moe_gu(MoeGu J) {
    constexpr int nb = kHidden / 32, ld = nb | 1;
    __shared__ float xs[kHidden], xt[32 * ld], red[8];
    const int lane = threadIdx.x & 31, row = (blockIdx.x * 8 + (threadIdx.x >> 5)) * kGuR;
    const uint8_t *G = nullptr, *U = nullptr;
    int fmt;
    float cl;
    size_t rb;
    if (row < kTopK * kFfExp) {
        const int sl = J.slot[J.idx[row / kFfExp]];
        fmt = J.fmt_e;
        cl = J.clamp_e;
        rb = row_bytes(fmt, kHidden);
        if (sl >= 0) { G = J.ge + (size_t(sl) * kFfExp + row % kFfExp) * rb; U = J.ue + (size_t(sl) * kFfExp + row % kFfExp) * rb; }
    } else {
        fmt = J.fmt_s;
        cl = J.clamp_s;
        rb = row_bytes(fmt, kHidden);
        G = J.gs + size_t(row - kTopK * kFfExp) * rb;
        U = J.us + size_t(row - kTopK * kFfExp) * rb;
    }
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
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        const float v = xs[i] * r * J.nw[i];
        xs[i] = v;
        xt[(i & 31) * ld + (i >> 5)] = v;
    }
    __syncthreads();
    if (!G) {   // (an expert the card does not hold: the CPU computes it)
        if (lane < kGuR) J.act[row + lane] = 0.f;
        return;
    }
    float g[kGuR] = {}, u[kGuR] = {};
    if (is_blk(fmt) && (rb & 15) == 0) {
        any_rows<kGuR>(fmt, G, rb, nb, ld, xt, lane, g);
        any_rows<kGuR>(fmt, U, rb, nb, ld, xt, lane, u);
    } else
        for (int k = 0; k < kGuR; ++k) {
            g[k] = row_dot(fmt, G + size_t(k) * rb, xs, kHidden, lane);
            u[k] = row_dot(fmt, U + size_t(k) * rb, xs, kHidden, lane);
        }
#pragma unroll
    for (int k = 0; k < kGuR; ++k) {
        float gv = warp_sum(g[k]), uv = warp_sum(u[k]);
        if (cl > 0.f) { gv = fminf(gv, cl); uv = fmaxf(-cl, fminf(uv, cl)); }
        if (lane == 0) J.act[row + k] = siluf(gv) * uv;
    }
}
struct MoeDown {
    float* h;
    const float* act;
    const uint8_t *de, *ds;
    int fmt_e, fmt_s;
    const int *idx, *slot;
    const float* wts;
};
// h[r] += sum_k w_k down_k(act_k)[r] (the card's experts) + down_shared(act_s)[r]; kDnR rows a warp. A row's 8 x 24
// expert blocks and the shared expert's 24 are spread over the lanes (a block a lane), act rearranged by block
constexpr int kDnR = 2;
__global__ void __launch_bounds__(256) k_moe_down(MoeDown J) {
    constexpr int nbe = kFfExp / 32, nbt = kActRows / 32, ld = nbt | 1;   // 24 a row, 216 in all
    __shared__ float xt[32 * ld];
    __shared__ const uint8_t* base[kTopK + 1];
    __shared__ float wk[kTopK + 1];
    for (int i = threadIdx.x; i < kActRows; i += 256) xt[(i & 31) * ld + (i >> 5)] = J.act[i];
    if (threadIdx.x <= kTopK) {
        const int k = threadIdx.x;
        if (k < kTopK) {
            const int sl = J.slot[J.idx[k]];
            base[k] = sl < 0 ? nullptr : J.de + size_t(sl) * kHidden * row_bytes(J.fmt_e, kFfExp);
            wk[k] = J.wts[k];
        } else { base[k] = J.ds; wk[k] = 1.f; }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * kDnR;
    const size_t rbe = row_bytes(J.fmt_e, kFfExp), rbs = row_bytes(J.fmt_s, kFfShared);
    float tot[kDnR] = {};
    if (is_blk(J.fmt_e) && is_blk(J.fmt_s) && (rbe & 15) == 0 && (rbs & 15) == 0) {
        for (int p = lane; p < nbt; p += 32) {
            const int k = p / nbe, b = p - k * nbe;
            const uint8_t* B = base[k];
            if (!B) continue;
            const int fmt = k < kTopK ? J.fmt_e : J.fmt_s;
            const size_t rbk = k < kTopK ? rbe : rbs;
            float xv[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) xv[i] = xt[i * ld + p];
#pragma unroll
            for (int r = 0; r < kDnR; ++r) tot[r] += wk[k] * blk_any(fmt, B + size_t(r0 + r) * rbk, b, nbe, xv);
        }
    } else {   // (other formats: a row at a time, act read back in order)
        __shared__ float xs[kActRows];
        for (int i = threadIdx.x; i < kActRows; i += 256) xs[i] = J.act[i];
        __syncthreads();
        for (int k = 0; k <= kTopK; ++k) {
            if (!base[k]) continue;
            for (int r = 0; r < kDnR; ++r)
                tot[r] += wk[k] * row_dot(k < kTopK ? J.fmt_e : J.fmt_s, base[k] + size_t(r0 + r) * (k < kTopK ? rbe : rbs),
                                          xs + k * kFfExp, k < kTopK ? kFfExp : kFfShared, lane);
        }
    }
#pragma unroll
    for (int r = 0; r < kDnR; ++r) {
        const float v = warp_sum(tot[r]);
        if (lane == 0) J.h[r0 + r] += v;
    }
}

__global__ void k_embed(const uint8_t* table, int fmt, const int* state, float* h) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;   // 8-value group
    if (g >= kHidden / 8) return;
    float v[8];
    deq8(fmt, table + size_t(state[1]) * row_bytes(fmt, kHidden), g, v, kHidden);
    for (int i = 0; i < 8; ++i) h[g * 8 + i] = v[i];
}
// the GGUF block layout of `rows` rows of K values into the aligned form (one thread a block; Q6_K: a 210 -> 224 copy)
__global__ void k_repack(const uint8_t* src, uint8_t* dst, size_t rows, int K, int fmt) {
    const int nb = fmt == Q6_K ? K / 256 : K / 32;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * nb) return;
    const size_t r = i / nb;
    const int b = int(i % nb);
    const uint8_t* s = src + r * row_bytes(fmt, K);
    uint8_t* d = dst + r * row_bytes(aligned_fmt(fmt), K);
    switch (fmt) {
        case Q4_0: for (int j = 0; j < 16; ++j) d[b * 16 + j] = s[b * 18 + 2 + j];
                   d[nb * 16 + b * 2] = s[b * 18]; d[nb * 16 + b * 2 + 1] = s[b * 18 + 1]; break;
        case Q4_1: for (int j = 0; j < 16; ++j) d[b * 16 + j] = s[b * 20 + 4 + j];
                   for (int j = 0; j < 4; ++j) d[nb * 16 + b * 4 + j] = s[b * 20 + j]; break;
        case Q5_0: for (int j = 0; j < 16; ++j) d[b * 16 + j] = s[b * 22 + 6 + j];
                   for (int j = 0; j < 4; ++j) d[nb * 16 + b * 4 + j] = s[b * 22 + 2 + j];
                   d[nb * 20 + b * 2] = s[b * 22]; d[nb * 20 + b * 2 + 1] = s[b * 22 + 1]; break;
        case Q8_0: for (int j = 0; j < 32; ++j) d[b * 32 + j] = s[b * 34 + 2 + j];
                   d[nb * 32 + b * 2] = s[b * 34]; d[nb * 32 + b * 2 + 1] = s[b * 34 + 1]; break;
        case Q6_K: for (int j = 0; j < 210; ++j) d[b * 224 + j] = s[b * 210 + j]; break;
    }
}
// (checking) rows 0 .. 3 of a tensor decoded from the GGUF layout and from the aligned one
__global__ void k_fmt_test(const uint8_t* raw, const uint8_t* al, int fmt, int K, float* out) {
    const int r = blockIdx.x, g = threadIdx.x;
    if (g >= K / 8) return;
    float a[8], b[8];
    deq8(fmt, raw + size_t(r) * row_bytes(fmt, K), g, a, K);
    deq8(aligned_fmt(fmt), al + size_t(r) * row_bytes(aligned_fmt(fmt), K), g, b, K);
    for (int i = 0; i < 8; ++i) { out[(size_t(r) * K + g * 8 + i) * 2] = a[i]; out[(size_t(r) * K + g * 8 + i) * 2 + 1] = b[i]; }
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

// ---- the card / CPU hand-off (strata-flint's): host memory the card reads and writes directly. The card posts a MoE
// layer's input and experts and raises in_seq; the CPU computes the experts the card does not hold and raises out_seq;
// a card kernel spins on it and adds y. No host CUDA call inside the step, so it can be one graph.
struct HyShared {
    float h[kHidden];
    float y[kHidden];
    int idx[kTopK];
    float w[kTopK];
    alignas(128) uint32_t in_seq;
    alignas(128) uint32_t out_seq;
    uint32_t has;
};
__device__ __forceinline__ void hy_pause() {
#if defined(__HIPCC__)
    __builtin_amdgcn_s_sleep(1);
#else
    __nanosleep(100);
#endif
}
__global__ void __launch_bounds__(256) k_hy_post(const float* h, const int* idx, const float* wts, HyShared* S, uint32_t* ctr) {
    const int t = threadIdx.x;
    for (int i = t; i < kHidden; i += 256) S->h[i] = h[i];
    if (t < kTopK) { S->idx[t] = idx[t]; S->w[t] = wts[t]; }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        const uint32_t s = *ctr + 1;
        *ctr = s;
        *(volatile uint32_t*) &S->in_seq = s;
        __threadfence_system();
    }
}
__global__ void __launch_bounds__(256) k_hy_wait_add(float* h, HyShared* S, const uint32_t* ctr) {
    __shared__ uint32_t has;
    if (threadIdx.x == 0) {
        const uint32_t s = *ctr;
        while (*(volatile uint32_t*) &S->out_seq < s) hy_pause();
        __threadfence_system();
        has = *(volatile uint32_t*) &S->has;
    }
    __syncthreads();
    if (!has) return;
    const int i = blockIdx.x * 256 + threadIdx.x;
    h[i] += ((volatile float*) S->y)[i];
}

// the writing step's routing and hand-off in one: the experts as route_block picks them (each warp its 32 experts' top 8,
// then warp 0 the best 8 of those 128 - the same choice, ties to the lower index, with two block syncs instead of 17),
// then the MoE input, choices and weights posted to the CPU as k_hy_post does
__global__ void __launch_bounds__(512) k_route_post(const float* logits, const float* bias, int* idx, float* wts, const float* h,
                                                    HyShared* S, uint32_t* ctr) {
    __shared__ float sel[kExperts], prob[kExperts], gsc[kGroups], cv[128];
    __shared__ int ci[128], keep[kGroups], top[kTopK];
    const int e = threadIdx.x, lane = e & 31, w = e >> 5;
    for (int i = e; i < kHidden; i += kExperts) S->h[i] = h[i];   // (the input first: it does not wait on the choice)
    prob[e] = sigmoidf(logits[e]);
    sel[e] = prob[e] + bias[e];
    __syncthreads();
    if (w < kGroups) {   // warp w: group w's top-2 sum
        const float a = sel[w * 64 + lane], b = sel[w * 64 + 32 + lane];
        float m1 = fmaxf(a, b), m2 = fminf(a, b);
        for (int o = 16; o > 0; o >>= 1) {
            const float o1 = fl_shfl_xor(m1, o, 32), o2 = fl_shfl_xor(m2, o, 32);
            const float n1 = fmaxf(m1, o1), n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
            m1 = n1; m2 = n2;
        }
        if (lane == 0) gsc[w] = m1 + m2;
    }
    __syncthreads();
    if (e == 0) {
        int used = 0;
        for (int k = 0; k < kGroupsUsed; ++k) {
            int bi = -1;
            for (int gi = 0; gi < kGroups; ++gi) if (!(used >> gi & 1) && (bi < 0 || gsc[gi] > gsc[bi])) bi = gi;
            used |= 1 << bi;
        }
        for (int gi = 0; gi < kGroups; ++gi) keep[gi] = used >> gi & 1;
    }
    __syncthreads();
    // this warp's best 8 (warp-local, no block syncs)
    float v = keep[e >> 6] ? sel[e] : -INFINITY;
    for (int k = 0; k < kTopK; ++k) {
        float bv = v;
        int bi = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = fl_shfl_xor(bv, o, 32);
            const int oi = fl_shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { cv[w * kTopK + k] = bv; ci[w * kTopK + k] = bi; }
        if (e == bi) v = -INFINITY;
    }
    __syncthreads();
    if (w == 0) {   // the best 8 of the 16 warps' 8: four candidates a lane
        float lv[4];
        int li[4];
        for (int u = 0; u < 4; ++u) { lv[u] = cv[lane * 4 + u]; li[u] = ci[lane * 4 + u]; }
        for (int k = 0; k < kTopK; ++k) {
            int bu = 0;
            for (int u = 1; u < 4; ++u) if (lv[u] > lv[bu] || (lv[u] == lv[bu] && li[u] < li[bu])) bu = u;
            float bv = lv[bu];
            int bi = li[bu], bl = lane;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = fl_shfl_xor(bv, o, 32);
                const int oi = fl_shfl_xor(bi, o, 32), ol = fl_shfl_xor(bl, o, 32);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bl = ol; }
            }
            if (lane == 0) top[k] = bi;
            if (lane == bl) { lv[bu] = -INFINITY; li[bu] = 0x7fffffff; }
        }
        if (lane == 0) {
            float tot = 0;
            for (int k = 0; k < kTopK; ++k) tot += prob[top[k]];
            for (int k = 0; k < kTopK; ++k) {
                const float wk = prob[top[k]] / tot * kExpScale;
                idx[k] = top[k]; wts[k] = wk;
                S->idx[k] = top[k]; S->w[k] = wk;
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (e == 0) {
        const uint32_t sq = *ctr + 1;
        *ctr = sq;
        *(volatile uint32_t*) &S->in_seq = sq;
        __threadfence_system();
    }
}

#include "ling_prefill.cuh"

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
    std::vector<float> heat;              // the adaptive tier: each expert's recent routing (Model::adapt)
    std::vector<float> ffn_norm_h;
};

struct Model {
    std::vector<std::unique_ptr<strata::GgufFile>> parts;
    Layer L[kLayers];
    Tensor embed, head;
    const float* out_norm;
    int max_ctx;
    cudaStream_t st;
    float* mla_part = nullptr;
    float *h, *x1, *qb, *kb, *vb, *fb, *gb, *betab, *ob, *qa, *ol, *act, *logits, *rl, *wts, *hy;
    int *idx, *state, *host_state;
    int pos = 0;
    size_t loaded = 0;
    std::vector<int> layer_kvh;
    std::vector<float> clamp_exp, clamp_shexp;
    HyShared *hy_s = nullptr, *hy_s_dev = nullptr;   // the hand-off (mapped host memory; the card's view)
    uint32_t* hy_ctr = nullptr;
    uint32_t hy_seq = 0;
    cudaGraphExec_t graph = nullptr;
    size_t hy_cpu_experts = 0, hy_card_experts = 0, n_steps = 0;
    double hy_t_wait = 0, hy_t_cpu = 0;
    std::vector<std::vector<int>> plan;
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
        if (fmt == F32) {
            CK(cudaMalloc(&d, bytes));
            CK(cudaMemcpy(d, o->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
            loaded += bytes;
            return {(const uint8_t*) d, fmt};
        }
        // quantized: to the card as it is, then rearranged into the aligned form (LING_RAW_FMTS=6,14: those GGUF types
        // stay as they are - for checking)
        static const std::string raw_fmts = std::getenv("LING_RAW_FMTS") ? std::string(",") + std::getenv("LING_RAW_FMTS") + "," : "";
        if (raw_fmts.find("," + std::to_string(fmt) + ",") != std::string::npos) {
            CK(cudaMalloc(&d, bytes));
            CK(cudaMemcpy(d, o->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
            loaded += bytes;
            return {(const uint8_t*) d, fmt};
        }
        const int K = int(t->shape[0]);
        const size_t rows = bytes / row_bytes(fmt, K);
        const size_t abytes = rows * row_bytes(aligned_fmt(fmt), K);
        void* raw = nullptr;
        CK(cudaMalloc(&raw, bytes));
        CK(cudaMemcpy(raw, o->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d, abytes));
        repack((const uint8_t*) raw, (uint8_t*) d, rows, K, fmt);
        CK(cudaDeviceSynchronize());
        CK(cudaFree(raw));
        loaded += abytes;
        return {(const uint8_t*) d, aligned_fmt(fmt)};
    }
    void repack(const uint8_t* raw, uint8_t* dst, size_t rows, int K, int fmt, cudaStream_t s = 0) {
        const int nb = fmt == Q6_K ? K / 256 : K / 32;
        const size_t n = rows * nb;
        k_repack<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(raw, dst, rows, K, fmt);
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
        plan.assign(kLayers, {});
        if (const char* pf = std::getenv("LING_PLAN")) {   // lines "layer expert expert ..." (tools/flint/make_plan.py --model ling)
            FILE* f = std::fopen(pf, "r");
            if (!f) { std::fprintf(stderr, "strata-ling: cannot read the plan %s\n", pf); std::exit(1); }
            static char line[1 << 16];
            while (std::fgets(line, sizeof line, f)) {
                std::stringstream ss(line);
                int li;
                if (!(ss >> li) || li < 0 || li >= kLayers) continue;
                for (int e; ss >> e;) if (e >= 0 && e < kExperts) plan[li].push_back(e);
            }
            std::fclose(f);
        }
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
                // the plan's experts packed on the card (slot order); the rest stay in the file mapping for the CPU
                l.slot_h.assign(kExperts, -1);
                l.heat.assign(kExperts, 0.f);
                const std::vector<int>& R = plan[i];
                for (size_t j = 0; j < R.size(); ++j) l.slot_h[R[j]] = int(j);
                CK(cudaMalloc(&l.slot, kExperts * 4));
                CK(cudaMemcpy(l.slot, l.slot_h.data(), kExperts * 4, cudaMemcpyHostToDevice));
                l.ge = l.ue = l.de = nullptr;
                if (!R.empty()) {
                    const size_t eg = size_t(kFfExp) * row_bytes(l.fmt_eg, kHidden), ed = size_t(kHidden) * row_bytes(l.fmt_ed, kFfExp);
                    // (to the card as they are, then rearranged into the aligned form: same sizes)
                    uint8_t *cg, *cu, *cd, *raw;
                    CK(cudaMalloc(&cg, R.size() * eg)); CK(cudaMalloc(&cu, R.size() * eg)); CK(cudaMalloc(&cd, R.size() * ed));
                    CK(cudaMalloc(&raw, R.size() * std::max(eg, ed)));
                    const uint8_t* src[3] = {l.ge_h, l.ue_h, l.de_h};
                    uint8_t* dst[3] = {cg, cu, cd};
                    for (int m = 0; m < 3; ++m) {
                        const size_t eb = m < 2 ? eg : ed;
                        for (size_t j = 0; j < R.size(); ++j) CK(cudaMemcpy(raw + j * eb, src[m] + size_t(R[j]) * eb, eb, cudaMemcpyHostToDevice));
                        if (m < 2) repack(raw, dst[m], R.size() * kFfExp, kHidden, l.fmt_eg);
                        else repack(raw, dst[m], R.size() * kHidden, kFfExp, l.fmt_ed);
                        CK(cudaDeviceSynchronize());
                    }
                    CK(cudaFree(raw));
                    loaded += R.size() * (2 * eg + ed);
                    l.ge = cg; l.ue = cu; l.de = cd;
                }
                const float* nf = (const float*) host(b + "ffn_norm.weight");
                l.ffn_norm_h.assign(nf, nf + kHidden);
            }
        }
        auto f = [&](float*& p, size_t n) { CK(cudaMalloc(&p, n * 4)); };
        f(h, kHidden); f(x1, kHidden); f(qb, kTopK * kFfExp > kHeads * kQkHd ? kTopK * kFfExp : kHeads * kQkHd); f(kb, kInner); f(vb, kInner);
        f(fb, kInner); f(gb, kInner); f(betab, kHeads); f(ob, kInner > kFfDense ? kInner : kFfDense); f(qa, kHeads * kKvA);
        f(ol, kHeads * kLat); f(act, kActRows > kFfDense ? kActRows : kFfDense); f(logits, 157184); f(rl, kExperts); f(wts, kTopK);
        f(hy, kHidden);
        f(mla_part, size_t(kMlaChunks) * kHeads * kMlaPart);
        CK(cudaMalloc(&idx, kTopK * 4));
        CK(cudaMalloc(&state, 8)); CK(cudaMemset(state, 0, 8));
        CK(cudaMallocHost(&host_state, 8));
        CK(cudaHostAlloc((void**) &hy_s, sizeof(HyShared), cudaHostAllocMapped));
        std::memset((void*) hy_s, 0, sizeof(HyShared));
        CK(cudaHostGetDevicePointer((void**) &hy_s_dev, hy_s, 0));
        CK(cudaMalloc(&hy_ctr, 4));
        CK(cudaMemset(hy_ctr, 0, 4));
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
        k_mvt<<<total / kMvRows, 256, size_t(K) * 8 + 32 * 4, st>>>(J);
    }

    // one token (state[1]) through the model at state[0]; logits, then argmax into state[1] and state[0] + 1. The card's
    // work is one graph (record); this thread serves the MoE layers' CPU halves while it runs (hy_serve). LING_SYNC=1:
    // the older layer-by-layer path with host syncs (and LING_DUMP_H), for checking
    void step() {
        static const bool sync = std::getenv("LING_SYNC") != nullptr;
        if (sync) { step_sync(); return; }
        static const bool no_graph = std::getenv("LING_NO_GRAPH") != nullptr;
        flint_cpu::set_active(true);
        if (no_graph) record();
        else {
            if (!graph) {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                record();
                CK(cudaStreamEndCapture(st, &g));
                CK(cudaGraphInstantiate(&graph, g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
            }
            CK(cudaGraphLaunch(graph, st));
        }
        hy_serve();
        flint_cpu::set_active(false);
        ++pos;
        ++n_steps;
        adapt_after_step();
    }
    void record() {
        k_embed<<<(kHidden / 8 + 63) / 64, 64, 0, st>>>(embed.d, embed.fmt, state, h);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            attn(l);
            if (l.dense) { dense(l); continue; }
            mv({{Tensor{(const uint8_t*) l.router, F32}, rl}}, {kExperts}, h, kHidden, l.ffn_norm);
            k_route_post<<<1, kExperts, 0, st>>>(rl, l.bias, idx, wts, h, hy_s_dev, hy_ctr);
            k_moe_gu<<<kActRows / (8 * kGuR), 256, 0, st>>>(MoeGu{h, l.ffn_norm, l.ge, l.ue, l.gs.d, l.us.d, aligned_fmt(l.fmt_eg), l.gs.fmt, idx,
                                                         l.slot, l.clamp_e, l.clamp_s, act});
            k_moe_down<<<kHidden / (8 * kDnR), 256, 0, st>>>(MoeDown{h, act, l.de, l.ds.d, aligned_fmt(l.fmt_ed), l.ds.fmt, idx, l.slot, wts});
            k_hy_wait_add<<<kHidden / 256, 256, 0, st>>>(h, hy_s_dev, hy_ctr);
        }
        mv({{head, logits}}, {157184}, h, kHidden, out_norm);
        k_argmax<<<1, 1024, 0, st>>>(logits, 157184, state);
    }
    // the CPU's half of each MoE layer: wait for the card's post, compute the experts it does not hold, hand the sum back
    void hy_serve() {
        volatile HyShared* S = hy_s;
        for (int i = kDense; i < kLayers; ++i) {
            const Layer& l = L[i];
            const uint32_t sq = ++hy_seq;
            const auto w0 = std::chrono::steady_clock::now();
            for (long spin = 0; S->in_seq < sq; ++spin) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
                if ((spin & 0xfffff) == 0xfffff && std::chrono::steady_clock::now() - w0 > std::chrono::seconds(30)) {
                    std::fprintf(stderr, "strata-ling: the card never posted layer %d (%s)\n", i, cudaGetErrorString(cudaStreamQuery(st)));
                    std::exit(1);
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const auto w1 = std::chrono::steady_clock::now();
            hy_t_wait += std::chrono::duration<double>(w1 - w0).count();
            static FILE* route_f = std::getenv("LING_ROUTE_DUMP") ? std::fopen(std::getenv("LING_ROUTE_DUMP"), "ab") : nullptr;
            if (route_f) {
                int16_t rec[2 + kTopK] = {int16_t(i), 1};
                for (int k = 0; k < kTopK; ++k) rec[2 + k] = int16_t(S->idx[k]);
                std::fwrite(rec, 2, 2 + kTopK, route_f);
            }
            flint_cpu::Expert ex[kTopK];
            float w[kTopK];
            int n = 0;
            const size_t eb_gu = size_t(kFfExp) * row_bytes(l.fmt_eg, kHidden), eb_d = size_t(kHidden) * row_bytes(l.fmt_ed, kFfExp);
            for (int k = 0; k < kTopK; ++k) {
                const int e = S->idx[k];
                heat_l(i)[e] += heat_inc;
                if (l.slot_h[e] >= 0) { ++hy_card_experts; continue; }
                missed.push_back({i, e});
                ex[n] = {l.ge_h + size_t(e) * eb_gu, l.ue_h + size_t(e) * eb_gu, l.de_h + size_t(e) * eb_d};
                w[n++] = S->w[k];
            }
            hy_cpu_experts += n;
            if (n) flint_cpu::moe_q((const float*) S->h, l.ffn_norm_h.data(), kHidden, kFfExp, n, ex, w, (float*) S->y, l.fmt_eg, l.fmt_ed, l.clamp_e);
            S->has = n > 0;
            hy_t_cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - w1).count();
            std::atomic_thread_fence(std::memory_order_release);
            S->out_seq = sq;
        }
    }
    void attn(Layer& l) {
        if (l.kda) {
            mv({{l.wq, qb}, {l.wk, kb}, {l.wv, vb}, {l.wf, fb}, {l.wg, gb}, {Tensor{(const uint8_t*) l.beta_w, F32}, betab}},
               {kInner, kInner, kInner, kInner, kInner, kHeads}, h, kHidden, l.attn_norm);
            k_kda<<<kHeads, kHd, 0, st>>>(Kda{qb, kb, vb, fb, gb, betab, l.cq, l.ck, l.cv, l.dtb, l.a, l.onorm, l.cs, l.S, ob});
        } else {
            mv({{l.q, qb}, {l.kva, kb}, {l.gate, betab}}, {kHeads * kQkHd, kKvA, kHeads}, h, kHidden, l.attn_norm);
            k_mla_kv<<<1, kLat, 0, st>>>(kb, l.kv_norm, l.cache, state);
            k_mla_q<<<kHeads, 256, 0, st>>>(qb, l.kb.d, l.kb.fmt, qa, state);
            k_mla_attn_part<<<kMlaChunks, 256, 0, st>>>(qa, l.cache, mla_part, state);
            k_mla_merge<<<kHeads, 256, 0, st>>>(mla_part, ol);
            k_mla_out<<<kHeads, 256, 0, st>>>(ol, l.vb.d, l.vb.fmt, betab, ob);
        }
        mv({{l.wo, h}}, {kHidden}, ob, kInner, nullptr, true);
    }
    void dense(Layer& l) {
        mv({{l.fg, act}, {l.fu, ob}}, {kFfDense, kFfDense}, h, kHidden, l.ffn_norm);
        k_swiglu<<<(kFfDense + 255) / 256, 256, 0, st>>>(act, ob, kFfDense);
        mv({{l.fd, h}}, {kHidden}, act, kFfDense, nullptr, true);
    }
    void step_sync() {
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
            k_moe_gu<<<kActRows / (8 * kGuR), 256, 0, st>>>(MoeGu{h, l.ffn_norm, l.ge, l.ue, l.gs.d, l.us.d, aligned_fmt(l.fmt_eg), l.gs.fmt, idx,
                                                         l.slot, l.clamp_e, l.clamp_s, act});
            k_moe_down<<<kHidden / (8 * kDnR), 256, 0, st>>>(MoeDown{h, act, l.de, l.ds.d, aligned_fmt(l.fmt_ed), l.ds.fmt, idx, l.slot, wts});
            CK(cudaStreamSynchronize(st));
            // LING_ROUTE_DUMP=FILE: each token's 8 experts a layer, as int16 records [layer, 1, 8 ids] (tools/flint/make_plan.py)
            static FILE* route_f = std::getenv("LING_ROUTE_DUMP") ? std::fopen(std::getenv("LING_ROUTE_DUMP"), "ab") : nullptr;
            if (route_f) {
                int16_t rec[2 + kTopK] = {int16_t(i), 1};
                for (int k = 0; k < kTopK; ++k) rec[2 + k] = int16_t(hy_idx[k]);
                std::fwrite(rec, 2, 2 + kTopK, route_f);
            }
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
    // ------------------------------------------------ the prompt path (ling_prefill.cuh): chunks of up to pf_t tokens
    int pf_t = 0;
    float *pH = nullptr, *pX = nullptr, *pQ = nullptr, *pK = nullptr, *pV = nullptr, *pF = nullptr, *pG = nullptr, *pB = nullptr,
          *pO = nullptr, *pQa = nullptr, *pOl = nullptr, *pRL = nullptr, *pW = nullptr, *pAG = nullptr, *pAU = nullptr, *pY = nullptr,
          *pSG = nullptr, *pSU = nullptr, *pD1 = nullptr, *pD2 = nullptr, *pXc = nullptr;
    int *pIds = nullptr, *pIdx = nullptr, *pCount = nullptr, *pBounds = nullptr, *pFill = nullptr, *pTiles = nullptr, *pNT = nullptr,
        *pSrc = nullptr, *pSlot = nullptr, *hCount = nullptr;
    uint16_t *pXH = nullptr, *pXL = nullptr;   // the matrix-core products' split inputs (k_split16)
    float* pXS = nullptr;
    int split_K = 0;
    const uint8_t** wptr = nullptr;      // the grouped products' expert pointers: gate, up, down (3 x 512), device
    const uint8_t** wptr_h = nullptr;    // (pinned host copy)
    uint8_t* stg = nullptr;              // the chunk's routed experts the card does not hold, copied from the mapping
    uint8_t* stg_raw = nullptr;          // (as copied, before the rearrangement into stg)
    size_t stg_bytes = 0;
    static constexpr int kStgPiece = 32;
    // the staging areas, once: stg for the most experts a layer leaves off the card, stg_raw for one piece
    void stage_alloc() {
        if (stg) return;
        size_t most = 0, per = 0;
        for (const auto& l : L) {
            if (l.fmt_eg < 0) continue;   // (the dense layers)
            size_t off = 0;
            for (int e = 0; e < kExperts; ++e) off += l.slot_h.empty() || l.slot_h[e] < 0;
            most = std::max(most, off);
            per = std::max(per, 2 * size_t(kFfExp) * row_bytes(l.fmt_eg, kHidden) + size_t(kHidden) * row_bytes(l.fmt_ed, kFfExp));
        }
        stg_bytes = std::max<size_t>(1, most) * per;
        CK(cudaMalloc(&stg, stg_bytes));
        CK(cudaMalloc(&stg_raw, size_t(kStgPiece) * per));
    }
    float* pArena = nullptr;             // the attention's scratch, and (dead by then) the FFN's in the same place
    void pf_alloc(int T) {
        if (T <= pf_t) return;
        pf_t = T;
        auto f = [&](float*& p, size_t n) { if (p) CK(cudaFree(p)); CK(cudaMalloc(&p, n * 4)); };
        auto fi = [&](int*& p, size_t n) { if (p) CK(cudaFree(p)); CK(cudaMalloc(&p, n * 4)); };
        const size_t S = size_t(T) * kTopK, t = size_t(T);
        f(pH, t * kHidden); f(pX, t * kHidden);
        // attention: KDA's q k v f g, conv outputs and o, or MLA's q, kv, absorbed q and latent out; then the FFN's
        const size_t attn = t * (kHeads * kQkHd + 9 * size_t(kInner) + kHeads + kHeads * kKvA + kHeads * kLat);
        const size_t ffn = t * (2 * size_t(kFfDense) + kExperts) + S * (1 + 2 * kFfExp + kHidden);
        f(pArena, std::max(attn, ffn));
        float* a = pArena;
        auto take = [&](float*& p, size_t n) { p = a; a += n; };
        take(pQ, t * kHeads * kQkHd); take(pK, t * kInner); take(pV, t * kInner); take(pF, t * kInner); take(pG, t * kInner);
        take(pO, t * kInner); take(pD1, t * kInner); take(pD2, t * kInner); take(pXc, t * kInner); take(pB, t * kHeads);
        take(pQa, t * kHeads * kKvA); take(pOl, t * kHeads * kLat);
        a = pArena;
        take(pSG, t * kFfDense); take(pSU, t * kFfDense); take(pRL, t * kExperts); take(pW, S); take(pAG, S * kFfExp);
        take(pAU, S * kFfExp); take(pY, S * kHidden);
        {
            const size_t n = std::max({t * kFfDense, t * kInner, S * kFfExp});
            if (pXH) { CK(cudaFree(pXH)); CK(cudaFree(pXL)); CK(cudaFree(pXS)); }
            CK(cudaMalloc(&pXH, n * 2)); CK(cudaMalloc(&pXL, n * 2)); CK(cudaMalloc(&pXS, std::max<size_t>(S, T) * 4));
        }
        fi(pIds, T); fi(pIdx, S); fi(pCount, kExperts); fi(pBounds, kExperts + 1); fi(pFill, kExperts);
        fi(pTiles, 2 * (S / 64 + kExperts + 1)); fi(pNT, 1); fi(pSrc, S); fi(pSlot, S);
        if (!hCount) CK(cudaMallocHost(&hCount, kExperts * 4));
        if (!wptr) { CK(cudaMalloc(&wptr, 3 * kExperts * sizeof(void*))); CK(cudaMallocHost(&wptr_h, 3 * kExperts * sizeof(void*))); }
    }
    void gemm(const Tensor& w, int N, int K, const float* X, int ldx, int T, float* Y, int ldy, bool add) {
        Gemm J{};
        J.W = w.d; J.fmt = w.fmt; J.N = N; J.K = K; J.X = X; J.ldx = ldx; J.T = T; J.Y = Y; J.ldy = ldy; J.add = add;
        k_gemm<<<dim3((N + 63) / 64, (T + 63) / 64), 256, 0, st>>>(J);
    }
    // the matrix-core form: split X once (split), then any number of products on it (gemm_w); LING_PF_SIMT=1 keeps the fp32
    // SIMT products everywhere
    static bool simt() { static const bool v = std::getenv("LING_PF_SIMT") != nullptr; return v; }
    void split(const float* X, int ldx, int K, int rows) {
        k_split16<<<rows, 256, 0, st>>>(X, ldx, K, pXH, pXL, pXS);
        split_K = K;
    }
    void gemm_w(const Tensor& w, int N, int K, int T, float* Y, int ldy, bool add) {
        GemmW J{};
        J.W = w.d; J.fmt = w.fmt; J.N = N; J.K = K; J.xh = pXH; J.xl = pXL; J.xs = pXS; J.T = T; J.Y = Y; J.ldy = ldy; J.add = add;
        k_gemm_w<false><<<dim3((N + 255) / 256, (T + 63) / 64), 256, 0, st>>>(J);
    }
    // (SIMT or matrix cores) Y = W . X for X already split when matrix cores
    void prod(const Tensor& w, int N, int K, const float* X, int ldx, int T, float* Y, int ldy, bool add) {
        if (simt()) gemm(w, N, K, X, ldx, T, Y, ldy, add);
        else gemm_w(w, N, K, T, Y, ldy, add);
    }
    void gemm_grp_w(const uint8_t* const* wp, int fmt, int N, int K, const int* src, float* Y, int ldy, int S) {
        GemmW J{};
        J.wp = wp; J.fmt = fmt; J.N = N; J.K = K; J.xh = pXH; J.xl = pXL; J.xs = pXS; J.Y = Y; J.ldy = ldy; J.src = src;
        J.bounds = pBounds; J.tiles = pTiles; J.n_tiles = pNT;
        k_gemm_w<true><<<dim3((N + 255) / 256, S / 64 + kExperts + 1), 256, 0, st>>>(J);
    }
    void gemm_grp(const uint8_t* const* wp, int fmt, int N, int K, const float* X, int ldx, const int* src, float* Y, int ldy, int S) {
        Gemm J{};
        J.wp = wp; J.fmt = fmt; J.N = N; J.K = K; J.X = X; J.ldx = ldx; J.Y = Y; J.ldy = ldy; J.src = src;
        J.bounds = pBounds; J.tiles = pTiles; J.n_tiles = pNT;
        k_gemm<<<dim3((N + 63) / 64, S / 64 + kExperts + 1), 256, 0, st>>>(J);
    }
    void prefill_chunk(const int* ids, int T) {
        if (pos + T > max_ctx) { std::fprintf(stderr, "strata-ling: context full (%d)\n", max_ctx); std::exit(1); }
        pf_alloc(T);
        const int p0 = pos;
        CK(cudaMemcpyAsync(pIds, ids, T * 4, cudaMemcpyHostToDevice, st));
        k_pembed<<<T, kHidden / 8, 0, st>>>(embed.d, embed.fmt, pIds, pH);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            k_rms_rows<<<T, 256, 0, st>>>(pH, l.attn_norm, pX, kHidden);
            if (!simt()) split(pX, kHidden, kHidden, T);
            if (l.kda) {
                prod(l.wq, kInner, kHidden, pX, kHidden, T, pQ, kInner, false);
                prod(l.wk, kInner, kHidden, pX, kHidden, T, pK, kInner, false);
                prod(l.wv, kInner, kHidden, pX, kHidden, T, pV, kInner, false);
                prod(l.wf, kInner, kHidden, pX, kHidden, T, pF, kInner, false);
                prod(l.wg, kInner, kHidden, pX, kHidden, T, pG, kInner, false);
                gemm(Tensor{(const uint8_t*) l.beta_w, F32}, kHeads, kHidden, pX, kHidden, T, pB, kHeads, false);
                k_kda_conv<<<kInner / 256, 256, 0, st>>>(pQ, l.cq, l.cs, pD1, T);
                k_kda_conv<<<kInner / 256, 256, 0, st>>>(pK, l.ck, l.cs + 3 * kInner, pD2, T);
                k_kda_conv<<<kInner / 256, 256, 0, st>>>(pV, l.cv, l.cs + 6 * kInner, pXc, T);
                k_kda_conv_state<<<kInner / 256, 256, 0, st>>>(pQ, l.cs, T);
                k_kda_conv_state<<<kInner / 256, 256, 0, st>>>(pK, l.cs + 3 * kInner, T);
                k_kda_conv_state<<<kInner / 256, 256, 0, st>>>(pV, l.cs + 6 * kInner, T);
                k_kda_prep<<<T * kHeads, kHd, 0, st>>>(pD1, pD2, pF, l.dtb, l.a, pB);
                k_kda_scan<<<kHeads, kHd, 0, st>>>(pD1, pD2, pXc, pF, pB, l.S, pO, T);
                k_kda_out<<<T * kHeads, kHd, 0, st>>>(pO, pG, l.onorm);
            } else {
                prod(l.q, kHeads * kQkHd, kHidden, pX, kHidden, T, pQ, kHeads * kQkHd, false);
                prod(l.kva, kKvA, kHidden, pX, kHidden, T, pK, kKvA, false);
                gemm(l.gate, kHeads, kHidden, pX, kHidden, T, pB, kHeads, false);
                k_pmla_kv<<<T, kLat, 0, st>>>(pK, l.kv_norm, l.cache, p0);
                const size_t rbk = row_bytes(l.kb.fmt, kQkNope), rbv = row_bytes(l.vb.fmt, kLat);
                for (int hh = 0; hh < kHeads; ++hh)
                    gemm(Tensor{l.kb.d + size_t(hh) * kLat * rbk, l.kb.fmt}, kLat, kQkNope, pQ + hh * kQkHd, kHeads * kQkHd, T,
                         pQa + hh * kKvA, kHeads * kKvA, false);
                k_pmla_qrope<<<T * kHeads, 32, 0, st>>>(pQ, pQa, p0);
                k_pmla_attn<<<T, 256, 0, st>>>(pQa, l.cache, pOl, p0);
                for (int hh = 0; hh < kHeads; ++hh)
                    gemm(Tensor{l.vb.d + size_t(hh) * kHd * rbv, l.vb.fmt}, kHd, kLat, pOl + hh * kLat, kHeads * kLat, T,
                         pO + hh * kHd, kInner, false);
                k_pmla_gate<<<T, 256, 0, st>>>(pO, pB);
            }
            if (!simt()) split(pO, kInner, kInner, T);
            prod(l.wo, kHidden, kInner, pO, kInner, T, pH, kHidden, true);
            k_rms_rows<<<T, 256, 0, st>>>(pH, l.ffn_norm, pX, kHidden);
            if (!simt()) split(pX, kHidden, kHidden, T);
            if (l.dense) {
                prod(l.fg, kFfDense, kHidden, pX, kHidden, T, pSG, kFfDense, false);
                prod(l.fu, kFfDense, kHidden, pX, kHidden, T, pSU, kFfDense, false);
                k_pswiglu<<<(T * kFfDense + 255) / 256, 256, 0, st>>>(pSG, pSU, size_t(T) * kFfDense, 0.f);
                if (!simt()) split(pSG, kFfDense, kFfDense, T);
                prod(l.fd, kHidden, kFfDense, pSG, kFfDense, T, pH, kHidden, true);
                continue;
            }
            // shared expert, added into h (its act split afterwards: the routed experts' gate / up need pX's split first)
            prod(l.gs, kFfShared, kHidden, pX, kHidden, T, pSG, kFfShared, false);
            prod(l.us, kFfShared, kHidden, pX, kHidden, T, pSU, kFfShared, false);
            k_pswiglu<<<(T * kFfShared + 255) / 256, 256, 0, st>>>(pSG, pSU, size_t(T) * kFfShared, l.clamp_s);
            // routing, then the rows grouped by expert
            gemm(Tensor{(const uint8_t*) l.router, F32}, kExperts, kHidden, pX, kHidden, T, pRL, kExperts, false);
            CK(cudaMemsetAsync(pCount, 0, kExperts * 4, st));
            k_proute<<<T, kExperts, 0, st>>>(pRL, l.bias, pIdx, pW, pCount);
            k_pbounds<<<1, 512, 0, st>>>(pCount, pBounds, pFill, pTiles, pNT);
            k_pscatter<<<(T * kTopK + 255) / 256, 256, 0, st>>>(pIdx, T, pBounds, pFill, pSrc, pSlot);
            CK(cudaMemcpyAsync(hCount, pCount, kExperts * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            static FILE* route_f = std::getenv("LING_ROUTE_DUMP") ? std::fopen(std::getenv("LING_ROUTE_DUMP"), "ab") : nullptr;
            if (route_f) {   // [layer, T, T x 8 ids] (as the writing path's records)
                std::vector<int> ix(size_t(T) * kTopK);
                CK(cudaMemcpy(ix.data(), pIdx, ix.size() * 4, cudaMemcpyDeviceToHost));
                std::vector<int16_t> rec(2 + ix.size());
                rec[0] = int16_t(i); rec[1] = int16_t(T);
                for (size_t j = 0; j < ix.size(); ++j) rec[2 + j] = int16_t(ix[j]);
                std::fwrite(rec.data(), 2, rec.size(), route_f);
            }
            // each used expert's three matrices: on the card already, or copied into the staging area
            const size_t eg = size_t(kFfExp) * row_bytes(l.fmt_eg, kHidden), ed = size_t(kHidden) * row_bytes(l.fmt_ed, kFfExp);
            stage_alloc();
            // staged experts in pieces of kStgPiece: each piece's gates, ups and downs copied as they are into stg_raw,
            // then rearranged into its own place in stg (gates, ups, downs a piece)
            const size_t piece = 2 * eg + ed;
            int j = 0, j0 = 0;
            auto flush = [&]() {
                const int np = j - j0;
                if (!np) return;
                uint8_t* dst = stg + size_t(j0) * piece;
                repack(stg_raw, dst, size_t(2 * np) * kFfExp, kHidden, l.fmt_eg, st);   // (gates and ups together)
                repack(stg_raw + size_t(2 * np) * eg, dst + size_t(2 * np) * eg, size_t(np) * kHidden, kFfExp, l.fmt_ed, st);
                j0 = j;
            };
            std::vector<int> pend;
            for (int e = 0; e < kExperts; ++e) {
                wptr_h[e] = wptr_h[kExperts + e] = wptr_h[2 * kExperts + e] = nullptr;
                if (!hCount[e]) continue;
                if (l.slot_h[e] >= 0) {
                    const size_t sl = size_t(l.slot_h[e]);
                    wptr_h[e] = l.ge + sl * eg; wptr_h[kExperts + e] = l.ue + sl * eg; wptr_h[2 * kExperts + e] = l.de + sl * ed;
                    continue;
                }
                pend.push_back(e);
            }
            for (size_t q = 0; q < pend.size(); q += kStgPiece) {
                const int np = int(std::min(pend.size() - q, size_t(kStgPiece)));
                uint8_t* dst = stg + size_t(j) * piece;
                for (int u = 0; u < np; ++u) {
                    const int e = pend[q + u];
                    CK(cudaMemcpyAsync(stg_raw + u * eg, l.ge_h + size_t(e) * eg, eg, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(stg_raw + (np + u) * eg, l.ue_h + size_t(e) * eg, eg, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(stg_raw + 2 * np * eg + u * ed, l.de_h + size_t(e) * ed, ed, cudaMemcpyHostToDevice, st));
                    wptr_h[e] = dst + u * eg;
                    wptr_h[kExperts + e] = dst + (np + u) * eg;
                    wptr_h[2 * kExperts + e] = dst + 2 * np * eg + u * ed;
                }
                j += np;
                flush();
            }
            CK(cudaMemcpyAsync(wptr, wptr_h, 3 * kExperts * sizeof(void*), cudaMemcpyHostToDevice, st));
            const int S = T * kTopK;
            if (simt()) {
                gemm(l.ds, kHidden, kFfShared, pSG, kFfShared, T, pH, kHidden, true);
                gemm_grp(wptr, aligned_fmt(l.fmt_eg), kFfExp, kHidden, pX, kHidden, pSrc, pAG, kFfExp, S);
                gemm_grp(wptr + kExperts, aligned_fmt(l.fmt_eg), kFfExp, kHidden, pX, kHidden, pSrc, pAU, kFfExp, S);
                k_pswiglu<<<(unsigned) ((size_t(S) * kFfExp + 255) / 256), 256, 0, st>>>(pAG, pAU, size_t(S) * kFfExp, l.clamp_e);
                gemm_grp(wptr + 2 * kExperts, aligned_fmt(l.fmt_ed), kHidden, kFfExp, pAG, kFfExp, nullptr, pY, kHidden, S);
            } else {
                gemm_grp_w(wptr, aligned_fmt(l.fmt_eg), kFfExp, kHidden, pSrc, pAG, kFfExp, S);   // (pX's split)
                gemm_grp_w(wptr + kExperts, aligned_fmt(l.fmt_eg), kFfExp, kHidden, pSrc, pAU, kFfExp, S);
                split(pSG, kFfShared, kFfShared, T);
                gemm_w(l.ds, kHidden, kFfShared, T, pH, kHidden, true);
                k_pswiglu<<<(unsigned) ((size_t(S) * kFfExp + 255) / 256), 256, 0, st>>>(pAG, pAU, size_t(S) * kFfExp, l.clamp_e);
                split(pAG, kFfExp, kFfExp, S);
                gemm_grp_w(wptr + 2 * kExperts, aligned_fmt(l.fmt_ed), kHidden, kFfExp, nullptr, pY, kHidden, S);
            }
            k_pcombine<<<T, 256, 0, st>>>(pH, pY, pSlot, pW);
            // the adaptive tier: this chunk's routing counted, then its hottest staged experts promoted (copied on the card)
            for (int e = 0; e < kExperts; ++e) l.heat[e] += float(hCount[e]) * heat_inc;
            if (adapt_on() && l.ge) {
                static const int budget = std::getenv("LING_SWAPS_PROMPT") ? std::atoi(std::getenv("LING_SWAPS_PROMPT")) : 64;
                std::sort(pend.begin(), pend.end(), [&](int a, int b) { return l.heat[a] > l.heat[b]; });
                int done = 0;
                for (int e : pend) {
                    if (done >= budget) break;
                    const int c = coldest(l);
                    if (c < 0 || !worth(l, e, c)) break;
                    swap_in(l, e, c, wptr_h[e], wptr_h[kExperts + e], wptr_h[2 * kExperts + e]);
                    ++done;
                }
                if (done) push_slots(l);
            }
            CK(cudaStreamSynchronize(st));   // (wptr_h and the staging area are reused by the next layer)
        }
        // the last row through the head (the writing path's kernel), its argmax the next token
        mv({{head, logits}}, {157184}, pH + size_t(T - 1) * kHidden, kHidden, out_norm);
        const int st0[2] = {p0 + T - 1, 0};
        CK(cudaMemcpyAsync(state, st0, 4, cudaMemcpyHostToDevice, st));
        k_argmax<<<1, 1024, 0, st>>>(logits, 157184, state);
        CK(cudaStreamSynchronize(st));
        pos = p0 + T;
        heat_tick(T);
    }
    // ---- the adaptive tier (as Strata's for Qwen Next): the card's expert slots follow the routing. Each layer keeps
    // its plan's number of slots; an expert the CPU computed that is now hotter than the layer's coldest resident one
    // takes that one's slot. heat decays (the increment grows 0.2% a token: a half-life of about 350 tokens). Writing:
    // up to LING_SWAPS (2) swaps after a token, copied from the file mapping. The prompt path: up to LING_SWAPS_PROMPT
    // (64) a layer a chunk, copied from its staging area on the card. LING_ADAPT=0 keeps the plan fixed.
    float heat_inc = 1.f;
    std::vector<std::pair<int, int>> missed;            // this token's CPU experts (layer, expert)
    size_t n_swaps = 0;
    std::vector<float>& heat_l(int i) { return L[i].heat; }
    bool adapt_on() const { static const bool on = !(std::getenv("LING_ADAPT") && std::atoi(std::getenv("LING_ADAPT")) == 0); return on; }
    void heat_tick(int tokens) {
        heat_inc *= std::pow(1.002f, float(tokens));
        if (heat_inc > 1e20f) {   // (rescaled before it overflows)
            for (auto& l : L) for (auto& v : l.heat) v /= heat_inc;
            heat_inc = 1.f;
        }
    }
    // the layer's coldest resident expert, or -1 (no slots)
    int coldest(const Layer& l) const {
        int c = -1;
        for (int e = 0; e < kExperts; ++e) if (l.slot_h[e] >= 0 && (c < 0 || l.heat[e] < l.heat[c])) c = e;
        return c;
    }
    bool worth(const Layer& l, int e_in, int e_out) const { return l.heat[e_in] > 1.25f * l.heat[e_out] + 2.f * heat_inc; }
    // e_in into e_out's slot: its three matrices from the device copies given (aligned already), or from the mapping
    void swap_in(Layer& l, int e_in, int e_out, const uint8_t* dg = nullptr, const uint8_t* du = nullptr, const uint8_t* dd = nullptr) {
        const int sl = l.slot_h[e_out];
        const size_t eg = size_t(kFfExp) * row_bytes(l.fmt_eg, kHidden), ed = size_t(kHidden) * row_bytes(l.fmt_ed, kFfExp);
        uint8_t* g = (uint8_t*) l.ge + size_t(sl) * eg;
        uint8_t* u = (uint8_t*) l.ue + size_t(sl) * eg;
        uint8_t* d = (uint8_t*) l.de + size_t(sl) * ed;
        if (dg) {
            CK(cudaMemcpyAsync(g, dg, eg, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(u, du, eg, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(d, dd, ed, cudaMemcpyDeviceToDevice, st));
        } else {
            stage_alloc();
            CK(cudaMemcpyAsync(stg_raw, l.ge_h + size_t(e_in) * eg, eg, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(stg_raw + eg, l.ue_h + size_t(e_in) * eg, eg, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(stg_raw + 2 * eg, l.de_h + size_t(e_in) * ed, ed, cudaMemcpyHostToDevice, st));
            repack(stg_raw, g, kFfExp, kHidden, l.fmt_eg, st);
            repack(stg_raw + eg, u, kFfExp, kHidden, l.fmt_eg, st);
            repack(stg_raw + 2 * eg, d, kHidden, kFfExp, l.fmt_ed, st);
        }
        l.slot_h[e_in] = sl;
        l.slot_h[e_out] = -1;
        ++n_swaps;
    }
    void push_slots(Layer& l) { CK(cudaMemcpyAsync(l.slot, l.slot_h.data(), kExperts * 4, cudaMemcpyHostToDevice, st)); }
    void adapt_after_step() {
        heat_tick(1);
        if (!adapt_on() || missed.empty()) { missed.clear(); return; }
        static const int budget = std::getenv("LING_SWAPS") ? std::atoi(std::getenv("LING_SWAPS")) : 2;
        // the hottest of this token's misses first
        std::sort(missed.begin(), missed.end(), [&](const auto& a, const auto& b) { return L[a.first].heat[a.second] > L[b.first].heat[b.second]; });
        int done = 0;
        for (const auto& m : missed) {
            if (done >= budget) break;
            Layer& l = L[m.first];
            if (l.slot_h[m.second] >= 0) continue;
            const int c = coldest(l);
            if (c < 0 || !worth(l, m.second, c)) continue;
            swap_in(l, m.second, c);
            push_slots(l);
            ++done;
        }
        missed.clear();
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

    // the KDA layers' running state (S and the conv inputs) - all of the model's memory of the past besides the MLA
    // cache, whose rows stay valid below a position - copied out and back for conversation reuse (ling_serve.cuh)
    static constexpr size_t kSnapS = size_t(kHeads) * kHd * kHd, kSnapCs = size_t(9) * kInner;
    static size_t snap_floats() { return size_t(kLayers - kKdaFree) * (kSnapS + kSnapCs); }
    static constexpr int kKdaFree = 7;   // (the MLA layers: no running state)
    void set_pos(int p) {
        pos = p;
        const int v[2] = {p, 0};
        CK(cudaMemcpyAsync(state, v, 8, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
    }
    void clear_state() {
        for (auto& l : L) if (l.kda) {
            CK(cudaMemsetAsync(l.S, 0, kSnapS * 4, st));
            CK(cudaMemsetAsync(l.cs, 0, kSnapCs * 4, st));
        }
        set_pos(0);
    }
    void snap_copy(float* buf, bool save) {
        size_t o = 0;
        for (auto& l : L) if (l.kda) {
            CK(cudaMemcpyAsync(save ? buf + o : l.S, save ? l.S : buf + o, kSnapS * 4, cudaMemcpyDeviceToDevice, st));
            o += kSnapS;
            CK(cudaMemcpyAsync(save ? buf + o : l.cs, save ? l.cs : buf + o, kSnapCs * 4, cudaMemcpyDeviceToDevice, st));
            o += kSnapCs;
        }
        CK(cudaStreamSynchronize(st));
    }
};

#include "ling_serve.cuh"

}  // namespace

int main(int argc, char** argv) {
    std::string path, ids_s, logits_path;
    int n_new = 0, ctx = 4096;
    bool serve_mode = false;
    std::vector<int> eos_ids = {156895, 156892};   // <|role_end|>, <|endoftext|>
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
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos-ids") {
            eos_ids.clear();
            std::stringstream es(next());
            for (std::string v; std::getline(es, v, ',');) eos_ids.push_back(std::stoi(v));
        }
        else { std::fprintf(stderr, "usage: strata-ling --model PART1.gguf --ids 1,2,3|@FILE [--new N] [--max-ctx N] [--logits FILE] | --serve [--eos-ids A,B]\n"); return 2; }
    }
    std::vector<int> ids;
    { std::stringstream ss(ids_s); for (std::string v; std::getline(ss, v, ',');) if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids.push_back(std::stoi(v)); }
    if (path.empty() || (ids.empty() && !serve_mode)) { std::fprintf(stderr, "strata-ling: --model and --ids (or --serve) are needed\n"); return 2; }
    const auto t0 = std::chrono::steady_clock::now();
    if (const char* tn = std::getenv("LING_FMT_TEST")) {   // (checking) one tensor's rows 0..3, both layouts
        strata::GgufFile g1(path);
        std::string p2 = path; p2.replace(p2.find("-00001-of-"), 10, "-00002-of-");
        strata::GgufFile g2(p2);
        const strata::GgufFile* o = g1.find(tn) ? &g1 : &g2;
        const auto* t = o->find(tn);
        const int fmt = int(t->type), K = int(t->shape[0]);
        const size_t rb = row_bytes(fmt, K) * 4;
        uint8_t *raw, *al;
        float* out;
        CK(cudaMalloc(&raw, rb)); CK(cudaMalloc(&al, rb)); CK(cudaMalloc(&out, size_t(4) * K * 8));
        CK(cudaMemcpy(raw, o->tensor_data(*t), rb, cudaMemcpyHostToDevice));
        k_repack<<<(4 * (fmt == Q6_K ? K / 256 : K / 32) + 255) / 256, 256>>>(raw, al, 4, K, fmt);
        k_fmt_test<<<4, (K / 8 + 31) / 32 * 32>>>(raw, al, fmt, K, out);
        std::vector<float> h(size_t(4) * K * 2);
        CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < size_t(4) * K; ++i)
            if (h[2 * i] != h[2 * i + 1] && bad++ < 8) std::printf("value %zu (row %zu, col %zu): %g vs %g\n", i, i / K, i % K, h[2 * i], h[2 * i + 1]);
        std::printf("%s (type %d, K %d): %d of %d values differ\n", tn, fmt, K, bad, 4 * K);
        return 0;
    }
    Model m(path, ctx);
    std::fprintf(stderr, "strata-ling: %.1f GB on the card in %.1f s\n", m.loaded / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (serve_mode) return serve(m, eos_ids, std::getenv("LING_CHUNK") ? std::atoi(std::getenv("LING_CHUNK")) : 8192);
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    std::vector<float> lg(157184);
    const auto t1 = std::chrono::steady_clock::now();
    static const bool tokwise = std::getenv("LING_TOKWISE") != nullptr;
    const int chunk = std::getenv("LING_CHUNK") ? std::atoi(std::getenv("LING_CHUNK")) : 8192;
    if (!tokwise) {   // (with --logits: only the last position's, the prompt path computes no others)
        for (size_t i = 0; i < ids.size(); i += chunk) m.prefill_chunk(ids.data() + i, int(std::min(ids.size() - i, size_t(chunk))));
        if (lf) {
            CK(cudaMemcpyAsync(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost, m.st));
            CK(cudaStreamSynchronize(m.st));
            std::fwrite(lg.data(), 4, lg.size(), lf);
        }
    } else for (int t : ids) {
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
    if (m.n_swaps) std::fprintf(stderr, "strata-ling: %zu experts moved onto the card\n", m.n_swaps);
    if (m.hy_card_experts + m.hy_cpu_experts)
        std::fprintf(stderr, "strata-ling: %.0f%% of the chosen experts on the card; a token: %.2f ms waiting for the card, %.2f ms on the CPU\n",
                     100.0 * m.hy_card_experts / double(m.hy_card_experts + m.hy_cpu_experts), m.hy_t_wait * 1e3 / std::max<size_t>(1, m.n_steps),
                     m.hy_t_cpu * 1e3 / std::max<size_t>(1, m.n_steps));
    return 0;
}
