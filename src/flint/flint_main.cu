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
#include <atomic>
#include <sys/mman.h>
#include <thread>
#include "flint_cpu.hpp"
#include <unordered_map>
#include <string>
#include <vector>

namespace {
// ---- the few things that differ between AMD (HIP, wave32) and NVIDIA (CUDA built by clang, see
// tools/cuda/build-flint.sh): warp shuffles, lane swaps, byte permutes, sleeps, barriers, the 16x16x16 matrix product
#if defined(__HIPCC__)
#define FL_HIP 1
template <class T> __device__ __forceinline__ T fl_shfl_xor(T v, int m, int w = 32) { return __shfl_xor(v, m, w); }
__device__ __forceinline__ bool fl_any(bool p) { return __any(p); }
__device__ __forceinline__ float fl_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ void fl_wave_barrier() { __builtin_amdgcn_wave_barrier(); }
__device__ __forceinline__ uint32_t fl_perm(uint32_t a, uint32_t b, uint32_t sel) { return __builtin_amdgcn_perm(a, b, sel); }
__device__ __forceinline__ float fl_swap16(float v) {   // lane l <-> lane l ^ 16
    return __int_as_float((int) __builtin_amdgcn_permlanex16((unsigned) __float_as_int(v), (unsigned) __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}
#define fl_sleep(n) __builtin_amdgcn_s_sleep(n)
#else
#define FL_HIP 0
template <class T> __device__ __forceinline__ T fl_shfl_xor(T v, int m, int w = 32) { return __shfl_xor_sync(0xffffffffu, v, m, w); }
__device__ __forceinline__ bool fl_any(bool p) { return __any_sync(0xffffffffu, p); }
__device__ __forceinline__ float fl_exp2(float x) { return exp2f(x); }
__device__ __forceinline__ void fl_wave_barrier() { __syncwarp(); }
// AMD's v_perm_b32: bytes 0-3 of {a:b} are b's, 4-7 a's; a selector byte 0-7 picks one (every use here stays in 0-7)
__device__ __forceinline__ uint32_t fl_perm(uint32_t a, uint32_t b, uint32_t sel) {
    const uint32_t s = (sel & 7) | ((sel >> 8) & 7) << 4 | ((sel >> 16) & 7) << 8 | ((sel >> 24) & 7) << 12;
    return __byte_perm(b, a, s);
}
__device__ __forceinline__ float fl_swap16(float v) { return __shfl_xor_sync(0xffffffffu, v, 16); }
#define fl_sleep(n) __nanosleep(64 * (n))
// (clang 19's NVPTX back end has no acquire load: a volatile read then a fence; a fence then a volatile write)
template <class T> __device__ __forceinline__ T fl_ld_acq(const T* p) { const T v = *(const volatile T*) p; __threadfence(); return v; }
template <class T> __device__ __forceinline__ void fl_st_rel(T* p, T v) { __threadfence(); *(volatile T*) p = v; }
#define __hip_atomic_load(p, order, scope) fl_ld_acq(p)
#define __hip_atomic_store(p, v, order, scope) fl_st_rel((p), (v))
#define __HIP_MEMORY_SCOPE_AGENT 0
#endif


#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "strata-flint: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kHidden = 2048, kLayers = 49, kDense = 2;
constexpr int kHeads = 32, kKv = 4, kHd = 128, kQDim = kHeads * kHd, kKvDim = kKv * kHd;
constexpr int kFfDense = 6144, kExperts = 256, kTopK = 6, kFfExp = 512, kFfShared = 1024;
constexpr int kWindow = 513;
#ifndef FLINT_SELFROUND
#define FLINT_SELFROUND 0   // (the current position's V rounded as stored, in its own attention)
#endif
#ifndef FLINT_CHUNKS
#define FLINT_CHUNKS 16   // attention: a query's keys in up to this many parts
#endif
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
// (experiment) FLINT_Q8SIM=1: the full layers' K and V rounded to 8 bits with one scale per head row before storing
__constant__ int q8sim = 0, q8grp = 128;
__device__ inline float q8rt(float x, float amax, int bit = 3) {
    if (!(q8sim & bit) || amax == 0.f) return x;
    const float d = amax / 127.f;
    return rintf(x / d) * d;
}
// The full layers' V in 8 bits (values a head row of 128 share one scale): their K stays fp16 (8-bit K cost KL 0.011
// on Flint, 8-bit V 0.007 against 0.006 for fp16). q null: V is the plain cache.
// FLINT_K8: K too (kq null: K is the plain cache) - one scale a head row, scales grouped as V's, a row's values in the
// order the writing attention's lanes read them (lane half g: dims s * 16 + g * 8 + i, s = 0..7, as 64 bytes)
struct V8 { int8_t* q; float* s; int8_t* kq = nullptr; float* ks = nullptr; };
// Stored in groups of 16 positions, dimension by dimension: a group's 16 values of one dim (and K/V head) sit side by
// side, so attention reads V^T straight from memory. The scales are grouped the same way.
__device__ inline size_t v8_qi(size_t slot, int kh, int d) { return ((slot >> 4) * kKv + kh) * (kHd * 16) + size_t(d) * 16 + (slot & 15); }
__device__ inline size_t v8_si(size_t slot, int kh) { return ((slot >> 4) * kKv + kh) * 16 + (slot & 15); }
// v (128 values of one position and K/V head, one a thread of 128) into the 8-bit store; amax over the 128 given
__device__ inline void v8_put(V8 st, size_t slot, int kh, int i, float v, float amax) {
    const float d = amax > 0.f ? amax / 127.f : 1.f;
    st.q[v8_qi(slot, kh, i)] = int8_t(max(-127.f, min(127.f, rintf(v / d))));
    if (i == 0) st.s[v8_si(slot, kh)] = d;
}
// loads that pass through without settling in the caches (the K/V history: it would push out the prefetched weights)
typedef unsigned int nt_u4 __attribute__((ext_vector_type(4)));
typedef unsigned int nt_u2 __attribute__((ext_vector_type(2)));
typedef float nt_f4 __attribute__((ext_vector_type(4)));
#ifndef FLINT_KV_NT
#define FLINT_KV_NT 0
#endif
__device__ inline uint4 ld_nt(const void* p) {
    if (!FLINT_KV_NT) return *(const uint4*) p;
    const nt_u4 v = __builtin_nontemporal_load((const nt_u4*) p);
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ inline uint2 ld_nt2(const void* p) {
    if (!FLINT_KV_NT) return *(const uint2*) p;
    const nt_u2 v = __builtin_nontemporal_load((const nt_u2*) p);
    return make_uint2(v.x, v.y);
}
__device__ inline float4 ld_ntf(const void* p) {
    if (!FLINT_KV_NT) return *(const float4*) p;
    const nt_f4 v = __builtin_nontemporal_load((const nt_f4*) p);
    return make_float4(v.x, v.y, v.z, v.w);
}
__device__ inline int k8_perm(int d) { return ((d >> 3) & 1) * 64 + (d >> 4) * 8 + (d & 7); }
// a key row's 128 bytes (head kh, position slot): the 16 rows of a group and head side by side, as V's groups
__device__ inline size_t k8_row(size_t slot, int kh) { return ((slot >> 4) * kKv + kh) * (kHd * 16) + (slot & 15) * kHd; }
__device__ inline void k8_put(V8 st, size_t slot, int kh, int i, float k, float amax) {
    const float d = amax > 0.f ? amax / 127.f : 1.f;
    st.kq[k8_row(slot, kh) + k8_perm(i)] = int8_t(max(-127.f, min(127.f, rintf(k / d))));
    if (i == 0) st.ks[v8_si(slot, kh)] = d;
}
template <class T> __device__ inline T f2c(float v);
template <> __device__ inline __half f2c<__half>(float v) { return __float2half(v); }
template <> __device__ inline float f2c<float>(float v) { return v; }   // sliding layers' K/V slots: the window plus the largest prompt chunk (8192)
constexpr float kEps = 1e-6f, kRopeBase = 10000.f;
constexpr int kQ4_1 = 3, kF32 = 0;

struct block_q4_1 { __half d, m; uint8_t qs[16]; };     // value = d * q + m, q in 0..15 (low nibbles: 0-15, high: 16-31)
static_assert(sizeof(block_q4_1) == 20, "block layout");

// ---------------------------------------------------------------------------------------------------- kernels
__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += fl_shfl_xor(v, o, 32);
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
// D = A . B + C, 16 x 16 x 16, fp16 in, fp32 sums, in gfx12's wave32 WMMA layout: lane (l16, g = lane >> 4) holds
// A[l16][g * 8 + i], B[g * 8 + i][l16] and C/D[g * 8 + i][l16]
__device__ __forceinline__ pf_v8f pf_wmma(const pf_v8h& a, const pf_v8h& b, const pf_v8f& c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#elif !FL_HIP
    // NVIDIA: the operands through shared memory into mma.sync m16n8k16's layout (two products for the 16 columns),
    // the result back the same way. Exact, not fast: the hot kernels get native-layout versions.
    __shared__ __align__(16) float wb[8][256];   // a warp's: A, B as fp16, then D as fp32 (blocks of up to 256)
    const int lane = threadIdx.x & 31, w = (threadIdx.x >> 5) & 7, l16 = lane & 15, g = lane >> 4;
    const int gid = lane >> 2, tig = lane & 3;
    _Float16* AB = (_Float16*) wb[w];
#pragma unroll
    for (int i = 0; i < 8; ++i) { AB[l16 * 16 + g * 8 + i] = a[i]; AB[256 + l16 * 16 + g * 8 + i] = b[i]; }   // A[m][k], Bt[n][k]
    __syncwarp();
    const uint32_t* A32 = (const uint32_t*) AB;
    const uint32_t a0 = A32[(gid * 16 + 2 * tig) / 2], a1 = A32[((gid + 8) * 16 + 2 * tig) / 2];
    const uint32_t a2 = A32[(gid * 16 + 2 * tig + 8) / 2], a3 = A32[((gid + 8) * 16 + 2 * tig + 8) / 2];
    float d[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t b0 = A32[(256 + (8 * h + gid) * 16 + 2 * tig) / 2], b1 = A32[(256 + (8 * h + gid) * 16 + 2 * tig + 8) / 2];
        d[h][0] = d[h][1] = d[h][2] = d[h][3] = 0.f;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[h][0]), "+f"(d[h][1]), "+f"(d[h][2]), "+f"(d[h][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncwarp();
    float* D = wb[w];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        D[gid * 16 + 8 * h + 2 * tig] = d[h][0]; D[gid * 16 + 8 * h + 2 * tig + 1] = d[h][1];
        D[(gid + 8) * 16 + 8 * h + 2 * tig] = d[h][2]; D[(gid + 8) * 16 + 8 * h + 2 * tig + 1] = d[h][3];
    }
    __syncwarp();
    pf_v8f out;
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = c[i] + D[(g * 8 + i) * 16 + l16];
    __syncwarp();
    return out;
#else
    __trap();
    return c;
#endif
}
// LDS-only barrier (a __syncthreads' fence also covers global memory and invalidates the vector caches on gfx12)
__device__ __forceinline__ void pf_sync() {
#if FL_HIP
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
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
        const int sl = idx[row / kFfExp];
        if (sl < 0) {   // (hybrid: an expert the card does not hold - the CPU computes it)
            if (lane == 0) for (int r = 0; r < kMvRpw; ++r) act[row + r] = 0.f;
            return;
        }
        const size_t e = size_t(sl) * kFfExp + row % kFfExp;
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
        if (idx[k] < 0) continue;   // (hybrid: on the CPU)
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

// hybrid: the card and the CPU meet in host memory the card reads and writes directly (as Strata's verify windows
// do): the card posts a layer's MoE input and raises in_seq; the CPU computes the experts the card does not hold and
// raises out_seq; a card kernel spins on it and adds the CPU's y. No host CUDA call inside the step, so it stays a graph.
struct HyShared {
    float h[kHidden];
    float y[kHidden];
    int idx[kTopK];
    float w[kTopK];
    alignas(128) uint32_t in_seq;    // card -> CPU: this layer's input is posted
    alignas(128) uint32_t out_seq;   // CPU -> card: its share is in y (has: anything at all)
    uint32_t has;
};
__device__ __forceinline__ void hy_pause() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    fl_sleep(1);
#else
    __nanosleep(100);
#endif
}
// the layer's input, its experts and their weights to the CPU; each chosen expert's card slot (or -1)
__global__ void __launch_bounds__(256) k_hy_post(const float* h, const int* idx, const float* wts, const int* slot, int* ridx,
                                                 HyShared* S, uint32_t* ctr) {
    const int t = threadIdx.x;
    for (int i = t; i < kHidden; i += 256) S->h[i] = h[i];
    if (t < kTopK) { S->idx[t] = idx[t]; S->w[t] = wts[t]; ridx[t] = slot[idx[t]]; }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        const uint32_t s = *ctr + 1;
        *ctr = s;
        *(volatile uint32_t*) &S->in_seq = s;
        __threadfence_system();
    }
}
// wait for the CPU's share of this layer, then h += y
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
            const float ov = fl_shfl_xor(v, o, 32);
            const int oi = fl_shfl_xor(i, o, 32);
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
                           vc_t* V, const int* state, bool rope, int kv_mask, int zs = 0, V8 v8 = {}) {
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
        const float vv = v[kh * kHd + i];
        float ak = 0, av = 0;
        if (q8sim && !rope) {
            __shared__ float mk[4], mv[4];
            float a = fabsf(y), b = fabsf(vv);
            for (int o = 16; o > 0; o >>= 1) { a = fmaxf(a, fl_shfl_xor(a, o, 32)); b = fmaxf(b, fl_shfl_xor(b, o, 32)); }
            if ((i & 31) == 0) { mk[i >> 5] = a; mv[i >> 5] = b; }
            __syncthreads();
            if (q8grp == 32) { ak = mk[i >> 5]; av = mv[i >> 5]; }
            else { ak = fmaxf(fmaxf(mk[0], mk[1]), fmaxf(mk[2], mk[3])); av = fmaxf(fmaxf(mv[0], mv[1]), fmaxf(mv[2], mv[3])); }
        }
        if (v8.kq) {
            __shared__ float n8[4];
            float a = fabsf(y);
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, fl_shfl_xor(a, o, 32));
            if ((i & 31) == 0) n8[i >> 5] = a;
            __syncthreads();
            k8_put(v8, size_t(pos & kv_mask), kh, i, y, fmaxf(fmaxf(n8[0], n8[1]), fmaxf(n8[2], n8[3])));
        } else K[r] = f2c<kc_t>(q8rt(y, ak, 1));
        if (v8.q) {
            __shared__ float m8[4];
            float a = fabsf(vv);
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, fl_shfl_xor(a, o, 32));
            if ((i & 31) == 0) m8[i >> 5] = a;
            __syncthreads();
            v8_put(v8, size_t(pos & kv_mask), kh, i, vv, fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])));
        } else V[r] = f2c<vc_t>(q8rt(vv, av, 2));
    }
}

// Split attention for one query position: block (h, c) takes chunk c of the keys lo..pos (kAttnChunks equal parts)
// for query head h; each warp walks its keys with an online softmax (lane = 4 of the 128 dims), the 8 warps merge in
// LDS, and the chunk's (max, sum, unnormalized output) goes to part. k_attn_merge adds the chunks and the sink.
constexpr int kAttnChunks = FLINT_CHUNKS, kPart = kHd + 4, kMinChunk = FLINT_MINCHUNK;   // a chunk's record: max, sum, 2 spare, 128 outputs
__device__ inline float4 h4(const float* p) { return *(const float4*) p; }
__device__ inline float4 h4(const __half* p) {
    const uint2 u = *(const uint2*) p;
    __half2 ha, hb;
    std::memcpy(&ha, &u.x, 4);
    std::memcpy(&hb, &u.y, 4);
    const float2 a = __half22float2(ha), b = __half22float2(hb);
    return make_float4(a.x, a.y, b.x, b.y);
}
// The writing path's attention also does what k_qkv_post did when F.qn is set: q (its head) and this position's k
// RMSNorm (+ RoPE on sliding layers) from the raw products, k / v of this position into the cache (one block a K/V
// head writes them), and that position's key used from LDS (other blocks may not have written it yet).
struct AttnFuse { const float *k, *v, *qn, *kn; kc_t* Kw; vc_t* Vw; };   // (the 8-bit V store comes as the kernel's v8)
// rmsnorm * w (+ NEOX RoPE at pos) of 128 values in place, by one warp (4 a lane)
__device__ inline void head_norm_warp(float* t, const float* w, int pos, bool rope, int lane) {
    float y[4], ss = 0;
    for (int j = 0; j < 4; ++j) { y[j] = t[lane + 32 * j]; ss += y[j] * y[j]; }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / kHd + kEps);
    for (int j = 0; j < 4; ++j) y[j] = y[j] * r * w[lane + 32 * j];
    if (rope) {
        float o4[4];
        for (int j = 0; j < 2; ++j) {
            const int i = lane + 32 * j;
            const float ang = float(pos) * powf(kRopeBase, -2.f * i / kHd);
            float sn, cs;
            sincosf(ang, &sn, &cs);
            o4[j] = y[j] * cs - y[j + 2] * sn;
            o4[j + 2] = y[j] * sn + y[j + 2] * cs;
        }
        for (int j = 0; j < 4; ++j) y[j] = o4[j];
    }
    for (int j = 0; j < 4; ++j) t[lane + 32 * j] = y[j];
}
__global__ void __launch_bounds__(256) k_attn_part(const float* q, const kc_t* K, const vc_t* V, float* part,
                                                   const int* state, bool swa, int kv_mask, int zs = 0, AttnFuse F = {}) {
    q += blockIdx.z * zs;
    part += size_t(blockIdx.z) * kHeads * kAttnChunks * kPart;
    const int h = blockIdx.x, c = blockIdx.y, kh = h / (kHeads / kKv);
    const int pos = state[0] + blockIdx.z, lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks), a = lo + c * per, b = min(pos + 1, a + per);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __shared__ __align__(16) float qf[kHd], kf[kHd], vf[kHd];
    const bool fused = F.qn != nullptr;
    if (fused) {
        if (threadIdx.x < kHd) { qf[threadIdx.x] = q[h * kHd + threadIdx.x]; kf[threadIdx.x] = F.k[kh * kHd + threadIdx.x]; vf[threadIdx.x] = F.v[kh * kHd + threadIdx.x]; }
        __syncthreads();
        if (w == 0) head_norm_warp(qf, F.qn, pos, swa, lane);
        else if (w == 1) head_norm_warp(kf, F.kn, pos, swa, lane);
        __syncthreads();
        if (c == 0 && h % (kHeads / kKv) == 0 && threadIdx.x < kHd) {
            const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd + threadIdx.x;
            F.Kw[r] = f2c<kc_t>(kf[threadIdx.x]);
            F.Vw[r] = f2c<vc_t>(vf[threadIdx.x]);
        }
    }
    const float4 qv = fused ? *(const float4*) (qf + lane * 4) : *(const float4*) (q + h * kHd + lane * 4);
    const float scale = rsqrtf(float(kHd));
    float m = -INFINITY, l = 0;
    float4 o = make_float4(0, 0, 0, 0);
    for (int p = a + w; p < b; p += 8) {
        // (this position, fused: its k / v rounded as the cache will hold them)
        const bool here = fused && p == pos;
        const float4 kv = here ? make_float4(c2f(f2c<kc_t>(kf[lane * 4])), c2f(f2c<kc_t>(kf[lane * 4 + 1])), c2f(f2c<kc_t>(kf[lane * 4 + 2])), c2f(f2c<kc_t>(kf[lane * 4 + 3])))
                               : h4(K + size_t(p & kv_mask) * kKvDim + kh * kHd + lane * 4);
        const float d = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w) * scale;
        const float mn = fmaxf(m, d), cf = expf(m - mn), e = expf(d - mn);
        const float4 vv = here ? make_float4(c2f(f2c<vc_t>(vf[lane * 4])), c2f(f2c<vc_t>(vf[lane * 4 + 1])), c2f(f2c<vc_t>(vf[lane * 4 + 2])), c2f(f2c<vc_t>(vf[lane * 4 + 3])))
                               : h4(V + size_t(p & kv_mask) * kKvDim + kh * kHd + lane * 4);
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
                                                    const int* __restrict__ state, int swa, int kv_mask, int zs = 0,
                                                    AttnFuse F = {}, V8 v8 = {}) {
    q += blockIdx.z * zs;
    part += size_t(blockIdx.z) * kHeads * kAttnChunks * kPart;
    __shared__ __align__(16) float qf8[8][kHd], kf[kHd], vf[kHd];
    const bool fused = F.qn != nullptr;
    // a warp's V tile (transposed) during the loop; the same LDS holds the warps' outputs for the merge after it
    __shared__ __align__(16) char lds[8 * kHd * (16 + 8) * 2];
    _Float16 (*vt)[kHd][16 + 8] = (_Float16 (*)[kHd][16 + 8]) lds;
    float (*wo)[8][kHd] = (float (*)[8][kHd]) lds;
    __shared__ float wm[8][8], wl[8][8];
    const int kh = blockIdx.x, c = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int pos = state[0] + blockIdx.z, lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks);
    if (v8.q) per = (per + 15) & ~15;   // (tiles start on a group of 16 positions)
    const int a0 = lo + c * per, b0 = min(pos + 1, a0 + per);
    const int hq = kh * (kHeads / kKv) + l16;                // this lane's query head (columns 8-15: none)
    if (fused) {   // the 8 query heads (warp w: head kh * 8 + w) and this position's k, normed; k / v into the cache
        for (int i = lane; i < kHd; i += 32) qf8[warp][i] = q[(kh * 8 + warp) * kHd + i];
        if (warp == 0) for (int i = lane; i < kHd; i += 32) { kf[i] = F.k[kh * kHd + i]; vf[i] = F.v[kh * kHd + i]; }
        __syncthreads();
        head_norm_warp(qf8[warp], F.qn, pos, false, lane);
        if (warp == 0) head_norm_warp(kf, F.kn, pos, false, lane);
        __syncthreads();
        if (c == 0 && tid < kHd) {
            const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd + tid;
            float ak = 0, av = 0;
            if (q8sim) { const int g0 = tid / q8grp * q8grp; for (int i = g0; i < g0 + q8grp; ++i) { ak = fmaxf(ak, fabsf(kf[i])); av = fmaxf(av, fabsf(vf[i])); } }
            if (v8.kq) {
                float a8 = 0;
                for (int i = 0; i < kHd; ++i) a8 = fmaxf(a8, fabsf(kf[i]));
                k8_put(v8, size_t(pos & kv_mask), kh, tid, kf[tid], a8);
            } else F.Kw[r] = f2c<kc_t>(q8rt(kf[tid], ak, 1));
            if (v8.q) {
                float a8 = 0;
                for (int i = 0; i < kHd; ++i) a8 = fmaxf(a8, fabsf(vf[i]));
                v8_put(v8, size_t(pos & kv_mask), kh, tid, vf[tid], a8);
            } else F.Vw[r] = f2c<vc_t>(q8rt(vf[tid], av, 2));
        }
    }
    pf_v8h qb[8];
#pragma unroll
    for (int s8 = 0; s8 < 8; ++s8)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s8][i] = (_Float16) (l16 < 8 ? (fused ? qf8[l16][s8 * 16 + g * 8 + i] : q[hq * kHd + s8 * 16 + g * 8 + i]) : 0.f);
    const float scale = rsqrtf(float(kHd));
    float m = -INFINITY, l = 0;
    pf_v8f o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    // the scores of a tile into the running max / sum: returns P (this lane: head l16, keys t0 + g * 8 + i)
    auto soft = [&](const pf_v8f& sc, int t0, float& cf) {
        float mx = -INFINITY, v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int p = t0 + g * 8 + i;
            v[i] = (l16 < 8 && p < b0) ? sc[i] * scale : -INFINITY;
            mx = fmaxf(mx, v[i]);
        }
        mx = fmaxf(mx, fl_shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx), ref = mn == -INFINITY ? 0.f : mn;
        cf = expf(m - ref);
        m = mn;
        pf_v8h pb;
        float ps = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) { const float e = expf(v[i] - ref); ps += e; pb[i] = (_Float16) e; }
        l = l * cf + ps;
        return pb;
    };
    if (sizeof(kc_t) == 2 && v8.q) {
        // 8-bit V (fp16 K): no LDS - lane (dim j * 16 + l16, keys t0 + g * 8 ..) reads its V^T operand straight from
        // the store, and the next tile's K and V are fetched into registers while this one is used
        uint4 kn[8];
        uint2 vn[8];
        float4 sn0, sn1, kn0, kn1;
        auto fetch = [&](int t0) {
            const int p = t0 + l16;
            const int pk = t0 + g * 8;
            const bool ok = pk < b0;
            const size_t sk = size_t(pk & kv_mask);
            if (v8.kq) {   // (8-bit K: the lane's 64 bytes, and the scales of keys pk ..)
                const uint4* kr = (const uint4*) (v8.kq + k8_row(size_t(p & kv_mask), kh) + g * 64);
#pragma unroll
                for (int s4 = 0; s4 < 4; ++s4) kn[s4] = p < b0 ? ld_nt(kr + s4) : make_uint4(0, 0, 0, 0);
                const float* kp = v8.ks + v8_si(sk, kh);
                kn0 = ok ? ld_ntf(kp) : make_float4(0, 0, 0, 0);
                kn1 = ok ? ld_ntf(kp + 4) : make_float4(0, 0, 0, 0);
            } else {
                const uint4* kr = (const uint4*) ((const char*) K + (size_t(p & kv_mask) * kKvDim + kh * kHd + g * 8) * 2);
#pragma unroll
                for (int s8 = 0; s8 < 8; ++s8) kn[s8] = p < b0 ? ld_nt(kr + s8 * 2) : make_uint4(0, 0, 0, 0);
            }
            const float* sp = v8.s + v8_si(sk, kh);
            sn0 = ok ? ld_ntf(sp) : make_float4(0, 0, 0, 0);
            sn1 = ok ? ld_ntf(sp + 4) : make_float4(0, 0, 0, 0);
            const int8_t* qb8 = v8.q + v8_qi(sk, kh, l16);
#pragma unroll
            for (int j = 0; j < 8; ++j) vn[j] = ok ? ld_nt2(qb8 + j * 256) : make_uint2(0, 0);
        };
        if (a0 + warp * 16 < b0) fetch(a0 + warp * 16);
        for (int t0 = a0 + warp * 16; t0 < b0; t0 += 8 * 16) {
            uint4 kc[8];
            uint2 vq[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { kc[i] = kn[i]; vq[i] = vn[i]; }
            float vs[8] = {sn0.x, sn0.y, sn0.z, sn0.w, sn1.x, sn1.y, sn1.z, sn1.w};
            float ksc[8] = {kn0.x, kn0.y, kn0.z, kn0.w, kn1.x, kn1.y, kn1.z, kn1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) if (t0 + g * 8 + i >= b0) vs[i] = 0.f;   // (past the chunk: never-written slots)
            if (t0 + 8 * 16 < b0) fetch(t0 + 8 * 16);
            pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
            const bool here = fused && t0 + l16 == pos;
#pragma unroll
            for (int s8 = 0; s8 < 8; ++s8) {
                pf_v8h ka;
                if (v8.kq) {
                    const int8_t* b8 = (const int8_t*) &kc[s8 >> 1] + (s8 & 1) * 8;
#pragma unroll
                    for (int i = 0; i < 8; ++i) ka[i] = (_Float16) b8[i];
                } else ka = __builtin_bit_cast(pf_v8h, kc[s8]);
                if (here)
#pragma unroll
                    for (int i = 0; i < 8; ++i) ka[i] = (_Float16) c2f(f2c<kc_t>(kf[s8 * 16 + g * 8 + i]));
                sc = pf_wmma(ka, qb[s8], sc);
            }
            // (fused: the position's own k and v are the exact ones - the 8-bit copies may not be written yet)
            const int self = fused ? pos - (t0 + g * 8) : -1;
            if (v8.kq)
#pragma unroll
                for (int i = 0; i < 8; ++i) sc[i] *= i == self ? 1.f : ksc[i];
            float cf;
            const pf_v8h pb = soft(sc, t0, cf);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                o[j] = o[j] * cf;
                const int8_t* b8 = (const int8_t*) &vq[j];
                pf_v8h va;
#pragma unroll
                for (int i = 0; i < 8; ++i) va[i] = (_Float16) (float(b8[i]) * vs[i]);
                if (self >= 0 && self < 8) va[self] = (_Float16) c2f(f2c<vc_t>(vf[j * 16 + l16]));
                o[j] = pf_wmma(va, pb, o[j]);
            }
        }
    } else
    for (int t0 = a0 + warp * 16; t0 < b0; t0 += 8 * 16) {
        {
        // V tile: lane takes key t0 + l16, dims g * 64 .. + 63, into vt transposed
            const int p = t0 + l16;
            const bool ok = p < b0;
            const vc_t* vr = V + size_t(p & kv_mask) * kKvDim + kh * kHd + g * 64;
            if (fused && p == pos) {
                float a8 = 0;
                if (v8.q) for (int i = 0; i < kHd; ++i) a8 = fmaxf(a8, fabsf(vf[i]));
                const float d8 = a8 > 0.f ? a8 / 127.f : 1.f;
                for (int d = 0; d < 64; ++d) {
                    const float x = vf[g * 64 + d];
                    static constexpr bool kSelfRound = FLINT_SELFROUND;
                    vt[warp][g * 64 + d][l16] = (_Float16) (v8.q && kSelfRound ? max(-127.f, min(127.f, rintf(x / d8))) * d8 : c2f(f2c<vc_t>(x)));
                }
            } else if constexpr (sizeof(vc_t) == 2) {
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
            const bool here = fused && p == pos;
#pragma unroll
            for (int s8 = 0; s8 < 8; ++s8) {
                pf_v8h ka;
                if (here)
#pragma unroll
                    for (int i = 0; i < 8; ++i) ka[i] = (_Float16) c2f(f2c<kc_t>(kf[s8 * 16 + g * 8 + i]));
                else if constexpr (sizeof(kc_t) == 2) ka = ok ? __builtin_bit_cast(pf_v8h, *(const uint4*) (kr + s8 * 16 + g * 8)) : pf_v8h{};
                else
#pragma unroll
                    for (int i = 0; i < 8; ++i) ka[i] = (_Float16) (ok ? c2f(kr[s8 * 16 + g * 8 + i]) : 0.f);
                sc = pf_wmma(ka, qb[s8], sc);
            }
        }
        float cf;
        const pf_v8h pb = soft(sc, t0, cf);
        fl_wave_barrier();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j] = o[j] * cf;
            const pf_v8h va = *(const pf_v8h*) &vt[warp][j * 16 + l16][g * 8];
            o[j] = pf_wmma(va, pb, o[j]);
        }
        fl_wave_barrier();
    }
    l += fl_shfl_xor(l, 16, 32);
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

// The checking pass's attention on the full layers (8-bit V): the 8 query heads of up to 2 * NCG tokens as the WMMA
// columns (column cg * 16 + l16: token col / 8, head kh * 8 + col % 8), so the history is read once for all of them
// instead of once a token. grid (kKv, kAttnChunks, ceil(T / (2 * NCG))), 256 threads; part as k_attn_dec_w's (merged by
// k_attn_merge per token). The tokens' own K/V are already in the cache (k_qkv_post).
template <int NCG>
__global__ void __launch_bounds__(256) k_attn_dec_m(const float* __restrict__ q, float* __restrict__ part,
                                                    const int* __restrict__ state, int kv_mask, int zs, int T,
                                                    const kc_t* __restrict__ K, V8 v8) {
    constexpr int NC = 16 * NCG, NZ = 2 * NCG;
    __shared__ __align__(16) _Float16 qs[NC][kHd + 8];
    __shared__ __align__(16) float wo[8][8][kHd];
    __shared__ float wm[8][NC], wl[8][NC];
    const int kh = blockIdx.x, c = blockIdx.y, z0 = blockIdx.z * NZ, nz = min(NZ, T - z0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int pos0 = state[0] + z0, n = pos0 + nz;   // keys 0 .. n - 1 (token z sees up to pos0 + z)
    int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks);
    per = (per + 15) & ~15;
    const int a0 = c * per, b0 = min(n, a0 + per);
    const float scale = rsqrtf(float(kHd));
    for (int e = tid; e < NC * kHd; e += 256) {
        const int col = e / kHd, d = e % kHd, zl = col >> 3;
        qs[col][d] = (_Float16) (zl < nz ? q[size_t(z0 + zl) * zs + (kh * 8 + (col & 7)) * kHd + d] * scale : 0.f);
    }
    __syncthreads();
    float m[NCG], l[NCG];
    int zlast[NCG];   // the last key this lane's column may see (-1: no token)
    pf_v8f o[NCG][8];
#pragma unroll
    for (int cg = 0; cg < NCG; ++cg) {
        m[cg] = -INFINITY; l[cg] = 0;
        const int zl = (cg * 16 + l16) >> 3;
        zlast[cg] = zl < nz ? pos0 + zl : -1;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[cg][j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    for (int t0 = a0 + warp * 16; t0 < b0; t0 += 8 * 16) {
        const int p = t0 + l16, pk = t0 + g * 8;
        const bool okk = p < b0, ok = pk < b0;
        const size_t sk = size_t(pk & kv_mask);
        // V^T operand bytes and scales (lane: dims j * 16 + l16, keys pk ..), K scales (8-bit K)
        uint2 vq[8];
        const int8_t* qb8 = v8.q + v8_qi(sk, kh, l16);
#pragma unroll
        for (int j = 0; j < 8; ++j) vq[j] = ok ? *(const uint2*) (qb8 + j * 256) : make_uint2(0, 0);
        float vs[8], ksc[8];
        {
            const float* sp = v8.s + v8_si(sk, kh);
            const float4 s0 = ok ? *(const float4*) sp : make_float4(0, 0, 0, 0), s1 = ok ? *(const float4*) (sp + 4) : make_float4(0, 0, 0, 0);
            vs[0] = s0.x; vs[1] = s0.y; vs[2] = s0.z; vs[3] = s0.w; vs[4] = s1.x; vs[5] = s1.y; vs[6] = s1.z; vs[7] = s1.w;
            if (v8.kq) {
                const float* kp = v8.ks + v8_si(sk, kh);
                const float4 k0 = ok ? *(const float4*) kp : make_float4(0, 0, 0, 0), k1 = ok ? *(const float4*) (kp + 4) : make_float4(0, 0, 0, 0);
                ksc[0] = k0.x; ksc[1] = k0.y; ksc[2] = k0.z; ksc[3] = k0.w; ksc[4] = k1.x; ksc[5] = k1.y; ksc[6] = k1.z; ksc[7] = k1.w;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) if (pk + i >= b0) vs[i] = 0.f;
        // S^T for the columns: A = K rows (lane: key p, dims s * 16 + g * 8 ..)
        pf_v8f sc[NCG];
#pragma unroll
        for (int cg = 0; cg < NCG; ++cg) sc[cg] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
        if (v8.kq) {
            const uint4* kr = (const uint4*) (v8.kq + k8_row(size_t(p & kv_mask), kh) + g * 64);
#pragma unroll
            for (int s4 = 0; s4 < 4; ++s4) {
                const uint4 u = okk ? kr[s4] : make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const int s8 = s4 * 2 + h2;
                    const int8_t* b8 = (const int8_t*) &u + h2 * 8;
                    pf_v8h ka;
#pragma unroll
                    for (int i = 0; i < 8; ++i) ka[i] = (_Float16) b8[i];
#pragma unroll
                    for (int cg = 0; cg < NCG; ++cg) sc[cg] = pf_wmma(ka, *(const pf_v8h*) &qs[cg * 16 + l16][s8 * 16 + g * 8], sc[cg]);
                }
            }
#pragma unroll
            for (int cg = 0; cg < NCG; ++cg)
#pragma unroll
                for (int i = 0; i < 8; ++i) sc[cg][i] *= ksc[i];
        } else {
            const uint4* kr = (const uint4*) ((const char*) K + (size_t(p & kv_mask) * kKvDim + kh * kHd + g * 8) * 2);
#pragma unroll
            for (int s8 = 0; s8 < 8; ++s8) {
                const pf_v8h ka = okk ? __builtin_bit_cast(pf_v8h, kr[s8 * 2]) : pf_v8h{};
#pragma unroll
                for (int cg = 0; cg < NCG; ++cg) sc[cg] = pf_wmma(ka, *(const pf_v8h*) &qs[cg * 16 + l16][s8 * 16 + g * 8], sc[cg]);
            }
        }
        // per column: running max / sum and P (keys pk + i)
        pf_v8h pb[NCG];
        float cf[NCG];
#pragma unroll
        for (int cg = 0; cg < NCG; ++cg) {
            float mx = -INFINITY, v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                v[i] = (pk + i < b0 && pk + i <= zlast[cg]) ? sc[cg][i] : -INFINITY;
                mx = fmaxf(mx, v[i]);
            }
            mx = fmaxf(mx, fl_shfl_xor(mx, 16, 32));
            const float mn = fmaxf(m[cg], mx), ref = mn == -INFINITY ? 0.f : mn;
            cf[cg] = expf(m[cg] - ref);
            m[cg] = mn;
            float ps = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { const float e = expf(v[i] - ref); ps += e; pb[cg][i] = (_Float16) e; }
            l[cg] = l[cg] * cf[cg] + ps;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int8_t* b8 = (const int8_t*) &vq[j];
            pf_v8h va;
#pragma unroll
            for (int i = 0; i < 8; ++i) va[i] = (_Float16) (float(b8[i]) * vs[i]);
#pragma unroll
            for (int cg = 0; cg < NCG; ++cg) o[cg][j] = pf_wmma(va, pb[cg], o[cg][j] * cf[cg]);
        }
    }
#pragma unroll
    for (int cg = 0; cg < NCG; ++cg) {
        l[cg] += fl_shfl_xor(l[cg], 16, 32);
        if (g == 0) { wm[warp][cg * 16 + l16] = m[cg]; wl[warp][cg * 16 + l16] = l[cg]; }
    }
    // merge the 8 warps a token at a time: lane (column l16 of group cg, half g) has dims j * 16 + g * 8 + i
#pragma unroll
    for (int zl = 0; zl < NZ; ++zl) {
        if (zl >= nz) break;
        __syncthreads();
        if ((l16 >> 3) == (zl & 1)) {
#pragma unroll
            for (int cg = 0; cg < NCG; ++cg)
                if (cg == (zl >> 1))
#pragma unroll
                    for (int j = 0; j < 8; ++j)
#pragma unroll
                        for (int i = 0; i < 8; ++i) wo[warp][l16 & 7][j * 16 + g * 8 + i] = o[cg][j][i];
        }
        __syncthreads();
        const int hh = tid >> 5, d0 = (tid & 31) * 4, col = zl * 8 + hh;
        float M = -INFINITY;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w][col]);
        float Ls = 0, O[4] = {};
        for (int w = 0; w < 8; ++w) {
            if (wm[w][col] == -INFINITY) continue;
            const float f = expf(wm[w][col] - M);
            Ls += wl[w][col] * f;
            for (int i = 0; i < 4; ++i) O[i] += wo[w][hh][d0 + i] * f;
        }
        float* P = part + size_t(z0 + zl) * kHeads * kAttnChunks * kPart + (size_t(kh * (kHeads / kKv) + hh) * kAttnChunks + c) * kPart;
        if (d0 == 0) { P[0] = M; P[1] = Ls; }
        for (int i = 0; i < 4; ++i) P[4 + d0 + i] = O[i];
    }
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
        const float ov = fl_shfl_xor(v, o, 32);
        const int oi = fl_shfl_xor(i, o, 32);
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
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, fl_shfl_xor(mx, o, 32));
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
    // (two 4-byte loads: a block is 20 bytes, so B + 4 is only 4-aligned - AMD reads it as 8, NVIDIA faults)
    const uint2 qq = make_uint2(*(const uint32_t*) (B + 4 + (j0 & 15)), *(const uint32_t*) (B + 8 + (j0 & 15)));
    const int sh = j0 >= 16 ? 4 : 0;
    const uint32_t u0 = (qq.x >> sh) & 0x0F0F0F0Fu, u1 = (qq.y >> sh) & 0x0F0F0F0Fu;
    // bytes q0 q1 q2 q3 -> halves (1024 + q0, 1024 + q1), (1024 + q2, 1024 + q3)
    const uint32_t c = 0x64646464u;
    const uint32_t h0 = fl_perm(c, u0, 0x07010700u), h1 = fl_perm(c, u0, 0x07030702u);
    const uint32_t h2 = fl_perm(c, u1, 0x07010700u), h3 = fl_perm(c, u1, 0x07030702u);
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
#if FL_HIP
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
#endif
#if !FL_HIP
// NVIDIA: the same product with mma.sync m16n8k16 in its native layout. A = weights, unpacked from Q4_1 straight into
// the fragment (lane (gid, tig): rows gid and gid + 8, k pairs 2 tig and 2 tig + 8), B = the tokens' x from LDS, the
// 16 outputs x 64 tokens of a row tile as 8 products of 8 tokens. Weights to fp16 as d * q + m, as the gfx12 kernel does.
__device__ __forceinline__ uint32_t nv_q41_pair(const uint8_t* blk, int byte, int sh, __half2 d2, __half2 m2) {
    const uint16_t q = *(const uint16_t*) (blk + 4 + byte);
    const __half2 v = __halves2half2(__ushort2half_rn((q >> sh) & 15), __ushort2half_rn((q >> (8 + sh)) & 15));
    const __half2 w = __hfma2(v, d2, m2);
    uint32_t r;
    std::memcpy(&r, &w, 4);
    return r;
}
// rows R0 (gid) and R1 (gid + 8), k0 .. k0 + 15: the four A registers
__device__ __forceinline__ void nv_q41_afrag(const uint8_t* R0, const uint8_t* R1, int k0, int tig, uint32_t* a) {
    const int sh = (k0 & 16) ? 4 : 0;
    const uint8_t* b0 = R0 + (k0 >> 5) * 20;
    const uint8_t* b1 = R1 + (k0 >> 5) * 20;
    __half2 d0, m0, d1, m1;
    {
        const uint32_t dm0 = *(const uint32_t*) b0, dm1 = *(const uint32_t*) b1;
        __half h;
        uint16_t u;
        u = uint16_t(dm0); std::memcpy(&h, &u, 2); d0 = __half2half2(h);
        u = uint16_t(dm0 >> 16); std::memcpy(&h, &u, 2); m0 = __half2half2(h);
        u = uint16_t(dm1); std::memcpy(&h, &u, 2); d1 = __half2half2(h);
        u = uint16_t(dm1 >> 16); std::memcpy(&h, &u, 2); m1 = __half2half2(h);
    }
    a[0] = nv_q41_pair(b0, 2 * tig, sh, d0, m0);
    a[1] = nv_q41_pair(b1, 2 * tig, sh, d1, m1);
    a[2] = nv_q41_pair(b0, 2 * tig + 8, sh, d0, m0);
    a[3] = nv_q41_pair(b1, 2 * tig + 8, sh, d1, m1);
}
__device__ __forceinline__ void nv_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool LO>
__global__ void __launch_bounds__(256) k_gemm_q41(const uint8_t* __restrict__ W, int N, int K, const uint16_t* __restrict__ xh,
                                                  const uint16_t* __restrict__ xl, const float* __restrict__ xs, int T,
                                                  float* __restrict__ Y, int ldy, int add) {
    constexpr int TA = 2, BN = 64, NT = BN / 8, BK = 64, LD = BK + 8;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int o0 = blockIdx.x * 256 + warp * TA * 16, t0 = blockIdx.y * BN;
    const size_t rowb = size_t(K / 32) * 20;
    const uint8_t* wr[TA][2];
#pragma unroll
    for (int a = 0; a < TA; ++a) {
        wr[a][0] = W + size_t(min(o0 + a * 16 + gid, N - 1)) * rowb;
        wr[a][1] = W + size_t(min(o0 + a * 16 + gid + 8, N - 1)) * rowb;
    }
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
    float acc[TA][NT][4];
#pragma unroll
    for (int a = 0; a < TA; ++a)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
    uint4 rb[4];
    fetch(0, rb);
    put(rb, 0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) fetch(k0 + BK, rb);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            uint32_t fa[TA][4];
#pragma unroll
            for (int a = 0; a < TA; ++a) nv_q41_afrag(wr[a][0], wr[a][1], k0 + kk * 16, tig, fa[a]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const uint32_t bh0 = *(const uint32_t*) &sx[buf][0][n * 8 + gid][kk * 16 + 2 * tig];
                const uint32_t bh1 = *(const uint32_t*) &sx[buf][0][n * 8 + gid][kk * 16 + 2 * tig + 8];
#pragma unroll
                for (int a = 0; a < TA; ++a) nv_mma(acc[a][n], fa[a], bh0, bh1);
                if (LO) {
                    const uint32_t bl0 = *(const uint32_t*) &sx[buf][1][n * 8 + gid][kk * 16 + 2 * tig];
                    const uint32_t bl1 = *(const uint32_t*) &sx[buf][1][n * 8 + gid][kk * 16 + 2 * tig + 8];
#pragma unroll
                    for (int a = 0; a < TA; ++a) nv_mma(acc[a][n], fa[a], bl0, bl1);
                }
            }
        }
        if (k0 + BK < K) put(rb, buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    // lane: outputs o0 + a * 16 + gid (+ 8), tokens t0 + n * 8 + 2 tig (+ 1)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = t0 + n * 8 + 2 * tig + j;
            if (t >= T) continue;
            const float sc = xs[t];
#pragma unroll
            for (int a = 0; a < TA; ++a)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int o = o0 + a * 16 + gid + 8 * r;
                    if (o >= N) continue;
                    float* y = Y + size_t(t) * ldy + o;
                    const float v = acc[a][n][2 * r + j] * sc;
                    *y = add ? *y + v : v;
                }
        }
}
#endif


// The routed experts' grouped product: rows of the grouped order (expert e's rows bounds[e] .. bounds[e + 1]) in tiles of
// 64 (tiles: (expert, first row) pairs, *n_tiles of them). Input row of grouped row r: src[r] (or r without src).
// GU: gate (W1) and up (W2) both, Y = silu(gate) * up; otherwise Y = W1 . x. W_e = W + e * mat_bytes.
#if FL_HIP
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
#endif
#if !FL_HIP
// NVIDIA's grouped product (see k_gemm_q41 above for the fragment layout)
template <bool LO, bool GU>
__global__ void __launch_bounds__(256) k_gemm_q41_grp(const uint8_t* __restrict__ W1, const uint8_t* __restrict__ W2, size_t mat_bytes,
                                                      int N, int K, const uint16_t* __restrict__ xh, const uint16_t* __restrict__ xl,
                                                      const float* __restrict__ xs, const int* __restrict__ src,
                                                      const int* __restrict__ bounds, const int* __restrict__ tiles,
                                                      const int* __restrict__ n_tiles, float* __restrict__ Y, int ldy) {
    constexpr int TA = 2, BN = 64, NT = BN / 8, BK = 64, LD = BK + 8, NM = GU ? 2 : 1;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];
    __shared__ int srow[BN];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int e = tiles[2 * tile], r0 = bounds[e] + tiles[2 * tile + 1], rn = min(BN, bounds[e + 1] - r0);
    const int o0 = blockIdx.x * 256 + warp * TA * 16;
    if (tid < BN) srow[tid] = tid < rn ? (src ? src[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const size_t rowb = size_t(K / 32) * 20;
    const uint8_t* wr[NM][TA][2];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int r = 0; r < 2; ++r) wr[mm][a][r] = (mm ? W2 : W1) + size_t(e) * mat_bytes + size_t(o0 + a * 16 + gid + 8 * r) * rowb;
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
    float acc[NM][TA][NT][4];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[mm][a][n][0] = acc[mm][a][n][1] = acc[mm][a][n][2] = acc[mm][a][n][3] = 0.f;
    uint4 rb[4];
    fetch(0, rb);
    put(rb, 0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) fetch(k0 + BK, rb);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            uint32_t fa[NM][TA][4];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) nv_q41_afrag(wr[mm][a][0], wr[mm][a][1], k0 + kk * 16, tig, fa[mm][a]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const uint32_t bh0 = *(const uint32_t*) &sx[buf][0][n * 8 + gid][kk * 16 + 2 * tig];
                const uint32_t bh1 = *(const uint32_t*) &sx[buf][0][n * 8 + gid][kk * 16 + 2 * tig + 8];
#pragma unroll
                for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                    for (int a = 0; a < TA; ++a) nv_mma(acc[mm][a][n], fa[mm][a], bh0, bh1);
                if (LO) {
                    const uint32_t bl0 = *(const uint32_t*) &sx[buf][1][n * 8 + gid][kk * 16 + 2 * tig];
                    const uint32_t bl1 = *(const uint32_t*) &sx[buf][1][n * 8 + gid][kk * 16 + 2 * tig + 8];
#pragma unroll
                    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                        for (int a = 0; a < TA; ++a) nv_mma(acc[mm][a][n], fa[mm][a], bl0, bl1);
                }
            }
        }
        if (k0 + BK < K) put(rb, buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = n * 8 + 2 * tig + j;
            if (t >= rn) continue;
            const float sc = xs[srow[t]];
#pragma unroll
            for (int a = 0; a < TA; ++a)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const float g0 = acc[0][a][n][2 * r + j] * sc;
                    const float v = GU ? silu(g0) * (acc[GU ? 1 : 0][a][n][2 * r + j] * sc) : g0;
                    Y[size_t(r0 + t) * ldy + o0 + a * 16 + gid + 8 * r] = v;
                }
        }
}
#endif

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
// The prompt path's router on the matrix cores: logits (T x 256) = W . x with W the fp16 router copy (exact, scaled by
// 2^we) and x the split rows already made for the shared expert (xh + xl, times xs: the normed input). Block: 16 tokens;
// warp w: experts w * 32 .. + 31 (two tiles of 16), A = router rows, B = the tokens' x^T.
__global__ void __launch_bounds__(256) k_pf_router_w(const uint16_t* __restrict__ xh, const uint16_t* __restrict__ xl,
                                                     const float* __restrict__ xs, const __half* __restrict__ W, int we,
                                                     int T, float* __restrict__ logits) {
    const int t0 = blockIdx.x * 16, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, l16 = lane & 15, g = lane >> 4;
    const int t = t0 + l16;
    const bool live = t < T;
    const uint16_t* bh = xh + size_t(min(t, T - 1)) * kHidden + g * 8;
    const uint16_t* bl = xl + size_t(min(t, T - 1)) * kHidden + g * 8;
    const __half* a0 = W + size_t(warp * 32 + l16) * kHidden + g * 8;
    const __half* a1 = a0 + size_t(16) * kHidden;
    pf_v8f c0 = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0}, c1 = c0;
#pragma unroll 4
    for (int k0 = 0; k0 < kHidden; k0 += 16) {
        const pf_v8h xb = live ? __builtin_bit_cast(pf_v8h, *(const uint4*) (bh + k0)) : pf_v8h{};
        const pf_v8h xc = live ? __builtin_bit_cast(pf_v8h, *(const uint4*) (bl + k0)) : pf_v8h{};
        const pf_v8h w0 = __builtin_bit_cast(pf_v8h, *(const uint4*) (a0 + k0)), w1 = __builtin_bit_cast(pf_v8h, *(const uint4*) (a1 + k0));
        c0 = pf_wmma(w0, xb, c0); c0 = pf_wmma(w0, xc, c0);
        c1 = pf_wmma(w1, xb, c1); c1 = pf_wmma(w1, xc, c1);
    }
    if (!live) return;
    // lane: token t, experts warp * 32 + j * 16 + g * 8 + i
    const float sc = ldexpf(xs[t], -we);
    float* o = logits + size_t(t) * kExperts + warp * 32 + g * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) { o[i] = c0[i] * sc; o[16 + i] = c1[i] * sc; }
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
            const float ov = fl_shfl_xor(v, o, 32);
            const int oi = fl_shfl_xor(i, o, 32);
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
__global__ void k_pf_qkpost(float* QKV, const float* qn, const float* kn, kc_t* K, vc_t* V, int p0, bool rope, int kv_mask, V8 v8 = {}) {
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
    if (isq || v8.kq) src[i] = y;   // (8-bit K: the chunk's own keys are read exact from here)
    if (!isq) {
        const int kh = hb - kHeads;
        const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd + i;
        const float vv = row[kQDim + kKvDim + kh * kHd + i];
        float ak = 0, av = 0;
        if (q8sim && !rope) {
            __shared__ float mk[4], mv[4];
            float a = fabsf(y), b = fabsf(vv);
            for (int o = 16; o > 0; o >>= 1) { a = fmaxf(a, fl_shfl_xor(a, o, 32)); b = fmaxf(b, fl_shfl_xor(b, o, 32)); }
            if ((i & 31) == 0) { mk[i >> 5] = a; mv[i >> 5] = b; }
            __syncthreads();
            if (q8grp == 32) { ak = mk[i >> 5]; av = mv[i >> 5]; }
            else { ak = fmaxf(fmaxf(mk[0], mk[1]), fmaxf(mk[2], mk[3])); av = fmaxf(fmaxf(mv[0], mv[1]), fmaxf(mv[2], mv[3])); }
        }
        if (v8.kq) {
            __shared__ float n8[4];
            float a = fabsf(y);
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, fl_shfl_xor(a, o, 32));
            if ((i & 31) == 0) n8[i >> 5] = a;
            __syncthreads();
            k8_put(v8, size_t(pos & kv_mask), kh, i, y, fmaxf(fmaxf(n8[0], n8[1]), fmaxf(n8[2], n8[3])));
        } else K[r] = f2c<kc_t>(q8rt(y, ak, 1));
        if (v8.q) {
            __shared__ float m8[4];
            float a = fabsf(vv);
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, fl_shfl_xor(a, o, 32));
            if ((i & 31) == 0) m8[i >> 5] = a;
            __syncthreads();
            v8_put(v8, size_t(pos & kv_mask), kh, i, vv, fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])));
        } else V[r] = f2c<vc_t>(q8rt(vv, av, 2));
    }
}


// Attention for T queries on the matrix units: block = (K/V head, 16 queries), warp w = query head kh * 8 + w, so the 8
// heads sharing a K/V head read each key tile once. Per 16 keys a warp forms S^T = K Q^T (keys x queries: its C layout
// gives each lane one query's 8 keys, which is already the B operand of the next product), takes the online softmax,
// and adds O^T += V^T P^T (dims x queries). K and V go through LDS as fp16 (V transposed), Q and P are fp16, sums fp32.
__global__ void __launch_bounds__(256) k_pf_attn_w(const float* __restrict__ QKV, const kc_t* __restrict__ K,
                                                   const vc_t* __restrict__ V, const float* __restrict__ sinks,
                                                   float* __restrict__ ATT, int p0, int T, int swa, int kv_mask, V8 v8 = {}) {
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
            // (8-bit V: this chunk's own keys use their exact values from QKV - rounding a key's own value in its
            // attention cost KL 0.02 - and only older positions come from the 8-bit store)
            const bool own = v8.q && p >= p0;
            const float* vraw = own ? QKV + size_t(p - p0) * (kQDim + 2 * kKvDim) + kQDim + kKvDim + kh * kHd + c0 : nullptr;
            const float d8 = v8.q && ok && !own ? v8.s[v8_si(size_t(p & kv_mask), kh)] : 0.f;
            const int8_t* qr = v8.q && !own ? v8.q + v8_qi(size_t(p & kv_mask), kh, c0) : nullptr;
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                ks[key][c0 + c] = (_Float16) (ok ? c2f(kr[c]) : 0.f);
                vt[c0 + c][key] = (_Float16) (!ok ? 0.f : own ? vraw[c] : qr ? float(qr[c * 16]) * d8 : c2f(vr[c]));
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
            mx = fmaxf(mx, fl_shfl_xor(mx, 16, 32));
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
    l += fl_shfl_xor(l, 16, 32);
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
                                 float (*acc)[T], const float* rs = nullptr, const float* nw = nullptr) {
    const int qd = lane & 3, bl = lane >> 2;
    for (int kc = 0; kc < nb * 32; kc += kVSl) {
        const int kn = min(kVSl, nb * 32 - kc);
        for (int i = threadIdx.x; i < nt * (kn / 4); i += 256) {
            const int t = i / (kn / 4), c = (i % (kn / 4)) * 4;
            float4 v = *(const float4*) (xp[t] + kc + c);
            if (nw) {
                const float4 w = *(const float4*) (nw + kc + c);
                v = make_float4(v.x * rs[t] * w.x, v.y * rs[t] * w.y, v.z * rs[t] * w.z, v.w * rs[t] * w.w);
            }
            *(float4*) (sb + t * kVSl + c) = v;
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
    const float* nw;   // set: the inputs are rmsnorm(X) * nw (computed here)
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
    __shared__ float rs[T];
    if (J.nw) {   // each token's 1 / rms (warp t: token t)
        const int w_ = threadIdx.x >> 5;
        for (int t = w_; t < T; t += 8) {
            float ss = 0;
            for (int i = lane; i < J.n_in; i += 32) { const float x = J.X[size_t(t) * J.ldx + i]; ss += x * x; }
            ss = warp_sum(ss);
            if (lane == 0) rs[t] = rsqrtf(ss / J.n_in + kEps);
        }
        __syncthreads();
    }
    q4_rows_p<kMvRpw, T>(wr, nb, xp, T, sb, lane, acc, rs, J.nw);
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
                const float ov = fl_shfl_xor(v, o, 32);
                const int oi = fl_shfl_xor(i, o, 32);
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
        const float ov = fl_shfl_xor(v, o, 32);
        const int oi = fl_shfl_xor(i, o, 32);
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
                const float ov = fl_shfl_xor(v, o, 32);
                const int oi = fl_shfl_xor(i, o, 32);
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


// ---------------------------------------------------------------------------------------------------- one launch a token
// The whole writing step in one kernel of kMegaTot blocks (what fits the card at once: 4 a WGP at <= 192 registers).
// Blocks 0 .. G - 1 work: they walk the 49 layers phase by phase with a grid barrier between phases (about 0.6 us,
// against about 2 us a kernel launch). The rest prefetch: as the workers start layer i, they pick layer i's likely
// experts from its input (the router on the layer's input agrees with the real pick 84-87% of the time) and read them,
// then layer i + 1's fixed weights, into the memory-side cache. Each row is still one warp's dot product as in the
// separate kernels, so results match them (up to the router's summation order).
constexpr int kMegaTot = 128;
struct MegaLayer {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *sinks;
    const uint16_t* router;
    const uint8_t *wqkv, *wo, *gate, *up, *down, *gate_e, *up_e, *down_e, *gate_s, *up_s, *down_s;
    kc_t* K;
    vc_t* V;
    int swa, kv_mask;
};
struct MegaArgs {
    const MegaLayer* L;
    const float *embed, *out_norm;
    const uint8_t* head;
    int vocab, G;
    int* state;
    float *h, *qkv, *att, *part, *rl, *act, *logits, *plog;
    float2* best;
    unsigned* ctr;   // [0] workers' barrier, [1] progress (layer + 1), [2] argmax done, [3] prefetchers' barrier
};
__device__ inline void mega_bar(unsigned* cnt, unsigned& gen, unsigned n) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned target = (gen + 1) * n;
        atomicAdd(cnt, 1u);
        while (__hip_atomic_load(cnt, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) fl_sleep(1);
        __threadfence();
    }
    ++gen;
    __syncthreads();
}
// R rows given by their own pointers (GGUF Q4_1 rows of nb blocks) against xs (LDS or memory); warp sums in out
template <int R>
__device__ inline void q4_rows_ptr(const uint8_t* const* W, int nb, const float* xs, int lane, float* out) {
    const int qd = lane & 3;
    float acc[R] = {};
    for (int b = lane >> 2; b < nb; b += 8) {
        const float4 xa = *(const float4*) (xs + b * 32 + qd * 4);
        const float4 xb = *(const float4*) (xs + b * 32 + 16 + qd * 4);
        const float sx = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* B = W[r] + size_t(b) * 20;
            const uint32_t dm = *(const uint32_t*) B, q = *(const uint32_t*) (B + 4 + qd * 4);
            const float d = __half2float(__ushort_as_half(uint16_t(dm))), m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
            const float t = float(q & 15) * xa.x + float((q >> 8) & 15) * xa.y + float((q >> 16) & 15) * xa.z + float((q >> 24) & 15) * xa.w
                          + float((q >> 4) & 15) * xb.x + float((q >> 12) & 15) * xb.y + float((q >> 20) & 15) * xb.z + float(q >> 28) * xb.w;
            acc[r] += d * t + m * sx;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) out[r] = warp_sum(acc[r]);
}
__device__ inline void mega_touch(const void* p, size_t bytes, int b, int nb, uint32_t& a) {
    const uint4* q = (const uint4*) p;
    for (size_t i = size_t(b) * 256 + threadIdx.x; i < bytes / 16; i += size_t(nb) * 256) { const uint4 v = q[i]; a ^= v.x ^ v.w; }
}
// top 6 of 256 logits (lg in LDS) by sigmoid score, one warp; lane k < 6 gets the k-th index and normalized weight
__device__ inline void mega_top6(const float* lg, int lane, int& oi, float& ow) {
    float sc[8];
    for (int j = 0; j < 8; ++j) sc[j] = 1.f / (1.f + expf(-lg[lane * 8 + j]));
    float pv = 0, tot = 0;
    int pi = 0;
    for (int k = 0; k < kTopK; ++k) {
        float v = -2.f;
        int i = 0;
        for (int j = 0; j < 8; ++j) if (sc[j] > v) { v = sc[j]; i = lane * 8 + j; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = fl_shfl_xor(v, o, 32);
            const int oi_ = fl_shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi_ < i)) { v = ov; i = oi_; }
        }
        if (lane == i / 8) sc[i % 8] = -1.f;
        if (lane == k) { pv = v; pi = i; }
        tot += v;
    }
    oi = pi; ow = pv / tot;
}
__device__ inline void mega_router_row(const uint16_t* W, int r, const float* xs, int lane, float* out) {
    float acc = 0;
    for (int k = lane * 8; k < kHidden; k += 256) {
        const uint4 w = *(const uint4*) (W + size_t(r) * kHidden + k);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) acc += __uint_as_float(ww[j] << 16) * xs[k + 2 * j] + __uint_as_float(ww[j] & 0xffff0000u) * xs[k + 2 * j + 1];
    }
    acc = warp_sum(acc);
    if (lane == 0) out[r] = acc;
}
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8))) k_token(MegaArgs A) {
    extern __shared__ float xs[];
    __shared__ __align__(16) float qf[kHd], kf[kHd], vf[kHd];
    __shared__ int sidx[kTopK];
    __shared__ float swts[kTopK];
    __shared__ float lg256[kExperts];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = A.G, NW = G * 8;
    constexpr size_t r2k = size_t(kHidden / 32) * 20;
    const int pos = A.state[0];
    unsigned gen = 0;
    if (int(blockIdx.x) >= G) {
        // ------------------------------------------------ prefetchers
        const int P = kMegaTot - G, b = blockIdx.x - G;
        uint32_t a = 0;
        __shared__ int at;
        for (int i = 0; i < kLayers; ++i) {
            if (tid == 0) {
                unsigned p;
                while ((p = __hip_atomic_load(A.ctr + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < unsigned(i + 1)) fl_sleep(2);
                at = int(p) - 1;   // the layer the workers are on
            }
            __syncthreads();
            if (at > i) continue;   // fallen behind: never hold the workers' step up, skip to where they are
            const MegaLayer& l = A.L[i];
            if (i >= kDense && false) {
                // the router on rmsnorm(h) * ffn_norm as the workers start the layer (a guess: h is still changing)
                stage_in(xs, A.h, l.ffn_norm, nullptr, kHidden);
                for (int r = b * 8 + warp; r < kExperts; r += P * 8) mega_router_row(l.router, r, xs, lane, A.plog);
                mega_bar(A.ctr + 3, gen, P);
                if (warp == 0) {
                    for (int j = 0; j < 8; ++j) lg256[lane * 8 + j] = ((volatile float*) A.plog)[lane * 8 + j];
                    int oi; float ow;
                    mega_top6(lg256, lane, oi, ow);
                    if (lane < kTopK) sidx[lane] = oi;
                }
                __syncthreads();
                constexpr size_t mb = size_t(kFfExp) * r2k;
                for (int k = 0; k < kTopK; ++k) {
                    const size_t e = size_t(sidx[k]);
                    mega_touch(l.gate_e + e * mb, mb, b, P, a);
                    mega_touch(l.up_e + e * mb, mb, b, P, a);
                    mega_touch(l.down_e + e * mb, mb, b, P, a);
                }
                mega_bar(A.ctr + 3, gen, P);   // (plog is rewritten next layer only after every prefetcher read it)
            }
            if (i + 1 < kLayers) {
                const MegaLayer& n = A.L[i + 1];
                mega_touch(n.wqkv, size_t(kQDim + 2 * kKvDim) * r2k, b, P, a);
                mega_touch(n.wo, size_t(kHidden) * (kQDim / 32) * 20, b, P, a);
                if (i + 1 < kDense) {
                    mega_touch(n.gate, kFfDense * r2k, b, P, a); mega_touch(n.up, kFfDense * r2k, b, P, a);
                    mega_touch(n.down, size_t(kHidden) * (kFfDense / 32) * 20, b, P, a);
                } else {
                    mega_touch(n.router, size_t(kExperts) * kHidden * 2, b, P, a);
                    mega_touch(n.gate_s, kFfShared * r2k, b, P, a); mega_touch(n.up_s, kFfShared * r2k, b, P, a);
                    mega_touch(n.down_s, size_t(kHidden) * (kFfShared / 32) * 20, b, P, a);
                }
            }
        }
        if (a == 0x9e3779b9u) A.state[2] = int(a);
        return;
    }
    // ------------------------------------------------ workers
    const int gw = blockIdx.x * 8 + warp;
    const int tok = A.state[1];
    float* h = A.h;
    for (int i = blockIdx.x * 256 + tid; i < kHidden; i += G * 256) h[i] = A.embed[size_t(tok) * kHidden + i];
    mega_bar(A.ctr, gen, G);
    for (int il = 0; il < kLayers; ++il) {
        const MegaLayer& l = A.L[il];
        if (blockIdx.x == 0 && tid == 0) __hip_atomic_store(A.ctr + 1, unsigned(il + 1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        // A: q/k/v rows (5120), 6 a warp unit, on rmsnorm(h) * attn_norm
        stage_in(xs, h, l.attn_norm, nullptr, kHidden);
        for (int u = gw; u * 6 < kQDim + 2 * kKvDim; u += NW) {
            const uint8_t* W[6];
            for (int r = 0; r < 6; ++r) W[r] = l.wqkv + size_t(min(u * 6 + r, kQDim + 2 * kKvDim - 1)) * r2k;
            float o[6];
            q4_rows_ptr<6>(W, kHidden / 32, xs, lane, o);
            if (lane == 0) for (int r = 0; r < 6 && u * 6 + r < kQDim + 2 * kKvDim; ++r) A.qkv[u * 6 + r] = o[r];
        }
        mega_bar(A.ctr, gen, G);
        // B: attention partials, items (head, chunk); q / this position's k and v normed here; the chunk-0 item of a
        // K/V head's first query head writes the cache
        {
            const int lo = l.swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
            const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks);
            const int nch = (n + per - 1) / per;
            float4 (*so)[32] = (float4 (*)[32]) xs;
            float* sm = xs + 8 * 32 * 4;
            float* sl = sm + 8;
            for (int it = blockIdx.x; it < kHeads * kAttnChunks; it += G) {
                const int hh = it / kAttnChunks, c = it % kAttnChunks, kh = hh / (kHeads / kKv);
                float* P = A.part + (size_t(hh) * kAttnChunks + c) * kPart;
                if (c >= nch) { if (tid == 0) { P[0] = -INFINITY; P[1] = 0; } continue; }
                if (tid < kHd) { qf[tid] = A.qkv[hh * kHd + tid]; kf[tid] = A.qkv[kQDim + kh * kHd + tid]; vf[tid] = A.qkv[kQDim + kKvDim + kh * kHd + tid]; }
                __syncthreads();
                if (warp == 0) head_norm_warp(qf, l.q_norm, pos, l.swa, lane);
                else if (warp == 1) head_norm_warp(kf, l.k_norm, pos, l.swa, lane);
                __syncthreads();
                if (c == 0 && hh % (kHeads / kKv) == 0 && tid < kHd) {
                    const size_t r = size_t(pos & l.kv_mask) * kKvDim + kh * kHd + tid;
                    l.K[r] = f2c<kc_t>(kf[tid]);
                    l.V[r] = f2c<vc_t>(vf[tid]);
                }
                const int a0 = lo + c * per, b0 = min(pos + 1, a0 + per);
                const float4 qv = *(const float4*) (qf + lane * 4);
                const float scale = rsqrtf(float(kHd));
                float m = -INFINITY, sum = 0;
                float4 o = make_float4(0, 0, 0, 0);
                for (int p = a0 + warp; p < b0; p += 8) {
                    const bool here = p == pos;
                    const float4 kk = here ? make_float4(c2f(f2c<kc_t>(kf[lane * 4])), c2f(f2c<kc_t>(kf[lane * 4 + 1])), c2f(f2c<kc_t>(kf[lane * 4 + 2])), c2f(f2c<kc_t>(kf[lane * 4 + 3])))
                                           : h4(l.K + size_t(p & l.kv_mask) * kKvDim + kh * kHd + lane * 4);
                    const float d = warp_sum(qv.x * kk.x + qv.y * kk.y + qv.z * kk.z + qv.w * kk.w) * scale;
                    const float mn = fmaxf(m, d), cf = expf(m - mn), e = expf(d - mn);
                    const float4 vv = here ? make_float4(c2f(f2c<vc_t>(vf[lane * 4])), c2f(f2c<vc_t>(vf[lane * 4 + 1])), c2f(f2c<vc_t>(vf[lane * 4 + 2])), c2f(f2c<vc_t>(vf[lane * 4 + 3])))
                                           : h4(l.V + size_t(p & l.kv_mask) * kKvDim + kh * kHd + lane * 4);
                    o.x = o.x * cf + e * vv.x; o.y = o.y * cf + e * vv.y; o.z = o.z * cf + e * vv.z; o.w = o.w * cf + e * vv.w;
                    sum = sum * cf + e;
                    m = mn;
                }
                if (lane == 0) { sm[warp] = m; sl[warp] = sum; }
                so[warp][lane] = o;
                __syncthreads();
                if (warp == 0) {
                    float M = -INFINITY;
                    for (int i = 0; i < 8; ++i) M = fmaxf(M, sm[i]);
                    float Ls = 0;
                    float4 O = make_float4(0, 0, 0, 0);
                    for (int i = 0; i < 8; ++i) {
                        if (sm[i] == -INFINITY) continue;
                        const float f = expf(sm[i] - M);
                        Ls += sl[i] * f;
                        O.x += so[i][lane].x * f; O.y += so[i][lane].y * f; O.z += so[i][lane].z * f; O.w += so[i][lane].w * f;
                    }
                    if (lane == 0) { P[0] = M; P[1] = Ls; }
                    *(float4*) (P + 4 + lane * 4) = O;
                }
                __syncthreads();
            }
        }
        mega_bar(A.ctr, gen, G);
        // C: the chunks merged with the sink, a head a block
        for (int hh = blockIdx.x; hh < kHeads; hh += G) {
            if (tid >= kHd) continue;
            const float* P = A.part + size_t(hh) * kAttnChunks * kPart;
            float M = l.sinks[hh];
            for (int c = 0; c < kAttnChunks; ++c) M = fmaxf(M, P[c * kPart]);
            float Ls = expf(l.sinks[hh] - M), O = 0;
            for (int c = 0; c < kAttnChunks; ++c) {
                const float mc = P[c * kPart];
                if (mc == -INFINITY) continue;
                const float f = expf(mc - M);
                Ls += P[c * kPart + 1] * f;
                O += P[c * kPart + 4 + tid] * f;
            }
            A.att[hh * kHd + tid] = O / Ls;
        }
        mega_bar(A.ctr, gen, G);
        // D: o rows (2048), 3 a warp unit, added into h
        stage_in(xs, A.att, nullptr, nullptr, kQDim);
        for (int u = gw; u * 3 < kHidden; u += NW) {
            const uint8_t* W[3];
            for (int r = 0; r < 3; ++r) W[r] = l.wo + size_t(min(u * 3 + r, kHidden - 1)) * (kQDim / 32) * 20;
            float o[3];
            q4_rows_ptr<3>(W, kQDim / 32, xs, lane, o);
            if (lane == 0) for (int r = 0; r < 3 && u * 3 + r < kHidden; ++r) h[u * 3 + r] += o[r];
        }
        mega_bar(A.ctr, gen, G);
        stage_in(xs, h, l.ffn_norm, nullptr, kHidden);
        if (il < kDense) {
            // dense gate/up (6144 each), 7 a warp unit: act = silu(gate) * up
            for (int u = gw; u * 7 < kFfDense; u += NW) {
                const uint8_t* W[14];
                for (int r = 0; r < 7; ++r) { const int row = min(u * 7 + r, kFfDense - 1); W[r] = l.gate + size_t(row) * r2k; W[7 + r] = l.up + size_t(row) * r2k; }
                float o[14];
                q4_rows_ptr<14>(W, kHidden / 32, xs, lane, o);
                if (lane == 0) for (int r = 0; r < 7 && u * 7 + r < kFfDense; ++r) A.act[u * 7 + r] = silu(o[r]) * o[7 + r];
            }
            mega_bar(A.ctr, gen, G);
            for (int u = gw; u * 3 < kHidden; u += NW) {   // (6144 inputs read from memory: LDS stays small)
                const uint8_t* W[3];
                for (int r = 0; r < 3; ++r) W[r] = l.down + size_t(min(u * 3 + r, kHidden - 1)) * (kFfDense / 32) * 20;
                float o[3];
                q4_rows_ptr<3>(W, kFfDense / 32, A.act, lane, o);
                if (lane == 0) for (int r = 0; r < 3 && u * 3 + r < kHidden; ++r) h[u * 3 + r] += o[r];
            }
            mega_bar(A.ctr, gen, G);
            continue;
        }
        // E: router rows (256: one a warp) on the staged rmsnorm(h)
        for (int r = gw; r < kExperts; r += NW) mega_router_row(l.router, r, xs, lane, A.rl);
        mega_bar(A.ctr, gen, G);
        // F: each block picks the top 6 (warp 0), then gate/up rows of the 6 experts and the shared one (4096 act rows,
        // 5 a warp unit; a row's expert from its index), on the still-staged x
        if (warp == 0) {
            for (int j = 0; j < 8; ++j) lg256[lane * 8 + j] = A.rl[lane * 8 + j];
            int oi; float ow;
            mega_top6(lg256, lane, oi, ow);
            if (lane < kTopK) { sidx[lane] = oi; swts[lane] = ow; }
        }
        __syncthreads();
        for (int u = gw; u * 5 < kActRows; u += NW) {
            const uint8_t* W[10];
            for (int r = 0; r < 5; ++r) {
                const int row = min(u * 5 + r, kActRows - 1);
                if (row < kTopK * kFfExp) {
                    const size_t e = size_t(sidx[row / kFfExp]) * kFfExp + row % kFfExp;
                    W[r] = l.gate_e + e * r2k; W[5 + r] = l.up_e + e * r2k;
                } else { W[r] = l.gate_s + size_t(row - kTopK * kFfExp) * r2k; W[5 + r] = l.up_s + size_t(row - kTopK * kFfExp) * r2k; }
            }
            float o[10];
            q4_rows_ptr<10>(W, kHidden / 32, xs, lane, o);
            if (lane == 0) for (int r = 0; r < 5 && u * 5 + r < kActRows; ++r) A.act[u * 5 + r] = silu(o[r]) * o[5 + r];
        }
        mega_bar(A.ctr, gen, G);
        // G: down rows (2048), 3 a warp unit: the 6 experts weighted plus the shared one, added into h
        stage_in(xs, A.act, nullptr, nullptr, kActRows);
        {
            constexpr size_t rowe = size_t(kFfExp / 32) * 20, rows_ = size_t(kFfShared / 32) * 20;
            for (int u = gw; u * 3 < kHidden; u += NW) {
                float tot[3] = {};
                for (int k = 0; k < kTopK; ++k) {
                    const uint8_t* W[3];
                    for (int r = 0; r < 3; ++r) W[r] = l.down_e + (size_t(sidx[k]) * kHidden + min(u * 3 + r, kHidden - 1)) * rowe;
                    float o[3];
                    q4_rows_ptr<3>(W, kFfExp / 32, xs + k * kFfExp, lane, o);
                    for (int r = 0; r < 3; ++r) tot[r] += swts[k] * o[r];
                }
                const uint8_t* W[3];
                for (int r = 0; r < 3; ++r) W[r] = l.down_s + size_t(min(u * 3 + r, kHidden - 1)) * rows_;
                float o[3];
                q4_rows_ptr<3>(W, kFfShared / 32, xs + kTopK * kFfExp, lane, o);
                if (lane == 0) for (int r = 0; r < 3 && u * 3 + r < kHidden; ++r) h[u * 3 + r] += tot[r] + o[r];
            }
        }
        mega_bar(A.ctr, gen, G);
    }
    // the head, 4 rows a warp step, and this block's best; the last block picks the token
    stage_in(xs, h, A.out_norm, nullptr, kHidden);
    float bv = -INFINITY;
    int bi = 0;
    for (int r0 = gw * 4; r0 < A.vocab; r0 += NW * 4) {
        const uint8_t* W[4];
        for (int r = 0; r < 4; ++r) W[r] = A.head + size_t(r0 + r) * r2k;
        float o[4];
        q4_rows_ptr<4>(W, kHidden / 32, xs, lane, o);
        if (lane == 0)
            for (int r = 0; r < 4; ++r) {
                A.logits[r0 + r] = o[r];
                if (o[r] > bv) { bv = o[r]; bi = r0 + r; }
            }
    }
    __shared__ float wbv[8];
    __shared__ int wbi[8];
    __shared__ bool last;
    if (lane == 0) { wbv[warp] = bv; wbi[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 8; ++w) if (wbv[w] > bv || (wbv[w] == bv && wbi[w] < bi)) { bv = wbv[w]; bi = wbi[w]; }
        A.best[blockIdx.x] = make_float2(bv, __int_as_float(bi));
        __threadfence();
        last = atomicAdd(A.ctr + 2, 1u) == unsigned(G) - 1;
    }
    __syncthreads();
    if (!last || tid != 0) return;
    __threadfence();
    float v = -INFINITY;
    int i = 0;
    for (int b = 0; b < G; ++b) {
        const float cv = ((volatile float*) A.best)[2 * b];
        const int ci = __float_as_int(((volatile float*) A.best)[2 * b + 1]);
        if (cv > v || (cv == v && ci < i)) { v = cv; i = ci; }
    }
    A.state[1] = i;
    A.state[0] = pos + 1;
}


// The prompt attention, second form: QT query tiles of 16 a warp (so each K/V tile in LDS serves 16 * QT queries of 8
// heads), keys in tiles of 32, double-buffered: the next tile's K and V are loaded into registers while the current
// one is used, then stored to the other LDS buffer. Same arithmetic per query as k_pf_attn_w.
template <int QT>
__global__ void __launch_bounds__(256) k_pf_attn_w2(const float* __restrict__ QKV, const kc_t* __restrict__ K,
                                                    const vc_t* __restrict__ V, const float* __restrict__ sinks,
                                                    float* __restrict__ ATT, int p0, int T, int swa, int kv_mask, V8 v8) {
    constexpr int KT = 32, NQ = 16 * QT, LDQ = kQDim + 2 * kKvDim;
    __shared__ __align__(16) _Float16 ks[2][KT][kHd + 8];
    __shared__ __align__(16) _Float16 vt[2][kHd][KT + 8];
    __shared__ float kss[2][KT];   // (8-bit K: the tile's key scales, applied to the scores)
    const int kh = blockIdx.x, q0 = blockIdx.y * NQ;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int h = kh * (kHeads / kKv) + warp;
    int qpos[QT], qlo[QT];
    bool live[QT];
    pf_v8h qb[QT][8];
#pragma unroll
    for (int j = 0; j < QT; ++j) {
        const int tq = q0 + j * 16 + l16;
        live[j] = tq < T;
        qpos[j] = p0 + min(tq, T - 1);
        qlo[j] = swa ? max(0, qpos[j] - (kWindow - 1)) : 0;
        const float* qr = QKV + size_t(min(tq, T - 1)) * LDQ + h * kHd;
        // (1 / sqrt(128) and log2(e) folded into q: the scores come out in the log2 domain, for the bare exp2)
        const float qs = rsqrtf(float(kHd)) * 1.4426950408889634f;
#pragma unroll
        for (int s8 = 0; s8 < 8; ++s8)
#pragma unroll
            for (int i = 0; i < 8; ++i) qb[j][s8][i] = (_Float16) (qr[s8 * 16 + g * 8 + i] * qs);
    }
    // the tiles every query of this block sees whole (no mask needed): keys [all_lo, all_hi]
    const int all_hi = p0 + q0, all_lo = swa ? p0 + min(q0 + NQ - 1, T - 1) - (kWindow - 1) : 0;
    float m[QT], l[QT];
    pf_v8f o[QT][8];
#pragma unroll
    for (int j = 0; j < QT; ++j) {
        m[j] = -INFINITY; l[j] = 0;
#pragma unroll
        for (int d = 0; d < 8; ++d) o[j][d] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    const int last = p0 + min(q0 + NQ - 1, T - 1);
    const int first = swa ? max(0, p0 + q0 - (kWindow - 1)) : 0;
    // a thread's share of a tile: key lk, dims lc .. lc + 15
    const int lk = tid >> 3, lc = (tid & 7) * 16;
    const int vd = tid >> 1, vh = tid & 1;   // (8-bit V: dim vd, keys vh * 16 .. + 15)
    uint4 rk[2];
    uint32_t rv[8];
    float rks = 0;
    auto load = [&](int k0) {
        const int p = k0 + lk;
        const bool ok = p <= last;
        const kc_t* kr = K + size_t(p & kv_mask) * kKvDim + kh * kHd + lc;
        if (v8.kq && ok && p >= p0) {   // this chunk's own keys: exact, in the same two runs of 8 dims
            const int d0 = ((lc & 63) >> 3) * 16 + (lc >> 6) * 8;
            const float* kx = QKV + size_t(p - p0) * LDQ + kQDim + kh * kHd + d0;
            pf_v8h a, b;
            for (int i = 0; i < 8; ++i) { a[i] = (_Float16) kx[i]; b[i] = (_Float16) kx[16 + i]; }
            rk[0] = __builtin_bit_cast(uint4, a); rk[1] = __builtin_bit_cast(uint4, b);
            rks = 1.f;
        } else if (v8.kq) {   // stored positions lc .. + 15: dims s * 16 + h * 8 + i and (s + 1) * 16 + h * 8 + i
            const uint4 u = ok ? *(const uint4*) (v8.kq + k8_row(size_t(p & kv_mask), kh) + lc) : make_uint4(0, 0, 0, 0);
            const int8_t* b8 = (const int8_t*) &u;
            pf_v8h a, b;
            for (int i = 0; i < 8; ++i) { a[i] = (_Float16) b8[i]; b[i] = (_Float16) b8[8 + i]; }
            rk[0] = __builtin_bit_cast(uint4, a); rk[1] = __builtin_bit_cast(uint4, b);
            rks = ok ? v8.ks[v8_si(size_t(p & kv_mask), kh)] : 0.f;
        } else if constexpr (sizeof(kc_t) == 2) {
            rk[0] = ok ? *(const uint4*) kr : make_uint4(0, 0, 0, 0);
            rk[1] = ok ? *(const uint4*) (kr + 8) : make_uint4(0, 0, 0, 0);
        } else {
            pf_v8h a, b;
            for (int i = 0; i < 8; ++i) { a[i] = (_Float16) (ok ? c2f(kr[i]) : 0.f); b[i] = (_Float16) (ok ? c2f(kr[8 + i]) : 0.f); }
            rk[0] = __builtin_bit_cast(uint4, a); rk[1] = __builtin_bit_cast(uint4, b);
        }
        float vv[16];
        if (v8.q) {   // 8-bit V: dim vd, keys k0 + vh * 16 .. + 15 (one group: k0 steps by 32 from 0 on the full layers)
            const int pg = k0 + vh * 16;
            const size_t sg = size_t(pg & kv_mask);
            if (pg + 15 < p0 && pg + 15 <= last) {
                const uint4 u = *(const uint4*) (v8.q + v8_qi(sg, kh, vd));
                const int8_t* b = (const int8_t*) &u;
                const float* sp = v8.s + v8_si(sg, kh);
                for (int i = 0; i < 16; i += 4) {
                    const float4 f = *(const float4*) (sp + i);
                    vv[i] = float(b[i]) * f.x; vv[i + 1] = float(b[i + 1]) * f.y; vv[i + 2] = float(b[i + 2]) * f.z; vv[i + 3] = float(b[i + 3]) * f.w;
                }
            } else   // (this chunk's own keys: exact values from QKV)
                for (int i = 0; i < 16; ++i) {
                    const int q = pg + i;
                    vv[i] = q > last ? 0.f : q >= p0 ? QKV[size_t(q - p0) * LDQ + kQDim + kKvDim + kh * kHd + vd]
                                                     : float(v8.q[v8_qi(sg + i, kh, vd)]) * v8.s[v8_si(sg + i, kh)];
                }
        } else if (!ok) for (int i = 0; i < 16; ++i) vv[i] = 0.f;
        else {
            const vc_t* vr = V + size_t(p & kv_mask) * kKvDim + kh * kHd + lc;
            for (int i = 0; i < 16; ++i) vv[i] = c2f(vr[i]);
        }
        for (int i = 0; i < 8; ++i) rv[i] = __builtin_bit_cast(uint32_t, pf_v2h{(_Float16) vv[2 * i], (_Float16) vv[2 * i + 1]});
    };
    auto store = [&](int buf) {
        if (v8.kq) {
            const int d0 = ((lc & 63) >> 3) * 16 + (lc >> 6) * 8;
            *(uint4*) &ks[buf][lk][d0] = rk[0];
            *(uint4*) &ks[buf][lk][d0 + 16] = rk[1];
            if (lc == 0) kss[buf][lk] = rks;
        } else {
            *(uint4*) &ks[buf][lk][lc] = rk[0];
            *(uint4*) &ks[buf][lk][lc + 8] = rk[1];
        }
        if (v8.q) {
            *(uint4*) &vt[buf][vd][vh * 16] = make_uint4(rv[0], rv[1], rv[2], rv[3]);
            *(uint4*) &vt[buf][vd][vh * 16 + 8] = make_uint4(rv[4], rv[5], rv[6], rv[7]);
        } else
            for (int i = 0; i < 8; ++i) {
                const pf_v2h v2 = __builtin_bit_cast(pf_v2h, rv[i]);
                vt[buf][lc + 2 * i][lk] = v2[0];
                vt[buf][lc + 2 * i + 1][lk] = v2[1];
            }
    };
    load(first);
    store(0);
    pf_sync();
    int buf = 0;
    for (int k0 = first; k0 <= last; k0 += KT) {
        const bool more = k0 + KT <= last;
        if (more) load(k0 + KT);
#pragma unroll
        for (int sub = 0; sub < KT / 16; ++sub) {
            pf_v8f sc[QT];
#pragma unroll
            for (int j = 0; j < QT; ++j) sc[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s8 = 0; s8 < 8; ++s8) {
                const pf_v8h ka = *(const pf_v8h*) &ks[buf][sub * 16 + l16][s8 * 16 + g * 8];
#pragma unroll
                for (int j = 0; j < QT; ++j) sc[j] = pf_wmma(ka, qb[j][s8], sc[j]);
            }
            pf_v8h pb[QT];
            float cf[QT];
            const int kb0 = k0 + sub * 16;
            if (v8.kq)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float f = kss[buf][sub * 16 + g * 8 + i];
#pragma unroll
                    for (int j = 0; j < QT; ++j) sc[j][i] *= f;
                }
            const bool whole = kb0 >= all_lo && kb0 + 15 <= all_hi;   // (masks only on the edge tiles)
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                float mx = -INFINITY, v[8];
                if (whole && live[j]) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { v[i] = sc[j][i]; mx = fmaxf(mx, v[i]); }
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int p = kb0 + g * 8 + i;
                        v[i] = (live[j] && p >= qlo[j] && p <= qpos[j]) ? sc[j][i] : -INFINITY;
                        mx = fmaxf(mx, v[i]);
                    }
                }
                mx = fmaxf(mx, fl_swap16(mx));
                const float mn = fmaxf(m[j], mx), ref = mn == -INFINITY ? 0.f : mn;
                cf[j] = fl_exp2(m[j] - ref);
                m[j] = mn;
                float ps = 0;
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    const float e0 = fl_exp2(v[i] - ref), e1 = fl_exp2(v[i + 1] - ref);
                    ps += e0 + e1;
                    w[i / 2] = __builtin_bit_cast(uint32_t, pf_v2h{(_Float16) e0, (_Float16) e1});   // (round to nearest: rtz cost KL)
                }
                pb[j] = __builtin_bit_cast(pf_v8h, make_uint4(w[0], w[1], w[2], w[3]));
                l[j] = l[j] * cf[j] + ps;
            }
#pragma unroll
            for (int j = 0; j < QT; ++j)
                if (fl_any(cf[j] != 1.f))   // (rare after the first tiles)
#pragma unroll
                    for (int d = 0; d < 8; ++d) o[j][d] = o[j][d] * cf[j];
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const pf_v8h va = *(const pf_v8h*) &vt[buf][d * 16 + l16][sub * 16 + g * 8];
#pragma unroll
                for (int j = 0; j < QT; ++j) o[j][d] = pf_wmma(va, pb[j], o[j][d]);
            }
        }
        if (more) store(buf ^ 1);
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int j = 0; j < QT; ++j) {
        const float lt = l[j] + fl_shfl_xor(l[j], 16, 32);
        if (!live[j]) continue;
        const float sk = sinks[h] * 1.4426950408889634f, M = fmaxf(m[j], sk);   // (log2 domain)
        const float f = exp2f(m[j] - M), L = lt * f + exp2f(sk - M);
        float* out = ATT + size_t(q0 + j * 16 + l16) * kQDim + h * kHd;
#pragma unroll
        for (int d = 0; d < 8; ++d)
#pragma unroll
            for (int i = 0; i < 8; ++i) out[d * 16 + g * 8 + i] = o[j][d][i] * f / L;
    }
}

// ---------------------------------------------------------------------------------------------------- the model
struct Layer {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *sinks;
    const uint16_t* router = nullptr;   // bf16
    const uint8_t *wq, *wk, *wv, *wo;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;              // dense (layers 0-1)
    const uint8_t *gate_e = nullptr, *up_e = nullptr, *down_e = nullptr;        // experts (hybrid: the card's, packed)
    // hybrid (FLINT_PLAN): every expert in host memory (the GGUF mapping), which the card holds (slot, -1: the CPU)
    const uint8_t *gate_h = nullptr, *up_h = nullptr, *down_h = nullptr;
    int* slot = nullptr;
    std::vector<int> slot_h;
    std::vector<float> ffn_norm_h;
    const uint8_t *gate_s = nullptr, *up_s = nullptr, *down_s = nullptr;        // shared expert
    bool swa = false;
    kc_t* K = nullptr;   // [slot][K/V head][128]
    vc_t* V = nullptr;
    int kv_mask = 0x7fffffff;              // slot = position & kv_mask (a ring on the sliding layers)
    const __half* router16 = nullptr;      // the router as fp16, times 2^router_e (every bf16 value exact): prompt path
    int router_e = 0;
    V8 v8{};                               // the full layers' 8-bit V (then V is unused)
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
    MegaLayer* mega_layers = nullptr;
    unsigned* mega_ctr = nullptr;
    float *mega_plog = nullptr, *mega_qkv = nullptr;
    float2* mega_best = nullptr;
    int mega_g = 0;   // worker blocks (0: the separate kernels)
    int* pidx = nullptr;
    float *pw = nullptr, *prl = nullptr;
    unsigned* pcnt = nullptr;
    unsigned* pdone = nullptr;
    float *act, *part, *h, *q, *k, *v, *att, *f1, *f2, *logits, *rl, *wts;
    float2* best;
    unsigned *done, *done2;
    int* idx;
    // hybrid mode (FLINT_PLAN=FILE, lines "layer expert expert ..."): only the listed experts on the card, the rest
    // computed by the CPU from the GGUF mapping while the card computes its own; no step graph, no drafts
    bool hybrid = false;
    std::vector<std::vector<int>> plan;
    int* ridx = nullptr;          // the chosen experts' card slots
    HyShared* hy_s = nullptr;     // host memory (mapped): the card's and the CPU's meeting place; hy_s_dev the card's view
    HyShared* hy_s_dev = nullptr;
    uint32_t* hy_ctr = nullptr;   // on the card: the layers posted so far (hy_seq: the CPU's count)
    uint32_t hy_seq = 0;
    uint8_t *stg_g = nullptr, *stg_u = nullptr, *stg_d = nullptr;                 // prompt path: a layer's experts
    uint8_t* hy_arena = nullptr;  // every routed expert, host side (large pages; pinned unless FLINT_HY_PIN=0)
    // prompt path: the experts copied on their own stream, overlapping the card's work - gate/up for the next layer
    // while this one's down (and the next attention) runs, down while gate/up runs; the events keep each buffer's
    // copy and its use apart
    cudaStream_t hy_cp = nullptr;
    cudaEvent_t ev_gu_ready = nullptr, ev_gu_free = nullptr, ev_d_ready = nullptr, ev_d_free = nullptr;
    void hy_copy_gu(const Layer& l) {
        const size_t lb = size_t(kExperts) * kFfExp * (kHidden / 32) * 20;
        CK(cudaStreamWaitEvent(hy_cp, ev_gu_free, 0));
        CK(cudaMemcpyAsync(stg_g, l.gate_h, lb, cudaMemcpyHostToDevice, hy_cp));
        CK(cudaMemcpyAsync(stg_u, l.up_h, lb, cudaMemcpyHostToDevice, hy_cp));
        CK(cudaEventRecord(ev_gu_ready, hy_cp));
    }
    void hy_copy_d(const Layer& l) {
        const size_t lb = size_t(kExperts) * kFfExp * (kHidden / 32) * 20;
        CK(cudaStreamWaitEvent(hy_cp, ev_d_free, 0));
        CK(cudaMemcpyAsync(stg_d, l.down_h, lb, cudaMemcpyHostToDevice, hy_cp));
        CK(cudaEventRecord(ev_d_ready, hy_cp));
    }
    size_t hy_arena_bytes = 0;
    size_t hy_cpu_experts = 0, hy_card_experts = 0;   // (counted over the writing steps)
    double hy_t_wait = 0, hy_t_cpu = 0;               // (seconds: waiting for the card's hand-over, the CPU's share)
    int pos = 0;
    int* state;          // on the card: [0] the position, [1] the token going in next
    int* host_state;     // pinned copy
    cudaGraphExec_t graph = nullptr;

    int last_router_e = 0;   // (up_tensor's last router: its fp16 copy's power-of-two scale)
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
            // and as fp16 in the allocation's other half, scaled by a power of two so the largest is near 2^14 (each
            // bf16 value then lands exactly): the prompt path's matrix-core router
            float mx = 0.f;
            for (size_t i = 0; i < v.size(); ++i) mx = std::max(mx, std::fabs(f[i]));
            last_router_e = mx > 0.f ? 14 - std::ilogb(mx) : 0;
            std::vector<__half> hv(v.size());
            for (size_t i = 0; i < v.size(); ++i) hv[i] = __float2half(std::ldexp(f[i], last_router_e));
            CK(cudaMemcpy((char*) d + v.size() * 2, hv.data(), v.size() * 2, cudaMemcpyHostToDevice));
            return d;
        }
        CK(cudaMemcpy(d, g->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
        loaded += bytes;
        return d;
    }

    Model(const std::string& path, int ctx) : max_ctx(ctx) {
        if (std::getenv("FLINT_Q8SIM")) {
            const int bits = std::atoi(std::getenv("FLINT_Q8SIM")), grp = std::getenv("FLINT_Q8GRP") ? std::atoi(std::getenv("FLINT_Q8GRP")) : 128;
            CK(cudaMemcpyToSymbol(q8sim, &bits, 4));
            CK(cudaMemcpyToSymbol(q8grp, &grp, 4));
        }
        g = std::make_unique<strata::GgufFile>(path);
        if (const char* pf = std::getenv("FLINT_PLAN")) {
            FILE* f = std::fopen(pf, "r");
            if (!f) { std::fprintf(stderr, "strata-flint: cannot read the plan %s\n", pf); std::exit(1); }
            plan.assign(kLayers, {});
            char line[1 << 16];
            while (std::fgets(line, sizeof line, f)) {
                std::stringstream ss(line);
                int li;
                if (!(ss >> li) || li < 0 || li >= kLayers) continue;
                for (int e; ss >> e;) if (e >= 0 && e < kExperts) plan[li].push_back(e);
            }
            std::fclose(f);
            hybrid = true;
            flint_cpu::init(std::getenv("FLINT_CPU_THREADS") ? std::atoi(std::getenv("FLINT_CPU_THREADS")) : 0);
        }
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
            {   // q, k and v rows in one buffer (the one-launch step reads them as 5120 rows)
                const size_t r2k = size_t(kHidden / 32) * 20;
                uint8_t* qkvw = nullptr;
                CK(cudaMalloc(&qkvw, size_t(kQDim + 2 * kKvDim) * r2k));
                const char* nm[3] = {"attn_q.weight", "attn_k.weight", "attn_v.weight"};
                size_t off = 0;
                for (int j = 0; j < 3; ++j) {
                    const strata::TensorInfo* t = g->find(b + nm[j]);
                    if (!t || int(t->type) != kQ4_1) { std::fprintf(stderr, "strata-flint: %s%s missing or not Q4_1\n", b.c_str(), nm[j]); std::exit(1); }
                    const size_t bytes = strata::tensor_payload_bytes(*t);
                    CK(cudaMemcpy(qkvw + off, g->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
                    (j == 0 ? l.wq : j == 1 ? l.wk : l.wv) = qkvw + off;
                    off += bytes;
                    loaded += bytes;
                }
            }
            l.wo = (const uint8_t*) up_tensor(b + "attn_output.weight", kQ4_1);
            if (i < kDense) {
                l.gate = (const uint8_t*) up_tensor(b + "ffn_gate.weight", kQ4_1);
                l.up = (const uint8_t*) up_tensor(b + "ffn_up.weight", kQ4_1);
                l.down = (const uint8_t*) up_tensor(b + "ffn_down.weight", kQ4_1);
            } else {
                l.router = (const uint16_t*) up_tensor(b + "ffn_gate_inp.weight", kF32);
                l.router16 = (const __half*) (l.router + size_t(kExperts) * kHidden);
                l.router_e = last_router_e;
                if (!hybrid) {
                    l.gate_e = (const uint8_t*) up_tensor(b + "ffn_gate_exps.weight", kQ4_1);
                    l.up_e = (const uint8_t*) up_tensor(b + "ffn_up_exps.weight", kQ4_1);
                    l.down_e = (const uint8_t*) up_tensor(b + "ffn_down_exps.weight", kQ4_1);
                } else {   // the plan's experts packed on the card; all of them stay in the mapping for the CPU
                    auto host = [&](const char* nm) {
                        const strata::TensorInfo* t = g->find(b + nm);
                        if (!t || int(t->type) != kQ4_1) { std::fprintf(stderr, "strata-flint: %s%s missing or not Q4_1\n", b.c_str(), nm); std::exit(1); }
                        return (const uint8_t*) g->tensor_data(*t);
                    };
                    l.gate_h = host("ffn_gate_exps.weight"); l.up_h = host("ffn_up_exps.weight"); l.down_h = host("ffn_down_exps.weight");
                    {   // into the arena (large pages, pinned below): the CPU reads it faster than the file mapping, and
                        // the prompt path copies from it at full bus speed
                        const size_t lb = size_t(kExperts) * kFfExp * (kHidden / 32) * 20;
                        if (!hy_arena) {
                            hy_arena_bytes = size_t(kLayers - kDense) * 3 * lb;
                            void* a = mmap(nullptr, hy_arena_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                            if (a == MAP_FAILED) { std::fprintf(stderr, "strata-flint: no room for the %.1f GB expert arena\n", hy_arena_bytes / 1e9); std::exit(1); }
                            madvise(a, hy_arena_bytes, MADV_HUGEPAGE);
                            hy_arena = (uint8_t*) a;
                        }
                        uint8_t* dst = hy_arena + size_t(i - kDense) * 3 * lb;
                        const uint8_t* src[3] = {l.gate_h, l.up_h, l.down_h};
                        std::vector<std::thread> cp;
                        for (int t = 0; t < 6; ++t)
                            cp.emplace_back([&, t] {
                                for (int m = 0; m < 3; ++m) {
                                    const size_t a = lb * t / 6, z = lb * (t + 1) / 6;
                                    std::memcpy(dst + m * lb + a, src[m] + a, z - a);
                                }
                            });
                        for (auto& c : cp) c.join();
                        l.gate_h = dst; l.up_h = dst + lb; l.down_h = dst + 2 * lb;
                    }
                    const std::vector<int>& R = plan[i];
                    l.slot_h.assign(kExperts, -1);
                    for (size_t j = 0; j < R.size(); ++j) l.slot_h[R[j]] = int(j);
                    const size_t eb = size_t(kFfExp) * (kHidden / 32) * 20;   // (an expert's gate, up or down: same size)
                    uint8_t *cg = nullptr, *cu = nullptr, *cd = nullptr;
                    if (!R.empty()) {
                        CK(cudaMalloc(&cg, R.size() * eb)); CK(cudaMalloc(&cu, R.size() * eb)); CK(cudaMalloc(&cd, R.size() * eb));
                        for (size_t j = 0; j < R.size(); ++j) {
                            CK(cudaMemcpy(cg + j * eb, l.gate_h + size_t(R[j]) * eb, eb, cudaMemcpyHostToDevice));
                            CK(cudaMemcpy(cu + j * eb, l.up_h + size_t(R[j]) * eb, eb, cudaMemcpyHostToDevice));
                            CK(cudaMemcpy(cd + j * eb, l.down_h + size_t(R[j]) * eb, eb, cudaMemcpyHostToDevice));
                        }
                        loaded += 3 * R.size() * eb;
                    }
                    l.gate_e = cg; l.up_e = cu; l.down_e = cd;
                    CK(cudaMalloc(&l.slot, kExperts * 4));
                    CK(cudaMemcpy(l.slot, l.slot_h.data(), kExperts * 4, cudaMemcpyHostToDevice));
                    const strata::TensorInfo* tn = g->find(b + "ffn_norm.weight");
                    const float* nf = (const float*) g->tensor_data(*tn);
                    l.ffn_norm_h.assign(nf, nf + kHidden);
                }
                l.gate_s = (const uint8_t*) up_tensor(b + "ffn_gate_shexp.weight", kQ4_1);
                l.up_s = (const uint8_t*) up_tensor(b + "ffn_up_shexp.weight", kQ4_1);
                l.down_s = (const uint8_t*) up_tensor(b + "ffn_down_shexp.weight", kQ4_1);
            }
            // the sliding layers look back 513 positions: a ring of kSwaRing slots (a prompt chunk plus the window fits)
            static const int ring = std::getenv("FLINT_SWA_RING") ? std::atoi(std::getenv("FLINT_SWA_RING")) : kSwaRing;   // (a power of two)
            const size_t rows = l.swa && max_ctx > ring ? ring : max_ctx;
            if (l.swa && rows == size_t(ring)) l.kv_mask = ring - 1;
            static const bool v16 = std::getenv("FLINT_V16") != nullptr || !FL_HIP;   // (NVIDIA: fp16 V, the per-head attention)   // (fp16 V on the full layers too)
            static const bool k8 = std::getenv("FLINT_K8") != nullptr;      // (8-bit K too: about a third less to read)
            if (k8 && (v16 || sizeof(kc_t) != 2 || std::getenv("FLINT_ATTN_OLD") ||
                       (std::getenv("FLINT_PF_QT") && !std::strchr("124", std::getenv("FLINT_PF_QT")[0])))) {
                std::fprintf(stderr, "strata-flint: FLINT_K8 needs 8-bit V, fp16 K and the default attention kernels\n");
                std::exit(2);
            }
            if (l.swa || v16 || !k8) CK(cudaMalloc(&l.K, rows * kKvDim * sizeof(kc_t)));
            if (!l.swa && !v16) {
                const size_t rg = (rows + 15) & ~size_t(15);   // (whole groups of 16 positions)
                CK(cudaMalloc(&l.v8.q, rg * kKvDim));
                CK(cudaMalloc(&l.v8.s, rg * kKv * 4));
                kv_bytes += rg * kKvDim + rg * kKv * 4;
                l.V = nullptr;
                if (k8) {
                    CK(cudaMalloc(&l.v8.kq, rg * kKvDim));
                    CK(cudaMalloc(&l.v8.ks, rg * kKv * 4));
                    kv_bytes += rg * kKvDim + rg * kKv * 4;
                } else kv_bytes += rows * kKvDim * sizeof(kc_t);
            } else {
                CK(cudaMalloc(&l.V, rows * kKvDim * sizeof(vc_t)));
                kv_bytes += rows * kKvDim * (sizeof(kc_t) + sizeof(vc_t));
            }
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
        if (!hybrid) g.reset();   // (everything is on the card or in the pinned token table: the file's mapping goes)
        else {
            g.reset();   // (the experts are in the arena now)
            static const bool pin = !(std::getenv("FLINT_HY_PIN") && std::atoi(std::getenv("FLINT_HY_PIN")) == 0);
            if (pin && cudaHostRegister(hy_arena, hy_arena_bytes, 0) != cudaSuccess) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "strata-flint: could not pin the expert arena (prompts copy slower)\n");
            }   // the CPU's buffers, and the prompt path's staging for one layer's experts
            CK(cudaMalloc(&ridx, kTopK * 4));
            CK(cudaHostAlloc((void**) &hy_s, sizeof(HyShared), cudaHostAllocMapped));
            std::memset((void*) hy_s, 0, sizeof(HyShared));
            CK(cudaHostGetDevicePointer((void**) &hy_s_dev, hy_s, 0));
            CK(cudaMalloc(&hy_ctr, 4));
            CK(cudaMemset(hy_ctr, 0, 4));
            const size_t lb = size_t(kExperts) * kFfExp * (kHidden / 32) * 20;
            CK(cudaMalloc(&stg_g, lb)); CK(cudaMalloc(&stg_u, lb)); CK(cudaMalloc(&stg_d, lb));
            CK(cudaStreamCreateWithFlags(&hy_cp, cudaStreamNonBlocking));
            for (cudaEvent_t* e : {&ev_gu_ready, &ev_gu_free, &ev_d_ready, &ev_d_free}) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_gu_free, st));
            CK(cudaEventRecord(ev_d_free, st));
            size_t n = 0;
            for (const auto& r : plan) n += r.size();
            std::fprintf(stderr, "strata-flint: hybrid - %zu of %d experts on the card, the rest on %s\n", n, (kLayers - kDense) * kExperts,
                         "the CPU");
        }
        auto f = [&](float*& p, size_t n) { CK(cudaMalloc(&p, n * 4)); };
        f(h, kHidden); f(q, kQDim); f(k, kKvDim); f(v, kKvDim); f(att, kQDim); f(f1, kFfDense); f(f2, kFfDense);
        f(logits, vocab); f(rl, kExperts); f(wts, kTopK);
        CK(cudaMalloc(&idx, kTopK * 4));
        CK(cudaMalloc(&part, size_t(kHeads) * kAttnChunks * kPart * 4));
        CK(cudaMalloc(&act, kFfDense * 4));   // (the one-launch step's dense layers use 6144)
        CK(cudaMalloc(&done, 4));
        CK(cudaMemset(done, 0, 4));
        CK(cudaMalloc(&state, 8));
        CK(cudaMemset(state, 0, 8));
        CK(cudaMallocHost(&host_state, 8));
        CK(cudaMalloc(&pidx, kTopK * 4)); CK(cudaMalloc(&pw, kTopK * 4)); CK(cudaMalloc(&prl, kExperts * 4));
        CK(cudaMalloc(&pdone, 4)); CK(cudaMemset(pdone, 0, 4));
        if (std::getenv("FLINT_PRED")) { CK(cudaMalloc(&pcnt, 8)); CK(cudaMemset(pcnt, 0, 8)); }
        {   // the one-launch step (FLINT_MEGA=G worker blocks of kMegaTot; 0: off) when kMegaTot blocks fit at once
            const int want = std::getenv("FLINT_MEGA") ? std::atoi(std::getenv("FLINT_MEGA")) : 0;
            int dev = 0, per = 0;
            cudaDeviceProp pr;
            CK(cudaGetDevice(&dev));
            CK(cudaGetDeviceProperties(&pr, dev));
#if FL_HIP
            CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_token, 256, kActRows * 4));
            hipFuncAttributes fa;
            CK(hipFuncGetAttributes(&fa, (const void*) k_token));
#else
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_token, 256, kActRows * 4));
            cudaFuncAttributes fa;
            CK(cudaFuncGetAttributes(&fa, (const void*) k_token));
#endif
            if (want > 0 && want < kMegaTot && per * pr.multiProcessorCount >= kMegaTot && std::getenv("FLINT_V16")) mega_g = want;   // (fp16 V only)
            if (want > 0) std::fprintf(stderr, "strata-flint: one-launch step %s (%d workers, %d prefetchers; %d blocks fit a unit x %d, %d registers)\n",
                                       mega_g ? "on" : "off", mega_g, kMegaTot - mega_g, per, pr.multiProcessorCount, fa.numRegs);
            std::vector<MegaLayer> ml(kLayers);
            for (int i = 0; i < kLayers; ++i) {
                const Layer& l = L[i];
                ml[i] = MegaLayer{l.attn_norm, l.ffn_norm, l.q_norm, l.k_norm, l.sinks, l.router, l.wq, l.wo, l.gate, l.up,
                                  l.down, l.gate_e, l.up_e, l.down_e, l.gate_s, l.up_s, l.down_s, l.K, l.V, l.swa ? 1 : 0, l.kv_mask};
            }
            CK(cudaMalloc(&mega_layers, kLayers * sizeof(MegaLayer)));
            CK(cudaMemcpy(mega_layers, ml.data(), kLayers * sizeof(MegaLayer), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&mega_ctr, 16));
            CK(cudaMalloc(&mega_plog, kExperts * 4));
            CK(cudaMalloc(&mega_qkv, (kQDim + 2 * kKvDim) * 4));
            CK(cudaMalloc(&mega_best, kMegaTot * sizeof(float2)));
        }
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
    // the CPU's half of each MoE layer of a writing step: wait for the card's post, compute the experts it does not
    // hold, hand the sum back
    void hy_serve() {
        volatile HyShared* S = hy_s;
        const size_t eb = size_t(kFfExp) * (kHidden / 32) * 20;
        for (int i = kDense; i < n_layers(); ++i) {
            const Layer& l = L[i];
            const uint32_t sq = ++hy_seq;
            const auto w0 = std::chrono::steady_clock::now();
            for (long spin = 0; S->in_seq < sq; ++spin) {
                __builtin_ia32_pause();
                if ((spin & 0xfffff) == 0xfffff && std::chrono::steady_clock::now() - w0 > std::chrono::seconds(30)) {
                    std::fprintf(stderr, "strata-flint: hybrid - the card never posted layer %d (%s)\n", i, cudaGetErrorString(cudaStreamQuery(st)));
                    std::exit(1);
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const auto w1 = std::chrono::steady_clock::now();
            hy_t_wait += std::chrono::duration<double>(w1 - w0).count();
            flint_cpu::Expert ex[kTopK];
            float w[kTopK];
            int n = 0;
            for (int k = 0; k < kTopK; ++k) {
                const int e = S->idx[k];
                if (l.slot_h[e] >= 0) { ++hy_card_experts; continue; }
                ex[n] = {l.gate_h + size_t(e) * eb, l.up_h + size_t(e) * eb, l.down_h + size_t(e) * eb};
                w[n++] = S->w[k];
            }
            hy_cpu_experts += n;
            if (n) flint_cpu::moe((const float*) S->h, l.ffn_norm_h.data(), kHidden, kFfExp, n, ex, w, (float*) S->y);
            S->has = n > 0;
            hy_t_cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - w1).count();
            std::atomic_thread_fence(std::memory_order_release);
            S->out_seq = sq;
        }
    }
    void step(int tok) {
        if (pos >= max_ctx) { std::fprintf(stderr, "strata-flint: context full (%d)\n", max_ctx); std::exit(1); }
        if (tok >= 0) { feed({tok}); return; }
        static const bool no_graph = std::getenv("FLINT_NO_GRAPH") != nullptr;
        if (hybrid) {   // the step's graph runs while this thread serves its CPU halves, layer by layer
            if (!graph) {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                record();
                CK(cudaStreamEndCapture(st, &g));
                CK(cudaGraphInstantiate(&graph, g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
            }
            flint_cpu::set_active(true);
            CK(cudaGraphLaunch(graph, st));
            hy_serve();
            flint_cpu::set_active(false);
            ++pos;
            return;
        }
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
              const float* X, int ldx, int n_in, int T, int ldy, bool accum, const float* nw = nullptr) {
        MvV J{};
        int j = 0, total = 0;
        for (auto& m_ : mats) { J.w[j] = m_.first; J.y[j] = m_.second; ++j; }
        j = 0;
        for (int r : rows) { J.rows[j++] = r; total += r; }
        J.nmat = j; J.X = X; J.ldx = ldx; J.n_in = n_in; J.T = T; J.ldy = ldy; J.acc = accum; J.nw = nw;
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
            k_qkv_post<<<dim3(kHeads + kKv, 1, T), kHd, 0, st>>>(vQKV, vQKV + kQDim, vQKV + kQDim + kKvDim, l.q_norm, l.k_norm, l.K, l.V, state, l.swa, l.kv_mask, NQKV, l.v8);
            if (l.swa) k_attn_part<<<dim3(kHeads, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV);
            else if (l.v8.q && !std::getenv("FLINT_VATT_OLD")) {   // (the tokens share one read of the history)
                if (T <= 2) k_attn_dec_m<1><<<dim3(kKv, kAttnChunks, (T + 1) / 2), 256, 0, st>>>(vQKV, vPart, state, l.kv_mask, NQKV, T, l.K, l.v8);
                else k_attn_dec_m<2><<<dim3(kKv, kAttnChunks, (T + 3) / 4), 256, 0, st>>>(vQKV, vPart, state, l.kv_mask, NQKV, T, l.K, l.v8);
            } else k_attn_dec_w<<<dim3(kKv, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV, {}, l.v8);
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
        static const int pf = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : (FL_HIP ? 32 : 0);   // (NVIDIA: no 64 MB memory-side cache to fill)
        static const int pe = std::getenv("FLINT_PREDPF") ? std::atoi(std::getenv("FLINT_PREDPF")) : (FL_HIP ? 64 : 0);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            if ((pf || pe) && i + 1 < kLayers) {   // as on the writing path; the guess is the first token's
                CK(cudaEventRecord(ev[i], st));
                CK(cudaStreamWaitEvent(side, ev[i], 0));
                if (pe && i >= kDense) {
                    k_router<<<kExperts / 4, 128, 0, side>>>(vH, l.ffn_norm, l.router, prl, pidx, pw, pdone);
                    k_touch_pred<<<pe, 256, 0, side>>>(l.gate_e, l.up_e, l.down_e, pidx, sink);
                }
                if (pf) prefetch(i + 1);
            }
            v_mv({{l.wq, vQKV}, {l.wk, vQKV + kQDim}, {l.wv, vQKV + kQDim + kKvDim}}, {kQDim, kKvDim, kKvDim}, vH, kHidden, kHidden, T, NQKV, false, l.attn_norm);
            k_qkv_post<<<dim3(kHeads + kKv, 1, T), kHd, 0, st>>>(vQKV, vQKV + kQDim, vQKV + kQDim + kKvDim, l.q_norm, l.k_norm, l.K, l.V, state, l.swa, l.kv_mask, NQKV, l.v8);
            if (l.swa) k_attn_part<<<dim3(kHeads, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV);
            else if (l.v8.q && !std::getenv("FLINT_VATT_OLD")) {   // (the tokens share one read of the history)
                if (T <= 2) k_attn_dec_m<1><<<dim3(kKv, kAttnChunks, (T + 1) / 2), 256, 0, st>>>(vQKV, vPart, state, l.kv_mask, NQKV, T, l.K, l.v8);
                else k_attn_dec_m<2><<<dim3(kKv, kAttnChunks, (T + 3) / 4), 256, 0, st>>>(vQKV, vPart, state, l.kv_mask, NQKV, T, l.K, l.v8);
            } else k_attn_dec_w<<<dim3(kKv, kAttnChunks, T), 256, 0, st>>>(vQKV, l.K, l.V, vPart, state, l.swa, l.kv_mask, NQKV, {}, l.v8);
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
        if (pf || pe) {
            CK(cudaEventRecord(ev[kLayers], side));
            CK(cudaStreamWaitEvent(st, ev[kLayers], 0));
        }
        v_mv({{head, vLG}}, {vocab}, vH, kHidden, kHidden, T, vocab, false, out_norm);
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
    // FLINT_PF_HI: the prompt path's inputs without their fp16 remainder - "all", "dense" (attention, shared and dense
    // FFN products) or "experts" (the routed experts); unset: hi + lo everywhere (exact)
    static int pf_hi() {
        static const int v = [] {
            const char* e = std::getenv("FLINT_PF_HI");
            if (!e) return 0;
            const std::string s(e);
            return s == "dense" ? 1 : s == "experts" ? 2 : 3;
        }();
        return v;
    }
    static bool pf_lo() { return !(pf_hi() & 1); }        // (the dense products)
    static bool pf_lo_e() { return !(pf_hi() & 2); }      // (the routed experts)
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
        if (hybrid && n_layers() > kDense) hy_copy_gu(L[kDense]);   // (the first expert layer's gate/up)
        for (int i = 0; i < n_layers(); ++i) {
            Layer& l = L[i];
            if (hybrid && i >= kDense) hy_copy_d(l);   // (this layer's down, during its attention and gate/up)
            pf_split(pH, nullptr, l.attn_norm, kHidden, T);
            pf_gemm(l.wq, kQDim, kHidden, T, pQKV, NQKV, 0);
            pf_gemm(l.wk, kKvDim, kHidden, T, pQKV + kQDim, NQKV, 0);
            pf_gemm(l.wv, kKvDim, kHidden, T, pQKV + kQDim + kKvDim, NQKV, 0);
            k_pf_qkpost<<<dim3(kHeads + kKv, T), kHd, 0, st>>>(pQKV, l.q_norm, l.k_norm, l.K, l.V, p0, l.swa, l.kv_mask, l.v8);
            static const int aqt = std::getenv("FLINT_PF_QT") ? std::atoi(std::getenv("FLINT_PF_QT")) : 1;
            if (aqt == 1) k_pf_attn_w2<1><<<dim3(kKv, (T + 15) / 16), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa, l.kv_mask, l.v8);
            else if (aqt == 2) k_pf_attn_w2<2><<<dim3(kKv, (T + 31) / 32), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa, l.kv_mask, l.v8);
            else if (aqt == 4) k_pf_attn_w2<4><<<dim3(kKv, (T + 63) / 64), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa, l.kv_mask, l.v8);
            else k_pf_attn_w<<<dim3(kKv, (T + 15) / 16), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa, l.kv_mask, l.v8);
            // FLINT_DUMP_ATTN=DIR with FLINT_DUMP_AT=N: the prompt chunk ending at N writes, per full layer, the last 64
            // rows' q, the K cache and the 8-bit V store (for studying which keys the queries use)
            static const char* dump_dir = std::getenv("FLINT_DUMP_ATTN");
            static const int dump_at = std::getenv("FLINT_DUMP_AT") ? std::atoi(std::getenv("FLINT_DUMP_AT")) : -1;
            if (dump_dir && p0 + T == dump_at && !l.swa && l.K && l.v8.q && T >= 64) {
                CK(cudaStreamSynchronize(st));
                const size_t n = size_t(p0 + T), ng = (n + 15) & ~size_t(15);
                std::vector<float> qh(size_t(64) * kQDim);
                for (int r = 0; r < 64; ++r)
                    CK(cudaMemcpy(qh.data() + size_t(r) * kQDim, pQKV + size_t(T - 64 + r) * NQKV, kQDim * 4, cudaMemcpyDeviceToHost));
                std::vector<uint8_t> kh_(n * kKvDim * sizeof(kc_t)), vq(ng * kKvDim), vs(ng * kKv * 4), sk(kHeads * 4);
                CK(cudaMemcpy(kh_.data(), l.K, kh_.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(vq.data(), l.v8.q, vq.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(vs.data(), l.v8.s, vs.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(sk.data(), l.sinks, sk.size(), cudaMemcpyDeviceToHost));
                auto put = [&](const char* nm, const void* d, size_t bytes) {
                    const std::string fn = std::string(dump_dir) + "/L" + std::to_string(i) + "_" + nm;
                    FILE* f = std::fopen(fn.c_str(), "wb");
                    if (!f) { std::fprintf(stderr, "strata-flint: cannot write %s\n", fn.c_str()); std::exit(1); }
                    std::fwrite(d, 1, bytes, f);
                    std::fclose(f);
                };
                put("q.f32", qh.data(), qh.size() * 4);
                put("k.f16", kh_.data(), kh_.size());
                put("vq.i8", vq.data(), vq.size());
                put("vs.f32", vs.data(), vs.size());
                put("sink.f32", sk.data(), sk.size());
                std::fprintf(stderr, "strata-flint: layer %d attention dumped (%zu positions)\n", i, n);
            }
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
            static const bool router_old = std::getenv("FLINT_PF_ROUTER_OLD") != nullptr;
            if (router_old) {
                k_pf_rms<<<(T + 7) / 8, 256, 0, st>>>(pH, T, pRN);
                k_pf_router<<<(T + 15) / 16, 256, 0, st>>>(pH, pRN, l.ffn_norm, l.router, T, pRL);
            } else k_pf_router_w<<<(T + 15) / 16, 256, 0, st>>>(pXH, pXL, pXS, l.router16, l.router_e, T, pRL);
            CK(cudaMemsetAsync(pCount, 0, kExperts * 4, st));
            k_pf_topk<<<(T + 7) / 8, 256, 0, st>>>(pRL, T, pIdx, pW, pCount);
            // FLINT_ROUTE_DUMP=FILE: every token's 6 experts, appended per (chunk, layer) as int16 records
            // [layer, T] then T x 6 ids (for choosing which experts live on the card)
            static FILE* route_f = std::getenv("FLINT_ROUTE_DUMP") ? std::fopen(std::getenv("FLINT_ROUTE_DUMP"), "ab") : nullptr;
            if (route_f) {
                std::vector<int> ix(size_t(T) * kTopK);
                CK(cudaMemcpyAsync(ix.data(), pIdx, ix.size() * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::vector<int16_t> rec(2 + ix.size());
                rec[0] = int16_t(i); rec[1] = int16_t(T);
                for (size_t j = 0; j < ix.size(); ++j) rec[2 + j] = int16_t(ix[j]);
                std::fwrite(rec.data(), 2, rec.size(), route_f);
                std::fflush(route_f);
            }
            k_pf_bounds<<<1, kExperts, 0, st>>>(pCount, pBounds, pFill, pTiles, pNT);
            k_pf_scatter<<<(S + 255) / 256, 256, 0, st>>>(pIdx, T, pBounds, pFill, pSrc, pSlot);
            // experts' gate/up on the split rows of each routed token
            // hybrid: the layer's experts are not all on the card - copy the whole layer into the staging buffers
            const uint8_t *eg = l.gate_e, *eu = l.up_e, *ed = l.down_e;
            if (hybrid) {   // (copied on hy_cp: see hy_copy_gu / hy_copy_d)
                CK(cudaStreamWaitEvent(st, ev_gu_ready, 0));
                eg = stg_g; eu = stg_u; ed = stg_d;
            }
            if (pf_lo_e()) k_gemm_q41_grp<true, true><<<dim3(kFfExp / 256, max_tiles), 256, 0, st>>>(eg, eu, gu_bytes, kFfExp, kHidden, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT, kFfExp);
            else k_gemm_q41_grp<false, true><<<dim3(kFfExp / 256, max_tiles), 256, 0, st>>>(eg, eu, gu_bytes, kFfExp, kHidden, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT, kFfExp);
            if (hybrid) {   // gate/up's buffers are free: the next expert layer's go in while this layer finishes
                CK(cudaEventRecord(ev_gu_free, st));
                if (i + 1 < n_layers()) hy_copy_gu(L[i + 1]);
            }
            // shared expert's down, added into h
            pf_split(pG, pU, nullptr, kFfShared, T);
            pf_gemm(l.down_s, kHidden, kFfShared, T, pH, kHidden, 1);
            // experts' down on their act rows (grouped order), then the weighted sum into h
            pf_split(pACT, nullptr, nullptr, kFfExp, S);
            if (hybrid) CK(cudaStreamWaitEvent(st, ev_d_ready, 0));
            if (pf_lo_e()) k_gemm_q41_grp<true, false><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(ed, nullptr, d_bytes, kHidden, kFfExp, pXH, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY, kHidden);
            else k_gemm_q41_grp<false, false><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(ed, nullptr, d_bytes, kHidden, kFfExp, pXH, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY, kHidden);
            if (hybrid) CK(cudaEventRecord(ev_d_free, st));
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
        static const int nb = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : (FL_HIP ? 32 : 0);   // (NVIDIA: no 64 MB memory-side cache to fill)
        k_touch<<<nb, 256, 0, side>>>(T, sink);
    }
    void record() {
        if (mega_g) {
            CK(cudaMemsetAsync(mega_ctr, 0, 16, st));
            MegaArgs A{mega_layers, embed, out_norm, head, vocab, mega_g, state, h, mega_qkv, att, part, rl, act, logits, mega_plog, mega_best, mega_ctr};
            k_token<<<kMegaTot, 256, kActRows * 4, st>>>(A);
            return;
        }
        static const int pf = std::getenv("FLINT_PREFETCH") ? std::atoi(std::getenv("FLINT_PREFETCH")) : (FL_HIP ? 32 : 0);   // (NVIDIA: no 64 MB memory-side cache to fill)
        k_embed<<<kHidden / 256, 256, 0, st>>>(embed, state, h);
        for (int i = 0; i < n_layers(); ++i) {
            Layer& l = L[i];
            static const int pe = std::getenv("FLINT_PREDPF") ? std::atoi(std::getenv("FLINT_PREDPF")) : (FL_HIP ? 64 : 0);
            if ((pf || pe) && i + 1 < kLayers) {   // from the start of layer i, on the side stream:
                CK(cudaEventRecord(ev[i], st));
                CK(cudaStreamWaitEvent(side, ev[i], 0));
                if (pe && i >= kDense && !hybrid) {   // this layer's likely experts, picked from its input
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
            // (q / k norm, RoPE and the cache write happen inside the attention kernels: AttnFuse)
            const AttnFuse F{k, v, l.q_norm, l.k_norm, l.K, l.V};
            // the sliding layers (at most 513 keys) keep the per-head kernel; the full layers, whose keys grow with the
            // context, read each key once for the 8 heads sharing it
            static const bool attn_old = std::getenv("FLINT_ATTN_OLD") != nullptr;
            if ((attn_old && !l.v8.q) || l.swa || !FL_HIP) k_attn_part<<<dim3(kHeads, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa, l.kv_mask, 0, F);
            else k_attn_dec_w<<<dim3(kKv, kAttnChunks), 256, 0, st>>>(q, l.K, l.V, part, state, l.swa, l.kv_mask, 0, F, l.v8);
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
            if (hybrid) {   // the card computes the experts it holds while the CPU computes the others (hy_serve)
                k_hy_post<<<1, 256, 0, st>>>(h, idx, wts, l.slot, ridx, hy_s_dev, hy_ctr);
                k_moe_gu<<<kActRows / kMvRows, 256, kHidden * 4, st>>>(h, l.ffn_norm, l.gate_e, l.up_e, l.gate_s, l.up_s, ridx, act);
                k_moe_down<<<kHidden / kMvRows, 256, 0, st>>>(h, act, l.down_e, l.down_s, ridx, wts);
                k_hy_wait_add<<<kHidden / 256, 256, 0, st>>>(h, hy_s_dev, hy_ctr);
            } else {
                k_moe_gu<<<kActRows / kMvRows, 256, kHidden * 4, st>>>(h, l.ffn_norm, l.gate_e, l.up_e, l.gate_s, l.up_s, idx, act);
                k_moe_down<<<kHidden / kMvRows, 256, 0, st>>>(h, act, l.down_e, l.down_s, idx, wts);
            }
            if (std::getenv("FLINT_DUMP_H")) dump_rows(h, 1);
        }
        static const int pe2 = std::getenv("FLINT_PREDPF") ? std::atoi(std::getenv("FLINT_PREDPF")) : (FL_HIP ? 64 : 0);
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
    {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-flint: card memory %.2f GB free of %.2f GB after load\n", fr / 1e9, tot / 1e9);
    }
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
        for (size_t i = 0; i < ids.size(); i += chunk)   // (in pieces of FLINT_CHUNK, as a long prompt goes)
            m.prefill_chunk(ids.data() + i, int(std::min(ids.size() - i, size_t(chunk))), lga + i * size_t(m.vocab));
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
    if (m.hybrid && n_new > 0)
        std::fprintf(stderr, "strata-flint: hybrid - %.0f%% of the chosen experts on the card; a token: %.2f ms waiting for the card's "
                     "hand-over, %.2f ms on the CPU\n", 100.0 * m.hy_card_experts / std::max<size_t>(1, m.hy_card_experts + m.hy_cpu_experts),
                     m.hy_t_wait * 1e3 / n_new, m.hy_t_cpu * 1e3 / n_new);
    return 0;
}
