// src/nemotron/nemo_main.cu - strata-nemo: NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB's language model on one card.
//
// v1, the correctness build: one position at a time through all 88 blocks (40 Mamba-2, 40 latent MoE, 8 attention),
// every weight resident in VRAM in its stored format. It is checked block by block against the float32 reference in
// ~/nemotron-ref/ref.py (same maths, written as plainly as possible) before anything is made fast.
//
//   strata-nemo --model DIR --ids 1784,8961,1307 [--new N] [--dump FILE] [--max-ctx N]
//
// DIR holds the safetensors files and strata-manifest.txt (written by ~/nemotron-ref/manifest.py). --dump writes, per
// position, the 88 block outputs then the logits, all float32, after a header "NEMO1 positions blocks hidden vocab".
//
// Reused Strata kernels: iq_mmvq (IQ2_XXS experts) after quantize_q8_1_rows, native_q8_0_f32 (attention, latent
// projections, the output head). New here: RMSNorm, the gated group norm, NVFP4 and BF16 GEMVs, the Mamba-2 conv and
// state step, NoPE grouped-query attention, relu^2 and the small vector helpers.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

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
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "strata-nemo: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kIQ2_XXS = 16;
constexpr int kHidden = 4096, kVocab = 131072;
constexpr int kMHeads = 128, kMHeadDim = 64, kMGroups = 8, kMState = 128, kConvK = 4;
constexpr int kMInter = kMHeads * kMHeadDim;                       // 8192
constexpr int kConvDim = kMInter + 2 * kMGroups * kMState;         // 10240
constexpr int kInProj = kMInter + kConvDim + kMHeads;              // 18560
constexpr int kAHeads = 32, kAKv = 2, kAHeadDim = 128;
constexpr int kLatent = 1024, kShared = 5376;
constexpr float kEps = 1e-5f, kRouteScale = 5.0f;

// ---------------------------------------------------------------------------------------------------- kernels
__device__ inline float bf16f(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }

__device__ inline float e4m3f(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r;
    if (e == 0) r = ldexpf(float(m) / 8.f, -6);
    else if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
    else r = ldexpf(1.f + float(m) / 8.f, e - 7);
    return s ? -r : r;
}

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

template <int N>
__device__ inline float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = N / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

// y = w * x / rms(x); w bf16
__global__ void k_rmsnorm(const float* x, const uint16_t* w, float* y, int n) {
    __shared__ float sh[1024];
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) a += x[i] * x[i];
    const float inv = rsqrtf(block_sum<1024>(a, sh) / n + kEps);
    for (int i = threadIdx.x; i < n; i += 1024) y[i] = bf16f(w[i]) * (x[i] * inv);
}

// y[r] = W[r] . x, W bf16 (rows x n)
__global__ void k_bf16_gemv(const uint16_t* W, const float* x, float* y, int n) {
    __shared__ float sh[256];
    const uint16_t* row = W + size_t(blockIdx.x) * n;
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) a += bf16f(row[i]) * x[i];
    a = block_sum<256>(a, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = a;
}

// y[r] = scale2 * sum_g E4M3(s[r][g]) * sum_{j in g} E2M1(w[r][j]) * x[j]; two values per byte, low nibble first
__global__ void k_nvfp4_gemv(const uint8_t* W, const uint8_t* S, float scale2, const float* x, float* y, int n) {
    __shared__ float sh[256];
    const uint8_t* row = W + size_t(blockIdx.x) * (n / 2);
    const uint8_t* srow = S + size_t(blockIdx.x) * (n / 16);
    float a = 0.f;
    for (int g = threadIdx.x; g < n / 16; g += 256) {           // one 16-value group per step
        const uint8_t* b = row + g * 8;
        const float* xv = x + g * 16;
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) d += kE2M1[b[j] & 15] * xv[2 * j] + kE2M1[b[j] >> 4] * xv[2 * j + 1];
        a += d * e4m3f(srow[g]);
    }
    a = block_sum<256>(a, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = a * scale2;
}

__global__ void k_embed(const uint16_t* table, int token, float* y) {
    const uint16_t* r = table + size_t(token) * kHidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kHidden; i += gridDim.x * blockDim.x) y[i] = bf16f(r[i]);
}

__global__ void k_add(float* a, const float* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] += b[i];
}
__global__ void k_axpy(float* a, const float* b, float w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] += w * b[i];
}
// row r of logits: -log p(next[r]) and whether it is the top token, added to acc[0], acc[1] (rows), acc[2] (top-1);
// rows with no next token (left <= r) are skipped
__global__ void __launch_bounds__(1024) k_nll_rows(const float* logits, int V, const int32_t* next, int left, double* acc) {
    __shared__ float sm[1024], ss[1024];
    __shared__ int si[1024];
    const int r = blockIdx.x;
    if (r >= left) return;
    const float* x = logits + size_t(r) * V;
    float m = -INFINITY; int mi = 0;
    for (int i = threadIdx.x; i < V; i += 1024) if (x[i] > m) { m = x[i]; mi = i; }
    sm[threadIdx.x] = m; si[threadIdx.x] = mi;
    __syncthreads();
    for (int o = 512; o > 0; o >>= 1) {
        if (threadIdx.x < o && sm[threadIdx.x + o] > sm[threadIdx.x]) { sm[threadIdx.x] = sm[threadIdx.x + o]; si[threadIdx.x] = si[threadIdx.x + o]; }
        __syncthreads();
    }
    const float M = sm[0];
    float z = 0.f;
    for (int i = threadIdx.x; i < V; i += 1024) z += expf(x[i] - M);
    ss[threadIdx.x] = z;
    __syncthreads();
    for (int o = 512; o > 0; o >>= 1) {
        if (threadIdx.x < o) ss[threadIdx.x] += ss[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int tgt = next[r];
        atomicAdd(&acc[0], double(M) + log(double(ss[0])) - double(x[tgt]));
        atomicAdd(&acc[1], 1.0);
        atomicAdd(&acc[2], si[0] == tgt ? 1.0 : 0.0);
    }
}
__global__ void k_relu2(float* a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { float v = fmaxf(a[i], 0.f); a[i] = v * v; }
}
__global__ void k_sigmoid(float* a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] = 1.f / (1.f + expf(-a[i]));
}

// Mamba-2 depthwise causal conv over xBC (state holds the last kConvK-1 inputs, oldest first), then SiLU
__global__ void k_mamba_conv(const float* xBC, float* state, const uint16_t* cw, const uint16_t* cb, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConvDim) return;
    float* st = state + size_t(c) * (kConvK - 1);
    const float v[kConvK] = {st[0], st[1], st[2], xBC[c]};
    float a = bf16f(cb[c]);
#pragma unroll
    for (int k = 0; k < kConvK; ++k) a += bf16f(cw[c * kConvK + k]) * v[k];
    st[0] = v[1]; st[1] = v[2]; st[2] = v[3];
    out[c] = a / (1.f + expf(-a));
}

// one head per block, one head-dim row per thread: h = exp(dt*A) h + dt B x ; y = C.h + D x
__global__ void k_mamba_step(const float* xBC, const float* dt_raw, const uint16_t* A_log, const uint16_t* D,
                             const uint16_t* dt_bias, float* ssm, float* y) {
    const int hd = blockIdx.x, p = threadIdx.x;                     // p < kMHeadDim
    const int g = hd / (kMHeads / kMGroups);
    const float* xs = xBC;                                          // (kMInter)
    const float* B = xBC + kMInter + g * kMState;
    const float* C = xBC + kMInter + kMGroups * kMState + g * kMState;
    float dt = dt_raw[hd] + bf16f(dt_bias[hd]);
    dt = dt > 20.f ? dt : log1pf(expf(dt));                         // softplus, as torch's (threshold 20)
    const float A = -expf(bf16f(A_log[hd]));
    const float dA = expf(dt * A);
    const float x = xs[hd * kMHeadDim + p];
    float* h = ssm + (size_t(hd) * kMHeadDim + p) * kMState;
    float acc = 0.f;
    for (int n = 0; n < kMState; ++n) {
        const float v = h[n] * dA + dt * B[n] * x;
        h[n] = v;
        acc += v * C[n];
    }
    y[hd * kMHeadDim + p] = acc + x * bf16f(D[hd]);
}

// y = groupnorm(y * silu(z)) * w, groups of kMInter / kMGroups
__global__ void k_gated_norm(float* y, const float* z, const uint16_t* w) {
    __shared__ float sh[256];
    const int G = kMInter / kMGroups, base = blockIdx.x * G;
    float a = 0.f;
    for (int i = threadIdx.x; i < G; i += 256) {
        const float zz = z[base + i];
        const float v = y[base + i] * (zz / (1.f + expf(-zz)));
        y[base + i] = v;
        a += v * v;
    }
    const float inv = rsqrtf(block_sum<256>(a, sh) / G + kEps);
    for (int i = threadIdx.x; i < G; i += 256) y[base + i] = y[base + i] * inv * bf16f(w[base + i]);
}

// attention over positions [0, T): scores, softmax, weighted values. K/V: (kAKv, max_ctx, kAHeadDim) float32
__global__ void k_att_scores(const float* q, const float* K, float* sc, int T, int max_ctx) {
    const int h = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* k = K + (size_t(h / (kAHeads / kAKv)) * max_ctx + t) * kAHeadDim;
    const float* qq = q + h * kAHeadDim;
    float a = 0.f;
    for (int d = 0; d < kAHeadDim; ++d) a += qq[d] * k[d];
    sc[size_t(h) * max_ctx + t] = a * rsqrtf(float(kAHeadDim));
}
__global__ void k_att_softmax(float* sc, int T, int max_ctx) {
    __shared__ float sh[256];
    float* s = sc + size_t(blockIdx.x) * max_ctx;
    float m = -INFINITY;
    for (int t = threadIdx.x; t < T; t += 256) m = fmaxf(m, s[t]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0];
    __syncthreads();
    float a = 0.f;
    for (int t = threadIdx.x; t < T; t += 256) { const float e = expf(s[t] - m); s[t] = e; a += e; }
    const float inv = 1.f / block_sum<256>(a, sh);
    for (int t = threadIdx.x; t < T; t += 256) s[t] *= inv;
}
__global__ void k_att_out(const float* sc, const float* V, float* o, int T, int max_ctx) {
    const int h = blockIdx.x, d = threadIdx.x;                      // d < kAHeadDim
    const float* v = V + size_t(h / (kAHeads / kAKv)) * max_ctx * kAHeadDim + d;
    const float* s = sc + size_t(h) * max_ctx;
    float a = 0.f;
    for (int t = 0; t < T; ++t) a += s[t] * v[size_t(t) * kAHeadDim];
    o[h * kAHeadDim + d] = a;
}


// router: sigmoid, top-k by sigmoid + bias (n_group 1), weights = the chosen sigmoids normalised to 1, times 5.
// One block of 512 threads (n <= 512); ties go to the lower expert number.
__global__ void k_route(const float* raw, const float* bias, int n, int k, int32_t* idx, float* wts) {
    __shared__ float c[512], sg[512], bv[512];
    __shared__ int bi[512];
    __shared__ int chosen[16];
    const int t = threadIdx.x;
    const float s = t < n ? 1.f / (1.f + expf(-raw[t])) : 0.f;
    sg[t] = s;
    c[t] = t < n ? s + bias[t] : -INFINITY;
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        bv[t] = c[t]; bi[t] = t;
        __syncthreads();
        for (int st = 256; st > 0; st >>= 1) {
            if (t < st && (bv[t + st] > bv[t] || (bv[t + st] == bv[t] && bi[t + st] < bi[t]))) { bv[t] = bv[t + st]; bi[t] = bi[t + st]; }
            __syncthreads();
        }
        if (t == 0) { chosen[r] = bi[0]; c[bi[0]] = -INFINITY; }
        __syncthreads();
    }
    if (t == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += sg[chosen[r]];
        for (int r = 0; r < k; ++r) { idx[r] = chosen[r]; wts[r] = sg[chosen[r]] / (sum + 1e-20f) * kRouteScale; }
    }
}

// eo = sum_j w[j] * y[j], y: (k, n)
__global__ void k_wsum(float* eo, const float* y, const float* w, int k, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < k; ++j) a += w[j] * y[size_t(j) * n + i];
        eo[i] = a;
    }
}


// NVFP4 GEMV, one 32-lane warp per row, 16-byte loads: lane i reads 32 values (16 bytes) and their two E4M3 scales
// per step, so a warp reads 512 contiguous bytes at a time.  The E4M3 and E2M1 decodes come from per-block tables.
__global__ void __launch_bounds__(256) k_nvfp4_gemv2(const uint8_t* W, const uint8_t* S, float scale2, const float* x,
                                                     float* y, int n, int rows) {
    __shared__ float e4[256], e2[16];
    e4[threadIdx.x] = e4m3f(uint8_t(threadIdx.x));
    if (threadIdx.x < 16) e2[threadIdx.x] = kE2M1[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + size_t(row) * (n / 2));
    const uint16_t* s2 = reinterpret_cast<const uint16_t*>(S + size_t(row) * (n / 16));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
    for (int i = lane; i < n / 32; i += 32) {
        const uint4 q = w4[i];
        const uint16_t sc = s2[i];
        const uint32_t word[4] = {q.x, q.y, q.z, q.w};
        float part[2] = {0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {                                // 8 values per 32-bit word
            const float4 xa = x4[i * 8 + 2 * k], xb = x4[i * 8 + 2 * k + 1];
            const uint32_t b = word[k];
            part[k >> 1] += e2[b & 15] * xa.x + e2[(b >> 4) & 15] * xa.y + e2[(b >> 8) & 15] * xa.z + e2[(b >> 12) & 15] * xa.w
                          + e2[(b >> 16) & 15] * xb.x + e2[(b >> 20) & 15] * xb.y + e2[(b >> 24) & 15] * xb.z + e2[b >> 28] * xb.w;
        }
        acc += part[0] * e4[sc & 255] + part[1] * e4[sc >> 8];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o, 32);
    if (lane == 0) y[row] = acc * scale2;
}

// Mamba-2 state step, one head per block, one state column per thread (128): reads and writes of the state are
// contiguous across the block; each head-dim row's C.h is summed per warp, then across the four warps.
__global__ void __launch_bounds__(128) k_mamba_step2(const float* xBC, const float* dt_raw, const uint16_t* A_log,
                                                     const uint16_t* D, const uint16_t* dt_bias, float* ssm, float* y) {
    __shared__ float xs[kMHeadDim], part[kMHeadDim][4];
    const int hd = blockIdx.x, n = threadIdx.x, warp = n / 32, lane = n % 32;
    const int g = hd / (kMHeads / kMGroups);
    constexpr int RP = kMHeadDim / 4;                               // head-dim rows per block (gridDim.y = 4)
    const int p0 = blockIdx.y * RP;
    if (n < kMHeadDim) xs[n] = xBC[hd * kMHeadDim + n];
    float dt = dt_raw[hd] + bf16f(dt_bias[hd]);
    dt = dt > 20.f ? dt : log1pf(expf(dt));
    const float dA = expf(dt * -expf(bf16f(A_log[hd])));
    const float dtB = dt * xBC[kMInter + g * kMState + n];
    const float Cn = xBC[kMInter + kMGroups * kMState + g * kMState + n];
    __syncthreads();
    float* h = ssm + size_t(hd) * kMHeadDim * kMState + n;
    for (int p = p0; p < p0 + RP; ++p) {
        const float v = h[p * kMState] * dA + dtB * xs[p];
        h[p * kMState] = v;
        float c = v * Cn;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffff, c, o, 32);
        if (lane == 0) part[p][warp] = c;
    }
    __syncthreads();
    if (n >= p0 && n < p0 + RP) y[hd * kMHeadDim + n] = part[n][0] + part[n][1] + part[n][2] + part[n][3] + xs[n] * bf16f(D[hd]);
}


// graph-friendly pieces: the step's token and position live on the card (written before each replay)
__global__ void k_embed_dev(const uint16_t* table, const int* tok, float* y) {
    const uint16_t* r = table + size_t(*tok) * kHidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kHidden; i += gridDim.x * blockDim.x) y[i] = bf16f(r[i]);
}
__global__ void k_kv_append(const float* k, const float* v, float* K, float* V, const int* pos, int max_ctx) {
    const int i = threadIdx.x, kv = blockIdx.x;                     // kAKv blocks of kAHeadDim threads
    const size_t at = (size_t(kv) * max_ctx + *pos) * kAHeadDim + i;
    K[at] = k[kv * kAHeadDim + i];
    V[at] = v[kv * kAHeadDim + i];
}
__global__ void k_att_scores_dev(const float* q, const float* K, float* sc, const int* pos, int max_ctx) {
    const int h = blockIdx.y, T = *pos + 1;
    const float* qq = q + h * kAHeadDim;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T; t += gridDim.x * blockDim.x) {
        const float* k = K + (size_t(h / (kAHeads / kAKv)) * max_ctx + t) * kAHeadDim;
        float a = 0.f;
        for (int d = 0; d < kAHeadDim; ++d) a += qq[d] * k[d];
        sc[size_t(h) * max_ctx + t] = a * rsqrtf(float(kAHeadDim));
    }
}
__global__ void k_att_softmax_dev(float* sc, const int* pos, int max_ctx) {
    __shared__ float sh[256];
    const int T = *pos + 1;
    float* s = sc + size_t(blockIdx.x) * max_ctx;
    float m = -INFINITY;
    for (int t = threadIdx.x; t < T; t += 256) m = fmaxf(m, s[t]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0];
    __syncthreads();
    float a = 0.f;
    for (int t = threadIdx.x; t < T; t += 256) { const float e = expf(s[t] - m); s[t] = e; a += e; }
    const float inv = 1.f / block_sum<256>(a, sh);
    for (int t = threadIdx.x; t < T; t += 256) s[t] *= inv;
}
__global__ void k_att_out_dev(const float* sc, const float* V, float* o, const int* pos, int max_ctx) {
    const int h = blockIdx.x, d = threadIdx.x, T = *pos + 1;
    const float* v = V + size_t(h / (kAHeads / kAKv)) * max_ctx * kAHeadDim + d;
    const float* s = sc + size_t(h) * max_ctx;
    float a = 0.f;
    for (int t = 0; t < T; ++t) a += s[t] * v[size_t(t) * kAHeadDim];
    o[h * kAHeadDim + d] = a;
}


// h += d (+ e if given); y = w * h / rms(h)
__global__ void k_add_rmsnorm(float* h, const float* d, const float* e, const uint16_t* w, float* y, int n) {
    __shared__ float sh[1024];
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = h[i] + d[i] + (e ? e[i] : 0.f);
        h[i] = v;
        a += v * v;
    }
    const float inv = rsqrtf(block_sum<1024>(a, sh) / n + kEps);
    for (int i = threadIdx.x; i < n; i += 1024) y[i] = bf16f(w[i]) * (h[i] * inv);
}

#include "nemo_batch.cuh"
#include "nemo_draft.cuh"

// ---------------------------------------------------------------------------------------------------- weights
struct Tensor { std::string file, fmt; size_t off = 0, bytes = 0; std::vector<int64_t> shape; void* dev = nullptr; };

struct Store {
    std::string dir;
    std::map<std::string, Tensor> t;
    size_t loaded = 0;

    explicit Store(const std::string& d) : dir(d) {
        std::ifstream in(d + "/strata-manifest.txt");
        if (!in) { std::fprintf(stderr, "strata-nemo: no %s/strata-manifest.txt (run manifest.py)\n", d.c_str()); std::exit(1); }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::string name, dtype, shape;
            Tensor x;
            ss >> name >> x.file >> x.off >> x.bytes >> dtype >> x.fmt >> shape;
            std::stringstream sh(shape);
            for (std::string v; std::getline(sh, v, ',');) x.shape.push_back(std::stoll(v));
            t[name] = x;
        }
    }
    Tensor& at(const std::string& name) {
        auto it = t.find(name);
        if (it == t.end()) { std::fprintf(stderr, "strata-nemo: tensor %s missing\n", name.c_str()); std::exit(1); }
        return it->second;
    }
    // the tensor's bytes into `to` on the card
    void load_into(const std::string& name, void* to) {
        Tensor& x = at(name);
        std::vector<char> buf(x.bytes);
        const int fd = ::open((dir + "/" + x.file).c_str(), O_RDONLY);
        if (fd < 0 || ::pread(fd, buf.data(), x.bytes, off_t(x.off)) != ssize_t(x.bytes)) {
            std::fprintf(stderr, "strata-nemo: reading %s failed\n", name.c_str()); std::exit(1);
        }
        ::close(fd);
        CK(cudaMemcpy(to, buf.data(), x.bytes, cudaMemcpyHostToDevice));
        x.dev = to;
        loaded += x.bytes;
    }
    // the tensor's bytes on the card (read once), followed by kMmqTail zero bytes
    void* dev(const std::string& name) {
        Tensor& x = at(name);
        if (x.dev) return x.dev;
        void* p = nullptr;
        CK(cudaMalloc(&p, x.bytes + kMmqTail));
        CK(cudaMemset((char*) p + x.bytes, 0, kMmqTail));
        load_into(name, p);
        return p;
    }
    float f32(const std::string& name) {
        Tensor& x = at(name);
        float v = 0;
        const int fd = ::open((dir + "/" + x.file).c_str(), O_RDONLY);
        if (fd < 0 || ::pread(fd, &v, 4, off_t(x.off)) != 4) { std::fprintf(stderr, "strata-nemo: reading %s failed\n", name.c_str()); std::exit(1); }
        ::close(fd);
        return v;
    }
};

// ---------------------------------------------------------------------------------------------------- model
enum Kind { MAMBA, MOE, ATTN };

struct Block {
    Kind kind;
    const uint16_t* norm = nullptr;
    // mamba
    const uint8_t *in_w = nullptr, *in_s = nullptr, *out_w = nullptr, *out_s = nullptr;
    float in_s2 = 0, out_s2 = 0;
    const uint16_t *conv_w = nullptr, *conv_b = nullptr, *A_log = nullptr, *D = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
    float *conv_state = nullptr, *ssm = nullptr;
    // moe
    int n_exp = 0, width = 0, topk = 0;
    const uint16_t* router = nullptr;
    std::vector<float> bias;
    float* bias_dev = nullptr;
    unsigned long long *up_tab = nullptr, *down_tab = nullptr;
    const void *fc1 = nullptr, *fc2 = nullptr, *sh_up = nullptr, *sh_down = nullptr;
    std::vector<const void*> up, down;
    uint8_t *up_all = nullptr, *down_all = nullptr;   // the experts packed, for MMQ
    size_t up_bytes = 0, down_bytes = 0;
    // attention
    const void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    uint16_t *K = nullptr, *V = nullptr;   // attention: the KV cache in BF16, kv_rows rows a KV head
    int win = -1;                     // index of this Mamba layer's window buffers
};

std::vector<std::string> read_block_types(const std::string& dir, std::vector<std::pair<int, int>>& moe_sizes,
                                          std::vector<int>& topk) {
    // config.json is huge (per-layer quantization notes); the two lists needed are found by plain text search
    std::ifstream in(dir + "/config.json");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    std::vector<std::string> types;
    size_t p = s.find("\"block_configs\"");
    if (p == std::string::npos) { std::fprintf(stderr, "strata-nemo: no block_configs in config.json\n"); std::exit(1); }
    const size_t end = s.find(']', p);
    while (true) {
        const size_t b = s.find('{', p);
        if (b == std::string::npos || b > end) break;
        const size_t e = s.find('}', b);
        const std::string obj = s.substr(b, e - b);
        auto num = [&](const char* key) {
            const size_t k = obj.find(key);
            return k == std::string::npos ? 0 : std::atoi(obj.c_str() + obj.find(':', k) + 1);
        };
        const size_t bt = obj.find("\"block_type\"");
        const size_t q1 = obj.find('"', obj.find(':', bt) + 1), q2 = obj.find('"', q1 + 1);
        types.push_back(obj.substr(q1 + 1, q2 - q1 - 1));
        moe_sizes.push_back({num("\"n_routed_experts\""), num("\"moe_intermediate_size\"")});
        topk.push_back(num("\"num_experts_per_tok\""));
        p = e;
    }
    return types;
}

struct Model {
    Store st;
    std::vector<Block> blocks;
    int max_ctx;
    int kv_rows() const { return max_ctx + 64; }   // the BF16 KV cache's rows a KV head (k_attn_flash reads up to 63 past)
    const uint16_t *embed = nullptr, *norm_f = nullptr;
    const void* head = nullptr;
    // scratch, kTMax rows each
    float *h, *n, *d, *y, *lat, *eo, *tmp, *tmp2, *scores, *att_sc, *qkv, *kbuf, *vbuf, *logits;
    float *ex_up, *ex_down, *route_w;
    int32_t* route_idx;
    int32_t *g_n, *g_exp, *g_cnt, *g_tok, *g_slot;   // the step's expert groups
    void *q8, *q8b;
    void* q8s = nullptr;                   // the shared expert's packed inputs (its own: it runs beside the routed path)
    cudaStream_t stream2 = nullptr;         // the shared expert's stream in a small step
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    cudaStream_t stream = nullptr;
    int *d_toks = nullptr, *d_pos = nullptr, *h_tp = nullptr;   // [kTMax tokens, position of the first]
    // per Mamba layer: the window's projections and conv outputs, kept until the accepted count is known
    std::vector<float*> win_proj, win_xbc;
    // the drafter's context: the stream after blocks kTaps, (rows, 6 x 4096); on only when a drafter is attached
    float* aux = nullptr;          // kTMax rows (small steps)
    float* aux_big = nullptr;      // kTBig rows (prompt chunks)

    Model(const std::string& dir, int ctx) : st(dir), max_ctx(ctx) {
        CK(cudaStreamCreate(&stream));
        CK(cudaMalloc(&d_toks, (kTMax + 1) * sizeof(int)));
        d_pos = d_toks + kTMax;
        CK(cudaMallocHost(&h_tp, (kTMax + 1) * sizeof(int)));
        std::vector<std::pair<int, int>> sizes;
        std::vector<int> topk;
        const auto types = read_block_types(dir, sizes, topk);
        const std::string P = "language_model.backbone.";
        embed = (const uint16_t*) st.dev(P + "embeddings.weight");
        norm_f = (const uint16_t*) st.dev(P + "norm_f.weight");
        head = st.dev("language_model.lm_head.weight");
        for (size_t i = 0; i < types.size(); ++i) {
            const std::string L = P + "layers." + std::to_string(i) + ".";
            const std::string M = L + "mixer.";
            Block b;
            b.norm = (const uint16_t*) st.dev(L + "norm.weight");
            if (types[i] == "mamba") {
                b.kind = MAMBA;
                b.in_w = (const uint8_t*) st.dev(M + "in_proj.weight");
                b.in_s = (const uint8_t*) st.dev(M + "in_proj.weight_scale");
                b.in_s2 = st.f32(M + "in_proj.weight_scale_2");
                b.out_w = (const uint8_t*) st.dev(M + "out_proj.weight");
                b.out_s = (const uint8_t*) st.dev(M + "out_proj.weight_scale");
                b.out_s2 = st.f32(M + "out_proj.weight_scale_2");
                b.conv_w = (const uint16_t*) st.dev(M + "conv1d.weight");
                b.conv_b = (const uint16_t*) st.dev(M + "conv1d.bias");
                b.A_log = (const uint16_t*) st.dev(M + "A_log");
                b.D = (const uint16_t*) st.dev(M + "D");
                b.dt_bias = (const uint16_t*) st.dev(M + "dt_bias");
                b.gnorm = (const uint16_t*) st.dev(M + "norm.weight");
                CK(cudaMalloc(&b.conv_state, sizeof(float) * kConvDim * (kConvK - 1)));
                CK(cudaMalloc(&b.ssm, sizeof(float) * kMHeads * kMHeadDim * kMState));
                float *wp, *wx;
                CK(cudaMalloc(&wp, sizeof(float) * kTMax * kInProj));
                CK(cudaMalloc(&wx, sizeof(float) * kTMax * kConvDim));
                win_proj.push_back(wp); win_xbc.push_back(wx);
                b.win = int(win_proj.size()) - 1;
            } else if (types[i] == "moe") {
                b.kind = MOE;
                b.n_exp = sizes[i].first; b.width = sizes[i].second; b.topk = topk[i];
                b.router = (const uint16_t*) st.dev(M + "gate.weight");
                b.bias.resize(b.n_exp);
                {
                    Tensor& t = st.at(M + "gate.e_score_correction_bias");
                    const int fd = ::open((st.dir + "/" + t.file).c_str(), O_RDONLY);
                    if (::pread(fd, b.bias.data(), t.bytes, off_t(t.off)) != ssize_t(t.bytes)) { std::fprintf(stderr, "bias read\n"); std::exit(1); }
                    ::close(fd);
                }
                b.fc1 = st.dev(M + "fc1_latent_proj.weight");
                b.fc2 = st.dev(M + "fc2_latent_proj.weight");
                b.sh_up = st.dev(M + "shared_experts.up_proj.weight");
                b.sh_down = st.dev(M + "shared_experts.down_proj.weight");
                b.up_bytes = st.at(M + "experts.0.up_proj.weight").bytes;
                b.down_bytes = st.at(M + "experts.0.down_proj.weight").bytes;
                CK(cudaMalloc(&b.up_all, b.up_bytes * b.n_exp + kMmqTail));
                CK(cudaMalloc(&b.down_all, b.down_bytes * b.n_exp + kMmqTail));
                CK(cudaMemset(b.up_all + b.up_bytes * b.n_exp, 0, kMmqTail));
                CK(cudaMemset(b.down_all + b.down_bytes * b.n_exp, 0, kMmqTail));
                for (int e = 0; e < b.n_exp; ++e) {
                    const std::string E = M + "experts." + std::to_string(e) + ".";
                    if (st.at(E + "up_proj.weight").bytes != b.up_bytes || st.at(E + "down_proj.weight").bytes != b.down_bytes) {
                        std::fprintf(stderr, "strata-nemo: block %zu expert %d differs in size\n", i, e); std::exit(1);
                    }
                    st.load_into(E + "up_proj.weight", b.up_all + b.up_bytes * e);
                    st.load_into(E + "down_proj.weight", b.down_all + b.down_bytes * e);
                    b.up.push_back(b.up_all + b.up_bytes * e);
                    b.down.push_back(b.down_all + b.down_bytes * e);
                }
                std::vector<unsigned long long> u(b.n_exp), dn(b.n_exp);
                for (int e = 0; e < b.n_exp; ++e) { u[e] = (unsigned long long) b.up[e]; dn[e] = (unsigned long long) b.down[e]; }
                CK(cudaMalloc(&b.up_tab, b.n_exp * 8)); CK(cudaMalloc(&b.down_tab, b.n_exp * 8)); CK(cudaMalloc(&b.bias_dev, b.n_exp * 4));
                CK(cudaMemcpy(b.up_tab, u.data(), b.n_exp * 8, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(b.down_tab, dn.data(), b.n_exp * 8, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(b.bias_dev, b.bias.data(), b.n_exp * 4, cudaMemcpyHostToDevice));
            } else if (types[i] == "attention") {
                b.kind = ATTN;
                b.q = st.dev(M + "q_proj.weight"); b.k = st.dev(M + "k_proj.weight");
                b.v = st.dev(M + "v_proj.weight"); b.o = st.dev(M + "o_proj.weight");
                CK(cudaMalloc(&b.K, 2 * kAKv * size_t(kv_rows()) * kAHeadDim));
                CK(cudaMalloc(&b.V, 2 * kAKv * size_t(kv_rows()) * kAHeadDim));
                CK(cudaMemset(b.K, 0, 2 * kAKv * size_t(kv_rows()) * kAHeadDim));   // (finite past the last key)
                CK(cudaMemset(b.V, 0, 2 * kAKv * size_t(kv_rows()) * kAHeadDim));
            } else {
                std::fprintf(stderr, "strata-nemo: block %zu has unknown type %s\n", i, types[i].c_str()); std::exit(1);
            }
            blocks.push_back(std::move(b));
            if (i % 8 == 7) std::fprintf(stderr, "strata-nemo: loaded %zu of %zu blocks, %.1f GB\n", i + 1, types.size(), st.loaded / 1e9);
        }
        auto f = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        const size_t T = kTMax;
        f(&h, T * kHidden); f(&n, T * kHidden); f(&d, T * kHidden); f(&y, T * kMInter);
        f(&lat, T * kLatent); f(&eo, T * kLatent); f(&tmp, T * kShared); f(&tmp2, T * kShared); f(&scores, T * 512);
        f(&att_sc, T * kAHeads * std::max(size_t(max_ctx), size_t(kDecSplits) * (kAHeadDim + 2))); f(&qkv, T * kAHeads * kAHeadDim);
        f(&kbuf, T * kAKv * kAHeadDim); f(&vbuf, T * kAKv * kAHeadDim); f(&logits, T * size_t(kVocab));
        f(&ex_up, T * 16 * 2560); f(&ex_down, T * 16 * kLatent); f(&route_w, T * 16);
        CK(cudaMalloc(&route_idx, T * 16 * sizeof(int32_t)));
        CK(cudaMalloc(&best_dev, kTMax * 4));
        CK(cudaMalloc(&g_n, 4)); CK(cudaMalloc(&g_exp, 512 * 4)); CK(cudaMalloc(&g_cnt, 512 * 4));
        CK(cudaMalloc(&g_tok, T * 16 * 16 * 4)); CK(cudaMalloc(&g_slot, T * 16 * 16 * 4));
        CK(cudaMalloc(&q8, std::max(strata::kernels::native_q8_1_bytes(8192, 8), size_t(kTMax) * (kAux / 32) * 36)));   // up to 16 rows of 24,576 (the drafter's fc)
        CK(cudaMalloc(&q8b, size_t(T) * 16 * (2560 / 32) * 36));
        CK(cudaMalloc(&q8s, size_t(kTMax) * (kShared / 32) * 36));
        CK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
        reset();
    }

    int pos = 0;
    void reset() {
        for (auto& b : blocks) {
            if (b.kind == MAMBA) {
                CK(cudaMemset(b.conv_state, 0, sizeof(float) * kConvDim * (kConvK - 1)));
                CK(cudaMemset(b.ssm, 0, sizeof(float) * kMHeads * kMHeadDim * kMState));
            }
        }
        pos = 0;
    }

    // Strata's Q8_0 kernel takes at most 8 columns a call
    void q8_0(const void* w, const float* x, float* out, int n_in, int n_out, int T) {
        for (int c = 0; c < T; c += 8)
            strata::kernels::native_q8_0_f32(w, x + size_t(c) * n_in, q8, out + size_t(c) * n_out, n_in, n_out, std::min(8, T - c), stream);
    }
    // T > 1: several rows per warp share each activation load (NEMO_SH_NR 2 or 4, 0 = the one-row kernel)
    int grp_y = std::getenv("NEMO_GRP_Y") ? std::atoi(std::getenv("NEMO_GRP_Y")) : 16;
    int sh_nr = std::getenv("NEMO_SH_NR") ? std::atoi(std::getenv("NEMO_SH_NR")) : 4;
    void iq2(const void* w, const float* x, float* out, int n_in, int n_out, int T) {
        strata::kernels::quantize_q8_1_rows(x, T, n_in, q8, stream);
        iq2_q(w, q8, out, n_in, n_out, T);
    }
    bool sh_par = !std::getenv("NEMO_SH_PAR") || std::atoi(std::getenv("NEMO_SH_PAR")) != 0;   // shared expert on stream2
    void iq2_q_on(const void* w, const void* xq, float* out, int n_in, int n_out, int T, cudaStream_t s) {
        if (T > 1 && sh_nr) strata::kernels::iq_mmvq_rows(kIQ2_XXS, w, xq, out, n_in, n_out, T, sh_nr, s);
        else strata::kernels::iq_mmvq(kIQ2_XXS, w, xq, out, n_in, n_out, T, s);
    }
    // the same from inputs already in q8_1
    void iq2_q(const void* w, const void* xq, float* out, int n_in, int n_out, int T) {
        if (T > 1 && sh_nr) strata::kernels::iq_mmvq_rows(kIQ2_XXS, w, xq, out, n_in, n_out, T, sh_nr, stream);
        else strata::kernels::iq_mmvq(kIQ2_XXS, w, xq, out, n_in, n_out, T, stream);
    }
    // W (rows x n_in, BF16 or NVFP4) times T <= 16 rows of x into (T, rows): the skinny kernel, 8 columns a launch
    void skinny(int fmt, const void* w, const uint8_t* sc, float s2, const float* x, float* out, int n_in, int rows, int T) {
        int rb = rows / 256;                                       // aim for >= 256 blocks
        rb = std::max(16, std::min(kSkRB, rb / 16 * 16));
        const int grid = (rows + rb - 1) / rb;
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            const float* xc = x + size_t(c0) * n_in;
            float* yc = out + size_t(c0) * rows;
            if (fmt == 0) k_skinny<0><<<grid, 256, 0, stream>>>(w, nullptr, 1.f, xc, yc, n_in, rows, nc, rb);
            else k_skinny<1><<<grid, 256, 0, stream>>>(w, sc, s2, xc, yc, n_in, rows, nc, rb);
        }
    }
    bool use_skinny = std::getenv("NEMO_NO_SKINNY") == nullptr;

    // rows per warp at once in k_nvfp4_q8cols (1: 101 us a call, 2: 73, 4: 78)
    int nv_rw = std::getenv("NEMO_NV_RW") ? std::atoi(std::getenv("NEMO_NV_RW")) : 2;
    // zeroed: out is already zero (k_gated_norm_many's zero_out), so the slices need no fill first
    void nvfp4(const uint8_t* w, const uint8_t* s, float s2, const float* x, float* out, int n_in, int rows, int T, bool zeroed = false) {
        const int g = (rows + 7) / 8;
        // the skinny kernel measured slower than the q8 one for NVFP4 (40 vs 35 ms a check round): BF16 only
        if (T > 1 && use_skinny && std::getenv("NEMO_SKINNY_FP4") && n_in % 256 == 0) { skinny(1, w, s, s2, x, out, n_in, rows, T); return; }
        if (T == 1 || std::getenv("NEMO_OLD_COLS")) {
            if (T == 1) k_nvfp4_gemv2<<<g, 256, 0, stream>>>(w, s, s2, x, out, n_in, rows);
            else k_nvfp4_cols<8><<<g, 256, 0, stream>>>(w, s, s2, x, out, n_in, rows, T);
            return;
        }
        // inputs to q8_1 once, then the on-chip kernel: slices of 4096 add into a zeroed output
        strata::kernels::quantize_q8_1_rows(x, T, n_in, q8, stream);
        const int slices = n_in / 4096;
        if (slices > 1 && !zeroed) CK(cudaMemsetAsync(out, 0, size_t(T) * rows * 4, stream));
        // one block per compute unit fits (37 KB LDS): size the blocks so the launch is whole waves over the units
        static const int units = [] {
            if (const char* e = std::getenv("NEMO_UNITS")) return std::atoi(e);
            // RDNA reports work-group processors, two compute units each (and room for two of these blocks): 32 on the
            // R9700, where sizing for 64 measured best (32: 76 us a call, 64: 73, 128: 84)
            int d = 0, n = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n > 0 ? 2 * n : 64;
        }();
        const int per = (rows * slices * ((T + 7) / 8) + units - 1) / units;          // rows per block for one wave
        const int rpb = std::getenv("NEMO_RPB") ? std::atoi(std::getenv("NEMO_RPB")) : std::max(32, (per + 31) / 32 * 32);
        const dim3 grid((rows + rpb - 1) / rpb, slices, (T + 7) / 8);
        if (nv_rw == 4) k_nvfp4_q8cols<8, 4><<<grid, 1024, 0, stream>>>(w, s, s2, (const uint8_t*) q8, out, n_in, rows, T, slices > 1, rpb);
        else if (nv_rw == 2) k_nvfp4_q8cols<8, 2><<<grid, 1024, 0, stream>>>(w, s, s2, (const uint8_t*) q8, out, n_in, rows, T, slices > 1, rpb);
        else k_nvfp4_q8cols<8><<<grid, 1024, 0, stream>>>(w, s, s2, (const uint8_t*) q8, out, n_in, rows, T, slices > 1, rpb);
    }

    void mamba(Block& b, int T, int write_after) {
        float* proj = win_proj[b.win];
        float* xbc = win_xbc[b.win];
        nvfp4(b.in_w, b.in_s, b.in_s2, n, proj, kHidden, kInProj, T);
        k_mamba_conv_many<<<(kConvDim + 255) / 256, 256, 0, stream>>>(proj, b.conv_state, b.conv_w, b.conv_b, xbc, T, write_after);
        k_mamba_scan<<<dim3(kMHeads, kScanSlices), kMState, 0, stream>>>(xbc, proj, b.A_log, b.D, b.dt_bias, b.ssm, y, T, write_after);
        // (T > 1: the out_proj slices add into d, which the norm zeroes)
        k_gated_norm_many<<<dim3(kMGroups, T), 256, 0, stream>>>(y, proj, b.gnorm, T > 1 ? d : nullptr);
        nvfp4(b.out_w, b.out_s, b.out_s2, y, d, kMInter, kHidden, T, T > 1);
    }

    void attention(Block& b, int T) {
        q8_0(b.q, n, qkv, kHidden, kAHeads * kAHeadDim, T);
        q8_0(b.k, n, kbuf, kHidden, kAKv * kAHeadDim, T);
        q8_0(b.v, n, vbuf, kHidden, kAKv * kAHeadDim, T);
        k_kv_append_many<<<dim3(T, kAKv), kAHeadDim, 0, stream>>>(kbuf, vbuf, b.K, b.V, d_pos, kv_rows());
        // T <= 16 rows: one query tile, the keys present split into kDecSplits equal shares (an empty share is
        // weighted 0 by the combine)
        const int splits = kDecSplits;
        const size_t lds = size_t(T) * (5 * kAHeadDim + 8) * 4;
        if (T == 1) k_attn_dec<1><<<dim3(kAHeads, splits), 128, lds, stream>>>(qkv, b.K, b.V, att_sc, d_pos, kv_rows(), T, splits);
        else if (T <= 8) k_attn_dec<8><<<dim3(kAHeads, splits), 128, lds, stream>>>(qkv, b.K, b.V, att_sc, d_pos, kv_rows(), T, splits);
        else k_attn_dec<kTMax><<<dim3(kAHeads, splits), 128, lds, stream>>>(qkv, b.K, b.V, att_sc, d_pos, kv_rows(), T, splits);
        k_flash_combine<<<dim3(kAHeads, T), kAHeadDim, 0, stream>>>(att_sc, y, splits);
        q8_0(b.o, y, d, kAHeads * kAHeadDim, kHidden, T);
    }

    const float* extra = nullptr;     // a second output to add with d (the MoE's shared expert)

    // after block i: if it is one of the drafter's taps, its T rows go to column block (tap index) of dst
    void tap(size_t i, const float* src, float* dst, int T) {
        if (!dst) return;
        for (int t = 0; t < 6; ++t)
            if (int(i) == kTaps[t])
                CK(cudaMemcpy2DAsync(dst + size_t(t) * kHidden, size_t(kAux) * 4, src, size_t(kHidden) * 4, size_t(kHidden) * 4, T,
                                     cudaMemcpyDeviceToDevice, stream));
    }

    void moe(Block& b, int T) {
        const int k = b.topk, W = b.width;
        if (T == 1) k_bf16_cols<<<(b.n_exp + 7) / 8, 256, 0, stream>>>(b.router, n, scores, kHidden, b.n_exp, T);
        else if (use_skinny) skinny(0, b.router, nullptr, 1.f, n, scores, kHidden, b.n_exp, T);
        else k_bf16_rows<<<(b.n_exp + 7) / 8, 256, 0, stream>>>(b.router, n, scores, kHidden, b.n_exp, T);
        // the shared expert on the second stream, beside the routed path (they meet in the residual add): at these
        // sizes each product is a few us of ramp around a few us of reading, and two chains overlap those
        if (sh_par) {
            CK(cudaEventRecord(ev_fork, stream));
            CK(cudaStreamWaitEvent(stream2, ev_fork, 0));
            strata::kernels::quantize_q8_1_rows(n, T, kHidden, q8s, stream2);
            iq2_q_on(b.sh_up, q8s, tmp, kHidden, kShared, T, stream2);
            strata::kernels::relu2_quantize_q8_1_rows(tmp, T, kShared, q8s, stream2);
            iq2_q_on(b.sh_down, q8s, tmp2, kShared, kHidden, T, stream2);
            CK(cudaEventRecord(ev_join, stream2));
        }
        k_route_warp<<<(T + 7) / 8, 256, 0, stream>>>(scores, b.bias_dev, b.n_exp, k, T, route_idx, route_w);
        q8_0(b.fc1, n, lat, kHidden, kLatent, T);
        strata::kernels::quantize_q8_1_rows(lat, T, kLatent, q8, stream);
        if (T == 1) {
            strata::kernels::iq_mmvq_gather(kIQ2_XXS, b.up_tab, route_idx, k, q8, kLatent / 32, k, ex_up, kLatent, W, stream);
            strata::kernels::relu2_quantize_q8_1_rows(ex_up, k, W, q8b, stream);
            strata::kernels::iq_mmvq_gather(kIQ2_XXS, b.down_tab, route_idx, k, q8b, W / 32, 1, ex_down, W, kLatent, stream);
        } else {
            // the step's picks grouped by expert: each chosen expert is read once for every position that chose it
            // G: blocks across the groups (each block then loops over every G-th group, sharing its start-up)
            const int S = T * k, G = std::min({S, b.n_exp, grp_y});
            k_group_slots_par<<<1, 256, 0, stream>>>(route_idx, S, k, b.n_exp, g_n, g_exp, g_cnt, g_tok, g_slot);
            strata::kernels::iq_mmvq_grouped(kIQ2_XXS, b.up_tab, g_n, G, g_exp, g_cnt, g_tok, g_slot, q8, kLatent / 32, ex_up, kLatent, W, stream);
            strata::kernels::relu2_quantize_q8_1_rows(ex_up, S, W, q8b, stream);
            strata::kernels::iq_mmvq_grouped(kIQ2_XXS, b.down_tab, g_n, G, g_exp, g_cnt, g_slot, g_slot, q8b, W / 32, ex_down, W, kLatent, stream);
        }
        k_wsum_many<<<dim3(4, T), 256, 0, stream>>>(eo, ex_down, route_w, k, kLatent);
        q8_0(b.fc2, eo, d, kLatent, kHidden, T);
        // (llama.cpp's MMQ for these two at T = 8 measured slower: 46 vs 34 us a call)
        if (sh_par) CK(cudaStreamWaitEvent(stream, ev_join, 0));
        else {
            iq2(b.sh_up, n, tmp, kHidden, kShared, T);
            strata::kernels::relu2_quantize_q8_1_rows(tmp, T, kShared, q8, stream);
            iq2_q(b.sh_down, q8, tmp2, kShared, kHidden, T);
        }
        extra = tmp2;
    }

    // T positions from pos on the stream. write_after: positions after which the Mamba state is stored (T for a
    // prompt chunk or a plain step, 0 for a draft check). all_logits: every row's logits, else the last row's only.
    // dump: every block's output, rows in order (forces syncs, so never inside a graph).
    void body(int T, int write_after, bool all_logits, std::vector<float>* dump) {
        k_embed_many<<<dim3(16, T), 256, 0, stream>>>(embed, d_toks, h);
        k_rmsnorm_many<<<T, 1024, 0, stream>>>(h, blocks[0].norm, n, kHidden);
        for (size_t i = 0; i < blocks.size(); ++i) {
            Block& b = blocks[i];
            extra = nullptr;
            if (b.kind == MAMBA) mamba(b, T, write_after);
            else if (b.kind == MOE) moe(b, T);
            else attention(b, T);
            const uint16_t* nw = i + 1 < blocks.size() ? blocks[i + 1].norm : norm_f;
            k_add_rmsnorm_many<<<T, 1024, 0, stream>>>(h, d, extra, nw, n, kHidden);
            tap(i, h, aux, T);
            if (dump) {
                std::vector<float> rows(size_t(T) * kHidden);
                CK(cudaMemcpyAsync(rows.data(), h, rows.size() * 4, cudaMemcpyDeviceToHost, stream));
                CK(cudaStreamSynchronize(stream));
                dump->insert(dump->end(), rows.begin(), rows.end());
            }
        }
        if (all_logits) q8_0(head, n, logits, kHidden, kVocab, T);
        else q8_0(head, n + size_t(T - 1) * kHidden, logits, kHidden, kVocab, 1);
    }

    std::map<int, cudaGraphExec_t> graphs;   // key: T * 4 + (write_after == T) * 2 + all_logits
    bool use_graph = std::getenv("NEMO_NO_GRAPH") == nullptr;

    // run body() on the stream, recorded as a graph the first time (e) and replayed after; body must take every
    // per-call value from device memory, since the launches' own arguments are frozen at the recording
    template <class F> void graphed(cudaGraphExec_t& e, F&& body) {
        if (!use_graph) { body(); return; }
        if (!e) {
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(stream, &g));
            CK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
            CK(cudaGraphDestroy(g));
        }
        CK(cudaGraphLaunch(e, stream));
    }
    cudaGraphExec_t commit_graph[kTMax + 1] = {};

    // run tokens (up to kTMax) at pos; returns the logits rows (all of them, or the last); does not move pos
    std::vector<float> forward(const std::vector<int>& toks, int write_after, bool all_logits, std::vector<float>* dump = nullptr) {
        const int T = int(toks.size());
        if (T < 1 || T > kTMax) { std::fprintf(stderr, "strata-nemo: %d positions in one step (1..%d)\n", T, kTMax); std::exit(1); }
        if (pos + T > max_ctx) { std::fprintf(stderr, "strata-nemo: context full (%d)\n", max_ctx); std::exit(1); }
        for (int t = 0; t < T; ++t) h_tp[t] = toks[t];
        h_tp[kTMax] = pos;
        CK(cudaMemcpyAsync(d_toks, h_tp, (kTMax + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        if (dump || !use_graph) {
            body(T, write_after, all_logits, dump);
        } else {
            const int key = T * 4 + (write_after == T) * 2 + (all_logits ? 1 : 0);
            if (write_after != 0 && write_after != T) { std::fprintf(stderr, "strata-nemo: write_after %d\n", write_after); std::exit(1); }
            auto it = graphs.find(key);
            if (it == graphs.end()) {
                cudaGraph_t g;
                cudaGraphExec_t e;
                CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                body(T, write_after, all_logits, nullptr);
                CK(cudaStreamEndCapture(stream, &g));
                CK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
                it = graphs.emplace(key, e).first;
            }
            CK(cudaGraphLaunch(it->second, stream));
        }
        std::vector<float> out(size_t(all_logits ? T : 1) * kVocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        return out;
    }

    // like forward(..., all_logits), but only each row's best token comes back (the logits stay on the card)
    std::vector<int> forward_argmax(const std::vector<int>& toks, int write_after) {
        const int T = int(toks.size());
        for (int t = 0; t < T; ++t) h_tp[t] = toks[t];
        h_tp[kTMax] = pos;
        CK(cudaMemcpyAsync(d_toks, h_tp, (kTMax + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        const int key = T * 4 + (write_after == T) * 2 + 1;
        auto it = graphs.find(key);
        if (!use_graph) body(T, write_after, true, nullptr);
        else {
            if (it == graphs.end()) {
                cudaGraph_t g; cudaGraphExec_t e;
                CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                body(T, write_after, true, nullptr);
                CK(cudaStreamEndCapture(stream, &g));
                CK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
                it = graphs.emplace(key, e).first;
            }
            CK(cudaGraphLaunch(it->second, stream));
        }
        k_argmax_rows<<<T, 1024, 0, stream>>>(logits, kVocab, best_dev);
        std::vector<int> out(T);
        CK(cudaMemcpyAsync(out.data(), best_dev, T * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        return out;
    }
    int32_t* best_dev = nullptr;

    // after a draft check (write_after 0): move every Mamba state forward over the first m positions of that window
    void commit(int m) {
        graphed(commit_graph[m], [&] {
            for (auto& b : blocks) {
                if (b.kind != MAMBA) continue;
                float* proj = win_proj[b.win];
                float* xbc = win_xbc[b.win];
                k_mamba_commit<<<dim3(kMHeads, kScanSlices), kMState, 0, stream>>>(xbc, proj, b.A_log, b.dt_bias, b.ssm, b.conv_state, m);
            }
        });
        pos += m;
    }

    // ---------------------------------------------------------------- the prompt path (chunks of up to kTBig)
    struct Big {
        float *h, *n, *d, *proj, *xbc, *y, *lat, *eo, *tmp, *tmp2, *scores, *exu, *exd, *rw;
        uint16_t* xb;                 // BF16 activations for hipBLASLt (up to kTBig x kMInter)
        uint16_t *kb, *vtb, *qb, *P;  // attention: K, V^T, Q of one KV head and the probabilities, BF16
        uint16_t *kb2, *vtb2;         // k_attn_flash: K and V^T of both KV heads, BF16
        float *S, *O;
        float *ssd_dt, *ssd_acs, *ssd_ea, *ssd_cb, *ssd_S;   // the SSD Mamba path
        int32_t *ridx, *bounds, *src, *rowof, *ident, *one_bounds, *toks;
        int32_t* hbounds;             // pinned host copy of bounds
        int32_t* ccnt;                // the sort's per-chunk expert counts / offsets
        cudaEvent_t counted;          // bounds copied to hbounds
        int32_t *mtiles, *mtiles_n;   // the matrix-core expert kernel's work list
        uint8_t *w8 = nullptr, *x8 = nullptr;   // NEMO_FP8: E4M3 weight and activations, row scales and alpha, amax
        float* rscale = nullptr;
        uint32_t* amax = nullptr;
        // per-row FP8 activations (x8 with these row scales): of n (in_proj / shared up), of the gated norm's output
        // (out_proj), of relu^2 of the shared up output (shared down)
        float *rs_n = nullptr, *rs_y = nullptr, *rs_t = nullptr;
        // the FP8 weights of a chunk's products, converted on a side stream a few products ahead (w8_*)
        struct W8Job { bool iq2; const void* w; const uint8_t* sc; int rows, K; };
        struct W8Slot { uint8_t* w8; float* rscale; cudaEvent_t ready, free_; };
        static constexpr int kW8Slots = 3;
        W8Slot slot[kW8Slots];
        std::vector<W8Job> jobs;
        cudaStream_t side = nullptr;
        float* alpha = nullptr;
        int use = 0, issued = 0;
        bool pipe = false;
        uint16_t *lat16, *exh;        // its inputs in the scaled FP16 form, and their scales
        float *lats, *exs;
        int32_t* small_bounds;        // {0, T} for T <= kTMax, at 2 T
        void *xq, *xq2;
        strata::prefill::Gemm gemm;
        strata::prefill::mmq::Context mmq;
    };
    Big* B = nullptr;

    void init_big() {
        B = new Big();
        auto f = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        const size_t T = kTBig, S = T * 9;
        f(&B->h, T * kHidden); f(&B->n, T * kHidden); f(&B->d, T * kHidden); f(&B->proj, T * kInProj);
        f(&B->xbc, T * kConvDim); f(&B->y, T * kMInter); f(&B->lat, T * kLatent); f(&B->eo, T * kLatent);
        f(&B->tmp, T * kShared); f(&B->tmp2, T * kShared); f(&B->scores, T * 512);
        f(&B->exu, S * 2560); f(&B->exd, S * kLatent); f(&B->rw, S);
        CK(cudaMalloc(&B->xb, T * kAux * 2));
        {
            const size_t nc = (T + kSq - 1) / kSq;
            f(&B->ssd_dt, T * kMHeads); f(&B->ssd_acs, T * kMHeads); f(&B->ssd_ea, nc * kMHeads);
            f(&B->ssd_cb, size_t(kMGroups) * nc * kSq * kSq); f(&B->ssd_S, size_t(kMHeads) * nc * kMState * kMHeadDim);
        }         // BF16 activations up to kTBig x 24,576 (the drafter's fc)
        // the prompt attention products hold (16 T x max_ctx) scores: this caps the context they serve
        const size_t Lmax = size_t(max_ctx + 64);
        CK(cudaMalloc(&B->vtb2, kAKv * Lmax * kAHeadDim * 2));   // (the keys are read from the BF16 cache itself)
        if (att_old) {   // the two-GEMM attention's score tables (24 KB a context position) only when it runs
            CK(cudaMalloc(&B->kb, Lmax * kAHeadDim * 2)); CK(cudaMalloc(&B->vtb, Lmax * kAHeadDim * 2));
            CK(cudaMalloc(&B->qb, 16 * kAttSub * kAHeadDim * 2)); CK(cudaMalloc(&B->O, 16 * kAttSub * kAHeadDim * 4));
            CK(cudaMalloc(&B->S, 16 * kAttSub * Lmax * 4)); CK(cudaMalloc(&B->P, 16 * kAttSub * Lmax * 2));
        }
        CK(cudaMalloc(&B->ridx, S * 4)); CK(cudaMalloc(&B->bounds, 513 * 4)); CK(cudaMalloc(&B->src, S * 4));
        CK(cudaMalloc(&B->rowof, S * 4)); CK(cudaMalloc(&B->ident, S * 4)); CK(cudaMalloc(&B->one_bounds, 2 * 4));
        CK(cudaMalloc(&B->toks, kTBig * 4));
        CK(cudaMallocHost(&B->hbounds, 513 * 4));
        CK(cudaMalloc(&B->ccnt, (S / kSortChunk + 1) * 512 * 4));
        CK(cudaEventCreateWithFlags(&B->counted, cudaEventDisableTiming));
        CK(cudaMalloc(&B->mtiles, 2 * strata::kernels::moe_wmma_max_tiles(S, 512) * 4)); CK(cudaMalloc(&B->mtiles_n, 4));
        CK(cudaMalloc(&B->lat16, T * kLatent * 2)); f(&B->lats, T * kLatent / 128);
        CK(cudaMalloc(&B->exh, S * 2560 * 2)); f(&B->exs, S * 2560 / 128);
        {
            std::vector<int32_t> sb(2 * (kTMax + 1));
            for (int t = 0; t <= kTMax; ++t) { sb[2 * t] = 0; sb[2 * t + 1] = t; }
            CK(cudaMalloc(&B->small_bounds, sb.size() * 4));
            CK(cudaMemcpy(B->small_bounds, sb.data(), sb.size() * 4, cudaMemcpyHostToDevice));
        }
        strata::prefill::mmq::iota(B->ident, int64_t(S), stream);
        const size_t xqb = std::max(strata::prefill::mmq::q8_bytes(int64_t(S), 2560), strata::prefill::mmq::q8_bytes(int64_t(T), kShared));
        CK(cudaMalloc(&B->xq, xqb)); CK(cudaMalloc(&B->xq2, xqb));
        std::string err;
        if (!B->gemm.init(stream, std::max<int64_t>(int64_t(kInProj) * kHidden, int64_t(kHidden) * kAux), err)) { std::fprintf(stderr, "strata-nemo: %s\n", err.c_str()); std::exit(1); }
        if (!strata::prefill::mmq::built()) { std::fprintf(stderr, "strata-nemo: this build has no MMQ\n"); std::exit(1); }
        // the FP8 weight pipeline: slots, a side stream, and the products in the order the prompt pass runs them
        if (!B->w8) {
            CK(cudaMalloc(&B->w8, size_t(kInProj) * kHidden)); CK(cudaMalloc(&B->x8, size_t(kTBig) * kMInter));
            CK(cudaMalloc(&B->rscale, 2 * size_t(kInProj) * 4)); CK(cudaMalloc(&B->amax, 4));
        }
        CK(cudaMalloc(&B->alpha, size_t(kInProj) * 4));
        f(&B->rs_n, T); f(&B->rs_y, T); f(&B->rs_t, T);
        CK(cudaStreamCreateWithFlags(&B->side, cudaStreamNonBlocking));

        for (auto& sl : B->slot) {
            CK(cudaMalloc(&sl.w8, size_t(kInProj) * kHidden)); CK(cudaMalloc(&sl.rscale, size_t(kInProj) * 4));
            CK(cudaEventCreateWithFlags(&sl.ready, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&sl.free_, cudaEventDisableTiming));
        }
        for (auto& b : blocks) {
            if (b.kind == MAMBA) {
                if (fp8_mode == 1 || fp8_mode == 2) B->jobs.push_back({false, b.in_w, b.in_s, kInProj, kHidden});
                if (fp8_mode == 1 || fp8_mode == 3) B->jobs.push_back({false, b.out_w, b.out_s, kHidden, kMInter});
            } else if (b.kind == MOE && sh_fp8) {
                B->jobs.push_back({true, b.sh_up, nullptr, kShared, kHidden});
                B->jobs.push_back({true, b.sh_down, nullptr, kHidden, kShared});
            }
        }
        // one product now, so the library's first-use set-up (~0.4 s) is not paid inside the first prompt
        B->gemm.bf16(B->xb, B->gemm.scratch(), B->proj, kTBig, kInProj, kHidden, 0, 0.0f, 0, 1.0f);
        if (use_fp8) {   // and the FP8 products' (their first call cost ~0.45 s): both NVFP4 shapes at a full chunk
            CK(cudaMemsetAsync(B->n, 0, size_t(kTBig) * kHidden * 4, stream));
            CK(cudaMemsetAsync(B->y, 0, size_t(kTBig) * kMInter * 4, stream));
            for (auto& b : blocks)
                if (b.kind == MAMBA) {
                    nvfp4_fp8(b.in_w, b.in_s, b.in_s2, B->n, B->proj, kTBig, kInProj, kHidden);
                    nvfp4_fp8(b.out_w, b.out_s, b.out_s2, B->y, B->d, kTBig, kHidden, kMInter);
                    break;
                }
            // and the side-stream weight pipeline once: its first use cost ~11 ms of idle card in the first prompt
            if (use_w8_pipe && !B->jobs.empty()) {
                const Big::W8Job& j0 = B->jobs[0];
                float* X = j0.K == kHidden ? B->n : j0.K == kMInter ? B->y : B->tmp;
                float* Y = j0.rows == kInProj ? B->proj : j0.rows == kHidden ? B->d : B->tmp2;
                w8_begin(kTBig);
                if (j0.iq2) iq2_fp8(j0.w, X, Y, kTBig, j0.rows, j0.K, false);
                else nvfp4_fp8((const uint8_t*) j0.w, j0.sc, 1.f, X, Y, kTBig, j0.rows, j0.K);
                CK(cudaDeviceSynchronize());
            }
        }
        CK(cudaStreamSynchronize(stream));
    }
    // the chunk's slots sorted by expert into B->bounds / src / rowof (see k_sort_count)
    void sort_slots(const int32_t* idx, int S, int k, int n_exp) {
        const int nch = (S + kSortChunk - 1) / kSortChunk;
        k_sort_count<<<nch, 512, 0, stream>>>(idx, S, n_exp, B->ccnt);
        k_sort_offsets<<<1, 512, 0, stream>>>(B->ccnt, nch, n_exp, B->bounds);
        k_sort_place<<<nch, 512, 0, stream>>>(idx, S, k, B->ccnt, B->src, B->rowof);
    }

    // Y[T, N] = X[T, K] . W^T through MMQ, W one matrix of `type` (dense: one "expert" with all T rows)
    int one_bounds_t = -1;                                          // the T that B->one_bounds holds
    void mmq_dense(const void* W, int type, const float* X, float* Y, int T, int N, int K) {
        namespace mq = strata::prefill::mmq;
        mq::quantize(X, nullptr, B->xq2, type, K, K, T, stream);
        const int32_t* bounds = B->one_bounds;
        if (T <= kTMax) bounds = B->small_bounds + 2 * T;          // preset {0, T}: no host copy, so graphs can hold it
        else if (T != one_bounds_t) {                              // (once a chunk: each copy left a ~20 us gap)
            const int32_t hb[2] = {0, T};
            CK(cudaMemcpyAsync(B->one_bounds, hb, 8, cudaMemcpyHostToDevice, stream));
            one_bounds_t = T;
        }
        mq::Product p;
        p.w = W; p.type = type; p.w_rows = N; p.w_cols = K; p.expert_bytes = mq::matrix_bytes(type, N, K);
        p.n = 1; p.xq = B->xq2; p.bounds = bounds; p.ids = B->ident; p.total_rows = T; p.max_rows = T;
        p.dst = Y; p.ld_dst = N;
        B->mmq.run(p, stream);
    }
    // NVFP4 projection: unpacked to BF16 (exact) in the GEMM scratch, X to BF16, hipBLASLt times scale_2 (its alpha:
    // the same bits as scaling the fp32 product after, without the extra pass)
    // NEMO_FP8 (trial): the NVFP4 products in FP8 - the weight per row and the activations per chunk in E4M3,
    // hipBLASLt's FP8 product with alpha = row scale x activation scale x scale_2
    // NEMO_FP8 = 1 (default): both NVFP4 products of prompt chunks of 256+ rows in FP8, 2: in_proj only, 3: out_proj
    // only, 0: BF16. Measured (8K-token texts, next-token perplexity): README 4.5585 -> 4.5645, DETAILS.md 7.5497 ->
    // 7.5674 (+0.13%, +0.23%; top-1 66.03 -> 65.95%, 53.04 -> 53.14%); reading 8K 2,661 -> 3,064 tokens a second
    int fp8_mode = std::getenv("NEMO_FP8") ? std::atoi(std::getenv("NEMO_FP8")) : 1;
    bool use_fp8 = fp8_mode != 0;
    // the side stream's conversion of job j into its slot (after the slot's last product has read it)
    void w8_issue(int j) {
        const Big::W8Job& jb = B->jobs[j];
        Big::W8Slot& sl = B->slot[j % Big::kW8Slots];
        CK(cudaStreamWaitEvent(B->side, sl.free_, 0));
        if (jb.iq2) {
            strata::kernels::iq2xxs_e4m3_rows(jb.w, jb.rows, jb.K, sl.w8, sl.rscale, B->side);
        } else {
            k_nvfp4_to_e4m3_rows<<<jb.rows, 256, 0, B->side>>>((const uint8_t*) jb.w, jb.sc, jb.K, sl.w8, sl.rscale);
        }
        CK(cudaEventRecord(sl.ready, B->side));
    }
    // a chunk's prompt pass starts: the first few weights go to the side stream
    void w8_begin(int T) {
        B->pipe = T >= 256 && !B->jobs.empty();
        B->use = 0;
        if (!B->pipe) return;
        CK(cudaEventRecord(B->slot[0].free_, stream));            // (the side stream starts after the main stream's work so far)
        for (int k = 1; k < Big::kW8Slots; ++k) CK(cudaEventRecord(B->slot[k].free_, stream));
        for (B->issued = 0; B->issued < std::min<int>(Big::kW8Slots, int(B->jobs.size())); ++B->issued) w8_issue(B->issued);
    }
    // the slot holding weight w if it is the next one of the pipeline (else nullptr: convert in line)
    Big::W8Slot* w8_take(const void* w) {
        if (!B->pipe || B->use >= int(B->jobs.size()) || B->jobs[B->use].w != w) return nullptr;
        Big::W8Slot* sl = &B->slot[B->use % Big::kW8Slots];
        CK(cudaStreamWaitEvent(stream, sl->ready, 0));
        return sl;
    }
    // the product that read slot sl is queued: free it and convert the job kW8Slots ahead into it
    void w8_done(Big::W8Slot* sl) {
        CK(cudaEventRecord(sl->free_, stream));
        ++B->use;
        if (B->issued < int(B->jobs.size())) w8_issue(B->issued++);
    }
    // X null: B->x8 already holds the input rows in E4M3 with per-row scales (k_add_rmsnorm_rows / k_gated_norm_rows
    // / k_rows_e4m3), which the output's consumers apply; otherwise X is quantized here at one scale (amax / 448)
    void x8_from(const float* X, size_t n) {
        CK(cudaMemsetAsync(B->amax, 0, 4, stream));
        k_amax<<<1024, 256, 0, stream>>>(X, n, B->amax);
        k_to_e4m3<<<2048, 256, 0, stream>>>(X, n, B->amax, B->x8);
    }
    bool nvfp4_fp8(const uint8_t* w, const uint8_t* sc, float s2, const float* X, float* Y, int T, int rows, int K) {
        if (Big::W8Slot* sl = w8_take(w)) {
            if (X) x8_from(X, size_t(T) * K);
            k_fp8_alpha<<<(rows + 255) / 256, 256, 0, stream>>>(sl->rscale, rows, X ? B->amax : nullptr, s2, B->alpha);
            const bool ok = B->gemm.fp8(B->x8, sl->w8, B->alpha, Y, T, rows, K);
            w8_done(sl);
            if (!ok) { std::fprintf(stderr, "strata-nemo: no FP8 plan for a pipelined product\n"); std::exit(1); }
            return true;
        }
        if (!B->w8) {
            CK(cudaMalloc(&B->w8, size_t(kInProj) * kHidden)); CK(cudaMalloc(&B->x8, size_t(kTBig) * kMInter));
            CK(cudaMalloc(&B->rscale, 2 * size_t(kInProj) * 4)); CK(cudaMalloc(&B->amax, 4));
        }
        // (a product past the FP8 buffers - the drafter's fc, 24,576 wide - takes the BF16 path: it overran them)
        if (rows > kInProj || size_t(rows) * K > size_t(kInProj) * kHidden || size_t(T) * K > size_t(kTBig) * kMInter) return false;
        k_nvfp4_to_e4m3_rows<<<rows, 256, 0, stream>>>(w, sc, K, B->w8, B->rscale);
        if (X) x8_from(X, size_t(T) * K);
        float* alpha = B->rscale + kInProj;
        k_fp8_alpha<<<(rows + 255) / 256, 256, 0, stream>>>(B->rscale, rows, X ? B->amax : nullptr, s2, alpha);
        return B->gemm.fp8(B->x8, B->w8, alpha, Y, T, rows, K);
    }
    // the shared expert's IQ2_XXS products in FP8 (NEMO_SH_FP8): weight unpacked to BF16, then E4M3 per row; X in
    // E4M3 per chunk (relu^2 applied first, in place, when relu2)
    bool iq2_fp8(const void* w, float* X, float* Y, int T, int rows, int K, bool relu2) {
        if (Big::W8Slot* sl = w8_take(w)) {
            if (X) {
                CK(cudaMemsetAsync(B->amax, 0, 4, stream));
                const size_t n = size_t(T) * K;
                if (relu2) k_relu2_amax<<<1024, 256, 0, stream>>>(X, n, B->amax);
                else k_amax<<<1024, 256, 0, stream>>>(X, n, B->amax);
                k_to_e4m3<<<2048, 256, 0, stream>>>(X, n, B->amax, B->x8);
            }
            k_fp8_alpha<<<(rows + 255) / 256, 256, 0, stream>>>(sl->rscale, rows, X ? B->amax : nullptr, 1.f, B->alpha);
            const bool ok = B->gemm.fp8(B->x8, sl->w8, B->alpha, Y, T, rows, K);
            w8_done(sl);
            if (!ok) { std::fprintf(stderr, "strata-nemo: no FP8 plan for a pipelined product\n"); std::exit(1); }
            return true;
        }
        if (!B->w8) {
            CK(cudaMalloc(&B->w8, size_t(kInProj) * kHidden)); CK(cudaMalloc(&B->x8, size_t(kTBig) * kMInter));
            CK(cudaMalloc(&B->rscale, 2 * size_t(kInProj) * 4)); CK(cudaMalloc(&B->amax, 4));
        }
        strata::kernels::iq2xxs_e4m3_rows(w, rows, K, B->w8, B->rscale, stream);
        if (X) {
            CK(cudaMemsetAsync(B->amax, 0, 4, stream));
            const size_t n = size_t(T) * K;
            if (relu2) k_relu2_amax<<<1024, 256, 0, stream>>>(X, n, B->amax);
            else k_amax<<<1024, 256, 0, stream>>>(X, n, B->amax);
            k_to_e4m3<<<2048, 256, 0, stream>>>(X, n, B->amax, B->x8);
        }
        float* alpha = B->rscale + kInProj;
        k_fp8_alpha<<<(rows + 255) / 256, 256, 0, stream>>>(B->rscale, rows, X ? B->amax : nullptr, 1.f, alpha);
        return B->gemm.fp8(B->x8, B->w8, alpha, Y, T, rows, K);
    }
    // the shared expert in FP8 for chunks of 256+ rows (default; NEMO_SH_FP8=0: BF16). 8K texts: perplexity +0.07% /
    // +0.17% over the BF16 shared expert (all FP8 together vs all BF16: README 4.5585 -> 4.5710, DETAILS.md 7.5497 ->
    // 7.5795), reading 8K 3,392 -> 3,553 a second
    bool sh_fp8 = !std::getenv("NEMO_SH_FP8") || std::atoi(std::getenv("NEMO_SH_FP8")) != 0;
    // the FP8 weights converted on a side stream ahead of their products (NEMO_W8_PIPE=0: in line)
    bool use_w8_pipe = !std::getenv("NEMO_W8_PIPE") || std::atoi(std::getenv("NEMO_W8_PIPE")) != 0;
    void nvfp4_big(const uint8_t* w, const uint8_t* sc, float s2, const float* X, float* Y, int T, int rows, int K) {
        const bool fp8_here = T >= 256 && (fp8_mode == 1 || (fp8_mode == 2 && rows == kInProj) || (fp8_mode == 3 && rows == kHidden && K == kMInter));
        if (fp8_here && nvfp4_fp8(w, sc, s2, X, Y, T, rows, K)) return;
        const size_t nb = size_t(rows) * K / 2;
        k_nvfp4_to_bf16<<<2048, 256, 0, stream>>>(w, sc, B->gemm.scratch(), nb);
        k_to_bf16<<<1024, 256, 0, stream>>>(X, B->xb, size_t(T) * K);
        B->gemm.bf16(B->xb, B->gemm.scratch(), Y, T, rows, K, 0, 0.0f, 0, s2);   // scale_2 as the product's alpha
    }

    bool ssd_old = std::getenv("NEMO_SSD_OLD") != nullptr;
    bool att_old = std::getenv("NEMO_ATT_OLD") != nullptr;   // the two-GEMM prompt attention (before k_attn_flash)
    bool sh_mmq = std::getenv("NEMO_SH_MMQ") != nullptr;
    // routed experts on the FP16 matrix cores (strata::kernels::moe_wmma_iq2xxs); NEMO_MOE_MMQ = llama.cpp's MMQ.
    // 8K README: perplexity 4.5585 vs 4.5630, 2,675 vs 2,484 tokens a second
    bool moe_wmma = std::getenv("NEMO_MOE_MMQ") == nullptr;     // the shared expert through MMQ (before: BF16 GEMM)   // the three-kernel state hand-off, for comparison
    // which prompt products take per-row FP8 activations, written by their producers (NEMO_ROW_FP8=0: one scale a
    // chunk, quantized in front of each product as before)
    bool row_fp8 = !std::getenv("NEMO_ROW_FP8") || std::atoi(std::getenv("NEMO_ROW_FP8")) != 0;
    bool fp8_in(int T) const {
        return row_fp8 && T >= 256 && (fp8_mode == 1 || fp8_mode == 2) && !std::getenv("NEMO_SSD1") && !std::getenv("NEMO_SEQ_SCAN");
    }
    bool fp8_out(int T) const { return row_fp8 && T >= 256 && (fp8_mode == 1 || fp8_mode == 3); }
    bool fp8_sh(int T) const { return row_fp8 && sh_fp8 && !sh_mmq && T >= 256; }
    bool needs_x8(const Block& b, int T) const { return (b.kind == MAMBA && fp8_in(T)) || (b.kind == MOE && fp8_sh(T)); }
    const float* e_rs = nullptr;    // tmp2's row scales after a MoE block (null: true values)
    // NEMO_SSD_FP32=1: the fp32 SSD kernels instead of k_ssd_full
    bool ssd_fp32 = std::getenv("NEMO_SSD_FP32") && std::atoi(std::getenv("NEMO_SSD_FP32")) != 0;
    void mamba_big(Block& b, int T) {
        const bool fin = fp8_in(T), fout = fp8_out(T);
        const float* prs = fin ? B->rs_n : nullptr;                   // proj's row scales
        if (!fin) nvfp4_big(b.in_w, b.in_s, b.in_s2, B->n, B->proj, T, kInProj, kHidden);
        else if (!nvfp4_fp8(b.in_w, b.in_s, b.in_s2, nullptr, B->proj, T, kInProj, kHidden)) { std::fprintf(stderr, "strata-nemo: no FP8 in_proj\n"); std::exit(1); }
        k_mamba_conv_tiles<<<dim3((kConvDim + 255) / 256, (T + kConvTile - 1) / kConvTile), 256, 0, stream>>>(B->proj, b.conv_state, b.conv_w, b.conv_b, B->xbc, T, prs);
        k_conv_state_set<<<(kConvDim + 255) / 256, 256, 0, stream>>>(B->proj, b.conv_state, T, prs);
        // the step-by-step scan measured faster than this first SSD kernel (1,499 vs 1,065 a second on a 4K prompt)
        if (std::getenv("NEMO_SSD1")) k_mamba_ssd<<<dim3(kMHeads, kMHeadDim / 16), 256, 0, stream>>>(B->xbc, B->proj, b.A_log, b.D, b.dt_bias, b.ssm, B->y, T);
        else if (!std::getenv("NEMO_SEQ_SCAN") && T > 64) {
            // the Q-blocks in parallel, only the state hand-off in order (k_ssd_*)
            const int nc = (T + kSq - 1) / kSq;
            k_ssd_dt<<<dim3(kMHeads, nc), kSq, 0, stream>>>(B->proj, b.A_log, b.dt_bias, B->ssd_dt, B->ssd_acs, B->ssd_ea, T, prs);
            k_ssd_cb<<<dim3(kMGroups, nc), 256, 0, stream>>>(B->xbc, B->ssd_cb, T);
            if (ssd_old) {
                k_ssd_intra2<<<dim3(kMGroups, nc), 256, 0, stream>>>(B->xbc, B->ssd_cb, B->ssd_dt, B->ssd_acs, b.D, B->ssd_S, B->y, T, nc);
                k_ssd_pass<<<dim3(kMHeads, kMState * kMHeadDim / 256), 256, 0, stream>>>(B->ssd_S, B->ssd_ea, b.ssm, nc);
                k_ssd_out2<<<dim3(kMGroups, nc), 256, 0, stream>>>(B->xbc, B->ssd_acs, B->ssd_S, B->y, T, nc);
            } else {
                if (ssd_fp32) {
                    // fp32: the outputs' own part, then the state walk adding the carried-in part (half a head a
                    // block; a whole head at 512 threads measured slower)
                    k_ssd_intra2<<<dim3(kMGroups, nc), 256, 0, stream>>>(B->xbc, B->ssd_cb, B->ssd_dt, B->ssd_acs, b.D, B->ssd_S, B->y, T, nc, false);
                    k_ssd_state<32, 256><<<dim3(kMHeads, 2), 256, 0, stream>>>(B->xbc, B->ssd_dt, B->ssd_acs, B->ssd_ea, b.ssm, B->y, T, nc);
                } else {
                    // both parts in one pass on the BF16 matrix cores (fp32 values as high part + remainder)
                    k_ssd_full<2><<<kMHeads / 2, 256, 0, stream>>>(B->xbc, B->ssd_cb, B->ssd_dt, B->ssd_acs, B->ssd_ea, b.D, b.ssm, B->y, T, nc);
                }
            }
        }
        else k_mamba_scan<<<dim3(kMHeads, kScanSlices), kMState, 0, stream>>>(B->xbc, B->proj, b.A_log, b.D, b.dt_bias, b.ssm, B->y, T, T);
        if (fout) {
            // straight into E4M3 with the row scales (d's, applied by the next k_add_rmsnorm_rows)
            k_gated_norm_rows<<<T, 1024, 0, stream>>>(B->y, B->proj, prs, b.gnorm, B->x8, B->rs_y);
            if (!nvfp4_fp8(b.out_w, b.out_s, b.out_s2, nullptr, B->d, T, kHidden, kMInter)) { std::fprintf(stderr, "strata-nemo: no FP8 out_proj\n"); std::exit(1); }
        } else {
            k_gated_norm_many<<<dim3(kMGroups, T), 256, 0, stream>>>(B->y, B->proj, b.gnorm, nullptr, prs);
            nvfp4_big(b.out_w, b.out_s, b.out_s2, B->y, B->d, T, kHidden, kMInter);
        }
    }

    void attention_big(Block& b, int T) {
        float* q = B->tmp;                                           // (T, 4096)
        float* kk = B->tmp2;                                         // (T, 256)
        float* vv = B->tmp2 + size_t(kTBig) * kAKv * kAHeadDim;
        mmq_dense(b.q, 8, B->n, q, T, kAHeads * kAHeadDim, kHidden);
        mmq_dense(b.k, 8, B->n, kk, T, kAKv * kAHeadDim, kHidden);
        mmq_dense(b.v, 8, B->n, vv, T, kAKv * kAHeadDim, kHidden);
        k_kv_append_many<<<dim3(T, kAKv), kAHeadDim, 0, stream>>>(kk, vv, b.K, b.V, d_pos, kv_rows());
        const int Lp = (pos + T + 63) / 64 * 64;
        if (!att_old) {
            // one pass on the matrix cores per (KV head, 8 positions): the scores stay on the chip (k_attn_flash)
            const size_t kvs = size_t(max_ctx + 64) * kAHeadDim;
            // (the keys straight from the BF16 cache; V transposed for the product)
            for (int g = 0; g < kAKv; ++g)
                k_vt_bf16<<<1024, 256, 0, stream>>>(b.V + size_t(g) * kv_rows() * kAHeadDim, B->vtb2 + g * kvs, d_pos, T, Lp);
            k_attn_flash<<<dim3((T + kFwPos - 1) / kFwPos, kAKv), 256, 0, stream>>>(q, b.K, size_t(kv_rows()) * kAHeadDim, B->vtb2, kvs,
                                                                                     d_pos, T, Lp, B->y);
            mmq_dense(b.o, 8, B->y, B->d, T, kHidden, kAHeads * kAHeadDim);
            return;
        }
        // two products per KV head through hipBLASLt: S = Q K^T (16 T x L), causal softmax, O = P V
        for (int g = 0; g < kAKv; ++g) {
            const uint16_t* Kg = b.K + size_t(g) * kv_rows() * kAHeadDim;
            const uint16_t* Vg = b.V + size_t(g) * kv_rows() * kAHeadDim;
            k_k_bf16<<<1024, 256, 0, stream>>>(Kg, B->kb, d_pos, T, max_ctx, Lp);
            k_vt_bf16<<<1024, 256, 0, stream>>>(Vg, B->vtb, d_pos, T, Lp);
            for (int t0 = 0; t0 < T; t0 += kAttSub) {
                const int nt = std::min(kAttSub, T - t0);
                k_q_bf16<<<1024, 256, 0, stream>>>(q, B->qb, g, nt, t0);
                B->gemm.bf16(B->qb, B->kb, B->S, int64_t(16) * nt, Lp, kAHeadDim);
                k_softmax_causal<<<16 * nt, 256, 0, stream>>>(B->S, B->P, d_pos, nt, Lp, t0);
                B->gemm.bf16(B->P, B->vtb, B->O, int64_t(16) * nt, kAHeadDim, Lp);
                k_o_scatter<<<1024, 256, 0, stream>>>(B->O, B->y, g, nt, t0);
            }
        }
        mmq_dense(b.o, 8, B->y, B->d, T, kHidden, kAHeads * kAHeadDim);
    }

    // debug (NEMO_MW_CHECK): the matrix-core expert outputs (exu after relu^2 is applied inside the down call, so up's
    // raw output and the final exd) against llama.cpp's MMQ path on the same inputs, per row
    void mw_check(Block& b, int T, int S, int W) {
        namespace mq = strata::prefill::mmq;
        std::vector<float> u1(size_t(S) * W), d1(size_t(S) * kLatent), u2(u1.size()), d2(d1.size());
        {   // the matrix cores' up output: relu^2 in the scaled FP16 form -> fp32 (sqrt back for the comparisons below)
            std::vector<uint16_t> h(size_t(S) * W); std::vector<float> sc(size_t(S) * W / 128);
            CK(cudaStreamSynchronize(stream));
            CK(cudaMemcpy(h.data(), B->exh, h.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sc.data(), B->exs, sc.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < h.size(); ++i) { const float v = __half2float(__ushort_as_half(h[i])) * sc[i / 128]; u1[i] = std::sqrt(std::max(v, 0.f)); }
        }
        std::vector<int32_t> hb(b.n_exp + 1);
        CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(d1.data(), B->exd, d1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), B->bounds, hb.size() * 4, cudaMemcpyDeviceToHost));
        int maxr = 1;
        for (int e = 0; e < b.n_exp; ++e) maxr = std::max(maxr, hb[e + 1] - hb[e]);
        mq::quantize(B->lat, B->src, B->xq, kIQ2_XXS, kLatent, kLatent, S, stream);
        mq::Product up;
        up.w = b.up_all; up.type = kIQ2_XXS; up.w_rows = W; up.w_cols = kLatent; up.expert_bytes = b.up_bytes;
        up.n = b.n_exp; up.xq = B->xq; up.bounds = B->bounds; up.ids = B->ident; up.total_rows = S; up.max_rows = maxr;
        up.dst = B->exu; up.ld_dst = W;
        B->mmq.run(up, stream);
        CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(u2.data(), B->exu, u2.size() * 4, cudaMemcpyDeviceToHost));
        k_relu2<<<1024, 256, 0, stream>>>(B->exu, size_t(S) * W);
        mq::quantize(B->exu, nullptr, B->xq, kIQ2_XXS, W, W, S, stream);
        mq::Product dn = up;
        dn.w = b.down_all; dn.w_rows = kLatent; dn.w_cols = W; dn.expert_bytes = b.down_bytes; dn.dst = B->exd; dn.ld_dst = kLatent;
        B->mmq.run(dn, stream);
        CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(d2.data(), B->exd, d2.size() * 4, cudaMemcpyDeviceToHost));
        auto rel = [](const float* a, const float* b, int n) { double x = 0, y = 0; for (int i = 0; i < n; ++i) { x += double(a[i] - b[i]) * (a[i] - b[i]); y += double(b[i]) * b[i]; } return std::sqrt(x / (y + 1e-30)); };
        int bad = 0; double su = 0, sd = 0;
        for (int e = 0; e < b.n_exp; ++e)
            for (int r = hb[e]; r < hb[e + 1]; ++r) {
                const double eu = rel(&u1[size_t(r) * W], &u2[size_t(r) * W], W), ed = rel(&d1[size_t(r) * kLatent], &d2[size_t(r) * kLatent], kLatent);
                su += eu; sd += ed;
                if ((eu > 0.03 || ed > 0.05) && bad++ < 8) std::fprintf(stderr, "mw_check: expert %d row %d of %d: up %.4f down %.4f\n", e, r - hb[e], hb[e + 1] - hb[e], eu, ed);
            }
        std::fprintf(stderr, "mw_check: W %d, %d rows: mean up %.4f, down %.4f, %d rows off\n", W, S, su / S, sd / S, bad);
        // exact down products (fp64 on the host, each path's own up output as the input) for the first 48 rows
        static int layer = 0;
        if (layer++ < 3) {
            std::vector<float> wd(size_t(kLatent) * W);
            float* dw = nullptr;
            CK(cudaMalloc(&dw, wd.size() * 4));
            double e1 = 0, e2 = 0; int n = 0;
            for (int e = 0; e < b.n_exp && n < 48; ++e) {
                if (hb[e + 1] == hb[e]) continue;
                strata::kernels::iq_dequant_f32(kIQ2_XXS, (const uint8_t*) b.down_all + size_t(e) * b.down_bytes, int64_t(kLatent) * W, dw, stream);
                CK(cudaStreamSynchronize(stream));
                CK(cudaMemcpy(wd.data(), dw, wd.size() * 4, cudaMemcpyDeviceToHost));
                for (int r = hb[e]; r < hb[e + 1] && n < 48; ++r, ++n) {
                    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                    for (int o = 0; o < kLatent; ++o) {
                        double r1 = 0, r2 = 0;
                        for (int q = 0; q < W; ++q) {
                            const double a1 = std::max(0.f, u1[size_t(r) * W + q]), a2 = std::max(0.f, u2[size_t(r) * W + q]);
                            r1 += double(wd[size_t(o) * W + q]) * a1 * a1;
                            r2 += double(wd[size_t(o) * W + q]) * a2 * a2;
                        }
                        x1 += (d1[size_t(r) * kLatent + o] - r1) * (d1[size_t(r) * kLatent + o] - r1); y1 += r1 * r1;
                        x2 += (d2[size_t(r) * kLatent + o] - r2) * (d2[size_t(r) * kLatent + o] - r2); y2 += r2 * r2;
                    }
                    e1 += std::sqrt(x1 / (y1 + 1e-30)); e2 += std::sqrt(x2 / (y2 + 1e-30));
                }
            }
            CK(cudaFree(dw));
            std::fprintf(stderr, "mw_check: exact down, %d rows: matrix cores %.5f, MMQ %.5f (mean relative error)\n", n, e1 / n, e2 / n);
        }
    }

    void moe_big(Block& b, int T) {
        namespace mq = strata::prefill::mmq;
        e_rs = nullptr;
        const int k = b.topk, W = b.width, S = T * k;
        k_to_bf16<<<1024, 256, 0, stream>>>(B->n, B->xb, size_t(T) * kHidden);
        // the router through hipBLASLt (its weights are BF16 already)
        B->gemm.bf16(B->xb, b.router, B->scores, T, b.n_exp, kHidden);
        k_route_warp<<<(T + 7) / 8, 256, 0, stream>>>(B->scores, b.bias_dev, b.n_exp, k, T, B->ridx, B->rw);
        sort_slots(B->ridx, S, k, b.n_exp);
        if (!moe_wmma) {
            CK(cudaMemcpyAsync(B->hbounds, B->bounds, (b.n_exp + 1) * 4, cudaMemcpyDeviceToHost, stream));
            CK(cudaEventRecord(B->counted, stream));
        }
        // the work that does not need the counts goes in before the host waits for them, so the card is not left
        // idle while the host launches the rest (a stream sync here cost ~0.25 ms of idle card a layer)
        mmq_dense(b.fc1, 8, B->n, B->lat, T, kLatent, kHidden);
        if (!moe_wmma) mq::quantize(B->lat, B->src, B->xq, kIQ2_XXS, kLatent, kLatent, S, stream);
        if (sh_mmq) {
            mmq_dense(b.sh_up, kIQ2_XXS, B->n, B->tmp, T, kShared, kHidden);
            k_relu2<<<1024, 256, 0, stream>>>(B->tmp, size_t(T) * kShared);
            mmq_dense(b.sh_down, kIQ2_XXS, B->tmp, B->tmp2, T, kHidden, kShared);
        } else if (fp8_sh(T) && iq2_fp8(b.sh_up, nullptr, B->tmp, T, kShared, kHidden, false)) {
            // per-row FP8: n's rows came from k_add_rmsnorm_rows; relu^2 of tmp (times those scales) into E4M3 per row
            k_rows_e4m3<kShared, 448, true><<<T, 448, 0, stream>>>(B->tmp, B->rs_n, B->x8, B->rs_t);
            if (!iq2_fp8(b.sh_down, nullptr, B->tmp2, T, kHidden, kShared, true)) { std::fprintf(stderr, "strata-nemo: shared FP8\n"); std::exit(1); }
            e_rs = B->rs_t;
        } else if (!row_fp8 && sh_fp8 && T >= 256 && iq2_fp8(b.sh_up, B->n, B->tmp, T, kShared, kHidden, false)) {
            if (!iq2_fp8(b.sh_down, B->tmp, B->tmp2, T, kHidden, kShared, true)) { std::fprintf(stderr, "strata-nemo: shared FP8\n"); std::exit(1); }
        } else {
            // the shared expert unpacked to BF16 and through the BLAS GEMM (B->xb still holds n in BF16, from the
            // router): MMQ reached ~85 int8 TOPS on these two, the BF16 GEMM ~150 TFLOPS
            strata::kernels::iq_dequant_bf16(kIQ2_XXS, b.sh_up, int64_t(kShared) * kHidden, B->gemm.scratch(), stream);
            B->gemm.bf16(B->xb, B->gemm.scratch(), B->tmp, T, kShared, kHidden);
            k_relu2_to_bf16<<<1024, 256, 0, stream>>>(B->tmp, B->xb, size_t(T) * kShared);
            strata::kernels::iq_dequant_bf16(kIQ2_XXS, b.sh_down, int64_t(kHidden) * kShared, B->gemm.scratch(), stream);
            B->gemm.bf16(B->xb, B->gemm.scratch(), B->tmp2, T, kHidden, kShared);
        }
        if (moe_wmma) {
            // the routed experts on the matrix cores, their work list built on the card (no host wait at all)
            // (inputs in a scaled FP16 form: the latent rows once, the up outputs written in it with relu^2 applied)
            strata::kernels::moe_x16_rows(B->lat, int64_t(T) * kLatent, B->lat16, B->lats, stream);
            strata::kernels::moe_wmma_iq2xxs(b.up_all, b.up_bytes, b.n_exp, W, kLatent, B->lat16, B->lats, B->src, B->bounds, S,
                                             B->mtiles, B->mtiles_n, true, nullptr, B->exh, B->exs, W, stream);
            strata::kernels::moe_wmma_iq2xxs(b.down_all, b.down_bytes, b.n_exp, kLatent, W, B->exh, B->exs, nullptr, B->bounds, S,
                                             B->mtiles, B->mtiles_n, false, B->exd, nullptr, nullptr, kLatent, stream);
            if (std::getenv("NEMO_MW_CHECK")) mw_check(b, T, S, W);   // debug: against the MMQ path, then on with MMQ's
            k_wsum_sorted<<<dim3(4, T), 256, 0, stream>>>(B->eo, B->exd, B->rowof, B->rw, k, kLatent);
            mmq_dense(b.fc2, 8, B->eo, B->d, T, kHidden, kLatent);
            return;
        }
        CK(cudaEventSynchronize(B->counted));
        int maxr = 1;
        for (int e = 0; e < b.n_exp; ++e) maxr = std::max(maxr, B->hbounds[e + 1] - B->hbounds[e]);
        mq::Product up;
        up.w = b.up_all; up.type = kIQ2_XXS; up.w_rows = W; up.w_cols = kLatent; up.expert_bytes = b.up_bytes;
        up.n = b.n_exp; up.xq = B->xq; up.bounds = B->bounds; up.ids = B->ident; up.total_rows = S; up.max_rows = maxr;
        up.dst = B->exu; up.ld_dst = W;
        B->mmq.run(up, stream);
        k_relu2<<<1024, 256, 0, stream>>>(B->exu, size_t(S) * W);
        mq::quantize(B->exu, nullptr, B->xq, kIQ2_XXS, W, W, S, stream);
        mq::Product dn;
        dn.w = b.down_all; dn.type = kIQ2_XXS; dn.w_rows = kLatent; dn.w_cols = W; dn.expert_bytes = b.down_bytes;
        dn.n = b.n_exp; dn.xq = B->xq; dn.bounds = B->bounds; dn.ids = B->ident; dn.total_rows = S; dn.max_rows = maxr;
        dn.dst = B->exd; dn.ld_dst = kLatent;
        B->mmq.run(dn, stream);
        k_wsum_sorted<<<dim3(4, T), 256, 0, stream>>>(B->eo, B->exd, B->rowof, B->rw, k, kLatent);
        mmq_dense(b.fc2, 8, B->eo, B->d, T, kHidden, kLatent);
    }

    // --ppl: the text's ids on the card, their count, and the sums (negative log-likelihood, rows, top-1 hits)
    const int32_t* ppl_ids = nullptr;
    int ppl_n = 0;
    double* ppl_acc = nullptr;

    // T (<= kTBig) prompt positions at pos, the state stored after them; returns the last position's logits
    std::vector<float> forward_big(const std::vector<int>& toks, std::vector<float>* dump = nullptr) {
        if (!B) init_big();
        const int T = int(toks.size());
        if (pos + T > max_ctx) { std::fprintf(stderr, "strata-nemo: context full (%d)\n", max_ctx); std::exit(1); }
        int* dt = B->toks;
        CK(cudaMemcpyAsync(dt, toks.data(), T * 4, cudaMemcpyHostToDevice, stream));
        h_tp[kTMax] = pos;
        CK(cudaMemcpyAsync(d_pos, h_tp + kTMax, 4, cudaMemcpyHostToDevice, stream));
        k_embed_many<<<dim3(16, T), 256, 0, stream>>>(embed, dt, B->h);
        k_rmsnorm_many<<<T, 1024, 0, stream>>>(B->h, blocks[0].norm, B->n, kHidden);
        if (needs_x8(blocks[0], T)) k_rows_e4m3<kHidden, 256, false><<<T, 256, 0, stream>>>(B->n, nullptr, B->x8, B->rs_n);
        if (use_w8_pipe) w8_begin(T);
        for (size_t i = 0; i < blocks.size(); ++i) {
            Block& b = blocks[i];
            const float* e = nullptr;
            if (b.kind == MAMBA) mamba_big(b, T);
            else if (b.kind == MOE) { moe_big(b, T); e = B->tmp2; }
            else attention_big(b, T);
            const uint16_t* nw = i + 1 < blocks.size() ? blocks[i + 1].norm : norm_f;
            // the next block's input: in E4M3 per row for its FP8 product, in fp32 unless only that product reads it
            const Block* nx = i + 1 < blocks.size() ? &blocks[i + 1] : nullptr;
            const bool x8 = nx && needs_x8(*nx, T), keep_n = !(x8 && nx->kind == MAMBA);
            const float* drs = b.kind == MAMBA && fp8_out(T) ? B->rs_y : nullptr;
            k_add_rmsnorm_rows<<<T, 256, 0, stream>>>(B->h, B->d, drs, e, b.kind == MOE ? e_rs : nullptr, nw, keep_n ? B->n : nullptr,
                                                       x8 ? B->x8 : nullptr, B->rs_n);
            tap(i, B->h, aux_big, T);
            if (dump) {
                std::vector<float> rows(size_t(T) * kHidden);
                CK(cudaMemcpyAsync(rows.data(), B->h, rows.size() * 4, cudaMemcpyDeviceToHost, stream));
                CK(cudaStreamSynchronize(stream));
                dump->insert(dump->end(), rows.begin(), rows.end());
            }
        }
        if (ppl_ids) {   // quality measure: every row's log-probability of the text's next token (chunks of 8 rows)
            for (int t0 = 0; t0 < T; t0 += 8) {
                const int nt = std::min(8, T - t0);
                q8_0(head, B->n + size_t(t0) * kHidden, logits, kHidden, kVocab, nt);
                k_nll_rows<<<nt, 1024, 0, stream>>>(logits, kVocab, ppl_ids + pos + t0 + 1, ppl_n - (pos + t0 + 1), ppl_acc);
            }
        }
        if (B->pipe && B->use != int(B->jobs.size())) { std::fprintf(stderr, "strata-nemo: the FP8 pipeline ran %d of %zu products\n", B->use, B->jobs.size()); std::exit(1); }
        B->pipe = false;
        std::vector<float> out;
        if (dump) {   // every row's logits, for the comparison
            out.resize(size_t(T) * kVocab);
            for (int t0 = 0; t0 < T; t0 += 8) {
                const int nt = std::min(8, T - t0);
                q8_0(head, B->n + size_t(t0) * kHidden, logits, kHidden, kVocab, nt);
                CK(cudaMemcpyAsync(out.data() + size_t(t0) * kVocab, logits, size_t(nt) * kVocab * 4, cudaMemcpyDeviceToHost, stream));
            }
        } else {
            out.resize(kVocab);
            q8_0(head, B->n + size_t(T - 1) * kHidden, logits, kHidden, kVocab, 1);
            CK(cudaMemcpyAsync(out.data(), logits, kVocab * 4, cudaMemcpyDeviceToHost, stream));
        }
        CK(cudaStreamSynchronize(stream));
        return out;
    }

    // a prompt: chunks of kTMax, state stored after each; returns the last position's logits
    std::vector<float> read_prompt(const std::vector<int>& ids, std::vector<float>* dump = nullptr) {
        std::vector<float> lg;
        for (size_t i = 0; i < ids.size(); i += kTMax) {
            const std::vector<int> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + kTMax));
            lg = forward(c, int(c.size()), dump != nullptr, dump);
            if (dump) lg.erase(lg.begin(), lg.end() - kVocab);
            pos += int(c.size());
        }
        return lg;
    }
};


// ---------------------------------------------------------------------------------------------------- the drafter
struct Drafter {
    Model& M;
    Store st;
    cudaStream_t stream;
    struct Nv { const uint8_t *w = nullptr, *s = nullptr; float s2 = 0; const uint16_t* b = nullptr; int n_in = 0, n_out = 0; };
    struct Layer {
        const uint16_t *in_ln, *post_ln, *ac_base, *ac_proj, *mc_base, *mc_proj, *q, *k, *v, *o, *q_norm, *k_norm, *sink;
        Nv gate, up, down;
        float *K, *V;
    };
    Nv fc;
    const uint16_t *hidden_norm, *norm;
    Layer L[kDLayers];
    // LiLiCorr
    Nv token_proj, pass_proj, context_proj, l_out[2], l_up[2], l_down[2];
    const uint16_t *fn_w, *fn_b, *fm_up_w, *fm_up_b, *fm_down_w, *fm_down_b, *slot_emb, *rank_emb;
    const uint16_t *l_attn_norm[2], *l_mlp_norm[2], *l_in_w[2], *l_in_b[2], *out_norm, *anchor_norm;
    const uint16_t *f_self, *f_anchor, *f_cross, *f_bias, *out_head_w, *out_head_b, *in_head_w, *in_head_b, *aout_w, *aout_b;
    float* att_bias;                   // (8 heads, 56, 56)
    // buffers
    float *comb, *xn, *kt, *vt, *anchor;                 // context insertion (up to kTBig rows)
    float *h, *res, *x, *xa, *coef, *q, *k, *v, *att, *o, *g, *u, *act, *dn, *out;
    int32_t *ids, *cand, *drafts;
    float *lp, *tokE, *tok, *pass, *feat, *featd, *hs, *lx, *qkv, *latt, *lo, *lu, *hsn, *an_raw, *an, *pre1, *pre2, *ar, *ov, *iv, *aout;
    int *d_p0, *h_p0;
    int32_t* h_ids;
    bool dumped = false;
    cudaGraphExec_t draft_graph = nullptr, ctx_graph[kTMax + 1] = {};

    Nv nv(const std::string& name, bool bias) {
        Nv r;
        Tensor& t = st.at(name + ".weight");
        if (t.fmt != "NVFP4") { std::fprintf(stderr, "strata-nemo: %s is %s, not NVFP4\n", name.c_str(), t.fmt.c_str()); std::exit(1); }
        r.w = (const uint8_t*) st.dev(name + ".weight");
        r.s = (const uint8_t*) st.dev(name + ".weight_scale");
        r.s2 = st.f32(name + ".weight_scale_2");
        r.n_out = int(t.shape[0]); r.n_in = int(t.shape[1]) * 2;
        if (bias) r.b = (const uint16_t*) st.dev(name + ".bias");
        return r;
    }
    const uint16_t* bf(const std::string& name) { return (const uint16_t*) st.dev(name); }

    Drafter(Model& m, const std::string& dir) : M(m), st(dir), stream(m.stream) {
        fc = nv("fc", false);
        hidden_norm = bf("hidden_norm.weight"); norm = bf("norm.weight");
        for (int l = 0; l < kDLayers; ++l) {
            const std::string P = "layers." + std::to_string(l) + ".";
            Layer& y = L[l];
            y.in_ln = bf(P + "input_layernorm.weight"); y.post_ln = bf(P + "post_attention_layernorm.weight");
            y.ac_base = bf(P + "attention_conv.base_kernel"); y.ac_proj = bf(P + "attention_conv.kernel_projection.weight");
            y.mc_base = bf(P + "mlp_conv.base_kernel"); y.mc_proj = bf(P + "mlp_conv.kernel_projection.weight");
            y.q = bf(P + "self_attn.q_proj.weight"); y.k = bf(P + "self_attn.k_proj.weight");
            y.v = bf(P + "self_attn.v_proj.weight"); y.o = bf(P + "self_attn.o_proj.weight");
            y.q_norm = bf(P + "self_attn.q_norm.weight"); y.k_norm = bf(P + "self_attn.k_norm.weight");
            y.sink = bf(P + "self_attn.attention_sink_bias");
            y.gate = nv(P + "mlp.gate_proj", false); y.up = nv(P + "mlp.up_proj", false); y.down = nv(P + "mlp.down_proj", false);
            CK(cudaMalloc(&y.K, sizeof(float) * kAKv * size_t(kDRing) * 128));   // (rings: see kDRing)
            CK(cudaMalloc(&y.V, sizeof(float) * kAKv * size_t(kDRing) * 128));
        }
        const std::string C = "lilicorr.";
        token_proj = nv(C + "token_proj", true); pass_proj = nv(C + "pass_hidden_proj", true); context_proj = nv(C + "context_proj", true);
        fn_w = bf(C + "feature_norm.weight"); fn_b = bf(C + "feature_norm.bias");
        fm_up_w = bf(C + "feature_mlp.up_proj.weight"); fm_up_b = bf(C + "feature_mlp.up_proj.bias");
        fm_down_w = bf(C + "feature_mlp.down_proj.weight"); fm_down_b = bf(C + "feature_mlp.down_proj.bias");
        slot_emb = bf(C + "slot_embedding"); rank_emb = bf(C + "rank_embedding");
        for (int i = 0; i < 2; ++i) {
            const std::string P = C + "layers." + std::to_string(i) + ".";
            l_attn_norm[i] = bf(P + "attn_norm.weight"); l_mlp_norm[i] = bf(P + "mlp_norm.weight");
            l_in_w[i] = bf(P + "attn.in_proj_weight"); l_in_b[i] = bf(P + "attn.in_proj_bias");
            l_out[i] = nv(P + "attn.out_proj", true); l_up[i] = nv(P + "mlp.up_proj", true); l_down[i] = nv(P + "mlp.down_proj", true);
        }
        out_norm = bf(C + "output_norm.weight"); anchor_norm = bf(C + "anchor_norm.weight");
        out_head_w = bf(C + "out_head.weight"); out_head_b = bf(C + "out_head.bias");
        in_head_w = bf(C + "in_head.weight"); in_head_b = bf(C + "in_head.bias");
        aout_w = bf(C + "anchor_out_head.weight"); aout_b = bf(C + "anchor_out_head.bias");
        f_bias = bf(C + "factor_input_proj.bias");
        {   // factor_input_proj (1024 x 3072) split into its three 1024-column blocks, each its own matrix
            Tensor& t = st.at(C + "factor_input_proj.weight");
            std::vector<uint16_t> w(t.bytes / 2);
            const int fd = ::open((st.dir + "/" + t.file).c_str(), O_RDONLY);
            if (::pread(fd, w.data(), t.bytes, off_t(t.off)) != ssize_t(t.bytes)) { std::fprintf(stderr, "factor read\n"); std::exit(1); }
            ::close(fd);
            const uint16_t** dst[3] = {&f_self, &f_anchor, &f_cross};
            for (int b = 0; b < 3; ++b) {
                std::vector<uint16_t> part(size_t(kLl) * kLl);
                for (int r = 0; r < kLl; ++r) std::memcpy(&part[size_t(r) * kLl], &w[size_t(r) * 3 * kLl + b * kLl], kLl * 2);
                uint16_t* p = nullptr;
                CK(cudaMalloc(&p, part.size() * 2));
                CK(cudaMemcpy(p, part.data(), part.size() * 2, cudaMemcpyHostToDevice));
                *dst[b] = p;
            }
        }
        {   // the lattice attention bias: relative_slot_bias[h][s_q - s_k + 7] + same_slot_bias[h] * (s_q == s_k)
            auto rd = [&](const std::string& n) {
                Tensor& t = st.at(n);
                std::vector<uint16_t> w(t.bytes / 2);
                const int fd = ::open((st.dir + "/" + t.file).c_str(), O_RDONLY);
                if (::pread(fd, w.data(), t.bytes, off_t(t.off)) != ssize_t(t.bytes)) { std::fprintf(stderr, "bias read\n"); std::exit(1); }
                ::close(fd);
                std::vector<float> f(w.size());
                for (size_t i = 0; i < w.size(); ++i) { uint32_t u = uint32_t(w[i]) << 16; std::memcpy(&f[i], &u, 4); }
                return f;
            };
            const auto rel = rd(C + "relative_slot_bias"), same = rd(C + "same_slot_bias");
            const int R = kDSlots * kDTopK;
            std::vector<float> b(size_t(8) * R * R);
            for (int hh = 0; hh < 8; ++hh)
                for (int a = 0; a < R; ++a)
                    for (int c = 0; c < R; ++c) {
                        const int sa = a / kDTopK, sc = c / kDTopK;
                        b[(size_t(hh) * R + a) * R + c] = rel[hh * (2 * kDBlock - 1) + (sa - sc + kDBlock - 1)] + (sa == sc ? same[hh] : 0.f);
                    }
            CK(cudaMalloc(&att_bias, b.size() * 4));
            CK(cudaMemcpy(att_bias, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
        }
        auto f = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        const size_t RB = kTBig, R8 = kDBlock, RL = kDSlots * kDTopK;
        f(&comb, RB * kHidden); f(&xn, RB * kHidden); f(&kt, RB * 256); f(&vt, RB * 256); f(&anchor, kHidden);
        f(&h, R8 * kHidden); f(&res, R8 * kHidden); f(&x, R8 * kHidden); f(&xa, R8 * kHidden); f(&coef, R8 * 1024);
        f(&q, R8 * kHidden); f(&k, R8 * 256); f(&v, R8 * 256); f(&att, R8 * kHidden); f(&o, R8 * kHidden);
        f(&g, R8 * kDFf); f(&u, R8 * kDFf); f(&act, R8 * kDFf); f(&dn, R8 * kHidden); f(&out, R8 * kHidden);
        f(&lp, RL); f(&tokE, RL * kHidden); f(&tok, RL * kLl); f(&pass, kDSlots * kLl); f(&feat, RL * kLl); f(&featd, RL * kLl);
        f(&hs, RL * kLl); f(&lx, RL * kLl); f(&qkv, RL * 3 * kLl); f(&latt, RL * kLl); f(&lo, RL * kLl); f(&lu, RL * 2 * kLl);
        f(&hsn, RL * kLl); f(&an_raw, kLl); f(&an, kLl); f(&pre1, RL * kLl); f(&pre2, RL * kLl); f(&ar, kLl);
        f(&ov, RL * kLl); f(&iv, RL * kLl); f(&aout, kLl);
        CK(cudaMalloc(&ids, kDBlock * 4)); CK(cudaMalloc(&cand, RL * 4)); CK(cudaMalloc(&drafts, kDSlots * 4));
        CK(cudaMalloc(&d_p0, 8)); CK(cudaMallocHost(&h_p0, 8)); CK(cudaMallocHost(&h_ids, kDBlock * 4));
        CK(cudaMalloc(&M.aux, size_t(kTMax) * kAux * 4));
        CK(cudaMalloc(&M.aux_big, size_t(kTBig) * kAux * 4));
        std::fprintf(stderr, "strata-nemo: drafter %.2f GB\n", st.loaded / 1e9);
    }

    // above lib_rows rows the products go to hipBLASLt (the small-step kernels re-read the weights every 8-16 rows)
    int lib_rows = std::getenv("NEMO_LIB_ROWS") ? std::atoi(std::getenv("NEMO_LIB_ROWS")) : 16;
    // Y[R, n_out] = X[R, n_in] . W^T (+ bias), W bf16
    void lin_bf16(const uint16_t* W, const float* X, float* Y, int R, int n_in, int n_out, const uint16_t* bias) {
        if (R > lib_rows && M.B && n_in % 64 == 0) {
            k_to_bf16<<<1024, 256, 0, stream>>>(X, M.B->xb, size_t(R) * n_in);
            M.B->gemm.bf16(M.B->xb, W, Y, R, n_out, n_in);
        } else {
            for (int r0 = 0; r0 < R; r0 += kTMax) {
                const int nr = std::min(kTMax, R - r0);
                if (M.use_skinny && n_in % 256 == 0 && nr > 1) M.skinny(0, W, nullptr, 1.f, X + size_t(r0) * n_in, Y + size_t(r0) * n_out, n_in, n_out, nr);
                else k_bf16_rows<<<(n_out + 7) / 8, 256, 0, stream>>>(W, X + size_t(r0) * n_in, Y + size_t(r0) * n_out, n_in, n_out, nr);
            }
        }
        if (bias) k_add_bias<<<dim3((n_out + 255) / 256, R), 256, 0, stream>>>(Y, bias, n_out);
    }
    // the same for an NVFP4 matrix
    void lin_nv(const Nv& W, const float* X, float* Y, int R) {
        if (R > lib_rows && M.B) M.nvfp4_big(W.w, W.s, W.s2, X, Y, R, W.n_out, W.n_in);
        else
            for (int r0 = 0; r0 < R; r0 += kTMax) {
                const int nr = std::min(kTMax, R - r0);
                const float* xr = X + size_t(r0) * W.n_in;
                float* yr = Y + size_t(r0) * W.n_out;
                if (W.n_in % 4096 == 0) M.nvfp4(W.w, W.s, W.s2, xr, yr, W.n_in, W.n_out, nr);
                else k_nvfp4_cols<8><<<(W.n_out + 7) / 8, 256, 0, stream>>>(W.w, W.s, W.s2, xr, yr, W.n_in, W.n_out, nr);
            }
        if (W.b) k_add_bias<<<dim3((W.n_out + 255) / 256, R), 256, 0, stream>>>(Y, W.b, W.n_out);
    }

    // the target's tapped stream for R accepted positions starting at p0 (rows of src, each kAux wide) into the
    // draft layers' K/V; the last row's normed combination becomes the next anchor
    void add_context(const float* src, int R, int p0) {
        // its own pinned slot: an async copy reads the host value when it runs, and draft() writes its own position
        // before this copy may have run (one shared slot filed every accepted position under the next draft's)
        h_p0[1] = p0;
        CK(cudaMemcpyAsync(d_p0 + 1, h_p0 + 1, 4, cudaMemcpyHostToDevice, stream));
        auto body = [&] {
            lin_nv(fc, src, comb, R);
            k_rms_rows<<<R, 1024, 0, stream>>>(comb, hidden_norm, xn, kHidden, kDEps);
            for (int l = 0; l < kDLayers; ++l) {
                lin_bf16(L[l].k, xn, kt, R, kHidden, 256, nullptr);
                lin_bf16(L[l].v, xn, vt, R, kHidden, 256, nullptr);
                k_head_norm_rope<<<dim3(R, kAKv), 128, 0, stream>>>(kt, L[l].k_norm, kAKv, d_p0 + 1, 0, L[l].K, M.max_ctx);
                k_v_store<<<dim3(R, kAKv), 128, 0, stream>>>(vt, L[l].V, d_p0 + 1, 0, M.max_ctx);
            }
            CK(cudaMemcpyAsync(anchor, xn + size_t(R - 1) * kHidden, kHidden * 4, cudaMemcpyDeviceToDevice, stream));
        };
        // the rounds' small updates always read the target's step rows (M.aux): one recording per row count
        if (R <= kTMax && src == M.aux) M.graphed(ctx_graph[R], body);
        else body();
    }

    // 7 drafts after `bonus`, which sits at position p0 (the context covers positions < p0)
    std::vector<int> draft(int bonus, int p0) {
        *h_p0 = p0;
        h_ids[0] = bonus;
        for (int i = 1; i < kDBlock; ++i) h_ids[i] = 990;          // dflash mask_token_id
        CK(cudaMemcpyAsync(d_p0, h_p0, 4, cudaMemcpyHostToDevice, stream));
        CK(cudaMemcpyAsync(ids, h_ids, kDBlock * 4, cudaMemcpyHostToDevice, stream));
        const int R = kDBlock;
        const int RL = kDSlots * kDTopK;
        const float* hid = out + kHidden;
        M.graphed(draft_graph, [&] {
            k_embed_ids<<<dim3(16, R), 256, 0, stream>>>(M.embed, ids, h, kHidden);
            for (int l = 0; l < kDLayers; ++l) {
                Layer& y = L[l];
                k_add_rms_rows<<<R, 1024, 0, stream>>>(res, h, y.in_ln, x, kHidden, kDEps, l == 0);
                lin_bf16(y.ac_proj, x, coef, R, kHidden, 1024, nullptr);
                k_dconv<<<dim3(16, R), 256, 0, stream>>>(x, coef, y.ac_base, 0, xa);
                lin_bf16(y.q, xa, q, R, kHidden, kHidden, nullptr);
                lin_bf16(y.k, xa, k, R, kHidden, 256, nullptr);
                lin_bf16(y.v, xa, v, R, kHidden, 256, nullptr);
                k_head_norm_rope<<<dim3(R, kAHeads), 128, 0, stream>>>(q, y.q_norm, kAHeads, d_p0, 0, nullptr, M.max_ctx);
                k_head_norm_rope<<<dim3(R, kAKv), 128, 0, stream>>>(k, y.k_norm, kAKv, d_p0, 0, y.K, M.max_ctx);
                k_v_store<<<dim3(R, kAKv), 128, 0, stream>>>(v, y.V, d_p0, 0, M.max_ctx);
                k_draft_attn<<<dim3(kAHeads, R), 128, 0, stream>>>(q, y.K, y.V, y.sink, att, d_p0, M.max_ctx);
                lin_bf16(y.o, att, o, R, kHidden, kHidden, nullptr);
                k_dconv<<<dim3(16, R), 256, 0, stream>>>(o, coef, y.ac_base, 1, h);
                k_add_rms_rows<<<R, 1024, 0, stream>>>(res, h, y.post_ln, x, kHidden, kDEps, 0);
                lin_bf16(y.mc_proj, x, coef, R, kHidden, 1024, nullptr);
                k_dconv<<<dim3(16, R), 256, 0, stream>>>(x, coef, y.mc_base, 0, xa);
                lin_nv(y.gate, xa, g, R);
                lin_nv(y.up, xa, u, R);
                k_swiglu<<<512, 256, 0, stream>>>(g, u, act, size_t(R) * kDFf);
                lin_nv(y.down, act, dn, R);
                k_dconv<<<dim3(16, R), 256, 0, stream>>>(dn, coef, y.mc_base, 1, h);
            }
            k_add_rms_rows<<<R, 1024, 0, stream>>>(res, h, norm, out, kHidden, kDEps, 0);
            // the target's head on the 7 mask slots, their top 8
            M.q8_0(M.head, hid, M.logits, kHidden, kVocab, kDSlots);
            k_logsoftmax_top8<<<kDSlots, 1024, 0, stream>>>(M.logits, kVocab, cand, lp);
            // LiLiCorr
            k_embed_ids<<<dim3(16, RL), 256, 0, stream>>>(M.embed, cand, tokE, kHidden);
            lin_nv(token_proj, tokE, tok, RL);
            lin_nv(pass_proj, hid, pass, kDSlots);
            k_ll_features<<<RL, 256, 0, stream>>>(lp, fn_w, fn_b, fm_up_w, fm_up_b, feat);
            lin_bf16(fm_down_w, feat, featd, RL, kLl, kLl, fm_down_b);
            k_ll_sum<<<RL, 256, 0, stream>>>(hs, tok, pass, featd, slot_emb, rank_emb);
            for (int i = 0; i < 2; ++i) {
                k_rms_rows<<<RL, 1024, 0, stream>>>(hs, l_attn_norm[i], lx, kLl, kDEps);
                lin_bf16(l_in_w[i], lx, qkv, RL, kLl, 3 * kLl, l_in_b[i]);
                k_ll_attn<<<dim3(8, RL), 128, 0, stream>>>(qkv, att_bias, latt, RL);
                lin_nv(l_out[i], latt, lo, RL);
                k_add<<<64, 256, 0, stream>>>(hs, lo, RL * kLl);
                k_rms_rows<<<RL, 1024, 0, stream>>>(hs, l_mlp_norm[i], lx, kLl, kDEps);
                lin_nv(l_up[i], lx, lu, RL);
                k_silu<<<256, 256, 0, stream>>>(lu, size_t(RL) * 2 * kLl);
                lin_nv(l_down[i], lu, lo, RL);
                k_add<<<64, 256, 0, stream>>>(hs, lo, RL * kLl);
            }
            k_rms_rows<<<RL, 1024, 0, stream>>>(hs, out_norm, hsn, kLl, kDEps);
            lin_nv(context_proj, anchor, an_raw, 1);
            k_rms_rows<<<1, 1024, 0, stream>>>(an_raw, anchor_norm, an, kLl, kDEps);
            lin_bf16(f_self, hsn, pre1, RL, kLl, kLl, f_bias);
            lin_bf16(f_anchor, an, ar, 1, kLl, kLl, nullptr);
            k_mul_row<<<RL, 256, 0, stream>>>(hsn, an, lx, kLl);
            lin_bf16(f_cross, lx, pre2, RL, kLl, kLl, nullptr);
            k_add3_silu<<<RL, 256, 0, stream>>>(pre1, pre2, ar, kLl);
            lin_bf16(out_head_w, pre1, ov, RL, kLl, kLl, out_head_b);
            lin_bf16(in_head_w, pre1, iv, RL, kLl, kLl, in_head_b);
            k_normalize_rows<<<RL, 256, 0, stream>>>(ov, kLl, 1e-4f);
            k_normalize_rows<<<RL, 256, 0, stream>>>(iv, kLl, 1e-4f);
            lin_bf16(aout_w, an, aout, 1, kLl, kLl, aout_b);
            k_normalize_rows<<<1, 256, 0, stream>>>(aout, kLl, 1e-4f);
            k_ll_walk<<<1, 256, 0, stream>>>(ov, iv, aout, cand, drafts);
        });
        if (const char* dp = std::getenv("NEMO_DDUMP"); dp && !dumped) {   // the first draft's pieces, for ref_draft.py
            dumped = true;
            CK(cudaStreamSynchronize(stream));
            FILE* f = std::fopen(dp, "wb");
            auto put = [&](const void* d, size_t n) { std::vector<char> b(n); CK(cudaMemcpy(b.data(), d, n, cudaMemcpyDeviceToHost)); std::fwrite(b.data(), 1, n, f); };
            put(anchor, kHidden * 4); put(out, size_t(kDBlock) * kHidden * 4); put(cand, RL * 4); put(lp, RL * 4); put(drafts, kDSlots * 4);
            put(L[0].K, size_t(p0 + kDBlock) * 128 * 4);   // layer 0, KV head 0, positions 0 .. p0 + 7
            std::fclose(f);
        }
        std::vector<int> r(kDSlots);
        CK(cudaMemcpyAsync(r.data(), drafts, kDSlots * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        return r;
    }
};
}  // namespace

static int argmax(const float* p, int n) { return int(std::max_element(p, p + n) - p); }

int main(int argc, char** argv) {
    std::string dir, ids_s, dump_path, mode = "plain", draft_dir;
    bool ppl = false;
    int n_new = 0, ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--model") dir = next();
        else if (a == "--ids") {   // (--ids @FILE: the comma-separated ids from a file, for prompts past the argument limit)
            ids_s = next();
            if (!ids_s.empty() && ids_s[0] == '@') {
                std::ifstream f(ids_s.substr(1));
                if (!f) { std::fprintf(stderr, "strata-nemo: cannot read %s\n", ids_s.c_str() + 1); return 1; }
                ids_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }
        }
        else if (a == "--new") n_new = std::stoi(next());
        else if (a == "--dump") dump_path = next();
        else if (a == "--max-ctx") ctx = std::stoi(next());
        else if (a == "--mode") mode = next();
        else if (a == "--draft") draft_dir = next();
        else if (a == "--ppl") ppl = true;
        else { std::fprintf(stderr, "usage: strata-nemo --model DIR --ids 1,2,3|@FILE [--new N] [--dump FILE] [--max-ctx N] [--mode plain|small|onebyone|spec] [--draft DIR] [--ppl]\n"); return 2; }
    }
    if (dir.empty() || ids_s.empty()) { std::fprintf(stderr, "strata-nemo: --model and --ids are needed\n"); return 2; }
    std::vector<int> ids;
    std::stringstream ss(ids_s);
    for (std::string v; std::getline(ss, v, ',');) ids.push_back(std::stoi(v));

    const auto t0 = std::chrono::steady_clock::now();
    // hipBLASLt's tuned solutions (and with them its FP8 products) for this card, unless the caller set a table
    if (!std::getenv("STRATA_HIPBLASLT_TUNING")) {
        char exe[4096] = {};
        const ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
        std::string tbl = std::string(exe, len > 0 ? size_t(len) : 0);
        tbl = tbl.substr(0, tbl.rfind('/')) + "/../tools/hip/gfx1201-hipblaslt-100500.txt";
        if (FILE* f = std::fopen(tbl.c_str(), "r")) { std::fclose(f); setenv("STRATA_HIPBLASLT_TUNING", tbl.c_str(), 0); }
    }
    Model m(dir, ctx);
    m.init_big();                              // the prompt path's buffers and library, before any prompt is timed
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-nemo: card memory %.1f GB free of %.1f GB after load\n", fr / 1e9, tot / 1e9);
    }
    std::fprintf(stderr, "strata-nemo: %.1f GB on the card in %.1f s\n", m.st.loaded / 1e9, load_s);

    if (ppl) {   // how well the prompt path predicts each next token of the text (quality checks of numeric changes)
        int32_t* d_ids = nullptr;
        double* d_acc = nullptr;
        CK(cudaMalloc(&d_ids, ids.size() * 4)); CK(cudaMalloc(&d_acc, 3 * 8));
        CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_acc, 0, 3 * 8));
        m.ppl_ids = d_ids; m.ppl_n = int(ids.size()); m.ppl_acc = d_acc;
        for (size_t i = 0; i < ids.size(); i += kTBig) {
            const std::vector<int> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + kTBig));
            if (c.size() <= size_t(kTMax)) break;          // (the last few tokens: the small path has no scoring)
            m.forward_big(c);
            m.pos += int(c.size());
        }
        double acc[3];
        CK(cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost));
        std::printf("ppl: %.0f tokens, mean NLL %.5f, perplexity %.4f, top-1 %.4f\n", acc[1], acc[0] / acc[1], std::exp(acc[0] / acc[1]), acc[2] / acc[1]);
        return 0;
    }
    // dump: positions in order, each the 88 block outputs then the logits (the reference's layout)
    const bool dumping = !dump_path.empty();
    std::vector<float> blocks_rows, logit_rows;
    auto keep = [&](const std::vector<float>& rows, const std::vector<float>& lg, int T) {
        (void) T;
        blocks_rows.insert(blocks_rows.end(), rows.begin(), rows.end());
        logit_rows.insert(logit_rows.end(), lg.begin(), lg.end());
    };
    // (blocks come back block-major within a step: reorder to position-major when writing)
    std::vector<std::pair<int, size_t>> steps;   // (T, offset into blocks_rows) per step


    if (mode == "spec") {
        // greedy speculative decoding with the LiLiCorr drafter: draft 7, check [bonus + 7] in one step, keep the
        // agreeing prefix plus the target's own next token, move the Mamba state forward over exactly those
        if (draft_dir.empty()) { std::fprintf(stderr, "strata-nemo: --mode spec needs --draft DIR\n"); return 2; }
        Drafter D(m, draft_dir);
        const auto s0 = std::chrono::steady_clock::now();
        std::vector<float> lg;
        for (size_t i = 0; i < ids.size(); i += kTBig) {
            const std::vector<int> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + kTBig));
            const int p0 = m.pos;
            if (c.size() > size_t(kTMax)) lg = m.forward_big(c);
            else lg = m.forward(c, int(c.size()), false);
            m.pos += int(c.size());
            D.add_context(c.size() > size_t(kTMax) ? m.aux_big : m.aux, int(c.size()), p0);
        }
        CK(cudaStreamSynchronize(m.stream));
        const double rs = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        const auto s1 = std::chrono::steady_clock::now();
        int bonus = argmax(lg.data(), kVocab);
        std::vector<int> out = {bonus};
        int rounds = 0, drafted_ok = 0;
        double t_draft = 0, t_check = 0;
        while (int(out.size()) < n_new) {
            const auto a0 = std::chrono::steady_clock::now();
            const std::vector<int> dr = D.draft(bonus, m.pos);
            const auto a1 = std::chrono::steady_clock::now();
            std::vector<int> win = {bonus};
            win.insert(win.end(), dr.begin(), dr.end());
            const std::vector<int> best = m.forward_argmax(win, 0);
            int acc = 0;
            while (acc < kDSlots && best[acc] == dr[acc]) ++acc;
            const int nb = best[acc];
            const int p0 = m.pos;
            m.commit(acc + 1);
            D.add_context(m.aux, acc + 1, p0);
            for (int j = 0; j < acc; ++j) out.push_back(dr[j]);
            out.push_back(nb);
            bonus = nb;
            ++rounds; drafted_ok += acc;
            const auto a2 = std::chrono::steady_clock::now();
            t_draft += std::chrono::duration<double>(a1 - a0).count();
            t_check += std::chrono::duration<double>(a2 - a1).count();
        }
        CK(cudaStreamSynchronize(m.stream));
        const double ws = std::chrono::duration<double>(std::chrono::steady_clock::now() - s1).count();
        out.resize(n_new);
        for (int k = 0; k < n_new; ++k) std::printf("%d%s", out[k], k + 1 < n_new ? "," : "\n");
        std::fprintf(stderr, "strata-nemo: read %zu in %.3f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n",
                     ids.size(), rs, ids.size() / rs, n_new, ws, n_new / ws);
        std::fprintf(stderr, "strata-nemo: %d rounds, %.2f tokens a round (%.2f drafts accepted), drafting %.1f ms + checking %.1f ms a round\n",
                     rounds, double(out.size()) / std::max(rounds, 1), double(drafted_ok) / std::max(rounds, 1),
                     1e3 * t_draft / std::max(rounds, 1), 1e3 * t_check / std::max(rounds, 1));
        return 0;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<float> lg;
    auto run = [&](const std::vector<int>& toks) {
        std::vector<float> rows;
        lg = m.forward(toks, int(toks.size()), dumping, dumping ? &rows : nullptr);
        if (dumping) { steps.push_back({int(toks.size()), blocks_rows.size()}); keep(rows, lg, int(toks.size())); lg.erase(lg.begin(), lg.end() - kVocab); }
        m.pos += int(toks.size());
    };
    auto run_big = [&](const std::vector<int>& toks) {
        std::vector<float> rows;
        lg = m.forward_big(toks, dumping ? &rows : nullptr);
        if (dumping) { steps.push_back({int(toks.size()), blocks_rows.size()}); keep(rows, lg, int(toks.size())); lg.erase(lg.begin(), lg.end() - kVocab); }
        m.pos += int(toks.size());
    };
    if (mode == "onebyone") for (int t : ids) run({t});
    else if (mode == "small") for (size_t i = 0; i < ids.size(); i += kTMax) run(std::vector<int>(ids.begin() + i, ids.begin() + std::min(ids.size(), i + kTMax)));
    else for (size_t i = 0; i < ids.size(); i += kTBig) {
        const std::vector<int> c(ids.begin() + i, ids.begin() + std::min(ids.size(), i + kTBig));
        if (c.size() > size_t(kTMax)) run_big(c); else run(c);
    }
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto t2 = std::chrono::steady_clock::now();
    for (int k = 0; k < n_new; ++k) {
        const int nxt = argmax(lg.data(), kVocab);
        std::printf("%d%s", nxt, k + 1 < n_new ? "," : "\n");
        std::fflush(stdout);
        run({nxt});
    }
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    std::fprintf(stderr, "strata-nemo: read %zu in %.3f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n",
                 ids.size(), read_s, ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    if (dumping) {
        FILE* f = std::fopen(dump_path.c_str(), "wb");
        int P = 0;
        for (auto& s : steps) P += s.first;
        std::fprintf(f, "NEMO1 %d %zu %d %d\n", P, m.blocks.size(), kHidden, kVocab);
        size_t lrow = 0;
        const size_t B = m.blocks.size();
        for (auto& s : steps) {
            for (int t = 0; t < s.first; ++t) {
                for (size_t b = 0; b < B; ++b) std::fwrite(&blocks_rows[s.second + (b * s.first + t) * kHidden], 4, kHidden, f);
                std::fwrite(&logit_rows[(lrow + t) * kVocab], 4, kVocab, f);
            }
            lrow += s.first;
        }
        std::fclose(f);
    }
    return 0;
}
