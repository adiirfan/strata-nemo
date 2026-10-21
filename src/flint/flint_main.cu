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
__global__ void k_head_norm_rope(float* v, const float* w, int pos, bool rope) {
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

// y[r] = W[r] . x for an fp32 matrix (the router); one warp a row
__global__ void k_f32_mv(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (r >= n_out) return;
    float acc = 0;
    for (int i = l; i < n_in; i += 32) acc += W[size_t(r) * n_in + i] * x[i];
    acc = warp_sum(acc);
    if (l == 0) y[r] = acc;
}

__device__ inline float silu(float x) { return x / (1.f + expf(-x)); }

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
__global__ void k_kv_put(const float* k, const float* v, float* K, float* V, int pos) {
    const int i = threadIdx.x + blockIdx.x * 256;
    if (i < kKvDim) { K[size_t(pos) * kKvDim + i] = k[i]; V[size_t(pos) * kKvDim + i] = v[i]; }
}

// one query position against keys lo..pos, one block (256) a query head; the head's sink logit joins the softmax's
// denominator only. sc: scratch of max_ctx floats a head.
__global__ void k_attn(const float* q, const float* K, const float* V, const float* sinks, float* out, float* sc,
                       int lo, int pos, int max_ctx) {
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

__global__ void k_embed(const float* table, int tok, float* h) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < kHidden) h[i] = table[size_t(tok) * kHidden + i];
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
    float *h, *x, *q, *k, *v, *att, *o, *f1, *f2, *f3, *ye, *ysh, *logits, *sc, *rl, *wts;
    int* idx;
    block_q8_1 *xq, *aq, *eq;
    int pos = 0;

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
        CK(cudaMalloc(&xq, (kFfDense / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&aq, (kQDim / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&eq, (kTopK * kFfExp / 32) * sizeof(block_q8_1)));
    }

    void quant(const float* a, block_q8_1* y, int n) {
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

    // one token through the model; logits on the card
    void step(int tok) {
        if (pos >= max_ctx) { std::fprintf(stderr, "strata-flint: context full (%d)\n", max_ctx); std::exit(1); }
        k_embed<<<kHidden / 256, 256, 0, st>>>(embed, tok, h);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            k_rmsnorm<<<1, 256, 0, st>>>(h, l.attn_norm, x, kHidden);
            quant(x, xq, kHidden);
            cls = 1; mv(l.wq, x, xq, q, kHidden, kQDim);
            mv(l.wk, x, xq, k, kHidden, kKvDim);
            mv(l.wv, x, xq, v, kHidden, kKvDim);
            k_head_norm_rope<<<kHeads, kHd, 0, st>>>(q, l.q_norm, pos, l.swa);
            k_head_norm_rope<<<kKv, kHd, 0, st>>>(k, l.k_norm, pos, l.swa);
            k_kv_put<<<(kKvDim + 255) / 256, 256, 0, st>>>(k, v, l.K, l.V, pos);
            const int lo = l.swa ? std::max(0, pos - (kWindow - 1)) : 0;
            k_attn<<<kHeads, 256, 0, st>>>(q, l.K, l.V, l.sinks, att, sc, lo, pos, max_ctx);
            quant(att, aq, kQDim);
            cls = 2; mv(l.wo, att, aq, o, kQDim, kHidden);
            k_add<<<kHidden / 256, 256, 0, st>>>(h, o, kHidden);
            k_rmsnorm<<<1, 256, 0, st>>>(h, l.ffn_norm, x, kHidden);
            quant(x, xq, kHidden);
            if (i < kDense) {
                cls = 4; mv(l.gate, x, xq, f1, kHidden, kFfDense);
                mv(l.up, x, xq, f2, kHidden, kFfDense);
                k_swiglu<<<kFfDense / 256, 256, 0, st>>>(f1, f2, f1, kFfDense);
                quant(f1, xq, kFfDense);
                cls = 8; mv(l.down, f1, xq, o, kFfDense, kHidden);
                k_add<<<kHidden / 256, 256, 0, st>>>(h, o, kHidden);
                continue;
            }
            k_f32_mv<<<kExperts / 8, 256, 0, st>>>(l.router, x, rl, kHidden, kExperts);
            k_route<<<1, 256, 0, st>>>(rl, idx, wts);
            const size_t gu_bytes = size_t(kFfExp) * q4_row(kHidden), d_bytes = size_t(kHidden) * q4_row(kFfExp);
            cls = 4; mvx(l.gate_e, gu_bytes, idx, x, xq, false, f1, kHidden, kFfExp, kTopK);
            mvx(l.up_e, gu_bytes, idx, x, xq, false, f2, kHidden, kFfExp, kTopK);
            k_swiglu<<<kTopK * kFfExp / 256, 256, 0, st>>>(f1, f2, f3, kTopK * kFfExp);
            quant(f3, eq, kTopK * kFfExp);
            cls = 8; mvx(l.down_e, d_bytes, idx, f3, eq, true, ye, kFfExp, kHidden, kTopK);
            // shared expert (its input xq is still the normed x)
            cls = 4; mv(l.gate_s, x, xq, f1, kHidden, kFfShared);
            mv(l.up_s, x, xq, f2, kHidden, kFfShared);
            k_swiglu<<<kFfShared / 256, 256, 0, st>>>(f1, f2, f1, kFfShared);
            quant(f1, aq, kFfShared);
            cls = 8; mv(l.down_s, f1, aq, ysh, kFfShared, kHidden);
            k_moe_sum<<<kHidden / 256, 256, 0, st>>>(h, ysh, ye, wts);
        }
        k_rmsnorm<<<1, 256, 0, st>>>(h, out_norm, x, kHidden);
        quant(x, xq, kHidden);
        cls = 16; mv(head, x, xq, logits, kHidden, vocab);
        ++pos;
    }
};

int argmax(const std::vector<float>& p) { return int(std::max_element(p.begin(), p.end()) - p.begin()); }

}  // namespace

int main(int argc, char** argv) {
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
        CK(cudaMemcpy(lg.data(), m.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
        const int nxt = argmax(lg);
        if (top) {
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
        m.step(nxt);
    }
    CK(cudaStreamSynchronize(m.st));
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    if (top) std::fclose(top);
    std::fprintf(stderr, "strata-flint: read %zu in %.3f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n",
                 ids.size(), read_s, ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    return 0;
}
