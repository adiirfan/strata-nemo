// src/nemotron/nemo_draft.cuh - NVIDIA's LiLiCorr drafter for Nemotron 3.5 Bolt (nvidia/...-24GB-LiLiCorr), the
// kernels it needs beyond the target's. Transcribed from vLLM 77154ca6 (qwen3_dflash.py, qwen3_dflash2.py,
// lilicorr.py, the DFlash / DFlash2 / LiLiCorr speculators) by way of a plain Python reference (not included), which
// matched the engine's own greedy text on a worked-arithmetic answer.
//
// A round: the block [bonus, mask x 7] at positions pos..pos+7 goes through 5 Qwen3-style layers (grouped dynamic
// convolutions around attention and MLP; attention bidirectional inside the block, a 1,024-position window over the
// context, per-head sink logits); the target's head gives each of the 7 mask slots its top 8 candidates; the LiLiCorr
// correlator scores candidate-to-candidate transitions and a greedy walk picks the 7 drafts.
// The context is the target's own stream after blocks 1, 18, 35, 51, 68, 85, combined by `fc` and stored as each
// draft layer's K/V at the accepted positions.
#pragma once

constexpr int kDBlock = 8, kDSlots = 7, kDTopK = 8, kDWindow = 1024, kDLayers = 5, kDFf = 12288;
// the draft layers attend over the last kDWindow positions only: their K / V live in a ring of kDRing slots
// (position p in slot p & (kDRing - 1)) instead of max_ctx rows - 10 KB a context position saved
constexpr int kDRing = 2048;
static_assert((kDRing & (kDRing - 1)) == 0 && kDRing >= kDWindow + kDBlock, "a power of two past the window and a block");
constexpr int kLl = 1024;                  // LiLiCorr width
constexpr float kDEps = 1e-6f;
constexpr int kTaps[6] = {1, 18, 35, 51, 68, 85};
constexpr int kAux = 6 * kHidden;          // 24576

__global__ void k_rms_rows(const float* x, const uint16_t* w, float* y, int n, float eps) {
    __shared__ float sh[1024];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * n;
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) a += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum<1024>(a, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) yr[i] = (w ? bf16f(w[i]) : 1.f) * (xr[i] * inv);
}
// res += x (or res = x when first); y = norm(res)
__global__ void k_add_rms_rows(float* res, const float* x, const uint16_t* w, float* y, int n, float eps, int first) {
    __shared__ float sh[1024];
    const size_t o = size_t(blockIdx.x) * n;
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = first ? x[o + i] : res[o + i] + x[o + i];
        res[o + i] = v;
        a += v * v;
    }
    const float inv = rsqrtf(block_sum<1024>(a, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) y[o + i] = bf16f(w[i]) * (res[o + i] * inv);
}
// y (rows, n) += b; grid (n / 256 rounded up, rows) (a flat loop with a 64-bit i % n took 12 us on 56 x 1024)
__global__ void k_add_bias(float* y, const uint16_t* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[size_t(blockIdx.y) * n + i] += bf16f(b[i]);
}
__global__ void k_silu(float* a, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) a[i] = a[i] / (1.f + expf(-a[i]));
}
__global__ void k_swiglu(const float* g, const float* u, float* y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}
// bf16 embedding rows for the ids
__global__ void k_embed_ids(const uint16_t* table, const int32_t* ids, float* y, int n) {
    const uint16_t* r = table + size_t(ids[blockIdx.y]) * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[size_t(blockIdx.y) * n + i] = bf16f(r[i]);
}

// per (row, head): RMSNorm over the 128 dims (eps 1e-6) then neox RoPE at position pos0 + row (theta 10000);
// x: (R, heads, 128) in place; when K is given the result is also stored at (head, pos0 + row) of the K ring (kDRing)
__global__ void k_head_norm_rope(float* x, const uint16_t* w, int heads, const int* pos0p, int pos0_add, float* K, int max_ctx) {
    __shared__ float v[128], sh[128];
    const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float* xr = x + (size_t(row) * heads + h) * 128;
    const float a = xr[d];
    sh[d] = a * a;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (d < s) sh[d] += sh[d + s]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / 128.f + kDEps);
    v[d] = bf16f(w[d]) * a * inv;
    __syncthreads();
    const int pos = *pos0p + pos0_add + row;
    const int i = d % 64;
    const float ang = float(pos) * powf(10000.f, -2.f * i / 128.f);
    const float c = cosf(ang), s = sinf(ang);
    const float out = d < 64 ? v[d] * c - v[d + 64] * s : v[d] * c + v[d - 64] * s;
    xr[d] = out;
    // (K: the draft ring - a call of more than kDRing rows stores only its last kDRing, so no two rows share a slot)
    if (K && row + kDRing >= int(gridDim.x)) K[(size_t(h) * kDRing + (pos & (kDRing - 1))) * 128 + d] = out;
}
// V rows straight into the cache at (head, pos0 + row)
__global__ void k_v_store(const float* v, float* V, const int* pos0p, int pos0_add, int max_ctx) {
    const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (row + kDRing < int(gridDim.x)) return;                       // (the ring keeps a call's last kDRing rows)
    const int pos = *pos0p + pos0_add + row;
    V[(size_t(h) * kDRing + (pos & (kDRing - 1))) * 128 + d] = v[(size_t(row) * kAKv + h) * 128 + d];
}

// grouped dynamic convolution over the 8-row block: out[r] = sum_tap (base[tap] + delta[r][tap][c / 16]) x[r - tap]
// coef: kernel_projection output (rows, 1024) = (side 2, tap 2, group 256); side picks which half is used
__global__ void k_dconv(const float* x, const float* coef, const uint16_t* base, int side, float* out) {
    const int r = blockIdx.y;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < kHidden; c += gridDim.x * blockDim.x) {
        const float* cr = coef + size_t(r) * 1024 + side * 512;
        const int g = c / 16;
        float a = (bf16f(base[(side * 2 + 0) * kHidden + c]) + cr[g]) * x[size_t(r) * kHidden + c];
        if (r >= 1) a += (bf16f(base[(side * 2 + 1) * kHidden + c]) + cr[256 + g]) * x[size_t(r - 1) * kHidden + c];
        out[size_t(r) * kHidden + c] = a;
    }
}

// draft attention: query row t of the block (position p = pos0 + t) over keys [max(0, p - 1023), pos0 + 8), the block's
// own keys included both ways (no causal mask inside it), plus a per-head sink logit in the softmax denominator.
// One block per (head, row); 4 warps stride the keys, lanes hold 4 dims; merged in LDS.
__global__ void __launch_bounds__(128) k_draft_attn(const float* q, const float* K, const float* V, const uint16_t* sink,
                                                    float* out, const int* pos0p, int max_ctx) {
    __shared__ float wm[4], wl[4], wacc[4][128];
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int pos0 = *pos0p, p = pos0 + t, kvh = h / (kAHeads / kAKv);
    const float scale = rsqrtf(128.f);
    const float4 qq = reinterpret_cast<const float4*>(q + (size_t(t) * kAHeads + h) * 128)[lane];
    const float* Kh = K + size_t(kvh) * kDRing * 128;               // (the ring: slot j & (kDRing - 1))
    const float* Vh = V + size_t(kvh) * kDRing * 128;
    float m = -1e30f, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int j = max(0, p - (kDWindow - 1)) + warp; j < pos0 + kDBlock; j += 4) {
        const float4 kv = reinterpret_cast<const float4*>(Kh + size_t(j & (kDRing - 1)) * 128)[lane];
        const float4 vv = reinterpret_cast<const float4*>(Vh + size_t(j & (kDRing - 1)) * 128)[lane];
        float s = qq.x * kv.x + qq.y * kv.y + qq.z * kv.z + qq.w * kv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o, 32);
        s *= scale;
        const float mn = fmaxf(m, s), e0 = expf(m - mn), e = expf(s - mn);
        l = l * e0 + e; m = mn;
        a0 = a0 * e0 + e * vv.x; a1 = a1 * e0 + e * vv.y; a2 = a2 * e0 + e * vv.z; a3 = a3 * e0 + e * vv.w;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    reinterpret_cast<float4*>(wacc[warp])[lane] = make_float4(a0, a1, a2, a3);
    __syncthreads();
    const float sk = bf16f(sink[h]);
    float M = sk;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w]);
    float num = 0.f, den = expf(sk - M);
    for (int w = 0; w < 4; ++w) {
        const float e = wl[w] > 0.f ? expf(wm[w] - M) : 0.f;
        num += wacc[w][tid] * e;
        den += wl[w] * e;
    }
    out[(size_t(t) * kAHeads + h) * 128 + tid] = num / den;
}

// per slot row: log-softmax over the vocabulary and its top 8 (ids, log-probs), ties to the lower id. Each thread keeps
// its own sorted top 8 in one pass; 8 rounds of a block argmax over those 1024 x 8 pick the result.
__global__ void __launch_bounds__(1024) k_logsoftmax_top8(const float* logits, int V, int32_t* ids, float* lp) {
    __shared__ float sh[1024];
    __shared__ int si[1024], sw[1024];
    const float* x = logits + size_t(blockIdx.x) * V;
    float tv[kDTopK]; int ti[kDTopK];
#pragma unroll
    for (int k = 0; k < kDTopK; ++k) { tv[k] = -INFINITY; ti[k] = 0x7fffffff; }
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += 1024) {
        const float v = x[i];
        m = fmaxf(m, v);
        if (v > tv[kDTopK - 1]) {                    // insert (ids ascend within a thread, so > keeps ties to the lower id)
            int k = kDTopK - 1;
            while (k > 0 && v > tv[k - 1]) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; --k; }
            tv[k] = v; ti[k] = i;
        }
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); __syncthreads(); }
    m = sh[0];
    __syncthreads();
    float a = 0.f;
    for (int i = threadIdx.x; i < V; i += 1024) a += expf(x[i] - m);
    const float lse = m + logf(block_sum<1024>(a, sh));
    int head = 0;                                    // next unused entry of this thread's list
    for (int k = 0; k < kDTopK; ++k) {
        sh[threadIdx.x] = head < kDTopK ? tv[head] : -INFINITY;
        si[threadIdx.x] = head < kDTopK ? ti[head] : 0x7fffffff;
        sw[threadIdx.x] = threadIdx.x;
        __syncthreads();
        for (int s = 512; s > 0; s >>= 1) {
            if (threadIdx.x < s && (sh[threadIdx.x + s] > sh[threadIdx.x] || (sh[threadIdx.x + s] == sh[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x]))) {
                sh[threadIdx.x] = sh[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s]; sw[threadIdx.x] = sw[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) { ids[blockIdx.x * kDTopK + k] = si[0]; lp[blockIdx.x * kDTopK + k] = sh[0] - lse; }
        if (threadIdx.x == sw[0]) ++head;
        __syncthreads();
    }
}

// LiLiCorr: the candidate features (5 per candidate) through LayerNorm(5) then feature_mlp.up (5 -> 1024, bias), silu
__global__ void k_ll_features(const float* lp, const uint16_t* nw, const uint16_t* nb, const uint16_t* upw, const uint16_t* upb, float* out) {
    const int row = blockIdx.x, s = row / kDTopK, k = row % kDTopK;              // row = slot * 8 + rank
    __shared__ float f[5];
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < kDTopK; ++j) mx = fmaxf(mx, lp[s * kDTopK + j]);
        const float v = lp[row];
        float raw[5] = {v, expf(v), v - mx, float(k) / float(kDTopK - 1), k == 0 ? 1.f : 0.f};
        float mean = 0.f; for (int j = 0; j < 5; ++j) mean += raw[j]; mean /= 5.f;
        float var = 0.f; for (int j = 0; j < 5; ++j) var += (raw[j] - mean) * (raw[j] - mean); var /= 5.f;
        const float inv = rsqrtf(var + 1e-5f);
        for (int j = 0; j < 5; ++j) f[j] = (raw[j] - mean) * inv * bf16f(nw[j]) + bf16f(nb[j]);
    }
    __syncthreads();
    for (int o = threadIdx.x; o < kLl; o += blockDim.x) {
        float a = bf16f(upb[o]);
        for (int j = 0; j < 5; ++j) a += bf16f(upw[o * 5 + j]) * f[j];
        out[size_t(row) * kLl + o] = a / (1.f + expf(-a));
    }
}
// hs = token + pass[slot] + feature + slot_embedding[slot] + rank_embedding[rank]
__global__ void k_ll_sum(float* hs, const float* tok, const float* pass, const float* feat, const uint16_t* slot_emb, const uint16_t* rank_emb) {
    const int row = blockIdx.x, s = row / kDTopK, k = row % kDTopK;
    for (int o = threadIdx.x; o < kLl; o += blockDim.x)
        hs[size_t(row) * kLl + o] = tok[size_t(row) * kLl + o] + pass[size_t(s) * kLl + o] + feat[size_t(row) * kLl + o] +
                                    bf16f(slot_emb[s * kLl + o]) + bf16f(rank_emb[k * kLl + o]);
}
// lattice attention: 8 heads of 128 over the 56 (slot, candidate) rows, an additive per-head bias, no mask
__global__ void k_ll_attn(const float* qkv, const float* bias, float* out, int R) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;      // 128 threads
    __shared__ float s[64];
    const float* q = qkv + size_t(r) * 3 * kLl + h * 128;
    if (d < R) {
        const float* k = qkv + size_t(d) * 3 * kLl + kLl + h * 128;
        float a = 0.f;
        for (int i = 0; i < 128; ++i) a += q[i] * k[i];
        s[d] = a * rsqrtf(128.f) + bias[(size_t(h) * R + r) * R + d];
    }
    __syncthreads();
    if (d == 0) {
        float m = -INFINITY; for (int j = 0; j < R; ++j) m = fmaxf(m, s[j]);
        float z = 0.f; for (int j = 0; j < R; ++j) { s[j] = expf(s[j] - m); z += s[j]; }
        for (int j = 0; j < R; ++j) s[j] /= z;
    }
    __syncthreads();
    float a = 0.f;
    for (int j = 0; j < R; ++j) a += s[j] * qkv[size_t(j) * 3 * kLl + 2 * kLl + h * 128 + d];
    out[size_t(r) * kLl + h * 128 + d] = a;
}
__global__ void k_mul_row(const float* x, const float* row, float* y, int n) {   // y[r] = x[r] * row
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[size_t(blockIdx.x) * n + i] = x[size_t(blockIdx.x) * n + i] * row[i];
}
__global__ void k_add3_silu(float* a, const float* b, const float* rowc, int n) {  // a[r] = silu(a[r] + b[r] + rowc)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = a[size_t(blockIdx.x) * n + i] + b[size_t(blockIdx.x) * n + i] + rowc[i];
        a[size_t(blockIdx.x) * n + i] = v / (1.f + expf(-v));
    }
}
__global__ void k_normalize_rows(float* x, int n, float eps) {   // F.normalize: x / max(||x||, eps)
    __shared__ float sh[256];
    float* xr = x + size_t(blockIdx.x) * n;
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) a += xr[i] * xr[i];
    const float nn = fmaxf(sqrtf(block_sum<256>(a, sh)), eps);
    for (int i = threadIdx.x; i < n; i += 256) xr[i] /= nn;
}
// the greedy walk: slot 0 scores anchor_out . in[0][j]; slot s > 0 scores out[s-1][prev] . in[s][j] (times 8); the
// best j each step (ties to the lower rank) names the draft token cand[s][j]
__global__ void k_ll_walk(const float* out_v, const float* in_v, const float* a_out, const int32_t* cand, int32_t* drafts) {
    __shared__ float sc[kDTopK];
    __shared__ int prev;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;   // 8 warps, one per candidate
    if (threadIdx.x == 0) prev = 0;
    __syncthreads();
    for (int s = 0; s < kDSlots; ++s) {
        const float* a = s == 0 ? a_out : out_v + (size_t(s - 1) * kDTopK + prev) * kLl;
        const float* b = in_v + (size_t(s) * kDTopK + warp) * kLl;
        float v = 0.f;
        for (int i = lane; i < kLl; i += 32) v += a[i] * b[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o, 32);
        if (lane == 0) sc[warp] = v * 8.f;
        __syncthreads();
        if (threadIdx.x == 0) {
            int bj = 0;
            for (int j = 1; j < kDTopK; ++j) if (sc[j] > sc[bj]) bj = j;
            prev = bj;
            drafts[s] = cand[s * kDTopK + bj];
        }
        __syncthreads();
    }
}
