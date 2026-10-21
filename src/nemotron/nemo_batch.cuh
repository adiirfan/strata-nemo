// src/nemotron/nemo_batch.cuh - the many-positions kernels of strata-nemo (included by nemo_main.cu after its
// single-position kernels and constants). Activations are row-major (T, dim): row t is position pos0 + t.
//
// The Mamba state is never written while a step runs unless the step says so (write_after = the number of positions
// after which the state is stored; 0 = leave it). A checked draft calls the scan once with write_after = 0 to get
// its outputs, then again on just the accepted positions to move the state forward: the same kernel and the same
// order of operations, so the stored state is exactly the one the accepted outputs were computed from.
#pragma once

constexpr int kTMax = 16;            // positions per step (a draft window of 8 + slack; prompt chunks)

__global__ void k_embed_many(const uint16_t* table, const int* toks, float* y) {
    const int t = blockIdx.y;
    const uint16_t* r = table + size_t(toks[t]) * kHidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kHidden; i += gridDim.x * blockDim.x) y[size_t(t) * kHidden + i] = bf16f(r[i]);
}

__global__ void k_rmsnorm_many(const float* x, const uint16_t* w, float* y, int n) {
    __shared__ float sh[1024];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * n;
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) a += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum<1024>(a, sh) / n + kEps);
    for (int i = threadIdx.x; i < n; i += 1024) yr[i] = bf16f(w[i]) * (xr[i] * inv);
}

// a q8_1 block (block_q8_1's layout: fp16 scale and sum, 32 int8 values)
struct Q8Blk { half2 ds; int8_t qs[32]; };
// h += d (+ e); y = norm(h) * w, one row per block; q8 (if given): y as q8_1 rows too - a warp holds 32 consecutive
// values, so each block is quantized exactly as quantize_q8_1_rows / native_quantize_q8_1 do it (same xor-tree max
// and sum, the same bits), for the products that read n (one launch instead of one per product)
__global__ void k_add_rmsnorm_many(float* h, const float* d, const float* e, const uint16_t* w, float* y, int n, void* q8 = nullptr) {
    __shared__ float sh[1024];
    const size_t o = size_t(blockIdx.x) * n;
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = h[o + i] + d[o + i] + (e ? e[o + i] : 0.f);
        h[o + i] = v;
        a += v * v;
    }
    const float inv = rsqrtf(block_sum<1024>(a, sh) / n + kEps);
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float xi = bf16f(w[i]) * (h[o + i] * inv);
        y[o + i] = xi;
        if (q8) {
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            const float dq = strata::kernels::q8_1_finite(amax / 127.0f);
            Q8Blk* yb = reinterpret_cast<Q8Blk*>(q8) + (o + i) / 32;
            yb->qs[i % 32] = strata::kernels::q8_1_quant(xi, dq, amax);
            if (i % 32 == 0) yb->ds = strata::kernels::q8_1_ds(dq, sum);
        }
    }
}

// NVFP4 times T columns: one warp per row, NC columns per pass over the row's weights (y is (T, rows))
template <int NC>
__global__ void __launch_bounds__(256) k_nvfp4_cols(const uint8_t* W, const uint8_t* S, float scale2, const float* x,
                                                    float* y, int n, int rows, int T) {
    __shared__ float e4[256], e2[16];
    e4[threadIdx.x] = e4m3f(uint8_t(threadIdx.x));
    if (threadIdx.x < 16) e2[threadIdx.x] = kE2M1[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + size_t(row) * (n / 2));
    const uint16_t* s2 = reinterpret_cast<const uint16_t*>(S + size_t(row) * (n / 16));
    for (int c0 = 0; c0 < T; c0 += NC) {
        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] = 0.f;
        for (int i = lane; i < n / 32; i += 32) {
            const uint4 q = w4[i];
            const uint16_t sc = s2[i];
            const float sa = e4[sc & 255], sb = e4[sc >> 8];
            float wv[32];
            const uint32_t word[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int j = 0; j < 8; ++j) wv[k * 8 + j] = e2[(word[k] >> (4 * j)) & 15] * (k < 2 ? sa : sb);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (c0 + c >= T) break;
                const float4* x4 = reinterpret_cast<const float4*>(x + size_t(c0 + c) * n) + i * 8;
                float a = 0.f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float4 xv = x4[k];
                    a += wv[4 * k] * xv.x + wv[4 * k + 1] * xv.y + wv[4 * k + 2] * xv.z + wv[4 * k + 3] * xv.w;
                }
                acc[c] += a;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float a = acc[c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o, 32);
            if (lane == 0 && c0 + c < T) y[size_t(c0 + c) * rows + row] = a * scale2;
        }
    }
}

// BF16 times T columns (the router): one warp per row, 8 weights (16 bytes) a lane a step (n % 256 == 0; 2-byte loads
// kept too little in flight: 20 us for a 4 MB router)
__global__ void __launch_bounds__(256) k_bf16_cols(const uint16_t* W, const float* x, float* y, int n, int rows, int T) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint4* wr = reinterpret_cast<const uint4*>(W + size_t(row) * n);
    for (int c = 0; c < T; ++c) {
        const float4* xr = reinterpret_cast<const float4*>(x + size_t(c) * n);
        float a = 0.f;
#pragma unroll 4
        for (int i = lane; i < n / 8; i += 32) {
            const uint4 w = wr[i];
            const float4 x0 = xr[2 * i], x1 = xr[2 * i + 1];
            a += __uint_as_float(w.x << 16) * x0.x + __uint_as_float(w.x & 0xffff0000u) * x0.y +
                 __uint_as_float(w.y << 16) * x0.z + __uint_as_float(w.y & 0xffff0000u) * x0.w +
                 __uint_as_float(w.z << 16) * x1.x + __uint_as_float(w.z & 0xffff0000u) * x1.y +
                 __uint_as_float(w.w << 16) * x1.z + __uint_as_float(w.w & 0xffff0000u) * x1.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o, 32);
        if (lane == 0) y[size_t(c) * rows + row] = a;
    }
}

// Mamba conv over T positions, per channel (raw xBC at proj + t * kInProj + kMInter); state written after
// write_after positions (0: not at all)
__global__ void k_mamba_conv_many(const float* proj, float* state, const uint16_t* cw, const uint16_t* cb, float* out,
                                  int T, int write_after) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConvDim) return;
    float* st = state + size_t(c) * (kConvK - 1);
    float v0 = st[0], v1 = st[1], v2 = st[2];
    const float w0 = bf16f(cw[c * kConvK]), w1 = bf16f(cw[c * kConvK + 1]), w2 = bf16f(cw[c * kConvK + 2]), w3 = bf16f(cw[c * kConvK + 3]);
    const float b = bf16f(cb[c]);
    for (int t = 0; t < T; ++t) {
        const float x = proj[size_t(t) * kInProj + kMInter + c];
        const float a = b + w0 * v0 + w1 * v1 + w2 * v2 + w3 * x;
        out[size_t(t) * kConvDim + c] = a / (1.f + expf(-a));
        v0 = v1; v1 = v2; v2 = x;
        if (t + 1 == write_after) { st[0] = v0; st[1] = v1; st[2] = v2; }
    }
}

// sum each of a[0..7] over the warp's 32 lanes in 9 shuffles (not 8 x 5): every step a lane keeps half of its values
// and trades the other half with its partner; afterwards lane l holds the sum of value
// ((l >> 4) & 1) * 4 + ((l >> 3) & 1) * 2 + ((l >> 2) & 1)
__device__ __forceinline__ float warp_sum8(float (&a)[8], int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float send = b4 ? a[i] : a[i + 4], keep = b4 ? a[i + 4] : a[i];
        a[i] = keep + __shfl_xor_sync(0xffffffff, send, 16, 32);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float send = b3 ? a[i] : a[i + 2], keep = b3 ? a[i + 2] : a[i];
        a[i] = keep + __shfl_xor_sync(0xffffffff, send, 8, 32);
    }
    const float send = b2 ? a[0] : a[1], keep = b2 ? a[1] : a[0];
    float v = keep + __shfl_xor_sync(0xffffffff, send, 4, 32);
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    return v;
}

// Mamba state scan over T positions: block (head, eighth of the head-dim rows), thread = state column. Each thread
// keeps its 8 rows of the state in registers through the window. Positions go kScanTile at a time: the tile's
// step sizes, decays and x values are read into LDS (and each thread's B, C into registers) before the steps, so a
// step has no memory wait; the per-row sums over the 128 columns go through LDS, double-buffered by position.
constexpr int kScanSlices = 8;       // blocks per head (each takes 64 / kScanSlices of the head-dim rows)
constexpr int kScanTile = 16;
__global__ void __launch_bounds__(128) k_mamba_scan(const float* xbc, const float* proj, const uint16_t* A_log,
                                                    const uint16_t* D, const uint16_t* dt_bias, float* ssm, float* y,
                                                    int T, int write_after) {
    constexpr int RP = kMHeadDim / kScanSlices;
    static_assert(RP == 8, "warp_sum8 sums 8 rows");
    __shared__ float part[2][RP][4];
    __shared__ float s_dt[kScanTile], s_dA[kScanTile], s_x[kScanTile][RP];
    const int hd = blockIdx.x, n = threadIdx.x, warp = n / 32, lane = n % 32;
    const int g = hd / (kMHeads / kMGroups), p0 = blockIdx.y * RP;
    const int row = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    const float A = -expf(bf16f(A_log[hd])), Dh = bf16f(D[hd]), db = bf16f(dt_bias[hd]);
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState + size_t(p0) * kMState + n;
    float hr[RP];
#pragma unroll
    for (int p = 0; p < RP; ++p) hr[p] = hp[p * kMState];
    for (int t0 = 0; t0 < T; t0 += kScanTile) {
        const int nt = min(kScanTile, T - t0);
        __syncthreads();                                              // the last tile's LDS reads are done
        if (n < nt) {
            float dt = proj[size_t(t0 + n) * kInProj + kMInter + kConvDim + hd] + db;
            dt = dt > 20.f ? dt : log1pf(expf(dt));
            s_dt[n] = dt;
            s_dA[n] = expf(dt * A);
        }
        for (int i = n; i < nt * RP; i += 128) s_x[i / RP][i % RP] = xbc[size_t(t0 + i / RP) * kConvDim + hd * kMHeadDim + p0 + i % RP];
        float Bn[kScanTile], Cn[kScanTile];
#pragma unroll
        for (int t = 0; t < kScanTile; ++t)
            if (t < nt) {
                const float* xt = xbc + size_t(t0 + t) * kConvDim;
                Bn[t] = xt[kMInter + g * kMState + n];
                Cn[t] = xt[kMInter + kMGroups * kMState + g * kMState + n];
            }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kScanTile; ++t) {
            if (t >= nt) break;
            const float dA = s_dA[t], dtB = s_dt[t] * Bn[t];
            float c[RP];
#pragma unroll
            for (int p = 0; p < RP; ++p) {
                hr[p] = hr[p] * dA + dtB * s_x[t][p];
                c[p] = hr[p] * Cn[t];
            }
            const float sum = warp_sum8(c, lane);
            float (*pt)[4] = part[t & 1];
            if ((lane & 3) == 0) pt[row][warp] = sum;
            __syncthreads();
            if (n < RP) y[size_t(t0 + t) * kMInter + hd * kMHeadDim + p0 + n] = pt[n][0] + pt[n][1] + pt[n][2] + pt[n][3] + s_x[t][n] * Dh;
            if (t0 + t + 1 == write_after) {
#pragma unroll
                for (int p = 0; p < RP; ++p) hp[p * kMState] = hr[p];
            }
        }
    }
}

// k_mamba_conv_many + k_mamba_scan in one launch for the small steps: each scan block runs the causal conv for the
// channels it reads - its 8 x channels and its group's 128 B and 128 C channels (one each a thread; the group's
// B / C are computed by all its 128 blocks, a few multiply-adds) - instead of reading xbc. The same arithmetic per
// channel as k_mamba_conv_many, so the same values. xbc is still written, by one owner each (the x channels by their
// block, B / C by the group's (first head, slice 0) block), for k_mamba_commit; the conv state of the x channels by
// their block, of B / C by k_gated_norm_many afterwards (here another block of the group could read it after its
// owner moved it on).
__global__ void __launch_bounds__(128) k_mamba_conv_scan(const float* proj, float* conv_state, const uint16_t* cw, const uint16_t* cb,
                                                         float* xbc, const uint16_t* A_log, const uint16_t* D,
                                                         const uint16_t* dt_bias, float* ssm, float* y, int T, int write_after) {
    constexpr int RP = kMHeadDim / kScanSlices;
    __shared__ float part[2][RP][4];
    __shared__ float s_dt[kScanTile], s_dA[kScanTile], s_x[kScanTile][RP];
    const int hd = blockIdx.x, n = threadIdx.x, warp = n / 32, lane = n % 32;
    const int g = hd / (kMHeads / kMGroups), p0 = blockIdx.y * RP;
    const int row = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    const float A = -expf(bf16f(A_log[hd])), Dh = bf16f(D[hd]), db = bf16f(dt_bias[hd]);
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState + size_t(p0) * kMState + n;
    float hr[RP];
#pragma unroll
    for (int p = 0; p < RP; ++p) hr[p] = hp[p * kMState];
    // this thread's conv channels: B (cB), C (cC), and for n < RP its x channel (cX)
    const int cB = kMInter + g * kMState + n, cC = kMInter + kMGroups * kMState + g * kMState + n, cX = hd * kMHeadDim + p0 + n;
    const bool bc_owner = (hd % (kMHeads / kMGroups)) == 0 && blockIdx.y == 0;
    struct Conv {
        float v0, v1, v2, w0, w1, w2, w3, b;
        float* st;
        int c;
        __device__ void init(float* state, const uint16_t* cw, const uint16_t* cb, int ch) {
            c = ch;
            st = state + size_t(ch) * (kConvK - 1);
            v0 = st[0]; v1 = st[1]; v2 = st[2];
            w0 = bf16f(cw[ch * kConvK]); w1 = bf16f(cw[ch * kConvK + 1]); w2 = bf16f(cw[ch * kConvK + 2]); w3 = bf16f(cw[ch * kConvK + 3]);
            b = bf16f(cb[ch]);
        }
        // position t's output (the raw input in, the window shifted)
        __device__ float step(const float* proj, int t) {
            const float x = proj[size_t(t) * kInProj + kMInter + c];
            const float a = b + w0 * v0 + w1 * v1 + w2 * v2 + w3 * x;
            v0 = v1; v1 = v2; v2 = x;
            return a / (1.f + expf(-a));
        }
        __device__ void save() { st[0] = v0; st[1] = v1; st[2] = v2; }
    };
    Conv cvB, cvC, cvX;
    cvB.init(conv_state, cw, cb, cB);
    cvC.init(conv_state, cw, cb, cC);
    if (n < RP) cvX.init(conv_state, cw, cb, cX);
    for (int t0 = 0; t0 < T; t0 += kScanTile) {
        const int nt = min(kScanTile, T - t0);
        __syncthreads();                                              // the last tile's LDS reads are done
        if (n < nt) {
            float dt = proj[size_t(t0 + n) * kInProj + kMInter + kConvDim + hd] + db;
            dt = dt > 20.f ? dt : log1pf(expf(dt));
            s_dt[n] = dt;
            s_dA[n] = expf(dt * A);
        }
        float Bn[kScanTile], Cn[kScanTile];
#pragma unroll
        for (int t = 0; t < kScanTile; ++t)
            if (t < nt) {
                Bn[t] = cvB.step(proj, t0 + t);
                Cn[t] = cvC.step(proj, t0 + t);
                if (bc_owner) {
                    xbc[size_t(t0 + t) * kConvDim + cB] = Bn[t];
                    xbc[size_t(t0 + t) * kConvDim + cC] = Cn[t];
                }
                if (n < RP) {
                    const float xv = cvX.step(proj, t0 + t);
                    s_x[t][n] = xv;
                    xbc[size_t(t0 + t) * kConvDim + cX] = xv;
                    if (t0 + t + 1 == write_after) cvX.save();
                }
            }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kScanTile; ++t) {
            if (t >= nt) break;
            const float dA = s_dA[t], dtB = s_dt[t] * Bn[t];
            float c[RP];
#pragma unroll
            for (int p = 0; p < RP; ++p) {
                hr[p] = hr[p] * dA + dtB * s_x[t][p];
                c[p] = hr[p] * Cn[t];
            }
            const float sum = warp_sum8(c, lane);
            float (*pt)[4] = part[t & 1];
            if ((lane & 3) == 0) pt[row][warp] = sum;
            __syncthreads();
            if (n < RP) y[size_t(t0 + t) * kMInter + hd * kMHeadDim + p0 + n] = pt[n][0] + pt[n][1] + pt[n][2] + pt[n][3] + s_x[t][n] * Dh;
            if (t0 + t + 1 == write_after) {
#pragma unroll
                for (int p = 0; p < RP; ++p) hp[p * kMState] = hr[p];
            }
        }
    }
}

// gated group norm for T rows: y (T, kMInter), z = proj + t * kInProj. zero_out (T, kHidden), if given, is zeroed
// for the out_proj slices to add into (saves a fill launch)
__global__ void k_gated_norm_many(float* y, const float* proj, const uint16_t* w, float* zero_out = nullptr, const float* prs = nullptr,
                                  float* conv_state = nullptr, int write_after = 0) {
    __shared__ float sh[256];
    const int G = kMInter / kMGroups, base = blockIdx.x * G, t = blockIdx.y;
    if (conv_state && write_after > 0 && t == 0) {
        // (after k_mamba_conv_scan) group blockIdx.x's B / C conv state moved on by write_after positions: the last
        // three of [state, raw inputs 0 .. write_after - 1]
        const int c = kMInter + (threadIdx.x < kMState ? blockIdx.x * kMState + threadIdx.x : kMGroups * kMState + blockIdx.x * kMState + threadIdx.x - kMState);
        float* st = conv_state + size_t(c) * (kConvK - 1);
        float v[3] = {st[0], st[1], st[2]};
        for (int i = 0; i < write_after; ++i) {
            v[0] = v[1]; v[1] = v[2];
            v[2] = proj[size_t(i) * kInProj + kMInter + c] * (prs ? prs[i] : 1.f);
        }
        st[0] = v[0]; st[1] = v[1]; st[2] = v[2];
    }
    const float zs = prs ? prs[t] : 1.f;                              // proj's row scale (per-row FP8 in_proj)
    if (zero_out)
        for (int i = threadIdx.x; i < kHidden / kMGroups; i += 256) zero_out[size_t(t) * kHidden + blockIdx.x * (kHidden / kMGroups) + i] = 0.f;
    float* yr = y + size_t(t) * kMInter;
    const float* z = proj + size_t(t) * kInProj;
    float a = 0.f;
    for (int i = threadIdx.x; i < G; i += 256) {
        const float zz = z[base + i] * zs;
        const float v = yr[base + i] * (zz / (1.f + expf(-zz)));
        yr[base + i] = v;
        a += v * v;
    }
    const float inv = rsqrtf(block_sum<256>(a, sh) / G + kEps);
    for (int i = threadIdx.x; i < G; i += 256) yr[base + i] = yr[base + i] * inv * bf16f(w[base + i]);
}

// attention for T new positions starting at *pos0: their keys / values are appended, then each attends causally
__device__ inline uint16_t f2bf(float f);   // (below)
// (the cache is BF16, as the reference keeps it: kv_rows = max_ctx + 64 rows a KV head, the 64 for k_attn_flash's
// reads past the last key)
// V is kept transposed in runs of 32 keys - (key / 32, dim, key % 32) - the layout both k_attn_flash and
// k_attn_dec_wm read their V operands in, so neither transposes it again
__device__ __forceinline__ size_t vt_at(int key, int d) { return (size_t(key >> 5) * kAHeadDim + d) * 32 + (key & 31); }
__global__ void k_kv_append_many(const float* k, const float* v, uint16_t* K, uint16_t* V, const int* pos0, int kv_rows) {
    const int t = blockIdx.x, kv = blockIdx.y, i = threadIdx.x, pos = *pos0 + t;
    K[(size_t(kv) * kv_rows + pos) * kAHeadDim + i] = f2bf(k[size_t(t) * kAKv * kAHeadDim + kv * kAHeadDim + i]);
    V[size_t(kv) * kv_rows * kAHeadDim + vt_at(pos, i)] = f2bf(v[size_t(t) * kAKv * kAHeadDim + kv * kAHeadDim + i]);
}
// sc: (T, kAHeads, max_ctx)
__global__ void k_att_scores_many(const float* q, const float* K, float* sc, const int* pos0, int max_ctx) {
    const int h = blockIdx.y, t = blockIdx.z, L = *pos0 + t + 1;
    const float* qq = q + (size_t(t) * kAHeads + h) * kAHeadDim;
    float* s = sc + (size_t(t) * kAHeads + h) * max_ctx;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < L; j += gridDim.x * blockDim.x) {
        const float* k = K + (size_t(h / (kAHeads / kAKv)) * max_ctx + j) * kAHeadDim;
        float a = 0.f;
        for (int d = 0; d < kAHeadDim; ++d) a += qq[d] * k[d];
        s[j] = a * rsqrtf(float(kAHeadDim));
    }
}
__global__ void k_att_softmax_many(float* sc, const int* pos0, int max_ctx) {
    __shared__ float sh[256];
    const int h = blockIdx.x, t = blockIdx.y, L = *pos0 + t + 1;
    float* s = sc + (size_t(t) * kAHeads + h) * max_ctx;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += 256) m = fmaxf(m, s[j]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0];
    __syncthreads();
    float a = 0.f;
    for (int j = threadIdx.x; j < L; j += 256) { const float e = expf(s[j] - m); s[j] = e; a += e; }
    const float inv = 1.f / block_sum<256>(a, sh);
    for (int j = threadIdx.x; j < L; j += 256) s[j] *= inv;
}
__global__ void k_att_out_many(const float* sc, const float* V, float* o, const int* pos0, int max_ctx) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x, L = *pos0 + t + 1;
    const float* v = V + size_t(h / (kAHeads / kAKv)) * max_ctx * kAHeadDim + d;
    const float* s = sc + (size_t(t) * kAHeads + h) * max_ctx;
    float a = 0.f;
    for (int j = 0; j < L; ++j) a += s[j] * v[size_t(j) * kAHeadDim];
    o[(size_t(t) * kAHeads + h) * kAHeadDim + d] = a;
}

// lane i's value from lane i ^ M (M < 16, within a row of 16: DPP row_xmask) and from lane i ^ 16 (permlanex16)
template <int M> __device__ __forceinline__ int rt_xmask(int v) { return __builtin_amdgcn_mov_dpp(v, 0x160 + M, 0xf, 0xf, false); }
template <int M> __device__ __forceinline__ float rt_xmask(float v) { return __int_as_float(rt_xmask<M>(__float_as_int(v))); }
__device__ __forceinline__ int rt_x16(int v) { return (int) __builtin_amdgcn_permlanex16((unsigned) v, (unsigned) v, 0x76543210u, 0xfedcba98u, false, false); }
__device__ __forceinline__ float rt_x16(float v) { return __int_as_float(rt_x16(__float_as_int(v))); }
// the router for T positions: sigmoid scores plus bias, the top k (larger first, ties to the smaller index), their
// sigmoids normalized x kRouteScale; idx / wts are (T, k). One warp a row (8 rows a block of 256, n <= 512, k <= 16):
// each lane holds n/32 scores, k rounds of a warp argmax by shuffles, no barriers (a 512-wide LDS reduction per round
// took ~80 barriers a row)
// rows of n floats (n % 1024 == 0; grid (n / 1024, rows)) from src (row stride lds) to dst (row stride ldd)
__global__ void __launch_bounds__(256) k_copy_rows(float* dst, int ldd, const float* src, int lds) {
    const int r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    reinterpret_cast<float4*>(dst + size_t(r) * ldd)[i] = reinterpret_cast<const float4*>(src + size_t(r) * lds)[i];
}
// one row's choice by one warp (k_route_warp's body): r is the row's raw scores, VOL reads them past the vector caches
template <bool VOL = false>
__device__ __forceinline__ void route_row(const float* r, const float* bias, int n, int k, int row, int lane, int32_t* idx, float* wts) {
    constexpr int M = 16;
    float c[M], sg[M];
    // (every load issued unconditionally at a clamped index, then masked: guarded loads became 16 branches whose
    // memory waits ran one after another)
    float rv[M], bv0[M];
#pragma unroll
    for (int j = 0; j < M; ++j) {
        const int e = min(lane + 32 * j, n - 1);
        rv[j] = VOL ? __builtin_nontemporal_load(r + e) : r[e];
        bv0[j] = bias[e];
    }
#pragma unroll
    for (int j = 0; j < M; ++j) {
        const bool in = lane + 32 * j < n;
        const float s = 1.f / (1.f + expf(-rv[j]));
        sg[j] = in ? s : 0.f;
        c[j] = in ? s + bv0[j] : -INFINITY;
    }
    float sum = 0.f, myw = 0.f;                                       // lane q keeps choice q
    int myi = 0;
    for (int q = 0; q < k; ++q) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < M; ++j)
            if (c[j] > bv) { bv = c[j]; bi = lane + 32 * j; }
        // (lane exchanges by DPP / permlanex16: a few cycles each, where a shuffle goes through LDS)
        auto take = [&](float ov, int oi) { if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } };
        take(rt_xmask<1>(bv), rt_xmask<1>(bi));
        take(rt_xmask<2>(bv), rt_xmask<2>(bi));
        take(rt_xmask<4>(bv), rt_xmask<4>(bi));
        take(rt_xmask<8>(bv), rt_xmask<8>(bi));
        take(rt_x16(bv), rt_x16(bi));
        float w = 0.f;
#pragma unroll
        for (int j = 0; j < M; ++j) {
            const bool hit = lane + 32 * j == bi;
            w = hit ? sg[j] : w;
            c[j] = hit ? -INFINITY : c[j];
        }
        w = __shfl_sync(0xffffffffu, w, bi & 31, 32);
        if (lane == q) { myw = w; myi = bi; }
        sum += w;
    }
    if (lane < k) {
        idx[row * k + lane] = myi;
        wts[row * k + lane] = myw / (sum + 1e-20f) * kRouteScale;
    }
}
__global__ void __launch_bounds__(256) k_route_warp(const float* raw, const float* bias, int n, int k, int T, int32_t* idx, float* wts) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= T) return;
    route_row(raw + size_t(row) * n, bias, n, k, row, lane, idx, wts);
}
// one position's routing in one launch: k_bf16_cols' scores (a warp a router row), then the last block to finish
// (a counter, which it puts back to 0) makes the choice from them
__global__ void __launch_bounds__(256) k_route1(const uint16_t* W, const float* x, float* raw, const float* bias, int n, int rows, int k,
                                                unsigned* done, int32_t* idx, float* wts) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp;
    if (row < rows) {
        const uint4* wr = reinterpret_cast<const uint4*>(W + size_t(row) * n);
        const float4* xr = reinterpret_cast<const float4*>(x);
        float a = 0.f;
#pragma unroll 4
        for (int i = lane; i < n / 8; i += 32) {
            const uint4 w = wr[i];
            const float4 x0 = xr[2 * i], x1 = xr[2 * i + 1];
            a += __uint_as_float(w.x << 16) * x0.x + __uint_as_float(w.x & 0xffff0000u) * x0.y +
                 __uint_as_float(w.y << 16) * x0.z + __uint_as_float(w.y & 0xffff0000u) * x0.w +
                 __uint_as_float(w.z << 16) * x1.x + __uint_as_float(w.z & 0xffff0000u) * x1.y +
                 __uint_as_float(w.w << 16) * x1.z + __uint_as_float(w.w & 0xffff0000u) * x1.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o, 32);
        if (lane == 0) raw[row] = a;
    }
    __shared__ unsigned last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last || warp) return;
    __threadfence();
    route_row<true>(raw, bias, rows, k, 0, lane, idx, wts);
    if (lane == 0) *done = 0;
}

// k_wsum_many with eo also as q8_1 rows (n == gridDim.x * blockDim.x: a warp holds 32 consecutive outputs; the same
// xor-tree quantizer as native_quantize_q8_1), for fc2 without its quantize launch
__global__ void k_wsum_q8(float* eo, const float* y, const float* w, int k, int n, void* q8) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.f;
    for (int j = 0; j < k; ++j) a += w[t * k + j] * y[(size_t(t) * k + j) * n + i];
    eo[size_t(t) * n + i] = a;
    float amax = fabsf(a), sum = a;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    const float dq = strata::kernels::q8_1_finite(amax / 127.0f);
    Q8Blk* yb = reinterpret_cast<Q8Blk*>(q8) + (size_t(t) * n + i) / 32;
    yb->qs[i % 32] = strata::kernels::q8_1_quant(a, dq, amax);
    if (i % 32 == 0) yb->ds = strata::kernels::q8_1_ds(dq, sum);
}
// eo[t] = sum_j w[t][j] * y[t * k + j], eo (T, n)
__global__ void k_wsum_many(float* eo, const float* y, const float* w, int k, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < k; ++j) a += w[t * k + j] * y[(size_t(t) * k + j) * n + i];
        eo[size_t(t) * n + i] = a;
    }
}

// ---- small-batch kernels with the inputs kept on chip -----------------------------------------------------------
// NVFP4 times up to NC columns of q8_1 inputs (quantize_q8_1_rows layout: column c is n/32 blocks of 36 bytes).
// A block keeps one 4096-wide slice of every column's inputs in LDS and runs RPB rows over it, so the inputs are read
// from memory once per block instead of once per row. E2M1 values times 2 are the integers 0..12, so each 16-value
// group is two byte-wise integer dots against the q8_1 values, scaled by the group's E4M3 scale and the input's d.
// grid: (row blocks, n / 4096 slices, column groups of NC); with more than one slice the slices add into y
// (zeroed first): two partial sums onto zero give the same float in either order.
constexpr int kNvRPB = 128;      // rows per block (32 warps, 4 rows each)
// E2M1 nibble times 2 as an integer: 0 1 2 3 4 6 8 12, the top bit the sign
__device__ __forceinline__ int e2m1x2(uint32_t n) {
    const int m = n & 7;
    const int v = m < 4 ? m : (2 + (m & 1)) << ((m >> 1) - 1);
    return (n & 8) ? -v : v;
}
template <int NC, int RW = 1>
__global__ void __launch_bounds__(1024) k_nvfp4_q8cols(const uint8_t* W, const uint8_t* S, float scale2, const uint8_t* xq,
                                                      float* y, int n, int rows, int T, int accumulate, int rpb) {
    constexpr int KS = 4096, NB = KS / 32;
    __shared__ __attribute__((aligned(16))) int xs[NC][NB][8];
    __shared__ float xd[NC][NB];
    __shared__ float e4[256];
    __shared__ int kv[16];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, nw = blockDim.x / 32;
    const int k0 = blockIdx.y * KS, c0 = blockIdx.z * NC;
    if (tid < 256) e4[tid] = e4m3f(uint8_t(tid));
    if (tid < 16) { const int v[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}; kv[tid] = v[tid]; }
    for (int idx = tid; idx < NC * NB; idx += blockDim.x) {
        const int c = idx / NB, b = idx % NB;
        if (c0 + c < T) {
            const uint8_t* blk = xq + (size_t(c0 + c) * (n / 32) + k0 / 32 + b) * 36;
            xd[c][b] = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
            const int* q = reinterpret_cast<const int*>(blk + 4);
#pragma unroll
            for (int w = 0; w < 8; ++w) xs[c][b][w] = q[w];
        } else {
            xd[c][b] = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) xs[c][b][w] = 0;
        }
    }
    __syncthreads();
    const int r_end = min(rows, int(blockIdx.x + 1) * rpb);
    // RW rows per warp at once (rows row0, row0 + nw, ...): their weight loads are issued together, so more bytes are
    // in flight per warp (one 16-byte load at a time kept the card at about half its memory speed). Per row the same
    // sums in the same order, so the output does not depend on RW.
    for (int row0 = blockIdx.x * rpb + warp; row0 < r_end; row0 += nw * RW) {
        const uint4* w4[RW];
        const uint16_t* s2[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = min(row0 + r * nw, r_end - 1);
            w4[r] = reinterpret_cast<const uint4*>(W + size_t(row) * (n / 2) + k0 / 2);
            s2[r] = reinterpret_cast<const uint16_t*>(S + size_t(row) * (n / 16) + k0 / 16);
        }
        // (loading a row's chunks up front spilled registers at 1,024 threads a block: 317 vs 153 us)
        float acc[RW][NC];
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[r][c] = 0.f;
        for (int i = lane; i < NB; i += 32) {
            uint4 q[RW];
            uint16_t sc[RW];
#pragma unroll
            for (int r = 0; r < RW; ++r) { q[r] = w4[r][i]; sc[r] = s2[r][i]; }
            float sa[RW], sb[RW];
            int wq[RW][8];
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                sa[r] = e4[sc[r] & 255];
                sb[r] = e4[sc[r] >> 8];
                const uint32_t word[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t bb = word[k];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t u = bb >> (16 * h);
                        wq[r][2 * k + h] = (kv[u & 15] & 255) | ((kv[(u >> 4) & 15] & 255) << 8) | ((kv[(u >> 8) & 15] & 255) << 16) |
                                           ((kv[(u >> 12) & 15] & 255) << 24);
                    }
                }
            }
            // each column's inputs read from LDS once for all RW rows
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int4 x0 = *reinterpret_cast<const int4*>(&xs[c][i][0]), x1 = *reinterpret_cast<const int4*>(&xs[c][i][4]);
                const int xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                const float d = xd[c][i];
#pragma unroll
                for (int r = 0; r < RW; ++r) {
                    int lo = 0, hi = 0;
#pragma unroll
                    for (int w = 0; w < 4; ++w) { lo = STRATA_DP4A(wq[r][w], xv[w], lo); hi = STRATA_DP4A(wq[r][w + 4], xv[w + 4], hi); }
                    acc[r][c] += d * (float(lo) * sa[r] + float(hi) * sb[r]);
                }
            }
        }
        static_assert(NC == 8, "the butterfly below is for 8 columns");
#pragma unroll
        for (int r = 0; r < RW; ++r) {
        const int row = row0 + r * nw;
        float* v = acc[r];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const bool up = lane & 16;
            const float got = __shfl_xor_sync(0xffffffff, up ? v[k] : v[k + 4], 16, 32);
            v[k] = (up ? v[k + 4] : v[k]) + got;
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const bool up = lane & 8;
            const float got = __shfl_xor_sync(0xffffffff, up ? v[k] : v[k + 2], 8, 32);
            v[k] = (up ? v[k + 2] : v[k]) + got;
        }
        {
            const bool up = lane & 4;
            const float got = __shfl_xor_sync(0xffffffff, up ? v[0] : v[1], 4, 32);
            v[0] = (up ? v[1] : v[0]) + got;
        }
        v[0] += __shfl_xor_sync(0xffffffff, v[0], 2, 32);
        v[0] += __shfl_xor_sync(0xffffffff, v[0], 1, 32);
        const int c = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
        if ((lane & 3) == 0 && c0 + c < T && row < r_end) {
            const float val = v[0] * scale2 * 0.5f;
            if (accumulate) atomicAdd(&y[size_t(c0 + c) * rows + row], val);
            else y[size_t(c0 + c) * rows + row] = val;
        }
        }
    }
}

// BF16 times T columns, the inputs staged in LDS 512 values at a time; one warp per row, each lane reading 8 weights
// (16 bytes) per step so a warp reads 512 contiguous bytes
__global__ void __launch_bounds__(256) k_bf16_tiled(const uint16_t* W, const float* x, float* y, int n, int rows, int T) {
    constexpr int KT = 512;
    __shared__ float xt[kTMax][KT];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp;
    float acc[kTMax];
#pragma unroll
    for (int c = 0; c < kTMax; ++c) acc[c] = 0.f;
    for (int k0 = 0; k0 < n; k0 += KT) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < T * KT; idx += 256) xt[idx / KT][idx % KT] = x[size_t(idx / KT) * n + k0 + idx % KT];
        __syncthreads();
        if (row < rows) {
            const uint4* wr = reinterpret_cast<const uint4*>(W + size_t(row) * n + k0);
            for (int i = lane; i < KT / 8; i += 32) {
                const uint4 q = wr[i];
                const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
                float w[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) { w[2 * j] = __uint_as_float(ws[j] << 16); w[2 * j + 1] = __uint_as_float(ws[j] & 0xffff0000u); }
#pragma unroll
                for (int c = 0; c < kTMax; ++c) {
                    if (c >= T) break;
                    const float4 xa = reinterpret_cast<const float4*>(&xt[c][i * 8])[0];
                    const float4 xb = reinterpret_cast<const float4*>(&xt[c][i * 8])[1];
                    acc[c] += w[0] * xa.x + w[1] * xa.y + w[2] * xa.z + w[3] * xa.w + w[4] * xb.x + w[5] * xb.y + w[6] * xb.z + w[7] * xb.w;
                }
            }
        }
    }
    if (row >= rows) return;
#pragma unroll
    for (int c = 0; c < kTMax; ++c) {
        if (c >= T) break;
        float a = acc[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o, 32);
        if (lane == 0) y[size_t(c) * rows + row] = a;
    }
}

// a step's routed slots grouped by expert, in slot order (one thread: at most 16 x 9 slots, deterministic).
// out: *ng groups; g_exp[g], g_cnt[g]; members g_tok[g][m] = slot / k (the token, for the up projection's input) and
// g_slot[g][m] = slot (the output row block, and the down projection's input)
__global__ void k_group_slots(const int32_t* idx, int S, int k, int n_exp, int32_t* ng, int32_t* g_exp, int32_t* g_cnt,
                              int32_t* g_tok, int32_t* g_slot) {
    __shared__ int gid[512];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) gid[e] = -1;
    __syncthreads();
    if (threadIdx.x != 0) return;
    int n = 0;
    for (int j = 0; j < S; ++j) {
        const int e = idx[j];
        int g = gid[e];
        if (g < 0) { g = n++; gid[e] = g; g_exp[g] = e; g_cnt[g] = 0; }
        const int m = g_cnt[g]++;
        g_tok[g * 16 + m] = j / k;
        g_slot[g * 16 + m] = j;
    }
    *ng = n;
}

// ---- the prompt path: chunks of up to kTBig positions through llama.cpp's MMQ and hipBLASLt --------------------
constexpr int kTBig = 4096;      // 8192 read an 8K prompt 5% faster but peaked at 31.4 of 32.6 GB with the drafter + 32K context
constexpr int kAttSub = 256;        // prompt attention runs this many query rows at a time (bounds the score table)
constexpr size_t kMmqTail = 4096;    // zero bytes after every MMQ weight buffer (its tiles read past the last row)

__device__ inline uint16_t f2bf(float f) {     // round to nearest even
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
// ---- FP8 (E4M3) operands for the prompt path's NVFP4 products ----
// four floats to E4M3 (OCP, as hipBLASLt's HIP_R_8F_E4M3 on gfx12), round to nearest even, by the card's converter
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    int v = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    v = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, v, true);
    return (uint32_t) v;
#else
    __trap();
    return 0;
#endif
}
// one block per weight row: the row unpacked (E2M1 x its E4M3 block scale; scale_2 is left to alpha), its largest
// magnitude / 448 as the row scale, the row in E4M3 at that scale
__global__ void __launch_bounds__(256) k_nvfp4_to_e4m3_rows(const uint8_t* W, const uint8_t* S, int K, uint8_t* w8, float* rscale) {
    __shared__ float red[256];
    const int64_t r = blockIdx.x;
    const uint8_t* wr = W + r * (K / 2);
    const uint8_t* sr = S + r * (K / 16);
    // the row's largest from the block scales alone (6 x scale: E2M1's largest; a block whose values stop short of 6
    // only leaves E4M3 headroom unused), so the 4-bit values are read once
    float m = 0.f;
    for (int i = threadIdx.x; i < K / 16; i += 256) m = fmaxf(m, 6.f * fabsf(e4m3f(sr[i])));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    const float rs = red[0] > 0.f ? red[0] / 448.f : 1.f, inv = 1.f / rs;
    for (int i = threadIdx.x; i < K / 4; i += 256) {                 // 4 values (2 bytes) a step
        const uint8_t b0 = wr[2 * i], b1 = wr[2 * i + 1];
        const float sc = e4m3f(sr[(4 * i) / 16]) * 0.5f * inv;
        reinterpret_cast<uint32_t*>(w8 + r * K)[i] = fp8x4(e2m1x2(b0 & 15) * sc, e2m1x2(b0 >> 4) * sc, e2m1x2(b1 & 15) * sc, e2m1x2(b1 >> 4) * sc);
    }
    if (threadIdx.x == 0) rscale[r] = rs;
}
// the largest |x| of n values into *amax (float bits)
__global__ void __launch_bounds__(256) k_amax(const float* x, size_t n, uint32_t* amax) {
    __shared__ float red[256];
    float m = 0.f;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) m = fmaxf(m, fabsf(x[i]));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    if (threadIdx.x == 0 && red[0] > 0.f) atomicMax(amax, __float_as_uint(red[0]));
}
// x in E4M3 at the scale amax / 448
__global__ void k_to_e4m3(const float* x, size_t n, const uint32_t* amax, uint8_t* x8) {
    const float m = __uint_as_float(*amax), inv = m > 0.f ? 448.f / m : 1.f;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n / 4; i += size_t(gridDim.x) * blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        reinterpret_cast<uint32_t*>(x8)[i] = fp8x4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
    }
}
// one block per row of a BF16 matrix (K wide): its largest magnitude / 448 as the row scale, the row in E4M3
__global__ void __launch_bounds__(256) k_bf16_to_e4m3_rows(const uint16_t* w, int K, uint8_t* w8, float* rscale) {
    __shared__ float red[256];
    const int64_t r = blockIdx.x;
    const uint16_t* wr = w + r * K;
    float m = 0.f;
    for (int i = threadIdx.x; i < K; i += 256) m = fmaxf(m, fabsf(__uint_as_float(uint32_t(wr[i]) << 16)));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    const float rs = red[0] > 0.f ? red[0] / 448.f : 1.f, inv = 1.f / rs;
    for (int i = threadIdx.x; i < K / 4; i += 256) {
        const uint2 v = reinterpret_cast<const uint2*>(wr)[i];
        reinterpret_cast<uint32_t*>(w8 + r * K)[i] = fp8x4(__uint_as_float(v.x << 16) * inv, __uint_as_float(v.x & 0xffff0000u) * inv,
                                                          __uint_as_float(v.y << 16) * inv, __uint_as_float(v.y & 0xffff0000u) * inv);
    }
    if (threadIdx.x == 0) rscale[r] = rs;
}
// x = relu(x)^2 in place and its largest value into *amax
__global__ void __launch_bounds__(256) k_relu2_amax(float* x, size_t n, uint32_t* amax) {
    __shared__ float red[256];
    float m = 0.f;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
        const float v = fmaxf(x[i], 0.f), q = v * v;
        x[i] = q;
        m = fmaxf(m, q);
    }
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    if (threadIdx.x == 0 && red[0] > 0.f) atomicMax(amax, __float_as_uint(red[0]));
}
// alpha[n] = row scale x activation scale x scale_2 (amax null: the activations' scales are per row, applied by
// the product's consumers)
__global__ void k_fp8_alpha(const float* rscale, int N, const uint32_t* amax, float s2, float* alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float m = amax ? __uint_as_float(*amax) : 448.f, xs = m > 0.f ? m / 448.f : 1.f;
    if (i < N) alpha[i] = rscale[i] * xs * s2;
}

// ---- FP8 activations with one scale per row ----------------------------------------------------------------
// The producers below write a row in E4M3 at (its largest magnitude / 448) with that scale in xrs; the product
// (Gemm::fp8, alpha = weight row scale only) then leaves each output row short of that factor, which its consumers
// multiply back in (the *rs arguments of k_add_rmsnorm_rows, k_gated_norm_*, k_mamba_conv_tiles, k_conv_state_set,
// k_ssd_dt, k_rows_e4m3). Against one scale a chunk this drops the amax and conversion passes, and small rows keep
// more of FP8's precision.
__device__ __forceinline__ float rw_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ float rw_max32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}
// the residual add and RMSNorm of one row (kHidden) a block of 256 (16 values a thread: four 16-byte loads of each
// input in flight; 1024 threads of 4: ~650 us a 4K chunk, this ~475): h += d * drs + e * ers (scales optional), then
// y (fp32, if given) and/or the row in E4M3 (x8, xrs; if given)
__global__ void __launch_bounds__(256) k_add_rmsnorm_rows(float* h, const float* d, const float* drs, const float* e, const float* ers,
                                                         const uint16_t* w, float* y, uint8_t* x8, float* xrs) {
    constexpr int NT = 256, V = kHidden / 4 / NT;
    __shared__ float ss[NT / 32], sm[NT / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t r = blockIdx.x;
    float4* h4 = reinterpret_cast<float4*>(h + r * kHidden);
    const float4* d4 = reinterpret_cast<const float4*>(d + r * kHidden);
    const float4* e4 = e ? reinterpret_cast<const float4*>(e + r * kHidden) : nullptr;
    const float ds = drs ? drs[r] : 1.f, es = ers ? ers[r] : 1.f;
    float4 v[V], dv[V], ev[V];
#pragma unroll
    for (int j = 0; j < V; ++j) {
        v[j] = h4[tid + j * NT];
        dv[j] = d4[tid + j * NT];
        ev[j] = e4 ? e4[tid + j * NT] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        v[j].x += dv[j].x * ds + ev[j].x * es; v[j].y += dv[j].y * ds + ev[j].y * es;
        v[j].z += dv[j].z * ds + ev[j].z * es; v[j].w += dv[j].w * ds + ev[j].w * es;
        h4[tid + j * NT] = v[j];
        a += v[j].x * v[j].x + v[j].y * v[j].y + v[j].z * v[j].z + v[j].w * v[j].w;
    }
    a = rw_sum32(a);
    if (lane == 0) ss[warp] = a;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; ++i) t += ss[i];
    const float inv = rsqrtf(t / kHidden + kEps);
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        const uint2 wb = reinterpret_cast<const uint2*>(w)[tid + j * NT];
        v[j] = make_float4(bf16f(uint16_t(wb.x)) * (v[j].x * inv), bf16f(uint16_t(wb.x >> 16)) * (v[j].y * inv),
                           bf16f(uint16_t(wb.y)) * (v[j].z * inv), bf16f(uint16_t(wb.y >> 16)) * (v[j].w * inv));
        if (y) reinterpret_cast<float4*>(y + r * kHidden)[tid + j * NT] = v[j];
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[j].x), fabsf(v[j].y)), fmaxf(fabsf(v[j].z), fabsf(v[j].w))));
    }
    if (x8) {
        m = rw_max32(m);
        if (lane == 0) sm[warp] = m;
        __syncthreads();
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < NT / 32; ++i) mx = fmaxf(mx, sm[i]);
        const float sc = mx > 0.f ? mx / 448.f : 1.f, i8 = 1.f / sc;
#pragma unroll
        for (int j = 0; j < V; ++j)
            reinterpret_cast<uint32_t*>(x8 + r * kHidden)[tid + j * NT] = fp8x4(v[j].x * i8, v[j].y * i8, v[j].z * i8, v[j].w * i8);
        if (tid == 0) xrs[r] = sc;
    }
}
// the gated group norm of one row (kMInter, 8 groups of 1024) a block of 1024, into E4M3 with the row's scale: z is
// proj's gate times its row scale prs (if given)
__global__ void __launch_bounds__(1024) k_gated_norm_rows(const float* y, const float* proj, const float* prs, const uint16_t* w,
                                                         uint8_t* x8, float* xrs) {
    __shared__ float ss[32], sm[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t r = blockIdx.x;
    const float zs = prs ? prs[r] : 1.f;
    const float* yr = y + r * kMInter + tid * 8;
    const float* zr = proj + r * kInProj + tid * 8;
    float v[8];
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float4 yy = *reinterpret_cast<const float4*>(yr + 4 * j), zz4 = *reinterpret_cast<const float4*>(zr + 4 * j);
        const float yv[4] = {yy.x, yy.y, yy.z, yy.w}, zv[4] = {zz4.x, zz4.y, zz4.z, zz4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float zz = zv[i] * zs;
            v[4 * j + i] = yv[i] * (zz / (1.f + expf(-zz)));
            a += v[4 * j + i] * v[4 * j + i];
        }
    }
    a = rw_sum32(a);
    if (lane == 0) ss[warp] = a;
    __syncthreads();
    const int g4 = (warp / 4) * 4;                                    // a group is 4 warps (128 threads x 8)
    const float inv = rsqrtf((ss[g4] + ss[g4 + 1] + ss[g4 + 2] + ss[g4 + 3]) / (kMInter / kMGroups) + kEps);
    const uint4 wb = *reinterpret_cast<const uint4*>(w + tid * 8);
    const uint32_t ww[4] = {wb.x, wb.y, wb.z, wb.w};
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = v[i] * inv * bf16f(uint16_t(ww[i / 2] >> (16 * (i % 2))));
        m = fmaxf(m, fabsf(v[i]));
    }
    m = rw_max32(m);
    if (lane == 0) sm[warp] = m;
    __syncthreads();
    const float mx = rw_max32(sm[lane]), sc = mx > 0.f ? mx / 448.f : 1.f, i8 = 1.f / sc;
    *reinterpret_cast<uint2*>(x8 + r * kMInter + tid * 8) = make_uint2(fp8x4(v[0] * i8, v[1] * i8, v[2] * i8, v[3] * i8),
                                                                       fp8x4(v[4] * i8, v[5] * i8, v[6] * i8, v[7] * i8));
    if (tid == 0) xrs[r] = sc;
}
// one row of K (times its row scale rs, if given; relu(.)^2 if RELU2) a block of NT, into E4M3 with the row's scale
template <int K, int NT, bool RELU2>
__global__ void __launch_bounds__(NT) k_rows_e4m3(const float* x, const float* rs, uint8_t* x8, float* xrs) {
    static_assert(K % (4 * NT) == 0 && NT % 32 == 0, "whole float4 steps");
    constexpr int V = K / 4 / NT;
    __shared__ float sm[NT / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t r = blockIdx.x;
    const float s = rs ? rs[r] : 1.f;
    float4 v[V];
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        float4 a = reinterpret_cast<const float4*>(x + r * K)[tid + j * NT];
        a.x *= s; a.y *= s; a.z *= s; a.w *= s;
        if (RELU2) {
            a.x = fmaxf(a.x, 0.f); a.y = fmaxf(a.y, 0.f); a.z = fmaxf(a.z, 0.f); a.w = fmaxf(a.w, 0.f);
            a.x *= a.x; a.y *= a.y; a.z *= a.z; a.w *= a.w;
        }
        v[j] = a;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
    }
    m = rw_max32(m);
    if (lane == 0) sm[warp] = m;
    __syncthreads();
    float mx = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; ++i) mx = fmaxf(mx, sm[i]);
    const float sc = mx > 0.f ? mx / 448.f : 1.f, i8 = 1.f / sc;
#pragma unroll
    for (int j = 0; j < V; ++j) reinterpret_cast<uint32_t*>(x8 + r * K)[tid + j * NT] = fp8x4(v[j].x * i8, v[j].y * i8, v[j].z * i8, v[j].w * i8);
    if (tid == 0) xrs[r] = sc;
}

__global__ void k_relu2_to_bf16(const float* x, uint16_t* y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) { const float v = fmaxf(x[i], 0.f); y[i] = f2bf(v * v); }
}
__global__ void k_to_bf16(const float* x, uint16_t* y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] = f2bf(x[i]);
}
// NVFP4 rows to BF16 WITHOUT the per-tensor scale_2: E2M1 (3 bits) times E4M3 (4 significant bits) fits BF16's 8
// exactly, so the only rounding is the GEMM's own; scale_2 is applied to the product (k_scale)
__global__ void k_nvfp4_to_bf16(const uint8_t* W, const uint8_t* S, uint16_t* out, size_t n_bytes) {
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x; j < n_bytes; j += size_t(gridDim.x) * blockDim.x) {
        const uint8_t b = W[j];
        const float s = e4m3f(S[j / 8]);             // 16 values = 8 bytes per scale, rows included
        out[2 * j] = f2bf(kE2M1[b & 15] * s);
        out[2 * j + 1] = f2bf(kE2M1[b >> 4] * s);
    }
}
__global__ void k_scale(float* y, float s, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] *= s;
}

// the chunk's S = T * k routed slots sorted by expert (stable, so deterministic): bounds[e]..bounds[e+1] are expert
// e's sorted rows; src[r] = the token of sorted row r (the up projection's input row); rowof[slot] = its sorted row
__global__ void k_sort_slots(const int32_t* idx, int S, int k, int n_exp, int32_t* bounds, int32_t* src, int32_t* rowof) {
    __shared__ int cnt[512], at[512];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < S; j += blockDim.x) atomicAdd(&cnt[idx[j]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int e = 0; e < n_exp; ++e) { bounds[e] = a; at[e] = a; a += cnt[e]; }
        bounds[n_exp] = a;
        for (int j = 0; j < S; ++j) { const int r = at[idx[j]]++; src[r] = j / k; rowof[j] = r; }
    }
}
// eo[t] = sum_j w[t][j] * Yd[rowof[t * k + j]]
__global__ void k_wsum_sorted(float* eo, const float* Yd, const int32_t* rowof, const float* w, int k, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < k; ++j) a += w[t * k + j] * Yd[size_t(rowof[t * k + j]) * n + i];
        eo[size_t(t) * n + i] = a;
    }
}

// attention for query rows [t_off, t_off + gridDim) of a chunk (scores for kTMax rows at a time)
__global__ void k_att_scores_off(const float* q, const float* K, float* sc, const int* pos0, int max_ctx, int t_off) {
    const int h = blockIdx.y, tl = blockIdx.z, t = t_off + tl, L = *pos0 + t + 1;
    const float* qq = q + (size_t(t) * kAHeads + h) * kAHeadDim;
    float* s = sc + (size_t(tl) * kAHeads + h) * max_ctx;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < L; j += gridDim.x * blockDim.x) {
        const float* k = K + (size_t(h / (kAHeads / kAKv)) * max_ctx + j) * kAHeadDim;
        float a = 0.f;
        for (int d = 0; d < kAHeadDim; ++d) a += qq[d] * k[d];
        s[j] = a * rsqrtf(float(kAHeadDim));
    }
}
__global__ void k_att_softmax_off(float* sc, const int* pos0, int max_ctx, int t_off) {
    __shared__ float sh[256];
    const int h = blockIdx.x, tl = blockIdx.y, L = *pos0 + t_off + tl + 1;
    float* s = sc + (size_t(tl) * kAHeads + h) * max_ctx;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += 256) m = fmaxf(m, s[j]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0];
    __syncthreads();
    float a = 0.f;
    for (int j = threadIdx.x; j < L; j += 256) { const float e = expf(s[j] - m); s[j] = e; a += e; }
    const float inv = 1.f / block_sum<256>(a, sh);
    for (int j = threadIdx.x; j < L; j += 256) s[j] *= inv;
}
__global__ void k_att_out_off(const float* sc, const float* V, float* o, const int* pos0, int max_ctx, int t_off) {
    const int h = blockIdx.x, tl = blockIdx.y, t = t_off + tl, d = threadIdx.x, L = *pos0 + t + 1;
    const float* v = V + size_t(h / (kAHeads / kAKv)) * max_ctx * kAHeadDim + d;
    const float* s = sc + (size_t(tl) * kAHeads + h) * max_ctx;
    float a = 0.f;
    for (int j = 0; j < L; ++j) a += s[j] * v[size_t(j) * kAHeadDim];
    o[(size_t(t) * kAHeads + h) * kAHeadDim + d] = a;
}

// ---- attention, tiled with a running softmax (no score table) ---------------------------------------------------
// Query rows t of the step (q: (T, kAHeads, kAHeadDim)), each attending to keys 0 .. *pos0 + t. Block = (16-row query
// tile, head, key split); 128 threads, one per head dim. Keys go through LDS 32 at a time. With splits == 1 the block
// writes the output; otherwise it writes its partial (running max, sum, unnormalised output) for k_flash_combine.
constexpr int kFaRows = 16, kFaKeys = 32, kFaSplitKeys = 512;
__global__ void __launch_bounds__(128) k_flash(const float* q, const float* K, const float* V, float* out,
                                               float* part, const int* pos0, int max_ctx, int T, int splits) {
    __shared__ float qs[kFaRows][kAHeadDim];
    __shared__ float ks[kFaKeys][kAHeadDim + 1];
    __shared__ float vs[kFaKeys][kAHeadDim];
    __shared__ float ps[kFaRows][kFaKeys];
    __shared__ float mrow[kFaRows], lrow[kFaRows], alpha[kFaRows];
    const int tile = blockIdx.x, h = blockIdx.y, split = blockIdx.z, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int t0 = tile * kFaRows, nr = min(kFaRows, T - t0), p0 = *pos0;
    const int kvh = h / (kAHeads / kAKv);
    const float scale = rsqrtf(float(kAHeadDim));
    for (int r = 0; r < kFaRows; ++r) qs[r][tid] = r < nr ? q[(size_t(t0 + r) * kAHeads + h) * kAHeadDim + tid] * scale : 0.f;
    if (tid < kFaRows) { mrow[tid] = -1e30f; lrow[tid] = 0.f; }
    float acc[kFaRows];
#pragma unroll
    for (int r = 0; r < kFaRows; ++r) acc[r] = 0.f;
    const int L = p0 + t0 + nr;                                       // keys the last row of the tile sees
    int k_begin = 0, k_end = L;
    if (splits > 1) { k_begin = split * kFaSplitKeys; k_end = min(L, k_begin + kFaSplitKeys); }
    const float* Kh = K + size_t(kvh) * max_ctx * kAHeadDim;
    const float* Vh = V + size_t(kvh) * max_ctx * kAHeadDim;
    __syncthreads();
    for (int k0 = k_begin; k0 < k_end; k0 += kFaKeys) {
        const int nk = min(kFaKeys, k_end - k0);
        for (int j = 0; j < kFaKeys; ++j) {
            ks[j][tid] = j < nk ? Kh[size_t(k0 + j) * kAHeadDim + tid] : 0.f;
            vs[j][tid] = j < nk ? Vh[size_t(k0 + j) * kAHeadDim + tid] : 0.f;
        }
        __syncthreads();
        // scores: warp w takes rows w, w+4, w+8, w+12; lane = key
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = warp + 4 * i;
            if (r >= nr) break;
            float s = -INFINITY;
            if (r < nr && lane < nk && k0 + lane <= p0 + t0 + r) {
                s = 0.f;
                for (int d = 0; d < kAHeadDim; ++d) s += qs[r][d] * ks[lane][d];
            }
            float m = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o, 32));
            const float m_old = mrow[r], m_new = fmaxf(m_old, m);
            const float p = s == -INFINITY ? 0.f : expf(s - m_new);
            float sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o, 32);
            ps[r][lane] = p;
            __syncwarp();
            if (lane == 0) { const float a = expf(m_old - m_new); alpha[r] = a; lrow[r] = lrow[r] * a + sum; mrow[r] = m_new; }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kFaRows; ++r) {
            if (r >= nr) break;
            float a = acc[r] * alpha[r];
            for (int j = 0; j < nk; ++j) a += ps[r][j] * vs[j][tid];
            acc[r] = a;
        }
        __syncthreads();
    }
    for (int r = 0; r < nr; ++r) {
        const size_t o = (size_t(t0 + r) * kAHeads + h);
        if (splits == 1) out[o * kAHeadDim + tid] = acc[r] / lrow[r];
        else {
            float* pp = part + (o * splits + split) * (kAHeadDim + 2);
            pp[tid] = acc[r];
            if (tid == 0) { pp[kAHeadDim] = mrow[r]; pp[kAHeadDim + 1] = lrow[r]; }
        }
    }
}
// the splits of each (row, head) merged: out = sum_s acc_s e^(m_s - M) / sum_s l_s e^(m_s - M)
__global__ void __launch_bounds__(512) k_flash_combine(const float* part, float* out, int splits) {
    // 4 groups of 128 (one per dim) take every 4th split with a running max, then merge (one group over 128+
    // splits was a 50 us latency chain a call)
    __shared__ float sm[4], sd[4], sn[4][kAHeadDim];
    const size_t o = size_t(blockIdx.y) * kAHeads + blockIdx.x;      // (row, head)
    const int d = threadIdx.x & 127, grp = threadIdx.x >> 7;
    const float* pp = part + o * splits * (kAHeadDim + 2);
    float M = -INFINITY, num = 0.f, den = 0.f;
    for (int s = grp; s < splits; s += 4) {
        const float* p = pp + s * (kAHeadDim + 2);
        const float ls = p[kAHeadDim + 1];
        if (!(ls > 0.f)) continue;
        const float ms = p[kAHeadDim], mn = fmaxf(M, ms), a = expf(M - mn), b = expf(ms - mn);
        num = num * a + p[d] * b;
        den = den * a + ls * b;
        M = mn;
    }
    sn[grp][d] = num;
    if (d == 0) { sm[grp] = M; sd[grp] = den; }
    __syncthreads();
    if (grp) return;
    float Mx = sm[0];
    for (int i = 1; i < 4; ++i) Mx = fmaxf(Mx, sm[i]);
    float n = 0.f, dn = 0.f;
    for (int i = 0; i < 4; ++i) {
        const float w = sd[i] > 0.f ? expf(sm[i] - Mx) : 0.f;
        n += sn[i][d] * w;
        dn += sd[i] * w;
    }
    out[o * kAHeadDim + d] = n / dn;
}

// the chunk's slots sorted by expert, one thread per expert (stable: each expert takes its slots in slot order);
// the pick list goes through LDS 8,192 at a time
__global__ void __launch_bounds__(512) k_sort_slots_par(const int32_t* idx, int S, int k, int n_exp, int32_t* bounds, int32_t* src, int32_t* rowof) {
    constexpr int CH = 8192;
    __shared__ int cnt[512];
    __shared__ int16_t ch[CH];
    const int e = threadIdx.x;
    int c = 0;
    for (int j0 = 0; j0 < S; j0 += CH) {
        const int n = min(CH, S - j0);
        __syncthreads();
        for (int j = e; j < n; j += 512) ch[j] = int16_t(idx[j0 + j]);
        __syncthreads();
        if (e < n_exp) for (int j = 0; j < n; ++j) c += ch[j] == e;
    }
    cnt[e] = c;
    __syncthreads();
    if (e == 0) { int a = 0; for (int i = 0; i < n_exp; ++i) { const int v = cnt[i]; cnt[i] = a; a += v; } bounds[n_exp] = a; }
    __syncthreads();
    int r = cnt[e];
    if (e < n_exp) bounds[e] = r;
    for (int j0 = 0; j0 < S; j0 += CH) {
        const int n = min(CH, S - j0);
        __syncthreads();
        for (int j = e; j < n; j += 512) ch[j] = int16_t(idx[j0 + j]);
        __syncthreads();
        if (e < n_exp) for (int j = 0; j < n; ++j) if (ch[j] == e) { src[r] = (j0 + j) / k; rowof[j0 + j] = r; ++r; }
    }
}

// The same sort (stable, so bitwise the same bounds / src / rowof) in three parallel steps; k_sort_slots_par walked
// every slot once per expert thread, 1.8 ms on a 4,096-token chunk. Per 512-slot chunk the experts' counts; per
// expert the running offsets over the chunks and the bounds; each slot's place = its expert's offset in its chunk +
// the earlier slots of its chunk with the same expert.
constexpr int kSortChunk = 512;
__global__ void __launch_bounds__(512) k_sort_count(const int32_t* idx, int S, int n_exp, int32_t* ccnt) {
    __shared__ int cnt[512];
    const int tid = threadIdx.x, j = blockIdx.x * kSortChunk + tid;
    cnt[tid] = 0;
    __syncthreads();
    if (j < S) atomicAdd(&cnt[idx[j]], 1);
    __syncthreads();
    if (tid < n_exp) ccnt[size_t(blockIdx.x) * 512 + tid] = cnt[tid];
}
__global__ void __launch_bounds__(512) k_sort_offsets(int32_t* ccnt, int nch, int n_exp, int32_t* bounds) {
    __shared__ int sc[512];
    const int e = threadIdx.x;
    int tot = 0;
    if (e < n_exp) for (int c = 0; c < nch; ++c) tot += ccnt[size_t(c) * 512 + e];
    sc[e] = tot;
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {                        // inclusive scan over the experts
        const int v = e >= off ? sc[e - off] : 0;
        __syncthreads();
        sc[e] += v;
        __syncthreads();
    }
    if (e < n_exp) {
        int run = sc[e] - tot;
        bounds[e] = run;
        for (int c = 0; c < nch; ++c) { const int v = ccnt[size_t(c) * 512 + e]; ccnt[size_t(c) * 512 + e] = run; run += v; }
    }
    if (e == n_exp - 1) bounds[n_exp] = sc[e];
}
__global__ void __launch_bounds__(512) k_sort_place(const int32_t* idx, int S, int k, const int32_t* ccnt, int32_t* src, int32_t* rowof) {
    __shared__ int sidx[kSortChunk];
    const int tid = threadIdx.x, j = blockIdx.x * kSortChunk + tid;
    sidx[tid] = j < S ? idx[j] : -1;
    __syncthreads();
    if (j >= S) return;
    const int e = sidx[tid];
    int r = ccnt[size_t(blockIdx.x) * 512 + e];
    for (int i = 0; i < tid; ++i) r += sidx[i] == e;
    src[r] = j / k;
    rowof[j] = r;
}

// k_mamba_conv_many for a prompt chunk, kConvTile positions per thread (the serial one walked all T positions of a
// channel on one thread: 0.94 ms at T = 4,096). Each tile starts from the three inputs before it (the stored state
// for the first). The same sums, so the same outputs; the state after T is written by k_conv_state_set (after this
// kernel: the first tile reads the old state).
constexpr int kConvTile = 64;
__global__ void k_mamba_conv_tiles(const float* proj, const float* state, const uint16_t* cw, const uint16_t* cb, float* out, int T,
                                   const float* prs = nullptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConvDim) return;
    const int t0 = blockIdx.y * kConvTile, t1 = min(T, t0 + kConvTile);
    const float* st = state + size_t(c) * (kConvK - 1);
    // (prs: proj's row scales, when in_proj ran in FP8 with per-row activation scales)
    auto in = [&](int t) { return t < 0 ? st[t + 3] : proj[size_t(t) * kInProj + kMInter + c] * (prs ? prs[t] : 1.f); };
    float v0 = in(t0 - 3), v1 = in(t0 - 2), v2 = in(t0 - 1);
    const float w0 = bf16f(cw[c * kConvK]), w1 = bf16f(cw[c * kConvK + 1]), w2 = bf16f(cw[c * kConvK + 2]), w3 = bf16f(cw[c * kConvK + 3]);
    const float b = bf16f(cb[c]);
    for (int t = t0; t < t1; ++t) {
        const float x = in(t);
        const float a = b + w0 * v0 + w1 * v1 + w2 * v2 + w3 * x;
        out[size_t(t) * kConvDim + c] = a / (1.f + expf(-a));
        v0 = v1; v1 = v2; v2 = x;
    }
}
// the conv state after T >= 3 positions: their last three inputs
__global__ void k_conv_state_set(const float* proj, float* state, int T, const float* prs = nullptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConvDim) return;
    float* st = state + size_t(c) * (kConvK - 1);
    for (int i = 0; i < 3; ++i) st[i] = proj[size_t(T - 3 + i) * kInProj + kMInter + c] * (prs ? prs[T - 3 + i] : 1.f);
}

// ---- attention v2 ----------------------------------------------------------------------------------------------
// Writing / small steps: query head h, split z. The keys go in runs of kDecRun at fixed positions, run c to split
// c % splits, so every split has work at any context length (256-key splits left most blocks empty and the rest
// grinding through 256 keys) and a key's place in the sums does not depend on T: one row at a time and a checked
// window of 8 add up identically. 4 warps, each takes every 4th key, lane l holds dims 4l..4l+3; T <= kTMax rows
// share each loaded key. Partials (acc, max, sum) per (row, head, split) as k_flash.
// NT >= T rows of running sums per thread, fixed at compile time so they stay in registers (a runtime-sized set
// spilled to scratch: 272 bytes, 36 us a call)
constexpr int kDecSplits = 16, kDecRun = 32;
template <int NT>
__global__ void __launch_bounds__(128) k_attn_dec(const float* q, const uint16_t* K, const uint16_t* V, float* part,
                                                  const int* pos0, int max_ctx, int T, int splits) {
    // LDS sized for T rows (dynamic): qs[T][128], wacc[4][T][128], wm[4][T], wl[4][T]
    extern __shared__ float dsh[];
    float* qs_ = dsh;
    float* wacc_ = qs_ + T * kAHeadDim;
    float* wm_ = wacc_ + 4 * T * kAHeadDim;
    float* wl_ = wm_ + 4 * T;
    auto qs = [&](int r) { return qs_ + r * kAHeadDim; };
    const int h = blockIdx.x, z = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int p0 = *pos0, kvh = h / (kAHeads / kAKv);
    const float scale = rsqrtf(float(kAHeadDim));
    for (int i = tid; i < T * kAHeadDim; i += 128) qs_[i] = q[(size_t(i / kAHeadDim) * kAHeads + h) * kAHeadDim + i % kAHeadDim] * scale;
    __syncthreads();
    float m[NT], l[NT], acc[NT][4];
#pragma unroll
    for (int r = 0; r < NT; ++r) { m[r] = -1e30f; l[r] = 0.f; acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f; }
    const uint16_t* Kh = K + size_t(kvh) * max_ctx * kAHeadDim;       // (max_ctx: the cache's rows a head)
    const uint16_t* Vh = V + size_t(kvh) * max_ctx * kAHeadDim;
    for (int j0 = z * kDecRun; j0 < p0 + T; j0 += splits * kDecRun)
    for (int j = j0 + warp; j < min(p0 + T, j0 + kDecRun); j += 4) {
        const uint2 kb = reinterpret_cast<const uint2*>(Kh + size_t(j) * kAHeadDim)[lane];
        const uint2 vb = make_uint2(uint32_t(Vh[vt_at(j, lane * 4)]) | (uint32_t(Vh[vt_at(j, lane * 4 + 1)]) << 16),
                                    uint32_t(Vh[vt_at(j, lane * 4 + 2)]) | (uint32_t(Vh[vt_at(j, lane * 4 + 3)]) << 16));
        const float4 kv = make_float4(__uint_as_float(kb.x << 16), __uint_as_float(kb.x & 0xffff0000u),
                                      __uint_as_float(kb.y << 16), __uint_as_float(kb.y & 0xffff0000u));
        const float4 vv = make_float4(__uint_as_float(vb.x << 16), __uint_as_float(vb.x & 0xffff0000u),
                                      __uint_as_float(vb.y << 16), __uint_as_float(vb.y & 0xffff0000u));
#pragma unroll
        for (int r = 0; r < NT; ++r) {
            if (r >= T) break;
            if (j > p0 + r) continue;                                 // causal: row r sees keys 0 .. p0 + r
            const float4 qq = reinterpret_cast<const float4*>(qs(r))[lane];
            float s = qq.x * kv.x + qq.y * kv.y + qq.z * kv.z + qq.w * kv.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o, 32);
            const float mn = fmaxf(m[r], s), a = expf(m[r] - mn), p = expf(s - mn);
            l[r] = l[r] * a + p;
            m[r] = mn;
            acc[r][0] = acc[r][0] * a + p * vv.x; acc[r][1] = acc[r][1] * a + p * vv.y;
            acc[r][2] = acc[r][2] * a + p * vv.z; acc[r][3] = acc[r][3] * a + p * vv.w;
        }
    }
    // the four warps' partials merged in LDS
#pragma unroll
    for (int r = 0; r < NT; ++r) {
        if (r >= T) break;
        if (lane == 0) { wm_[warp * T + r] = m[r]; wl_[warp * T + r] = l[r]; }
        reinterpret_cast<float4*>(wacc_ + (warp * T + r) * kAHeadDim)[lane] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
    __syncthreads();
    for (int r = 0; r < T; ++r) {
        float M = -1e30f;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, wm_[w * T + r]);
        float num = 0.f, den = 0.f;
        for (int w = 0; w < 4; ++w) {
            const float e = wl_[w * T + r] > 0.f ? expf(wm_[w * T + r] - M) : 0.f;
            num += wacc_[(w * T + r) * kAHeadDim + tid] * e;
            den += wl_[w * T + r] * e;
        }
        float* pp = part + ((size_t(r) * kAHeads + h) * splits + z) * (kAHeadDim + 2);
        pp[tid] = num;
        if (tid == 0) { pp[kAHeadDim] = M; pp[kAHeadDim + 1] = den; }
    }
}

// Prompt chunks: attention as two hipBLASLt products per KV head (see Model::attention_big).
// K rows [0, L) of KV head g to BF16, (Lp x 128), zero rows past L
__global__ void k_k_bf16(const uint16_t* K, uint16_t* out, const int* pos0, int T, int max_ctx, int Lp) {
    const int L = *pos0 + T;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(Lp) * kAHeadDim; i += size_t(gridDim.x) * blockDim.x) {
        const int j = int(i / kAHeadDim);
        out[i] = j < L ? K[i] : 0;
    }
}
// V rows [0, L) of KV head g transposed to BF16, (128 x Lp)
__global__ void k_vt_bf16(const uint16_t* V, uint16_t* out, const int* pos0, int T, int Lp) {
    const int L = *pos0 + T;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(Lp) * kAHeadDim; i += size_t(gridDim.x) * blockDim.x) {
        const int d = int(i / Lp), j = int(i % Lp);
        out[i] = j < L ? V[vt_at(j, d)] : 0;
    }
}
// the 16 query heads of KV head g to BF16, rows (head-in-group, t): (16 T x 128), the 1/sqrt(128) folded in
__global__ void k_q_bf16(const float* q, uint16_t* out, int g, int T, int t_off) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(16) * T * kAHeadDim; i += size_t(gridDim.x) * blockDim.x) {
        const int d = int(i % kAHeadDim), row = int(i / kAHeadDim), hl = row / T, t = t_off + row % T;
        out[i] = f2bf(q[(size_t(t) * kAHeads + g * 16 + hl) * kAHeadDim + d] * rsqrtf(float(kAHeadDim)));
    }
}
// Prompt attention in one pass on the BF16 matrix cores (gfx12), the scores never leaving the chip. Block: KV head
// blockIdx.y, 8 consecutive positions (one per warp); a warp's 16 rows are its position's 16 query heads (one WMMA tile,
// one causal limit). Keys go 32 at a time through LDS (K rows and V transposed, BF16 as k_k_bf16 / k_vt_bf16 write
// them), shared by the 8 warps; online softmax in fp32 per row; P to BF16 for the P.V product. The same BF16 operands
// as the two-GEMM path (k_q_bf16, the 1/sqrt(128) here applied to the fp32 scores instead).
typedef short att_v8s __attribute__((ext_vector_type(8)));
typedef float att_v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ att_v8f att_wmma(const att_v8s& a, const att_v8s& b, const att_v8f& c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    __trap();
    return c;
#endif
}
constexpr int kFwKeys = 32, kFwPos = 8, kFwKS = kAHeadDim + 8, kFwVS = kFwKeys + 8;   // LDS row strides (bf16)
// The transposed form (measured 61 -> 92 TFLOPS at a 64K context): S^T = K Q^T (keys as rows, the 16 heads as
// columns), so a lane holds 8 keys of ONE head (l16): the softmax is per lane (one shuffle with lane ^ 16 for the
// max, exp2 in the log2 domain), and P^T in that layout is already the B operand of O^T = V^T P^T - no LDS round
// trip for P, each lane rescales only its own head's outputs (and only when some row's max moved).
__global__ void __launch_bounds__(kFwPos * 32) k_attn_flash(const float* __restrict__ q, const uint16_t* __restrict__ kb, size_t k_stride,
                                                         const uint16_t* __restrict__ vtb, size_t kv_stride, const int* pos0,
                                                         int T, int Lp, float* __restrict__ y) {
    __shared__ __align__(16) uint16_t sk[kFwKeys][kFwKS];
    __shared__ __align__(16) uint16_t sv[kAHeadDim][kFwVS];
    const int g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, l16 = lane & 15, g8 = lane >> 4;
    const int t0 = blockIdx.x * kFwPos, t = t0 + warp, P0 = *pos0;
    const bool live = t < T;
    const int p = P0 + t;
    const int pmax = P0 + min(T - 1, t0 + kFwPos - 1);
    const uint16_t* K = kb + g * k_stride;
    const uint16_t* V = vtb + g * kv_stride;
    // Q^T as B operands: lane holds head l16's dims kk * 16 + g8 * 8 .. + 8
    // (scaled by 1/sqrt(128) and log2(e) before the rounding: the scores come out in the log2 domain, for exp2)
    att_v8s qb[kAHeadDim / 16];
    const float qs = rsqrtf(float(kAHeadDim)) * 1.4426950408889634f;
#pragma unroll
    for (int kk = 0; kk < kAHeadDim / 16; ++kk) {
        uint32_t w[4] = {0, 0, 0, 0};
        if (live) {
            const float* qr = q + (size_t(t) * kAHeads + g * 16 + l16) * kAHeadDim + kk * 16 + g8 * 8;
            float4 a = reinterpret_cast<const float4*>(qr)[0], b = reinterpret_cast<const float4*>(qr)[1];
            a.x *= qs; a.y *= qs; a.z *= qs; a.w *= qs; b.x *= qs; b.y *= qs; b.z *= qs; b.w *= qs;
            w[0] = f2bf(a.x) | (uint32_t(f2bf(a.y)) << 16); w[1] = f2bf(a.z) | (uint32_t(f2bf(a.w)) << 16);
            w[2] = f2bf(b.x) | (uint32_t(f2bf(b.y)) << 16); w[3] = f2bf(b.z) | (uint32_t(f2bf(b.w)) << 16);
        }
        qb[kk] = __builtin_bit_cast(att_v8s, make_uint4(w[0], w[1], w[2], w[3]));
    }
    att_v8f o[kAHeadDim / 16];                                     // O^T: lane = head l16, dims d * 16 + g8 * 8 + i
#pragma unroll
    for (int d = 0; d < kAHeadDim / 16; ++d) o[d] = att_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY, l = 0.f;                                  // head l16: running max, this lane's sum
    for (int j0 = 0; j0 <= pmax; j0 += kFwKeys) {
        __syncthreads();
        for (int i = tid; i < kFwKeys * (kAHeadDim / 8); i += kFwPos * 32) {
            const int r = i / (kAHeadDim / 8), c = i % (kAHeadDim / 8);
            *reinterpret_cast<uint4*>(&sk[r][c * 8]) = *reinterpret_cast<const uint4*>(K + size_t(j0 + r) * kAHeadDim + c * 8);
        }
        for (int i = tid; i < kAHeadDim * (kFwKeys / 8); i += kFwPos * 32) {
            const int d = i / (kFwKeys / 8), c = i % (kFwKeys / 8);
            *reinterpret_cast<uint4*>(&sv[d][c * 8]) = *reinterpret_cast<const uint4*>(V + (size_t(j0 / 32) * kAHeadDim + d) * 32 + c * 8);
        }
        __syncthreads();
        if (!live || j0 > p) continue;
        att_v8f st[2];                                             // S^T: keys j0 + nt * 16 + g8 * 8 + i, head l16
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            st[nt] = att_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < kAHeadDim / 16; ++kk)
                st[nt] = att_wmma(*reinterpret_cast<const att_v8s*>(&sk[nt * 16 + l16][kk * 16 + g8 * 8]), qb[kk], st[nt]);
        }
        float v[2][8], mx = -INFINITY;
        if (j0 + kFwKeys - 1 > p) {                                // (the causal mask on the run that reaches p only)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int key = j0 + nt * 16 + g8 * 8 + i;
                    v[nt][i] = key > p ? -INFINITY : st[nt][i];
                    mx = fmaxf(mx, v[nt][i]);
                }
        } else {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[nt][i] = st[nt][i];
                    mx = fmaxf(mx, v[nt][i]);
                }
        }
        // (the other half's max by permlanex16: a shuffle went through LDS, a wait on the softmax's critical path)
        mx = fmaxf(mx, __int_as_float((int) __builtin_amdgcn_permlanex16((unsigned) __float_as_int(mx), (unsigned) __float_as_int(mx),
                                                                            0x76543210u, 0xfedcba98u, false, false)));
        // (the bare v_exp_f32: exp2f's guard for results below the normal range is several instructions an exp, and
        // such results only need to come out as 0)
        const float mn = fmaxf(m, mx), alpha = __builtin_amdgcn_exp2f(m - mn);
        m = mn;
        att_v8s pb[2];
        float sum = 0.f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            uint32_t w[4];
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const float e0 = __builtin_amdgcn_exp2f(v[nt][i] - mn), e1 = __builtin_amdgcn_exp2f(v[nt][i + 1] - mn);
                sum += e0 + e1;
                // (to BF16 by adding half a unit and keeping the top halves: e is finite and in [0, 1])
                w[i / 2] = __builtin_amdgcn_perm(__float_as_uint(e1) + 0x8000u, __float_as_uint(e0) + 0x8000u, 0x07060302u);
            }
            pb[nt] = __builtin_bit_cast(att_v8s, make_uint4(w[0], w[1], w[2], w[3]));
        }
        l = l * alpha + sum;
        if (__any(alpha != 1.f))                                   // (rare after the first runs)
#pragma unroll
            for (int d = 0; d < kAHeadDim / 16; ++d) o[d] *= alpha;
#pragma unroll
        for (int d = 0; d < kAHeadDim / 16; ++d)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                o[d] = att_wmma(*reinterpret_cast<const att_v8s*>(&sv[d * 16 + l16][nt * 16 + g8 * 8]), pb[nt], o[d]);
    }
    if (!live) return;
    const float inv = 1.f / (l + __shfl_xor_sync(0xffffffffu, l, 16, 32));
    float* yr = y + (size_t(t) * kAHeads + g * 16 + l16) * kAHeadDim + g8 * 8;
#pragma unroll
    for (int d = 0; d < kAHeadDim / 16; ++d) {
        *reinterpret_cast<float4*>(yr + d * 16) = make_float4(o[d][0] * inv, o[d][1] * inv, o[d][2] * inv, o[d][3] * inv);
        *reinterpret_cast<float4*>(yr + d * 16 + 4) = make_float4(o[d][4] * inv, o[d][5] * inv, o[d][6] * inv, o[d][7] * inv);
    }
}

// Writing / small steps on the matrix cores: one block per (key split, KV head) serves all 16 query heads of the KV
// head from one read of each 32-key run (k_attn_dec read the cache once per query head: 16 times a layer). Both
// operands come straight from the cache (K rows; V in its transposed runs): each warp is independent, only its P
// goes through LDS. Warp w: position w / PH, run phase w % PH; position t's runs c = z * PH + phase + k * (splits *
// PH) - each (split, phase) a partial (unnormalized output, running max, sum) for k_flash_combine over splits * PH.
template <int NW, int PH>
__global__ void __launch_bounds__(NW * 32) k_attn_dec_wm(const float* __restrict__ q, const uint16_t* __restrict__ K,
                                                         const uint16_t* __restrict__ V, float* __restrict__ part,
                                                         const int* pos0, int kv_rows, int T, int splits) {
    __shared__ __align__(16) uint16_t sp[NW][16][kFwVS];
    const int z = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, l16 = lane & 15, g8 = lane >> 4;
    const int t = warp / PH, ph = warp % PH, P0 = *pos0, p = P0 + t;
    if (t >= T) return;                                            // (warps are independent: no block barrier)
    const int nsp = splits * PH, sidx = z * PH + ph;
    const uint16_t* Kh = K + size_t(g) * kv_rows * kAHeadDim;
    const uint16_t* Vh = V + size_t(g) * kv_rows * kAHeadDim;
    att_v8s qa[kAHeadDim / 16];
#pragma unroll
    for (int kk = 0; kk < kAHeadDim / 16; ++kk) {
        const float* qr = q + (size_t(t) * kAHeads + g * 16 + l16) * kAHeadDim + kk * 16 + g8 * 8;
        const float4 a = reinterpret_cast<const float4*>(qr)[0], b = reinterpret_cast<const float4*>(qr)[1];
        const uint32_t w0 = f2bf(a.x) | (uint32_t(f2bf(a.y)) << 16), w1 = f2bf(a.z) | (uint32_t(f2bf(a.w)) << 16);
        const uint32_t w2 = f2bf(b.x) | (uint32_t(f2bf(b.y)) << 16), w3 = f2bf(b.z) | (uint32_t(f2bf(b.w)) << 16);
        qa[kk] = __builtin_bit_cast(att_v8s, make_uint4(w0, w1, w2, w3));
    }
    const float scale = rsqrtf(float(kAHeadDim));
    att_v8f o[kAHeadDim / 16];
#pragma unroll
    for (int d = 0; d < kAHeadDim / 16; ++d) o[d] = att_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
    for (int j0 = sidx * kFwKeys; j0 <= p; j0 += nsp * kFwKeys) {
        // the run's operands: K rows j0 + nt * 16 + l16 (dims kk * 16 + g8 * 8 ..), V^T dims d * 16 + l16 (keys kk * 16 + g8 * 8 ..)
        att_v8s kb[2][kAHeadDim / 16], vb[kFwKeys / 16][kAHeadDim / 16];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int kk = 0; kk < kAHeadDim / 16; ++kk)
                kb[nt][kk] = *reinterpret_cast<const att_v8s*>(Kh + size_t(j0 + nt * 16 + l16) * kAHeadDim + kk * 16 + g8 * 8);
        const uint16_t* Vr = Vh + size_t(j0 / 32) * kAHeadDim * 32;
#pragma unroll
        for (int kk = 0; kk < kFwKeys / 16; ++kk)
#pragma unroll
            for (int d = 0; d < kAHeadDim / 16; ++d)
                vb[kk][d] = *reinterpret_cast<const att_v8s*>(Vr + (d * 16 + l16) * 32 + kk * 16 + g8 * 8);
        att_v8f s[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            s[nt] = att_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < kAHeadDim / 16; ++kk) s[nt] = att_wmma(qa[kk], kb[nt][kk], s[nt]);
        }
        float alpha[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v0 = s[0][i] * scale, v1 = s[1][i] * scale;
            if (j0 + l16 > p) v0 = -INFINITY;
            if (j0 + 16 + l16 > p) v1 = -INFINITY;
            float mx = fmaxf(v0, v1);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 32));
            const float mn = fmaxf(m[i], mx);
            alpha[i] = expf(m[i] - mn);
            m[i] = mn;
            const float e0 = expf(v0 - mn), e1 = expf(v1 - mn);
            l[i] = l[i] * alpha[i] + e0 + e1;
            sp[warp][g8 * 8 + i][l16] = f2bf(e0);
            sp[warp][g8 * 8 + i][16 + l16] = f2bf(e1);
        }
#pragma unroll
        for (int d = 0; d < kAHeadDim / 16; ++d)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[d][i] *= alpha[i];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
#pragma unroll
        for (int kk = 0; kk < kFwKeys / 16; ++kk) {
            const att_v8s pa = *reinterpret_cast<const att_v8s*>(&sp[warp][l16][kk * 16 + g8 * 8]);
#pragma unroll
            for (int d = 0; d < kAHeadDim / 16; ++d) o[d] = att_wmma(pa, vb[kk][d], o[d]);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");   // (sp is rewritten next run)
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
    }
    // this lane: heads g * 16 + g8 * 8 + i, dims d * 16 + l16; a split without keys leaves sum 0 (weight 0)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float sum = l[i];
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off, 32);
        float* pp = part + ((size_t(t) * kAHeads + g * 16 + g8 * 8 + i) * nsp + sidx) * (kAHeadDim + 2);
#pragma unroll
        for (int d = 0; d < kAHeadDim / 16; ++d) pp[d * 16 + l16] = o[d][i];
        if (l16 == 0) { pp[kAHeadDim] = m[i]; pp[kAHeadDim + 1] = sum; }
    }
}

// causal softmax of S (16 T x Lp, row (hl, t) sees keys 0 .. pos0 + t) into BF16 P
__global__ void k_softmax_causal(const float* S, uint16_t* P, const int* pos0, int T, int Lp, int t_off) {
    __shared__ float sh[256];
    const int row = blockIdx.x, t = t_off + row % T, L = *pos0 + t + 1;
    const float* s = S + size_t(row) * Lp;
    uint16_t* p = P + size_t(row) * Lp;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += 256) m = fmaxf(m, s[j]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0];
    __syncthreads();
    float a = 0.f;
    for (int j = threadIdx.x; j < L; j += 256) a += expf(s[j] - m);
    const float inv = 1.f / block_sum<256>(a, sh);
    for (int j = threadIdx.x; j < Lp; j += 256) p[j] = j < L ? f2bf(expf(s[j] - m) * inv) : 0;
}
// O_g (16 T x 128, rows (hl, t)) back into the (T, kAHeads, kAHeadDim) output
__global__ void k_o_scatter(const float* O, float* out, int g, int T, int t_off) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(16) * T * kAHeadDim; i += size_t(gridDim.x) * blockDim.x) {
        const int d = int(i % kAHeadDim), row = int(i / kAHeadDim), hl = row / T, t = t_off + row % T;
        out[(size_t(t) * kAHeads + g * 16 + hl) * kAHeadDim + d] = O[i];
    }
}

// ---- Mamba-2 over a prompt chunk, block by block (the SSD form) ------------------------------------------------
// The same recurrence as k_mamba_scan, rearranged: positions go Q at a time; with A_t the running sum of dt A inside
// the block,
//   y_t = sum_{s<=t} (C_t.B_s) e^(A_t - A_s) dt_s x_s + e^(A_t) C_t . H + D x_t
//   H'  = e^(A_Q) H + sum_s (e^(A_Q - A_s) dt_s B_s) x_s^T
// The decay factors are applied to B (and the per-position scalars) before the products, as llama.cpp's ssm-scan.cu
// does, so the inner loops are plain multiply-adds; the first version took an exponential per (state, row, position).
// Block = (head, 16 of its 64 rows); 256 threads; H (128 states x 16 rows) lives in LDS.
constexpr int kSsdQ = 32;
__global__ void __launch_bounds__(256) k_mamba_ssd(const float* xbc, const float* proj, const uint16_t* A_log,
                                                   const uint16_t* D, const uint16_t* dt_bias, float* ssm, float* y, int T) {
    constexpr int RP = 16, Q = kSsdQ;
    __shared__ float Bs[Q][kMState + 1], Cs[Q][kMState + 1];
    __shared__ float xs[Q][RP], dts[Q], acs[Q], wq[Q], M[Q][Q + 1];
    __shared__ float H[kMState][RP];
    const int hd = blockIdx.x, p0 = blockIdx.y * RP, tid = threadIdx.x;
    const int g = hd / (kMHeads / kMGroups);
    const float A = -expf(bf16f(A_log[hd])), Dh = bf16f(D[hd]), db = bf16f(dt_bias[hd]);
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState;            // layout [p][n]
    for (int i = tid; i < kMState * RP; i += 256) { const int n = i % kMState, p = i / kMState; H[n][p] = hp[size_t(p0 + p) * kMState + n]; }
    for (int c0 = 0; c0 < T; c0 += Q) {
        const int nq = min(Q, T - c0);
        __syncthreads();
        for (int i = tid; i < Q * kMState; i += 256) {
            const int t = i / kMState, n = i % kMState;
            const float* xt = xbc + size_t(c0 + t) * kConvDim;
            Bs[t][n] = t < nq ? xt[kMInter + g * kMState + n] : 0.f;
            Cs[t][n] = t < nq ? xt[kMInter + kMGroups * kMState + g * kMState + n] : 0.f;
        }
        for (int i = tid; i < Q * RP; i += 256) {
            const int t = i / RP, p = i % RP;
            xs[t][p] = t < nq ? xbc[size_t(c0 + t) * kConvDim + hd * kMHeadDim + p0 + p] : 0.f;
        }
        if (tid < Q) {
            float dt = 0.f;
            if (tid < nq) { dt = proj[size_t(c0 + tid) * kInProj + kMInter + kConvDim + hd] + db; dt = dt > 20.f ? dt : log1pf(expf(dt)); }
            dts[tid] = dt;
        }
        __syncthreads();
        if (tid == 0) { float a = 0.f; for (int t = 0; t < Q; ++t) { a += dts[t] * A; acs[t] = a; } }
        __syncthreads();
        const float aq = acs[nq - 1];
        if (tid < Q) wq[tid] = tid < nq ? expf(aq - acs[tid]) * dts[tid] : 0.f;   // the weight of position s in H'
        // M[t][s] = (C_t . B_s) e^(acs_t - acs_s) dt_s for s <= t (B not yet scaled)
        for (int i = tid; i < Q * Q; i += 256) {
            const int t = i / Q, s = i % Q;
            float v = 0.f;
            if (s <= t && t < nq) {
#pragma unroll 8
                for (int n = 0; n < kMState; ++n) v += Cs[t][n] * Bs[s][n];
                v *= expf(acs[t] - acs[s]) * dts[s];
            }
            M[t][s] = v;
        }
        __syncthreads();
        // outputs: y = M x + e^(acs_t) C_t . H + D x
        for (int i = tid; i < Q * RP; i += 256) {
            const int t = i / RP, p = i % RP;
            if (t >= nq) continue;
            float a = 0.f;
            for (int s = 0; s <= t; ++s) a += M[t][s] * xs[s][p];
            float o = 0.f;
#pragma unroll 8
            for (int n = 0; n < kMState; ++n) o += Cs[t][n] * H[n][p];
            y[size_t(c0 + t) * kMInter + hd * kMHeadDim + p0 + p] = a + expf(acs[t]) * o + Dh * xs[t][p];
        }
        __syncthreads();
        // H = e^(acs_last) H + sum_s wq_s B_s x_s
        const float ea = expf(aq);
        for (int i = tid; i < kMState * RP; i += 256) {
            const int n = i % kMState, p = i / kMState;
            float a = H[n][p] * ea;
            for (int s = 0; s < nq; ++s) a += wq[s] * Bs[s][n] * xs[s][p];
            H[n][p] = a;
        }
    }
    __syncthreads();
    for (int i = tid; i < kMState * RP; i += 256) { const int n = i % kMState, p = i / kMState; hp[size_t(p0 + p) * kMState + n] = H[n][p]; }
}

// BF16 times T (<= kTMax) columns, no LDS: one warp per row, each lane 8 weights (16 bytes) a step, the inputs read
// through the cache (they are T x n floats, tiny next to the matrix)
__global__ void __launch_bounds__(256) k_bf16_rows(const uint16_t* W, const float* x, float* y, int n, int rows, int T) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint4* wr = reinterpret_cast<const uint4*>(W + size_t(row) * n);
    float acc[kTMax];
#pragma unroll
    for (int c = 0; c < kTMax; ++c) acc[c] = 0.f;
    for (int i = lane; i < n / 8; i += 32) {
        const uint4 q = wr[i];
        const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
        float w[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) { w[2 * j] = __uint_as_float(ws[j] << 16); w[2 * j + 1] = __uint_as_float(ws[j] & 0xffff0000u); }
#pragma unroll
        for (int c = 0; c < kTMax; ++c) {
            if (c >= T) break;
            const float4* xp = reinterpret_cast<const float4*>(x + size_t(c) * n) + i * 2;
            const float4 xa = xp[0], xb = xp[1];
            acc[c] += w[0] * xa.x + w[1] * xa.y + w[2] * xa.z + w[3] * xa.w + w[4] * xb.x + w[5] * xb.y + w[6] * xb.z + w[7] * xb.w;
        }
    }
#pragma unroll
    for (int c = 0; c < kTMax; ++c) {
        if (c >= T) break;
        float a = acc[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o, 32);
        if (lane == 0) y[size_t(c) * rows + row] = a;
    }
}

// ---- skinny products: W (rows x n) times T <= 8 input rows -----------------------------------------------------
// Warp w takes the 256-wide K slices w, w + 8, ...; for its slice it holds the T inputs' 8 values per lane in
// registers and streams `rb` weight rows past them, so the inputs are read once per block, not once per row. Each
// (row, slice) dot for the 8 columns is summed across the warp with a transposing butterfly (9 shuffles for 8
// columns), added into the warp's LDS partial, and the 8 warps' partials are summed in a fixed order at the end.
// FMT 0: BF16 rows; FMT 1: NVFP4 rows (16-value E4M3 group scales, the tensor scale applied at the end).
constexpr int kSkRB = 128;           // most rows per block
// E2M1 nibble to float from its bits (sign, 2-bit exponent, 1-bit mantissa): no table, no divergent lookups
__device__ __forceinline__ float e2m1f(uint32_t n) {
    const uint32_t s = (n & 8) << 28, e = (n >> 1) & 3, m = n & 1;
    const uint32_t mag = e ? (((e + 126u) << 23) | (m << 22)) : (m ? (126u << 23) : 0u);
    return __uint_as_float(s | mag);
}
template <int FMT>
__global__ void __launch_bounds__(256) k_skinny(const void* W, const uint8_t* S, float scale2, const float* x, float* y,
                                                int n, int rows, int T, int rb) {
    __shared__ float part[8][kSkRB][8];
    __shared__ float e4[256];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int r0 = blockIdx.x * rb, nr = min(rb, rows - r0);
    if (FMT == 1) e4[tid] = e4m3f(uint8_t(tid));
    for (int i = tid; i < 8 * kSkRB * 8; i += 256) (&part[0][0][0])[i] = 0.f;
    __syncthreads();
    const int nslices = n / 256;
    const int col = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    for (int s = warp; s < nslices; s += 8) {
        float xr[8][8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < T) {
                const float4* xp = reinterpret_cast<const float4*>(x + size_t(c) * n + s * 256 + lane * 8);
                const float4 a = xp[0], b = xp[1];
                xr[c][0] = a.x; xr[c][1] = a.y; xr[c][2] = a.z; xr[c][3] = a.w; xr[c][4] = b.x; xr[c][5] = b.y; xr[c][6] = b.z; xr[c][7] = b.w;
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) xr[c][j] = 0.f;
            }
        }
        constexpr int U = 4;                                         // rows loaded together: 4 requests in flight
        for (int rr = 0; rr < nr; rr += U) {
            uint4 qb[U]; uint32_t qn[U]; float scv[U];
#pragma unroll
            for (int i = 0; i < U; ++i) {
                const int row = r0 + min(rr + i, nr - 1);
                if (FMT == 0) qb[i] = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(W) + size_t(row) * n + s * 256)[lane];
                else {
                    qn[i] = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(W) + size_t(row) * (n / 2) + s * 128)[lane];
                    scv[i] = e4[S[size_t(row) * (n / 16) + s * 16 + lane / 2]];
                }
            }
#pragma unroll
            for (int i = 0; i < U; ++i) {
                const int r = rr + i;
                if (r >= nr) break;
                float w[8];
                if (FMT == 0) {
                    const uint32_t ws[4] = {qb[i].x, qb[i].y, qb[i].z, qb[i].w};
#pragma unroll
                    for (int j = 0; j < 4; ++j) { w[2 * j] = __uint_as_float(ws[j] << 16); w[2 * j + 1] = __uint_as_float(ws[j] & 0xffff0000u); }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) w[j] = e2m1f((qn[i] >> (4 * j)) & 15) * scv[i];
                }
                float v[8];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float a = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) a += w[j] * xr[c][j];
                    v[c] = a;
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const bool hi = lane & 16;
                    const float send = hi ? v[k] : v[k + 4];
                    const float got = __shfl_xor_sync(0xffffffff, send, 16, 32);
                    v[k] = (hi ? v[k + 4] : v[k]) + got;
                }
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const bool hi = lane & 8;
                    const float send = hi ? v[k] : v[k + 2];
                    const float got = __shfl_xor_sync(0xffffffff, send, 8, 32);
                    v[k] = (hi ? v[k + 2] : v[k]) + got;
                }
                {
                    const bool hi = lane & 4;
                    const float send = hi ? v[0] : v[1];
                    const float got = __shfl_xor_sync(0xffffffff, send, 4, 32);
                    v[0] = (hi ? v[1] : v[0]) + got;
                }
                v[0] += __shfl_xor_sync(0xffffffff, v[0], 2, 32);
                v[0] += __shfl_xor_sync(0xffffffff, v[0], 1, 32);
                if ((lane & 3) == 0) part[warp][r][col] += v[0];
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < nr * T; i += 256) {
        const int r = i / T, c = i % T;
        float a = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) a += part[w][r][c];
        y[size_t(c) * rows + r0 + r] = FMT == 1 ? a * scale2 : a;
    }
}

// ---- Mamba-2 over a prompt chunk, SSD with the blocks in parallel ----------------------------------------------
// k_mamba_ssd kept one block per (head, row slice) walking all its Q-blocks in order (13 ms per 2,048 positions per
// layer). Here only the hand-off of H between Q-blocks is sequential:
//   1. k_ssd_cb:    CB[g][c][t][s] = C_t . B_s per group and Q-block (shared by the group's 16 heads)
//   2. k_ssd_intra: per (head, Q-block, row slice): the block's own outputs y_t = sum_{s<=t} CB e^(A_t-A_s) dt_s x_s
//                   + D x_t (written), and its state contribution S_c = sum_s e^(A_Q-A_s) dt_s B_s x_s^T
//   3. k_ssd_pass:  per (head, row slice): H_in[c] = H; H = e^(A_Q,c) H + S_c  (S_c's slot then holds H_in[c])
//   4. k_ssd_out:   y_t += e^(A_t) C_t . H_in[c]
// A_t is the running sum of dt A inside each Q-block (dt = softplus(dt + bias)), stored per (head, position).
constexpr int kSq = 32;              // positions per Q-block
__global__ void k_ssd_dt(const float* proj, const uint16_t* A_log, const uint16_t* dt_bias, float* dtv, float* acs, float* ea, int T,
                         const float* prs = nullptr) {
    // per (head, Q-block): dt and the running decay sum; ea = e^(A_Q) for the block
    const int hd = blockIdx.x, c = blockIdx.y, t = threadIdx.x;      // kSq threads
    __shared__ float a[kSq];
    const int pos = c * kSq + t;
    const float A = -expf(bf16f(A_log[hd])), db = bf16f(dt_bias[hd]);
    float dt = 0.f;
    if (pos < T) { dt = proj[size_t(pos) * kInProj + kMInter + kConvDim + hd] * (prs ? prs[pos] : 1.f) + db; dt = dt > 20.f ? dt : log1pf(expf(dt)); }
    a[t] = dt * A;
    __syncthreads();
    if (t == 0) { float s = 0.f; for (int i = 0; i < kSq; ++i) { s += a[i]; a[i] = s; } }
    __syncthreads();
    if (pos < T) { dtv[size_t(pos) * kMHeads + hd] = dt; acs[size_t(pos) * kMHeads + hd] = a[t]; }
    if (t == 0) { const int last = min(kSq, T - c * kSq) - 1; ea[size_t(c) * kMHeads + hd] = expf(a[last]); }
}
__global__ void k_ssd_cb(const float* xbc, float* CB, int T) {
    __shared__ float Bs[kSq][kMState + 1], Cs[kSq][kMState + 1];
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;      // 256 threads
    for (int i = tid; i < kSq * kMState; i += 256) {
        const int t = i / kMState, n = i % kMState, pos = c * kSq + t;
        const float* xt = xbc + size_t(pos) * kConvDim;
        Bs[t][n] = pos < T ? xt[kMInter + g * kMState + n] : 0.f;
        Cs[t][n] = pos < T ? xt[kMInter + kMGroups * kMState + g * kMState + n] : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < kSq * kSq; i += 256) {
        const int t = i / kSq, s = i % kSq;
        float v = 0.f;
        if (s <= t) {
#pragma unroll 8
            for (int n = 0; n < kMState; ++n) v += Cs[t][n] * Bs[s][n];
        }
        CB[((size_t(g) * gridDim.y + c) * kSq + t) * kSq + s] = v;
    }
}
// row slice of 16; S layout [head][Q-block][n][p (64)]
__global__ void __launch_bounds__(256) k_ssd_intra(const float* xbc, const float* CB, const float* dtv, const float* acs,
                                                   const uint16_t* D, float* S, float* y, int T, int nc) {
    constexpr int RP = 16;
    __shared__ float Bs[kSq][kMState + 1], xs[kSq][RP], M[kSq][kSq + 1], w[kSq], at[kSq];
    const int hd = blockIdx.x, c = blockIdx.y, p0 = blockIdx.z * RP, tid = threadIdx.x;
    const int g = hd / (kMHeads / kMGroups), c0 = c * kSq, nq = min(kSq, T - c0);
    for (int i = tid; i < kSq * kMState; i += 256) {
        const int t = i / kMState, n = i % kMState;
        Bs[t][n] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + g * kMState + n] : 0.f;
    }
    for (int i = tid; i < kSq * RP; i += 256) {
        const int t = i / RP, p = i % RP;
        xs[t][p] = t < nq ? xbc[size_t(c0 + t) * kConvDim + hd * kMHeadDim + p0 + p] : 0.f;
    }
    if (tid < kSq) at[tid] = tid < nq ? acs[size_t(c0 + tid) * kMHeads + hd] : 0.f;
    __syncthreads();
    const float aq = at[nq - 1];
    if (tid < kSq) w[tid] = tid < nq ? expf(aq - at[tid]) * dtv[size_t(c0 + tid) * kMHeads + hd] : 0.f;
    for (int i = tid; i < kSq * kSq; i += 256) {
        const int t = i / kSq, s = i % kSq;
        M[t][s] = (s <= t && t < nq) ? CB[((size_t(g) * nc + c) * kSq + t) * kSq + s] * expf(at[t] - at[s]) * dtv[size_t(c0 + s) * kMHeads + hd] : 0.f;
    }
    __syncthreads();
    const float Dh = bf16f(D[hd]);
    for (int i = tid; i < kSq * RP; i += 256) {
        const int t = i / RP, p = i % RP;
        if (t >= nq) continue;
        float a = Dh * xs[t][p];
        for (int s = 0; s <= t; ++s) a += M[t][s] * xs[s][p];
        y[size_t(c0 + t) * kMInter + hd * kMHeadDim + p0 + p] = a;
    }
    float* Sc = S + (size_t(hd) * nc + c) * kMState * kMHeadDim;
    for (int i = tid; i < kMState * RP; i += 256) {
        const int n = i / RP, p = i % RP;
        float a = 0.f;
        for (int s = 0; s < nq; ++s) a += w[s] * Bs[s][n] * xs[s][p];
        Sc[size_t(n) * kMHeadDim + p0 + p] = a;
    }
}
// the hand-off: one thread per (state, row) of a head, walking the Q-blocks; leaves H_in[c] in S's slot c
__global__ void k_ssd_pass(float* S, const float* ea, float* ssm, int nc) {
    const int hd = blockIdx.x, i = blockIdx.y * blockDim.x + threadIdx.x;    // i over 128 x 64
    if (i >= kMState * kMHeadDim) return;
    const int n = i / kMHeadDim, p = i % kMHeadDim;
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState + size_t(p) * kMState + n;   // ssm layout [p][n]
    float H = *hp;
    for (int c = 0; c < nc; ++c) {
        float* sl = S + (size_t(hd) * nc + c) * kMState * kMHeadDim + i;
        const float sc = *sl;
        *sl = H;
        H = ea[size_t(c) * kMHeads + hd] * H + sc;
    }
    *hp = H;
}
__global__ void __launch_bounds__(256) k_ssd_out(const float* xbc, const float* acs, const float* S, float* y, int T, int nc) {
    constexpr int RP = 16;
    __shared__ float Cs[kSq][kMState + 1], Hs[kMState][RP];
    const int hd = blockIdx.x, c = blockIdx.y, p0 = blockIdx.z * RP, tid = threadIdx.x;
    const int g = hd / (kMHeads / kMGroups), c0 = c * kSq, nq = min(kSq, T - c0);
    for (int i = tid; i < kSq * kMState; i += 256) {
        const int t = i / kMState, n = i % kMState;
        Cs[t][n] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + kMGroups * kMState + g * kMState + n] : 0.f;
    }
    const float* Hc = S + (size_t(hd) * nc + c) * kMState * kMHeadDim;
    for (int i = tid; i < kMState * RP; i += 256) { const int n = i / RP, p = i % RP; Hs[n][p] = Hc[size_t(n) * kMHeadDim + p0 + p]; }
    __syncthreads();
    for (int i = tid; i < kSq * RP; i += 256) {
        const int t = i / RP, p = i % RP;
        if (t >= nq) continue;
        float o = 0.f;
#pragma unroll 8
        for (int n = 0; n < kMState; ++n) o += Cs[t][n] * Hs[n][p];
        y[size_t(c0 + t) * kMInter + hd * kMHeadDim + p0 + p] += expf(acs[size_t(c0 + t) * kMHeads + hd]) * o;
    }
}

// group-level versions: block = (group, Q-block); B (or C) is loaded once and the group's 16 heads go in turn
__global__ void __launch_bounds__(256) k_ssd_intra2(const float* xbc, const float* CB, const float* dtv, const float* acs,
                                                    const uint16_t* D, float* S, float* y, int T, int nc, bool write_S = true) {
    __shared__ float Bs[kSq][kMState + 1], xs[kSq][kMHeadDim + 1], M[kSq][kSq + 1], w[kSq], at[kSq], cbt[kSq][kSq + 1];
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, c0 = c * kSq, nq = min(kSq, T - c0);
    for (int i = tid; i < kSq * kMState; i += 256) {
        const int t = i / kMState, n = i % kMState;
        Bs[t][n] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + g * kMState + n] : 0.f;
    }
    for (int i = tid; i < kSq * kSq; i += 256) cbt[i / kSq][i % kSq] = CB[((size_t(g) * nc + c) * kSq) * kSq + i];
    for (int hl = 0; hl < kMHeads / kMGroups; ++hl) {
        const int hd = g * (kMHeads / kMGroups) + hl;
        __syncthreads();
        for (int i = tid; i < kSq * kMHeadDim; i += 256) {
            const int t = i / kMHeadDim, p = i % kMHeadDim;
            xs[t][p] = t < nq ? xbc[size_t(c0 + t) * kConvDim + hd * kMHeadDim + p] : 0.f;
        }
        if (tid < kSq) at[tid] = tid < nq ? acs[size_t(c0 + tid) * kMHeads + hd] : 0.f;
        __syncthreads();
        const float aq = at[nq - 1];
        if (tid < kSq) w[tid] = tid < nq ? expf(aq - at[tid]) * dtv[size_t(c0 + tid) * kMHeads + hd] : 0.f;
        for (int i = tid; i < kSq * kSq; i += 256) {
            const int t = i / kSq, s = i % kSq;
            M[t][s] = (s <= t && t < nq) ? cbt[t][s] * expf(at[t] - at[s]) * dtv[size_t(c0 + s) * kMHeads + hd] : 0.f;
        }
        __syncthreads();
        const float Dh = bf16f(D[hd]);
        {   // Y (32 x 64): thread = 2 positions x 4 rows
            const int t0 = (tid / 16) * 2, pp = (tid % 16) * 4;
            float a[2][4] = {};
            for (int s = 0; s <= t0 + 1 && s < kSq; ++s) {
                const float m0 = M[t0][s], m1 = M[t0 + 1][s];
                const float x0 = xs[s][pp], x1 = xs[s][pp + 1], x2 = xs[s][pp + 2], x3 = xs[s][pp + 3];
                a[0][0] += m0 * x0; a[0][1] += m0 * x1; a[0][2] += m0 * x2; a[0][3] += m0 * x3;
                a[1][0] += m1 * x0; a[1][1] += m1 * x1; a[1][2] += m1 * x2; a[1][3] += m1 * x3;
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int t = t0 + i;
                if (t >= nq) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) y[size_t(c0 + t) * kMInter + hd * kMHeadDim + pp + j] = a[i][j] + Dh * xs[t][pp + j];
            }
        }
        if (!write_S) continue;                                       // (k_ssd_state does it)
        float* Sc = S + (size_t(hd) * nc + c) * kMState * kMHeadDim;
        {   // S (128 x 64) = sum_s w_s B_s x_s: thread = 4 states x 8 rows
            const int n0 = (tid / 8) * 4, pp = (tid % 8) * 8;
            float a[4][8] = {};
            for (int s = 0; s < kSq; ++s) {
                const float ws = w[s];
                float bv[4], xv[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) bv[i] = ws * Bs[s][n0 + i];
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = xs[s][pp + j];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) a[i][j] += bv[i] * xv[j];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) Sc[size_t(n0 + i) * kMHeadDim + pp + j] = a[i][j];
        }
    }
}
__global__ void __launch_bounds__(256) k_ssd_out2(const float* xbc, const float* acs, const float* S, float* y, int T, int nc) {
    __shared__ float Cs[kSq][kMState + 1], Hs[kMState][kMHeadDim + 1];
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, c0 = c * kSq, nq = min(kSq, T - c0);
    for (int i = tid; i < kSq * kMState; i += 256) {
        const int t = i / kMState, n = i % kMState;
        Cs[t][n] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + kMGroups * kMState + g * kMState + n] : 0.f;
    }
    for (int hl = 0; hl < kMHeads / kMGroups; ++hl) {
        const int hd = g * (kMHeads / kMGroups) + hl;
        __syncthreads();
        const float* Hc = S + (size_t(hd) * nc + c) * kMState * kMHeadDim;
        for (int i = tid; i < kMState * kMHeadDim; i += 256) Hs[i / kMHeadDim][i % kMHeadDim] = Hc[i];
        __syncthreads();
        {   // (32 x 64) = C (32 x 128) . H (128 x 64): thread = 2 positions x 4 rows
            const int t0 = (tid / 16) * 2, pp = (tid % 16) * 4;
            float a[2][4] = {};
            for (int n = 0; n < kMState; ++n) {
                const float c0v = Cs[t0][n], c1v = Cs[t0 + 1][n];
                const float h0 = Hs[n][pp], h1 = Hs[n][pp + 1], h2 = Hs[n][pp + 2], h3 = Hs[n][pp + 3];
                a[0][0] += c0v * h0; a[0][1] += c0v * h1; a[0][2] += c0v * h2; a[0][3] += c0v * h3;
                a[1][0] += c1v * h0; a[1][1] += c1v * h1; a[1][2] += c1v * h2; a[1][3] += c1v * h3;
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int t = t0 + i;
                if (t >= nq) continue;
                const float e = expf(acs[size_t(c0 + t) * kMHeads + hd]);
#pragma unroll
                for (int j = 0; j < 4; ++j) y[size_t(c0 + t) * kMInter + hd * kMHeadDim + pp + j] += e * a[i][j];
            }
        }
    }
}

// k_ssd_intra2's state part, k_ssd_pass and k_ssd_out2 in one: per (head, kSsdP rows of the head) a block walks the
// Q-blocks in order with its slice of the state H (128 x kSsdP) in registers: each Q-block's outputs
// y_t += e^(A_t) C_t . H, then H = e^(A_Q) H + S_c, S_c = sum_s w_s B_s x_s. The same sums in the same order as those
// three kernels, so the same bits, without writing and re-reading every Q-block's state (0.5 GB a layer per 4,096
// positions, ~7 ms).
// P rows of a head per block, NT threads: NT / 128 threads per state column, each P * 128 / NT rows
template <int P, int NT>
__global__ void __launch_bounds__(NT) k_ssd_state(const float* xbc, const float* dtv, const float* acs, const float* ea,
                                                   float* ssm, float* y, int T, int nc) {
    constexpr int TPN = NT / kMState, PT = P / TPN;                   // threads per state column, rows per thread
    __shared__ float Cs[kSq][kMState + 1], Hs[kMState][P + 1], xs[kSq][P], w[kSq], et[kSq];
    const int hd = blockIdx.x, p0 = blockIdx.y * P, tid = threadIdx.x;
    const int g = hd / (kMHeads / kMGroups);
    const int n = tid / TPN, pp = (tid % TPN) * PT;                   // thread: state column n, rows pp .. pp + PT - 1
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState;               // ssm layout [p][n]
    float H[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) H[j] = hp[size_t(p0 + pp + j) * kMState + n];
    // a Q-block's global reads go to registers one step ahead (issued before the step's products, used after the next
    // barrier), so each step's memory waits overlap the step before (waiting for them in the step cost ~1/3 of the
    // walk: 43 us a step)
    constexpr int CI = kSq * kMState / NT, XI = kSq * P / NT;
    float pc[CI], px[XI], pb[kSq], pat = 0.f, pdt = 0.f, paq = 0.f;
    auto fetch = [&](int c) {
        if (c >= nc) return;
        const int c0 = c * kSq, nq = min(kSq, T - c0);
#pragma unroll
        for (int j = 0; j < CI; ++j) {
            const int i = tid + j * NT, t = i / kMState, m = i % kMState;
            pc[j] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + kMGroups * kMState + g * kMState + m] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < XI; ++j) {
            const int i = tid + j * NT, t = i / P, q = i % P;
            px[j] = t < nq ? xbc[size_t(c0 + t) * kConvDim + hd * kMHeadDim + p0 + q] : 0.f;
        }
#pragma unroll
        for (int t = 0; t < kSq; ++t) pb[t] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + g * kMState + n] : 0.f;
        if (tid < kSq) {
            pat = tid < nq ? acs[size_t(c0 + tid) * kMHeads + hd] : 0.f;
            pdt = tid < nq ? dtv[size_t(c0 + tid) * kMHeads + hd] : 0.f;
            paq = acs[size_t(c0 + nq - 1) * kMHeads + hd];
        }
    };
    fetch(0);
    for (int c = 0; c < nc; ++c) {
        const int c0 = c * kSq, nq = min(kSq, T - c0);
        __syncthreads();                                              // the last Q-block's LDS reads are done
#pragma unroll
        for (int j = 0; j < PT; ++j) Hs[n][pp + j] = H[j];
#pragma unroll
        for (int j = 0; j < CI; ++j) { const int i = tid + j * NT; Cs[i / kMState][i % kMState] = pc[j]; }
#pragma unroll
        for (int j = 0; j < XI; ++j) { const int i = tid + j * NT; xs[i / P][i % P] = px[j]; }
        if (tid < kSq) {
            w[tid] = tid < nq ? expf(paq - pat) * pdt : 0.f;
            et[tid] = expf(pat);
        }
        float bn[kSq];                                                // B_s[n] for this thread's column
#pragma unroll
        for (int t = 0; t < kSq; ++t) bn[t] = pb[t];
        __syncthreads();
        fetch(c + 1);
        static_assert(kSq * P == NT * 4, "outputs: 4 per thread");
        {   // outputs (kSq x P) = C (kSq x 128) . H (128 x P): thread = 1 position x 4 rows
            const int t = tid / (P / 4), q0 = (tid % (P / 4)) * 4;
            float a[4] = {};
            for (int m = 0; m < kMState; ++m) {
                const float cv = Cs[t][m];
#pragma unroll
                for (int j = 0; j < 4; ++j) a[j] += cv * Hs[m][q0 + j];
            }
            if (t < nq)
#pragma unroll
                for (int j = 0; j < 4; ++j) y[size_t(c0 + t) * kMInter + hd * kMHeadDim + p0 + q0 + j] += et[t] * a[j];
        }
        float sc[PT] = {};
#pragma unroll
        for (int t = 0; t < kSq; ++t) {
            const float bv = w[t] * bn[t];
#pragma unroll
            for (int j = 0; j < PT; ++j) sc[j] += bv * xs[t][pp + j];
        }
        const float e = ea[size_t(c) * kMHeads + hd];
#pragma unroll
        for (int j = 0; j < PT; ++j) H[j] = e * H[j] + sc[j];
    }
#pragma unroll
    for (int j = 0; j < PT; ++j) hp[size_t(p0 + pp + j) * kMState + n] = H[j];
}

// BF16 matrix-core helpers for k_ssd_full: fp32 values go in as a BF16 high part plus a BF16 remainder (three
// products: hi.hi + hi.lo + lo.hi, ~2^-15 relative each), the sums are fp32.
typedef __bf16 sw_v8bf __attribute__((ext_vector_type(8)));
typedef float sw_v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ sw_v8f sw_wmma(const uint4& a, const uint4& b, const sw_v8f& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(sw_v8bf, a), __builtin_bit_cast(sw_v8bf, b), c);
}
// two floats to packed BF16 high parts (truncated) and packed BF16 remainders (truncated)
__device__ __forceinline__ void sw_split2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    const uint32_t b0 = __float_as_uint(x0), b1 = __float_as_uint(x1);
    hi = __builtin_amdgcn_perm(b1, b0, 0x07060302u);
    const float r0 = x0 - __uint_as_float(b0 & 0xffff0000u), r1 = x1 - __uint_as_float(b1 & 0xffff0000u);
    lo = __builtin_amdgcn_perm(__float_as_uint(r1), __float_as_uint(r0), 0x07060302u);
}
__device__ __forceinline__ void sw_split8(const float* v, uint4& hi, uint4& lo) {
    sw_split2(v[0], v[1], hi.x, lo.x); sw_split2(v[2], v[3], hi.y, lo.y);
    sw_split2(v[4], v[5], hi.z, lo.z); sw_split2(v[6], v[7], hi.w, lo.w);
}

// The whole SSD output in one pass on the BF16 matrix cores: the state walk plus each Q-block's own outputs (what
// k_ssd_intra2 + k_ssd_state do in fp32), so y is written once and x read once. Each warp keeps one (head, 16-row P
// slice) of the state as 8 accumulator tiles (16 states x 16 rows) for the whole walk; a tile's lane layout (states
// g*8 + i, row l16) is the B operand layout of the outputs' product, so the state never leaves the registers. HPB
// heads a block (one group: C and B shared through LDS), 4 warps a head. Per Q-block and head: M[t][s] = CB[t][s] e^(a_t - a_s) (s <= t, CB =
// C.B^T from k_ssd_cb) into LDS as BF16 high part + remainder; y[t][p] = D x[t][p] + sum_s M[t][s] (dt_s x[s][p])
// + et[t] sum_n C[t][n] H[n][p]; then H = e H + B^T (w x). Three syncs a Q-block: LDS free, C/B/step values in, M in.
template <int HPB>
__global__ void __launch_bounds__(HPB * 128) k_ssd_full(const float* xbc, const float* CB, const float* dtv, const float* acs,
                                                       const float* ea, const uint16_t* D, float* ssm, float* y, int T, int nc) {
    constexpr int NT = HPB * 128, LDC = kMState + 8, LDB = kSq + 8;
    __shared__ __align__(16) uint16_t Ch[kSq][LDC], Cl[kSq][LDC], Bh[kMState][LDB], Bl[kMState][LDB];
    __shared__ __align__(16) uint16_t Mh[HPB][kSq][LDB], Ml[HPB][kSq][LDB];
    __shared__ float w[HPB][kSq], et[HPB][kSq], at[HPB][kSq], dts[HPB][kSq];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int hl = warp / 4, hd = blockIdx.x * HPB + hl, grp = hd / (kMHeads / kMGroups);
    const int p = (warp % 4) * 16 + l16;                              // this lane's P row
    const float Dh = bf16f(D[hd]);
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState;               // ssm layout [p][n]
    sw_v8f H[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const float4 a = *reinterpret_cast<const float4*>(hp + size_t(p) * kMState + nt * 16 + g * 8);
        const float4 b = *reinterpret_cast<const float4*>(hp + size_t(p) * kMState + nt * 16 + g * 8 + 4);
        H[nt] = sw_v8f{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    }
    // reads one Q-block ahead. C: thread = 2 states, CT positions; B: thread = 1 state, BT positions; CB: thread = 4
    // adjacent entries of one row per CBI; x: this lane's 16 (position, row) values
    constexpr int CT = kSq / (NT / 64), BT = kSq / (NT / 128), CBI = kSq * kSq / 4 / NT;
    const int cn = (tid % 64) * 2, ct0 = (tid / 64) * CT, bn = tid % 128, bt0 = (tid / 128) * BT;
    float2 pc[CT];
    float4 pcb[CBI];
    float pb[BT], px[16], pat = 0.f, pdt = 0.f, paq = 0.f;
    auto fetch_cb = [&](int c) {
        if (c >= nc) return;
        const int c0 = c * kSq, nq = min(kSq, T - c0);
#pragma unroll
        for (int j = 0; j < CT; ++j) {
            const int t = ct0 + j;
            pc[j] = t < nq ? *reinterpret_cast<const float2*>(xbc + size_t(c0 + t) * kConvDim + kMInter + kMGroups * kMState + grp * kMState + cn)
                           : make_float2(0.f, 0.f);
        }
#pragma unroll
        for (int j = 0; j < BT; ++j) {
            const int t = bt0 + j;
            pb[j] = t < nq ? xbc[size_t(c0 + t) * kConvDim + kMInter + grp * kMState + bn] : 0.f;
        }
        const float4* cb = reinterpret_cast<const float4*>(CB + (size_t(grp) * nc + c) * kSq * kSq);
#pragma unroll
        for (int j = 0; j < CBI; ++j) pcb[j] = cb[tid + j * NT];
        if (lane < kSq && warp % 4 == 0) {
            pat = lane < nq ? acs[size_t(c0 + lane) * kMHeads + hd] : 0.f;
            pdt = lane < nq ? dtv[size_t(c0 + lane) * kMHeads + hd] : 0.f;
            paq = acs[size_t(c0 + nq - 1) * kMHeads + hd];
        }
    };
    auto fetch_x = [&](int c) {
        if (c >= nc) return;
        const int c0 = c * kSq, nq = min(kSq, T - c0);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int t = (j / 8) * 16 + g * 8 + j % 8;
            px[j] = t < nq ? xbc[size_t(c0 + t) * kConvDim + hd * kMHeadDim + p] : 0.f;
        }
    };
    fetch_cb(0);
    fetch_x(0);
    for (int c = 0; c < nc; ++c) {
        const int c0 = c * kSq, nq = min(kSq, T - c0);
        __syncthreads();                                              // the last Q-block's LDS reads are done
        if (lane < kSq && warp % 4 == 0) {
            w[hl][lane] = lane < nq ? expf(paq - pat) * pdt : 0.f;
            et[hl][lane] = expf(pat);
            at[hl][lane] = pat;
            dts[hl][lane] = pdt;
        }
#pragma unroll
        for (int j = 0; j < CT; ++j) {
            uint32_t hi, lo;
            sw_split2(pc[j].x, pc[j].y, hi, lo);
            *reinterpret_cast<uint32_t*>(&Ch[ct0 + j][cn]) = hi;
            *reinterpret_cast<uint32_t*>(&Cl[ct0 + j][cn]) = lo;
        }
#pragma unroll
        for (int j = 0; j < BT; j += 8) {
            uint4 hi, lo;
            sw_split8(pb + j, hi, lo);
            *reinterpret_cast<uint4*>(&Bh[bn][bt0 + j]) = hi;
            *reinterpret_cast<uint4*>(&Bl[bn][bt0 + j]) = lo;
        }
        __syncthreads();
        // M of every head of the block, from this thread's CB entries
#pragma unroll
        for (int j = 0; j < CBI; ++j) {
            const int i = (tid + j * NT) * 4, t = i / kSq, s = i % kSq;
            const float cv[4] = {pcb[j].x, pcb[j].y, pcb[j].z, pcb[j].w};
#pragma unroll
            for (int h = 0; h < HPB; ++h) {
                float m[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) m[k] = (s + k <= t && t < nq) ? cv[k] * expf(at[h][t] - at[h][s + k]) : 0.f;
                uint32_t h0, l0, h1, l1;
                sw_split2(m[0], m[1], h0, l0);
                sw_split2(m[2], m[3], h1, l1);
                *reinterpret_cast<uint2*>(&Mh[h][t][s]) = make_uint2(h0, h1);
                *reinterpret_cast<uint2*>(&Ml[h][t][s]) = make_uint2(l0, l1);
            }
        }
        fetch_cb(c + 1);
        __syncthreads();                                              // M is in
        // the Q-block's own outputs: D x, plus M times x scaled by dt
        sw_v8f Yi[2];
#pragma unroll
        for (int tt = 0; tt < 2; ++tt)
#pragma unroll
            for (int i = 0; i < 8; ++i) Yi[tt][i] = Dh * px[tt * 8 + i];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            float u[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) u[i] = px[ks * 8 + i] * dts[hl][ks * 16 + g * 8 + i];
            uint4 dh, dl;
            sw_split8(u, dh, dl);
#pragma unroll
            for (int tt = ks; tt < 2; ++tt) {
                const uint4 ah = *reinterpret_cast<const uint4*>(&Mh[hl][tt * 16 + l16][ks * 16 + g * 8]);
                const uint4 al = *reinterpret_cast<const uint4*>(&Ml[hl][tt * 16 + l16][ks * 16 + g * 8]);
                Yi[tt] = sw_wmma(ah, dh, Yi[tt]);
                Yi[tt] = sw_wmma(ah, dl, Yi[tt]);
                Yi[tt] = sw_wmma(al, dh, Yi[tt]);
            }
        }
        {   // the carried-in part, with the state before this Q-block
            sw_v8f Y[2] = {};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                float hv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = H[ks][i];
                uint4 hh, hlo;
                sw_split8(hv, hh, hlo);
#pragma unroll
                for (int tt = 0; tt < 2; ++tt) {
                    const uint4 ah = *reinterpret_cast<const uint4*>(&Ch[tt * 16 + l16][ks * 16 + g * 8]);
                    const uint4 al = *reinterpret_cast<const uint4*>(&Cl[tt * 16 + l16][ks * 16 + g * 8]);
                    Y[tt] = sw_wmma(ah, hh, Y[tt]);
                    Y[tt] = sw_wmma(ah, hlo, Y[tt]);
                    Y[tt] = sw_wmma(al, hh, Y[tt]);
                }
            }
#pragma unroll
            for (int tt = 0; tt < 2; ++tt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int t = tt * 16 + g * 8 + i;
                    if (t < nq) y[size_t(c0 + t) * kMInter + hd * kMHeadDim + p] = Yi[tt][i] + et[hl][t] * Y[tt][i];
                }
        }
        // the state: decay, then the block's inputs (x scaled by w)
        uint4 xh[2], xl[2];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = px[ks * 8 + i] * w[hl][ks * 16 + g * 8 + i];
            sw_split8(v, xh[ks], xl[ks]);
        }
        fetch_x(c + 1);
        const float e = ea[size_t(c) * kMHeads + hd];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            H[nt] = H[nt] * e;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const uint4 ah = *reinterpret_cast<const uint4*>(&Bh[nt * 16 + l16][ks * 16 + g * 8]);
                const uint4 al = *reinterpret_cast<const uint4*>(&Bl[nt * 16 + l16][ks * 16 + g * 8]);
                H[nt] = sw_wmma(ah, xh[ks], H[nt]);
                H[nt] = sw_wmma(ah, xl[ks], H[nt]);
                H[nt] = sw_wmma(al, xh[ks], H[nt]);
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<float4*>(hp + size_t(p) * kMState + nt * 16 + g * 8) = make_float4(H[nt][0], H[nt][1], H[nt][2], H[nt][3]);
        *reinterpret_cast<float4*>(hp + size_t(p) * kMState + nt * 16 + g * 8 + 4) = make_float4(H[nt][4], H[nt][5], H[nt][6], H[nt][7]);
    }
}

// commit: move the Mamba state forward over the first m <= kScanTile positions of a checked window (no outputs).
// The same per-step arithmetic as k_mamba_scan (h = h * dA + dt B x), so the stored state equals the scan's at the
// same step; the step values are read before the steps, as there.
__global__ void __launch_bounds__(128) k_mamba_commit(const float* xbc, const float* proj, const uint16_t* A_log,
                                                      const uint16_t* dt_bias, float* ssm, float* conv_state, int m) {
    constexpr int RP = kMHeadDim / kScanSlices;
    // the conv state too (k_mamba_conv_many's, after m positions): the last 3 raw inputs of [state, x_0 .. x_m-1],
    // one channel per thread of the grid. The window's conv outputs (xbc) are already the check's.
    if (const int ch = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; ch < kConvDim) {
        float* st = conv_state + size_t(ch) * (kConvK - 1);
        float v0 = st[0], v1 = st[1], v2 = st[2];
        for (int t = 0; t < m; ++t) { v0 = v1; v1 = v2; v2 = proj[size_t(t) * kInProj + kMInter + ch]; }
        st[0] = v0; st[1] = v1; st[2] = v2;
    }
    __shared__ float s_dt[kScanTile], s_dA[kScanTile], s_x[kScanTile][RP];
    const int hd = blockIdx.x, n = threadIdx.x;
    const int g = hd / (kMHeads / kMGroups), p0 = blockIdx.y * RP;
    const float A = -expf(bf16f(A_log[hd])), db = bf16f(dt_bias[hd]);
    float* hp = ssm + size_t(hd) * kMHeadDim * kMState + size_t(p0) * kMState + n;
    float hr[RP];
#pragma unroll
    for (int p = 0; p < RP; ++p) hr[p] = hp[p * kMState];
    if (n < m) {
        float dt = proj[size_t(n) * kInProj + kMInter + kConvDim + hd] + db;
        dt = dt > 20.f ? dt : log1pf(expf(dt));
        s_dt[n] = dt;
        s_dA[n] = expf(dt * A);
    }
    for (int i = n; i < m * RP; i += 128) s_x[i / RP][i % RP] = xbc[size_t(i / RP) * kConvDim + hd * kMHeadDim + p0 + i % RP];
    float Bn[kScanTile];
#pragma unroll
    for (int t = 0; t < kScanTile; ++t)
        if (t < m) Bn[t] = xbc[size_t(t) * kConvDim + kMInter + g * kMState + n];
    __syncthreads();
#pragma unroll
    for (int t = 0; t < kScanTile; ++t) {
        if (t >= m) break;
        const float dA = s_dA[t], dtB = s_dt[t] * Bn[t];
#pragma unroll
        for (int p = 0; p < RP; ++p) hr[p] = hr[p] * dA + dtB * s_x[t][p];
    }
#pragma unroll
    for (int p = 0; p < RP; ++p) hp[p * kMState] = hr[p];
}

// the index of each row's largest value (ties to the lower index, as std::max_element); one block of 1024 per row
__global__ void __launch_bounds__(1024) k_argmax_rows(const float* x, int n, int32_t* out) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const float* r = x + size_t(blockIdx.x) * n;
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024) { const float v = r[i]; if (v > bv) { bv = v; bi = i; } }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) {
        if (threadIdx.x < s && (sv[threadIdx.x + s] > sv[threadIdx.x] || (sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x]))) {
            sv[threadIdx.x] = sv[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}
// the step's picks grouped by expert (groups in order of first appearance, each taking its slots in slot order), one
// thread per slot: every loop is over the step's S <= 256 picks, not the layer's experts (33 us a call before, 27 on
// one thread)
__global__ void __launch_bounds__(256) k_group_slots_par(const int32_t* idx, int S, int k, int n_exp, int32_t* ng, int32_t* g_exp,
                                                         int32_t* g_cnt, int32_t* g_tok, int32_t* g_slot) {
    (void) n_exp;
    __shared__ int sidx[256], first[256], gof[256];
    const int j = threadIdx.x;
    if (j < S) sidx[j] = idx[j];
    __syncthreads();
    int e = -1, f = j, m = 0;                                        // f: the expert's first slot, m: rank among its slots
    if (j < S) {
        e = sidx[j];
        for (int i = j - 1; i >= 0; --i) if (sidx[i] == e) { f = i; ++m; }
    }
    first[j] = j < S && f == j;
    __syncthreads();
    if (first[j]) { int g = 0; for (int i = 0; i < j; ++i) g += first[i]; gof[j] = g; }
    __syncthreads();
    if (j < S) {
        const int g = gof[f];
        g_tok[g * 16 + m] = j / k;
        g_slot[g * 16 + m] = j;
        if (f == j) { int c = 0; for (int i = j; i < S; ++i) c += sidx[i] == e; g_exp[g] = e; g_cnt[g] = c; }
    }
    if (j == 0) { int n = 0; for (int i = 0; i < S; ++i) n += first[i]; *ng = n; }
}

// rows of kHidden floats into the residual stream at the positions map[t] >= 0 names (strata-nemo --serve: an image's
// embeddings in place of its <image> tokens)
__global__ void k_put_rows(float* h, const float* rows, const int32_t* map) {
    const int r = map[blockIdx.x];
    if (r < 0) return;
    for (int i = threadIdx.x; i < kHidden; i += blockDim.x) h[size_t(blockIdx.x) * kHidden + i] = rows[size_t(r) * kHidden + i];
}
