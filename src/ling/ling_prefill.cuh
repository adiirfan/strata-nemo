// src/ling/ling_prefill.cuh - strata-ling's prompt path: T tokens a chunk through every layer (included by ling_main.cu).
// First version: fp32 SIMT products on the GGUF's own formats (exact, not fast), KDA's recurrence as one block a head
// stepping through the chunk with its state column in registers, MLA's attention one block a query over all 32 heads
// (the latent is shared, so each key row is read once for the 32), the routed experts grouped by expert.

// ---- Y[t][n] (= or +=) sum_k W[n][k] X[t][k]: W N x K rows of format fmt, X T x K fp32 (row stride ldx), Y row stride
// ldy. Tiles of 64 rows x 64 tokens, k 32 a step through shared memory; a thread 4 rows x 4 tokens.
// With src / bounds / tiles: the routed experts' grouped form - tile i is (expert e, first row r0) of the grouped
// order, X row of grouped row r is src[r], W_e = W + e * mat_bytes (or wp[e] when wp is given: the expert's own pointer).
struct Gemm {
    const uint8_t* W;
    const uint8_t* const* wp;
    size_t mat_bytes;
    int fmt, N, K;
    const float* X;
    int ldx, T;
    float* Y;
    int ldy;
    bool add;
    const int *src, *bounds, *tiles, *n_tiles;
};
__global__ void __launch_bounds__(256) k_gemm(Gemm J) {
    __shared__ float ws[64][33], xs[64][33];
    __shared__ int srow[64];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;   // tx: 4 tokens each, ty: 4 rows each
    const int n0 = blockIdx.x * 64;
    int t0, tn;
    const uint8_t* W = J.W;
    if (J.tiles) {   // grouped
        const int tile = blockIdx.y;
        if (tile >= *J.n_tiles) return;
        const int e = J.tiles[2 * tile];
        t0 = J.bounds[e] + J.tiles[2 * tile + 1];
        tn = min(64, J.bounds[e + 1] - t0);
        W = J.wp ? J.wp[e] : J.W + size_t(e) * J.mat_bytes;
        if (tid < 64) srow[tid] = tid < tn ? (J.src ? J.src[t0 + tid] : t0 + tid) : -1;
    } else {
        t0 = blockIdx.y * 64;
        tn = min(64, J.T - t0);
        if (tid < 64) srow[tid] = tid < tn ? t0 + tid : -1;
    }
    __syncthreads();
    const size_t rb = row_bytes(J.fmt, J.K);
    float acc[4][4] = {};
    for (int k0 = 0; k0 < J.K; k0 += 32) {
        {   // weights: 64 rows x 32 values = 256 groups of 8, one a thread
            const int r = tid >> 2, g = tid & 3;
            float v[8];
            if (n0 + r < J.N) deq8(J.fmt, W + size_t(n0 + r) * rb, (k0 >> 3) + g, v, J.K);
            else for (int i = 0; i < 8; ++i) v[i] = 0.f;
            for (int i = 0; i < 8; ++i) ws[r][g * 8 + i] = v[i];
        }
        {   // inputs: 64 tokens x 32 values
            const int t = tid >> 2, c = (tid & 3) * 8;
            const int row = srow[t];
            for (int i = 0; i < 8; ++i) xs[t][c + i] = row >= 0 ? J.X[size_t(row) * J.ldx + k0 + c + i] : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = ws[ty * 4 + i][k]; b[i] = xs[tx * 4 + i][k]; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int j = 0; j < 4; ++j) {
        const int t = tx * 4 + j;
        if (t >= tn) continue;
        const size_t yrow = J.tiles ? size_t(t0 + t) : size_t(t0 + t);   // (grouped: the grouped order's row)
        for (int i = 0; i < 4; ++i) {
            const int n = n0 + ty * 4 + i;
            if (n >= J.N) continue;
            float* y = J.Y + yrow * J.ldy + n;
            *y = J.add ? *y + acc[i][j] : acc[i][j];
        }
    }
}

// rows of x: rmsnorm * w (T rows of K), 256 threads a row
__global__ void __launch_bounds__(256) k_rms_rows(const float* x, const float* w, float* y, int K) {
    __shared__ float red[8];
    const float* r = x + size_t(blockIdx.x) * K;
    float ss = 0;
    for (int i = threadIdx.x; i < K; i += 256) ss += r[i] * r[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < 8; ++i) t += red[i];
    const float s = rsqrtf(t / K + kEps);
    for (int i = threadIdx.x; i < K; i += 256) y[size_t(blockIdx.x) * K + i] = r[i] * s * w[i];
}

// ---- KDA over a chunk
// conv4 + silu for T tokens of one of q / k / v (the state: the 3 inputs before the chunk, oldest first), and the new state
__global__ void k_kda_conv(const float* xp, const float* w, const float* st_in, float* out, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kInner) return;
    for (int t = 0; t < T; ++t) {
        float o = 0;
        for (int j = 0; j < 4; ++j) {
            const int tt = t - 3 + j;
            const float x = tt >= 0 ? xp[size_t(tt) * kInner + c] : st_in[(3 + tt) * kInner + c];
            o += w[c * 4 + j] * x;
        }
        out[size_t(t) * kInner + c] = siluf(o);
    }
}
__global__ void k_kda_conv_state(const float* xp, float* st, int T) {   // the last 3 inputs (state's, then the chunk's)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kInner) return;
    float v[3];
    for (int j = 0; j < 3; ++j) {
        const int tt = T - 3 + j;
        v[j] = tt >= 0 ? xp[size_t(tt) * kInner + c] : st[(3 + tt) * kInner + c];
    }
    for (int j = 0; j < 3; ++j) st[j * kInner + c] = v[j];
}
// q, k: L2 norm a 128-head (block per token x head, 128 threads); decay = exp(-5 sigmoid(a_h (f + dt_b))) in f's place;
// beta = sigmoid(beta) in place
__global__ void __launch_bounds__(128) k_kda_prep(float* q, float* k, float* f, const float* dtb, const float* a, float* beta) {
    __shared__ float red[2][4];
    const int t = blockIdx.x / kHeads, h = blockIdx.x % kHeads, c = threadIdx.x;
    const size_t o = size_t(t) * kInner + h * kHd + c;
    const float qv = q[o], kv = k[o];
    float sq = warp_sum(qv * qv), sk = warp_sum(kv * kv);
    if ((c & 31) == 0) { red[0][c >> 5] = sq; red[1][c >> 5] = sk; }
    __syncthreads();
    sq = red[0][0] + red[0][1] + red[0][2] + red[0][3];
    sk = red[1][0] + red[1][1] + red[1][2] + red[1][3];
    q[o] = qv * rsqrtf(sq + kEps);
    k[o] = kv * rsqrtf(sk + kEps);
    f[o] = expf(kGateLow * sigmoidf(a[h] * (f[o] + dtb[h * kHd + c])));
    if (c == 0) beta[size_t(t) * kHeads + h] = sigmoidf(beta[size_t(t) * kHeads + h]);
}
// the delta rule through the chunk: block a head, thread j its state column S[:, j] in registers. o (T x 4096) is
// written as S^T q / sqrt(128)
__global__ void __launch_bounds__(128) k_kda_scan(const float* q, const float* k, const float* v, const float* dec,
                                                  const float* beta, float* S_all, float* o, int T) {
    __shared__ float ks[kHd], qs[kHd], ds[kHd];
    const int h = blockIdx.x, j = threadIdx.x;
    float* S = S_all + size_t(h) * kHd * kHd;
    float s[kHd];
#pragma unroll
    for (int i = 0; i < kHd; ++i) s[i] = S[i * kHd + j];
    const float sc = rsqrtf(float(kHd));
    for (int t = 0; t < T; ++t) {
        const size_t base = size_t(t) * kInner + h * kHd;
        __syncthreads();
        ks[j] = k[base + j];
        qs[j] = q[base + j];
        ds[j] = dec[base + j];
        __syncthreads();
        float sk = 0;
#pragma unroll
        for (int i = 0; i < kHd; ++i) { s[i] *= ds[i]; sk += s[i] * ks[i]; }
        const float d = (v[base + j] - sk) * beta[size_t(t) * kHeads + h];
        float ov = 0;
#pragma unroll
        for (int i = 0; i < kHd; ++i) { s[i] += ks[i] * d; ov += s[i] * qs[i]; }
        o[base + j] = ov * sc;
    }
#pragma unroll
    for (int i = 0; i < kHd; ++i) S[i * kHd + j] = s[i];
}
// o = rmsnorm_head(o) * onorm * sigmoid(g) (block per token x head)
__global__ void __launch_bounds__(128) k_kda_out(float* o, const float* g, const float* onorm) {
    __shared__ float red[4];
    const int t = blockIdx.x / kHeads, h = blockIdx.x % kHeads, c = threadIdx.x;
    const size_t i = size_t(t) * kInner + h * kHd + c;
    const float v = o[i];
    float s = warp_sum(v * v);
    if ((c & 31) == 0) red[c >> 5] = s;
    __syncthreads();
    s = red[0] + red[1] + red[2] + red[3];
    o[i] = v * rsqrtf(s / kHd + kEps) * onorm[c] * sigmoidf(g[i]);
}

// ---- MLA over a chunk
// the cache rows of positions p0 .. p0 + T - 1 (block a token, 512 threads)
__global__ void __launch_bounds__(512) k_pmla_kv(const float* kva, const float* norm, float* cache, int p0) {
    __shared__ float red[16], kr[kRope];
    const int t = threadIdx.x, tok = blockIdx.x, pos = p0 + tok;
    const float* in = kva + size_t(tok) * kKvA;
    const float x = in[t];
    float v = warp_sum(x * x);
    if ((t & 31) == 0) red[t >> 5] = v;
    if (t < kRope) kr[t] = in[kLat + t];
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
// the roped q_rope of every token and head into qa[t][h][512 ..] (block a token x head, 32 threads)
__global__ void k_pmla_qrope(const float* q, float* qa, int p0) {
    __shared__ float r[kRope];
    const int tok = blockIdx.x / kHeads, h = blockIdx.x % kHeads, t = threadIdx.x;
    const float* src = q + size_t(tok) * (kHeads * kQkHd) + h * kQkHd + kQkNope;
    r[t] = src[t];
    r[t + 32] = src[t + 32];
    __syncthreads();
    rope_neox(r, p0 + tok, t);
    __syncthreads();
    float* dst = qa + (size_t(tok) * kHeads + h) * kKvA + kLat;
    dst[t] = r[t];
    dst[t + 32] = r[t + 32];
}
// attention of one query (block) over its positions, all 32 heads: warp w heads w, w + 8, w + 16, w + 24; lane: 18 of
// the 576 key values and 16 of the 512 output values a head; online softmax. ol (T x 32 x 512)
__global__ void __launch_bounds__(256) k_pmla_attn(const float* qa, const float* cache, float* ol, int p0) {
    const int tok = blockIdx.x, n = p0 + tok + 1, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float scale = rsqrtf(float(kQkHd));
    float qv[4][18], acc[4][16], m[4], l[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float* qh = qa + (size_t(tok) * kHeads + w + 8 * a) * kKvA;
#pragma unroll
        for (int i = 0; i < 18; ++i) qv[a][i] = qh[lane + 32 * i] * scale;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[a][i] = 0.f;
        m[a] = -INFINITY;
        l[a] = 0.f;
    }
    for (int s = 0; s < n; ++s) {
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
            for (int i = 0; i < 16; ++i) acc[a][i] = acc[a][i] * cf + p * kv[i];   // (lane's 16 of the latent: 0..511)
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        float* dst = ol + (size_t(tok) * kHeads + w + 8 * a) * kLat;
#pragma unroll
        for (int i = 0; i < 16; ++i) dst[lane + 32 * i] = acc[a][i] / l[a];
    }
}
// out[t][h][0..128) *= sigmoid(gate[t][h]) (block a token, 128 x 32 threads... 256 threads looping)
__global__ void __launch_bounds__(256) k_pmla_gate(float* out, const float* gate) {
    const int tok = blockIdx.x;
    for (int i = threadIdx.x; i < kInner; i += 256) out[size_t(tok) * kInner + i] *= sigmoidf(gate[size_t(tok) * kHeads + i / kHd]);
}

// ---- MoE over a chunk: route every token, then group the rows by expert
__global__ void __launch_bounds__(512) k_proute(const float* logits, const float* bias, int* idx, float* wts, int* count) {
    const int tok = blockIdx.x;   // (k_route's rule, one block a token)
    route_block(logits + size_t(tok) * kExperts, bias, idx + tok * kTopK, wts + tok * kTopK, count);
}
// bounds (exclusive prefix sums of the counts) and the tiles of 64 rows; one block of 512
__global__ void __launch_bounds__(512) k_pbounds(const int* count, int* bounds, int* fill, int* tiles, int* n_tiles) {
    if (threadIdx.x != 0) return;
    int s = 0, nt = 0;
    for (int e = 0; e < kExperts; ++e) {
        bounds[e] = s;
        fill[e] = 0;
        for (int r = 0; r < count[e]; r += 64) { tiles[2 * nt] = e; tiles[2 * nt + 1] = r; ++nt; }
        s += count[e];
    }
    bounds[kExperts] = s;
    *n_tiles = nt;
}
// token t's k-th choice -> its grouped row; src[row] = t, slot[t][k] = row
__global__ void k_pscatter(const int* idx, int T, const int* bounds, int* fill, int* src, int* slot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * kTopK) return;
    const int e = idx[i], r = bounds[e] + atomicAdd(&fill[e], 1);
    src[r] = i / kTopK;
    slot[i] = r;
}
// act[r] = swiglu(gate[r], up[r]) with the clamp, both n_ff wide (gate in place)
__global__ void k_pswiglu(float* g, const float* u, size_t n, float clamp) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (clamp > 0.f) { gv = fminf(gv, clamp); uv = fmaxf(-clamp, fminf(uv, clamp)); }
    g[i] = siluf(gv) * uv;
}
// h[t] += sum_k w[t][k] * y[slot[t][k]] (block a token)
__global__ void __launch_bounds__(256) k_pcombine(float* h, const float* y, const int* slot, const float* wts) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        float s = 0;
        for (int k = 0; k < kTopK; ++k) s += wts[t * kTopK + k] * y[size_t(slot[t * kTopK + k]) * kHidden + i];
        h[size_t(t) * kHidden + i] += s;
    }
}
__global__ void k_pembed(const uint8_t* table, int fmt, const int* ids, float* h) {
    const int tok = blockIdx.x, g = threadIdx.x;   // (320 groups of 8 a row)
    if (g >= kHidden / 8) return;
    float v[8];
    deq8(fmt, table + size_t(ids[tok]) * row_bytes(fmt, kHidden), g, v, kHidden);
    for (int i = 0; i < 8; ++i) h[size_t(tok) * kHidden + g * 8 + i] = v[i];
}

// ---- the prompt path's products on the matrix cores (gfx12 WMMA 16x16x16, fp16 in, fp32 sums): X split per row into
// fp16 hi + lo with a power-of-2 scale (exact to ~22 bits, as strata-flint's prompt path), W unpacked by deq8 into the A
// operand (lane l16: row, g: k half - deq8's 8 consecutive values), as fp16. A block: 8 warps over 256 rows (2 tiles of
// 16 each), 64 tokens (4 tiles), k 64 a step through LDS (double-buffered). GROUPED: as k_gemm's grouped form.
typedef _Float16 lp_v8h __attribute__((ext_vector_type(8)));
typedef float lp_v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ lp_v8f lp_wmma(const lp_v8h& a, const lp_v8h& b, const lp_v8f& c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __trap();
    return c;
#endif
}
// x rows (T x K, row stride ldx) -> hi / lo fp16 (T x K) and each row's scale (xs[t] = 2^(e - 14), the row's max near 2^14)
__global__ void __launch_bounds__(256) k_split16(const float* x, int ldx, int K, uint16_t* xh, uint16_t* xl, float* xs) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    const float* r = x + size_t(t) * ldx;
    float mx = 0;
    for (int i = threadIdx.x; i < K; i += 256) mx = fmaxf(mx, fabsf(r[i]));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, fl_shfl_xor(mx, o, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    float m = 0;
    for (int i = 0; i < 8; ++i) m = fmaxf(m, red[i]);
    const int e = m > 0.f ? ilogbf(m) : 0;
    const float up = ldexpf(1.f, 14 - e);
    for (int i = threadIdx.x; i < K; i += 256) {
        const float a = r[i] * up;
        const __half hi = __float2half(a);
        const __half lo = __float2half(a - __half2float(hi));
        uint16_t uh, ul;
        std::memcpy(&uh, &hi, 2);
        std::memcpy(&ul, &lo, 2);
        xh[size_t(t) * K + i] = uh;
        xl[size_t(t) * K + i] = ul;
    }
    if (threadIdx.x == 0) xs[t] = ldexpf(1.f, e - 14);
}
// 8 consecutive weights straight to fp16 for the A operand. AQ4_0 / AQ4_1 / AQ8_0: packed fp16 maths on the exact
// codes (nibble n as the half 1024 + n, then - 1032 or - 1024: exact; then one rounding, as converting deq8's fp32 would);
// other formats through deq8.
__device__ __forceinline__ lp_v8h deq8h(int fmt, const uint8_t* row, int g, int K) {
    lp_v8h r;
    if (fmt == AQ4_0 || fmt == AQ4_1) {
        const int b = g >> 2, j0 = (g & 3) * 8, sh = j0 >= 16 ? 4 : 0, nb = K / 32;
        const uint2 q = *(const uint2*) (row + size_t(b) * 16 + (j0 & 15));
        const uint32_t u[2] = {(q.x >> sh) & 0x0F0F0F0Fu, (q.y >> sh) & 0x0F0F0F0Fu};
        const uint8_t* sc = row + size_t(nb) * 16;
        _Float16 d, m;
        if (fmt == AQ4_0) { d = *(const _Float16*) (sc + b * 2); m = (_Float16) 0.f; }
        else { d = *(const _Float16*) (sc + b * 4); m = *(const _Float16*) (sc + b * 4 + 2); }
        const _Float16 off = (_Float16) (fmt == AQ4_0 ? -1032.f : -1024.f);
#pragma unroll
        for (int w = 0; w < 2; ++w)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint16_t bits = uint16_t(0x6400u | ((u[w] >> (8 * k)) & 0xFFu));
                const _Float16 code = __builtin_bit_cast(_Float16, bits) + off;   // (exact: a small integer)
                r[w * 4 + k] = fmt == AQ4_0 ? code * d : code * d + m;
            }
        return r;
    }
    if (fmt == AQ8_0) {
        const int b = g >> 2, j0 = (g & 3) * 8, nb = K / 32;
        const uint2 q = *(const uint2*) (row + size_t(b) * 32 + j0);
        const int8_t* qb = (const int8_t*) &q;
        const _Float16 d = *(const _Float16*) (row + size_t(nb) * 32 + b * 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = (_Float16) qb[i] * d;
        return r;
    }
    float v[8];
    deq8(fmt, row, g, v, K);
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = (_Float16) v[i];
    return r;
}
struct GemmW {
    const uint8_t* W;
    const uint8_t* const* wp;
    int fmt, N, K;
    const uint16_t *xh, *xl;
    const float* xs;
    int T;
    float* Y;
    int ldy;
    bool add;
    const int *src, *bounds, *tiles, *n_tiles;
};
template <bool GROUPED>
__global__ void __launch_bounds__(256) k_gemm_w(GemmW J) {
    constexpr int TA = 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];
    __shared__ int srow[BN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int o0 = blockIdx.x * 256 + warp * TA * 16;
    int t0, tn;
    const uint8_t* W = J.W;
    if (GROUPED) {
        const int tile = blockIdx.y;
        if (tile >= *J.n_tiles) return;
        const int e = J.tiles[2 * tile];
        t0 = J.bounds[e] + J.tiles[2 * tile + 1];
        tn = min(BN, J.bounds[e + 1] - t0);
        W = J.wp[e];
        if (tid < BN) srow[tid] = tid < tn ? (J.src ? J.src[t0 + tid] : t0 + tid) : -1;
    } else {
        t0 = blockIdx.y * BN;
        tn = min(BN, J.T - t0);
        if (tid < BN) srow[tid] = tid < tn ? t0 + tid : -1;
    }
    __syncthreads();
    const size_t rb = row_bytes(J.fmt, J.K);
    const uint8_t* wr[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) wr[a] = W + size_t(min(o0 + a * 16 + l16, J.N - 1)) * rb;
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            const int row = srow[t];
            const uint16_t* sp = hl ? J.xl : J.xh;
            r[j] = row >= 0 ? *(const uint4*) (sp + size_t(row) * J.K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            *(uint4*) &sx[buf][hl][t][c] = r[j];
        }
    };
    lp_v8f acc[TA][TB];
#pragma unroll
    for (int a = 0; a < TA; ++a)
#pragma unroll
        for (int b = 0; b < TB; ++b) acc[a][b] = lp_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rbuf[4];
    fetch(0, rbuf);
    put(rbuf, 0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < J.K; k0 += BK) {
        if (k0 + BK < J.K) fetch(k0 + BK, rbuf);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            lp_v8h fa[TA];
#pragma unroll
            for (int a = 0; a < TA; ++a) fa[a] = deq8h(J.fmt, wr[a], (k0 + kk * 16) / 8 + g, J.K);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const lp_v8h bh = *(const lp_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
                const lp_v8h bl = *(const lp_v8h*) &sx[buf][1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int a = 0; a < TA; ++a) { acc[a][b] = lp_wmma(fa[a], bh, acc[a][b]); acc[a][b] = lp_wmma(fa[a], bl, acc[a][b]); }
            }
        }
        if (k0 + BK < J.K) put(rbuf, buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    // lane: outputs o0 + a * 16 + g * 8 + i, token tile b, column l16
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = b * 16 + l16;
        if (t >= tn) continue;
        const int row = srow[t];
        const float sc = J.xs[row];
        const size_t yrow = size_t(t0 + t);
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int o = o0 + a * 16 + g * 8 + i;
                if (o >= J.N) continue;
                float* y = J.Y + yrow * J.ldy + o;
                const float v = acc[a][b][i] * sc;
                *y = J.add ? *y + v : v;
            }
    }
}
