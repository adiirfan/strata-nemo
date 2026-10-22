"""tools/ling/ref_ling.py IDS OUT [--layers N] [--dump H]: Ling-3.0-flash-VL (bailingmoe3) in float64 numpy, every position's logits (float32,
n x vocab) to OUT - the exact answers the engine is checked against. Weights dequantized from the GGUF (both parts).
IDS: comma-separated token ids or @file. Follows llama.cpp's src/models/bailingmoe3.cpp + delta-net-base.cpp:

  KDA layers: q/k/v = silu(causal depthwise conv4(W x)); q, k L2-normed per 128-head; per-channel decay
  g = -5 * sigmoid(a_h * (W_f x + dt_b)); beta = sigmoid(W_beta x) a head; S[i][j] *= exp(g[i]); d = (v - S^T k) * beta;
  S += k d^T; o = S^T q / sqrt(128); o = rmsnorm_head(o) * sigmoid(W_g x); out = W_o o.
  MLA layers: q = W_q x (32 x [nope 128 | rope 64]); [lat 512 | k_rope 64] = W_kva x; lat = rmsnorm(lat); rope NEOX
  (dims i, i + 32; base 6e6) on q_rope, k_rope; score = (wk_b_h q_nope) . lat + q_rope . k_rope, / sqrt(192); out_h =
  wv_b_h (sum p lat) * sigmoid(W_gate x)_h; out = W_o out.
  MoE: p = sigmoid(W_r x); pick on p + bias: 8 groups of 64 scored by their top-2 sum, best 4 groups, top 8 experts;
  weights p (unbiased), normalized, x 2.5; swiglu with the layer's clamp (gate <= L, |up| <= L) when L > 0; + shared expert.
"""
import os
import sys
import numpy as np

if os.environ.get("GGUF_PY"):   # llama.cpp's gguf-py folder, if the gguf package is not installed
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader
from gguf.quants import dequantize

D = os.environ.get("LING_DIR", ".")   # the folder holding both GGUF parts
R = [GGUFReader(f"{D}/Ling-3.0-flash-VL-Q4_0-0000{i}-of-00002.gguf") for i in (1, 2)]
T = {t.name: t for r in R for t in r.tensors}
F = R[0].fields
def kv(name):
    return F["bailingmoe3." + name].contents()

H, NL, NH, EPS = 2560, int(kv("block_count")), 32, float(kv("attention.layer_norm_rms_epsilon"))
KVH = kv("attention.head_count_kv")
CLAMP_E, CLAMP_S = kv("swiglu_clamp_exp"), kv("swiglu_clamp_shexp")
BASE, NROT, NEXP, TOPK, NG, NGU = float(kv("rope.freq_base")), 64, 512, 8, 8, 4
SCALE_W = float(kv("expert_weights_scale"))
LB = float(kv("kda.gate_lower_bound"))


def W(name):
    t = T[name]
    return dequantize(t.data, t.tensor_type).astype(np.float64)


def expert(name, e):
    t = T[name]
    return dequantize(t.data[e], t.tensor_type).astype(np.float64)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1 + np.exp(-x))


def sig(x):
    return 1 / (1 + np.exp(-x))


def rope_neox(x, pos):   # x (n, heads, 64)
    j = np.arange(NROT // 2)
    ang = pos[:, None] * BASE ** (-2.0 * j / NROT)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :NROT // 2], x[..., NROT // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def swiglu(g, u, lim):
    if lim > 0:
        g = np.minimum(g, lim)
        u = np.clip(u, -lim, lim)
    return silu(g) * u


arg = sys.argv[1]
ids = [int(v) for v in (open(arg[1:]).read() if arg.startswith("@") else arg).replace("\n", "").split(",") if v.strip()]
n = len(ids)
nl = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else NL
pos = np.arange(n, dtype=np.float64)
emb = T["token_embd.weight"]
h = np.stack([dequantize(emb.data[i], emb.tensor_type) for i in ids]).astype(np.float64)
P = np.arange(n)
for il in range(nl):
    b = f"blk.{il}."
    x = rms(h, W(b + "attn_norm.weight"))
    if KVH[il] == 0:   # KDA
        def conv(wname, cname):
            xp = x @ W(b + wname).T                                  # (n, 4096)
            cw = W(b + cname).reshape(-1, 4)                         # (4096, 4)
            pad = np.concatenate([np.zeros((3, xp.shape[1])), xp])
            out = sum(pad[k:k + n] * cw[:, k] for k in range(4))
            return silu(out).reshape(n, NH, 128)
        q, k, v = conv("attn_q.weight", "ssm_conv1d_q.weight"), conv("attn_k.weight", "ssm_conv1d_k.weight"), conv("attn_v.weight", "ssm_conv1d_v.weight")
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + EPS)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + EPS)
        a = W(b + "ssm_a").reshape(NH)
        g = (x @ W(b + "ssm_f_a.weight").T + W(b + "ssm_dt.bias")).reshape(n, NH, 128)
        g = LB * sig(g * a[None, :, None])
        beta = sig(x @ W(b + "ssm_beta.weight").T)                  # (n, 32)
        o = np.zeros((n, NH, 128))
        for hh in range(NH):
            S = np.zeros((128, 128))                                 # S[i][j]: key i, value j
            for t in range(n):
                S *= np.exp(g[t, hh])[:, None]
                d = (v[t, hh] - S.T @ k[t, hh]) * beta[t, hh]
                S += np.outer(k[t, hh], d)
                o[t, hh] = S.T @ q[t, hh] / np.sqrt(128)
        o = rms(o, W(b + "ssm_norm.weight")) * sig((x @ W(b + "ssm_g_a.weight").T).reshape(n, NH, 128))
        att = o.reshape(n, NH * 128) @ W(b + "attn_output.weight").T
    else:              # gated MLA
        qa = (x @ W(b + "attn_q.weight").T).reshape(n, NH, 192)
        q_nope, q_pe = qa[..., :128], rope_neox(qa[..., 128:], pos)
        kva = x @ W(b + "attn_kv_a_mqa.weight").T
        lat = rms(kva[:, :512], W(b + "attn_kv_a_norm.weight"))
        k_pe = rope_neox(kva[:, None, 512:], pos)[:, 0]
        wkb = W(b + "attn_k_b.weight").reshape(NH, 512, 128)
        wvb = W(b + "attn_v_b.weight").reshape(NH, 128, 512)
        gate = sig(x @ W(b + "attn_gate.weight").T)                 # (n, 32)
        o = np.zeros((n, NH, 128))
        mask = P[None, :] <= P[:, None]
        for hh in range(NH):
            ql = q_nope[:, hh] @ wkb[hh].T                           # (n, 512): (wk_b_h q_nope)
            sc = (ql @ lat.T + q_pe[:, hh] @ k_pe.T) / np.sqrt(192)
            sc = np.where(mask, sc, -np.inf)
            p = np.exp(sc - sc.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            o[:, hh] = (p @ lat) @ wvb[hh].T * gate[:, hh:hh + 1]
        att = o.reshape(n, NH * 128) @ W(b + "attn_output.weight").T
    h = h + att
    x = rms(h, W(b + "ffn_norm.weight"))
    if b + "ffn_gate.weight" in T:   # dense
        h = h + swiglu(x @ W(b + "ffn_gate.weight").T, x @ W(b + "ffn_up.weight").T, 0) @ W(b + "ffn_down.weight").T
    else:
        out = swiglu(x @ W(b + "ffn_gate_shexp.weight").T, x @ W(b + "ffn_up_shexp.weight").T, CLAMP_S[il]) @ W(b + "ffn_down_shexp.weight").T
        prob = sig(x @ W(b + "ffn_gate_inp.weight").T)               # (n, 512)
        sel = prob + W(b + "exp_probs_b.bias")
        for t in range(n):
            gs = np.sort(sel[t].reshape(NG, -1), axis=1)[:, -2:].sum(1)
            keep_g = np.argsort(-gs, kind="stable")[:NGU]
            masked = np.full(NEXP, -np.inf)
            for gi in keep_g:
                masked[gi * 64:(gi + 1) * 64] = sel[t, gi * 64:(gi + 1) * 64]
            top = np.argsort(-masked, kind="stable")[:TOPK]
            w = prob[t, top] / prob[t, top].sum() * SCALE_W
            for e, we in zip(top, w):
                gv = expert(b + "ffn_gate_exps.weight", e) @ x[t]
                uv = expert(b + "ffn_up_exps.weight", e) @ x[t]
                out[t] += we * (expert(b + "ffn_down_exps.weight", e) @ swiglu(gv, uv, CLAMP_E[il]))
        h = h + out
    print(f"layer {il} ({'kda' if KVH[il] == 0 else 'mla'}): |h| {np.linalg.norm(h[-1]):.3f}", file=sys.stderr, flush=True)
    if "--dump" in sys.argv:   # every layer's output, all positions (float32, appended: layer-major)
        with open(sys.argv[sys.argv.index("--dump") + 1], "ab" if il else "wb") as f:
            h.astype(np.float32).tofile(f)
x = rms(h, W("output_norm.weight"))
Wo = W("output.weight")
with open(sys.argv[2], "wb") as f:
    for i in range(0, n, 64):
        (x[i:i + 64] @ Wo.T).astype(np.float32).tofile(f)
lg = x[-1] @ Wo.T
top = np.argsort(-lg)[:5]
print("last position, top 5:", " ".join(f"{t}:{lg[t]:.3f}" for t in top))
