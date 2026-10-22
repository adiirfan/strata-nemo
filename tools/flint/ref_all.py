"""ref_all.py IDSFILE OUT (FLINT_GGUF=model.gguf): Flint in plain numpy (float64 activations), every position's logits (float32, n x vocab) to OUT.
As ref.py, with the experts batched (each expert's tokens together) so a prompt of a few thousand is minutes, not days."""
import os
import sys
import numpy as np
if os.environ.get("GGUF_PY"):   # (a gguf-py checkout, when the gguf package is not installed)
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader
from gguf.quants import dequantize

R = GGUFReader(os.environ["FLINT_GGUF"])   # the Flint Q4_1 GGUF
T = {t.name: t for t in R.tensors}
H, NL, NH, NKV, HD, WIN, EPS, BASE = 2048, 49, 32, 4, 128, 513, 1e-6, 10000.0
swa = [bool(x) for x in R.fields["flint.attention.sliding_window_pattern"].contents()]


def W(name):
    t = T[name]
    return dequantize(t.data, t.tensor_type).astype(np.float64)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def rope(x, pos):
    j = np.arange(64)
    ang = pos[:, None] * BASE ** (-2.0 * j / HD)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :64], x[..., 64:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def silu(x):
    return x / (1 + np.exp(-x))


ids = [int(v) for v in open(sys.argv[1]).read().replace("\n", "").split(",") if v]
n = len(ids)
pos = np.arange(n, dtype=np.float64)
emb = T["token_embd.weight"].data
h = np.array([emb[i] for i in ids], dtype=np.float64)
P = np.arange(n)
for il in range(NL):
    b = f"blk.{il}."
    x = rms(h, W(b + "attn_norm.weight"))
    q = (x @ W(b + "attn_q.weight").T).reshape(n, NH, HD)
    k = (x @ W(b + "attn_k.weight").T).reshape(n, NKV, HD)
    v = (x @ W(b + "attn_v.weight").T).reshape(n, NKV, HD)
    q = rms(q, W(b + "attn_q_norm.weight"))
    k = rms(k, W(b + "attn_k_norm.weight"))
    if swa[il]:
        q, k = rope(q, pos), rope(k, pos)
    sinks = W(b + "attn_sinks.weight")
    mask = P[None, :] <= P[:, None]
    if swa[il]:
        mask &= P[None, :] >= P[:, None] - (WIN - 1)
    o = np.zeros((n, NH, HD))
    for hh in range(NH):
        kh = hh // (NH // NKV)
        sc = q[:, hh] @ k[:, kh].T / np.sqrt(HD)
        sc = np.where(mask, sc, -np.inf)
        m = np.maximum(sc.max(1), sinks[hh])
        e = np.exp(sc - m[:, None])
        o[:, hh] = (e @ v[:, kh]) / (e.sum(1) + np.exp(sinks[hh] - m))[:, None]
    h = h + o.reshape(n, NH * HD) @ W(b + "attn_output.weight").T
    x = rms(h, W(b + "ffn_norm.weight"))
    if il < 2:
        f = silu(x @ W(b + "ffn_gate.weight").T) * (x @ W(b + "ffn_up.weight").T)
        h = h + f @ W(b + "ffn_down.weight").T
    else:
        out = (silu(x @ W(b + "ffn_gate_shexp.weight").T) * (x @ W(b + "ffn_up_shexp.weight").T)) @ W(b + "ffn_down_shexp.weight").T
        prob = 1 / (1 + np.exp(-(x @ W(b + "ffn_gate_inp.weight").T)))
        top = np.argsort(-prob, axis=1)[:, :6]
        wt = np.take_along_axis(prob, top, 1)
        wt /= wt.sum(1, keepdims=True)
        tg, tu, td = (T[b + nm] for nm in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        for e in np.unique(top):
            rows, slot = np.nonzero(top == e)
            xe = x[rows]
            g = xe @ dequantize(tg.data[e], tg.tensor_type).astype(np.float64).T
            u = xe @ dequantize(tu.data[e], tu.tensor_type).astype(np.float64).T
            y = (silu(g) * u) @ dequantize(td.data[e], td.tensor_type).astype(np.float64).T
            np.add.at(out, rows, wt[rows, slot][:, None] * y)
        h = h + out
    print(f"layer {il} done", file=sys.stderr, flush=True)
x = rms(h, W("output_norm.weight"))
Wo = W("output.weight")
with open(sys.argv[2], "wb") as f:
    for i in range(0, n, 256):
        (x[i:i + 256] @ Wo.T).astype(np.float32).tofile(f)
