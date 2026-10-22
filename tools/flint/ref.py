"""ref.py IDS (FLINT_GGUF=model.gguf) [--dump FILE]: Flint in plain numpy (float64 activations, weights dequantized from the Q4_1 GGUF).
Prints the top-5 next tokens (log-probabilities) after the last id. --dump writes each layer's output for the last
position (float32, 49 x 2048) then the logits."""
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
    return dequantize(t.data, t.tensor_type).astype(np.float64)   # shape (rows, cols): rows = outputs


def expert(name, e):
    t = T[name]   # data (n_expert, rows, row bytes)
    return dequantize(t.data[e], t.tensor_type).astype(np.float64)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def rope(x, pos):   # x (n, heads, 128), NEOX halves
    j = np.arange(64)
    ang = pos[:, None] * BASE ** (-2.0 * j / HD)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :64], x[..., 64:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def silu(x):
    return x / (1 + np.exp(-x))


ids = [int(v) for v in sys.argv[1].split(",")]
n = len(ids)
pos = np.arange(n, dtype=np.float64)
emb = T["token_embd.weight"].data
h = np.array([emb[i] for i in ids], dtype=np.float64)
dump = []
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
    o = np.zeros((n, NH, HD))
    for hh in range(NH):
        kh = hh // (NH // NKV)
        sc = q[:, hh] @ k[:, kh].T / np.sqrt(HD)
        for p in range(n):
            lo = max(0, p - (WIN - 1)) if swa[il] else 0
            s = sc[p, lo:p + 1]
            m = max(s.max(), sinks[hh])
            e = np.exp(s - m)
            o[p, hh] = e @ v[lo:p + 1, kh] / (e.sum() + np.exp(sinks[hh] - m))
    h = h + o.reshape(n, NH * HD) @ W(b + "attn_output.weight").T
    x = rms(h, W(b + "ffn_norm.weight"))
    if il < 2:
        f = silu(x @ W(b + "ffn_gate.weight").T) * (x @ W(b + "ffn_up.weight").T)
        h = h + f @ W(b + "ffn_down.weight").T
    else:
        sh = silu(x @ W(b + "ffn_gate_shexp.weight").T) * (x @ W(b + "ffn_up_shexp.weight").T)
        out = sh @ W(b + "ffn_down_shexp.weight").T
        prob = 1 / (1 + np.exp(-(x @ W(b + "ffn_gate_inp.weight").T)))
        for p in range(n):
            top = np.argsort(-prob[p])[:6]
            w = prob[p, top] / prob[p, top].sum()
            for e, we in zip(top, w):
                g = expert(b + "ffn_gate_exps.weight", e) @ x[p]
                u = expert(b + "ffn_up_exps.weight", e) @ x[p]
                out[p] += we * (expert(b + "ffn_down_exps.weight", e) @ (silu(g) * u))
        h = h + out
    dump.append(h[-1].astype(np.float32))
    print(f"layer {il}: |h| {np.linalg.norm(h[-1]):.3f}", file=sys.stderr, flush=True)
x = rms(h[-1], W("output_norm.weight"))
lg = W("output.weight") @ x
lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
top = np.argsort(-lp)[:5]
print(" ".join(f"{t}:{lp[t]:.5f}" for t in top))
if "--dump" in sys.argv:
    with open(sys.argv[sys.argv.index("--dump") + 1], "wb") as f:
        np.stack(dump).tofile(f)
        lg.astype(np.float32).tofile(f)
