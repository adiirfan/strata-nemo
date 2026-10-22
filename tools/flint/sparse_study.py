"""sparse_study.py DIR: from strata-flint's FLINT_DUMP_ATTN files, how far attention that reads only the most promising
blocks of 16 keys (chosen by each block's per-dim K min / max bound, as Quest does) lands from the exact output, at a
few budgets - with 8-bit K's error alongside for scale (that error doubled Flint's KL against the exact model)."""
import sys, glob, re
import numpy as np
D = sys.argv[1]
KV, HD, HQ = 4, 128, 32
layers = sorted({int(re.search(r"L(\d+)_", f).group(1)) for f in glob.glob(D + "/L*_q.f32")})
budgets = [1 / 32, 1 / 16, 1 / 8, 1 / 4]


def attend(s, v, sink):   # s (rows, keys) scores, -inf masked; v (keys, 128); sink (rows,)
    m = np.maximum(s.max(1), sink)
    e = np.exp(s - m[:, None])
    return (e @ v) / (e.sum(1) + np.exp(sink - m))[:, None], e.sum(1) / (e.sum(1) + np.exp(sink - m))


print(f"{'layer':>5} " + " ".join(f"keep {b:>6.3f}" for b in budgets) + "  | 8-bit K   (relative output error, mean over 64 rows x 32 heads)")
for L in layers:
    q = np.fromfile(f"{D}/L{L}_q.f32", np.float32).reshape(64, HQ, HD)
    K = np.fromfile(f"{D}/L{L}_k.f16", np.float16).reshape(-1, KV, HD).astype(np.float32)
    n = K.shape[0]
    ng = (n + 15) // 16
    vq = np.fromfile(f"{D}/L{L}_vq.i8", np.int8).reshape(ng, KV, HD, 16).astype(np.float32)
    vs = np.fromfile(f"{D}/L{L}_vs.f32", np.float32).reshape(ng, KV, 16)
    V = (vq * vs[:, :, None, :]).transpose(0, 3, 1, 2).reshape(ng * 16, KV, HD)[:n]
    sink = np.fromfile(f"{D}/L{L}_sink.f32", np.float32)
    nb = n // 16
    pos = n - 64 + np.arange(64)
    errs = {b: [] for b in budgets}
    e8 = []
    for kh in range(KV):
        Kh = K[:, kh]
        Kb = Kh[: nb * 16].reshape(nb, 16, HD)
        kmin, kmax = Kb.min(1), Kb.max(1)
        a = np.abs(Kh).max(1, keepdims=True) / 127
        K8 = np.rint(Kh / np.where(a > 0, a, 1)) * a
        for hh in range(kh * 8, kh * 8 + 8):
            Q = q[:, hh] / np.sqrt(HD)                                   # (64, 128)
            s = Q @ Kh.T                                                 # (64, n)
            s[np.arange(n)[None, :] > pos[:, None]] = -np.inf
            sk = np.full(64, sink[hh])
            o, _ = attend(s, V[:, kh], sk)
            no = np.linalg.norm(o, axis=1)
            s8 = Q @ K8.T
            s8[np.isinf(s)] = -np.inf
            o8, _ = attend(s8, V[:, kh], sk)
            e8.append(np.linalg.norm(o8 - o, axis=1) / no)
            ub = np.maximum(Q, 0) @ kmax.T + np.minimum(Q, 0) @ kmin.T     # (64, nb) upper bound of each block's best score
            for b in budgets:
                k = max(1, int(nb * b))
                keep = np.zeros((64, nb), bool)
                top = np.argpartition(-ub, k, axis=1)[:, :k]
                np.put_along_axis(keep, top, True, 1)
                keep[:, 0] = True                                        # the first block, and the last 5 (recent keys)
                keep[:, nb - 5:] = True
                km = np.repeat(keep, 16, 1)
                km = np.concatenate([km, np.ones((64, n - nb * 16), bool)], 1)
                ss = np.where(km, s, -np.inf)
                os_, _ = attend(ss, V[:, kh], sk)
                errs[b].append(np.linalg.norm(os_ - o, axis=1) / no)
    print(f"{L:5d} " + " ".join(f"{np.mean(errs[b]):11.5f}" for b in budgets) + f"  | {np.mean(e8):.5f}", flush=True)
