"""make_plan.py OUT --experts N ROUTE... : which experts Flint's hybrid mode keeps on the card (FLINT_PLAN).

ROUTE files come from FLINT_ROUTE_DUMP (records: int16 layer, int16 T, then T x 6 expert ids). Each file's counts are
normalised to the same weight, then the N most used (layer, expert) pairs over all layers are kept. --experts may be
a count, or a size in GB ("5.5GB": 3 x 655,360 bytes an expert). Writes lines "layer expert expert ...".
"""
import argparse
import numpy as np

EXPERTS, EXPERT_BYTES, TOPK = 256, 3 * 655360, 6   # (Flint's; --model ling: Ling-3.0-flash-VL's)


def counts(path):
    a = np.fromfile(path, np.int16)
    c, i = {}, 0
    while i < len(a):
        layer, t = int(a[i]), int(a[i + 1])
        ids = a[i + 2:i + 2 + t * TOPK]
        i += 2 + t * TOPK
        c[layer] = c.get(layer, np.zeros(EXPERTS)) + np.bincount(ids, minlength=EXPERTS)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("routes", nargs="+")
    ap.add_argument("--experts", required=True)
    ap.add_argument("--model", default="flint", choices=["flint", "ling"])
    a = ap.parse_args()
    global EXPERTS, EXPERT_BYTES, TOPK
    if a.model == "ling":   # 512 experts, top 8; Q4_0 gate / up (768 x 2560) and down - ~3.3 MB an expert
        EXPERTS, EXPERT_BYTES, TOPK = 512, 3 * 768 * 2560 * 18 // 32, 8
    total = {}
    for p in a.routes:
        c = counts(p)
        s = sum(v.sum() for v in c.values())
        for layer, v in c.items():
            total[layer] = total.get(layer, np.zeros(EXPERTS)) + v / s
    layers = sorted(total)
    n = int(float(a.experts[:-2]) * 1e9 / EXPERT_BYTES) if a.experts.lower().endswith("gb") else int(a.experts)
    n = max(0, min(n, len(layers) * EXPERTS))
    flat = np.concatenate([total[layer] for layer in layers])
    keep = np.zeros(flat.size, bool)
    keep[np.argsort(-flat, kind="stable")[:n]] = True
    keep = keep.reshape(len(layers), EXPERTS)
    with open(a.out, "w") as f:
        for li, layer in enumerate(layers):
            f.write(" ".join([str(layer)] + [str(e) for e in np.nonzero(keep[li])[0]]) + "\n")
    print(f"{a.out}: {n} experts on the card ({n * EXPERT_BYTES / 1e9:.1f} GB), {len(layers)} layers")


if __name__ == "__main__":
    main()
