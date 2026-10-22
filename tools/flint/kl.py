"""kl.py BASE OTHER: mean / max KL(base || other) per position, top-1 agreement, of two --logits files."""
import sys, numpy as np
V = 150144
a = np.fromfile(sys.argv[1], np.float32).reshape(-1, V).astype(np.float64)
b = np.fromfile(sys.argv[2], np.float32).reshape(-1, V).astype(np.float64)
def ls(x): m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
la, lb = ls(a), ls(b)
kl = (np.exp(la) * (la - lb)).sum(1)
print(f"KL mean {kl.mean():.5f} median {np.median(kl):.5f} max {kl.max():.4f}; top-1 same {np.mean(a.argmax(1) == b.argmax(1)):.3f} over {len(kl)} positions")
