"""Value entropy with column-level side info (total-UMI / nnz ratio bucket)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp4 import rbin
from track_c_exp6 import cond_ent
def run(test, mem, nb=32):
    d = get(f'{test}__{mem}'); m, n = d['m'], d['n']; nnz = len(d['indices'])
    idx, col, v = d['indices'], d['col'], d['data'].astype(np.int64)
    b = bucketize(d, ('nz', nb, 0, 14)); k = np.bincount(idx, minlength=m); ce = rbin(k, n)
    vs = np.minimum(v, 9) - 1
    base_ctx = (ce[idx] // 4) * 9 + b[col] // 4
    e0 = cond_ent(base_ctx, vs, 9)
    ratio = d['tot'] / np.maximum(d['nz'], 1)
    out = [f"{test} {mem}: base {e0/nnz:.3f}"]
    for nr in (4, 8, 16):
        q = np.clip(np.floor(np.log2(np.maximum(ratio, 1.0)) * nr / 2.5), 0, nr - 1).astype(np.int64)   # ratio 1..5.6
        e1 = cond_ent(base_ctx * nr + q[col], vs, 9)
        # side info: entropy of q given b
        key = b.astype(np.int64) * nr + q; Ck = np.bincount(key, minlength=33 * nr).reshape(33, nr).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            H = -np.nansum(np.where(Ck > 0, Ck * np.log2(Ck / Ck.sum(1, keepdims=True)), 0))
        out.append(f"nr={nr}: val {e1/nnz:.3f} + side {H/nnz:.3f} = {(e1+H)/nnz:.3f}")
    print(' | '.join(out), flush=True)
for t in sys.argv[2:]: run(t, sys.argv[1])
