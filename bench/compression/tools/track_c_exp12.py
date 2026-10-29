"""Presence entropy with a second column covariate z = total UMI / nnz (side info), in-sample class tables."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp4 import rbin
from track_c_exp5 import tab_cost
def run(test, mem, nb=32):
    d = get(f'{test}__{mem}'); m, n = d['m'], d['n']; idx, col = d['indices'], d['col']; nnz = len(idx)
    b = bucketize(d, ('nz', nb, 0, 14)); k = np.bincount(idx, minlength=m); ce = rbin(k, n)
    nc = 121
    def ent(bb, nbb):
        key = bb[col].astype(np.int64) * nc + ce[idx]
        A = np.bincount(key, minlength=nbb * nc).reshape(nbb, nc).astype(float)
        Nb = np.bincount(bb, minlength=nbb).astype(float); mc = np.bincount(ce, minlength=nc).astype(float)
        M = Nb[:, None] * mc[None]; M[:, 0] = 1
        with np.errstate(divide='ignore', invalid='ignore'):
            h = -(np.where(A > 0, A * np.log2(A / M), 0) + np.where(M - A > 0, (M - A) * np.log2((M - A) / M), 0))
        return np.nansum(h), Nb
    e0, Nb0 = ent(b, nb + 1)
    ratio = d['tot'] / np.maximum(d['nz'], 1)
    out = [f"{test} {mem}: b only {e0/nnz:.3f}"]
    for nz_ in (3, 6):
        q = np.clip(np.floor(np.log2(np.maximum(ratio, 1.0)) * nz_ / 2.5), 0, nz_ - 1).astype(np.int64)
        bb = (b.astype(np.int64) * nz_ + q)
        e1, Nb1 = ent(bb.astype(np.int64), (nb + 1) * nz_)
        Ck = Nb1[Nb1 > 0]; Hq = -(Ck * np.log2(Ck / Ck.sum())).sum() - (-(Nb0[Nb0 > 0] * np.log2(Nb0[Nb0 > 0] / Nb0.sum())).sum())
        out.append(f"b x z{nz_}: {e1/nnz:.3f} (+{Hq/nnz:.3f} side, table cells {(nb+1)*nz_*nc}) = {(e1+Hq)/nnz:.3f}")
    print(' | '.join(out), flush=True)
for t in sys.argv[2:]: run(t, sys.argv[1])
