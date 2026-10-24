"""In-sample presence entropy: 1-D class (k/n) vs 2-D class (k_lo/n_lo, k_hi/n_hi) split by column depth."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp4 import rbin

def ent(A, M):
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(A > 0, A * np.log2(A / M), 0) + np.where(M - A > 0, (M - A) * np.log2((M - A) / M), 0))
    return np.nansum(h)

def run(test, mem, nb=32, step=0.5):
    d = get(f'{test}__{mem}'); m, n = d['m'], d['n']; idx, col = d['indices'], d['col']; nnz = len(idx)
    b = bucketize(d, ('nz', nb, 0, 14))
    k = np.bincount(idx, minlength=m); c1 = rbin(k, n, 0.25)
    A, Nb = counts_AN(d, b, c1, nb, 121); mc = np.bincount(c1, minlength=121).astype(float); M = Nb[:, None] * mc[None]; M[:, 0] = 1
    e1 = ent(A, M)
    out = [f"1D {e1/nnz:.3f}"]
    for split in (10, 40, 150):
        lo = d['nz'] < split
        nlo, nhi = lo.sum(), (~lo).sum()
        klo = np.bincount(idx[lo[col]], minlength=m); khi = k - klo
        cl = rbin(klo, max(nlo, 1), step); ch = rbin(khi, max(nhi, 1), step)
        nl, nh = cl.max() + 1, ch.max() + 1
        c2 = cl * nh + ch
        nc2 = nl * nh
        A2, Nb2 = counts_AN(d, b, c2, nb, nc2); mc2 = np.bincount(c2, minlength=nc2).astype(float); M2 = Nb2[:, None] * mc2[None]; M2[:, mc2 == 0] = 1
        # exclude all-zero pair class
        e2 = ent(A2, M2)
        out.append(f"2D split@{split} (cols lo {nlo}/hi {nhi}, classes used {int((mc2>0).sum())}) {e2/nnz:.3f}")
    print(f"{test:20s} {mem}: " + ' | '.join(out), flush=True)

if __name__ == '__main__':
    for t in sys.argv[2:]: run(t, sys.argv[1])
