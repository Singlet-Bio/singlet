"""Latent-cluster potential: split-half within file. Head features presence cost with/without cluster conditioning."""
import sys, os, numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *

def run(test, mem, H=2000, minnz=150, Ks=(1, 4, 8, 16, 32), ntier=4):
    d = get(f'{test}__{mem}'); m = d['m']; ip = d['indptr']
    k = np.bincount(d['indices'], minlength=m); head = np.argsort(-k)[:H]
    hid = np.full(m, -1); hid[head] = np.arange(H)
    cells = np.flatnonzero(d['nz'] >= minnz)
    rng = np.random.default_rng(0); perm = rng.permutation(len(cells)); half = len(cells) // 2
    A, B = cells[perm[:half]], cells[perm[half:]]
    def mat(cs):
        rows, cols, vals = [], [], []
        for i, c in enumerate(cs):
            s, e = ip[c], ip[c + 1]; ix = hid[d['indices'][s:e]]; ok = ix >= 0
            rows.append(np.full(ok.sum(), i)); cols.append(ix[ok]); vals.append(d['data'][s:e][ok])
        return sp.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cs), H))
    XA, XB = mat(A), mat(B)
    def feat(X):
        # log-normalised counts
        t = np.asarray(X.sum(1)).ravel(); Xn = sp.diags(1e4 / np.maximum(t, 1)) @ X
        Xn.data = np.log1p(Xn.data); return Xn
    svd = TruncatedSVD(20, random_state=0).fit(feat(XA)); ZA = svd.transform(feat(XA)); ZB = svd.transform(feat(XB))
    # tiers by nnz
    nzA, nzB = d['nz'][A], d['nz'][B]
    qs = np.quantile(nzA, np.linspace(0, 1, ntier + 1)[1:-1])
    tA, tB = np.searchsorted(qs, nzA), np.searchsorted(qs, nzB)
    PA = (XA > 0).astype(np.float32).tocsr(); PB = (XB > 0).astype(np.float32).tocsr()
    res = []
    for K in Ks:
        if K == 1: cA = np.zeros(len(A), int); cB = np.zeros(len(B), int)
        else:
            km = KMeans(K, n_init=2, random_state=0).fit(ZA); cA = km.labels_; cB = km.predict(ZB)
        # base p per (tier, feature) from A; cluster p smoothed toward base
        bits = 0.0
        for t in range(ntier):
            ma = tA == t; base = (np.asarray(PA[ma].sum(0)).ravel() + 0.5) / (ma.sum() + 1)
            for c in range(K):
                sel = ma & (cA == c); n = sel.sum()
                p1 = np.asarray(PA[sel].sum(0)).ravel() if n else np.zeros(H)
                p = (p1 + 3.0 * base) / (n + 3.0)
                tb = (tB == t) & (cB == c)
                if tb.sum() == 0: continue
                Pb = PB[tb]; n1 = np.asarray(Pb.sum(0)).ravel(); nb_ = tb.sum()
                bits += -(n1 * np.log2(p) + (nb_ - n1) * np.log2(1 - p)).sum()
        nnzB = PB.sum()
        side = len(B) * np.log2(K)
        res.append((K, bits / nnzB, side / nnzB))
    print(f"{test} {mem} cells(nnz>={minnz}) {len(cells)}, head nnz share in B {PB.sum()/d['nz'][B].sum():.2f}")
    for K, b_, s_ in res: print(f"   K={K:3d}  head presence {b_:.3f} bits/head-nnz + side {s_:.3f} = {b_+s_:.3f}")

if __name__ == '__main__':
    for t in sys.argv[2:]: run(t, sys.argv[1])
