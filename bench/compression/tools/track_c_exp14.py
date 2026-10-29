"""Rung 3 (latent structure), out-of-sample hard-cluster version. Train K-means (on SVD of log-normalised head counts) and per-(cluster,tier,feature)
presence rates on cells of OTHER studies; test cells of the held-out file are assigned by max likelihood (side info log2 K bits/cell).
Reports head-feature presence bits/head-nnz with and without clusters."""
import sys, os, numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp13 import cells_matrix

def run(test, mem, H=2000, Ks=(1, 4, 8, 16, 32), ntier=4, cap=3000, minnz=150):
    org = ORG[test]; rng = np.random.default_rng(0)
    pool = [k for k in STUDY if ORG[k] == org and STUDY[k] != STUDY[test]]
    prof = sum(np.bincount(get(f'{k}__{mem}')['indices'], minlength=get(f'{k}__{mem}')['m']) / len(get(f'{k}__{mem}')['indices']) for k in pool)
    head = np.argsort(-prof)[:H]
    Xs, nzs = [], []
    for k in pool:
        X, nz = cells_matrix(f'{k}__{mem}', head, cap, rng, minnz); Xs.append(X); nzs.append(nz)
    XA = sp.vstack(Xs).tocsr(); nzA = np.concatenate(nzs)
    XB, nzB = cells_matrix(f'{test}__{mem}', head, 4000, rng, minnz)
    def feat(X):
        t = np.asarray(X.sum(1)).ravel(); Xn = sp.diags(1e4 / np.maximum(t, 1)) @ X; Xn.data = np.log1p(Xn.data); return Xn
    svd = TruncatedSVD(20, random_state=0).fit(feat(XA)); ZA = svd.transform(feat(XA)); ZB = svd.transform(feat(XB))
    qs = np.quantile(nzA, np.linspace(0, 1, ntier + 1)[1:-1]); tA, tB = np.searchsorted(qs, nzA), np.searchsorted(qs, nzB)
    PA = (XA > 0).astype(np.float32).tocsr(); PB = (XB > 0).astype(np.float32).tocsr().toarray()
    nnzB = PB.sum()
    print(f"{test} {mem}: train cells {XA.shape[0]}, test cells {XB.shape[0]}, test head nnz {int(nnzB)}", flush=True)
    for K in Ks:
        if K == 1: cA = np.zeros(XA.shape[0], int)
        else: cA = KMeans(K, n_init=2, random_state=0).fit(ZA).labels_
        # rates table [K, tier, H] from train (smoothed toward tier base)
        R = np.zeros((K, ntier, H)); 
        for t in range(ntier):
            ma = tA == t; base = (np.asarray(PA[ma].sum(0)).ravel() + 0.5) / (ma.sum() + 1)
            for c in range(K):
                sel = ma & (cA == c); n = sel.sum()
                p1 = np.asarray(PA[sel].sum(0)).ravel() if n else np.zeros(H)
                R[c, t] = np.clip((p1 + 3.0 * base) / (n + 3.0), 1e-5, 1 - 1e-5)
        # assignment of test cells by max likelihood, cost
        L = np.log(R); L1 = np.log(1 - R)
        bits = 0.0
        for i in range(XB.shape[0]):
            t = tB[i]; x = PB[i]
            ll = (L[:, t, :] @ x) + (L1[:, t, :] @ (1 - x))
            bits += -ll.max() / np.log(2)
        side = XB.shape[0] * np.log2(K)
        print(f"   K={K:2d}: head presence {bits/nnzB:.3f} + side {side/nnzB:.3f} = {(bits+side)/nnzB:.3f} bits/head-nnz", flush=True)

if __name__ == '__main__':
    run(sys.argv[2], sys.argv[1])
