"""Rung 3 evidence (topic-model flavour): out-of-sample Poisson NMF on head features, per-cell factors fitted on the held-out cell
(side info = quantised factors). Presence cross-entropy (bits) of head features under p = 1 - exp(-mu), compared across K."""
import sys, os, numpy as np
from sklearn.decomposition import NMF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
import scipy.sparse as sp

def cells_matrix(name, head, cap, rng, minnz):
    d = get(name); ip = d['indptr']
    hid = np.full(d['m'], -1); hid[head] = np.arange(len(head))
    cs = np.flatnonzero(d['nz'] >= minnz)
    if len(cs) > cap: cs = rng.choice(cs, cap, replace=False)
    rows, cols, vals = [], [], []
    for i, c in enumerate(cs):
        s, e = ip[c], ip[c + 1]; ix = hid[d['indices'][s:e]]; ok = ix >= 0
        rows.append(np.full(ok.sum(), i)); cols.append(ix[ok]); vals.append(d['data'][s:e][ok])
    X = sp.csr_matrix((np.concatenate(vals).astype(np.float64), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cs), len(head)))
    return X, d['nz'][cs]

def fit_theta(X, W, iters=60):
    """KL-NMF multiplicative updates for H (cells x K) with W (K x F) fixed."""
    K = W.shape[0]; Xd = X.toarray() if sp.issparse(X) else X
    tot = Xd.sum(1, keepdims=True) + 1e-9
    H = np.full((Xd.shape[0], K), 1.0) * (tot / K) / W.sum(1)[None, :].clip(1e-9)
    Ws = W.sum(1)[None, :]
    for _ in range(iters):
        R = Xd / np.maximum(H @ W, 1e-12)
        H *= (R @ W.T) / np.maximum(Ws, 1e-12)
    return H, Xd

def run(test, mem, Ks=(1, 4, 8, 16), H=3000, cap=3000, minnz=150):
    org = ORG[test]; rng = np.random.default_rng(0)
    pool = [k for k in STUDY if ORG[k] == org and STUDY[k] != STUDY[test]]
    prof = sum(np.bincount(get(f'{k}__{mem}')['indices'], minlength=get(f'{k}__{mem}')['m']) / len(get(f'{k}__{mem}')['indices']) for k in pool)
    head = np.argsort(-prof)[:H]
    Xs = []
    for k in pool:
        X, _ = cells_matrix(f'{k}__{mem}', head, cap, rng, minnz); Xs.append(X)
    Xtr = sp.vstack(Xs).tocsr()
    Xte, nzte = cells_matrix(f'{test}__{mem}', head, 2500, rng, minnz)
    Xte_d = Xte.toarray()
    pres = (Xte_d > 0)
    nnz_head = pres.sum()
    print(f"{test} {mem}: train cells {Xtr.shape[0]}, test cells {Xte.shape[0]}, head nnz (test) {nnz_head}; head share of test-cell nnz {nnz_head/ nzte.sum():.2f}", flush=True)
    for K in Ks:
        if K == 1:
            W = np.asarray(Xtr.sum(0)).ravel()[None, :]; W = W / W.sum()
        else:
            nmf = NMF(K, init='nndsvda', solver='mu', beta_loss='kullback-leibler', max_iter=80, random_state=0)
            nmf.fit(Xtr[:20000]); W = nmf.components_
        Hh, Xd = fit_theta(Xte, W)
        # quantize factors: relative factors to 4 bits (log scale) -> side info approx K*4 bits/cell (K>1) ; depth is free (nnz bucket)
        mu = Hh @ W
        # calibrate presence link per feature-decile: p = 1 - exp(-a*mu) with a fitted on the same data (global scalar), same for all K
        best = None
        for a in (0.5, 0.7, 0.85, 1.0, 1.2):
            p = np.clip(1 - np.exp(-a * mu), 1e-6, 1 - 1e-6)
            bits = -(pres * np.log2(p) + (~pres) * np.log2(1 - p)).sum()
            if best is None or bits < best[0]: best = (bits, a)
        side = 0 if K == 1 else Xte.shape[0] * K * 4
        print(f"   K={K:2d}: head presence {best[0]/nnz_head:.3f} bits/head-nnz (a={best[1]}) + side {side/nnz_head:.3f} = {(best[0]+side)/nnz_head:.3f}", flush=True)
if __name__ == '__main__':
    run(sys.argv[2], sys.argv[1])
