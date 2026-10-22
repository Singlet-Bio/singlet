# Does cell-type structure (cluster id per column as side info) cut the head-row Bernoulli cost? Held-out by column parity.
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from singlet._pz import read_1pz

def run(path, R=4096, Ks=(1, 4, 8, 16, 32), kmin=30):
    r = read_1pz(path)
    ip = np.asarray(r['indptr'], dtype=np.int64); ix = np.asarray(r['indices']).astype(np.int64); d = np.asarray(r['data']).astype(np.float32)
    n, m = r['n'], r['m']
    cc = np.diff(ip); col = np.repeat(np.arange(n), cc)
    rn = np.bincount(ix, minlength=m)
    order = np.lexsort((np.arange(m), -rn)); rank = np.empty(m, np.int64); rank[order] = np.arange(m)
    rk = rank[ix]
    sel = rk < R
    X = np.zeros((n, R), np.uint8); X[col[sel], rk[sel]] = 1
    kb = np.minimum(np.floor(np.log2(cc + 1)).astype(np.int64), 11)
    cells = cc >= kmin
    print(os.path.basename(os.path.dirname(path)), "cells(k>=%d)=%d of %d; head nnz frac %.2f" % (kmin, cells.sum(), n, X.sum() / len(ix)))
    # feature matrix for clustering: top 1000 rows, log1p of counts normalized per column
    F = np.zeros((n, 1000), np.float32)
    s2 = rk < 1000
    F[col[s2], rk[s2]] = np.log1p(d[s2])
    F -= F.mean(0, keepdims=True) * 0
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    # PCA to 32 dims on cell columns
    Fc = F[cells]
    Fc0 = Fc - Fc.mean(0)
    U, S, Vt = np.linalg.svd(Fc0[:min(len(Fc0), 20000)], full_matrices=False)
    P = (F - Fc.mean(0)) @ Vt[:32].T
    rng = np.random.default_rng(1)
    tr = (np.arange(n) % 2 == 0); te = ~tr
    base = None
    for K in Ks:
        lab = np.full(n, K, np.int64)  # K = non-cell class
        if K > 1:
            Pc = P[cells]
            C = Pc[rng.choice(len(Pc), K, replace=False)]
            for it in range(15):
                dist = ((Pc[:, None, :] - C[None]) ** 2).sum(-1) if len(Pc) * K < 4e6 else None
                if dist is None:
                    dist = (Pc ** 2).sum(1)[:, None] - 2 * Pc @ C.T + (C ** 2).sum(1)[None]
                a = dist.argmin(1)
                for c in range(K):
                    if (a == c).any(): C[c] = Pc[a == c].mean(0)
            lab[cells] = a
        else:
            lab[:] = 0
        total = 0.0
        for c in np.unique(lab):
            for b in range(12):
                trm = tr & (lab == c) & (kb == b); tem = te & (lab == c) & (kb == b)
                if tem.sum() == 0: continue
                cnt = X[trm].sum(0).astype(np.float64) if trm.sum() else np.zeros(R)
                p = (cnt + 0.05) / (trm.sum() + 0.1)
                x = X[tem].astype(np.float64)
                total += -(x * np.log2(p) + (1 - x) * np.log2(1 - p)).sum()
        side = (te & cells).sum() * np.log2(max(K, 1)) if K > 1 else 0
        nh = X[te].sum()
        print(f"  K={K:2d}: head cost {total / nh:.3f} bits/head-nnz (+ side info {side / nh:.3f}) -> {(total + side) / nh:.3f}")

for p in sys.argv[1:]:
    run(p)
