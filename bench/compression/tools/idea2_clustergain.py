"""Upper-bound style check of how much cell-type (cluster) information helps presence coding.

Cross-validated (5-fold over cells) log-loss of P(feature f present in cell | depth bucket b, cluster c) on the top-T
features, with hierarchical smoothing  p_fbc = (n1_fbc + a p_fb) / (n_bc + a),  p_fb = (n1_fb + b p_f)/(n_b + b).
Compares K=1 (depth only) with K clusters from k-means on PCA of log-CPM. Reports bits per nnz of the cells.
usage: python -P tools/idea2_clustergain.py sample member [T]
"""
import sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz


def load(sample, member):
    r = read_1pz(f'corpus/{sample}/{member}_counts.1pz')
    m, n = r['m'], r['n']
    ip = np.asarray(r['indptr']).astype(np.int64)
    ix = np.asarray(r['indices']).astype(np.int64)
    d = np.asarray(r['data']).astype(np.float64)
    calls = set(l.split('\t')[0] for l in open(f'corpus/{sample}/cell_calls.tsv').read().splitlines()[1:])
    isc = np.array([c in calls for c in r['colnames']])
    return m, n, ip, ix, d, isc


def kmeans(X, K, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    cen = [X[rng.integers(n)]]
    d2 = ((X - cen[0]) ** 2).sum(1)
    for _ in range(1, K):
        c = X[rng.choice(n, p=d2 / d2.sum())]
        cen.append(c)
        d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
    cen = np.array(cen)
    for _ in range(iters):
        dist = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(2)
        lab = dist.argmin(1)
        for k in range(K):
            if (lab == k).any():
                cen[k] = X[lab == k].mean(0)
    return lab


def main():
    sample, member = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    m, n, ip, ix, d, isc = load(sample, member)
    cells = np.nonzero(isc)[0]
    nc = len(cells)
    z = np.zeros(m, np.int64)
    rows = []
    for j in cells:
        rows.append(ix[ip[j]:ip[j + 1]])
        z[ix[ip[j]:ip[j + 1]]] += 1
    order = np.argsort(-z, kind='stable')
    rank = np.full(m, -1, np.int64)
    rank[order[:T]] = np.arange(T)
    B = np.zeros((nc, T), np.bool_)
    nnz_total = 0
    kk = np.zeros(nc)
    for i, j in enumerate(cells):
        r = rank[ix[ip[j]:ip[j + 1]]]
        r = r[r >= 0]
        B[i, r] = True
        kk[i] = d[ip[j]:ip[j + 1]].sum()
        nnz_total += ip[j + 1] - ip[j]
    in_top = B.sum()
    print(f'{sample} {member}: cells {nc}, top-{T} nnz {in_top} of {nnz_total} ({in_top / nnz_total:.3f})')
    # embedding: log CPM on top 2000, PCA 20
    F = min(2000, T)
    X = np.zeros((nc, F), np.float32)
    for i, j in enumerate(cells):
        r = rank[ix[ip[j]:ip[j + 1]]]
        v = d[ip[j]:ip[j + 1]]
        msk = (r >= 0) & (r < F)
        X[i, r[msk]] = np.log1p(1e4 * v[msk] / kk[i])
    X -= X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = U[:, :20] * S[:20]
    nb = 8
    dq = np.quantile(kk, np.linspace(0, 1, nb + 1)[1:-1])
    bk = np.searchsorted(dq, kk)
    rng = np.random.default_rng(1)
    fold = rng.integers(0, 5, nc)
    for K in (1, 4, 8, 16, 32):
        lab = np.zeros(nc, np.int64) if K == 1 else kmeans(Y, K)
        best = None
        for a in (3, 10, 30):
            for b in (30,):
                loss = 0.0
                for f_ in range(5):
                    tr, te = fold != f_, fold == f_
                    pf = (B[tr].sum(0) + 0.5) / (tr.sum() + 1.0)
                    p_fb = np.zeros((nb, T))
                    for bb in range(nb):
                        s = tr & (bk == bb)
                        p_fb[bb] = (B[s].sum(0) + b * pf) / (s.sum() + b)
                    p_fbc = np.zeros((K, nb, T))
                    for c in range(K):
                        for bb in range(nb):
                            s = tr & (bk == bb) & (lab == c)
                            p_fbc[c, bb] = (B[s].sum(0) + a * p_fb[bb]) / (s.sum() + a)
                    idx = np.nonzero(te)[0]
                    P = p_fbc[lab[idx], bk[idx]]
                    P = np.clip(P, 1e-6, 1 - 1e-6)
                    Bi = B[idx]
                    loss += -(Bi * np.log2(P) + (~Bi) * np.log2(1 - P)).sum()
                if best is None or loss < best[0]:
                    best = (loss, a, b)
        lab_cost = nc * np.log2(K) if K > 1 else 0
        print(f'K={K:3d}: CV presence cost {best[0] / in_top:.4f} bits/top-nnz (a={best[1]}), + label cost {lab_cost / in_top:.4f} -> {(best[0] + lab_cost) / in_top:.4f}')


if __name__ == '__main__':
    main()
