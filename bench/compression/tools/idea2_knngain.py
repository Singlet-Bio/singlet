"""kNN upper-bound check: leave-one-out presence log-loss of cell j predicted from its k nearest neighbours (PCA space,
cosine) mixed with the (feature, depth-bucket) estimate. usage: python -u -P tools/idea2_knngain.py sample member [T]"""
import sys, numpy as np
sys.path.insert(0, '../../python')
sys.path.insert(0, 'tools')
from idea2_clustergain import load


def main():
    sample, member = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    m, n, ip, ix, d, isc = load(sample, member)
    cells = np.nonzero(isc)[0]; nc = len(cells)
    z = np.zeros(m, np.int64)
    for j in cells: z[ix[ip[j]:ip[j + 1]]] += 1
    order = np.argsort(-z, kind='stable'); rank = np.full(m, -1, np.int64); rank[order[:T]] = np.arange(T)
    B = np.zeros((nc, T), np.float32); kk = np.zeros(nc); nnz_total = 0
    for i, j in enumerate(cells):
        r = rank[ix[ip[j]:ip[j + 1]]]; r = r[r >= 0]; B[i, r] = 1; kk[i] = d[ip[j]:ip[j + 1]].sum(); nnz_total += ip[j + 1] - ip[j]
    in_top = B.sum()
    F = min(2000, T); X = np.zeros((nc, F), np.float32)
    for i, j in enumerate(cells):
        r = rank[ix[ip[j]:ip[j + 1]]]; v = d[ip[j]:ip[j + 1]]; msk = (r >= 0) & (r < F)
        X[i, r[msk]] = np.log1p(1e4 * v[msk] / kk[i])
    X -= X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = U[:, :20] * S[:20]; Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    nb = 8; dq = np.quantile(kk, np.linspace(0, 1, nb + 1)[1:-1]); bk = np.searchsorted(dq, kk)
    # leave-one-out (feature, depth-bucket) baseline
    cnt_fb = np.zeros((nb, T)); nb_ = np.zeros(nb)
    for bb in range(nb):
        s = bk == bb; cnt_fb[bb] = B[s].sum(0); nb_[bb] = s.sum()
    pf = (B.sum(0) + .5) / (nc + 1)
    def baseline(i):
        bb = bk[i]
        return (cnt_fb[bb] - B[i] + 30 * pf) / (nb_[bb] - 1 + 30)
    sim = Y @ Y.T
    np.fill_diagonal(sim, -9)
    res = {}
    base_loss = 0.0
    nbrs = {k: np.argsort(-sim, axis=1)[:, :k] for k in (30, 100)}
    losses = {(k, a): 0.0 for k in nbrs for a in (10, 30, 100)}
    for i in range(nc):
        pb = baseline(i)
        P = np.clip(pb, 1e-6, 1 - 1e-6)
        base_loss += -(B[i] * np.log2(P) + (1 - B[i]) * np.log2(1 - P)).sum()
        for k, nn in nbrs.items():
            # neighbours' presence relative to their own expected baseline is better, but keep it simple: raw average
            s = B[nn[i]].sum(0)
            for a in (10, 30, 100):
                Pn = np.clip((s + a * pb) / (k + a), 1e-6, 1 - 1e-6)
                losses[(k, a)] += -(B[i] * np.log2(Pn) + (1 - B[i]) * np.log2(1 - Pn)).sum()
    print(f'{sample} {member}: top-{T} nnz {int(in_top)}  LOO (feature x depth) baseline {base_loss / in_top:.4f} bits/top-nnz')
    for (k, a), v in losses.items():
        print(f'  kNN k={k:3d} a={a:3d}: {v / in_top:.4f}  (delta {(v - base_loss) / in_top:+.4f})')

if __name__ == '__main__':
    main()
