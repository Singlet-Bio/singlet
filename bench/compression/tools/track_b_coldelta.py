# Column-to-column delta coding test (PLAN Track B, column order): for called cells ordered by a nearest-neighbour
# chain in PCA space, is the presence bit of head row r in column j cheaper given the same bit in column j-1?
# Static in-sample conditional entropies on the top-R rows (bits per head nnz), compared with the independent
# Bernoulli(rank bucket x depth bucket) model and with the oracle (per rank, per depth) model.
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from singlet._pz import read_1pz


def H(cnt):
    """entropy (bits) of a (ctx, 2) count table"""
    cnt = cnt.astype(np.float64)
    tot = cnt.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        l = np.where(cnt > 0, cnt * np.log2(cnt / np.maximum(tot, 1)), 0.0)
    return -l.sum()


def run(path, R=4096, kmin=200):
    r = read_1pz(path)
    ip = np.asarray(r['indptr'], dtype=np.int64); ix = np.asarray(r['indices']).astype(np.int64); d = np.asarray(r['data']).astype(np.float32)
    n, m = r['n'], r['m']
    cc = np.diff(ip); col = np.repeat(np.arange(n), cc)
    rn = np.bincount(ix, minlength=m)
    order = np.lexsort((np.arange(m), -rn)); rank = np.empty(m, np.int64); rank[order] = np.arange(m)
    rk = rank[ix]
    cells = np.nonzero(cc >= kmin)[0]
    sel = (rk < R) & (cc[col] >= kmin)
    cidx = np.full(n, -1); cidx[cells] = np.arange(len(cells))
    X = np.zeros((len(cells), R), np.uint8); X[cidx[col[sel]], rk[sel]] = 1
    kb = np.minimum(np.floor(np.log2(cc[cells] + 1)).astype(np.int64), 11)
    nh = X.sum()
    print(f"== {os.path.basename(os.path.dirname(path))}: {len(cells)} cells with k>={kmin}, head nnz {nh} ({nh / r['nnz']:.2f} of all nnz)")
    # PCA on log1p of top-1000 ranks
    F = np.zeros((len(cells), 1000), np.float32)
    s2 = (rk < 1000) & (cc[col] >= kmin)
    F[cidx[col[s2]], rk[s2]] = np.log1p(d[s2])
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    F0 = F - F.mean(0)
    U, S, Vt = np.linalg.svd(F0[:min(len(F0), 15000)], full_matrices=False)
    P = F0 @ Vt[:32].T
    # greedy nearest-neighbour chain (start from the deepest cell)
    rng = np.random.default_rng(0)
    left = np.ones(len(cells), bool)
    cur = int(np.argmax(cc[cells])); chain = [cur]; left[cur] = False
    P2 = (P ** 2).sum(1)
    for _ in range(len(cells) - 1):
        dd = P2 - 2 * P @ P[cur] + P2[cur]
        dd[~left] = np.inf
        cur = int(np.argmin(dd)); chain.append(cur); left[cur] = False
    chain = np.array(chain)
    rb = np.minimum(np.floor(2 * np.log2(np.arange(R) + 2)).astype(np.int64), 39)  # rank bucket per head row
    def cost(orderX, name):
        Xo = X[orderX]; kbo = kb[orderX]
        prev = np.zeros_like(Xo); prev[1:] = Xo[:-1]
        # contexts: (rank bucket, depth bucket, prev bit) vs (rank bucket, depth bucket)
        rbm = np.broadcast_to(rb[None, :], Xo.shape)
        kbm = np.broadcast_to(kbo[:, None], Xo.shape)
        c_ind = (rbm * 12 + kbm).ravel()
        c_del = (c_ind * 2 + prev.ravel())
        x = Xo.ravel().astype(np.int64)
        t_ind = np.zeros((40 * 12, 2)); np.add.at(t_ind, (c_ind, x), 1)
        t_del = np.zeros((40 * 12 * 2, 2)); np.add.at(t_del, (c_del, x), 1)
        # per-rank oracle (rank x depth), the independent-feature ceiling
        c_or = (np.broadcast_to(np.arange(R)[None, :], Xo.shape) * 12 + kbm).ravel()
        t_or = np.zeros((R * 12, 2)); np.add.at(t_or, (c_or, x), 1)
        c_ord = c_or * 2 + prev.ravel()
        t_ord = np.zeros((R * 12 * 2, 2)); np.add.at(t_ord, (c_ord, x), 1)
        print(f"  [{name:22s}] bucket model {H(t_ind) / nh:.3f} | bucket + prev-column bit {H(t_del) / nh:.3f} | per-rank model {H(t_or) / nh:.3f} | per-rank + prev bit {H(t_ord) / nh:.3f}  bits/head nnz")
    cost(np.arange(len(cells)), "stored order")
    cost(np.argsort(-cc[cells], kind='stable'), "by depth")
    cost(chain, "NN chain (PCA-32)")
    cost(rng.permutation(len(cells)), "random")


for p in sys.argv[1:]:
    run(p)
