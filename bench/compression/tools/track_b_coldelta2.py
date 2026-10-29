# Column-to-column delta test (PLAN Track B, column order), efficient rewrite of track_b_coldelta.py (np.add.at replaced by
# bincount, int32 buffers, optional subsampling of called cells).
# For called cells (k >= kmin) ordered by (a) stored order, (b) depth, (c) greedy nearest-neighbour chain in PCA space,
# (d) random: is the presence bit of head row r in column j cheaper given the same bit in column j-1?
# Static in-sample conditional entropies, bits per head nnz, for: bucket model (rank bucket x depth bucket), the same plus the
# previous column's bit, and the per-rank oracles (rank x depth) with / without the previous bit.
#   python -P tools/track_b_coldelta2.py file.1pz [max_cells]
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from singlet._pz import read_1pz


def H(cnt):
    cnt = cnt.astype(np.float64)
    tot = cnt.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        l = np.where(cnt > 0, cnt * np.log2(cnt / np.maximum(tot, 1)), 0.0)
    return -l.sum()


def table(ctx, x, nctx):
    return np.bincount(ctx.astype(np.int64) * 2 + x, minlength=nctx * 2).reshape(nctx, 2)


def run(path, R=4096, kmin=200, max_cells=6000):
    r = read_1pz(path)
    ip = np.asarray(r['indptr'], dtype=np.int64); ix = np.asarray(r['indices']).astype(np.int64); d = np.asarray(r['data']).astype(np.float32)
    n, m = r['n'], r['m']
    cc = np.diff(ip); col = np.repeat(np.arange(n), cc)
    rn = np.bincount(ix, minlength=m)
    order = np.lexsort((np.arange(m), -rn)); rank = np.empty(m, np.int64); rank[order] = np.arange(m)
    rk = rank[ix]
    cells = np.nonzero(cc >= kmin)[0]
    rng = np.random.default_rng(0)
    if len(cells) > max_cells:
        cells = np.sort(rng.choice(cells, max_cells, replace=False))
    cidx = np.full(n, -1); cidx[cells] = np.arange(len(cells))
    selc = cidx[col] >= 0
    sel = (rk < R) & selc
    X = np.zeros((len(cells), R), np.uint8); X[cidx[col[sel]], rk[sel]] = 1
    kb = np.minimum(np.floor(np.log2(cc[cells] + 1)).astype(np.int64), 11)
    nh = int(X.sum())
    print(f"== {os.path.basename(os.path.dirname(path))}: {len(cells)} cells with k>={kmin}, head nnz {nh} ({nh / r['nnz']:.2f} of all nnz)", flush=True)
    F = np.zeros((len(cells), 1000), np.float32)
    s2 = (rk < 1000) & selc
    F[cidx[col[s2]], rk[s2]] = np.log1p(d[s2])
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    F0 = F - F.mean(0)
    U, S, Vt = np.linalg.svd(F0[:min(len(F0), 8000)], full_matrices=False)
    P = F0 @ Vt[:32].T
    left = np.ones(len(cells), bool)
    cur = int(np.argmax(cc[cells])); chain = [cur]; left[cur] = False
    P2 = (P ** 2).sum(1)
    for _ in range(len(cells) - 1):
        dd = P2 - 2 * P @ P[cur] + P2[cur]
        dd[~left] = np.inf
        cur = int(np.argmin(dd)); chain.append(cur); left[cur] = False
    chain = np.array(chain)
    rb = np.minimum(np.floor(2 * np.log2(np.arange(R) + 2)).astype(np.int64), 39)

    def cost(orderX, name):
        Xo = X[orderX]
        kbo = kb[orderX]
        prev = np.zeros_like(Xo); prev[1:] = Xo[:-1]
        x = Xo.ravel().astype(np.int64)
        pv = prev.ravel().astype(np.int64)
        c_ind = (rb[None, :] * 12 + kbo[:, None]).ravel()
        c_or = (np.arange(R)[None, :] * 12 + kbo[:, None]).ravel()
        t1 = H(table(c_ind, x, 40 * 12)); t2 = H(table(c_ind * 2 + pv, x, 40 * 12 * 2))
        t3 = H(table(c_or, x, R * 12)); t4 = H(table(c_or * 2 + pv, x, R * 12 * 2))
        print(f"  [{name:20s}] bucket model {t1 / nh:.3f} | + prev-column bit {t2 / nh:.3f} | per-rank oracle {t3 / nh:.3f} | + prev bit {t4 / nh:.3f}  bits/head nnz", flush=True)

    cost(np.arange(len(cells)), "stored order")
    cost(np.argsort(-cc[cells], kind='stable'), "by depth")
    cost(chain, "NN chain (PCA-32)")
    cost(rng.permutation(len(cells)), "random")


if __name__ == '__main__':
    run(sys.argv[1], max_cells=int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
