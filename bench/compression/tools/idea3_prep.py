"""Idea 3: one-time per-sample cache of sufficient statistics (interval and gene level), used by the measurement
scripts for leave-one-study-out out-of-sample estimates.
Per sample and per matrix kind (exon/intron) store:
  T_i      (m,)   total UMIs per interval
  Tg       (G,)   total UMIs per gene
  kI, kG   (n,)   nnz per cell at interval / gene level
  presI    (B,m)  #cells in bucket b (by kI) with interval i present     (sparse)
  presG    (B,G)  same at gene level (bucket by kG)
  NI, NG   (B,)   #cells per bucket
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea3_common import *
import scipy.sparse as sp

B = 24
def bucket(k):
    k = np.asarray(k, np.int64)
    b = np.where(k <= 4, k - 1, 4 + np.floor(2 * np.log2(np.maximum(k, 1) / 4.0) + 1e-9).astype(np.int64))
    b = np.minimum(b, B - 1)
    return np.where(k <= 0, -1, b)

def entries(M):
    """col, row, x arrays of a CSC matrix (sorted by col then row)."""
    col = np.repeat(np.arange(M.shape[1]), np.diff(M.indptr))
    return col, M.indices.astype(np.int64), M.data.astype(np.int64)

def gene_groups(col, row, x, gene_of_row):
    g = gene_of_row[row].astype(np.int64)
    new = np.ones(len(col), bool)
    new[1:] = (col[1:] != col[:-1]) | (g[1:] != g[:-1])
    st = np.flatnonzero(new)
    gc = np.add.reduceat(x, st)
    return col[st], g[st], gc, st

def stats_for(M, gene_of_row, G):
    m, n = M.shape
    col, row, x = entries(M)
    gcol, gg, gc, st = gene_groups(col, row, x, gene_of_row)
    kI = np.diff(M.indptr)
    kG = np.bincount(gcol, minlength=n)
    bI, bG = bucket(kI), bucket(kG)
    T_i = np.bincount(row, weights=x, minlength=m).astype(np.int64)
    Tg = np.bincount(gg, weights=gc, minlength=G).astype(np.int64)
    NI = np.bincount(bI[bI >= 0], minlength=B); NG = np.bincount(bG[bG >= 0], minlength=B)
    # presence counts
    ok = bI[col] >= 0
    presI = np.bincount(bI[col][ok] * m + row[ok], minlength=B * m).reshape(B, m).astype(np.int32)
    ok = bG[gcol] >= 0
    presG = np.bincount(bG[gcol][ok] * G + gg[ok], minlength=B * G).reshape(B, G).astype(np.int32)
    return dict(T_i=T_i, Tg=Tg, kI=kI, kG=kG, presI=sp.csr_matrix(presI), presG=sp.csr_matrix(presG), NI=NI, NG=NG)

def cache_path(sid, kind):
    return os.path.join(CACHE, f'{sid}_{kind}.npz')

def save(sid, kind, d):
    flat = {}
    for k, v in d.items():
        if sp.issparse(v):
            sp.save_npz(os.path.join(CACHE, f'{sid}_{kind}_{k}.npz'), v)
        else:
            flat[k] = v
    np.savez(cache_path(sid, kind), **flat)

def load(sid, kind):
    f = np.load(cache_path(sid, kind))
    d = {k: f[k] for k in f.files}
    for k in ('presI', 'presG'):
        d[k] = sp.load_npz(os.path.join(CACHE, f'{sid}_{kind}_{k}.npz'))
    return d

if __name__ == '__main__':
    sids = sys.argv[1:] or TEST + TRAIN_H + TRAIN_M
    for sid in sids:
        if all(os.path.exists(cache_path(sid, k)) for k in ('exon', 'intron')):
            continue
        t = time.time()
        S = Sample(sid)
        Ge, Gi = len(S.gids_e), len(S.gids_i)
        save(sid, 'exon', stats_for(S.E, S.egene, Ge))
        save(sid, 'intron', stats_for(S.I, S.igene, Gi))
        print(sid, 'cached', round(time.time() - t, 1), 's', flush=True)
