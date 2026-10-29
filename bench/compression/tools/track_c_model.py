"""Rung-1/2 modelling primitives (numpy, used for design + cross-entropy evaluation)."""
import numpy as np, os
from track_c_common import *

_gene_cache = {}
def gene_index(org, mem):
    key = (org, mem)
    if key not in _gene_cache:
        ref = 'pbmc_v2_typical' if org == 'human' else 'brain_sn_v3_deep'
        rows = open(os.path.join(SCR, f'{ref}__{mem}.rows.txt')).read().split('\n')
        ids = {}
        g = np.array([ids.setdefault(r.split('_')[0], len(ids)) for r in rows], dtype=np.int64)
        _gene_cache[key] = (g, len(ids))
    return _gene_cache[key]

_data_cache = {}
def get(name):
    if name not in _data_cache:
        d = load(name); d['nz'], d['tot'] = col_stats(d); d['col'] = colids(d['indptr']); _data_cache[name] = d
    return _data_cache[name]

def profile(names, m):
    """mean share of nnz per feature over files"""
    s = np.zeros(m)
    for nm in names:
        d = get(nm); s += np.bincount(d['indices'], minlength=m) / len(d['indices'])
    return s / len(names)

def make_classes(s, g, ng, step=0.25, nlev=96, unseen_lev=12):
    """feature classes from profile s (share of nnz) and gene index g. returns uint8 class per feature and #classes"""
    m = len(s)
    c = np.zeros(m, dtype=np.int32)
    seen = s > 0
    top = np.log2(s[seen].max())
    lev = np.clip(np.round((np.log2(s[seen]) - top) / step * -1), 0, nlev - 1).astype(np.int32)   # 0 = highest
    c[seen] = lev
    # unseen: by gene mean rate over intervals (all intervals of gene)
    gs = np.bincount(g, weights=s, minlength=ng) / np.maximum(np.bincount(g, minlength=ng), 1)
    gf = gs[g]
    un = ~seen
    with np.errstate(divide='ignore'):
        ul = np.where(gf > 0, np.clip(np.round((np.log2(np.where(gf > 0, gf, 1)) - top) / -1.0), 0, unseen_lev - 2), unseen_lev - 1).astype(np.int32)
    c[un] = nlev + ul[un]
    return c, nlev + unseen_lev

def bucketize(d, scheme):
    kind, nb, lo, hi = scheme
    x = d['tot'] if kind == 'tot' else d['nz']
    return log2_bucket(x, nb, lo, hi)

def counts_AN(d, b, c, nb, nc):
    """A[b,c] presence counts, Nb columns per bucket"""
    key = b[d['col']].astype(np.int64) * nc + c[d['indices']]
    A = np.bincount(key, minlength=nb * nc).reshape(nb, nc).astype(np.float64)
    Nb = np.bincount(b, minlength=nb).astype(np.float64)
    return A, Nb

def presence_cost_bits(A, Nb, mc, P):
    """cross entropy (bits) of counts A under presence probs P[b,c]; mc = features per class"""
    absent = Nb[:, None] * mc[None, :] - A
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -(np.where(A > 0, A * np.log2(P), 0) + np.where(absent > 0, absent * np.log2(1 - P), 0))
    return t.sum()

def entropy_bits(counts):
    counts = counts[counts > 0]; p = counts / counts.sum(); return -(counts * np.log2(p)).sum()
