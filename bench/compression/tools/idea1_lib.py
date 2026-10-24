"""Idea 1 analysis library: depth buckets, training counts, smoothing, cross-entropy of the presence/count models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from idea1_data import load, colsums, TEST, TRAIN, ALL

NB = 16
# bucket edges on column total UMI D: bucket b holds EDGES[b] <= D < EDGES[b+1]
EDGES16 = np.array([0, 1, 2, 3, 4, 5, 7, 9, 13, 21, 36, 61, 111, 201, 401, 801, 10**9])

def bucket_of(D, edges=EDGES16):
    return np.searchsorted(edges, D, side='right') - 1

def col_info(d, edges=EDGES16):
    D = colsums(d)
    k = np.diff(d['indptr'])
    b = bucket_of(D, edges)
    return D, k, b

def nnz_bucket(d, b):
    return np.repeat(b, np.diff(d['indptr']))

def counts_table(d, edges=EDGES16, nb=NB):
    """n[f, b] presence counts, N[b] columns per bucket"""
    D, k, b = col_info(d, edges)
    nb_nnz = nnz_bucket(d, b)
    n = np.bincount(d['indices'].astype(np.int64) * nb + nb_nnz, minlength=d['m'] * nb).reshape(d['m'], nb)
    N = np.bincount(b, minlength=nb)
    return n.astype(np.float64), N.astype(np.float64)

def pool(samples, member, edges=EDGES16, weight=None):
    n = None
    for sid in samples:
        d = load(sid, member)
        t, N = counts_table(d, edges)
        w = 1.0 if weight is None else weight(d)
        if n is None: n, NN = t * w, N * w
        else: n += t * w; NN += N * w
    return n, NN

def train_set(test_sid, member, in_sample=False):
    org = ALL[test_sid][1]
    gse = ALL[test_sid][0]
    out = []
    for sid, (g, o) in ALL.items():
        if o != org: continue
        if g == gse and not in_sample: continue
        out.append(sid)
    if in_sample and test_sid not in out: out.append(test_sid)
    return out

def ce_presence(d, p, edges=EDGES16):
    """bits for presence/absence of all (feature, column) under p[f, b]; returns total bits"""
    D, k, b = col_info(d, edges)
    N = np.bincount(b, minlength=p.shape[1]).astype(np.float64)
    A = (-np.log2(1 - p)).sum(axis=0)           # per bucket: bits for all-absent column
    absent = (N * A).sum()
    nb_nnz = nnz_bucket(d, b)
    pp = p[d['indices'], nb_nnz]
    present_extra = (np.log2((1 - pp) / pp)).sum()
    return absent + present_extra

def model_order0(n, N):
    tot = n.sum(1); NN = N.sum()
    return np.repeat(((tot + 0.1) / (NN + 1.0))[:, None], n.shape[1], 1)
