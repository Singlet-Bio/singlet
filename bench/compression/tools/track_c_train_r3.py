#!/usr/bin/env python
"""Rung 3 model: out-of-sample cell-type clusters for the head features.

    python -P tools/track_c_train_r3.py ORG MEM HOLDOUT [--K=16] [--H=2048]

Writes results/track_c_r3_<ORG>_<MEM>_<HOLDOUT>.bin:
    'TCR3' u32 H u32 K u32 T  u32 head[H] (feature ids, by training-pool share)  u8 R[K*T*H]  (code v: p = 2^(-v/16))
Clusters: k-means (K) on a 20-d SVD of log-normalised head counts of the training pool's cells with nnz >= 128 (bucket >= 17);
R[k, t, f] = smoothed presence rate of head feature f in cluster k, depth tier t (tiers: bucket 17-19, 20-22, 23-25, >= 26).
The encoder assigns each column of the test file to the max-likelihood cluster under R (transmitted as side info); the decoder
only reads R. Single-threaded (set OMP_NUM_THREADS=1).
"""
import sys, os, struct, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_train import bucket_fast, RES
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

T = 12
def tier_of(b, T=T):
    """mirror of the decoder's tier_of(): T == 4 -> 3-bucket groups, otherwise one tier per bucket (capped)"""
    if T == 4: return np.where(b <= 19, 0, np.where(b <= 22, 1, np.where(b <= 25, 2, 3)))
    return np.minimum(np.maximum(b - 17, 0), T - 1)

def head_matrix(name, head, cap, rng, min_bucket=17):
    d = get(name); ip = d['indptr']; m = d['m']
    hid = np.full(m, -1); hid[head] = np.arange(len(head))
    b = bucket_fast(d['nz']); cs = np.flatnonzero(b >= min_bucket)
    if len(cs) > cap: cs = np.sort(rng.choice(cs, cap, replace=False))
    rows, cols, vals = [], [], []
    for i, c in enumerate(cs):
        s, e = ip[c], ip[c + 1]; ix = hid[d['indices'][s:e]]; ok = ix >= 0
        rows.append(np.full(ok.sum(), i)); cols.append(ix[ok]); vals.append(d['data'][s:e][ok])
    X = sp.csr_matrix((np.concatenate(vals).astype(np.float64), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cs), len(head)))
    return X, tier_of(b[cs], T)

def main(org, mem, holdout, K=16, H=2048, cap=4000, suffix=''):
    rng = np.random.default_rng(0)
    pool = [k for k in STUDY if ORG[k] == org and (holdout == 'none' or STUDY[k] != holdout)]
    names = [f'{k}__{mem}' for k in pool]; m = get(names[0])['m']
    prof = sum(np.bincount(get(nm)['indices'], minlength=m) / len(get(nm)['indices']) for nm in names)
    head = np.argsort(-prof, kind='stable')[:H].astype(np.uint32)
    Xs, ts = [], []
    for nm in names:
        X, t = head_matrix(nm, head, cap, rng); Xs.append(X); ts.append(t)
    X = sp.vstack(Xs).tocsr(); t = np.concatenate(ts)
    tot = np.asarray(X.sum(1)).ravel(); Xn = sp.diags(1e4 / np.maximum(tot, 1)) @ X; Xn.data = np.log1p(Xn.data)
    Z = TruncatedSVD(20, random_state=0).fit_transform(Xn)
    lab = KMeans(K, n_init=2, random_state=0).fit(Z).labels_ if K > 1 else np.zeros(Z.shape[0], dtype=int)
    P = (X > 0).astype(np.float64).tocsr()
    R = np.zeros((K, T, H))
    for tt in range(T):
        mt = t == tt; base = (np.asarray(P[mt].sum(0)).ravel() + 0.5) / (mt.sum() + 1) if mt.sum() else np.full(H, 0.05)
        for k in range(K):
            sel = mt & (lab == k); n = sel.sum()
            p1 = np.asarray(P[sel].sum(0)).ravel() if n else np.zeros(H)
            R[k, tt] = (p1 + 3.0 * base) / (n + 3.0)
    code = np.clip(np.round(-16 * np.log2(np.clip(R, 2 ** -16, 1))), 0, 255).astype(np.uint8)
    out = os.path.join(RES, f'track_c_r3_{org}_{mem}_{holdout}{suffix}.bin')
    if suffix:   # a variant tag needs its own copy of the main model so the codec resolves <org>_<mem>_<holdout><suffix>
        import shutil; src = os.path.join(RES, f'track_c_model_{org}_{mem}_{holdout}.bin'); dst = os.path.join(RES, f'track_c_model_{org}_{mem}_{holdout}{suffix}.bin')
        if os.path.exists(src) and not os.path.exists(dst): shutil.copyfile(src, dst)
    with open(out, 'wb') as f:
        f.write(b'TCR3'); f.write(struct.pack('<III', H, K, T)); f.write(head.tobytes()); f.write(code.tobytes())
    sizes = np.bincount(lab, minlength=K)
    print(f'wrote {out} ({os.path.getsize(out)} B): pool {pool}, {X.shape[0]} cells, cluster sizes {sizes.tolist()}', flush=True)

if __name__ == '__main__':
    kw = {}
    for a in sys.argv[4:]:
        if a.startswith('--K='): kw['K'] = int(a[4:])
        if a.startswith('--H='): kw['H'] = int(a[4:])
        if a.startswith('--T='): T = int(a[4:])
        if a.startswith('--suffix='): kw['suffix'] = a[9:]
    main(sys.argv[1], sys.argv[2], sys.argv[3], **kw)
