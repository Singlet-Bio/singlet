# Exploratory: how much of test nnz falls on features unseen in training; zero-order entropies
import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_common import *
tests = {'pbmc_v2_typical':'GSE142444','pbmc_v2_shallow':'GSE142444','pbmc_v3_lowgenes':'GSE244263','tcell_5p':'GSE152522','tumor_3p':'GSE145370'}
for mem in ['exon','intron']:
    D = {t: load(f'{t}__{mem}') for t in tests}
    for t in tests:
        d = D[t]; m = d['m']
        kf = np.bincount(d['indices'], minlength=m).astype(np.float64)
        # train = other studies
        tr = np.zeros(m)
        for u in tests:
            if tests[u] != tests[t]:
                du = D[u]; tr += np.bincount(du['indices'], minlength=m) / len(du['indices'])
        unseen = (tr == 0)
        print(f"{mem:6s} {t:18s} nnz={kf.sum():.0f} rows used={np.sum(kf>0)} frac nnz on train-unseen feats={kf[unseen].sum()/kf.sum():.4f} rows(unseen&used)={np.sum(unseen&(kf>0))}")
        # order0 per-feature in-sample entropy (bits/nnz) ignoring depth
        n = d['n']
        p = kf / n
        with np.errstate(divide='ignore', invalid='ignore'):
            h = -(kf*np.log2(p) + (n-kf)*np.log2(1-p)); h = np.nansum(h)
        print(f"        in-sample per-feature presence entropy {h/kf.sum():.3f} bits/nnz (no depth cond.)")
