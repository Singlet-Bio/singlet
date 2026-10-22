"""Does the training mix matter?  Out-of-sample presence CE (bits/nnz) of cloglog models trained on ONE other-study sample at a time,
vs the pooled model (all other studies), on each human exon test file.  best-of-K = min over the K single-sample models (cost log2 K bits/file).
   python -P tools/idea1_select.py OUT.tsv"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idea1_train as T
from idea1_fit import fit_cloglog
from idea1_data import load, colsums, ALL, genes, CACHE

member = 'exon'
hs = [s for s, (g, o) in ALL.items() if o == 'human']
g = genes('human', member)
fits = {}
def fit_one(sid):
    cp = os.path.join(CACHE, f'fit1_{sid}_{member}.npz')
    if os.path.exists(cp):
        z = np.load(cp); return z['a'], z['c']
    n, N = T.counts_cached(sid, member)
    tot = n.sum(1)
    gm = T.gene_prior(tot, g)
    eps = np.where(tot > 0, 0.0, 0.2 * gm / (1 + gm / 50.0))
    a, c = fit_cloglog(n, N, eps=eps, iters=8)
    np.savez(cp, a=a, c=c)
    return a, c

def pres_bits(sid, a, c):
    p = np.clip(T.presence_p(a, c), 1e-9, 0.9997)
    d = load(sid, member); kk = np.diff(d['indptr'])
    b = np.searchsorted(T.EDGES, colsums(d), side='right') - 1
    Nb = np.bincount(b, minlength=16)
    bits = (Nb * (-np.log2(1 - p)).sum(0)).sum()
    pp = p[d['indices'], np.repeat(b, kk)]
    return (bits + np.log2((1 - pp) / pp).sum()) / len(d['indices'])

rows = []
for test in ('pbmc_v2_typical', 'pbmc_v2_shallow', 'tcell_5p', 'pbmc_v3_lowgenes', 'tumor_3p'):
    gse = ALL[test][0]
    cands = [s for s in hs if ALL[s][0] != gse]
    res = {}
    for s in cands:
        a, c = fit_one(s)
        res[s] = pres_bits(test, a, c)
    # pooled (already cached by the main trainer)
    z = np.load(os.path.join(CACHE, f'fit_human_{member}_{gse}.npz'))
    pooled = pres_bits(test, z['a'], z['c'])
    best = min(res, key=res.get)
    print(test, 'pooled %.3f' % pooled, 'best single %s %.3f' % (best, res[best]), ' '.join('%s:%.2f' % (k, v) for k, v in sorted(res.items(), key=lambda kv: kv[1])), flush=True)
    rows.append([test, pooled, best, res[best]] + [res[s] for s in sorted(res)])
with open(sys.argv[1], 'w') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['test', 'pooled', 'best_single', 'best_single_bits', 'all_singles_sorted_by_name']); w.writerows(rows)
