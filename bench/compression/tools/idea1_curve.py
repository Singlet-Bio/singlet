"""Learning curve: out-of-sample presence cross-entropy of the cloglog model on pbmc_v2_typical exon (and brain_sn_v3_deep exon)
as a function of the number of training studies, then adding the sibling sample of the same study (in-study).
   python -P tools/idea1_curve.py OUT.tsv"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idea1_train as T
import idea1_lib as L
from idea1_fit import fit_cloglog
from idea1_data import load, genes

def run(test, member, order):
    d = load(test, member); nnz = len(d['indices'])
    g = genes('human' if test.startswith(('pbmc', 'tcell', 'tumor')) else 'mouse', member)
    out = []
    for k in range(1, len(order) + 1):
        samples = order[:k]
        n, N = T.pooled_counts(samples, member)
        tot = n.sum(1)
        gm = T.gene_prior(tot, g)
        eps = np.where(tot > 0, 0.0, 0.2 * gm / (1 + gm / 50.0))
        a, c = fit_cloglog(n, N, eps=eps, iters=8)
        p = np.clip(T.presence_p(a, c), 1e-9, 0.9997)
        Nb = np.bincount(np.searchsorted(T.EDGES, T.colsums(d), side='right') - 1, minlength=16) if hasattr(T, 'colsums') else None
        D = T.colsums(d); kk = np.diff(d['indptr'])
        b = np.searchsorted(T.EDGES, D, side='right') - 1
        Nb = np.bincount(b, minlength=16)
        bits = (Nb * (-np.log2(1 - p)).sum(0)).sum()
        pp = p[d['indices'], np.repeat(b, kk)]
        bits += np.log2((1 - pp) / pp).sum()
        out.append((test, member, k, samples[-1], bits / nnz, float((tot > 0).mean())))
        print(out[-1], flush=True)
    return out

rows = []
rows += run('pbmc_v2_typical', 'exon', ['h_pbmc_a', 'h_c', 'h_b', 'tcell_5p', 'pbmc_v3_lowgenes', 'tumor_3p', 'pbmc_v2_shallow'])
rows += run('brain_sn_v3_deep', 'exon', ['m_brain_a', 'm_brain_c', 'm_brain_d', 'brain_sn_v3_intronic'])
with open(sys.argv[1], 'w') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['test', 'member', 'n_train', 'last_added', 'presence_bits_per_nnz', 'frac_features_seen']); w.writerows(rows)
