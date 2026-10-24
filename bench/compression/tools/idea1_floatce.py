"""Presence cross-entropy (bits/nnz, incl. END of column) of the UNQUANTISED cloglog fit  p_fb = 1-exp(-exp(a_f+c_b))
vs the quantised dense (8-bit logit) and rank-1 (8-bit a_f + 16 C_b) models, on a held-out file.
   python -P tools/idea1_floatce.py OUT.tsv   (all 14 test files, out-of-sample models)"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idea1_ce as C
import idea1_train as T
from idea1_data import load, colsums, TEST, CACHE, ROOT

def float_presence(hold, org, member, sid):
    z = np.load(os.path.join(CACHE, f'fit_{org}_{member}_{hold}.npz'))
    p = np.clip(T.presence_p(z['a'], z['c']), 1e-9, 0.9997)
    d = load(sid, member); nnz = len(d['indices'])
    D = colsums(d); k = np.diff(d['indptr'])
    b = np.searchsorted(T.EDGES, D, side='right') - 1
    Nb = np.bincount(b, minlength=16)
    absent = -np.log2(1 - p)
    bits = (Nb * absent.sum(0)).sum()
    bn = np.repeat(b, k)
    pp = p[d['indices'], bn]
    bits += (np.log2((1 - pp) / pp)).sum()
    return bits / nnz

import idea1_lib as L
def oracle_presence(sid, member):
    """presence entropy of the best per-(feature, depth bucket) Bernoulli table fitted on the file itself (no model cost)"""
    d = load(sid, member); nnz = len(d['indices'])
    n, N = L.counts_table(d, T.EDGES)
    p = np.clip(n / np.maximum(N, 1)[None, :], 0, 1 - 1e-12)
    absent = (N * (-np.log2(1 - p)).sum(0)).sum()
    D = colsums(d); k = np.diff(d['indptr'])
    b = np.searchsorted(T.EDGES, D, side='right') - 1
    pp = p[d['indices'], np.repeat(b, k)]
    return (absent + np.log2((1 - pp) / pp).sum()) / nnz

rows = []
for sid, (gse, org) in TEST.items():
    for member in ('exon', 'intron'):
        om = f'{org}_{member}'
        try:
            fl = float_presence(gse, org, member, sid)
            qd = C.ce(os.path.join(ROOT, 'results', 'idea1_models', f'{om}_s3_{gse}.bin'), sid, member)
            qr = C.ce(os.path.join(ROOT, 'results', 'idea1_models', f'{om}_s3r_{gse}.bin'), sid, member)
        except Exception as e:
            print('skip', sid, member, e); continue
        orc = oracle_presence(sid, member)
        rows.append(dict(sample=sid, member=member, file_oracle_fxd=orc, float_presence=fl, dense_presence=qd['presence'], rank1_presence=qr['presence']))
        print('%-20s %-6s oracle(fxd on file) %.3f  float %.3f  dense8 %.3f (%+.3f)  rank1 %.3f (%+.3f)' % (sid, member, orc, fl, qd['presence'], qd['presence'] - fl, qr['presence'], qr['presence'] - fl), flush=True)
with open(sys.argv[1], 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(rows)
