# Decompose the shipped-vs-file-adaptive gap into gene profile and interval split (2-fold cross-fit, Poisson, called cells).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
from scipy.special import gammaln
import idea4_lib as L, idea4_pp_export as EX, idea4_nmf as NM
test = sys.argv[1]; fold = sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 0
A = 300.0; AQ = 3.0
d = EX.read_model('results/idea4_pp_model_%s_k0.bin' % fold)
org = L.ALL[test][2]; ref = EX.get_ref(org); go, st, G, m = ref.go, ref.st, ref.G, ref.m
pi0d = np.exp(-d['code0'].astype(np.float64) / 2048.0)
w = np.exp2(-d['qcode'].astype(np.float64) / 16.0); qd = w / np.add.reduceat(w, st[:-1])[go]
pi0g = pi0d / pi0d.sum()
S = L.load_sample(test)
X = L.csr_of(S, S.called).tocsr(); X.sort_indices(); n = X.shape[0]
N = np.asarray(X.sum(1)).ravel(); nnz = X.nnz; xi = X.data.astype(np.float64)
rows = np.repeat(np.arange(n), np.diff(X.indptr))
const = (N.sum() + gammaln(xi + 1).sum()) * L.LOG2E
rng = np.random.default_rng(0); fold_id = rng.integers(0, 2, n)
res = {}
def add(key, v): res[key] = res.get(key, 0.0) + v
for f in (0, 1):
    te = fold_id == f; tr = ~te
    Xtr = X[np.where(tr)[0]]
    cnt_i = np.asarray(Xtr.sum(0)).ravel()
    cnt_g = np.add.reduceat(cnt_i, st[:-1])
    pg_file = cnt_g + A * pi0g; pg_file /= pg_file.sum()
    qv = cnt_i + AQ * qd; qf = qv / np.add.reduceat(qv, st[:-1])[go]
    sel = te[rows]
    gi = go[X.indices[sel]]; ci = X.indices[sel]; x = xi[sel]; r = rows[sel]
    for gname, pg in (('shipped_gene', pi0g), ('file_gene', pg_file)):
        for qname, q in (('shipped_q', qd), ('file_q', qf)):
            bits = -(x * (np.log(N[r]) + np.log(pg[gi]) + np.log(q[ci]))).sum() * L.LOG2E
            add(gname + '+' + qname, bits)
for k, v in res.items():
    print(test, k, '%.3f bits/nnz' % ((v + const) / nnz))
