import sys; sys.path.insert(0, 'tools')
from e2_common import *
def H(x):
    _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum()
def Hc(x, y):
    # H(x | y)
    key = y.astype(np.int64) * 100000 + x
    return H(key) - H(y)
for s in SAMPLES:
    d = load(s, 'exon')
    N = d['N']; n = d['n']
    # log-bucket with 8 mantissa levels: e = floor(log2(N+1)), top 3 mantissa bits in symbol, rest raw
    v = N + 1
    e = np.floor(np.log2(v)).astype(int)
    mant = ((v >> np.maximum(e - 3, 0)) & 7) * (e >= 3)
    sym = e * 8 + mant
    raw = np.maximum(e - 3, 0)
    h_marg = H(sym) + raw.mean()
    # context: previous column's symbol bucket (e only)
    prev = np.concatenate([[0], e[:-1]])
    h_ctx = Hc(sym, prev) + raw.mean()
    # exact entropy of N itself
    print(f"{s:22s} n={n} H(N)exact={H(N):.2f}  coded-marg={h_marg:.2f} coded|prev e={h_ctx:.2f} bits/col; total KB marg={h_marg*n/8192:.1f} ctx={h_ctx*n/8192:.1f}  (payload nnz {d['k'].sum()}, {h_marg*n/d['k'].sum():.3f} b/nnz)")
