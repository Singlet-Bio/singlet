# How much of the shipped-static vs file-adaptive gap is a file-specific base profile? (Poisson, interval level, called cells)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
from scipy.special import gammaln
import idea4_lib as L, idea4_pp_export as EX
test = sys.argv[1]
org = L.ALL[test][2]
trs = ['h_pbmc_a','h_pbmc_b','h_b','h_c','h_d'] if org == 'human' else ['m_brain_a','m_brain_b','m_brain_c','m_brain_d']
ref = EX.get_ref(org)
cnt = np.zeros(ref.m)
for t in trs:
    cnt += EX.get_piece(t)['cnt_i']
S = L.load_sample(test)
X = L.csr_of(S, S.called).tocsr()
N = np.asarray(X.sum(1)).ravel(); nnz = X.nnz
rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr)); x = X.data.astype(np.float64)
const = (N.sum() + gammaln(x + 1).sum()) * L.LOG2E
par = EX.static_params(ref, cnt)
P0 = par['pi0d'][ref.go] * par['qd']; P0 /= P0.sum()
def cost(P): return (const - (x * np.log(N[rows] * P[X.indices])).sum() * L.LOG2E) / nnz
print(test, 'static (shipped, out-of-sample) %.3f' % cost(P0))
own = np.asarray(X.sum(0)).ravel()
for a in (10, 100, 1000):
    p = own + a * P0; p /= p.sum()
    print('  oracle file profile (+%d*P0)   %.3f   [in-sample, profile side info not counted]' % (a, cost(p)))
# leave-self-out profile (each cell excluded from its own profile): in-sample bias removed
tot = own.copy()
Xc = X.tocsr()
p_full = own + 100 * P0
cost_lso = 0.0
Pn = np.zeros(nnz)
# per-nnz leave-one-out probability: (own_j - x + a P0_j)/(sum - N + a)
den = own.sum() + 100 - N[rows]
Pn = (own[X.indices] - x + 100 * P0[X.indices]) / den
print('  leave-one-cell-out file profile  %.3f' % ((const - (x * np.log(N[rows] * Pn)).sum() * L.LOG2E) / nnz))
