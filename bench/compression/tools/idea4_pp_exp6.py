# Hybrid test: shipped topic ratios + file-specific base profile (2-fold cross-fit so a cell never sees its own counts).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
from scipy.special import gammaln
import idea4_lib as L, idea4_pp_export as EX, idea4_nmf as NM
test = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 16; A = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
mod = np.load('results/idea4_exp2_model_k%d.npz' % k)
Phi, pi0g = mod['Phi'], mod['pi0']
ref = EX.get_ref('human'); go, G = ref.go, ref.G
S = L.load_sample(test)
X = L.csr_of(S, S.called).tocsr(); X.sort_indices()
Xg = L.gene_matrix(go, G, X.astype(np.float32)); Xg.sort_indices()
n = Xg.shape[0]; N = np.asarray(Xg.sum(1)).ravel(); nnz = X.nnz
xi = X.data.astype(np.float64)
const = (N.sum() + gammaln(xi + 1).sum()) * L.LOG2E
# static q from the training set (as in the codec) -> split bits
cnt_i = sum(EX.get_piece(t)['cnt_i'] for t in ['h_pbmc_a','h_pbmc_b','h_b','h_c','h_d'])
par = EX.static_params(ref, cnt_i)
split = -(xi * np.log(par['qd'][X.indices])).sum() * L.LOG2E
rng = np.random.default_rng(0); fold = rng.integers(0, 2, n)
res = {'static_shipped': 0.0, 'static_file': 0.0, 'nmf_shipped': 0.0, 'nmf_filebase': 0.0}
for f in (0, 1):
    te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    Ct = NM.Csr(Xg[te])
    own = np.asarray(Xg[tr].sum(0)).ravel()
    pf = own + A * pi0g; pf = pf / pf.sum()
    rg = np.repeat(np.arange(Ct.n), np.diff(Ct.indptr)); xg = Ct.data.astype(np.float64); Nt = Ct.N
    def bits(P): return -(xg * (np.log(Nt[rg]) + np.log(np.maximum(P, 1e-30)))).sum() * L.LOG2E
    res['static_shipped'] += bits(pi0g[Ct.indices])
    res['static_file'] += bits(pf[Ct.indices])
    Th = NM.fit_theta(Ct, Phi, 30); res['nmf_shipped'] += bits(NM.pred_nz(Ct, Th, Phi).astype(np.float64))
    r = Phi / pi0g[:, None]
    Phi2 = pf[:, None] * r; Phi2 = (Phi2 / Phi2.sum(0, keepdims=True)).astype(np.float32)
    Th2 = NM.fit_theta(Ct, Phi2, 30); res['nmf_filebase'] += bits(NM.pred_nz(Ct, Th2, Phi2).astype(np.float64))
print(test, 'k=%d' % k, ' '.join('%s %.3f' % (a, (b + split + const) / nnz) for a, b in res.items()), '(bits/interval-nnz, Poisson, cells only; file profile side info NOT counted)')
