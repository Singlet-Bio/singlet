"""Side-information alternatives for theta (gene-level Poisson cross-entropy, called cells of one test sample).

python -P tools/idea4_pp_codebook.py <fold> <k> <test_sample>
Compares, for the shipped (quantised) topics of the fold's model:
  static (K=0 model) | float theta | 15-level geometric theta (codec) | 256-entry k-means codebook of theta (1 byte/cell)
reporting gross bits/interval-nnz and side-information bits/cell (entropy of the level codes under per-component
adaptive models is measured by the codec itself; here we use order-0 entropy + MDL as a proxy).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
from scipy.special import gammaln
import idea4_lib as L, idea4_pp_export as EX, idea4_nmf as NM

fold, K, test = sys.argv[1], int(sys.argv[2]), sys.argv[3]
org = L.ALL[test][2]
d = EX.read_model('results/idea4_pp_model_%s_k%d.bin' % (fold, K))
d0 = EX.read_model('results/idea4_pp_model_%s_k0.bin' % fold)
ref = EX.get_ref(org)
pi0d = np.exp(-d['code0'].astype(np.float64) / 2048.0)
Phi = EX.dequant_phi(d['Rcode'], d['tscale'], pi0d)
w = np.exp2(-d['qcode'].astype(np.float64) / 16.0)
qd = w / np.add.reduceat(w, ref.st[:-1])[ref.go]
S = L.load_sample(test)
X = L.csr_of(S, S.called).tocsr(); X.sort_indices()
xi = X.data.astype(np.float64)
N = np.asarray(X.sum(1)).ravel(); nnz = X.nnz
split = -(xi * np.log(qd[X.indices])).sum() * L.LOG2E
const = (N.sum() + gammaln(xi + 1).sum()) * L.LOG2E
Xg = L.gene_matrix(ref.go, ref.G, X.astype(np.float32)); Xg.sort_indices()
C = NM.Csr(Xg)
rg = np.repeat(np.arange(C.n), np.diff(C.indptr)); xg = C.data.astype(np.float64)


def bpn(P):
    return (-(xg * (np.log(N[rg]) + np.log(np.maximum(P, 1e-30)))).sum() * L.LOG2E + split + const) / nnz


pi0g = np.add.reduceat(pi0d[ref.go] * qd, ref.st[:-1]); pi0g /= pi0g.sum()
print(test, 'cells', C.n, 'static %.3f' % bpn(pi0g[C.indices]))
Th = NM.fit_theta(C, Phi, 30)
print('float theta  %.3f' % bpn(NM.pred_nz(C, Th, Phi).astype(np.float64)))
codes = EX.theta_codes(Th)
wq = EX.LEVEL_TQ[codes]
Thq = (wq / np.maximum(wq.sum(1, keepdims=True), 1e-9)).astype(np.float32)
print('15-level geometric theta  %.3f' % bpn(NM.pred_nz(C, Thq, Phi).astype(np.float64)))
side = 0.0
for k in range(K):
    h = np.bincount(codes[:, k], minlength=16).astype(float); p = h[h > 0] / h.sum()
    side += float(-(h[h > 0] * np.log2(p)).sum() + 0.5 * np.log2(C.n) * (h > 0).sum())
print('  order-0 side info: %.1f bits/cell (%.3f bits/nnz)' % (side / C.n, side / nnz))
# codebook: k-means over sqrt(theta) of the test cells' own fits is optimistic; train the codebook on other cells instead:
# use the shipped training-fold cells (called, capped) from the cache
trs = [t for t in L.ALL if L.ALL[t][2] == org and t != test and t in
       ('h_pbmc_a', 'h_pbmc_b', 'h_b', 'h_c', 'h_d', 'm_brain_a', 'm_brain_b', 'm_brain_c', 'm_brain_d')]
Xtr = sp.vstack([L.gene_matrix(ref.go, ref.G, EX.get_piece(t)['Xc'].astype(np.float32)) for t in trs]).tocsr(); Xtr.sort_indices()
Ctr = NM.Csr(Xtr)
Tht = NM.fit_theta(Ctr, Phi, 30)
Z = np.sqrt(Tht)
rng = np.random.default_rng(0)
cent = Z[rng.choice(len(Z), 256, replace=False)].copy()
for it in range(12):
    dist = (Z ** 2).sum(1, keepdims=True) - 2 * Z @ cent.T + (cent ** 2).sum(1)[None]
    lab = dist.argmin(1)
    for j in range(256):
        sel = lab == j
        if sel.any():
            cent[j] = Z[sel].mean(0)
cb = cent ** 2
cb /= cb.sum(1, keepdims=True)
cb = cb.astype(np.float32)
# nearest codebook entry in Hellinger distance, then choose the best of the 4 nearest by actual likelihood
Zt = np.sqrt(Th)
dist = (Zt ** 2).sum(1, keepdims=True) - 2 * Zt @ np.sqrt(cb).T + (np.sqrt(cb) ** 2).sum(1)[None]
cand = np.argsort(dist, axis=1)[:, :4]
best = None
for r in range(4):
    Pn = NM.pred_nz(C, cb[cand[:, r]], Phi).astype(np.float64)
    ll = np.bincount(rg, weights=xg * np.log(np.maximum(Pn, 1e-30)), minlength=C.n)
    best = ll if best is None else np.maximum(best, ll)
tot = -(xg * np.log(N[rg])).sum() * L.LOG2E - best.sum() * L.LOG2E
print('256-codebook theta (1 B/cell, trained on other cells)  %.3f  (+8 bits/cell = %.3f b/nnz)' % ((tot + split + const) / nnz, 8.0 * C.n / nnz))
