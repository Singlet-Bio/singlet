"""Per-cell gross gain of the manifold prior versus cell depth (gene-level Poisson, float theta), shipped base vs file base.

python -P tools/idea4_pp_breakeven.py <fold> <K> <test_sample>
Prints, per nnz-quantile bin of the test cells: mean gross gain in bits per cell (manifold - static), per nnz, and the break-even
against the measured side information (default 30 bits/cell).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
import idea4_lib as L, idea4_pp_export as EX, idea4_nmf as NM

fold, K, test = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SIDE = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
org = L.ALL[test][2]
d = EX.read_model('results/idea4_pp_model_%s_k%d.bin' % (fold, K))
ref = EX.get_ref(org)
pi0d = np.exp(-d['code0'].astype(np.float64) / 2048.0)
Phi = EX.dequant_phi(d['Rcode'], d['tscale'], pi0d)
S = L.load_sample(test)
X = L.csr_of(S, S.called).tocsr(); X.sort_indices()
Xg = L.gene_matrix(ref.go, ref.G, X.astype(np.float32)); Xg.sort_indices()
n = Xg.shape[0]
nnz_cell = np.diff(X.indptr)
rng = np.random.default_rng(0); fid = rng.integers(0, 2, n)
pi0g = pi0d / pi0d.sum()
r = Phi / pi0g[:, None]
out = {'shipped': np.zeros(n), 'filebase': np.zeros(n)}
for f in (0, 1):
    te = np.where(fid == f)[0]; tr = np.where(fid != f)[0]
    C = NM.Csr(Xg[te])
    own = np.asarray(Xg[tr].sum(0)).ravel()
    pf = own + 300.0 * pi0g; pf /= pf.sum()
    for name, base in (('shipped', pi0g), ('filebase', pf)):
        Ph = (base[:, None] * r); Ph = (Ph / Ph.sum(0, keepdims=True)).astype(np.float32)
        Th = NM.fit_theta(C, Ph, 30)
        ll_n = NM.cell_ll(C, Th, Ph)
        ll_s = NM.cell_ll(C, np.ones((C.n, 1), dtype=np.float32), base[:, None].astype(np.float32).copy())
        out[name][te] = (ll_n - ll_s) * L.LOG2E
qs = np.quantile(nnz_cell, np.linspace(0, 1, 9))
print(test, 'K=%d' % K, 'cells', n, 'side-info assumption %.0f bits/cell' % SIDE)
print('nnz/cell bin      cells  gross gain bits/cell (shipped base | file base)   per nnz (shipped | file)   net/cell after side (shipped | file)')
for i in range(8):
    sel = (nnz_cell >= qs[i]) & ((nnz_cell < qs[i + 1]) if i < 7 else (nnz_cell <= qs[i + 1]))
    if not sel.any():
        continue
    a, b = out['shipped'][sel].mean(), out['filebase'][sel].mean()
    nz = nnz_cell[sel].mean()
    print('%5.0f-%5.0f  %6d   %8.1f | %8.1f         %6.3f | %6.3f      %8.1f | %8.1f' % (qs[i], qs[i + 1], sel.sum(), a, b, a / nz, b / nz, a - SIDE, b - SIDE))
print('overall gross gain bits/nnz: shipped %.3f  filebase %.3f' % (out['shipped'].sum() / nnz_cell.sum(), out['filebase'].sum() / nnz_cell.sum()))
