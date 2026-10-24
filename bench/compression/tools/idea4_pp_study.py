"""Out-of-sample cross-entropy study of the manifold prior at gene level (float arithmetic, Poisson counts).

python -P tools/idea4_study.py --test pbmc_v2_typical --train h_pbmc_a,h_pbmc_b,h_b,h_c,h_d --ks 8,16,32 --betas 500
Reports, for the called cells of the test sample (bits per interval-level nonzero):
  static (global profile x depth), depth-binned static, manifold with float theta, with quantised theta,
  side-information entropy (theta levels + depth scale) and the net gain.
"""
import argparse, os, sys, time, json
import numpy as np, scipy.sparse as sp
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_lib as L
import idea4_nmf as NM
import idea4_pp_export as EX

LOG2E = L.LOG2E


def entropy_bits(codes, nsym):
    """empirical order-0 entropy (bits) of a code column set, plus 0.5*log2(n) per used symbol (MDL)"""
    h = np.bincount(codes, minlength=nsym).astype(np.float64)
    n = h.sum()
    p = h[h > 0] / n
    return float(-(h[h > 0] * np.log2(p)).sum() + 0.5 * np.log2(max(n, 2)) * (h > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--ks', default='16')
    ap.add_argument('--betas', default='500')
    ap.add_argument('--cap', type=int, default=2500)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--test_cap', type=int, default=0)
    ap.add_argument('--insample', action='store_true', help='train on the test sample itself (upper bound)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    t0 = time.time()
    ids = a.train.split(',')
    S0 = L.load_sample(ids[0])
    ref = EX.Ref(S0.rownames)
    go, G, m = ref.go, ref.G, ref.m
    cnt_i = np.zeros(m)
    Xg_tr = []
    for sid in ids:
        S = S0 if sid == ids[0] else L.load_sample(sid)
        sel = np.where(S.called)[0]
        Xall = L.csr_of(S, sel).tocsr()
        cnt_i += np.asarray(Xall.sum(0)).ravel()
        idx = np.arange(len(sel))
        if len(idx) > a.cap:
            idx = np.sort(np.random.default_rng(1).choice(idx, a.cap, replace=False))
        Xg_tr.append(L.gene_matrix(go, G, Xall[idx].astype(np.float32)))
    Xtr = sp.vstack(Xg_tr).tocsr(); Xtr.sort_indices()
    T = L.load_sample(a.test)
    selt = np.where(T.called)[0]
    if a.test_cap and len(selt) > a.test_cap:
        selt = np.sort(np.random.default_rng(3).choice(selt, a.test_cap, replace=False))
    Xi = L.csr_of(T, selt).tocsr(); Xi.sort_indices()
    Xg = L.gene_matrix(go, G, Xi.astype(np.float32)); Xg.sort_indices()
    if a.insample:   # upper bound: also train on the test sample's own cells
        cnt_i += np.asarray(Xi.sum(0)).ravel()
        cap_idx = np.sort(np.random.default_rng(4).choice(Xg.shape[0], min(Xg.shape[0], a.cap), replace=False))
        Xtr = sp.vstack([Xtr, Xg[cap_idx]]).tocsr(); Xtr.sort_indices()
    par = EX.static_params(ref, cnt_i)
    pi0d, qd = par['pi0d'], par['qd']
    P0 = pi0d[go] * qd; P0 /= P0.sum()
    pi0g = np.add.reduceat(P0, ref.st[:-1])
    N = np.asarray(Xi.sum(1)).ravel()
    nnz = Xi.nnz
    xi = Xi.data.astype(np.float64)
    rows_i = np.repeat(np.arange(Xi.shape[0]), np.diff(Xi.indptr))
    split_bits = -(xi * np.log(qd[Xi.indices])).sum() * LOG2E
    const_bits = (N.sum() + gammaln(xi + 1).sum()) * LOG2E
    Cg = NM.Csr(Xg)
    xg = Cg.data.astype(np.float64)
    rg = np.repeat(np.arange(Cg.n), np.diff(Cg.indptr))
    res = {'test': a.test, 'cells': int(Xi.shape[0]), 'nnz': int(nnz), 'gene_nnz': int(Cg.data.shape[0]), 'train_cells': int(Xtr.shape[0]), 'insample': a.insample}

    def total(gene_bits):
        return (gene_bits + split_bits + const_bits)

    stat_bits = -(xg * (np.log(N[rg]) + np.log(pi0g[Cg.indices]))).sum() * LOG2E
    res['static_bpn'] = total(stat_bits) / nnz
    # depth-binned static profile (nonparametric per-gene x depth): 8 log-depth bins from the training cells
    Ntr = np.asarray(Xtr.sum(1)).ravel()
    edges = np.quantile(np.log(np.maximum(Ntr, 1)), np.linspace(0, 1, 9)[1:-1])
    btr = np.searchsorted(edges, np.log(np.maximum(Ntr, 1)))
    prof = np.zeros((8, G))
    for b in range(8):
        prof[b] = np.asarray(Xtr[btr == b].sum(0)).ravel() + 500.0 * pi0g
        prof[b] /= prof[b].sum()
    bt = np.searchsorted(edges, np.log(np.maximum(N, 1)))
    db_bits = -(xg * (np.log(N[rg]) + np.log(prof[bt[rg], Cg.indices]))).sum() * LOG2E
    res['depthbin_bpn'] = total(db_bits) / nnz
    # scale side information (1/8 octave class entropy), common to all models
    s_code = np.round(8 * np.log2(np.maximum(N, 1))).astype(np.int64)
    res['scale_side_bits_per_cell'] = entropy_bits(s_code - s_code.min(), int(s_code.max() - s_code.min() + 1)) / Xi.shape[0]
    res['models'] = {}
    for beta in [float(b) for b in a.betas.split(',')]:
        for K in [int(k) for k in a.ks.split(',')]:
            C = NM.Csr(Xtr)
            tt = time.time()
            Phi, _ = NM.train(C, K, iters=a.iters, beta=beta, pi0=pi0g, verbose=False)
            Rc, ts = EX.quantise_phi(Phi, pi0d)
            Phiq = EX.dequant_phi(Rc, ts, pi0d)
            out = {}
            for nm_, Ph in (('float', Phi), ('quant', Phiq)):
                Th = NM.fit_theta(Cg, Ph, iters=30)
                Pn = NM.pred_nz(Cg, Th, Ph).astype(np.float64)
                gb = -(xg * (np.log(N[rg]) + np.log(np.maximum(Pn, 1e-30)))).sum() * LOG2E
                out[nm_ + '_bpn'] = total(gb) / nnz
                if nm_ == 'quant':
                    # theta quantisation to the codec's 15 levels (relative to the max component)
                    codes = EX.theta_codes(Th)
                    w = EX.LEVEL_TQ[codes]
                    Thq = (w / np.maximum(w.sum(1, keepdims=True), 1e-9)).astype(np.float32)
                    Pn = NM.pred_nz(Cg, Thq, Ph).astype(np.float64)
                    gb = -(xg * (np.log(N[rg]) + np.log(np.maximum(Pn, 1e-30)))).sum() * LOG2E
                    out['thq_bpn'] = total(gb) / nnz
                    side = sum(entropy_bits(codes[:, k], 16) for k in range(K))
                    out['theta_side_bits_per_cell'] = side / Xi.shape[0]
                    out['theta_nonzero_per_cell'] = float((codes > 0).sum(1).mean())
            out['train_s'] = time.time() - tt
            res['models'][f'k{K}_beta{int(beta)}'] = out
            net = res['static_bpn'] - out['thq_bpn'] - out['theta_side_bits_per_cell'] * Xi.shape[0] / nnz
            print(f"{a.test} k={K} beta={beta}: static {res['static_bpn']:.3f} depthbin {res['depthbin_bpn']:.3f} | manifold float {out['float_bpn']:.3f} "
                  f"quantPhi {out['quant_bpn']:.3f} +thetaQ {out['thq_bpn']:.3f} | side {out['theta_side_bits_per_cell']:.1f} b/cell "
                  f"({out['theta_side_bits_per_cell'] * Xi.shape[0] / nnz:.3f} b/nnz) | net gain vs static {net:.3f} b/nnz  [{time.time() - t0:.0f}s]", flush=True)
    if a.out:
        with open(a.out, 'a') as f:
            f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
