"""Train and export Idea 4 models (static K=0 and manifold K>0) for one training set.

python -P tools/idea4_pp_build.py --train h_pbmc_a,h_pbmc_b --name human_all --ks 0,16 [--cap 2500] [--iters 40] [--beta 500]
Writes results/idea4_pp_model_<name>_k<K>.bin
"""
import argparse, os, sys, time
import numpy as np, scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_lib as L
import idea4_nmf as NM
import idea4_pp_export as EX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--ks', default='0,16')
    ap.add_argument('--cap', type=int, default=2500)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--beta', type=float, default=500.0)
    ap.add_argument('--nth', type=int, default=2)
    ap.add_argument('--amb_cols', type=int, default=20000)
    ap.add_argument('--kc', type=float, default=20000)
    ap.add_argument('--k0', type=float, default=2000)
    a = ap.parse_args()
    ids = a.train.split(',')
    ks = [int(k) for k in a.ks.split(',')]
    t0 = time.time()
    org = L.ALL[ids[0]][2]
    ref = EX.get_ref(org)
    m, G, go = ref.m, ref.G, ref.go
    cnt_i = np.zeros(m)
    Xcap_g, Xcap_i, ambs = [], [], []
    for sid in ids:
        pc = EX.get_piece(sid, a.cap, a.amb_cols)
        assert pc['m'] == m
        cnt_i += pc['cnt_i']
        Xcap_i.append(pc['Xc'])
        Xcap_g.append(L.gene_matrix(go, G, pc['Xc'].astype(np.float32)))
        ambs.append((pc['cnt_i'], pc['Xe']))
        print(f'  piece {sid}: called {pc["called_total"]} cap {pc["Xc"].shape[0]} empties {pc["Xe"].shape[0]}  {time.time() - t0:.0f}s', flush=True)
    par = EX.static_params(ref, cnt_i)
    pi0d, qd = par['pi0d'], par['qd']
    P0 = pi0d[go] * qd                       # static interval profile
    P0 = P0 / P0.sum()

    # ---- ambient count table (mode 1): profile = kc * called-cells profile + k0 * P0, scaled like the codec
    Hamb = np.zeros((EX.NCB, EX.NSYM))
    for Acell, Xe in ambs:
        prof = a.kc * Acell / max(Acell.sum(), 1) + a.k0 * P0
        prof = prof / prof.sum()
        N = np.asarray(Xe.sum(1)).ravel()
        rows = np.repeat(np.arange(Xe.shape[0]), np.diff(Xe.indptr))
        lam = N[rows] * prof[Xe.indices]
        Hamb += EX.hist_counts(EX.cbucket(lam), Xe.data.astype(np.int64))
    ctab_amb = EX.make_ctab(Hamb)

    # ---- static model (K=0)
    Xi = sp.vstack(Xcap_i).tocsr()
    N = np.asarray(Xi.sum(1)).ravel()
    rows = np.repeat(np.arange(Xi.shape[0]), np.diff(Xi.indptr))
    lam = N[rows] * P0[Xi.indices]
    Hc = EX.hist_counts(EX.cbucket(lam), Xi.data.astype(np.int64))
    ctab_cell0 = EX.make_ctab(Hc)
    if 0 in ks:
        p = f'results/idea4_pp_model_{a.name}_k0.bin'
        nb = EX.write_model(p, ref, par, 0, None, None, None, [ctab_cell0, ctab_amb])
        print(f'wrote {p} {nb} bytes', flush=True)

    # ---- manifold models
    Xtr = sp.vstack(Xcap_g).tocsr(); Xtr.sort_indices()
    pi0g = np.add.reduceat(P0, ref.st[:-1])                         # gene-level static profile (normalised)
    for K in ks:
        if K == 0:
            continue
        C = NM.Csr(Xtr)
        Phi, Th = NM.train(C, K, iters=a.iters, beta=a.beta, pi0=pi0g, nth=a.nth)
        Rcode, tscale = EX.quantise_phi(Phi, pi0d)
        Phiq = EX.dequant_phi(Rcode, tscale, pi0d)
        # refit theta with the dequantised model; derive theta prior and count tables
        Th = NM.fit_theta(C, Phiq, iters=12, nth=a.nth)
        codes = EX.theta_codes(Th)
        thprior = np.zeros((K, EX.NLEV))
        for k in range(K):
            thprior[k] = np.bincount(codes[:, k], minlength=EX.NLEV)
        thprior = np.round(thprior / thprior.sum(1, keepdims=True) * 256).astype(np.uint16)
        # lambda at interval nnz of the training cells under the model
        Pg = NM.pred_nz(C, Th, Phiq)                                  # gene-level probs at gene nnz
        pos = EX.map_gene_nnz_to_interval(Xi, Xtr, go)
        lam = N[rows] * Pg[pos].astype(np.float64) * qd[Xi.indices]
        Hc = EX.hist_counts(EX.cbucket(lam), Xi.data.astype(np.int64))
        ctab_cell = EX.make_ctab(Hc)
        p = f'results/idea4_pp_model_{a.name}_k{K}.bin'
        nb = EX.write_model(p, ref, par, K, Rcode, tscale, thprior, [ctab_cell, ctab_amb])
        print(f'wrote {p} {nb} bytes  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
