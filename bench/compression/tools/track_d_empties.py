#!/usr/bin/env python
"""Track D, task 1b: how much does relaxing 'empty droplets are exchangeable multinomial draws' buy?

Same held-out protocol as track_d_cells.py, applied to the empty droplets (columns that are not called
cells and have >= 1 UMI): code length of each held-out droplet, given its total k_j, under
  E_mult1   one pooled ambient profile (the exchangeable-multinomial assumption), smoothed toward the
            pooled profile of the called cells (which a decoder has already seen)
  E_multC   mixture of C multinomial profiles (droplet heterogeneity: cell debris, partial lysis)
Rows go to results/track_d_curve.tsv with model names E_mult1, E_mult4, ... and to results/track_d_asymptote.tsv.

    python -P tools/track_d_empties.py [sample:member ...]
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import track_d_cells as tc
from track_d_common import corpus_files, load, ROOT
from track_d_lib import *

FR = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
CS = [4, 16]


def prior_pooled(train, prior, beta_rel):
    F = train['m']
    T = np.bincount(train['indices'], weights=train['data'], minlength=F).astype(np.float64)
    N = T.sum()
    return (T + beta_rel * N * prior) / (N * (1 + beta_rel))


def main():
    want = sys.argv[1:]
    fc = open(tc.CURVE, 'a'); fa = open(tc.ASYM, 'a')
    if os.path.getsize(tc.CURVE) == 0:
        fc.write('sample\tmember\tmodel\tC\tn_train_cells\tnnz_train\tn_test_cells\tnnz_test\tbits_test\tbits_per_nnz\tbits_per_cell\talpha\tb\tnote\n')
    for s, mem, p in corpus_files():
        key = f'{s}:{mem}'
        if want and key not in want and s not in want:
            continue
        t0 = time.time()
        d = load(s, mem)
        m = int(d['m'])
        ic = d['is_cell']
        cc = np.diff(d['indptr'])
        rn = np.bincount(d['indices'], minlength=m)
        fmap = np.full(m, -1, dtype=np.int64)
        F = int((rn > 0).sum())
        fmap[rn > 0] = np.arange(F)
        base = dict(indptr=d['indptr'], indices=fmap[d['indices']], data=d['data'], m=F)
        emp_idx = np.flatnonzero((~ic) & (cc > 0))
        cell_idx = np.flatnonzero(ic)
        cells = tc.take_cols(base, cell_idx)
        Cc = np.bincount(cells['indices'], weights=cells['data'], minlength=F).astype(np.float64)
        prior = (Cc + 0.05) / (Cc.sum() + 0.05 * F)
        rng = np.random.default_rng(2)
        perm = emp_idx[rng.permutation(len(emp_idx))]
        nE = len(perm)
        nT = int(0.25 * nE); nV = int(0.10 * nE)
        test = tc.take_cols(base, np.sort(perm[:nT]))
        val = tc.take_cols(base, np.sort(perm[nT:nT + nV]))
        pool = perm[nT + nV:]
        nz_test = test['indices'].size
        print(f'{key}: empties={nE} test={nT} pool={len(pool)} F={F}', flush=True)
        percell = {}

        def emit(model, C, ntr, nnztr, costs, alpha, b):
            tot = float(np.sum(costs))
            percell[(model, ntr)] = np.asarray(costs, dtype=np.float64)
            fc.write(f'{s}\t{mem}\t{model}\t{C}\t{ntr}\t{nnztr}\t{test["n"]}\t{nz_test}\t{tot:.1f}\t{tot / nz_test:.5f}\t{tot / test["n"]:.3f}\t{alpha}\t{b}\t\n')
            fc.flush()

        for fr in FR:
            ntr = int(round(fr * len(pool)))
            if ntr < 50:
                continue
            tr = tc.take_cols(base, np.sort(pool[:ntr]))
            nnztr = tr['indices'].size
            best = None
            for br in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
                q = prior_pooled(tr, prior, br)
                c = tc.mult_cost_cells(val, np.log2(q)[val['indices']]).sum()
                if best is None or c < best[0]:
                    best = (c, br)
            br = best[1]
            q = prior_pooled(tr, prior, br)
            emit('E_mult1', 1, ntr, nnztr, tc.mult_cost_cells(test, np.log2(q)[test['indices']]), '', br)
            for C in CS:
                if ntr < 20 * C:
                    continue
                N, Nc, w, qq = em_fit(tr, C, q)
                bv = None
                for al in (10, 30, 100, 300, 1000, 3000, 10000, 1e5, 1e6):
                    cv = tc.mixture_cells_cost(val, N, Nc, w, qq, al).sum()
                    if bv is None or cv < bv[0]:
                        bv = (cv, al)
                emit(f'E_mult{C}', C, ntr, nnztr, tc.mixture_cells_cost(test, N, Nc, w, qq, bv[1]), bv[1], br)
            print(f'  ntr={ntr} done ({time.time() - t0:.0f}s)', flush=True)
        for model in sorted({k[0] for k in percell}):
            sizes = sorted(k[1] for k in percell if k[0] == model)
            if len(sizes) < 4:
                continue
            Ls = [percell[(model, n_)].sum() / nz_test for n_ in sizes]
            pf = tc.fit_asym(sizes, Ls); p05 = tc.fit_asym(sizes, Ls, 0.5); p1 = tc.fit_asym(sizes, Ls, 1.0)
            brng = np.random.default_rng(7)
            nzc = test['nnzc'].astype(np.float64)
            Lin = []
            for _ in range(100):
                bi = brng.integers(0, test['n'], size=test['n'])
                Lin.append(tc.fit_asym(sizes, [percell[(model, n_)][bi].sum() / nzc[bi].sum() for n_ in sizes])[0])
            Lin = np.array(Lin); Lin = Lin[np.isfinite(Lin)]
            lo, hi = (np.percentile(Lin, 2.5), np.percentile(Lin, 97.5)) if len(Lin) > 20 else (float('nan'),) * 2
            fa.write(f'{s}\t{mem}\t{model}\t{len(pool)}\t{Ls[-1]:.4f}\t{pf[0]:.4f}\t{lo:.4f}\t{hi:.4f}\t{p05[0]:.4f}\t{p1[0]:.4f}\t{pf[2]:.3f}\t{pf[1]:.3f}\t\n')
            fa.flush()
        print(f'{key} finished in {time.time() - t0:.0f}s', flush=True)


def em_fit(train, C, q, alpha_em=100.0, iters=10, seed=0):
    """EM for a multinomial mixture starting from hard assignments by cosine k-means++ seeding (same as cells)."""
    import scipy.sparse as sp
    from scipy.special import logsumexp
    rng = np.random.default_rng(seed)
    X = tc.csr_of(train); Xt = X.T.tocsr()
    Xs = X.copy(); Xs.data = np.sqrt(Xs.data)
    nr = np.sqrt(np.asarray(Xs.multiply(Xs).sum(1)).ravel()) + 1e-9
    Xn = (sp.diags(1.0 / nr) @ Xs).tocsr()
    seeds = tc.kmeanspp_seeds(Xn, C, rng)
    S = np.asarray((Xn @ Xn[seeds].T).todense())
    R = np.zeros_like(S, dtype=np.float32)
    R[np.arange(S.shape[0]), S.argmax(1)] = 1.0
    q32 = q.astype(np.float32)
    for it in range(iters):
        N = np.asarray(Xt @ R, dtype=np.float32)
        Nc = N.sum(0)
        w = (R.sum(0) + 1.0) / (R.shape[0] + C)
        logP = np.log2((N + alpha_em * q32[:, None]) / (Nc[None, :] + alpha_em)).astype(np.float32)
        L = np.asarray(X @ logP, dtype=np.float32) + np.log2(w)[None, :].astype(np.float32)
        lse = logsumexp(L * LN2, axis=1) / LN2
        R = np.exp2(L - lse[:, None].astype(np.float32)).astype(np.float32)
    N = np.asarray(Xt @ R, dtype=np.float32)
    w = (R.sum(0) + 1.0) / (R.shape[0] + C)
    return N, N.sum(0), w, q


if __name__ == '__main__':
    main()
