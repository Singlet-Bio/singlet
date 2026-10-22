#!/usr/bin/env python
"""Track D, task 2: called cells, diminishing returns.

Cross-entropy (bits per nnz of held-out cells) of successively richer models as a function of the number of
training cells, plus asymptote fits with a bootstrap interval.

    python -P tools/track_d_cells.py [sample:member ...]

Protocol (per matrix, called cells only, columns restricted to the F rows that have >= 1 nnz anywhere in the file;
that set is known to a decoder from the per-row counts):
  random order of cells (seed 1); test = 25%, validation = 10%, training pool = 65%.
  Training sets are the first f * pool cells, f in {1/32, 1/16, 1/8, 1/4, 1/2, 1}.
  Hyper-parameters (smoothing strengths) are chosen on the validation cells, never on the test cells.
Models (all give a code length for the unordered multiset of UMIs of each held-out cell, given its total k_j):
  comb      uniform over all columns with the same (nnz_j, k_j): log2 C(F, nnz) + log2 C(k-1, nnz-1)   (no learning)
  M1        per-feature Bernoulli presence + order-0 value table
  M2        per-feature x depth-bucket Bernoulli presence + value table by rate context log2(k_j q_f)
  mult1     multinomial with ONE pooled profile (count-level; presence + values together)
  multC     mixture of C multinomial profiles (latent cell types), C in {4, 16, 64}; EM on the training cells
Output: results/track_d_curve.tsv (one row per model x training size), results/track_d_asymptote.tsv.
"""
import os, sys, time, math
import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln, logsumexp
from scipy.optimize import curve_fit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import corpus_files, load, ROOT
from track_d_lib import *

CURVE = os.path.join(ROOT, 'results', 'track_d_curve.tsv')
ASYM = os.path.join(ROOT, 'results', 'track_d_asymptote.tsv')
FRACS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
CS = [4, 16, 64]


def take_cols(base, idx):
    """Column subset of a CSC dict (base) by explicit column index array."""
    ip = base['indptr']
    st = ip[idx]
    ln = ip[idx + 1] - st
    newp = np.zeros(len(idx) + 1, dtype=np.int64)
    newp[1:] = np.cumsum(ln)
    tot = int(newp[-1])
    pos = np.arange(tot) - np.repeat(newp[:-1], ln) + np.repeat(st, ln)
    ncol = np.repeat(np.arange(len(idx)), ln)
    dat = base['data'][pos]
    k = np.bincount(ncol, weights=dat, minlength=len(idx)).astype(np.int64)
    return dict(n=len(idx), indptr=newp, indices=base['indices'][pos], data=dat, k=k, nnzc=ln, col=ncol,
                m=base['m'])


def csr_of(s):
    return sp.csr_matrix((s['data'].astype(np.float32), s['indices'], s['indptr']), shape=(s['n'], s['m']))


def coef_bits(s):
    """Per-cell multinomial coefficient term log2 k! - sum log2 x! (bits, positive)."""
    lf = log2_fact(s['k'])
    sx = np.bincount(s['col'], weights=log2_fact(s['data']), minlength=s['n'])
    return lf - sx


# ------------------------------------------------------------------ count-level multinomial models
def pooled_profile(train, b):
    F = train['m']
    T = np.bincount(train['indices'], weights=train['data'], minlength=F).astype(np.float64)
    return (T + b) / (T.sum() + b * F)


def mult_cost_cells(test, logp_ent):
    """-log2 Mult(x_j | k_j, p) per cell given log2 p at every nonzero entry."""
    sp_ = np.bincount(test['col'], weights=test['data'] * logp_ent, minlength=test['n'])
    return -(coef_bits(test) + sp_)


def dm_cost_cells(test, q, sconc):
    """Dirichlet-multinomial (burstiness) code length per cell: concentration sconc around profile q."""
    a = sconc * q[test['indices']]
    lg = lambda v: gammaln(v) / LN2
    t = np.bincount(test['col'], weights=lg(test['data'] + a) - lg(a), minlength=test['n'])
    k = test['k'].astype(np.float64)
    return -(coef_bits(test) + lg(sconc) - lg(k + sconc) + t)


def fit_mult1(train, val):
    best = None
    for b in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        q = pooled_profile(train, b)
        c = mult_cost_cells(val, np.log2(q)[val['indices']]).sum()
        if best is None or c < best[0]:
            best = (c, b)
    return best[1]


def kmeanspp_seeds(Xn, C, rng):
    n = Xn.shape[0]
    first = rng.integers(n)
    chosen = [first]
    sim = np.asarray((Xn @ Xn[first].T).todense()).ravel()
    best = sim.copy()
    for _ in range(1, C):
        w = np.maximum(1.0 - best, 0) ** 2
        if w.sum() <= 0:
            nxt = rng.integers(n)
        else:
            nxt = rng.choice(n, p=w / w.sum())
        chosen.append(nxt)
        sim = np.asarray((Xn @ Xn[nxt].T).todense()).ravel()
        best = np.maximum(best, sim)
    return chosen


def em_mixture(train, C, b, alpha_em=100.0, iters=12, seed=0):
    """EM for a mixture of multinomials with hierarchical (Dirichlet(alpha q)) smoothing. Returns (Ncf, Nc, w, q)."""
    F = train['m']
    rng = np.random.default_rng(seed)
    q = pooled_profile(train, b)
    X = csr_of(train)
    Xt = X.T.tocsr()
    # sqrt-normalised rows for cosine seeding
    Xs = X.copy()
    Xs.data = np.sqrt(Xs.data)
    nr = np.sqrt(np.asarray(Xs.multiply(Xs).sum(1)).ravel()) + 1e-9
    Xn = sp.diags(1.0 / nr) @ Xs
    Xn = Xn.tocsr()
    seeds = kmeanspp_seeds(Xn, C, rng)
    S = np.asarray((Xn @ Xn[seeds].T).todense())     # n x C
    R = np.zeros_like(S, dtype=np.float32)
    R[np.arange(S.shape[0]), S.argmax(1)] = 1.0
    lq = np.log2(q).astype(np.float32)
    for it in range(iters):
        N = np.asarray(Xt @ R, dtype=np.float32)               # F x C
        Nc = N.sum(0)                                          # C
        w = (R.sum(0) + 1.0) / (R.shape[0] + C)
        logP = np.log2((N + alpha_em * q[:, None].astype(np.float32)) / (Nc[None, :] + alpha_em)).astype(np.float32)
        L = np.asarray(X @ logP, dtype=np.float32) + np.log2(w)[None, :].astype(np.float32)
        lse = logsumexp(L * LN2, axis=1) / LN2
        R = np.exp2(L - lse[:, None].astype(np.float32)).astype(np.float32)
    N = np.asarray(Xt @ R, dtype=np.float32)
    w = (R.sum(0) + 1.0) / (R.shape[0] + C)
    return N, N.sum(0), w, q


def mixture_cells_cost(test, N, Nc, w, q, alpha):
    X = csr_of(test)
    logP = np.log2((N + alpha * q[:, None].astype(np.float32)) / (Nc[None, :] + alpha)).astype(np.float32)
    L = np.asarray(X @ logP, dtype=np.float64) + np.log2(w)[None, :]
    lse = logsumexp(L * LN2, axis=1) / LN2
    return -(coef_bits(test) + lse)


# ------------------------------------------------------------------ asymptote fit
def powerlaw(N, Linf, a, b):
    return Linf + a * np.power(N, -b)


def fit_asym(Ns, Ls, bfix=None):
    Ns = np.asarray(Ns, dtype=np.float64); Ls = np.asarray(Ls, dtype=np.float64)
    x = Ns / Ns.max()
    try:
        if bfix is None:
            p, _ = curve_fit(powerlaw, x, Ls, p0=[Ls.min() * 0.9, (Ls.max() - Ls.min()), 0.5],
                             bounds=([0, 0, 0.1], [Ls.min() + 1e-9, 1e3, 3.0]), maxfev=20000)
        else:
            f = lambda N, Linf, a: powerlaw(N, Linf, a, bfix)
            p, _ = curve_fit(f, x, Ls, p0=[Ls.min() * 0.9, (Ls.max() - Ls.min())],
                             bounds=([0, 0], [Ls.min() + 1e-9, 1e3]), maxfev=20000)
            p = [p[0], p[1], bfix]
        return p
    except Exception:
        return [float('nan')] * 3


def main():
    want = sys.argv[1:]
    cur_new = not os.path.exists(CURVE)
    asy_new = not os.path.exists(ASYM)
    fc = open(CURVE, 'a'); fa = open(ASYM, 'a')
    if cur_new:
        fc.write('sample\tmember\tmodel\tC\tn_train_cells\tnnz_train\tn_test_cells\tnnz_test\tbits_test\tbits_per_nnz\tbits_per_cell\talpha\tb\tnote\n')
    if asy_new:
        fa.write('sample\tmember\tmodel\tn_cells_pool\tL_at_max\tLinf_free\tLinf_lo95\tLinf_hi95\tLinf_b05\tLinf_b1\tb_free\ta_free\tnote\n')
    for s, mem, p in corpus_files():
        key = f'{s}:{mem}'
        if want and key not in want and s not in want:
            continue
        t0 = time.time()
        d = load(s, mem)
        m = int(d['m'])
        ic = d['is_cell']
        col = columns_of(d)
        cc = np.diff(d['indptr'])
        # feature space F = rows with >=1 nnz anywhere in the file
        rn = np.bincount(d['indices'], minlength=m)
        fmap = np.full(m, -1, dtype=np.int64)
        F = int((rn > 0).sum())
        fmap[rn > 0] = np.arange(F)
        cells = np.flatnonzero(ic)
        base_all = dict(indptr=d['indptr'], indices=fmap[d['indices']], data=d['data'], m=F)
        rng = np.random.default_rng(1)
        perm = cells[rng.permutation(len(cells))]
        nC = len(perm)
        nT = int(0.25 * nC); nV = int(0.10 * nC)
        test = take_cols(base_all, np.sort(perm[:nT]))
        val = take_cols(base_all, np.sort(perm[nT:nT + nV]))
        poolidx = perm[nT + nV:]
        print(f'{key}: F={F} cells={nC} test={nT} val={nV} pool={len(poolidx)} test_nnz={test["indices"].size}', flush=True)
        if nT < 200 or len(poolidx) < 400:
            print('  too few cells, skipping'); continue
        nz_test = test['indices'].size
        percell = {}     # (model, ntrain) -> per-cell cost array on test

        def emit(model, C, ntr, nnztr, costs, alpha, b, note=''):
            tot = float(np.sum(costs))
            percell[(model, ntr)] = np.asarray(costs, dtype=np.float64)
            fc.write(f'{s}\t{mem}\t{model}\t{C}\t{ntr}\t{nnztr}\t{test["n"]}\t{nz_test}\t{tot:.1f}\t{tot / nz_test:.5f}\t{tot / test["n"]:.3f}\t{alpha}\t{b}\t{note}\n')
            fc.flush()

        # --- no-learning combinatorial
        nj = test['nnzc'].astype(np.float64); kj = test['k'].astype(np.float64)
        comb = log2_binom(F, nj) + log2_binom(np.maximum(kj - 1, 0), np.maximum(nj - 1, 0))
        emit('comb', 0, 0, 0, comb, '', '', 'log2 C(F,nnz)+log2 C(k-1,nnz-1)')
        side_k = h0_symbols(np.concatenate([test['k']]))
        gamma_bits = None

        for fr in FRACS:
            ntr = int(round(fr * len(poolidx)))
            if ntr < 20:
                continue
            tr = take_cols(base_all, np.sort(poolidx[:ntr]))
            nnztr = tr['indices'].size
            # ---- M1 / M2 factorised (smoothing strengths chosen on the validation cells)
            r1 = presence_cost_feature(tr, val, all_alphas=True)
            a1 = min(r1, key=r1.get)
            ctx = rate_ctx_factory(tr)
            ve0 = value_cost_entries(tr, test, lambda s_: np.zeros(s_['indices'].size, dtype=np.int64))
            ver = value_cost_entries(tr, test, ctx)
            pc1 = presence_percell_feature(tr, test, a1) + np.bincount(test['col'], weights=ve0, minlength=test['n'])
            emit('M1', 0, ntr, nnztr, pc1, a1, '', 'per-feature presence + order-0 values')
            r2 = presence_cost_feature_depth(tr, val, all_alphas=True, alpha1=a1)
            a2 = min(r2, key=r2.get)
            pc2 = presence_percell_featdepth(tr, test, a2, a1) + np.bincount(test['col'], weights=ver, minlength=test['n'])
            emit('M2', 0, ntr, nnztr, pc2, a2, '', 'per-feature x depth presence + rate-context values')
            # ---- mult1
            bb = fit_mult1(tr, val)
            q = pooled_profile(tr, bb)
            emit('mult1', 1, ntr, nnztr, mult_cost_cells(test, np.log2(q)[test['indices']]), '', bb)
            # ---- Dirichlet-multinomial around the pooled profile (adds burstiness / overdispersion)
            bs = None
            for sc_ in (30, 100, 300, 1000, 3000, 1e4, 1e5, 1e7):
                cv = dm_cost_cells(val, q, sc_).sum()
                if bs is None or cv < bs[0]:
                    bs = (cv, sc_)
            emit('dm1', 1, ntr, nnztr, dm_cost_cells(test, q, bs[1]), bs[1], bb)
            # ---- mixtures
            for C in CS:
                if ntr < 8 * C:
                    continue
                N, Nc, w, qq = em_mixture(tr, C, bb, seed=C)
                bestv = None
                for al in (10, 30, 100, 300, 1000, 3000, 10000, 1e5, 1e6):
                    cv = mixture_cells_cost(val, N, Nc, w, qq, al).sum()
                    if bestv is None or cv < bestv[0]:
                        bestv = (cv, al)
                emit(f'mult{C}', C, ntr, nnztr, mixture_cells_cost(test, N, Nc, w, qq, bestv[1]), bestv[1], bb)
            print(f'  ntr={ntr:6d} done  ({time.time() - t0:.0f}s)', flush=True)
        # side info
        # ---- asymptote fits with bootstrap over test cells
        models = sorted({k[0] for k in percell if k[0] != 'comb'})
        brng = np.random.default_rng(7)
        boots = [brng.integers(0, test['n'], size=test['n']) for _ in range(200)]
        for model in models:
            sizes = sorted(k[1] for k in percell if k[0] == model)
            if len(sizes) < 4:
                continue
            nzc = test['nnzc'].astype(np.float64)
            Ls = [percell[(model, n_)].sum() / nz_test for n_ in sizes]
            p_free = fit_asym(sizes, Ls)
            p05 = fit_asym(sizes, Ls, 0.5)
            p1 = fit_asym(sizes, Ls, 1.0)
            Lin = []
            for bi in boots:
                Lb = [percell[(model, n_)][bi].sum() / nzc[bi].sum() for n_ in sizes]
                pb = fit_asym(sizes, Lb)
                Lin.append(pb[0])
            Lin = np.array(Lin)
            Lin = Lin[np.isfinite(Lin)]
            lo, hi = (np.percentile(Lin, 2.5), np.percentile(Lin, 97.5)) if len(Lin) > 20 else (float('nan'), float('nan'))
            fa.write(f'{s}\t{mem}\t{model}\t{len(poolidx)}\t{Ls[-1]:.4f}\t{p_free[0]:.4f}\t{lo:.4f}\t{hi:.4f}\t{p05[0]:.4f}\t{p1[0]:.4f}\t{p_free[2]:.3f}\t{p_free[1]:.3f}\t\n')
            fa.flush()
        print(f'{key} finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
