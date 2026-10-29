#!/usr/bin/env python
"""Track D, task 1: entropy estimates per .1pz file.

For each corpus matrix and each subset of columns (all / called cells / empty droplets) this writes
long-format rows (sample, member, subset, quantity, bits, bits_per_nnz, bytes, note) to
results/track_d_entropy.tsv, and the empty-droplet goodness-of-fit tests to results/track_d_gof.tsv.

    python -P tools/track_d_entropy.py [sample[:member] ...]

Honesty notes (also repeated in notes/track_d.md):
  *_plugin  : in-sample empirical entropy; optimistic (ignores the cost of learning the parameters).
  *_cv      : 2-fold cross-validated code length (train on interleaved half of the columns, score the other
              half, sum both); pessimistic (the model has seen only half the data). One or two smoothing
              scalars are tuned on the scored fold from a 9-point grid; with 10^5-10^6 parameters this is a
              negligible optimism but it is not zero.
  amb_*     : multinomial ambient floor, see ambient_inputs() in track_d_lib.py.
"""
import os, sys, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import corpus_files, load, ROOT
from track_d_lib import *

OUT = os.path.join(ROOT, 'results', 'track_d_entropy.tsv')
GOF = os.path.join(ROOT, 'results', 'track_d_gof.tsv')
rows = []
gof_rows = []


def add(sample, member, subset, q, bits, nnz, note=''):
    rows.append((sample, member, subset, q, float(bits), float(bits) / max(nnz, 1), float(bits) / 8.0, int(nnz), note))


def folds(colmask, n_total):
    idx = np.flatnonzero(colmask)
    a = np.zeros(n_total, dtype=bool)
    b = np.zeros(n_total, dtype=bool)
    a[idx[0::2]] = True
    b[idx[1::2]] = True
    return a, b


def cv_models(d, col, colmask, nb=16):
    """2-fold CV code lengths for the factorised models. Returns dict of bits."""
    n = int(d['n'])
    A, B = folds(colmask, n)
    out = dict(feat=0.0, featdepth=0.0, val0=0.0, valrate=0.0)
    alph = {}
    for tr_m, te_m in ((A, B), (B, A)):
        tr = subcsc(d, tr_m, col)
        te = subcsc(d, te_m, col)
        if te['indices'].size == 0 or tr['indices'].size == 0:
            continue
        b1, a1 = presence_cost_feature(tr, te)
        b2, a2 = presence_cost_feature_depth(tr, te, nb, alpha1=a1)
        out['feat'] += b1
        out['featdepth'] += b2
        alph.setdefault('feat', []).append(a1)
        alph.setdefault('featdepth', []).append(a2)
        out['val0'] += value_cost(tr, te, lambda s: np.zeros(s['indices'].size, dtype=np.int64))
        out['valrate'] += value_cost(tr, te, rate_ctx_factory(tr))
    out['alpha'] = alph
    return out


def process(sample, member):
    t0 = time.time()
    d = load(sample, member)
    m = int(d['m'])
    n = int(d['n'])
    ic = d['is_cell']
    col = columns_of(d)
    cc = np.diff(d['indptr'])
    k_col = np.bincount(col, weights=d['data'], minlength=n).astype(np.int64)
    m_eff = int((np.bincount(d['indices'], minlength=m) > 0).sum())
    subsets = {'all': np.ones(n, bool), 'cells': ic.copy(), 'empties': ~ic}
    nnz_of = {s: int(cc[mk].sum()) for s, mk in subsets.items()}

    # ---- writer symbol streams, order-0 entropies
    w = writer_streams(d, col)
    gap = w['gap']
    gcol = w['gap_col']
    gval = w['gap_val']
    for s, mk in subsets.items():
        sel = mk[gcol]
        nz = nnz_of[s]
        if nz == 0:
            continue
        g = gap[sel]
        v = gval[sel]
        add(sample, member, s, 'w_H0_gap_pooled', h0_symbols(g), nz, 'order-0 of all gap symbols, one table')
        add(sample, member, s, 'w_H0_gap_pooled_mdl', h0_symbols(g, True), nz, '+0.5 log2(n) per used symbol')
        one = (v == 1)
        add(sample, member, s, 'w_H0_gap_by_valclass', h0_symbols(g[one]) + h0_symbols(g[~one]), nz,
            'separate tables for value==1 groups and value>1 groups')
        add(sample, member, s, 'w_H0_gap_by_valclass_mdl', h0_symbols(g[one], True) + h0_symbols(g[~one], True), nz, '')
        gs = mk[w['g_col']]
        ngc = np.bincount(w['g_col'][gs], minlength=n)[mk]
        meta = h0_symbols(ngc) + h0_symbols(w['g_val'][gs]) + h0_symbols(w['g_cnt'][gs])
        add(sample, member, s, 'w_H0_meta', meta, nz,
            'order-0: groups-per-column + value + run-count symbols (separate tables)')
        add(sample, member, s, 'w_H0_total', h0_symbols(g[one]) + h0_symbols(g[~one]) + meta, nz, 'by-valclass gaps + meta')
        add(sample, member, s, 'w_H0_total_mdl', h0_symbols(g[one], True) + h0_symbols(g[~one], True) + meta, nz,
            'with MDL param cost on gaps')

    # ---- combinatorial and side-information quantities
    for s, mk in subsets.items():
        nz = nnz_of[s]
        if nz == 0:
            continue
        nj = cc[mk].astype(np.float64)
        kj = k_col[mk].astype(np.float64)
        pos = nj > 0
        add(sample, member, s, 'comb_presence_m', float(log2_binom(m, nj[pos]).sum()), nz,
            'sum_j log2 C(m, nnz_j), m=all rows')
        add(sample, member, s, 'comb_presence_meff', float(log2_binom(m_eff, nj[pos]).sum()), nz,
            f'sum_j log2 C(m_eff, nnz_j), m_eff={m_eff} rows with >=1 nnz')
        comp = np.where(pos, log2_binom(np.maximum(kj - 1, 0), np.maximum(nj - 1, 0)), 0.0)
        add(sample, member, s, 'comb_composition', float(comp.sum()), nz,
            'sum_j log2 C(k_j-1, nnz_j-1): ways to split k_j UMIs over nnz_j features (uniform)')
        add(sample, member, s, 'side_H0_nnzj', h0_symbols(cc[mk]), nz, 'order-0 of per-column nnz')
        add(sample, member, s, 'side_H0_kj', h0_symbols(k_col[mk]), nz, 'order-0 of per-column UMI total')
        add(sample, member, s, 'side_H0_kj_mdl', h0_symbols(k_col[mk], True), nz, 'with MDL param cost')

    # ---- factorised models: plug-in (optimistic) and 2-fold CV (pessimistic)
    for s in ('cells', 'empties', 'all'):
        mk = subsets[s]
        nz = nnz_of[s]
        if nz < 1000:
            continue
        sub = subcsc(d, mk, col)
        add(sample, member, s, 'pres_feat_plugin', plugin_feature_bits(sub), nz, 'sum_f N h(p_f), in-sample')
        add(sample, member, s, 'pres_featdepth_plugin', plugin_feature_bits(sub, 16), nz,
            'sum_{f,b} N_b h(p_fb), 16 depth buckets, in-sample')
        vv = h0_symbols(np.minimum(sub['data'], 4096))
        add(sample, member, s, 'val_H0_plugin', vv, nz, 'order-0 of values (capped 4096)')
        cv = cv_models(d, col, mk)
        add(sample, member, s, 'pres_feat_cv', cv['feat'], nz, f"alpha={cv['alpha'].get('feat')}")
        add(sample, member, s, 'pres_featdepth_cv', cv['featdepth'], nz, f"alpha={cv['alpha'].get('featdepth')}")
        add(sample, member, s, 'val_global_cv', cv['val0'], nz, f'order-0 value table, alphabet 1..{VMAX}+')
        add(sample, member, s, 'val_ratectx_cv', cv['valrate'], nz, 'value table by context log2(k_j*q_f)')
        add(sample, member, s, 'M1_feat_cv_total', cv['feat'] + cv['val0'], nz, 'per-feature presence + order-0 values, CV')
        add(sample, member, s, 'M2_featdepth_cv_total', cv['featdepth'] + cv['valrate'], nz,
            'per-feature x depth presence + rate-context values, CV')

    # ---- ambient floor on the empty droplets
    ambient_floor(sample, member, d, col, m, nnz_of['empties'])
    print(f'{sample:22s}{member:7s} done in {time.time()-t0:5.1f}s', flush=True)


def ambient_floor(sample, member, d, col, m, nz_emp, R=3, seed=12345):
    emp, Cc, zero_cols = ambient_inputs(d, col)
    if emp['n'] == 0 or nz_emp == 0:
        return
    f, x, ec, k = emp['indices'], emp['data'], emp['col'], emp['k']
    T = np.bincount(f, weights=x, minlength=m).astype(np.float64)
    N = T.sum()
    # choose delta, gamma by minimum LOO cost
    best = (1e300, None, None)
    for delta in (0.01, 0.1):
        r = prior_profile(Cc, m, delta)
        for g in (1e1, 1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7):
            g_ = g * (N / 5e6)          # scale with the amount of ambient data
            c = droplet_cost_bits(f, x, ec, k, loo_logp(f, x, ec, k, T, N, r, g_)).sum()
            if c < best[0]:
                best = (c, delta, g_)
    loo_bits, delta, gamma = best
    r = prior_profile(Cc, m, delta)
    plug_bits = droplet_cost_bits(f, x, ec, k, plugin_logp(f, T, N, r, gamma)).sum()
    plug0_bits = droplet_cost_bits(f, x, ec, k, plugin_logp(f, T, N, r, 1e-3)).sum()
    add(sample, member, 'empties', 'amb_loo_bits', loo_bits, nz_emp,
        f'leave-one-droplet-out multinomial, prior=cells profile delta={delta} gamma={gamma:.3g}')
    add(sample, member, 'empties', 'amb_plugin_bits', plug_bits, nz_emp, 'same smoothing, in-sample p (optimistic)')
    add(sample, member, 'empties', 'amb_plugin_unsmoothed_bits', plug0_bits, nz_emp, 'in-sample p, no prior (most optimistic)')
    # expected entropy of Mult(k_j, p*) by Monte Carlo at the smoothed in-sample profile
    pstar = (T + gamma * r) / (N + gamma)
    lp = np.log2(pstar)
    rng = np.random.default_rng(seed)
    mc = []
    syn = []
    for rep in range(R):
        sf, sx, sc = synth_droplets(rng, pstar, k)
        mc.append(droplet_cost_bits(sf, sx, sc, k, lp[sf]).sum())
        syn.append((sf, sx, sc))
    add(sample, member, 'empties', 'amb_entropy_mc_bits', np.mean(mc), nz_emp,
        f'sum_j H(Mult(k_j,p*)) Monte Carlo R={R}, sd={np.std(mc):.0f} bits; exact only if droplets are exchangeable multinomial draws')
    phat = T / N
    lph = np.where(phat > 0, np.log2(np.maximum(phat, 1e-300)), 0.0)
    mc2 = []
    for rep in range(R):
        sf, sx, sc = synth_droplets(rng, phat, k)
        mc2.append(droplet_cost_bits(sf, sx, sc, k, lph[sf]).sum())
    add(sample, member, 'empties', 'amb_entropy_mc_mle_bits', np.mean(mc2), nz_emp,
        f'sum_j H(Mult(k_j, p_hat)) with the unsmoothed in-sample MLE profile (R={R}, sd={np.std(mc2):.0f}); lower-bound flavour')
    add(sample, member, 'empties', 'amb_side_H0_kj', h0_symbols(k), nz_emp,
        f'order-0 of k_j over {emp["n"]} nonzero empties (+{zero_cols} all-zero columns not counted)')
    gof_tests(sample, member, d, emp, Cc, T, N, delta, gamma, pstar, syn, nz_emp, m, rng, loo_bits)


KB = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 10**9)]


def kbucket(k):
    b = np.zeros(len(k), dtype=np.int64)
    for i, (lo, hi) in enumerate(KB):
        b[(k >= lo) & (k <= hi)] = i
    return b


def blabel(lo, hi):
    return f'k{lo}-{hi if hi < 10**8 else "inf"}'


def _wstats(fidx, x, dcol, kW, nW, pW, binid, J, ndrop):
    """Per-droplet statistics restricted to the well-estimated feature set W.
    fidx: feature index in W per entry; dcol: droplet index per entry (0..ndrop-1)."""
    lf = log2_fact(kW)
    sx = np.bincount(dcol, weights=log2_fact(x), minlength=ndrop)
    sp = np.bincount(dcol, weights=x * np.log2(pW[fidx]), minlength=ndrop)
    cost = -(lf - sx + sp)
    D = np.bincount(dcol, minlength=ndrop).astype(np.float64)
    rep = np.bincount(dcol, weights=(x > 1), minlength=ndrop)
    q = np.bincount(binid, weights=pW, minlength=J)
    y = np.bincount(dcol * J + binid[fidx], weights=x, minlength=ndrop * J).reshape(ndrop, J)
    nz = q > 0
    ex = kW[:, None] * q[None, nz]
    x2 = ((y[:, nz] - ex) ** 2 / ex).sum(1) / (nz.sum() - 1)     # mean is exactly 1 under the multinomial
    return np.stack([cost, D, rep, x2], axis=1)


def gof_tests(sample, member, d, emp, Cc, T, N, delta, gamma, pstar, syn, nz_emp, m, rng, loo_bits, R=5, wmin=10, J=40):
    f, x, ec, k = emp['indices'], emp['data'], emp['col'], emp['k']
    ne = emp['n']
    r = prior_profile(Cc, m, delta)

    def put(test, b, n_d, real, sm, ssd):
        z = (real - sm) / ssd if ssd > 0 else float('nan')
        gof_rows.append((sample, member, test, b, n_d, real, sm, ssd, z))

    # --- T1: code-length parametric bootstrap with the MLE profile (no smoothing in the generator);
    #         the same smoothed leave-one-out estimator is applied to real and to synthetic droplets.
    cost_real = droplet_cost_bits(f, x, ec, k, loo_logp(f, x, ec, k, T, N, r, gamma))
    phat = T / N
    synth_cost = []
    for rep in range(3):
        sf, sx, sc = synth_droplets(rng, phat, k)
        Ts = np.bincount(sf, weights=sx, minlength=m).astype(np.float64)
        synth_cost.append(droplet_cost_bits(sf, sx, sc, k, loo_logp(sf, sx, sc, k, Ts, N, r, gamma)))
    tot = [c.sum() for c in synth_cost]
    put('loo_cost_bits_total', 'all', ne, cost_real.sum(), np.mean(tot), np.std(tot, ddof=1))
    kb = kbucket(k)
    for i, (lo, hi) in enumerate(KB):
        sel = kb == i
        if sel.sum() < 20:
            continue
        cs_ = [c[sel].mean() for c in synth_cost]
        put('loo_cost_bits_per_droplet', blabel(lo, hi), int(sel.sum()), cost_real[sel].mean(), np.mean(cs_), np.std(cs_, ddof=1))

    # --- T2/T3: cross-fitted, W-restricted tests (no smoothing anywhere)
    perm = rng.permutation(ne)
    half = np.zeros(ne, bool)
    half[perm[: ne // 2]] = True
    allreal = []
    allsyn = [[] for _ in range(R)]
    allk = []
    cover_num = 0.0
    cover_den = 0.0
    for fit_mask in (half, ~half):
        test_mask = ~fit_mask
        mfit = fit_mask[ec]
        Tfit = np.bincount(f[mfit], weights=x[mfit], minlength=m).astype(np.float64)
        W = Tfit >= wmin
        nW = int(W.sum())
        if nW < 2 * J:
            continue
        widx = np.full(m, -1, dtype=np.int64)
        widx[W] = np.arange(nW)
        pW = Tfit[W] / Tfit[W].sum()
        order = np.argsort(-pW)
        cum = np.cumsum(pW[order])
        binid = np.empty(nW, dtype=np.int64)
        binid[order] = np.minimum((cum * J).astype(np.int64), J - 1)
        mt = test_mask[ec] & W[f]
        te_d = np.flatnonzero(test_mask)
        remap = np.full(ne, -1, dtype=np.int64)
        remap[te_d] = np.arange(len(te_d))
        dcol = remap[ec[mt]]
        fidx = widx[f[mt]]
        xx = x[mt]
        kW = np.bincount(dcol, weights=xx, minlength=len(te_d))
        cover_num += kW.sum()
        cover_den += k[te_d].sum()
        keep = kW >= 1
        nd = len(te_d)
        real = _wstats(fidx, xx, dcol, kW, nW, pW, binid, J, nd)[keep]
        allreal.append(real)
        allk.append(kW[keep])
        kWi = kW.astype(np.int64)
        for rep in range(R):
            sf, sx, sc = synth_droplets(rng, pW, kWi)
            st = _wstats(sf, sx.astype(np.float64), sc, kW, nW, pW, binid, J, nd)[keep]
            allsyn[rep].append(st)
    if not allreal:
        return
    real = np.concatenate(allreal)
    kW = np.concatenate(allk)
    syn_ = [np.concatenate(a) for a in allsyn]
    put('W_umi_coverage', 'all', len(kW), cover_num / cover_den, cover_num / cover_den, 0.0)
    bkt = kbucket(kW.astype(np.int64))
    labels = [(None, None, np.ones(len(kW), bool))] + [(lo, hi, bkt == i) for i, (lo, hi) in enumerate(KB)]
    for lo, hi, sel in labels:
        if sel.sum() < 20:
            continue
        lab = 'all' if lo is None else blabel(lo, hi)
        nsel = int(sel.sum())
        ksum = kW[sel].sum()
        put('W_cost_bits_per_droplet', lab, nsel, real[sel, 0].mean(), np.mean([s_[sel, 0].mean() for s_ in syn_]), np.std([s_[sel, 0].mean() for s_ in syn_], ddof=1))
        put('W_distinct_per_umi', lab, nsel, real[sel, 1].sum() / ksum, np.mean([s_[sel, 1].sum() / ksum for s_ in syn_]), np.std([s_[sel, 1].sum() / ksum for s_ in syn_], ddof=1))
        put('W_repeat_per_umi', lab, nsel, real[sel, 2].sum() / ksum, np.mean([s_[sel, 2].sum() / ksum for s_ in syn_]), np.std([s_[sel, 2].sum() / ksum for s_ in syn_], ddof=1))
        put('W_pearson_dispersion', lab, nsel, real[sel, 3].mean(), np.mean([s_[sel, 3].mean() for s_ in syn_]), np.std([s_[sel, 3].mean() for s_ in syn_], ddof=1))


def write_all():
    with open(OUT, 'w') as fo:
        fo.write('sample\tmember\tsubset\tquantity\tbits\tbits_per_nnz\tbytes\tnnz\tnote\n')
        for r_ in rows:
            fo.write('\t'.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in r_) + '\n')
    with open(GOF, 'w') as fo:
        fo.write('sample\tmember\ttest\tbucket\tn_droplets\treal\tsynth_mean\tsynth_sd\tz\n')
        for r_ in gof_rows:
            fo.write('\t'.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in r_) + '\n')


def main():
    want = sys.argv[1:]
    for s, mem, p in corpus_files():
        if want and not any(w == s or w == f'{s}:{mem}' for w in want):
            continue
        process(s, mem)
        write_all()


if __name__ == '__main__':
    main()
