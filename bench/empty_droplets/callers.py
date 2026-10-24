# SPDX-License-Identifier: MIT
"""Independent cell callers on a genes x barcodes count matrix (the stored, discovered barcodes of one sample).

* ``emptydrops``: EmptyDrops as in DropletUtils (Lun et al. 2019, Genome Biology 20:63): ambient profile = Good-Turing
  smoothed proportions of the summed counts of barcodes with total <= ``lower``; Dirichlet-multinomial concentration ``alpha``
  by maximum likelihood over those ambient barcodes; Monte Carlo p-value of each tested barcode's DM log-probability
  (``niters`` draws, one Polya-urn path per iteration recorded at every tested total); BH FDR; barcodes with total >= ``retain``
  get p = 0 (DropletUtils' ``retain``, default the knee).
* ``ordmag``: Cell Ranger 2 / STARsolo ``CellRanger2.2 3000 0.99 10``: threshold = (0.99 quantile of the top ``n_expected``
  totals) / 10. STARsolo applies this to STAR's own UMI counts; here it runs on the pileup counts stored in the bundle.
* ``cr3_like``: Cell Ranger 3 (and STARsolo ``EmptyDrops_CR``): OrdMag cells, then EmptyDrops on the next barcodes (up to 20000,
  total >= ``umi_min`` and >= ``umi_min_frac_median`` x median OrdMag total), ambient = barcodes ranked ``ind_min``..``ind_max``.
* ``barcode_ranks``: DropletUtils knee and inflection of the log-log rank plot.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


# ----------------------------------------------------------------------------------------------------------- ambient + alpha
def good_turing_proportions(counts: np.ndarray) -> np.ndarray:
    """Simple Good-Turing (Gale & Sampson 1995) smoothed proportions, as ``edgeR::goodTuringProportions`` (zeros get the
    unseen mass shared equally)."""
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.sum()
    if n == 0:
        return np.full(len(counts), 1.0 / len(counts))
    r, nr = np.unique(counts[counts > 0], return_counts=True)
    n0 = int((counts == 0).sum())
    p0 = nr[0] / n if r[0] == 1 else 0.0
    # Z_r smoothing and the log-log regression
    z = np.empty(len(r))
    for i in range(len(r)):
        q = r[i - 1] if i > 0 else 0
        t = r[i + 1] if i < len(r) - 1 else 2 * r[i] - q
        z[i] = nr[i] / (0.5 * (t - q))
    if len(r) > 1:
        b, a = np.polyfit(np.log(r), np.log(z), 1)
    else:
        b, a = -1.0, math.log(z[0])
    rstar = np.empty(len(r))
    use_lgt = False
    for i in range(len(r)):
        y = (r[i] + 1) * math.exp(a + b * math.log(r[i] + 1)) / math.exp(a + b * math.log(r[i]))
        if not use_lgt and i < len(r) - 1 and r[i + 1] == r[i] + 1:
            x = (r[i] + 1) * nr[i + 1] / nr[i]
            sd = 1.96 * math.sqrt((r[i] + 1) ** 2 * nr[i + 1] / nr[i] ** 2 * (1 + nr[i + 1] / nr[i]))
            if abs(x - y) > sd:
                rstar[i] = x
                continue
        use_lgt = True
        rstar[i] = y
    nprime = (nr * rstar).sum()
    prop_r = (1 - p0) * rstar / nprime
    out = np.empty(len(counts))
    lookup = dict(zip(r.tolist(), prop_r.tolist()))
    pos = counts > 0
    out[pos] = [lookup[c] for c in counts[pos]]
    out[~pos] = p0 / n0 if n0 else 0.0
    return out / out.sum()


def dm_alpha_mle(Y: sp.csc_matrix, prop: np.ndarray) -> float:
    """MLE of the DM concentration over the ambient barcodes (columns of ``Y``), as DropletUtils ``.estimate_alpha``."""
    Y = Y.tocsc()
    t = np.asarray(Y.sum(axis=0)).ravel()
    keep = t > 0
    Y = Y[:, keep]
    t = t[keep]
    rows = Y.indices
    y = Y.data.astype(np.float64)
    pr = prop[rows]

    def nll(la):
        a = math.exp(la)
        ll = len(t) * gammaln(a) - gammaln(t + a).sum() + (gammaln(y + a * pr) - gammaln(a * pr)).sum()
        return -ll

    res = minimize_scalar(nll, bounds=(math.log(1e-2), math.log(1e7)), method="bounded", options={"xatol": 1e-4})
    return float(math.exp(res.x))


def dm_logprob(Y: sp.csc_matrix, prop: np.ndarray, alpha: float) -> np.ndarray:
    """Full DM log-probability of every column (multinomial coefficient included)."""
    Y = Y.tocsc()
    t = np.asarray(Y.sum(axis=0)).ravel().astype(np.float64)
    ap = alpha * prop[Y.indices]
    y = Y.data.astype(np.float64)
    per = gammaln(y + ap) - gammaln(ap) - gammaln(y + 1)
    col = np.repeat(np.arange(Y.shape[1]), np.diff(Y.indptr))
    s = np.bincount(col, weights=per, minlength=Y.shape[1])
    return gammaln(t + 1) + gammaln(alpha) - gammaln(t + alpha) + s


# ----------------------------------------------------------------------------------------------------------- MC sampler
def alias_table(p: np.ndarray):
    p = np.asarray(p, dtype=np.float64)
    K = len(p)
    q = p * K
    alias = np.zeros(K, np.int64)
    prob = np.zeros(K)
    small = [i for i in range(K) if q[i] < 1.0]
    large = [i for i in range(K) if q[i] >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = q[s]
        alias[s] = l
        q[l] = q[l] + q[s] - 1.0
        (small if q[l] < 1.0 else large).append(l)
    for i in large + small:
        prob[i] = 1.0
    return prob, alias


@nb.njit(cache=True, nogil=True)
def _draw(prob, alias, K):
    i = np.random.randint(0, K)
    return i if np.random.random() < prob[i] else alias[i]


@nb.njit(parallel=True, cache=True)
def _mc_paths(prob, alias, prop, alpha, totals_sorted, niters, seed):
    """For each iteration one Polya-urn DM path to totals_sorted[-1]; returns logprob at each requested total (niters x T)."""
    K = len(prop)
    T = len(totals_sorted)
    tmax = totals_sorted[T - 1]
    out = np.empty((niters, T), np.float64)
    lg_alpha = math.lgamma(alpha)
    for it in nb.prange(niters):
        np.random.seed(seed + it)
        counts = np.zeros(K, np.int32)
        draws = np.empty(tmax, np.int64)
        lp = lg_alpha - math.lgamma(alpha)  # = 0, log P at total 0
        k = 0
        for n in range(tmax):
            if np.random.random() < alpha / (alpha + n):
                g = _draw(prob, alias, K)
            else:
                g = draws[np.random.randint(0, n)]
            c = counts[g]
            lp += math.log(n + 1) - math.log(n + alpha) + math.log(c + alpha * prop[g]) - math.log(c + 1)
            counts[g] = c + 1
            draws[n] = g
            while k < T and totals_sorted[k] == n + 1:
                out[it, k] = lp
                k += 1
    return out


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# ----------------------------------------------------------------------------------------------------------- callers
def barcode_ranks(totals: np.ndarray, lower: int = 100, exclude_from: int = 50, df: int = 20):
    """DropletUtils ``barcodeRanks``: knee (max curvature) and inflection (min first derivative) of log10 total vs log10 rank,
    on totals > lower, after dropping the top ``exclude_from`` unique points. Returns (knee, inflection) totals."""
    from scipy.interpolate import UnivariateSpline

    t = np.sort(totals[totals > 0])[::-1]
    if len(t) < 10:
        return float("nan"), float("nan")
    ranks = np.arange(1, len(t) + 1)
    u, first = np.unique(t[::-1], return_index=True)  # ascending unique totals
    # average rank of each unique total
    rr = []
    for v in u:
        idx = np.nonzero(t == v)[0]
        rr.append((idx[0] + idx[-1]) / 2 + 1)
    u = u[::-1]
    rr = np.array(rr[::-1])
    keep = u > lower
    u, rr = u[keep], rr[keep]
    if len(u) <= exclude_from + 5:
        return float("nan"), float("nan")
    x = np.log10(rr)
    y = np.log10(u)
    xs, ys = x[exclude_from:], y[exclude_from:]
    d1n = np.diff(ys) / np.diff(xs)
    infl = 10 ** ys[np.argmin(d1n)] if len(d1n) else float("nan")
    try:
        spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.002)
        d1 = spl.derivative(1)(xs)
        d2 = spl.derivative(2)(xs)
        curv = d2 / (1 + d1**2) ** 1.5
        knee = 10 ** ys[np.argmin(curv)]
    except Exception:
        knee = float("nan")
    return float(knee), float(infl)


def ordmag(totals: np.ndarray, n_expected: int = 3000, q: float = 0.99, ratio: float = 10.0):
    t = np.sort(totals)[::-1][:n_expected]
    if len(t) == 0:
        return np.zeros(len(totals), bool), float("nan")
    thr = np.quantile(t, q) / ratio  # quantile over the top n_expected (STARsolo: nUMImax at the maxPercentile position)
    thr = max(thr, 1.0)
    return totals >= thr, float(thr)


def emptydrops(Y: sp.csc_matrix, lower: float = 100, niters: int = 10000, retain=None, test_ambient=False,
               ambient_mask=None, test_mask=None, seed: int = 1, max_total: int = 50000):
    """EmptyDrops on genes x barcodes ``Y``. Returns dict with arrays over barcodes: total, tested, logprob, pvalue, fdr
    (NaN where not tested), and alpha, ambient size. ``ambient_mask``/``test_mask`` override the default pools
    (total <= lower / total > lower). Tested totals above ``max_total`` are not simulated: they get p = 1/(niters+1)."""
    Y = Y.tocsc()
    tot = np.asarray(Y.sum(axis=0)).ravel().astype(np.int64)
    amb = (tot <= lower) & (tot > 0) if ambient_mask is None else ambient_mask
    test = (tot > lower) if test_mask is None else test_mask
    res = {"total": tot, "tested": test.copy(), "n_ambient": int(amb.sum()), "ambient_umi": int(tot[amb].sum())}
    nb_ = Y.shape[1]
    res["pvalue"] = np.full(nb_, np.nan)
    res["fdr"] = np.full(nb_, np.nan)
    res["logprob"] = np.full(nb_, np.nan)
    if amb.sum() < 10 or tot[amb].sum() < 100 or test.sum() == 0:
        res["error"] = "ambient pool too small"
        return res
    asum = np.asarray(Y[:, amb].sum(axis=1)).ravel().astype(np.int64)
    prop = good_turing_proportions(asum)
    alpha = dm_alpha_mle(Y[:, amb], prop)
    res["alpha"] = alpha
    res["ambient_profile"] = prop
    ti = np.nonzero(test)[0]
    lp = dm_logprob(Y[:, ti], prop, alpha)
    res["logprob"][ti] = lp
    tt = tot[ti]
    sim_mask = tt <= max_total
    ut = np.unique(tt[sim_mask])
    pv = np.full(len(ti), 1.0 / (niters + 1))
    if len(ut):
        prob, alias = alias_table(prop)
        sims = _mc_paths(prob, alias, prop, float(alpha), ut.astype(np.int64), int(niters), int(seed))
        sims.sort(axis=0)
        col = np.searchsorted(ut, tt[sim_mask])
        obs = lp[sim_mask]
        n_le = np.array([np.searchsorted(sims[:, c], o, side="right") for c, o in zip(col, obs)])
        pv[sim_mask] = (n_le + 1) / (niters + 1)
    if retain is not None:
        pv[tt >= retain] = 0.0
    res["pvalue"][ti] = pv
    res["fdr"][ti] = bh(pv)
    return res


def cr3_like(Y: sp.csc_matrix, n_expected: int = 3000, umi_min: int = 500, umi_min_frac_median: float = 0.01,
             cand_max: int = 20000, ind_min: int = 45000, ind_max: int = 90000, fdr: float = 0.01, niters: int = 10000, seed=2):
    """Cell Ranger 3 / STARsolo EmptyDrops_CR defaults. With fewer than ``ind_min`` stored barcodes the ambient pool falls back
    to the lowest-ranked half of the barcodes beyond the candidates (recorded in ``ambient_rule``)."""
    tot = np.asarray(Y.sum(axis=0)).ravel().astype(np.int64)
    om, thr = ordmag(tot, n_expected)
    order = np.argsort(-tot, kind="stable")
    rank = np.empty(len(tot), np.int64)
    rank[order] = np.arange(len(tot))
    n_om = int(om.sum())
    med = float(np.median(tot[om])) if n_om else 0.0
    cand = (~om) & (rank < n_om + cand_max) & (tot >= umi_min) & (tot >= umi_min_frac_median * med)
    if len(tot) > ind_min + 1000:
        amb = (rank >= ind_min) & (rank < ind_max) & (tot > 0)
        rule = f"ranks {ind_min}-{ind_max}"
    else:
        start = n_om + cand_max if len(tot) > n_om + cand_max + 1000 else (len(tot) + n_om) // 2
        amb = (rank >= start) & (tot > 0)
        rule = f"fallback ranks >= {start} of {len(tot)}"
    out = {"ordmag_threshold": thr, "n_ordmag": n_om, "n_candidates": int(cand.sum()), "ambient_rule": rule}
    called = om.copy()
    if cand.sum():
        ed = emptydrops(Y, ambient_mask=amb, test_mask=cand, niters=niters, seed=seed)
        if "error" not in ed:
            called |= np.nan_to_num(ed["fdr"], nan=1.0) <= fdr
            out["alpha"] = ed["alpha"]
        else:
            out["error"] = ed["error"]
    out["called"] = called
    return out
