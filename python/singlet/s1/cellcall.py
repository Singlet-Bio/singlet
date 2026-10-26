# SPDX-License-Identifier: MIT
"""``called_ed``: an EmptyDrops cell call from the counts a sample stores (docs/called-ed.md).

The stored ``called`` flag is STARsolo's default ``CellRanger2.2 3000 0.99 10`` (one UMI threshold), which misses low-RNA cells.
``called_ed`` keeps every ``called`` barcode and adds the uncalled barcodes whose expression profile rejects "ambient RNA only"
(EmptyDrops, Lun et al. 2019, Genome Biology 20:63, as in DropletUtils):

* counts: exon + intron, aggregated to genes (gene = feature name before the first ``_``);
* ambient pool: uncalled barcodes with 0 < total <= ``lower``; ``lower`` = 3 x the mode of log10(total) among barcodes below
  half the 5th percentile of the called totals, clipped to [5, 100] (the plateau of the barcode-rank curve);
* tested: uncalled barcodes with total > max(``lower``, 50); barcodes at or above the knee get p = 0 (``retain``); the knee is
  fitted within DropletUtils' curve bounds (``.find_curve_bounds``), as ``barcodeRanks`` does;
* ambient profile: Good-Turing proportions of the pool's summed counts; Dirichlet-multinomial concentration ``alpha`` by maximum
  likelihood over the pool; Monte Carlo p-values from 10,000 Polya-urn paths (one path per draw, reseeded per draw, so the result
  does not depend on threads); a barcode below every simulated path gets p = min(Phi(z), 1 / (niters + 1)), z from the mean and
  sd of the simulated log-probabilities at its total; Benjamini-Hochberg over the tested barcodes; flag at FDR <= 0.001.

The Monte Carlo loop needs numba (``pip install singlet-bio[migrate]``); everything else is numpy/scipy. Validated on 65 samples
against the authors' Cell Ranger calls and hashtags (docs/called-ed.md) and against ground truth on six benchmark datasets
(docs/called-ed-truth.md, reference bench/called_ed_truth/common.py). Golden vectors shared with the C++ port
(include/singlet/pileup/called_ed.h): tests/data/called_ed_golden/.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

METHOD = "singlet-emptydrops"
VERSION = "2.1"  # 2.1: knee within the DropletUtils curve bounds; tail p-values below every simulated path
DEFAULTS = {
    "lower_rule": "plateau",
    "test_min": 50,
    "fdr": 0.001,
    "retain": "knee",
    "niters": 10000,
    "seed": 20261023,
}
MAX_SIMULATED_TOTAL = 50000  # tested totals above this are not simulated: p = 1 / (niters + 1)
COLUMNS = ("called_ed", "called_ed_fdr")


class CellCallUnavailable(RuntimeError):
    """numba is not installed."""


def protocol_family(protocol: Optional[str]) -> tuple:
    """(family, validated). Combinatorial indexing (Parse, SPLiT-seq) has no ambient pool among the stored barcodes: the stored
    call is kept. inDrop and Drop-seq are computed but unvalidated (no ground truth; one inDrop and the one Drop-seq sample in
    the validation added ambient-like barcodes)."""
    p = (protocol or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    if "parse" in p or "splitseq" in p or "evercode" in p:
        return "combinatorial", False
    if "indrop" in p:
        return "indrop", False
    if "dropseq" in p:
        return "dropseq", False
    return "droplet", True


def gene_key(name: str) -> str:
    return name.split("_", 1)[0]


# ----------------------------------------------------------------------------------------------------------- inputs
def gene_counts(exon, intron, barcodes: list):
    """genes x barcodes scipy CSC of exon + intron counts aggregated to genes. ``exon`` / ``intron`` are ``matrix.Csc`` (or None);
    a matrix whose columns differ from ``barcodes`` is aligned by name (missing = 0)."""
    import scipy.sparse as sp

    mats = [m for m in (exon, intron) if m is not None and m.n > 0 and m.rownames is not None]
    genes = sorted({gene_key(r) for m in mats for r in m.rownames})
    gi = {g: i for i, g in enumerate(genes)}
    n = len(barcodes)
    X = sp.csc_matrix((len(genes), n), dtype=np.int64)
    for m in mats:
        rows = np.fromiter((gi[gene_key(r)] for r in m.rownames), dtype=np.int64, count=m.m)
        M = sp.csc_matrix(
            (m.data.astype(np.int64), m.indices.astype(np.int64), m.indptr.astype(np.int64)),
            shape=(m.m, m.n),
        )
        if m.colnames is not None and list(m.colnames) != list(barcodes):
            pos = {b: i for i, b in enumerate(m.colnames)}
            src = np.array([pos.get(b, -1) for b in barcodes], dtype=np.int64)
            ok = src >= 0
            sel = sp.csc_matrix(
                (np.ones(int(ok.sum()), np.int64), (src[ok], np.nonzero(ok)[0])), shape=(m.n, n)
            )
            M = (M @ sel).tocsc()
        agg = sp.csr_matrix(
            (np.ones(m.m, np.int64), (rows, np.arange(m.m))), shape=(len(genes), m.m)
        )
        X = X + (agg @ M).tocsc()
    X = X.tocsc()
    X.eliminate_zeros()
    X.sort_indices()
    return X


# ----------------------------------------------------------------------------------------------------------- ambient model
def good_turing_proportions(counts: np.ndarray) -> np.ndarray:
    """Simple Good-Turing (Gale & Sampson 1995) smoothed proportions, as ``edgeR::goodTuringProportions`` (zeros share the
    unseen mass equally)."""
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.sum()
    if n == 0:
        return np.full(len(counts), 1.0 / len(counts))
    r, nr = np.unique(counts[counts > 0], return_counts=True)
    n0 = int((counts == 0).sum())
    p0 = nr[0] / n if r[0] == 1 else 0.0
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
            sd = 1.96 * math.sqrt(
                (r[i] + 1) ** 2 * nr[i + 1] / nr[i] ** 2 * (1 + nr[i + 1] / nr[i])
            )
            if abs(x - y) > sd:
                rstar[i] = x
                continue
        use_lgt = True
        rstar[i] = y
    prop_r = (1 - p0) * rstar / (nr * rstar).sum()
    lookup = dict(zip(r.tolist(), prop_r.tolist()))
    out = np.empty(len(counts))
    pos = counts > 0
    out[pos] = [lookup[c] for c in counts[pos]]
    out[~pos] = p0 / n0 if n0 else 0.0
    return out / out.sum()


def dm_alpha_mle(Y, prop: np.ndarray) -> float:
    """MLE of the Dirichlet-multinomial concentration over the ambient barcodes (columns of ``Y``), as DropletUtils."""
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    Y = Y.tocsc()
    t = np.asarray(Y.sum(axis=0)).ravel()
    keep = t > 0
    Y = Y[:, keep]
    t = t[keep]
    y = Y.data.astype(np.float64)
    pr = prop[Y.indices]

    def nll(la):
        a = math.exp(la)
        return -(
            len(t) * gammaln(a)
            - gammaln(t + a).sum()
            + (gammaln(y + a * pr) - gammaln(a * pr)).sum()
        )

    res = minimize_scalar(
        nll, bounds=(math.log(1e-2), math.log(1e7)), method="bounded", options={"xatol": 1e-4}
    )
    return float(math.exp(res.x))


def dm_logprob(Y, prop: np.ndarray, alpha: float) -> np.ndarray:
    """Full DM log-probability of every column (multinomial coefficient included)."""
    from scipy.special import gammaln

    Y = Y.tocsc()
    t = np.asarray(Y.sum(axis=0)).ravel().astype(np.float64)
    ap = alpha * prop[Y.indices]
    y = Y.data.astype(np.float64)
    per = gammaln(y + ap) - gammaln(ap) - gammaln(y + 1)
    col = np.repeat(np.arange(Y.shape[1]), np.diff(Y.indptr))
    s = np.bincount(col, weights=per, minlength=Y.shape[1])
    return gammaln(t + 1) + gammaln(alpha) - gammaln(t + alpha) + s


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
        lg = large.pop()
        prob[s] = q[s]
        alias[s] = lg
        q[lg] = q[lg] + q[s] - 1.0
        (small if q[lg] < 1.0 else large).append(lg)
    for i in large + small:
        prob[i] = 1.0
    return prob, alias


_MC = None


def _mc_kernel():
    """The Polya-urn kernel, compiled on first use."""
    global _MC
    if _MC is not None:
        return _MC
    try:
        import numba as nb
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise CellCallUnavailable(
            "called_ed needs numba (pip install 'singlet-bio[migrate]')"
        ) from e

    @nb.njit(nogil=True)
    def _draw(prob, alias, K):
        i = np.random.randint(0, K)
        return i if np.random.random() < prob[i] else alias[i]

    @nb.njit(parallel=True)
    def _mc_paths(prob, alias, prop, alpha, totals_sorted, niters, seed):
        """For each draw one Polya-urn DM path to the largest total; the log-probability at every requested total."""
        K = len(prop)
        T = len(totals_sorted)
        tmax = totals_sorted[T - 1]
        out = np.empty((niters, T), np.float64)
        for it in nb.prange(niters):
            np.random.seed(seed + it)
            counts = np.zeros(K, np.int32)
            draws = np.empty(tmax, np.int64)
            lp = 0.0
            k = 0
            for n in range(tmax):
                if np.random.random() < alpha / (alpha + n):
                    g = _draw(prob, alias, K)
                else:
                    g = draws[np.random.randint(0, n)]
                c = counts[g]
                lp += (
                    math.log(n + 1)
                    - math.log(n + alpha)
                    + math.log(c + alpha * prop[g])
                    - math.log(c + 1)
                )
                counts[g] = c + 1
                draws[n] = g
                while k < T and totals_sorted[k] == n + 1:
                    out[it, k] = lp
                    k += 1
        return out

    _MC = _mc_paths
    return _MC


def available() -> bool:
    try:
        _mc_kernel()
        return True
    except CellCallUnavailable:
        return False


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p, kind="stable")
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# ----------------------------------------------------------------------------------------------------------- bounds
def _rank_curve(totals: np.ndarray, lower: float):
    """Unique totals > ``lower`` (descending) and their average ranks among the nonzero totals (ties share the mean rank)."""
    t = np.sort(totals[totals > 0])[::-1]
    u = np.unique(t)[::-1]  # descending unique totals
    first = np.searchsorted(-t, -u, side="left")
    last = np.searchsorted(-t, -u, side="right") - 1
    rr = (first + last) / 2 + 1
    keep = u > lower
    return u[keep], rr[keep]


def _spline_knee(xs: np.ndarray, ys: np.ndarray) -> float:
    """Total at the minimum signed curvature of a cubic smoothing spline through (log10 rank, log10 total)."""
    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.002)
    d1 = spl.derivative(1)(xs)
    d2 = spl.derivative(2)(xs)
    return float(10 ** ys[np.argmin(d2 / (1 + d1**2) ** 1.5)])


def curve_bounds(totals: np.ndarray, lower: float, exclude_from: int = 50) -> tuple:
    """Indices ``(left, right)`` into ``_rank_curve(totals, lower)`` of DropletUtils' fit bounds (``.find_curve_bounds``):
    on the (log10 rank, log10 total) points, ``right`` is the steepest drop and ``left`` the largest slope before it, both after
    the points with log10 rank <= log10(``exclude_from``). ``None`` with fewer than 10 points."""
    u, rr = _rank_curve(np.asarray(totals), lower)
    if len(u) < 10:
        return None
    x, y = np.log10(rr), np.log10(u)
    d1n = np.diff(y) / np.diff(x)
    skip = min(len(d1n) - 1, int((x <= math.log10(exclude_from)).sum()))
    d = d1n[skip:]
    right = int(np.argmin(d))
    left = int(np.argmax(d[: right + 1]))
    return left + skip, right + skip


def knee(totals: np.ndarray, lower: float, exclude_from: int = 50) -> float:
    """DropletUtils ``barcodeRanks`` knee: maximum curvature of log10 total vs log10 rank (average rank of ties), fitted on the
    totals > ``lower`` between the curve bounds (``curve_bounds``) only. Fewer than 6 points between the bounds: the total at
    the left bound. Without the bounds the fit reaches into the ambient shoulder when ``lower`` is small, and the knee can land
    just above ``lower`` (docs/called-ed-truth.md). Reference: bench/called_ed_truth/common.py ``knee_bounded``."""
    totals = np.asarray(totals)
    b = curve_bounds(totals, lower, exclude_from)
    if b is None:
        return float("nan")
    u, rr = _rank_curve(totals, lower)
    x, y = np.log10(rr), np.log10(u)
    xs, ys = x[b[0] : b[1] + 1], y[b[0] : b[1] + 1]
    try:
        return _spline_knee(xs, ys) if len(xs) >= 6 else float(10 ** ys[0])
    except Exception:
        return float("nan")


def plateau_lower(tot: np.ndarray, called: np.ndarray) -> int:
    """Upper total of the ambient pool: 3 x the mode of log10(total) among barcodes below half the 5th percentile of the called
    totals, clipped to [5, 100]."""
    q05 = float(np.quantile(tot[called], 0.05)) if called.any() else 200.0
    t = tot[(tot > 0) & (tot < max(4.0, 0.5 * q05))]
    if len(t) < 100:
        return int(np.clip(0.5 * q05, 5, 100))
    lt = np.log10(t)
    h, e = np.histogram(lt, bins=np.arange(0, lt.max() + 0.1, 0.1))
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    j = int(np.argmax(h))
    mode = 10 ** (0.5 * (e[j] + e[j + 1]))
    return int(np.clip(round(3 * mode), 5, 100))


# ----------------------------------------------------------------------------------------------------------- the call
def call(X, called: np.ndarray, protocol: Optional[str] = None, **over) -> tuple:
    """``called_ed`` from a genes x barcodes scipy CSC of counts and the stored ``called`` flags.

    Returns ``(called_ed bool[n], fdr float32[n] (NaN = not tested), params dict)``. ``params`` records everything needed to
    reproduce the call, including the fitted ``alpha`` and the ambient pool; ``params["validated"]`` is False for protocols
    without ground truth (inDrop, Drop-seq) and ``params["method"] == "stored-call"`` where the stored call is kept."""
    cv2, pv, q, params = call_detail(X, called, protocol, **over)
    return cv2, q.astype(np.float32), params


def call_detail(X, called: np.ndarray, protocol: Optional[str] = None, **over) -> tuple:
    """``call`` with the per-barcode p-values: ``(called_ed bool[n], pvalue float64[n], fdr float64[n], params)``; p-value and
    FDR are NaN where a barcode is not tested. Used by the golden vectors (tests/data/called_ed_golden/)."""
    P = {**DEFAULTS, **over}
    called = np.asarray(called, bool)
    n = len(called)
    fam, validated = protocol_family(protocol)
    params = {
        "method": METHOD,
        "version": VERSION,
        "counts": "exon+intron",
        "protocol": protocol or "",
        "protocol_family": fam,
        "validated": validated,
        **P,
    }
    fdr = np.full(n, np.nan)
    pvalue = np.full(n, np.nan)
    if fam == "combinatorial" or n == 0:
        params["method"] = "stored-call"
        params["n_called_ed"] = int(called.sum())
        return called.copy(), pvalue, fdr, params
    X = X.tocsc()
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    lower = plateau_lower(tot, called) if P["lower_rule"] == "plateau" else int(P["lower_rule"])
    ambient = (tot > 0) & (tot <= lower) & ~called
    test = (tot > max(lower, int(P["test_min"]))) & ~called
    kn = knee(tot, lower) if P["retain"] == "knee" else float("nan")
    retain = float(kn) if np.isfinite(kn) and kn > lower else None
    params.update(
        {
            "lower": lower,
            "retain_total": retain,
            "n_ambient": int(ambient.sum()),
            "ambient_umi": int(tot[ambient].sum()),
            "n_tested": int(test.sum()),
        }
    )
    if ambient.sum() < 10 or tot[ambient].sum() < 100 or not test.any():
        params["error"] = "ambient pool too small" if test.any() else "nothing to test"
        params["n_called_ed"] = int(called.sum())
        return called.copy(), pvalue, fdr, params
    mc = _mc_kernel()
    asum = np.asarray(X[:, ambient].sum(axis=1)).ravel().astype(np.int64)
    prop = good_turing_proportions(asum)
    alpha = dm_alpha_mle(X[:, ambient], prop)
    params["alpha"] = alpha
    ti = np.nonzero(test)[0]
    tt = tot[ti]
    niters = int(P["niters"])
    pv = np.full(len(ti), 1.0 / (niters + 1))
    sim = tt <= MAX_SIMULATED_TOTAL
    if retain is not None:
        pv[tt >= retain] = 0.0
        sim &= tt < retain
    n_tail = 0
    if sim.any():
        lp = dm_logprob(X[:, ti[sim]], prop, alpha)
        ut = np.unique(tt[sim])
        prob, alias = alias_table(prop)
        sims = mc(prob, alias, prop, float(alpha), ut.astype(np.int64), niters, int(P["seed"]))
        sims.sort(axis=0)
        col = np.searchsorted(ut, tt[sim])
        n_le = np.fromiter(
            (np.searchsorted(sims[:, c], o, side="right") for c, o in zip(col, lp)),
            dtype=np.int64,
            count=len(lp),
        )
        ps = (n_le + 1) / (niters + 1)
        # below every simulated path: extrapolate the null tail with a normal approximation of the simulated log-probabilities
        # at that total (mean, population sd), capped at the Monte Carlo floor 1 / (niters + 1). Without it BH can pass nothing
        # when fewer than n_tested * fdr * (niters + 1) barcodes sit at the floor (docs/called-ed-truth.md).
        floor = n_le == 0
        if floor.any():
            from scipy.stats import norm

            mu, sd = sims.mean(axis=0), sims.std(axis=0)
            z = (lp[floor] - mu[col[floor]]) / np.maximum(sd[col[floor]], 1e-12)
            ps[floor] = np.minimum(norm.cdf(z), 1.0 / (niters + 1))
            n_tail = int(floor.sum())
        pv[sim] = ps
    params["n_tail"] = n_tail
    q = bh(pv)
    pvalue[ti] = pv
    fdr[ti] = q
    cv2 = called.copy()
    cv2[ti[q <= P["fdr"]]] = True
    params["n_called_ed"] = int(cv2.sum())
    params["n_added"] = int((cv2 & ~called).sum())
    return cv2, pvalue, fdr, params
