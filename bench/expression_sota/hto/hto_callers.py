# SPDX-License-Identifier: MIT
"""Hashtag callers and scoring for the independent-set evaluation (bench/expression_sota/hto/PREREG.md).

Every caller maps a tag x cell count matrix X (float, raw counts of the called cells) to an int array:
tag index (singlet), NEG = -1 (negative), DBL = -2 (doublet).

Callers (all implemented here from published descriptions; no code of a GPL tool was read or copied):

* ``current``       Singlet's caller at the repo default (hto_demux.h ``HtoDemuxParams``): the numpy reference in
                    bench/adt/hto_lodo.py, which the header reproduces call for call.
* ``hasheddrops``   fold-change rule of DropletUtils::hashedDrops with its documented defaults, no mixture veto.
* ``gmm``           per-tag two-component Gaussian mixture on log1p counts (posterior >= 0.5 = positive); the number of
                    positive tags gives negative / singlet / doublet (the GMM-Demux-style baseline, without its
                    droplet-level model).
* ``htodemux``      Stoeckius et al. 2018 / Seurat HTODemux: CLR over cells, k-means (k = tags + 1), the cluster with the
                    lowest mean is background, negative binomial fitted to its counts after dropping the top 0.5%, the
                    0.99 quantile is the tag's threshold.
* ``bff_raw``, ``bff_cluster``  Boggy et al. 2022, from the Methods section (see the docstrings for the places where the
                    description is ambiguous and what was chosen).
* ``bff_lodo``      the BFF-style family of PREREG section 6 (BFF raw thresholds with a ratio / margin refinement).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

NEG, DBL = -1, -2
MIN_CLASS = 1  # minimum number of truth cells for a class to enter the macro F1 (eval_independent.py sets 30)
_HERE = os.path.dirname(os.path.abspath(__file__))


def _lodo():
    p = os.path.join(_HERE, "..", "..", "adt", "hto_lodo.py")
    spec = importlib.util.spec_from_file_location("hto_lodo_ref", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _lodo()

# ------------------------------------------------------------------------------------------------ frozen defaults
CURRENT_DEFAULT = {"confident_min": 0.5, "doublet_nmads": 3.0, "doublet_min": 1.0, "veto": 0.5}  # hto_demux.h


def proportion_rule(X):
    """hto_demux.h fallback for fewer than 20 cells or one tag."""
    k, n = X.shape
    mx = X.max(axis=0)
    pos = (X >= 5.0) & (X >= 0.3 * mx)
    npos = pos.sum(axis=0)
    best = np.where(pos, X, -1).argmax(axis=0)
    return np.where(npos == 1, best, np.where(npos >= 2, DBL, NEG)).astype(np.int64)


def current(X):
    if X.shape[1] < 20 or X.shape[0] < 2:
        return proportion_rule(X)
    return L.call_hashtags(X, CURRENT_DEFAULT)


def hasheddrops(X):
    """DropletUtils::hashedDrops defaults: doublet.nmads 3, doublet.min 2, confident.nmads 3, confident.min 2, pseudo 5."""
    return L.hashed_drops(X, confident_min=2.0, doublet_nmads=3.0, doublet_min=2.0)


def _count_rule(pos):
    npos = pos.sum(axis=0)
    return np.where(npos == 1, pos.argmax(axis=0), np.where(npos >= 2, DBL, NEG)).astype(np.int64)


def gmm(X):
    q = L.mixture_posteriors(X)
    return _count_rule(q >= 0.5)


# ----------------------------------------------------------------------------------------------------- HTODemux
def _clr(X):
    """Seurat NormalizeData(normalization.method = 'CLR') with its default margin = 1 (across the tags of each cell):
    log1p(x / exp(mean(log1p(x)))) per cell, as in the HTODemux vignette."""
    return np.log1p(X / np.exp(np.log1p(X).mean(axis=0, keepdims=True)))


def _nb_fit(x):
    """Method-of-moments start then MLE of NB(mu, size)."""
    m, v = x.mean(), x.var()
    if m <= 0:
        return None
    r0 = m * m / max(v - m, 1e-3 * m) if v > m else 1e3

    def nll(p):
        r, mu = np.exp(p)
        return -stats.nbinom.logpmf(x, r, r / (r + mu)).sum()

    res = optimize.minimize(nll, np.log([max(r0, 1e-2), m]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
    r, mu = np.exp(res.x)
    return r, mu


def htodemux(X, quantile=0.99):
    """HTODemux (Stoeckius et al. 2018): cells are clustered by k-means (k = tags + 1, 50 starts, fixed seed) in the CLR space of
    all tags; for each tag the cluster with the lowest average CLR of that tag is its background, a negative binomial is fitted to
    that tag's raw counts in the background cluster after dropping the top 0.5%, and the 0.99 quantile is the threshold."""
    from sklearn.cluster import KMeans

    k, n = X.shape
    clr = _clr(X)
    km = KMeans(n_clusters=min(k + 1, n), n_init=50, random_state=0).fit(clr.T)
    lab = km.labels_
    pos = np.zeros((k, n), dtype=bool)
    for j in range(k):
        means = [clr[j][lab == m].mean() if (lab == m).any() else np.inf for m in range(km.n_clusters)]
        bg = X[j][lab == int(np.argmin(means))].astype(float)
        if len(bg) < 3:
            continue
        bg = bg[bg <= np.quantile(bg, 0.995)]
        fit = _nb_fit(bg)
        if fit is None:
            thr = 1.0
        else:
            r, mu = fit
            thr = stats.nbinom.ppf(quantile, r, r / (r + mu))
        pos[j] = X[j] > thr
    return _count_rule(pos)


# ---------------------------------------------------------------------------------------------------------- BFF
def _bw_sj(x, nb=1000):
    """Sheather-Jones solve-the-equation bandwidth (R bw.SJ(method='ste')) on a binned sample."""
    n = len(x)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return 1e-3
    cnt, edges = np.histogram(x, bins=nb, range=(lo, hi))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    d = np.abs(ctr[:, None] - ctr[None])
    w = cnt[:, None] * cnt[None]
    dens = lambda z: np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    phi4 = lambda z: (z**4 - 6 * z**2 + 3) * dens(z)
    phi6 = lambda z: (z**6 - 15 * z**4 + 45 * z**2 - 15) * dens(z)
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
    scale = min(sd, iqr) if iqr > 0 else sd
    a, b = 1.24 * scale * n ** (-1 / 7), 1.23 * scale * n ** (-1 / 9)
    SD = lambda h: ((w * phi4(d / h)).sum() - n * phi4(0)) / (n * (n - 1) * h**5)
    TD = lambda h: -((w * phi6(d / h)).sum() - n * phi6(0)) / (n * (n - 1) * h**7)
    alph2 = 1.357 * (SD(a) / TD(b)) ** (1 / 7)
    c1 = 1 / (2 * np.sqrt(np.pi) * n)
    f = lambda h: (c1 / SD(alph2 * h ** (5 / 7))) ** (1 / 5) - h
    hmax = 1.144 * scale * n ** (-1 / 5)
    lo_, hi_ = 0.1 * hmax, hmax
    for _ in range(12):
        if f(lo_) * f(hi_) <= 0:
            break
        hi_ *= 1.2
        lo_ /= 1.2
    try:
        return float(optimize.brentq(f, lo_, hi_))
    except ValueError:
        return float(hmax)


def _kde(x, grid, h):
    z = (grid[:, None] - x[None]) / h
    return np.exp(-0.5 * z * z).sum(1) / (len(x) * h * np.sqrt(2 * np.pi))


def _kde_fast(x, grid, h, nb=2000):
    """KDE of binned data (nb bins over the grid range) to keep it O(grid * bins)."""
    cnt, edges = np.histogram(x, bins=nb, range=(grid[0], grid[-1]))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    z = (grid[:, None] - ctr[None]) / h
    return (np.exp(-0.5 * z * z) * cnt[None]).sum(1) / (len(x) * h * np.sqrt(2 * np.pi))


def _local_max(y):
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def bff_threshold(c, rng, max_peaks=4, ngrid=512):
    """Count threshold of one barcode (BFF Methods): y = log10(c + 1 + U(0,1)); Gaussian KDE with the SJ bandwidth, multiplied
    up in 10% steps until the first derivative of the smoothed curve has fewer than ``max_peaks`` local maxima; the threshold is
    the density minimum between the two tallest modes. Returns the threshold on the count scale, or inf if the smoothed density
    has one mode (the tag is not bimodal; nothing is positive)."""
    y = np.log10(c + 1.0 + rng.random(len(c)))
    grid = np.linspace(y.min() - 0.05, y.max() + 0.05, ngrid)
    h0 = _bw_sj(y)
    mult = 1.0
    for _ in range(60):
        dens = _kde_fast(y, grid, h0 * mult)
        if len(_local_max(np.gradient(dens))) < max_peaks:
            break
        mult *= 1.1
    pk = _local_max(dens)
    if len(pk) < 2:
        return np.inf
    top2 = np.sort(pk[np.argsort(dens[pk])[-2:]])
    seg = dens[top2[0]:top2[1] + 1]
    xm = grid[top2[0] + int(np.argmin(seg))]
    return float(10**xm - 1.0)


def bff_raw(X, seed=0):
    rng = np.random.default_rng(seed)
    thr = np.array([bff_threshold(X[j], rng) for j in range(X.shape[0])])
    return _count_rule(X > thr[:, None]), thr


def _bqn(X, thr):
    """Bimodal quantile normalisation: the positive entries (>= thr) of every barcode are quantile-normalised across barcodes,
    and so are the negative entries, separately; the two are summed back. Quantile normalisation of unequal-length sets maps each
    value to the mean over barcodes of the quantile at its within-barcode rank."""
    out = np.zeros_like(X, dtype=float)
    for part in (True, False):
        sel = [np.where((X[j] >= thr[j]) == part)[0] for j in range(X.shape[0])]
        if any(len(s) == 0 for s in sel):
            sel_nonempty = [j for j, s in enumerate(sel) if len(s)]
        else:
            sel_nonempty = list(range(X.shape[0]))
        if not sel_nonempty:
            continue
        probs = (np.arange(1000) + 0.5) / 1000
        ref = np.mean([np.quantile(np.log10(X[j][sel[j]] + 1.0), probs) for j in sel_nonempty], axis=0)
        for j in sel_nonempty:
            v = X[j][sel[j]]
            r = stats.rankdata(v, method="average")
            out[j][sel[j]] = np.interp((r - 0.5) / len(v), probs, ref)
    return out


def bff_prepare(X, seed=0):
    """Thresholds, bimodal quantile normalisation and the hi / 2nd densities of BFFcluster (everything that does not depend on
    alpha, beta or the reading of the Tn / T2 rules); cached by callers that scan those."""
    k, n = X.shape
    rng = np.random.default_rng(seed)
    thr = np.array([bff_threshold(X[j], rng) for j in range(k)])
    thr_fin = np.where(np.isfinite(thr), thr, X.max(axis=1) + 1)
    Z = _bqn(X, thr_fin)
    o = np.argsort(-Z, axis=0, kind="stable")
    srt = np.take_along_axis(Z, o, axis=0)
    hi, sec = srt[0], srt[1] if k > 1 else np.zeros(n)
    grid = np.linspace(0, max(hi.max(), 1e-6) * 1.02, 512)
    h = _bw_sj(hi + 1e-9) if hi.std() > 0 else 0.05
    return {"o0": o[0], "hi": hi, "sec": sec, "grid": grid, "dh": _kde_fast(hi, grid, h), "d2": _kde_fast(sec, grid, h), "thr": thr}


def bff_decide(P, alpha=0.05, beta=0.05, tn_rule="edge", t2_rule="top"):
    hi, sec, grid, dh, d2 = P["hi"], P["sec"], P["grid"], P["dh"], P["d2"]
    n = len(hi)
    pk = _local_max(dh)
    peak_i = pk[np.argmax(dh[pk])] if len(pk) else int(np.argmax(dh))
    M = dh[peak_i]
    if tn_rule == "valley":
        lower = [p for p in pk if p < peak_i and dh[p] > 0.05 * M]
        if lower:
            lo_p = lower[int(np.argmax(dh[lower]))]
            Tn = grid[lo_p + int(np.argmin(dh[lo_p:peak_i + 1]))]
        else:
            tn_rule = "edge"
    if tn_rule == "edge":
        below = np.where(dh[:peak_i] <= alpha * M)[0]
        Tn = grid[below[-1]] if len(below) else grid[0]
    p2 = int(np.argmax(d2))
    if t2_rule == "valley":
        pk2 = [p for p in _local_max(d2) if p > p2 and d2[p] >= 0.03 * d2[p2]]
        if pk2:
            T2 = grid[p2 + int(np.argmin(d2[p2:pk2[0] + 1]))]
        else:
            t2_rule = "edge_own"
    if t2_rule in ("edge_M", "edge_own"):
        ref = beta * (M if t2_rule == "edge_M" else d2[p2])
        after = np.where(d2[p2:] < ref)[0]
        T2 = grid[p2 + after[0]] if len(after) else grid[-1]
    elif t2_rule == "top":
        above = np.where(d2 >= beta * M)[0]
        T2 = grid[above[-1]] if len(above) else grid[-1]
    call = np.full(n, NEG, dtype=np.int64)
    sing = (hi > Tn) & (sec < T2)
    call[sing] = P["o0"][sing]
    call[(hi > Tn) & (sec >= T2)] = DBL
    return call, {"Tn": float(Tn), "T2": float(T2)}


def bff_cluster(X, alpha=0.05, beta=0.05, seed=0, tn_rule="edge", t2_rule="top"):
    """BFFcluster (Boggy et al. 2022): thresholds as bff_raw, bimodal quantile normalisation, then per droplet the highest (hi) and
    second highest (2nd) normalised value. Singlet = hi > Tn and 2nd < T2; doublet = 2nd >= T2 (and hi > Tn); negative = hi <= Tn.
    The text defines Tn and T2 through densities relative to the peak ("max singlet") density M of hi, and does not say on which
    side of what; the readings below are implemented and the default (edge / top) is the one with the best mean class agreement with
    the authors' published BFF_cluster calls on Howitt 2023 (fidelity.py; no truth labels used). No reading reproduces the published
    doublet counts, so this is a BFF-style approximation, not the cellhashR output.

    tn_rule  edge    scanning down from the singlet peak of hi, the first value where the density of hi is <= alpha * M
             valley  density minimum between the singlet peak and a lower (negative) mode of hi
    t2_rule  edge_M  scanning up from the main mode of 2nd, the first value where the density of 2nd is < beta * M
             edge_own  same with beta * (peak of the density of 2nd)
             valley  density minimum between the main mode of 2nd and the next prominent higher mode
             top     the highest value where the density of 2nd is >= beta * M
    """
    P = bff_prepare(X, seed)
    call, info = bff_decide(P, alpha, beta, tn_rule, t2_rule)
    info["thr"] = P["thr"].tolist()
    return call, info


# ----------------------------------------------------------------------------------------------------- scoring
def classes(call, truth):
    """call: int array. truth: array of donor/line names, 'Doublet', 'Negative'. -> (call class, truth class) in {S, D, N}."""
    c = np.where(call >= 0, "S", np.where(call == DBL, "D", "N"))
    t = np.where(truth == "Doublet", "D", np.where(truth == "Negative", "N", "S"))
    return c, t


def macro_f1(call, truth, tag_to_truth=None):
    """Primary metric (PREREG section 3): macro F1 over singlet / doublet / negative. A singlet call counts as correct only if
    its tag maps to the cell's genotype donor (``tag_to_truth``: tag index -> truth name); a singlet call with the wrong donor
    is a false positive for singlet and a false negative for singlet."""
    call = np.asarray(call)
    truth = np.asarray(truth, dtype=object)
    c, t = classes(call, truth)
    ok = np.ones(len(call), dtype=bool)
    if tag_to_truth is not None:
        mx = max(tag_to_truth) + 1
        lut = np.array([tag_to_truth.get(i, "") for i in range(mx)], dtype=object)
        sing = c == "S"
        ok[sing] = (t[sing] == "S") & (lut[np.clip(call[sing], 0, mx - 1)] == truth[sing])
    f = {}
    for cl in "SDN":
        if (t == cl).sum() < MIN_CLASS:  # classes absent from the truth (or tiny: PREREG Amendment 3) are not averaged
            f[cl] = np.nan
            continue
        tp = ((c == cl) & (t == cl) & ok).sum()
        fp = ((c == cl) & ~((t == cl) & ok)).sum()
        fn = ((t == cl) & ~((c == cl) & ok)).sum()
        f[cl] = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else np.nan
    return float(np.nanmean(list(f.values()))), f


def genetic_doublets_called_singlet(call, truth):
    d = np.asarray(truth, dtype=object) == "Doublet"
    if d.sum() < MIN_CLASS:
        return float("nan")
    return float(((np.asarray(call) >= 0) & d).sum() / max(d.sum(), 1))


def bootstrap(call, truth, tag_to_truth=None, n_boot=1000, seed=1):
    """95% percentile CI of macro F1 and the doublet-called-singlet rate over resamples of cells."""
    rng = np.random.default_rng(seed)
    call, truth = np.asarray(call), np.asarray(truth, dtype=object)
    n = len(call)
    m, s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        m.append(macro_f1(call[i], truth[i], tag_to_truth)[0])
        s.append(genetic_doublets_called_singlet(call[i], truth[i]))
    return np.percentile(m, [2.5, 97.5]).tolist(), np.percentile(s, [2.5, 97.5]).tolist()


def paired_bootstrap_diff(call_a, call_b, truth, tag_to_truth=None, n_boot=1000, seed=2):
    """CI of macro-F1(a) - macro-F1(b) on the same resampled cells."""
    rng = np.random.default_rng(seed)
    a, b, t = np.asarray(call_a), np.asarray(call_b), np.asarray(truth, dtype=object)
    n = len(a)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(macro_f1(a[i], t[i], tag_to_truth)[0] - macro_f1(b[i], t[i], tag_to_truth)[0])
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]).tolist()


CALLERS = {"current": current, "hasheddrops": hasheddrops, "gmm": gmm, "htodemux": htodemux,
           "bff_raw": lambda X: bff_raw(X)[0], "bff_cluster": lambda X: bff_cluster(X)[0]}
