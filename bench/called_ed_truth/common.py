# SPDX-License-Identifier: MIT
"""Shared code for the called_ed ground-truth study (docs/called-ed-truth.md).

* ``load_run``: exon + intron gene matrix, barcodes, the stored ``called`` flag and per-barcode QC from a Singlet output folder;
* ``knee_current`` (the knee ``called_ed`` uses now: python/singlet/store/cellcall.py ``knee``, identical to the C++ port) and
  ``knee_bounded`` (the same spline restricted to DropletUtils' curve bounds: left edge = max slope before the steepest drop,
  right edge = the steepest drop, as ``DropletUtils:::.find_curve_bounds``);
* ``raw_pvalues``: EmptyDrops Monte Carlo p-values for every uncalled barcode above ``lower`` with the production model
  (Good-Turing ambient profile, DM alpha MLE, 10,000 Polya-urn paths, seed 20261023), without any retain shortcut;
* ``finish`` / ``VARIANTS``: retain, test floor and BH applied afterwards, so all variants share one Monte Carlo run.

``cellcall`` (python/singlet/store/cellcall.py from branch claude/called-v2-migration) is imported from ``$CELLCALL_DIR``.
"""

from __future__ import annotations

import gzip
import math
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.environ.get("CELLCALL_DIR", "."))
import cellcall as CC  # noqa: E402


# ----------------------------------------------------------------------------------------------------------- inputs
def _lines(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as fh:
        return [ln.rstrip("\n").split("\t")[0] for ln in fh if ln.strip()]


def read_mtx(path):
    with gzip.open(path, "rt") as fh:
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        m, n, _ = (int(v) for v in line.split())
    df = pd.read_csv(path, sep=" ", comment="%", header=None, skiprows=1, dtype=np.int64, engine="c")
    df = df.iloc[1:] if len(df) and df.iloc[0, 0] == m and df.iloc[0, 1] == n else df
    r, c, v = df[0].to_numpy() - 1, df[1].to_numpy() - 1, df[2].to_numpy()
    return sp.csc_matrix((v, (r, c)), shape=(m, n))


def load_run(d):
    """genes x barcodes exon+intron CSC aggregated to genes (gene = feature name before the first ``_``), as called_ed does."""
    mats, genes_all, sym = [], {}, {}
    bcs = None
    for kind in ("exon", "intron"):
        feats = _lines(os.path.join(d, f"{kind}_counts_features.tsv.gz"))
        b = np.array(_lines(os.path.join(d, f"{kind}_counts_barcodes.tsv.gz")))
        if bcs is None:
            bcs = b
        assert len(b) == len(bcs) and (b == bcs).all(), "exon/intron barcodes differ"
        M = read_mtx(os.path.join(d, f"{kind}_counts.mtx.gz"))
        keys = [f.split("_", 1)[0] for f in feats]
        for f, k in zip(feats, keys):
            parts = f.split("_")
            sym.setdefault(k, parts[1] if len(parts) > 1 else k)
            genes_all.setdefault(k, len(genes_all))
        mats.append((kind, M, keys))
    G = len(genes_all)
    X = sp.csc_matrix((G, len(bcs)), dtype=np.int64)
    parts = {}
    for kind, M, keys in mats:
        rows = np.array([genes_all[k] for k in keys])
        A = sp.csr_matrix((np.ones(len(keys), np.int64), (rows, np.arange(len(keys)))), shape=(G, len(keys)))
        Y = (A @ M).tocsc()
        parts[kind] = np.asarray(Y.sum(axis=0)).ravel()
        X = X + Y
    X = X.tocsc()
    X.eliminate_zeros()
    X.sort_indices()
    genes = np.array(list(genes_all))
    symbols = np.array([sym[g] for g in genes])
    cc = pd.read_csv(os.path.join(d, "cell_calls.tsv"), sep="\t", dtype={"barcode": str})
    cc = cc.set_index("barcode").reindex(bcs)
    called = cc["called"].fillna(0).astype(int).to_numpy().astype(bool)
    prod_v2 = cc["called_ed"].fillna(0).astype(int).to_numpy().astype(bool)
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    ngenes = np.diff(X.indptr)
    mt = np.array([s.upper().startswith("MT-") for s in symbols])
    mtc = np.asarray(X[mt].sum(axis=0)).ravel()
    return {
        "X": X, "barcodes": bcs, "genes": genes, "symbols": symbols, "called": called, "prod_v2": prod_v2,
        "tot": tot, "ngenes": ngenes, "pct_mt": 100.0 * mtc / np.maximum(tot, 1),
        "intron_frac": parts["intron"] / np.maximum(parts["exon"] + parts["intron"], 1),
    }


# ----------------------------------------------------------------------------------------------------------- knees
def _rank_curve(totals, lower):
    t = np.sort(totals[totals > 0])[::-1]
    u = np.unique(t)[::-1]
    first = np.searchsorted(-t, -u, side="left")
    last = np.searchsorted(-t, -u, side="right") - 1
    rr = (first + last) / 2 + 1
    keep = u > lower
    return u[keep], rr[keep]


def _spline_knee(xs, ys):
    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.002)
    d1 = spl.derivative(1)(xs)
    d2 = spl.derivative(2)(xs)
    return float(10 ** ys[np.argmin(d2 / (1 + d1**2) ** 1.5)])


def knee_current(totals, lower, exclude_from=50):
    """What called_ed 2.0 used (cellcall.knee before the fix: the spline over every total > lower). Computed here, so the
    study reproduces with the fixed cellcall (2.1, whose ``knee`` is ``knee_bounded``) as well."""
    u, rr = _rank_curve(np.asarray(totals), lower)
    if len(u) <= exclude_from + 5:
        return float("nan")
    try:
        return _spline_knee(np.log10(rr)[exclude_from:], np.log10(u)[exclude_from:])
    except Exception:
        return float("nan")


def knee_bounded(totals, lower, exclude_from=50, return_bounds=False):
    """cellcall.knee restricted to DropletUtils' fit bounds (barcodeRanks: .find_curve_bounds)."""
    u, rr = _rank_curve(np.asarray(totals), lower)
    nan = (float("nan"), float("nan"), float("nan")) if return_bounds else float("nan")
    if len(u) < 10:
        return nan
    x, y = np.log10(rr), np.log10(u)
    d1n = np.diff(y) / np.diff(x)
    skip = min(len(d1n) - 1, int((x <= math.log10(exclude_from)).sum()))
    d = d1n[skip:]
    right = int(np.argmin(d))
    left = int(np.argmax(d[: right + 1]))
    left += skip
    right += skip
    xs, ys = x[left : right + 1], y[left : right + 1]
    try:
        k = _spline_knee(xs, ys) if len(xs) >= 6 else float(10 ** ys[0])
    except Exception:
        k = float("nan")
    if return_bounds:
        return k, float(10 ** y[left]), float(10 ** y[right])
    return k


# ----------------------------------------------------------------------------------------------------------- p-values
def raw_pvalues(X, called, lower, niters=10000, seed=20261023, test_min=0):
    """Monte Carlo p-value (NaN = not tested) for every uncalled barcode with total > max(lower, test_min)."""
    X = X.tocsc()
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    ambient = (tot > 0) & (tot <= lower) & ~called
    test = (tot > max(lower, test_min)) & ~called
    asum = np.asarray(X[:, ambient].sum(axis=1)).ravel().astype(np.int64)
    prop = CC.good_turing_proportions(asum)
    alpha = CC.dm_alpha_mle(X[:, ambient], prop)
    ti = np.nonzero(test)[0]
    tt = tot[ti]
    pv = np.full(len(tot), np.nan)
    pv[ti] = 1.0 / (niters + 1)
    sim = tt <= CC.MAX_SIMULATED_TOTAL
    lp = CC.dm_logprob(X[:, ti[sim]], prop, alpha)
    ut = np.unique(tt[sim])
    prob, alias = CC.alias_table(prop)
    sims = CC._mc_kernel()(prob, alias, prop, float(alpha), ut.astype(np.int64), int(niters), int(seed))
    sims.sort(axis=0)
    col = np.searchsorted(ut, tt[sim])
    n_le = np.fromiter((np.searchsorted(sims[:, c], o, side="right") for c, o in zip(col, lp)), dtype=np.int64, count=len(lp))
    pv[ti[sim]] = (n_le + 1) / (niters + 1)
    # tail extrapolation for barcodes below every simulated path: normal approximation of the null log-probability at that
    # total (mean / sd of the 10,000 paths), capped at the Monte Carlo floor so it only refines p-values at the floor
    from scipy.stats import norm

    mu, sd = sims.mean(axis=0), sims.std(axis=0)
    z = (lp - mu[col]) / np.maximum(sd[col], 1e-12)
    pt = pv.copy()
    floor = n_le == 0
    pt[ti[sim][floor]] = np.minimum(norm.cdf(z[floor]), 1.0 / (niters + 1))
    pt[ti[~sim]] = 0.0  # above MAX_SIMULATED_TOTAL: certainly not ambient
    return pv, {"pv_tail": pt, "alpha": float(alpha), "n_ambient": int(ambient.sum()), "ambient_umi": int(tot[ambient].sum()),
                "prop": prop}


def finish(tot, called, pv, lower, fdr=0.001, retain=None, test_min=50):  # noqa: D401
    """called_ed from raw p-values: test floor, retain (p = 0 at or above), BH, union with called."""
    t = (~called) & (tot > max(lower, test_min)) & np.isfinite(pv)
    p = pv.copy()
    if retain is not None and np.isfinite(retain) and retain > lower:
        p[t & (tot >= retain)] = 0.0
    q = np.full(len(tot), np.nan)
    q[t] = CC.bh(p[t])
    return called | (np.nan_to_num(q, nan=1.0) <= fdr), q


def variant_specs(tot, called, lower):
    """name -> (fdr, retain total or None, test_min, tail-extrapolated p-values)."""
    kc = knee_current(tot, lower)
    kb, le, re = knee_bounded(tot, lower, return_bounds=True)
    k100 = knee_bounded(tot, 100)
    umin = float(tot[called].min()) if called.any() else float("nan")
    specs = {
        "prod (knee, fdr .001)": (0.001, kc, 50, False),
        "retain none, fdr .001": (0.001, None, 50, False),
        "bounded knee, fdr .001": (0.001, kb, 50, False),
        "bounded knee, fdr .01": (0.01, kb, 50, False),
        "retain none, fdr .01": (0.01, None, 50, False),
        "bounded knee, test>100": (0.001, kb, 100, False),
        "bounded knee + tail p, fdr .001": (0.001, kb, 50, True),
        "retain none + tail p, fdr .001": (0.001, None, 50, True),
        "bounded knee + tail p, fdr .01": (0.01, kb, 50, True),
    }
    info = {"knee_current": kc, "knee_bounded": kb, "bounds": [le, re], "knee_bounded_lower100": k100, "called_min_umi": umin}
    return specs, info
