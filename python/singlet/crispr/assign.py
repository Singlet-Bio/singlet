# SPDX-License-Identifier: MIT
"""Per-cell guide assignment.

Three callers on the same guides x cells UMI matrix (``M``, scipy sparse, cells = called GEX columns):

* :func:`call_ambient` (**Singlet**, the stored call): every guide's UMIs in a cell are tested against what ambient guide
  molecules alone would give that cell. The ambient profile ``p`` (share of each guide among the molecules of empty
  droplets: whitelist barcodes outside the called cells) is measured on the library itself, so abundant guides need
  more UMIs to be called than rare ones. A cell's ambient load ``b_c`` is the larger of the empty droplets' load and
  the cell's own estimate from its minor guides (never the tested one). Guide ``g`` is called in cell ``c``
  when ``x_gc >= min_umis`` and ``P(Poisson(b_c p_g) >= x_gc) < alpha / n_guides`` and ``x_gc >= min_share`` of the
  cell's guide UMIs.
* :func:`call_gmm` (Cell Ranger's protospacer calling): per guide, a two-component Gaussian mixture on log10 UMIs of the
  cells with UMIs; cells in the upper component with >= 3 UMIs are called (Cell Ranger 6+, ``min_crispr_umi`` 3).
* :func:`call_poisson_gaussian` (Replogle 2022 / Norman 2019 style): per guide, a mixture of a Poisson background and a
  Gaussian on log2(UMIs + 1) for real guide expression, fitted by EM; cells with posterior > 0.5 are called.

Each returns a boolean sparse guides x cells matrix. :func:`summarize` turns it into per-cell labels: ``n_guides``,
``guide_ids`` (``|``-joined), ``target`` and ``class`` (``unassigned`` / ``single`` / ``multiple``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def ambient_profile(ambient_umis: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    a = np.asarray(ambient_umis, dtype=np.float64) + pseudo
    return a / a.sum()


def ambient_load(totals_empty: np.ndarray, q: float = 0.95) -> float:
    """Guide UMIs an ambient-only droplet carries at most (quantile ``q`` of the empty droplets' totals, zeros
    included): ambient RNA lands in every droplet at about the same rate, cells or not."""
    t = np.asarray(totals_empty, dtype=np.float64)
    return float(np.quantile(t, q)) if len(t) else 0.0


def call_ambient(M, p: np.ndarray, b_empty: float = 0.0, min_umis: int = 3, alpha: float = 1e-3,
                 min_share: float = 0.05, top: int = 2):
    """Ambient-calibrated Poisson test per (guide, cell). ``p``: ambient profile (:func:`ambient_profile`);
    ``b_empty``: the ambient load of a droplet (:func:`ambient_load` on the empty droplets).

    A cell's ambient load is the larger of ``b_empty`` and the cell's own estimate: the UMIs of all its guides but
    the ``top`` most abundant (real guides: one, two in a doublet or a dual-guide vector), scaled by the ambient share
    of the guides counted. The cell's estimate never includes the guide being tested, so with few guides (A375: 2) a
    real second guide is not explained away by itself."""
    from scipy import sparse
    from scipy.stats import poisson

    X = sparse.csc_matrix(M, dtype=np.float64)
    G = X.shape[0]
    tot = np.asarray(X.sum(axis=0)).ravel()
    thr = alpha / max(1, G)
    rows = X.indices
    cols = np.repeat(np.arange(X.shape[1]), np.diff(X.indptr))
    x = X.data
    # rank of each entry within its cell (0 = most UMIs)
    o = np.lexsort((-x, cols))
    rank = np.empty(len(x), np.int64)
    start = np.searchsorted(cols[o], np.arange(X.shape[1]))
    rank[o] = np.arange(len(x)) - start[cols[o]]
    rest = rank >= top
    rest_umis = np.bincount(cols[rest], weights=x[rest], minlength=X.shape[1])
    rest_share = np.bincount(cols[rest], weights=p[rows[rest]], minlength=X.shape[1])
    # leave the tested guide out of the cell's estimate when it is among the rest
    own = np.where(rest, x, 0.0)
    own_p = np.where(rest, p[rows], 0.0)
    b_cell = (rest_umis[cols] - own) / np.maximum(1e-9, rest_share[cols] - own_p)
    b = np.maximum(b_empty, np.where(rest_share[cols] - own_p > 0.05, b_cell, 0.0))
    lam = b * p[rows]
    pv = poisson.sf(x - 1, np.maximum(lam, 1e-12))
    is_called = (x >= min_umis) & (pv < thr) & (x >= min_share * tot[cols])
    return sparse.csc_matrix((is_called[is_called], (rows[is_called], cols[is_called])), shape=X.shape, dtype=bool)


def call_gmm(M, min_umis: int = 3, seed: int = 0):
    """Cell Ranger-style per-guide two-component GMM on log10 UMIs."""
    from scipy import sparse
    from sklearn.mixture import GaussianMixture

    X = sparse.csr_matrix(M)
    out_r, out_c = [], []
    for g in range(X.shape[0]):
        lo, hi = X.indptr[g], X.indptr[g + 1]
        cols, v = X.indices[lo:hi], X.data[lo:hi].astype(np.float64)
        if len(v) < 2:
            if len(v) == 1 and v[0] >= min_umis:
                out_r.append([g])
                out_c.append(cols)
            continue
        lv = np.log10(v).reshape(-1, 1)
        if np.ptp(lv) == 0:
            pos = v >= min_umis
        else:
            gm = GaussianMixture(2, random_state=seed).fit(lv)
            hi_comp = int(np.argmax(gm.means_.ravel()))
            pos = (gm.predict(lv) == hi_comp) & (v >= min_umis)
        out_r.append(np.full(int(pos.sum()), g))
        out_c.append(cols[pos])
    r = np.concatenate(out_r) if out_r else np.zeros(0, int)
    c = np.concatenate(out_c) if out_c else np.zeros(0, int)
    return sparse.csc_matrix((np.ones(len(r), bool), (r, c)), shape=X.shape, dtype=bool)


def _pg_em(v: np.ndarray, n_zero: int, iters: int = 100):
    """EM for counts = pi * Poisson(lam) + (1-pi) * Normal(mu, sd) on log2(v+1). ``v`` holds the nonzero counts, zeros
    are ``n_zero`` cells (Poisson component)."""
    from scipy.stats import norm, poisson

    lv = np.log2(v + 1)
    allv = np.concatenate([np.zeros(n_zero), v])
    alllv = np.concatenate([np.zeros(n_zero), lv])
    q = np.quantile(lv, 0.9) if len(lv) else 1.0
    mu, sd = max(q, 1.0), 1.0
    lam = max(0.05, float(np.mean(allv[alllv < q])) if (alllv < q).any() else 0.1)
    pi = 0.9
    for _ in range(iters):
        a = pi * poisson.pmf(allv, lam)
        # density of counts under the lognormal-ish component, on the count scale (Jacobian of log2(v+1))
        b = (1 - pi) * norm.pdf(alllv, mu, sd) / ((allv + 1) * np.log(2))
        w = b / np.maximum(a + b, 1e-300)
        pi_n = 1 - w.mean()
        if w.sum() < 1e-9:
            break
        mu_n = float((w * alllv).sum() / w.sum())
        sd_n = float(np.sqrt(max(1e-4, (w * (alllv - mu_n) ** 2).sum() / w.sum())))
        lam_n = float(((1 - w) * allv).sum() / max(1e-9, (1 - w).sum()))
        done = abs(mu_n - mu) < 1e-6 and abs(lam_n - lam) < 1e-6
        pi, mu, sd, lam = pi_n, mu_n, sd_n, max(lam_n, 1e-3)
        if done:
            break
    return pi, lam, mu, sd


def call_poisson_gaussian(M, min_umis: int = 1):
    from scipy import sparse
    from scipy.stats import norm, poisson

    X = sparse.csr_matrix(M)
    n = X.shape[1]
    out_r, out_c = [], []
    for g in range(X.shape[0]):
        lo, hi = X.indptr[g], X.indptr[g + 1]
        cols, v = X.indices[lo:hi], X.data[lo:hi].astype(np.float64)
        if len(v) == 0:
            continue
        pi, lam, mu, sd = _pg_em(v, n - len(v))
        a = pi * poisson.pmf(v, lam)
        b = (1 - pi) * norm.pdf(np.log2(v + 1), mu, sd) / ((v + 1) * np.log(2))
        pos = (b > a) & (v >= min_umis) & (v > lam)
        out_r.append(np.full(int(pos.sum()), g))
        out_c.append(cols[pos])
    r = np.concatenate(out_r) if out_r else np.zeros(0, int)
    c = np.concatenate(out_c) if out_c else np.zeros(0, int)
    return sparse.csc_matrix((np.ones(len(r), bool), (r, c)), shape=X.shape, dtype=bool)


def summarize(called, guide_ids: Sequence[str], targets: Optional[Sequence[str]] = None):
    """Per-cell labels from a boolean guides x cells matrix -> pandas DataFrame (one row per cell)."""
    import pandas as pd
    from scipy import sparse

    C = sparse.csc_matrix(called)
    n = np.diff(C.indptr)
    ids = np.asarray(guide_ids)
    tg = np.asarray(targets) if targets is not None else None
    gl, tl = [], []
    for j in range(C.shape[1]):
        rr = C.indices[C.indptr[j]:C.indptr[j + 1]]
        gl.append("|".join(ids[rr]))
        tl.append("|".join(sorted(set(tg[rr]))) if tg is not None else "")
    cls = np.where(n == 0, "unassigned", np.where(n == 1, "single", "multiple"))
    return pd.DataFrame({"n_guides": n, "guide_ids": gl, "target": tl, "class": cls})


def agreement(ours, ref) -> Dict[str, float]:
    """Cell-level agreement of two label tables (``summarize`` output, same cells): among cells the reference calls
    single, the share we call the same single guide (recall), and among our single calls, the share the reference
    calls the same (precision); plus the assigned and multiple-guide rates of both."""
    rs = ref["class"].to_numpy() == "single"
    os_ = ours["class"].to_numpy() == "single"
    same = ours["guide_ids"].to_numpy() == ref["guide_ids"].to_numpy()
    return {
        "cells": int(len(ref)),
        "ref_single": int(rs.sum()),
        "ours_single": int(os_.sum()),
        "recall_single": float((same & rs & os_).sum() / max(1, rs.sum())),
        "precision_single": float((same & rs & os_).sum() / max(1, (os_ & (ref["class"].to_numpy() != "unassigned")).sum())),
        "ref_contains_ours": float(np.mean([g in r.split("|") for g, r, o in zip(ours["guide_ids"], ref["guide_ids"], os_) if o])
                                   if os_.any() else 0.0),
        "assigned_ours": float((ours["class"] != "unassigned").mean()),
        "assigned_ref": float((ref["class"] != "unassigned").mean()),
        "multiple_ours": float((ours["class"] == "multiple").mean()),
        "multiple_ref": float((ref["class"] == "multiple").mean()),
    }
