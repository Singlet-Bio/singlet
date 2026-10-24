# SPDX-License-Identifier: MIT
"""Derived ADT layers. Raw UMI counts are the stored truth; these are recipes a client computes on demand.

* ``clr``: Seurat's centred log-ratio across the features of each cell (``NormalizeData(method="CLR", margin=2)``):
  ``log1p(x / exp(mean(log1p(x))))``.
* ``dsb``: Mulè, Martins & Tsang (2022, Nat Commun 13:2099), ``dsb::DSBNormalizeProtein`` defaults: per protein,
  ``log(x + 10)`` of cells standardised by the mean and SD of the same in empty droplets; then each cell's technical
  component (the mean of the background population of a two-component Gaussian mixture over its proteins, together with
  the isotype controls when the panel has them: first principal component) is regressed out of every protein.

Empty droplets are columns of the sample that are not called cells and whose ADT total lies within the background
mode (see :func:`empty_droplets`), the population DSB uses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DSB_PSEUDOCOUNT = 10.0
RECIPES = {"clr": "adt-clr-v1", "dsb": "adt-dsb-v1"}


def clr(X: np.ndarray) -> np.ndarray:
    """X: features x cells (dense counts) -> CLR per cell."""
    X = np.asarray(X, dtype=np.float64)
    lg = np.log1p(X)
    gm = np.exp(lg.mean(axis=0, keepdims=True))
    return np.log1p(X / gm)


def empty_droplets(totals: np.ndarray, called: np.ndarray, min_total: int = 1, max_quantile: float = 0.99) -> np.ndarray:
    """Background droplets for DSB: uncalled columns with an ADT total in [min_total, the background mode's upper tail].

    The mode is found on log10 totals of uncalled columns; droplets far above it (unannotated cells, doublets missed by
    calling) are excluded with a cut at the ``max_quantile`` of uncalled totals and below the 5th percentile of cells."""
    t = np.asarray(totals, dtype=float)
    called = np.asarray(called, dtype=bool)
    cand = (~called) & (t >= min_total)
    if cand.sum() < 50:
        return cand
    hi = np.quantile(t[cand], max_quantile)
    if called.any():
        hi = min(hi, np.quantile(t[called], 0.05))
    return cand & (t <= hi)


def _two_gauss_low_mean(v: np.ndarray, iters: int = 100) -> float:
    """Mean of the lower component of a 2-component 1-D Gaussian mixture (EM, deterministic init at quartiles)."""
    v = np.asarray(v, dtype=float)
    if len(v) < 4 or v.std() == 0:
        return float(v.mean())
    m = np.quantile(v, [0.25, 0.75])
    s = np.array([v.std(), v.std()]) / 2 + 1e-6
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        p = w / s * np.exp(-0.5 * ((v[:, None] - m) / s) ** 2) + 1e-300
        r = p / p.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-12
        m_new = (r * v[:, None]).sum(axis=0) / nk
        s = np.sqrt((r * (v[:, None] - m_new) ** 2).sum(axis=0) / nk) + 1e-6
        w = nk / nk.sum()
        if np.allclose(m_new, m, atol=1e-6):
            m = m_new
            break
        m = m_new
    return float(m.min())


def dsb(cells: np.ndarray, empties: np.ndarray, isotype_rows: Optional[np.ndarray] = None,
        denoise: bool = True) -> np.ndarray:
    """cells, empties: features x droplets (counts). Returns DSB values, features x cells."""
    lc = np.log(np.asarray(cells, dtype=np.float64) + DSB_PSEUDOCOUNT)
    le = np.log(np.asarray(empties, dtype=np.float64) + DSB_PSEUDOCOUNT)
    mu = le.mean(axis=1, keepdims=True)
    sd = le.std(axis=1, ddof=1, keepdims=True) if le.shape[1] > 1 else np.ones_like(mu)
    sd[sd == 0] = 1.0
    z = (lc - mu) / sd
    if not denoise:
        return z
    bg = np.array([_two_gauss_low_mean(z[:, j]) for j in range(z.shape[1])])
    if isotype_rows is not None and len(isotype_rows) > 0:
        tech = np.vstack([bg, z[isotype_rows]]).T  # cells x (1 + n_isotypes)
        tech = tech - tech.mean(axis=0)
        u, sv, vt = np.linalg.svd(tech, full_matrices=False)
        comp = u[:, 0] * sv[0]
        if np.corrcoef(comp, bg)[0, 1] < 0:
            comp = -comp
    else:
        comp = bg - bg.mean()
    # regress the technical component out of each protein (lm per protein, keep the intercept)
    c = comp - comp.mean()
    denom = (c ** 2).sum()
    if denom == 0:
        return z
    beta = (z - z.mean(axis=1, keepdims=True)) @ c / denom
    return z - beta[:, None] * c[None, :]
