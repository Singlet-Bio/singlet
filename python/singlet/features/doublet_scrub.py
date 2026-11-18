# SPDX-License-Identifier: MIT
"""scrub: the Scrublet doublet score as a numpy / scikit-learn input of ``doublet/score_v2``.

Port of the method of Scrublet (Wolock, Lopez and Klein, Cell Systems 8:281-291, 2019; github.com/swolock/scrublet, MIT licence,
Copyright (c) 2018 Samuel L Wolock), credited in NOTICE. Same steps and defaults as ``Scrublet.scrub_doublets``:

1. counts per cell scaled to the mean library size; genes kept when seen in at least 3 cells with at least 3 counts and in the top
   15% of the v-score (Klein et al. 2015, the Fano factor above the low-expression baseline);
2. ``sim_ratio`` (2.0) simulated doublets per observed cell, each the raw-count sum of two random observed cells, scaled alike;
3. z-scoring with the observed cells' gene means and standard deviations; PCA (30 components) fitted on the observed cells;
4. k = round(0.5 sqrt(n_obs)) nearest neighbours in the joint space, scaled by (1 + n_sim / n_obs); the score of a cell is the
   Bayes-adjusted fraction of simulated doublets among them (Scrublet's ``Ld``, monotone in the neighbour fraction).

One departure for speed on large samples: the fit (genes, PCA, simulated doublets, neighbour reference) uses at most ``cap`` observed
cells (a random subset); every other cell is projected and scored against that fixed reference, so the cost stays linear in cells."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _running_quantile(x: np.ndarray, y: np.ndarray, p: float, n_bins: int):
    o = np.argsort(x)
    x, y = x[o], y[o]
    dx = (x[-1] - x[0]) / n_bins
    xo = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, n_bins)
    yo = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (x >= xo[i] - dx) & (x < xo[i] + dx)
        if m.any():
            yo[i] = np.percentile(y[m], p * 100)
        elif i > 0:
            yo[i] = yo[i - 1]
    return xo, yo


def v_scores(
    E: sp.csr_matrix, n_bins: int = 50, fit_percentile: float = 0.1, error_wt: float = 1.0
) -> np.ndarray:
    """Scrublet's ``get_vscores`` on a (cells x genes) normalised matrix; NaN for genes with zero mean."""
    from scipy.optimize import minimize

    mu = np.asarray(E.mean(0)).ravel()
    var = np.asarray(E.multiply(E).mean(0)).ravel() - mu**2
    out = np.full(len(mu), np.nan)
    ok = mu > 0
    ff = var[ok] / mu[ok]
    pos = ff > 0
    if pos.sum() < 100:
        return out
    gx, gy = np.log(mu[ok][pos]), np.log(ff[pos] / mu[ok][pos])
    x, y = _running_quantile(gx, gy, fit_percentile, n_bins)
    keep = ~np.isnan(y)
    x, y = x[keep], y[keep]
    h, b = np.histogram(np.log(ff[pos]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    c = max(float(np.exp(b[np.argmax(h)])), 1.0)
    err = lambda b2: float(np.sum(np.abs(np.log(c * np.exp(-x) + b2[0]) - y) ** error_wt))  # noqa: E731
    bb = float(
        minimize(err, [0.1], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6)).x[0]
    )
    a = c / (1 + bb) - 1
    v = np.full(int(ok.sum()), np.nan)
    v[pos] = ff[pos] / ((1 + a) * (1 + bb) + bb * mu[ok][pos])
    out[np.flatnonzero(ok)] = v
    return out


def score(
    X: sp.csr_matrix,
    seed: int = 0,
    sim_ratio: float = 2.0,
    n_pc: int = 30,
    cap: int = 8000,
    min_counts: int = 3,
    min_cells: int = 3,
    vscore_pctl: float = 85.0,
    expected_rate: float = 0.1,
) -> np.ndarray:
    """Scrublet doublet score for every row of ``X`` (raw counts); higher = more doublet-like."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    X = sp.csr_matrix(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    sub = np.sort(rng.choice(N, cap, replace=False)) if N > cap else np.arange(N)
    rest = np.setdiff1d(np.arange(N), sub)
    Ns = len(sub)
    Xs = X[sub]
    tot = np.asarray(X.sum(1)).ravel().astype(np.float64)
    target = float(tot[sub].mean())

    def norm(M, t):
        return sp.csr_matrix(sp.diags(target / np.maximum(t, 1)) @ M)

    En = norm(Xs, tot[sub])
    counts = np.asarray(Xs.sum(0)).ravel()
    cells = np.asarray((Xs > 0).sum(0)).ravel()
    v = v_scores(En)
    cand = (counts >= min_counts) & (cells >= min_cells) & np.isfinite(v)
    if cand.sum() < 50:
        return np.full(N, np.nan, np.float32)
    genes = np.flatnonzero(cand & (v >= np.percentile(v[cand], vscore_pctl)))
    Eg = En[:, genes]
    mu = np.asarray(Eg.mean(0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(Eg.multiply(Eg).mean(0)).ravel() - mu**2, 0))
    sd[sd == 0] = 1.0

    def z(M, t):
        return ((norm(M, t)[:, genes].toarray() - mu) / sd).astype(np.float32)

    Zs = z(Xs, tot[sub])
    pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=seed).fit(Zs)
    P_obs = pca.transform(Zs)
    n_sim = int(sim_ratio * Ns)
    a = rng.integers(0, Ns, n_sim)
    b = rng.integers(0, Ns, n_sim)
    S = (Xs[a] + Xs[b]).tocsr()
    P_sim = pca.transform(z(S, tot[sub][a] + tot[sub][b]))
    ref = np.vstack([P_obs, P_sim]).astype(np.float32)
    is_sim = np.r_[np.zeros(Ns, bool), np.ones(n_sim, bool)]
    k = int(round(0.5 * np.sqrt(Ns)))
    k_adj = int(round(k * (1 + n_sim / Ns)))
    nn = NearestNeighbors(n_neighbors=k_adj + 1, n_jobs=-1).fit(ref)

    def ld(Pq, self_in_ref):
        ind = nn.kneighbors(Pq, return_distance=False)
        ind = ind[:, 1:] if self_in_ref else ind[:, :k_adj]
        q = (is_sim[ind].sum(1) + 1) / (k_adj + 2)
        r = expected_rate / sim_ratio
        return q * r / (1 - expected_rate - q * (1 - expected_rate - r))

    out = np.zeros(N, np.float64)
    out[sub] = ld(P_obs.astype(np.float32), True)
    for i in range(0, len(rest), 20000):
        ch = rest[i : i + 20000]
        out[ch] = ld(pca.transform(z(X[ch], tot[ch])).astype(np.float32), False)
    return out.astype(np.float32)
