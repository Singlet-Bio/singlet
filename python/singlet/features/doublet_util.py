# SPDX-License-Identifier: MIT
"""Count-matrix helpers shared by the doublet scorers (ports of bench/latent_qc/doublets/dbl_common.py, unchanged numerics)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def lognorm(X: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    n = np.asarray(X.sum(1)).ravel()
    Y = sp.csr_matrix(sp.diags(target / np.maximum(n, 1)) @ X.astype(np.float32))
    Y.data = np.log1p(Y.data)
    return Y


def hvg(X: sp.csr_matrix, n_top: int = 2000) -> np.ndarray:
    """Top genes by variance of log-normalised expression relative to a mean trend (binned dispersion residual)."""
    Y = lognorm(X)
    mu = np.asarray(Y.mean(0)).ravel()
    var = np.asarray(Y.multiply(Y).mean(0)).ravel() - mu**2
    ok = mu > 0
    bins = np.digitize(mu, np.quantile(mu[ok], np.linspace(0, 1, 21)[1:-1]))
    z = np.full(len(mu), -np.inf)
    for b in np.unique(bins[ok]):
        m = ok & (bins == b)
        v = var[m]
        z[m] = (v - np.median(v)) / (np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-9)
    return np.argsort(-z)[: min(n_top, int(ok.sum()))]


def simulate_doublets(X: sp.csr_matrix, n: int, rng):
    """Sum the raw counts of ``n`` random cell pairs (Scrublet / Solo). Returns (CSR, parent index pairs)."""
    N = X.shape[0]
    a = rng.integers(0, N, n)
    b = rng.integers(0, N, n)
    b = np.where(a == b, (b + 1) % N, b)
    return (X[a] + X[b]).tocsr(), np.stack([a, b], 1)
