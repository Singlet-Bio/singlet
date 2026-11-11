# SPDX-License-Identifier: MIT
"""Per-cell agreement between two tools' gene count matrices (Clipper side, numpy / scipy only)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from runio import GeneMat


def align(a: GeneMat, b: GeneMat) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Restrict both matrices to the genes both tools annotate, in the same order. Returns (A, B, genes)."""
    common = np.intersect1d(a.genes, b.genes)
    pos_a = {g: i for i, g in enumerate(a.genes)}
    pos_b = {g: i for i, g in enumerate(b.genes)}
    ia = np.array([pos_a[g] for g in common], dtype=np.int64)
    ib = np.array([pos_b[g] for g in common], dtype=np.int64)
    return a.X[:, ia].tocsr(), b.X[:, ib].tocsr(), common


def percell_log_corr(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    """Pearson correlation, per row, of log1p counts over all G genes (zeros included). Rows with no variance give NaN."""
    G = A.shape[1]
    x, y = A.copy().astype(np.float64), B.copy().astype(np.float64)
    x.data = np.log1p(x.data)
    y.data = np.log1p(y.data)
    sx, sy = np.asarray(x.sum(1)).ravel(), np.asarray(y.sum(1)).ravel()
    sxx = np.asarray(x.multiply(x).sum(1)).ravel()
    syy = np.asarray(y.multiply(y).sum(1)).ravel()
    sxy = np.asarray(x.multiply(y).sum(1)).ravel()
    cov = sxy - sx * sy / G
    vx, vy = sxx - sx * sx / G, syy - sy * sy / G
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(vx * vy)
    r[(vx <= 1e-12) | (vy <= 1e-12)] = np.nan
    return r


def pair_corr(a: GeneMat, b: GeneMat, mask: np.ndarray) -> np.ndarray:
    """Per-cell log1p correlation of a and b on ``mask`` cells (NaN elsewhere), over the genes both annotate."""
    A, B, _ = align(a, b)
    out = np.full(A.shape[0], np.nan)
    idx = np.nonzero(mask)[0]
    if len(idx):
        out[idx] = percell_log_corr(A[idx], B[idx])
    return out


def percell_log_corr_union(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    """Per-row Pearson correlation of log1p counts over the features non-zero in either matrix (sparse vectors such as
    junction counts, where the all-zero majority would otherwise dominate). Rows with fewer than 3 such features give NaN."""
    x, y = A.copy().astype(np.float64), B.copy().astype(np.float64)
    x.data = np.log1p(x.data)
    y.data = np.log1p(y.data)
    nx = np.diff(x.indptr)
    ny = np.diff(y.indptr)
    xb, yb = x.copy(), y.copy()
    xb.data[:] = 1.0
    yb.data[:] = 1.0
    nboth = np.asarray(xb.multiply(yb).sum(1)).ravel()
    n = nx + ny - nboth
    sx, sy = np.asarray(x.sum(1)).ravel(), np.asarray(y.sum(1)).ravel()
    sxx = np.asarray(x.multiply(x).sum(1)).ravel()
    syy = np.asarray(y.multiply(y).sum(1)).ravel()
    sxy = np.asarray(x.multiply(y).sum(1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sy / n
        vx, vy = sxx - sx * sx / n, syy - sy * sy / n
        r = cov / np.sqrt(vx * vy)
    r[(n < 3) | (vx <= 1e-12) | (vy <= 1e-12)] = np.nan
    return r
