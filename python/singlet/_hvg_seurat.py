# SPDX-License-Identifier: MIT
"""Seurat v3 highly variable genes selection."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def highly_variable_genes_seurat_v3(
    adata: AnnData,
    *,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    span: float = 0.3,
    layer: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Select highly variable genes using the Seurat v3 VST method.

    Fits a local regression of log(variance) vs log(mean) and selects
    genes with the largest residual variance (most variable relative
    to their expression level).

    Parameters
    ----------
    adata
        Annotated data matrix. Should contain raw counts (not normalized).
    n_top_genes
        Number of highly variable genes to select.
    batch_key
        Key in .obs for batch. If provided, HVG selection is done per-batch
        and combined.
    span
        Span parameter for loess-like smoothing (fraction of data used
        for each local fit).
    layer
        Layer to use. None uses .X.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Adds columns to .var:
        - 'highly_variable_seurat_v3' (bool)
        - 'means_seurat_v3' (float)
        - 'variances_seurat_v3' (float)
        - 'variances_norm_seurat_v3' (float) — normalized (residual) variance
    """
    from scipy.sparse import issparse

    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if batch_key is not None and batch_key in adata.obs.columns:
        # Per-batch HVG selection
        batches = adata.obs[batch_key].unique()
        all_norm_vars = np.zeros((len(batches), adata.n_vars))

        for i, batch in enumerate(batches):
            mask = (adata.obs[batch_key] == batch).values
            if issparse(X):
                X_batch = np.asarray(X[mask].todense())
            else:
                X_batch = np.asarray(X[mask])

            _, _, norm_var = _vst_single(X_batch, span)
            all_norm_vars[i] = norm_var

        # Rank genes by median normalized variance across batches
        median_norm_var = np.median(all_norm_vars, axis=0)

        # Compute overall stats
        if issparse(X):
            X_dense = np.asarray(X.todense())
        else:
            X_dense = np.asarray(X)

        means = X_dense.mean(axis=0)
        variances = X_dense.var(axis=0)
        norm_variances = median_norm_var
    else:
        if issparse(X):
            X_dense = np.asarray(X.todense())
        else:
            X_dense = np.asarray(X)

        means, variances, norm_variances = _vst_single(X_dense, span)

    # Select top genes by normalized variance
    n_select = min(n_top_genes, len(norm_variances))
    top_idx = np.argsort(norm_variances)[::-1][:n_select]

    highly_variable = np.zeros(adata.n_vars, dtype=bool)
    highly_variable[top_idx] = True

    adata.var["highly_variable_seurat_v3"] = highly_variable
    adata.var["means_seurat_v3"] = means.astype(np.float32)
    adata.var["variances_seurat_v3"] = variances.astype(np.float32)
    adata.var["variances_norm_seurat_v3"] = norm_variances.astype(np.float32)

    return adata if copy else None


def _vst_single(X: np.ndarray, span: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variance-stabilizing transformation for a single batch.

    Returns (means, variances, normalized_variances).
    """
    means = X.mean(axis=0)
    variances = X.var(axis=0, ddof=1)

    # Avoid log(0)
    valid = (means > 0) & (variances > 0)
    log_means = np.full_like(means, 0.0)
    log_vars = np.full_like(variances, 0.0)
    log_means[valid] = np.log10(means[valid])
    log_vars[valid] = np.log10(variances[valid])

    # Fit local polynomial (simplified loess: use rolling window)
    fitted_log_var = _local_regression(log_means, log_vars, valid, span)

    # Compute standardized values (clip variance from below)
    fitted_var = np.power(10, fitted_log_var)

    # Clip values at sqrt(fitted_var) for numerical stability
    clipped = np.clip(X, a_min=None, a_max=np.sqrt(fitted_var)[None, :])

    # Standardize: (x - mean) / sqrt(fitted_var)
    std_vals = np.sqrt(fitted_var)
    std_vals[std_vals == 0] = 1

    standardized = (clipped - means[None, :]) / std_vals[None, :]

    # Variance of standardized values = normalized variance
    norm_var = np.var(standardized, axis=0, ddof=1)

    return means, variances, norm_var


def _local_regression(x: np.ndarray, y: np.ndarray, valid: np.ndarray, span: float) -> np.ndarray:
    """Simple local polynomial regression (loess approximation).

    Uses a moving window with polynomial degree 2.
    """
    n = len(x)
    window_size = max(3, int(np.sum(valid) * span))

    # Sort by x for windowed fitting
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    valid_sorted = valid[order]

    fitted = np.zeros(n)

    for i in range(n):
        if not valid_sorted[i]:
            fitted[i] = 0.0
            continue

        # Find window around this point
        half_w = window_size // 2
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)

        # Only use valid points in window
        w_valid = valid_sorted[start:end]
        if w_valid.sum() < 3:
            fitted[i] = y_sorted[i]
            continue

        x_w = x_sorted[start:end][w_valid]
        y_w = y_sorted[start:end][w_valid]

        # Tricube weights
        x_center = x_sorted[i]
        max_dist = np.max(np.abs(x_w - x_center)) + 1e-10
        u = np.abs(x_w - x_center) / max_dist
        weights = (1 - u**3) ** 3

        # Weighted polynomial fit (degree 2)
        try:
            coeffs = np.polyfit(x_w, y_w, deg=min(2, len(x_w) - 1), w=weights)
            fitted[i] = np.polyval(coeffs, x_center)
        except (np.linalg.LinAlgError, ValueError):
            fitted[i] = y_sorted[i]

    # Unsort
    result = np.zeros(n)
    result[order] = fitted

    return result
