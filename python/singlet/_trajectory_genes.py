# SPDX-License-Identifier: MIT
"""Trajectory gene detection for AnnData objects.

Provides singlet.trajectory_genes() — identifies genes with significant
expression variation along a pseudotime trajectory.
"""

from __future__ import annotations


def trajectory_genes(
    adata,
    *,
    pseudotime_key: str = "dpt_pseudotime",
    n_bins: int = 50,
    n_top_genes: int = 100,
    method: str = "gam",
):
    """Find genes varying significantly along a trajectory.

    Bins cells by pseudotime, computes smoothed expression trends, and
    identifies genes with significant variation using an F-test.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must contain pseudotime in adata.obs.
    pseudotime_key : str, default 'dpt_pseudotime'
        Key in adata.obs containing pseudotime values.
    n_bins : int, default 50
        Number of bins for smoothing expression along pseudotime.
    n_top_genes : int, default 100
        Number of top trajectory genes to return.
    method : str, default 'gam'
        Smoothing method: 'gam' (rolling window) or 'spline' (cubic spline).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: gene, pvalue, fdr, trend_score, max_pseudotime.
        Sorted by fdr ascending, then trend_score descending.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.dpt(adata)
    >>> result = singlet.trajectory_genes(adata)
    >>> result.head()
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import f as f_dist

    # Validate inputs
    if pseudotime_key not in adata.obs.columns:
        raise KeyError(
            f"Pseudotime key '{pseudotime_key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )

    if method not in ("gam", "spline"):
        raise ValueError(f"method must be 'gam' or 'spline', got '{method}'")

    if n_bins < 3:
        raise ValueError(f"n_bins must be >= 3, got {n_bins}")

    # Get pseudotime values, filter out NaN
    pseudotime = adata.obs[pseudotime_key].values.astype(np.float64)
    valid_mask = ~np.isnan(pseudotime) & ~np.isinf(pseudotime)

    if valid_mask.sum() < n_bins:
        raise ValueError(
            f"Only {valid_mask.sum()} cells have valid pseudotime values, need at least {n_bins}."
        )

    # Get expression matrix
    X = adata.X
    if sp.issparse(X):
        X_dense = np.asarray(X[valid_mask].todense())
    else:
        X_dense = np.asarray(X[valid_mask])

    pt_valid = pseudotime[valid_mask]
    n_cells_valid, n_genes = X_dense.shape

    # Bin cells by pseudotime
    bin_edges = np.linspace(pt_valid.min(), pt_valid.max() + 1e-10, n_bins + 1)
    bin_indices = np.digitize(pt_valid, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute mean expression per bin
    bin_means = np.zeros((n_bins, n_genes), dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for b_idx in range(n_bins):
        mask = bin_indices == b_idx
        count = mask.sum()
        bin_counts[b_idx] = count
        if count > 0:
            bin_means[b_idx] = X_dense[mask].mean(axis=0)

    # Handle empty bins via interpolation
    nonempty = bin_counts > 0
    if not nonempty.all():
        for g_idx in range(n_genes):
            if nonempty.sum() >= 2:
                bin_means[~nonempty, g_idx] = np.interp(
                    bin_centers[~nonempty],
                    bin_centers[nonempty],
                    bin_means[nonempty, g_idx],
                )

    # Smooth expression trends
    smoothed = np.zeros_like(bin_means)

    if method == "gam":
        # Rolling window smoothing (window = max(3, n_bins//10))
        window = max(3, n_bins // 10)
        for g_idx in range(n_genes):
            kernel = np.ones(window) / window
            # Pad for edge handling
            padded = np.pad(bin_means[:, g_idx], window // 2, mode="edge")
            smoothed[:, g_idx] = np.convolve(padded, kernel, mode="valid")[:n_bins]
    else:
        # Spline smoothing
        for g_idx in range(n_genes):
            gene_vals = bin_means[:, g_idx]
            variance = np.var(gene_vals)
            if variance < 1e-10:
                smoothed[:, g_idx] = gene_vals
            else:
                smoothing_factor = n_bins * variance * 0.5
                try:
                    spl = UnivariateSpline(bin_centers, gene_vals, s=smoothing_factor, k=3)
                    smoothed[:, g_idx] = spl(bin_centers)
                except Exception:
                    smoothed[:, g_idx] = gene_vals

    # Compute F-statistic for each gene (variation of smoothed trend vs flat)
    grand_mean = smoothed.mean(axis=0)
    ss_trend = np.sum((smoothed - grand_mean[np.newaxis, :]) ** 2, axis=0)
    # Residual: bin_means - smoothed
    ss_resid = np.sum((bin_means - smoothed) ** 2, axis=0)

    df_trend = n_bins - 1
    df_resid = max(1, n_bins - (n_bins // 10 + 1))  # approximate df for smoothing

    ms_trend = ss_trend / max(df_trend, 1)
    ms_resid = ss_resid / max(df_resid, 1) + 1e-10  # avoid division by zero

    f_stats = ms_trend / ms_resid

    # Compute p-values from F-distribution
    pvalues = 1 - f_dist.cdf(f_stats, df_trend, df_resid)
    pvalues = np.clip(pvalues, 0, 1)

    # Trend score: normalized range of smoothed expression
    trend_range = smoothed.max(axis=0) - smoothed.min(axis=0)
    gene_std = bin_means.std(axis=0) + 1e-10
    trend_score = trend_range / gene_std

    # Max pseudotime: bin center where smoothed expression is maximal
    max_bin_idx = np.argmax(smoothed, axis=0)
    max_pseudotime = bin_centers[max_bin_idx]

    # FDR correction (Benjamini-Hochberg)
    n_tested = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    fdr = np.zeros(n_tested, dtype=np.float64)
    for rank, idx in enumerate(sorted_idx, 1):
        fdr[idx] = pvalues[idx] * n_tested / rank
    # Ensure monotonicity
    for rank_idx in range(n_tested - 2, -1, -1):
        idx = sorted_idx[rank_idx]
        next_idx = sorted_idx[rank_idx + 1]
        if fdr[idx] > fdr[next_idx]:
            fdr[idx] = fdr[next_idx]
    fdr = np.clip(fdr, 0, 1)

    # Build result DataFrame
    gene_names = list(adata.var_names)
    result_df = pd.DataFrame(
        {
            "gene": gene_names,
            "pvalue": pvalues,
            "fdr": fdr,
            "trend_score": trend_score,
            "max_pseudotime": max_pseudotime,
        }
    )

    # Sort by fdr ascending, then trend_score descending
    result_df = result_df.sort_values(["fdr", "trend_score"], ascending=[True, False]).reset_index(
        drop=True
    )

    # Take top N
    result_df = result_df.head(n_top_genes).reset_index(drop=True)

    # Store in adata.uns
    adata.uns["trajectory_genes"] = result_df

    # Store smoothed expression per cell (interpolate from bin smoothed)
    smoothed_layer = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
    for g_idx in range(n_genes):
        # For valid cells, interpolate smoothed from bin centers
        smoothed_layer[valid_mask, g_idx] = np.interp(
            pt_valid, bin_centers, smoothed[:, g_idx]
        ).astype(np.float32)
        # For invalid cells (NaN pseudotime), use 0
    adata.layers["trajectory_smoothed"] = smoothed_layer

    return result_df
