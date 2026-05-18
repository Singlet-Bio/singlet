# SPDX-License-Identifier: MIT
"""Highly variable gene selection using the Cell Ranger method.

Provides singlet.highly_variable_genes_cell_ranger() — identifies genes
with high cell-to-cell variation using median/MAD normalization within
expression bins (the Cell Ranger variant of the dispersion approach).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


def highly_variable_genes_cell_ranger(
    adata,
    *,
    n_top_genes: int = 2000,
    n_bins: int = 20,
    inplace: bool = True,
) -> Optional[list[str]]:
    """Identify highly variable genes using the Cell Ranger method.

    Computes mean and dispersion for each gene, bins genes by log10
    mean expression, then normalizes dispersion within each bin using
    median and MAD (median absolute deviation). Selects top genes by
    normalized dispersion.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    n_top_genes : int, default 2000
        Number of highly variable genes to select.
    n_bins : int, default 20
        Number of equal-width bins for mean expression.
    inplace : bool, default True
        If True, adds columns to adata.var and returns None.
        If False, returns list of highly variable gene names.

    Returns
    -------
    list[str] or None
        List of HVG names (if inplace=False), or None (if inplace=True,
        results stored in adata.var).
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(
            "highly_variable_genes_cell_ranger() requires an "
            f"AnnData object, got {type(adata).__name__}"
        )

    X = adata.X
    n_cells, n_genes = X.shape

    if n_genes < n_top_genes:
        n_top_genes = n_genes

    # Compute mean and variance per gene
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        X_sq = X.copy()
        X_sq.data **= 2
        mean_sq = np.asarray(X_sq.mean(axis=0)).ravel()
        # Population variance then convert to sample variance
        var = (mean_sq - mean**2) * n_cells / (n_cells - 1)
    else:
        mean = np.asarray(X.mean(axis=0)).ravel()
        var = np.asarray(X.var(axis=0, ddof=1)).ravel()

    mean = mean.astype(np.float64)
    var = var.astype(np.float64)

    # Dispersion = variance / mean
    disp = np.zeros_like(mean)
    nonzero_mean = mean > 0
    disp[nonzero_mean] = var[nonzero_mean] / mean[nonzero_mean]

    # Log10 transform for binning
    log_mean = np.zeros_like(mean)
    log_mean[nonzero_mean] = np.log10(mean[nonzero_mean])

    log_disp = np.zeros_like(disp)
    pos_disp = disp > 0
    log_disp[pos_disp] = np.log10(disp[pos_disp])

    # Bin genes by log10 mean expression (equal-width bins)
    valid = nonzero_mean & pos_disp
    disp_norm = np.zeros_like(log_disp)

    if valid.sum() > 0:
        min_val = log_mean[valid].min()
        max_val = log_mean[valid].max()
        bins = np.linspace(min_val, max_val, n_bins + 1)
        # Extend edges to capture all genes
        bins[0] = -np.inf
        bins[-1] = np.inf
        bin_indices = np.digitize(log_mean, bins)

        for idx in range(1, n_bins + 1):
            in_bin = (bin_indices == idx) & valid
            if in_bin.sum() < 2:
                disp_norm[in_bin] = 0
                continue
            bin_disp = log_disp[in_bin]
            bin_median = np.median(bin_disp)
            mad = np.median(np.abs(bin_disp - bin_median))
            if mad == 0:
                disp_norm[in_bin] = 0
            else:
                disp_norm[in_bin] = (bin_disp - bin_median) / mad

    # Select top n_top_genes by normalized dispersion
    top_idx = np.argsort(disp_norm)[::-1][:n_top_genes]

    highly_variable = np.zeros(n_genes, dtype=bool)
    highly_variable[top_idx] = True

    if inplace:
        adata.var["highly_variable"] = highly_variable
        adata.var["means"] = mean.astype(np.float32)
        adata.var["dispersions"] = disp.astype(np.float32)
        adata.var["dispersions_norm"] = disp_norm.astype(np.float32)
        return None
    else:
        return list(adata.var_names[highly_variable])
