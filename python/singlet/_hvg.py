"""Highly variable gene selection for AnnData objects.

Provides singlet.highly_variable_genes() — identifies genes with high
cell-to-cell variation, the standard feature selection step before
dimensionality reduction (PCA, NMF, etc.).
"""

from __future__ import annotations

from typing import Optional


def highly_variable_genes(
    adata,
    *,
    n_top_genes: int = 2000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    flavor: str = "seurat",
    inplace: bool = True,
) -> Optional[list[str]]:
    """Identify highly variable genes (HVGs).

    Computes mean and dispersion for each gene, selects top variable genes.
    After calling this, ``adata.var['highly_variable']`` marks selected genes.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (log-normalized recommended for 'seurat' flavor).
    n_top_genes : int, default 2000
        Number of highly variable genes to select.
    min_mean : float, default 0.0125
        Minimum mean expression for candidate genes (seurat flavor).
    max_mean : float, default 3.0
        Maximum mean expression for candidate genes (seurat flavor).
    min_disp : float, default 0.5
        Minimum normalized dispersion for candidate genes (seurat flavor).
    flavor : str, default "seurat"
        Method for computing variability. Currently only "seurat" (variance/mean
        binned normalization) is supported.
    inplace : bool, default True
        If True, adds 'highly_variable', 'means', 'dispersions',
        'dispersions_norm' columns to adata.var and returns None.
        If False, returns list of highly variable gene names.

    Returns
    -------
    list[str] or None
        List of HVG names (if inplace=False), or None (if inplace=True,
        results stored in adata.var).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> adata[:, adata.var['highly_variable']]  # subset to HVGs
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"highly_variable_genes() requires an AnnData object, got {type(adata).__name__}"
        )

    if flavor != "seurat":
        raise ValueError(f"Unsupported flavor '{flavor}'. Use 'seurat'.")

    X = adata.X
    n_cells, n_genes = X.shape

    if n_genes < n_top_genes:
        n_top_genes = n_genes

    # Compute mean and variance per gene
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        # var = E[X^2] - E[X]^2
        X_sq = X.copy()
        X_sq.data **= 2
        mean_sq = np.asarray(X_sq.mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        mean = X.mean(axis=0)
        var = X.var(axis=0)

    # Avoid division by zero
    mean = np.asarray(mean, dtype=np.float64).ravel()
    var = np.asarray(var, dtype=np.float64).ravel()

    # Dispersion = variance / mean (coefficient of variation squared for Poisson)
    disp = np.zeros_like(mean)
    nonzero_mean = mean > 0
    disp[nonzero_mean] = var[nonzero_mean] / mean[nonzero_mean]

    # Log transform for binning
    mean_log = np.zeros_like(mean)
    mean_log[nonzero_mean] = np.log1p(mean[nonzero_mean])
    disp_log = np.zeros_like(disp)
    pos_disp = disp > 0
    disp_log[pos_disp] = np.log(disp[pos_disp])

    # Bin genes by mean expression, normalize dispersion within bins
    n_bins = 20
    bins = np.linspace(mean_log[nonzero_mean].min(), mean_log[nonzero_mean].max(), n_bins + 1)
    bins = np.concatenate([[-np.inf], bins, [np.inf]])
    bin_indices = np.digitize(mean_log, bins)

    disp_norm = np.zeros_like(disp_log)
    for i in range(1, len(bins)):
        in_bin = bin_indices == i
        if in_bin.sum() == 0:
            continue
        bin_disp = disp_log[in_bin]
        bin_mean = bin_disp.mean()
        bin_std = bin_disp.std()
        if bin_std == 0:
            disp_norm[in_bin] = 0
        else:
            disp_norm[in_bin] = (bin_disp - bin_mean) / bin_std

    # Apply mean/dispersion cutoffs
    candidates = (
        (mean_log >= np.log1p(min_mean))
        & (mean_log <= np.log1p(max_mean))
        & (disp_norm >= min_disp)
    )

    # Select top n_top_genes from candidates by normalized dispersion
    candidate_indices = np.where(candidates)[0]
    if len(candidate_indices) > n_top_genes:
        top_idx = candidate_indices[np.argsort(disp_norm[candidate_indices])[::-1][:n_top_genes]]
    else:
        # If fewer candidates than n_top_genes, take all candidates
        # plus fill from highest disp_norm non-candidates
        top_idx = candidate_indices
        if len(top_idx) < n_top_genes:
            remaining = np.setdiff1d(np.arange(n_genes), top_idx)
            extra = remaining[np.argsort(disp_norm[remaining])[::-1][: n_top_genes - len(top_idx)]]
            top_idx = np.concatenate([top_idx, extra])

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
