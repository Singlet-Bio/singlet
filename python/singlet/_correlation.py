# SPDX-License-Identifier: MIT
"""Correlation matrix computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def correlation_matrix(
    adata: AnnData,
    var_names: list[str] | None = None,
    *,
    groupby: str | None = None,
    layer: str | None = None,
    method: str = "pearson",
    use: str = "var",
) -> pd.DataFrame:
    """Compute correlation matrix between genes or observations.

    Parameters
    ----------
    adata
        Annotated data matrix.
    var_names
        Subset of genes to compute correlations for. If None and use='var',
        uses highly variable genes if available, else top 50 by variance.
    groupby
        Key in .obs. If provided, correlations are computed within each
        group and the per-group mean is returned.
    layer
        Layer to use. None uses .X.
    method
        Correlation method: 'pearson' or 'spearman'.
    use
        'var' for gene-gene correlations, 'obs' for cell-cell correlations.

    Returns
    -------
    DataFrame with correlation matrix.
    """
    from scipy.sparse import issparse
    from scipy.stats import spearmanr

    # Get expression matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        X = np.asarray(X.todense())
    else:
        X = np.asarray(X)

    if use == "var":
        # Gene-gene correlation
        if var_names is not None:
            gene_idx = [list(adata.var_names).index(g) for g in var_names if g in adata.var_names]
            if len(gene_idx) == 0:
                raise ValueError("None of the specified genes found.")
            names = [adata.var_names[i] for i in gene_idx]
            X_sub = X[:, gene_idx]
        else:
            # Use highly variable genes or top by variance
            if "highly_variable" in adata.var.columns:
                hv_mask = adata.var["highly_variable"].values
                names = list(adata.var_names[hv_mask][:50])
                gene_idx = [list(adata.var_names).index(g) for g in names]
                X_sub = X[:, gene_idx]
            else:
                variances = X.var(axis=0)
                top_idx = np.argsort(variances)[::-1][:50]
                names = [adata.var_names[i] for i in top_idx]
                X_sub = X[:, top_idx]

        if groupby is not None and groupby in adata.obs.columns:
            # Per-group correlation, return mean
            groups = adata.obs[groupby].unique()
            corr_sum = np.zeros((X_sub.shape[1], X_sub.shape[1]))

            for grp in groups:
                mask = (adata.obs[groupby] == grp).values
                X_grp = X_sub[mask]
                if X_grp.shape[0] < 3:
                    continue
                if method == "spearman":
                    c, _ = spearmanr(X_grp)
                    if X_sub.shape[1] == 2:
                        # spearmanr returns scalar for 2 variables
                        c = np.array([[1.0, c], [c, 1.0]])
                    elif c.ndim == 0:
                        c = np.array([[1.0]])
                else:
                    c = np.corrcoef(X_grp.T)
                corr_sum += np.nan_to_num(c)

            corr = corr_sum / len(groups)
        else:
            if method == "spearman":
                corr, _ = spearmanr(X_sub)
                if X_sub.shape[1] == 2:
                    # spearmanr returns scalar for 2 variables
                    corr = np.array([[1.0, corr], [corr, 1.0]])
                elif corr.ndim == 0:
                    corr = np.array([[1.0]])
            else:
                corr = np.corrcoef(X_sub.T)

        return pd.DataFrame(corr, index=names, columns=names)

    elif use == "obs":
        # Cell-cell correlation (on PCA for efficiency)
        if "X_pca" in adata.obsm:
            X_use = adata.obsm["X_pca"]
        else:
            X_use = X

        if method == "spearman":
            corr, _ = spearmanr(X_use.T)
            if corr.ndim == 0:
                corr = np.array([[1.0]])
        else:
            corr = np.corrcoef(X_use)

        obs_names = list(adata.obs_names)
        return pd.DataFrame(corr, index=obs_names, columns=obs_names)

    else:
        raise ValueError(f"'use' must be 'var' or 'obs', got '{use}'.")
