# SPDX-License-Identifier: MIT
"""Variance partitioning for gene expression data.

Provides singlet.variance_partition() — decompose gene expression variance
into components attributable to different categorical factors using
ANOVA-style decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def variance_partition(
    adata: "AnnData",
    keys: list[str],
    *,
    n_top_genes: int = 500,
    genes: list[str] | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """Decompose gene expression variance by categorical factors.

    For each gene, compute the fraction of variance explained (eta-squared)
    by each categorical variable in ``adata.obs`` using one-way ANOVA
    decomposition.

    Parameters
    ----------
    adata
        Annotated data matrix with gene expression data.
    keys
        List of column names in ``adata.obs`` to partition variance by.
        Each must be categorical or convertible to categorical.
    n_top_genes
        Number of top highly-variable genes to use. Ignored if ``genes``
        is provided.
    genes
        Explicit list of genes to analyze. If provided, ``n_top_genes``
        is ignored.
    layer
        Layer to use for gene expression. If None, uses ``.X``.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_genes, n_keys + 1) where columns are the
        fraction of variance explained by each key, plus 'residual'.
        Values sum to 1.0 per row. Also stored in
        ``adata.uns['variance_partition']``.

    Examples
    --------
    >>> import singlet
    >>> vp = singlet.variance_partition(adata, keys=["batch", "cell_type"])
    >>> vp.head()
              batch  cell_type  residual
    GENE1     0.12       0.45      0.43
    GENE2     0.03       0.78      0.19
    """
    import scipy.sparse as sp

    if not keys:
        msg = "keys must be a non-empty list of obs column names"
        raise ValueError(msg)

    for key in keys:
        if key not in adata.obs.columns:
            msg = f"Key {key!r} not found in adata.obs"
            raise KeyError(msg)

    # Select genes
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        selected_genes = adata.var_names[gene_mask].tolist()
    elif "highly_variable" in adata.var.columns:
        hv_genes = adata.var_names[adata.var["highly_variable"]].tolist()
        selected_genes = hv_genes[:n_top_genes]
    else:
        # Use top genes by variance
        if layer is not None:
            expr = adata.layers[layer]
        else:
            expr = adata.X
        if sp.issparse(expr):
            # Compute variance for sparse matrix
            mean = np.asarray(expr.mean(axis=0)).ravel()
            mean_sq = np.asarray(expr.multiply(expr).mean(axis=0)).ravel()
            var = mean_sq - mean**2
        else:
            var = np.var(np.asarray(expr), axis=0)
        top_idx = np.argsort(var)[::-1][:n_top_genes]
        selected_genes = adata.var_names[top_idx].tolist()

    if not selected_genes:
        msg = "No genes selected for variance partitioning"
        raise ValueError(msg)

    # Get expression matrix for selected genes
    gene_indices = [adata.var_names.get_loc(g) for g in selected_genes]

    if layer is not None:
        expr_full = adata.layers[layer]
    else:
        expr_full = adata.X

    if sp.issparse(expr_full):
        expr_mat = np.asarray(expr_full[:, gene_indices].toarray(), dtype=np.float64)
    else:
        expr_mat = np.asarray(expr_full[:, gene_indices], dtype=np.float64)

    n_cells, n_genes_sel = expr_mat.shape

    # Compute total variance per gene
    gene_means = expr_mat.mean(axis=0)
    total_ss = np.sum((expr_mat - gene_means[np.newaxis, :]) ** 2, axis=0)

    # Compute SS explained by each factor
    result = {}
    for key in keys:
        groups = adata.obs[key].values
        unique_groups = pd.Categorical(groups).categories
        ss_between = np.zeros(n_genes_sel, dtype=np.float64)

        for group in unique_groups:
            mask = np.asarray(groups == group)
            n_group = mask.sum()
            if n_group == 0:
                continue
            group_mean = expr_mat[mask, :].mean(axis=0)
            ss_between += n_group * (group_mean - gene_means) ** 2

        # Eta-squared: fraction of variance explained
        with np.errstate(divide="ignore", invalid="ignore"):
            eta_sq = np.where(total_ss > 0, ss_between / total_ss, 0.0)
        result[key] = eta_sq

    # Compute residual
    explained = np.sum(list(result.values()), axis=0)
    # Clamp to [0, 1] (overlapping factors can exceed 1 in non-orthogonal design)
    explained = np.minimum(explained, 1.0)
    result["residual"] = 1.0 - explained

    df = pd.DataFrame(result, index=selected_genes)
    df.index.name = "gene"

    # Store in adata
    adata.uns["variance_partition"] = df

    return df
