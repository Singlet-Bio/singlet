# SPDX-License-Identifier: MIT
"""Hotspot-style spatially variable gene detection.

Provides singlet.hotspot_genes() — identify genes with local autocorrelation
(local clusters of high expression) using Getis-Ord Gi* or local Moran's I.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def hotspot_genes(
    adata: AnnData,
    *,
    n_neighbors: int = 30,
    fdr_threshold: float = 0.05,
    use_rep: str = "X_pca",
    method: str = "gi_star",
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Identify genes with local autocorrelation (expression hotspots).

    Unlike global Moran's I which detects overall spatial patterning,
    this function detects genes that have localized clusters of high
    expression (hotspots) using the Getis-Ord Gi* statistic or local
    Moran's I.

    Parameters
    ----------
    adata
        Annotated data matrix with a representation in ``.obsm[use_rep]``.
    n_neighbors
        Number of neighbors for constructing the local neighborhood graph.
    fdr_threshold
        False discovery rate threshold for significance.
    use_rep
        Key in ``adata.obsm`` for computing the neighborhood graph.
    method
        Method for local autocorrelation: ``'gi_star'`` (Getis-Ord Gi*)
        or ``'local_morans'`` (local Moran's I).
    genes
        Genes to test. If None, uses highly variable genes or all genes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: gene, statistic, pvalue, fdr, n_hotspot_cells.
        Sorted by statistic descending.
        Also stored in ``adata.uns['hotspot_genes']``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> result = singlet.hotspot_genes(adata, method="gi_star")
    >>> result.head()  # top genes with expression hotspots
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    valid_methods = ("gi_star", "local_morans")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got {method!r}"
        raise ValueError(msg)

    if use_rep not in adata.obsm:
        msg = f"Representation {use_rep!r} not found in adata.obsm"
        raise KeyError(msg)

    if n_neighbors < 1:
        msg = f"n_neighbors must be >= 1, got {n_neighbors}"
        raise ValueError(msg)

    # Select genes
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        if gene_mask.sum() == 0:
            msg = "None of the specified genes found in adata.var_names"
            raise ValueError(msg)
        gene_names = adata.var_names[gene_mask].tolist()
        gene_idx = np.where(gene_mask)[0]
    elif "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values.astype(bool)
        gene_names = adata.var_names[hvg_mask].tolist()
        gene_idx = np.where(hvg_mask)[0]
    else:
        gene_names = adata.var_names.tolist()
        gene_idx = np.arange(adata.n_vars)

    n_cells = adata.n_obs
    n_genes = len(gene_names)
    k = min(n_neighbors, n_cells - 1)

    # Build kNN graph from representation
    rep = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    tree = cKDTree(rep)
    # Query k+1 because cKDTree includes self
    _, indices = tree.query(rep, k=k + 1)
    # Remove self (first column)
    nn_indices = indices[:, 1:]

    # Build binary weight matrix (row-standardized)
    row_idx = np.repeat(np.arange(n_cells), k)
    col_idx = nn_indices.ravel()
    data = np.ones(n_cells * k, dtype=np.float64) / k
    weights = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_cells, n_cells))

    # Extract expression matrix
    expr_mat = adata.X[:, gene_idx]
    if sp.issparse(expr_mat):
        expr_mat = np.asarray(expr_mat.todense())
    else:
        expr_mat = np.asarray(expr_mat, dtype=np.float64)

    # Compute local statistics for each gene
    statistics = np.zeros(n_genes, dtype=np.float64)
    pvalues = np.zeros(n_genes, dtype=np.float64)
    hotspot_counts = np.zeros(n_genes, dtype=np.int64)

    for gene_j in range(n_genes):
        values = expr_mat[:, gene_j]
        global_mean = values.mean()
        global_std = values.std()

        if global_std < 1e-10:
            # Constant gene — no spatial pattern
            statistics[gene_j] = 0.0
            pvalues[gene_j] = 1.0
            hotspot_counts[gene_j] = 0
            continue

        if method == "gi_star":
            # Getis-Ord Gi* for each cell
            # Gi* = (sum_j w_ij * x_j - mean * sum_j w_ij) /
            #        (std * sqrt((n * sum_j w_ij^2 - (sum_j w_ij)^2) / (n-1)))
            lag = np.asarray(weights @ values.reshape(-1, 1)).ravel()
            # For row-standardized weights, sum_j w_ij = 1
            gi_star = (lag - global_mean) / global_std

            # Gene-level statistic: max Gi* (captures strongest hotspot)
            statistics[gene_j] = np.max(gi_star)

            # Count significant hotspot cells (z > 1.96 ~ p < 0.05)
            hotspot_counts[gene_j] = int(np.sum(gi_star > 1.96))

            # Gene-level p-value from max statistic
            # Under null, Gi* ~ N(0,1), max of n iid normals
            # Use Gumbel approximation for max of normals
            from scipy.stats import norm

            # Bonferroni-style p-value for the max
            max_stat = statistics[gene_j]
            pvalues[gene_j] = min(1.0, n_cells * (1.0 - norm.cdf(max_stat)))

        else:
            # Local Moran's I
            centered = values - global_mean
            # Local Moran for cell i: I_i = (x_i - mean) * sum_j w_ij * (x_j - mean) / var
            lag_centered = np.asarray(weights @ centered.reshape(-1, 1)).ravel()
            variance = global_std**2
            local_mi = centered * lag_centered / variance

            # Gene-level statistic: mean of positive local I values
            pos_mask = local_mi > 0
            if pos_mask.sum() > 0:
                statistics[gene_j] = np.mean(local_mi[pos_mask])
            else:
                statistics[gene_j] = 0.0

            # Count significant hotspot cells (positive local I)
            # Use pseudo z-score: I_i / E[I_i] where E[I_i] = -1/(n-1)
            expected_local = -1.0 / (n_cells - 1) if n_cells > 1 else 0.0
            z_scores = (local_mi - expected_local) / (np.std(local_mi) + 1e-10)
            hotspot_counts[gene_j] = int(np.sum(z_scores > 1.96))

            from scipy.stats import norm

            max_z = np.max(z_scores)
            pvalues[gene_j] = min(1.0, n_cells * (1.0 - norm.cdf(max_z)))

    # FDR correction (Benjamini-Hochberg)
    if n_genes > 0:
        ranked = np.argsort(pvalues)
        fdr = np.ones(n_genes, dtype=np.float64)
        for rank_pos, orig_idx in enumerate(ranked):
            rank_val = rank_pos + 1
            fdr[orig_idx] = pvalues[orig_idx] * n_genes / rank_val
        # Enforce monotonicity
        fdr_sorted_idx = ranked[::-1]
        cummin = fdr[fdr_sorted_idx[0]]
        for sorted_idx in fdr_sorted_idx:
            if fdr[sorted_idx] < cummin:
                cummin = fdr[sorted_idx]
            else:
                fdr[sorted_idx] = cummin
        fdr = np.minimum(fdr, 1.0)
    else:
        fdr = np.array([], dtype=np.float64)

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "gene": gene_names,
            "statistic": statistics,
            "pvalue": pvalues,
            "fdr": fdr,
            "n_hotspot_cells": hotspot_counts,
        }
    )
    result = result.sort_values("statistic", ascending=False).reset_index(drop=True)

    # Store in adata.uns
    adata.uns["hotspot_genes"] = result

    return result
