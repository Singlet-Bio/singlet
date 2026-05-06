"""Moran's I spatial autocorrelation for gene expression.

Provides singlet.morans_i() — compute Moran's I statistic for genes on a
kNN or spatial graph to identify spatially/neighborhood-patterned genes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def morans_i(
    adata: AnnData,
    *,
    genes: list[str] | None = None,
    use_graph: bool = True,
    n_perms: int = 100,
) -> pd.DataFrame:
    """Compute Moran's I spatial autocorrelation for gene expression.

    Moran's I measures the degree to which gene expression values of
    neighboring cells are correlated. High positive values indicate
    spatially patterned genes; values near zero indicate random spatial
    distribution.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have a kNN graph in
        ``adata.obsp['connectivities']`` (via ``singlet.neighbors()``)
        or a spatial graph if ``use_graph=True``.
    genes : list of str or None, default None
        Genes to compute Moran's I for. If None, uses highly variable
        genes (``adata.var['highly_variable']``) or all genes if HVGs
        are not annotated.
    use_graph : bool, default True
        If True, uses the existing connectivities graph. If False,
        raises an error (graph is required).
    n_perms : int, default 100
        Number of random permutations for computing p-values via a
        permutation test. Set to 0 to skip p-value computation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: gene, morans_i, expected_i, pvalue, fdr.
        Also stored in ``adata.uns['morans_i']``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> result = singlet.morans_i(adata, n_perms=200)
    >>> result.head()  # genes sorted by Moran's I descending
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    # Validate graph exists
    if use_graph:
        if "connectivities" not in adata.obsp:
            msg = (
                "No kNN graph found in adata.obsp['connectivities']. "
                "Run singlet.neighbors(adata) first."
            )
            raise KeyError(msg)
        weights = adata.obsp["connectivities"]
    else:
        msg = "use_graph=False is not supported; a connectivity graph is required."
        raise ValueError(msg)

    # Ensure sparse CSR
    if not sp.issparse(weights):
        weights = sp.csr_matrix(weights)
    else:
        weights = weights.tocsr()

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

    # Total weight
    total_w = weights.sum()

    # Expected Moran's I under null
    expected = -1.0 / (n_cells - 1) if n_cells > 1 else 0.0

    # Extract expression matrix for selected genes
    expr_mat = adata.X[:, gene_idx]
    if sp.issparse(expr_mat):
        expr_mat = np.asarray(expr_mat.todense())
    else:
        expr_mat = np.asarray(expr_mat, dtype=np.float64)

    def _compute_morans_i_single(values: np.ndarray) -> float:
        """Compute Moran's I for a single gene's expression vector."""
        mean_val = values.mean()
        deviations = values - mean_val
        denom = np.sum(deviations**2)
        if denom == 0 or total_w == 0:
            return 0.0
        # Numerator: sum of w_ij * (x_i - mean) * (x_j - mean)
        # Using sparse matrix multiplication: deviations @ W @ deviations
        # More efficient: (deviations * W) . deviations
        dev_sparse = sp.csr_matrix(deviations.reshape(1, -1))
        numerator = (dev_sparse @ weights).dot(deviations.reshape(-1, 1))[0, 0]
        return (n_cells / total_w) * (numerator / denom)

    # Compute Moran's I for all genes
    morans_values = np.zeros(n_genes, dtype=np.float64)
    for idx in range(n_genes):
        morans_values[idx] = _compute_morans_i_single(expr_mat[:, idx])

    # Permutation test for p-values
    rng = np.random.default_rng(42)
    if n_perms > 0:
        pvalues = np.ones(n_genes, dtype=np.float64)
        for gene_j in range(n_genes):
            values = expr_mat[:, gene_j]
            count_extreme = 0
            for _perm in range(n_perms):
                perm_values = rng.permutation(values)
                perm_mi = _compute_morans_i_single(perm_values)
                if perm_mi >= morans_values[gene_j]:
                    count_extreme += 1
            pvalues[gene_j] = (count_extreme + 1) / (n_perms + 1)
    else:
        pvalues = np.full(n_genes, np.nan)

    # FDR correction (Benjamini-Hochberg)
    if n_perms > 0 and n_genes > 0:
        ranked_pvals = np.argsort(pvalues)
        fdr = np.ones(n_genes, dtype=np.float64)
        for rank_pos, orig_idx in enumerate(ranked_pvals):
            rank_val = rank_pos + 1
            fdr[orig_idx] = pvalues[orig_idx] * n_genes / rank_val
        # Enforce monotonicity (step-up)
        fdr_sorted_idx = ranked_pvals[::-1]
        cummin = fdr[fdr_sorted_idx[0]]
        for sorted_idx in fdr_sorted_idx:
            if fdr[sorted_idx] < cummin:
                cummin = fdr[sorted_idx]
            else:
                fdr[sorted_idx] = cummin
        fdr = np.minimum(fdr, 1.0)
    else:
        fdr = np.full(n_genes, np.nan)

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "gene": gene_names,
            "morans_i": morans_values,
            "expected_i": expected,
            "pvalue": pvalues,
            "fdr": fdr,
        }
    )
    result = result.sort_values("morans_i", ascending=False).reset_index(drop=True)

    # Store in adata
    adata.uns["morans_i"] = result

    return result
