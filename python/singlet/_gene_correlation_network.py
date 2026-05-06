"""Gene-gene correlation network construction.

Provides singlet.gene_correlation_network() — compute pairwise gene correlations
and threshold into a binary adjacency network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def gene_correlation_network(
    adata: "AnnData",
    genes: list[str] | None = None,
    *,
    threshold: float = 0.3,
    method: str = "pearson",
    n_top_genes: int = 200,
    layer: str | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Compute gene-gene correlation network.

    Builds a pairwise correlation matrix for specified genes (or highly
    variable genes), then thresholds to create a binary adjacency matrix
    representing the gene network.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        List of gene names to include. If None, uses highly variable genes
        (if available) or top genes by variance.
    threshold
        Absolute correlation threshold for edge inclusion in the network.
        Pairs with |correlation| >= threshold get an edge (value 1).
    method
        Correlation method: 'pearson' or 'spearman'.
    n_top_genes
        Number of top variable genes to use when genes=None and no HVGs
        are annotated.
    layer
        Layer to use for expression values. None uses .X.

    Returns
    -------
    tuple of (correlation_df, adjacency_df)
        correlation_df : pd.DataFrame
            Square correlation matrix (genes x genes).
        adjacency_df : pd.DataFrame
            Binary adjacency matrix (1 where |corr| >= threshold, 0 otherwise).
            Diagonal is set to 0 (no self-loops).

    Notes
    -----
    Results are also stored in:
    - adata.varp['gene_correlations'] : correlation matrix (numpy array)
    - adata.varp['gene_network'] : adjacency matrix (numpy array)

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> corr_df, adj_df = singlet.gene_correlation_network(adata, threshold=0.5)
    >>> adj_df.sum().sum()  # total number of edges * 2
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse
    from scipy.stats import spearmanr

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"gene_correlation_network() requires an AnnData object, got {type(adata).__name__}"
        )

    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")

    if threshold < 0 or threshold > 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    # Get expression matrix
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        mat = adata.layers[layer]
    else:
        mat = adata.X

    if issparse(mat):
        mat = np.asarray(mat.todense())
    else:
        mat = np.asarray(mat)

    # Select genes
    if genes is not None:
        var_names_list = list(adata.var_names)
        gene_idx = []
        valid_genes = []
        for gene in genes:
            if gene in var_names_list:
                gene_idx.append(var_names_list.index(gene))
                valid_genes.append(gene)
        if len(valid_genes) == 0:
            raise ValueError("None of the specified genes found in adata.var_names")
        selected_names = valid_genes
        mat_sub = mat[:, gene_idx]
    else:
        # Use HVGs or top by variance
        if "highly_variable" in adata.var.columns:
            hv_mask = adata.var["highly_variable"].values.astype(bool)
            hv_indices = np.where(hv_mask)[0][:n_top_genes]
            selected_names = list(adata.var_names[hv_indices])
            mat_sub = mat[:, hv_indices]
        else:
            variances = np.var(mat, axis=0)
            n_select = min(n_top_genes, mat.shape[1])
            top_idx = np.argsort(variances)[::-1][:n_select]
            selected_names = [adata.var_names[idx] for idx in top_idx]
            mat_sub = mat[:, top_idx]

    n_genes_selected = len(selected_names)

    # Compute correlation matrix
    if n_genes_selected < 2:
        corr_matrix = np.ones((n_genes_selected, n_genes_selected))
    elif method == "spearman":
        corr_matrix, _ = spearmanr(mat_sub)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])
        elif n_genes_selected == 2:
            # spearmanr returns scalar for 2 variables
            rho = float(corr_matrix) if corr_matrix.ndim == 0 else corr_matrix
            if np.ndim(rho) == 0:
                corr_matrix = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    else:
        corr_matrix = np.corrcoef(mat_sub.T)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])

    # Handle NaN (constant genes)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Build adjacency matrix
    adjacency = (np.abs(corr_matrix) >= threshold).astype(np.float64)
    # Remove self-loops
    np.fill_diagonal(adjacency, 0.0)

    # Create DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=selected_names, columns=selected_names)
    adj_df = pd.DataFrame(adjacency, index=selected_names, columns=selected_names)

    # Store in adata.varp (subset to match adata.var dimension)
    # varp must be n_vars x n_vars — we store full-var-space sparse matrices
    # Actually, varp requires shape (n_vars, n_vars). We store the subset matrices
    # with gene names tracked in uns for reference.
    import scipy.sparse as sp

    n_vars = adata.shape[1]
    var_names_list = list(adata.var_names)

    # Map selected genes to indices in full var space
    full_idx = [var_names_list.index(g) for g in selected_names]

    # Build sparse matrices in full var space
    corr_sparse = sp.lil_matrix((n_vars, n_vars), dtype=np.float64)
    adj_sparse = sp.lil_matrix((n_vars, n_vars), dtype=np.float64)

    for ii, gi in enumerate(full_idx):
        for jj, gj in enumerate(full_idx):
            if corr_matrix[ii, jj] != 0:
                corr_sparse[gi, gj] = corr_matrix[ii, jj]
            if adjacency[ii, jj] != 0:
                adj_sparse[gi, gj] = adjacency[ii, jj]

    adata.varp["gene_correlations"] = corr_sparse.tocsr()
    adata.varp["gene_network"] = adj_sparse.tocsr()

    return corr_df, adj_df
