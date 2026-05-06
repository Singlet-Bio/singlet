"""Cross-species/dataset correlation analysis.

Provides singlet.cross_species_correlation() — correlate gene programs or
cluster centroids between two datasets, optionally with gene name mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def cross_species_correlation(
    adata1: "AnnData",
    adata2: "AnnData",
    *,
    gene_mapping: dict[str, str] | None = None,
    method: str = "pearson",
    groupby1: str | None = None,
    groupby2: str | None = None,
) -> "pd.DataFrame":
    """Correlate gene programs or cluster centroids between two datasets.

    Computes pairwise correlation between groups (clusters) or between
    the overall expression profiles of two datasets. Useful for comparing
    cell types across species, batches, or experimental conditions.

    Parameters
    ----------
    adata1
        First annotated data matrix.
    adata2
        Second annotated data matrix.
    gene_mapping
        Dictionary mapping gene names in adata1 to gene names in adata2.
        If None, assumes gene names are shared between datasets.
    method
        Correlation method: 'pearson' or 'spearman'.
    groupby1
        Key in adata1.obs for grouping. If provided, computes cluster
        centroids (mean expression per group) before correlation.
    groupby2
        Key in adata2.obs for grouping. If provided, computes cluster
        centroids for adata2.

    Returns
    -------
    pd.DataFrame
        Correlation matrix. Rows correspond to groups/cells from adata1,
        columns from adata2. If groupby is used, index/columns are
        group labels.

    Raises
    ------
    ValueError
        If method is not 'pearson' or 'spearman'.
        If no shared genes are found between datasets.

    Examples
    --------
    >>> import singlet
    >>> # Compare mouse and human datasets
    >>> human = singlet.load("GSE264667")
    >>> mouse = singlet.load("GSE264668")
    >>> mapping = {"GAPDH": "Gapdh", "TP53": "Trp53"}
    >>> corr = singlet.cross_species_correlation(
    ...     human, mouse, gene_mapping=mapping,
    ...     groupby1="cell_type", groupby2="cell_type"
    ... )
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.stats import spearmanr

    valid_methods = ("pearson", "spearman")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    # Determine shared genes
    genes1 = list(adata1.var_names)
    genes2 = list(adata2.var_names)

    if gene_mapping is not None:
        # Map adata1 gene names → adata2 gene names
        mapped_pairs = []
        genes2_set = set(genes2)
        for g1 in genes1:
            g2 = gene_mapping.get(g1, g1)
            if g2 in genes2_set:
                mapped_pairs.append((g1, g2))
    else:
        # Use shared gene names directly
        shared = set(genes1) & set(genes2)
        mapped_pairs = [(g, g) for g in sorted(shared)]

    if not mapped_pairs:
        msg = "No shared genes found between datasets"
        raise ValueError(msg)

    # Get indices for each dataset
    var_idx1 = {name: idx for idx, name in enumerate(genes1)}
    var_idx2 = {name: idx for idx, name in enumerate(genes2)}

    idx1 = [var_idx1[g1] for g1, _ in mapped_pairs]
    idx2 = [var_idx2[g2] for _, g2 in mapped_pairs]

    def _get_matrix(adata, indices):
        """Extract dense submatrix for given gene indices."""
        mat = adata.X[:, indices]
        if sp.issparse(mat):
            mat = mat.toarray()
        return np.asarray(mat, dtype=np.float64)

    def _compute_centroids(mat, groups):
        """Compute mean expression per group."""
        unique_groups = sorted(groups.unique())
        centroids = np.zeros((len(unique_groups), mat.shape[1]), dtype=np.float64)
        for idx, grp in enumerate(unique_groups):
            mask = groups.values == grp
            centroids[idx] = mat[mask].mean(axis=0)
        return centroids, unique_groups

    # Extract submatrices
    mat1 = _get_matrix(adata1, idx1)
    mat2 = _get_matrix(adata2, idx2)

    # Compute centroids if groupby specified
    if groupby1 is not None:
        mat1, labels1 = _compute_centroids(mat1, adata1.obs[groupby1])
    else:
        labels1 = list(adata1.obs_names)

    if groupby2 is not None:
        mat2, labels2 = _compute_centroids(mat2, adata2.obs[groupby2])
    else:
        labels2 = list(adata2.obs_names)

    # Compute pairwise correlation
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]
    corr_matrix = np.zeros((n1, n2), dtype=np.float64)

    if method == "pearson":
        # Standardize rows for fast pearson via dot product
        mat1_centered = mat1 - mat1.mean(axis=1, keepdims=True)
        mat2_centered = mat2 - mat2.mean(axis=1, keepdims=True)

        norms1 = np.linalg.norm(mat1_centered, axis=1, keepdims=True)
        norms2 = np.linalg.norm(mat2_centered, axis=1, keepdims=True)

        # Avoid division by zero
        norms1[norms1 == 0] = 1.0
        norms2[norms2 == 0] = 1.0

        mat1_normed = mat1_centered / norms1
        mat2_normed = mat2_centered / norms2

        corr_matrix = mat1_normed @ mat2_normed.T

    elif method == "spearman":
        for row_idx in range(n1):
            for col_idx in range(n2):
                rho, _ = spearmanr(mat1[row_idx], mat2[col_idx])
                corr_matrix[row_idx, col_idx] = rho if not np.isnan(rho) else 0.0

    return pd.DataFrame(corr_matrix, index=labels1, columns=labels2)
