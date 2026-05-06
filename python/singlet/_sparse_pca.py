"""Sparse PCA for interpretable dimensionality reduction.

Provides singlet.sparse_pca() — wrapper around sklearn SparsePCA that
produces interpretable components with sparse loadings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def sparse_pca(
    adata,
    *,
    n_components: int = 50,
    alpha: float = 1.0,
    random_state: int = 0,
    max_iter: int = 100,
):
    """Compute Sparse PCA for interpretable components.

    Uses sklearn's SparsePCA to produce components with sparse loadings,
    making each component easier to interpret biologically as it loads
    on fewer genes.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should be normalized and log-transformed.
    n_components : int, default 50
        Number of sparse components to compute.
    alpha : float, default 1.0
        Sparsity controlling parameter (L1 penalty). Higher values give
        sparser components.
    random_state : int, default 0
        Random seed for reproducibility.
    max_iter : int, default 100
        Maximum number of iterations.

    Returns
    -------
    anndata.AnnData
        The input adata with:
        - adata.obsm['X_sparse_pca']: cell embeddings (n_cells, n_components)
        - adata.varm['sparse_pca_loadings']: gene loadings (n_genes, n_components)
        - adata.uns['sparse_pca']: dict with parameters

    Raises
    ------
    TypeError
        If adata is not an AnnData object.
    ValueError
        If n_components is invalid.

    Examples
    --------
    >>> import singlet
    >>> singlet.normalize(adata)
    >>> singlet.sparse_pca(adata, n_components=20, alpha=0.5)
    >>> adata.obsm['X_sparse_pca'].shape  # (n_cells, 20)
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.decomposition import SparsePCA as SklearnSparsePCA

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"sparse_pca() requires an AnnData object, got {type(adata).__name__}")

    n_cells, n_genes = adata.shape

    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    n_components = min(n_components, n_cells, n_genes)

    # Get data matrix
    if "highly_variable" in adata.var.columns:
        var_mask = adata.var["highly_variable"].values
        gene_subset = adata[:, var_mask].X
    else:
        var_mask = None
        gene_subset = adata.X

    # Convert to dense if sparse
    if sp.issparse(gene_subset):
        gene_subset = gene_subset.toarray()
    gene_subset = np.asarray(gene_subset, dtype=np.float64)

    # Fit Sparse PCA
    spca = SklearnSparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=random_state,
        max_iter=max_iter,
    )
    embedding = spca.fit_transform(gene_subset)

    # Get loadings (components_ is n_components x n_features)
    components = spca.components_  # (n_components, n_features_used)

    # Store results
    adata.obsm["X_sparse_pca"] = embedding.astype(np.float32)

    # Store loadings - expand to full gene space if using HVGs
    if var_mask is not None:
        full_loadings = np.zeros((n_genes, n_components), dtype=np.float32)
        full_loadings[var_mask, :] = components.T.astype(np.float32)
        adata.varm["sparse_pca_loadings"] = full_loadings
    else:
        adata.varm["sparse_pca_loadings"] = components.T.astype(np.float32)

    adata.uns["sparse_pca"] = {
        "n_components": n_components,
        "alpha": alpha,
        "random_state": random_state,
        "max_iter": max_iter,
    }

    return adata
