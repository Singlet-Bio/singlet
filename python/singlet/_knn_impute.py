"""kNN-based expression imputation.

Provides singlet.knn_impute() — for each cell, impute expression by
averaging its k nearest neighbors weighted by distance. Simpler and
faster than MAGIC (no diffusion power steps).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def knn_impute(
    adata: AnnData,
    *,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    layer: str | None = None,
    weights: str = "distance",
) -> AnnData:
    """Impute expression using k-nearest-neighbor averaging.

    For each cell, computes a weighted average of expression values from
    its k nearest neighbors. Distance-weighted averaging gives more
    influence to closer neighbors.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    n_neighbors : int, default 15
        Number of nearest neighbors to use for imputation.
    use_rep : str, default 'X_pca'
        Representation in adata.obsm to use for kNN computation.
    layer : str or None, default None
        Which layer to impute. If None, uses adata.X.
    weights : str, default 'distance'
        Weighting scheme for neighbor contributions:
        - 'distance': weight inversely proportional to distance
        - 'uniform': equal weight for all neighbors

    Returns
    -------
    anndata.AnnData
        The input adata with adata.layers['knn_imputed'] set to the
        imputed expression matrix.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.knn_impute(adata, n_neighbors=15)
    >>> adata.layers['knn_imputed'].shape == adata.X.shape
    True
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    if weights not in ("distance", "uniform"):
        msg = f"weights must be 'distance' or 'uniform', got {weights!r}"
        raise ValueError(msg)

    if use_rep not in adata.obsm:
        msg = f"Representation {use_rep!r} not found in adata.obsm. Run singlet.pca(adata) first."
        raise KeyError(msg)

    # Get expression matrix to impute
    mat = adata.layers[layer] if layer is not None else adata.X
    if mat is None:
        msg = "adata.X is None; provide a layer or set adata.X"
        raise ValueError(msg)

    coords = np.asarray(adata.obsm[use_rep])
    n_cells = coords.shape[0]

    # Cap n_neighbors
    k_actual = min(n_neighbors, n_cells - 1)
    if k_actual < 1:
        # Only one cell — imputed is just itself
        if sp.issparse(mat):
            adata.layers["knn_imputed"] = np.asarray(mat.todense(), dtype=np.float32)
        else:
            adata.layers["knn_imputed"] = np.asarray(mat, dtype=np.float32)
        return adata

    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Convert expression to dense for indexing
    if sp.issparse(mat):
        dense = np.asarray(mat.todense(), dtype=np.float64)
    else:
        dense = np.asarray(mat, dtype=np.float64)

    n_genes = dense.shape[1]
    imputed = np.zeros((n_cells, n_genes), dtype=np.float64)

    if weights == "uniform":
        # Simple average of neighbors
        for idx in range(n_cells):
            neighbor_idx = indices[idx]
            imputed[idx] = dense[neighbor_idx].mean(axis=0)
    else:
        # Distance-weighted average
        for idx in range(n_cells):
            neighbor_idx = indices[idx]
            dists = distances[idx]

            # Convert distances to weights (inverse distance)
            # Add small epsilon to avoid division by zero for self
            eps = 1e-10
            inv_dists = 1.0 / (dists + eps)
            wts = inv_dists / inv_dists.sum()

            imputed[idx] = (dense[neighbor_idx] * wts[:, np.newaxis]).sum(axis=0)

    adata.layers["knn_imputed"] = imputed.astype(np.float32)
    return adata
