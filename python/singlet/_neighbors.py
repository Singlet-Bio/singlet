# SPDX-License-Identifier: MIT
"""k-Nearest neighbors graph construction for AnnData objects.

Provides singlet.neighbors() — computes a kNN graph in PCA space,
enabling downstream clustering (Leiden) and visualization (UMAP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


def neighbors(
    adata,
    *,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    metric: str = "euclidean",
    use_rep: str = "X_pca",
    inplace: bool = True,
) -> Optional[dict]:
    """Compute k-nearest neighbors graph.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have `obsm[use_rep]` (run pca() first).
    n_neighbors : int, default 15
        Number of nearest neighbors.
    n_pcs : int or None, default None
        Number of PCs to use. If None, uses all available PCs in use_rep.
    metric : str, default "euclidean"
        Distance metric. Supports any metric from scipy.spatial.distance
        (e.g., "euclidean", "cosine", "cityblock").
    use_rep : str, default "X_pca"
        Key in adata.obsm for the representation to use.
    inplace : bool, default True
        If True, stores connectivities and distances in adata.obsp
        and parameters in adata.uns['neighbors']. Returns None.
        If False, returns dict with 'connectivities' and 'distances'.

    Returns
    -------
    dict or None
        Dict with 'connectivities' and 'distances' sparse matrices
        (if inplace=False), or None (if inplace=True).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> adata.obsp['connectivities']  # sparse kNN graph
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"neighbors() requires an AnnData object, got {type(adata).__name__}")

    if use_rep not in adata.obsm:
        raise KeyError(f"'{use_rep}' not found in adata.obsm. Run singlet.pca(adata) first.")

    X = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    if n_pcs is not None:
        X = X[:, :n_pcs]

    n_cells = X.shape[0]
    k = min(n_neighbors, n_cells - 1)

    if k < 1:
        raise ValueError(f"Need at least 2 cells for neighbors, got {n_cells}.")

    # Compute pairwise distances in chunks to manage memory
    # For datasets < 10000 cells, compute full distance matrix
    # For larger, use brute-force kNN with chunking
    if n_cells <= 10000:
        D = cdist(X, X, metric=metric)
        # Get k nearest neighbors for each cell (excluding self)
        indices = np.zeros((n_cells, k), dtype=np.int64)
        distances = np.zeros((n_cells, k), dtype=np.float64)
        for i in range(n_cells):
            d = D[i]
            d[i] = np.inf  # exclude self
            nn_idx = np.argpartition(d, k)[:k]
            nn_idx = nn_idx[np.argsort(d[nn_idx])]
            indices[i] = nn_idx
            distances[i] = d[nn_idx]
    else:
        # Chunked computation for large datasets
        chunk_size = 1000
        indices = np.zeros((n_cells, k), dtype=np.int64)
        distances = np.zeros((n_cells, k), dtype=np.float64)
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            D_chunk = cdist(X[start:end], X, metric=metric)
            for i in range(end - start):
                d = D_chunk[i]
                d[start + i] = np.inf  # exclude self
                nn_idx = np.argpartition(d, k)[:k]
                nn_idx = nn_idx[np.argsort(d[nn_idx])]
                indices[start + i] = nn_idx
                distances[start + i] = d[nn_idx]

    # Build sparse distance matrix
    row_idx = np.repeat(np.arange(n_cells), k)
    col_idx = indices.ravel()
    dist_vals = distances.ravel()
    dist_matrix = sp.csr_matrix((dist_vals, (row_idx, col_idx)), shape=(n_cells, n_cells))

    # Build connectivity matrix (Gaussian kernel on distances)
    sigma = np.median(distances[:, -1]) if distances[:, -1].max() > 0 else 1.0
    conn_vals = np.exp(-(dist_vals**2) / (2 * sigma**2))
    conn_matrix = sp.csr_matrix((conn_vals, (row_idx, col_idx)), shape=(n_cells, n_cells))

    # Symmetrize
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    conn_matrix = (conn_matrix + conn_matrix.T) / 2

    result = {
        "connectivities": conn_matrix,
        "distances": dist_matrix,
    }

    if inplace:
        adata.obsp["connectivities"] = conn_matrix
        adata.obsp["distances"] = dist_matrix
        adata.uns["neighbors"] = {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "use_rep": use_rep,
            "n_pcs": n_pcs or X.shape[1],
        }
        return None
    else:
        return result
