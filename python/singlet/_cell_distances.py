"""Pairwise cell and group distance computation.

Provides singlet.cell_distances() — compute distances between cells or
between cluster centroids using various metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import scipy.sparse as sp
    from anndata import AnnData


def cell_distances(
    adata: "AnnData",
    *,
    groupby: str | None = None,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    n_neighbors: int = 15,
) -> "pd.DataFrame | sp.csr_matrix":
    """Compute pairwise distances between cells or group centroids.

    When ``groupby`` is specified, computes distances between group centroids
    (mean representations) and returns a labeled DataFrame. When ``groupby``
    is None, computes a sparse kNN distance matrix for scalability.

    Parameters
    ----------
    adata
        Annotated data matrix with a representation in ``.obsm[use_rep]``.
    groupby
        Column in ``adata.obs`` to group cells by. If provided, computes
        centroid-to-centroid distances and returns a DataFrame.
        If None, computes cell-level kNN distances (sparse).
    use_rep
        Key in ``adata.obsm`` for the cell representation.
    metric
        Distance metric: 'euclidean', 'cosine', or 'correlation'.
    n_neighbors
        Number of neighbors for kNN approximation (cell-level only).

    Returns
    -------
    pd.DataFrame or scipy.sparse.csr_matrix
        If ``groupby`` is set: DataFrame with group×group distances.
        If ``groupby`` is None: sparse CSR matrix stored in
        ``adata.obsp['distances_{metric}']``.

    Examples
    --------
    >>> import singlet
    >>> # Group-level distances
    >>> dist_df = singlet.cell_distances(adata, groupby="leiden")
    >>> dist_df.loc["0", "1"]  # distance between clusters 0 and 1

    >>> # Cell-level kNN distances
    >>> sparse_dist = singlet.cell_distances(adata, metric="cosine")
    >>> adata.obsp["distances_cosine"]
    """
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist, pdist, squareform

    valid_metrics = ("euclidean", "cosine", "correlation")
    if metric not in valid_metrics:
        msg = f"metric must be one of {valid_metrics}, got {metric!r}"
        raise ValueError(msg)

    if use_rep not in adata.obsm:
        msg = f"Representation {use_rep!r} not found in adata.obsm"
        raise KeyError(msg)

    rep = np.asarray(adata.obsm[use_rep], dtype=np.float64)

    if groupby is not None:
        # Centroid distance mode
        if groupby not in adata.obs.columns:
            msg = f"Key {groupby!r} not found in adata.obs"
            raise KeyError(msg)

        groups = adata.obs[groupby]
        unique_groups = sorted(groups.unique(), key=str)
        n_groups = len(unique_groups)

        # Compute centroids
        centroids = np.zeros((n_groups, rep.shape[1]), dtype=np.float64)
        for idx, group in enumerate(unique_groups):
            mask = np.asarray(groups == group)
            centroids[idx] = rep[mask].mean(axis=0)

        # Compute pairwise distances
        dist_matrix = squareform(pdist(centroids, metric=metric))

        labels = [str(g) for g in unique_groups]
        result = pd.DataFrame(dist_matrix, index=labels, columns=labels)
        result.index.name = groupby
        result.columns.name = groupby

        return result

    # Cell-level kNN distance mode
    n_cells = rep.shape[0]
    k = min(n_neighbors, n_cells - 1)

    if k <= 0:
        # Trivial case: single cell
        sparse_dist = sp.csr_matrix((n_cells, n_cells), dtype=np.float64)
        adata.obsp[f"distances_{metric}"] = sparse_dist
        return sparse_dist

    # Build kNN using brute-force for small datasets or chunked for larger
    # For efficiency, process in chunks
    chunk_size = 1000
    rows = []
    cols = []
    vals = []

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = rep[start:end]

        # Compute distances from chunk to all cells
        dists = cdist(chunk, rep, metric=metric)

        for local_idx in range(end - start):
            global_idx = start + local_idx
            row_dists = dists[local_idx]
            # Exclude self
            row_dists[global_idx] = np.inf
            # Get k nearest neighbors
            nn_idx = np.argpartition(row_dists, k)[:k]
            nn_dists = row_dists[nn_idx]

            rows.extend([global_idx] * k)
            cols.extend(nn_idx.tolist())
            vals.extend(nn_dists.tolist())

    sparse_dist = sp.csr_matrix(
        (np.array(vals, dtype=np.float64), (rows, cols)),
        shape=(n_cells, n_cells),
    )

    adata.obsp[f"distances_{metric}"] = sparse_dist
    return sparse_dist
