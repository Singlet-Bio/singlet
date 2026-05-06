"""Spatial analysis utilities."""

from __future__ import annotations

from anndata import AnnData


def spatial_neighbors(
    adata: AnnData,
    *,
    coord_type: str = "generic",
    spatial_key: str = "spatial",
    n_neighbors: int = 6,
    radius: float | None = None,
    key_added: str = "spatial",
    copy: bool = False,
) -> AnnData | None:
    """Compute a spatial neighbor graph from coordinates.

    Parameters
    ----------
    adata
        Annotated data matrix with spatial coordinates.
    coord_type
        Type of coordinates: 'generic' (any 2D/3D), 'grid' (hexagonal/square grid).
    spatial_key
        Key in .obsm containing spatial coordinates.
    n_neighbors
        Number of spatial neighbors per cell (for KNN mode).
    radius
        Maximum distance for neighbors. If set, uses radius mode
        instead of KNN.
    key_added
        Prefix for keys in .obsp and .uns.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores:
        - ``.obsp['{key_added}_connectivities']``: connectivity matrix
        - ``.obsp['{key_added}_distances']``: distance matrix
        - ``.uns['{key_added}_neighbors']``: parameters
    """
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    adata = adata.copy() if copy else adata

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"'{spatial_key}' not found in .obsm. Store spatial coordinates there first."
        )

    coords = adata.obsm[spatial_key]
    n_obs = coords.shape[0]

    if radius is not None:
        # Radius-based neighbors
        nn = NearestNeighbors(radius=radius, metric="euclidean")
        nn.fit(coords)
        distances_sparse = nn.radius_neighbors_graph(coords, mode="distance")
        connectivity_sparse = nn.radius_neighbors_graph(coords, mode="connectivity")
    else:
        # KNN-based neighbors
        k = min(n_neighbors, n_obs - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        # Build sparse matrices (skip self-neighbor at index 0)
        rows = []
        cols = []
        dists = []

        for i in range(n_obs):
            for j_idx in range(1, k + 1):
                j = indices[i, j_idx]
                rows.append(i)
                cols.append(j)
                dists.append(distances[i, j_idx])

        distances_sparse = csr_matrix((dists, (rows, cols)), shape=(n_obs, n_obs))

        # Symmetrize
        distances_sparse = (distances_sparse + distances_sparse.T) / 2

        # Connectivity (binary)
        connectivity_sparse = distances_sparse.copy()
        connectivity_sparse.data[:] = 1.0

    # Store results
    conn_key = f"{key_added}_connectivities"
    dist_key = f"{key_added}_distances"

    adata.obsp[conn_key] = connectivity_sparse
    adata.obsp[dist_key] = distances_sparse

    adata.uns[f"{key_added}_neighbors"] = {
        "connectivities_key": conn_key,
        "distances_key": dist_key,
        "params": {
            "n_neighbors": n_neighbors,
            "coord_type": coord_type,
            "radius": radius,
        },
    }

    return adata if copy else None
