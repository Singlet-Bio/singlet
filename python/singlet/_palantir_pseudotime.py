# SPDX-License-Identifier: MIT
"""Palantir-inspired pseudotime computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    pass


def palantir_pseudotime(
    adata: AnnData,
    root_cell: int | str,
    *,
    n_neighbors: int = 30,
    use_rep: str = "X_pca",
    n_components: int = 10,
    n_waypoints: int = 500,
    random_state: int = 0,
    copy: bool = False,
) -> AnnData:
    """Palantir-inspired pseudotime computation.

    Computes diffusion pseudotime with waypoints for multi-lineage
    trajectories. Steps: diffusion map → select waypoints → shortest
    paths from root cell.

    Parameters
    ----------
    adata
        Annotated data matrix.
    root_cell
        Root cell for pseudotime computation. Can be an integer index
        into obs or a string matching an obs_name.
    n_neighbors
        Number of neighbors for kNN graph construction.
    use_rep
        Representation to use from ``adata.obsm``.
    n_components
        Number of diffusion components.
    n_waypoints
        Number of waypoints for trajectory inference.
    random_state
        Random seed for reproducibility.
    copy
        Return a copy instead of modifying in place.

    Returns
    -------
    AnnData with ``adata.obs['palantir_pseudotime']`` and
    ``adata.obsm['palantir_waypoint_distances']`` set.
    """
    from scipy.sparse import csr_matrix, issparse
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import NearestNeighbors

    adata = adata.copy() if copy else adata

    n_cells = adata.n_obs

    # Resolve root cell
    if isinstance(root_cell, str):
        if root_cell in adata.obs_names:
            root_idx = int(np.where(adata.obs_names == root_cell)[0][0])
        else:
            msg = f"root_cell '{root_cell}' not found in adata.obs_names"
            raise KeyError(msg)
    else:
        root_idx = int(root_cell)
        if root_idx < 0 or root_idx >= n_cells:
            msg = f"root_cell index {root_idx} out of bounds [0, {n_cells})"
            raise IndexError(msg)

    # Get representation
    if use_rep not in adata.obsm:
        msg = f"'{use_rep}' not found in adata.obsm. Run the appropriate embedding first."
        raise KeyError(msg)

    X_rep = np.asarray(adata.obsm[use_rep])

    # Limit dimensions if needed
    if X_rep.shape[1] > n_components:
        X_rep = X_rep[:, :n_components]

    # Build kNN graph
    n_neighbors_use = min(n_neighbors, n_cells - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors_use, metric="euclidean")
    nn.fit(X_rep)
    distances, indices = nn.kneighbors(X_rep)

    # Build symmetric adjacency with adaptive kernel
    # Adaptive bandwidth: sigma_i = distance to k-th neighbor
    sigma = distances[:, -1].copy()
    sigma[sigma == 0] = 1e-10

    # Construct weighted adjacency
    rows = []
    cols = []
    vals = []
    for idx in range(n_cells):
        for jj in range(n_neighbors_use):
            neighbor = indices[idx, jj]
            dist = distances[idx, jj]
            # Adaptive Gaussian kernel
            weight = np.exp(-(dist**2) / (sigma[idx] * sigma[neighbor] + 1e-10))
            rows.append(idx)
            cols.append(neighbor)
            vals.append(weight)

    adj = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    # Symmetrize
    adj = (adj + adj.T) / 2

    # Build transition matrix (row-stochastic)
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    diag_inv = csr_matrix(
        (1.0 / row_sums, (range(n_cells), range(n_cells))), shape=(n_cells, n_cells)
    )
    transition = diag_inv @ adj

    # Compute diffusion components via eigendecomposition
    from scipy.sparse.linalg import eigsh

    n_comps_compute = min(n_components + 1, n_cells - 2)
    try:
        eigenvalues, eigenvectors = eigsh(transition, k=n_comps_compute, which="LM")
    except Exception:
        # Fall back to dense
        transition_dense = transition.toarray()
        eig_vals, eig_vecs = np.linalg.eigh(transition_dense)
        idx_sorted = np.argsort(eig_vals)[::-1]
        eigenvalues = eig_vals[idx_sorted[:n_comps_compute]]
        eigenvectors = eig_vecs[:, idx_sorted[:n_comps_compute]]

    # Sort by eigenvalue magnitude (descending)
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Skip trivial eigenvalue (=1) component
    if np.abs(eigenvalues[0] - 1.0) < 1e-5 and eigenvectors.shape[1] > 1:
        ms_data = eigenvectors[:, 1:]
        ms_eigenvalues = eigenvalues[1:]
    else:
        ms_data = eigenvectors
        ms_eigenvalues = eigenvalues

    # Scale by eigenvalues (multiscale diffusion)
    n_use = min(n_components, ms_data.shape[1])
    ms_data = ms_data[:, :n_use] * ms_eigenvalues[:n_use][np.newaxis, :]

    # Select waypoints using max-min sampling on diffusion space
    rng = np.random.default_rng(random_state)
    n_wp = min(n_waypoints, n_cells)

    waypoint_indices = np.zeros(n_wp, dtype=int)
    waypoint_indices[0] = root_idx

    # Max-min sampling: each subsequent waypoint maximizes minimum distance
    min_dists = np.full(n_cells, np.inf)
    for wp_i in range(1, n_wp):
        last_wp = waypoint_indices[wp_i - 1]
        dists_to_last = np.sum((ms_data - ms_data[last_wp]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        # Add small noise to break ties
        min_dists_noisy = min_dists + rng.uniform(0, 1e-10, size=n_cells)
        # Don't reselect already chosen waypoints
        min_dists_noisy[waypoint_indices[:wp_i]] = -1
        waypoint_indices[wp_i] = np.argmax(min_dists_noisy)

    # Compute shortest path distances from root in the graph
    # Convert adjacency to distance graph (invert weights)
    adj_copy = adj.copy()
    if issparse(adj_copy):
        adj_copy = adj_copy.tocsr()
        # Convert weights to distances
        adj_copy.data = 1.0 / (adj_copy.data + 1e-10)
    else:
        adj_copy = 1.0 / (adj_copy + 1e-10)

    # Compute shortest paths from root
    dist_from_root = shortest_path(
        adj_copy,
        method="D",
        directed=False,
        indices=root_idx,
    )

    # Handle unreachable cells
    dist_from_root = dist_from_root.flatten()
    unreachable = np.isinf(dist_from_root)
    if unreachable.any():
        max_finite = np.max(dist_from_root[~unreachable]) if (~unreachable).any() else 1.0
        dist_from_root[unreachable] = max_finite

    # Normalize pseudotime to [0, 1]
    pt_min = dist_from_root.min()
    pt_max = dist_from_root.max()
    if pt_max - pt_min > 0:
        pseudotime = (dist_from_root - pt_min) / (pt_max - pt_min)
    else:
        pseudotime = np.zeros(n_cells)

    # Compute waypoint distances matrix (cells × waypoints)
    wp_data = ms_data[waypoint_indices]
    waypoint_dists = np.zeros((n_cells, n_wp))
    for wi in range(n_wp):
        waypoint_dists[:, wi] = np.sqrt(np.sum((ms_data - wp_data[wi]) ** 2, axis=1))

    # Store results
    adata.obs["palantir_pseudotime"] = pseudotime
    adata.obsm["palantir_waypoint_distances"] = waypoint_dists
    adata.uns["palantir_params"] = {
        "root_cell": root_idx,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "n_waypoints": n_wp,
    }

    return adata
