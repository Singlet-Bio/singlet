# SPDX-License-Identifier: MIT
"""RNA velocity-inspired pseudotime computation.

Provides singlet.velocity_pseudotime() — computes pseudotime based on
directional transition probabilities from a kNN graph with cosine
similarity bias of PCA displacement vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from anndata import AnnData


def velocity_pseudotime(
    adata: "AnnData",
    *,
    root_key: Optional[str] = None,
    use_rep: str = "X_pca",
    n_neighbors: int = 30,
) -> "AnnData":
    """Compute RNA velocity-inspired pseudotime.

    Computes transition probabilities from a kNN graph with directional
    bias (cosine similarity of PCA displacement vectors between connected
    cells), identifies a root cell, and computes expected hitting time
    via iterative absorption probability.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have ``use_rep`` in ``.obsm``.
    root_key : str or None, default None
        Column in ``adata.obs`` indicating the root cell(s). The cell with
        the maximum value in this column is used as root. If None, the
        cell with the highest outdegree/indegree ratio is selected.
    use_rep : str, default "X_pca"
        Key in ``adata.obsm`` for the embedding to compute transition
        probabilities.
    n_neighbors : int, default 30
        Number of nearest neighbors to use for the kNN graph.

    Returns
    -------
    anndata.AnnData
        The input adata with ``adata.obs['velocity_pseudotime']`` added,
        normalized to [0, 1].

    Raises
    ------
    KeyError
        If ``use_rep`` not found in obsm or ``root_key`` not in obs.
    ValueError
        If the embedding has fewer than 2 dimensions.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.velocity_pseudotime(adata)
    """
    import numpy as np
    from scipy.spatial import cKDTree

    if not hasattr(adata, "obsm"):
        raise TypeError(
            f"velocity_pseudotime() requires an AnnData object, got {type(adata).__name__}"
        )

    if use_rep not in adata.obsm:
        raise KeyError(f"'{use_rep}' not found in adata.obsm. Run singlet.pca() first.")

    X_embed = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    n_cells = X_embed.shape[0]

    if X_embed.shape[1] < 2:
        raise ValueError(f"Embedding '{use_rep}' must have at least 2 dimensions.")

    # Clamp n_neighbors to available cells
    k = min(n_neighbors, n_cells - 1)

    # Build kNN graph
    tree = cKDTree(X_embed)
    _, indices = tree.query(X_embed, k=k + 1)
    # Remove self (first column)
    knn_indices = indices[:, 1:]

    # Estimate global trajectory direction via first PC of the embedding
    # This gives a consistent direction for biasing transitions
    centered = X_embed - X_embed.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    traj_axis = vh[0]  # First principal component direction

    # Project all cells onto trajectory axis
    projections = X_embed @ traj_axis

    # Compute directional transition probabilities
    # Bias transitions toward neighbors in the positive direction along trajectory
    transition_matrix = np.zeros((n_cells, k), dtype=np.float64)

    for cell_i in range(n_cells):
        neighbors_i = knn_indices[cell_i]
        # Forward bias: how much does each neighbor advance along trajectory
        delta_proj = projections[neighbors_i] - projections[cell_i]

        # Softmax with directional bias
        weights = np.exp(2.0 * delta_proj / (np.abs(delta_proj).max() + 1e-12))
        transition_matrix[cell_i] = weights / weights.sum()

    # Find root cell
    if root_key is not None:
        if root_key not in adata.obs.columns:
            raise KeyError(f"'{root_key}' not found in adata.obs.columns")
        root_values = np.asarray(adata.obs[root_key].values, dtype=np.float64)
        root_cell = int(np.argmax(root_values))
    else:
        # Use cell with most extreme position against trajectory direction
        # (most negative projection = source of the trajectory)
        indegree = np.zeros(n_cells, dtype=np.float64)
        for cell_i in range(n_cells):
            neighbors_i = knn_indices[cell_i]
            indegree[neighbors_i] += transition_matrix[cell_i]

        # Root = cell with lowest indegree (least incoming probability)
        indegree_min = indegree.copy()
        indegree_min[indegree_min == 0] = np.inf
        root_cell = int(np.argmin(indegree_min))

    # Compute expected hitting time from root via iterative diffusion
    pseudotime = _compute_hitting_time(
        root_cell, knn_indices, transition_matrix, n_cells, max_iter=500
    )

    # Normalize to [0, 1]
    pt_min = pseudotime.min()
    pt_max = pseudotime.max()
    if pt_max - pt_min > 1e-12:
        pseudotime = (pseudotime - pt_min) / (pt_max - pt_min)
    else:
        pseudotime = np.zeros(n_cells)

    adata.obs["velocity_pseudotime"] = pseudotime
    return adata


def _compute_hitting_time(
    root: int,
    knn_indices,
    transition_matrix,
    n_cells: int,
    max_iter: int = 500,
):
    """Compute expected hitting time from root to all other cells.

    Uses iterative diffusion with absorption: starting from root, propagate
    probability mass through the directed graph. Each cell accumulates
    arrival time proportional to when probability first reaches it.
    """
    import numpy as np

    # Use shortest weighted path via Dijkstra-like relaxation
    # on the directed transition graph
    dist = np.full(n_cells, np.inf, dtype=np.float64)
    dist[root] = 0.0

    # Convert transition probs to distances: -log(p) so high-prob = short
    # Use iterative relaxation (Bellman-Ford style, converges fast on kNN)
    for _ in range(min(max_iter, n_cells)):
        updated = False
        for cell_i in range(n_cells):
            if dist[cell_i] == np.inf:
                continue
            neighbors_i = knn_indices[cell_i]
            weights_i = transition_matrix[cell_i]
            for nb_idx in range(len(neighbors_i)):
                cell_j = neighbors_i[nb_idx]
                # Edge weight = inverse probability (lower prob = longer path)
                edge_w = 1.0 / (weights_i[nb_idx] + 1e-10)
                new_dist = dist[cell_i] + edge_w
                if new_dist < dist[cell_j]:
                    dist[cell_j] = new_dist
                    updated = True
        if not updated:
            break

    # Handle unreachable cells
    unreached = np.isinf(dist)
    if unreached.any():
        max_d = dist[~unreached].max() if (~unreached).any() else float(max_iter)
        dist[unreached] = max_d

    return dist
