"""Weighted Nearest Neighbors (WNN) multi-modal integration.

Implements Seurat v4-style weighted nearest neighbors for integrating multiple
modalities (e.g. RNA + protein) into a unified nearest-neighbor graph. Per-cell
modality weights are learned based on how well each modality's local neighborhood
predicts structure across other modalities.

Reference:
    Hao et al., "Integrated analysis of multimodal single-cell data"
    Cell 184, 3573-3587 (2021).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray


def weighted_nearest_neighbors(
    adata: AnnData,
    modalities: list[str],
    *,
    weights: NDArray[np.floating] | None = None,
    n_neighbors: int = 20,
) -> AnnData:
    """Compute a weighted nearest-neighbor graph from multiple modalities.

    For each cell, learns how much to trust each modality based on how well
    its local structure predicts neighbors in the other modalities, then
    constructs a unified neighbor graph using those per-cell weights.

    Parameters
    ----------
    adata
        Annotated data matrix. Must contain the embeddings specified in
        *modalities* as keys in ``adata.obsm``.
    modalities
        List of ``adata.obsm`` keys to integrate (e.g.
        ``['X_pca', 'X_protein']``). Each must be a 2-D array with shape
        ``(n_cells, n_features_m)``.
    weights
        Fixed modality weights, array of shape ``(n_modalities,)``. If
        ``None`` (default), per-cell weights are learned automatically.
        When provided, weights are broadcast identically to every cell and
        normalized to sum to 1.
    n_neighbors
        Number of nearest neighbors to use for each modality graph.
        Clamped to ``n_cells - 1`` when the dataset is small.

    Returns
    -------
    AnnData
        The input *adata* object, modified in-place with the following
        additions:

        - ``adata.obsp['wnn_connectivities']`` — sparse CSR matrix of shape
          ``(n_cells, n_cells)`` holding the combined weighted similarities.
        - ``adata.obsm['wnn_weights']`` — ndarray of shape
          ``(n_cells, n_modalities)`` with the per-cell modality weights.

    Raises
    ------
    KeyError
        If any key in *modalities* is missing from ``adata.obsm``.
    ValueError
        If *weights* is provided but its length does not match the number of
        modalities.

    Examples
    --------
    >>> import anndata
    >>> import numpy as np
    >>> from singlet import weighted_nearest_neighbors
    >>> rng = np.random.default_rng(42)
    >>> adata = anndata.AnnData(np.zeros((100, 1)))
    >>> adata.obsm['X_pca'] = rng.standard_normal((100, 50))
    >>> adata.obsm['X_protein'] = rng.standard_normal((100, 20))
    >>> adata = weighted_nearest_neighbors(
    ...     adata, ['X_pca', 'X_protein'], n_neighbors=15
    ... )
    >>> adata.obsp['wnn_connectivities'].shape
    (100, 100)
    >>> adata.obsm['wnn_weights'].shape
    (100, 2)
    """
    n_modalities = len(modalities)
    n_cells = adata.n_obs

    # --- Validate inputs ---
    for key in modalities:
        if key not in adata.obsm:
            msg = (
                f"Modality key {key!r} not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
            raise KeyError(msg)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n_modalities:
            msg = (
                f"Length of weights ({weights.shape[0]}) must match "
                f"number of modalities ({n_modalities})."
            )
            raise ValueError(msg)

    k = min(n_neighbors, n_cells - 1)

    # --- Step 1: Compute kNN graphs for each modality ---
    knn_indices: list[NDArray] = []
    knn_distances: list[NDArray] = []

    for key in modalities:
        embedding = np.asarray(adata.obsm[key])
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(embedding)
        distances, indices = nn.kneighbors(embedding)
        knn_indices.append(indices)
        knn_distances.append(distances)

    # --- Step 2: Convert distances to kernel similarities ---
    similarity_matrices: list[csr_matrix] = []

    for mod_idx in range(n_modalities):
        distances = knn_distances[mod_idx]
        indices = knn_indices[mod_idx]

        # Bandwidth = median distance to the k-th neighbor (last column)
        bandwidth = np.median(distances[:, -1])
        if bandwidth == 0:
            bandwidth = 1.0

        # Gaussian kernel
        sim_values = np.exp(-(distances**2) / (2.0 * bandwidth**2))

        # Build sparse similarity matrix (n_cells x n_cells)
        row_idx = np.repeat(np.arange(n_cells), k)
        col_idx = indices.ravel()
        data = sim_values.ravel()

        sim_sparse = csr_matrix((data, (row_idx, col_idx)), shape=(n_cells, n_cells))
        similarity_matrices.append(sim_sparse)

    # --- Step 3: Determine per-cell modality weights ---
    if n_modalities == 1:
        # Trivial case: single modality
        cell_weights = np.ones((n_cells, 1), dtype=np.float64)

    elif weights is not None:
        # Fixed weights: normalize and broadcast to all cells
        normalized = weights / weights.sum()
        cell_weights = np.tile(normalized, (n_cells, 1))

    else:
        # Auto-learn per-cell weights
        cell_weights = np.zeros((n_cells, n_modalities), dtype=np.float64)

        for mod_idx in range(n_modalities):
            indices = knn_indices[mod_idx]  # (n_cells, k)

            # For each cell, score = mean similarity of its mod_idx neighbors
            # in all OTHER modalities (vectorized)
            scores = np.zeros(n_cells, dtype=np.float64)

            # Flat index arrays for batch sparse lookup
            row_idx = np.repeat(np.arange(n_cells), k)
            col_idx = indices.ravel()

            for other_idx in range(n_modalities):
                if other_idx == mod_idx:
                    continue
                other_sim = similarity_matrices[other_idx]

                # Batch extract: similarities from each cell to its neighbors
                sim_vals = np.asarray(other_sim[row_idx, col_idx]).ravel()
                # Reshape to (n_cells, k) and take mean per cell
                scores += sim_vals.reshape(n_cells, k).mean(axis=1)

            cell_weights[:, mod_idx] = scores

        # Normalize weights per cell to sum to 1
        row_sums = cell_weights.sum(axis=1, keepdims=True)
        # Avoid division by zero for cells where all scores are 0
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        cell_weights = cell_weights / row_sums

    # --- Step 4: Combine similarity matrices using per-cell weights ---
    # For efficiency, build the combined graph from the union of all kNN edges
    # Combined weight for edge (i,j) = sum_m(weights[i,m] * similarity_m[i,j])

    # Collect all possible edges from all modalities
    row_list = []
    col_list = []
    data_list = []

    for mod_idx in range(n_modalities):
        indices = knn_indices[mod_idx]
        sim = similarity_matrices[mod_idx]

        row_idx = np.repeat(np.arange(n_cells), k)
        col_idx = indices.ravel()

        # Per-cell weights for this modality, repeated for each neighbor
        mod_weights = cell_weights[:, mod_idx]
        per_edge_weights = np.repeat(mod_weights, k)

        # Extract similarities vectorized via sparse fancy indexing
        edge_sims = np.asarray(sim[row_idx, col_idx]).ravel()

        weighted_sims = per_edge_weights * edge_sims

        row_list.append(row_idx)
        col_list.append(col_idx)
        data_list.append(weighted_sims)

    # Concatenate and build combined sparse matrix
    all_rows = np.concatenate(row_list)
    all_cols = np.concatenate(col_list)
    all_data = np.concatenate(data_list)

    # csr_matrix sums duplicate entries automatically
    combined = csr_matrix((all_data, (all_rows, all_cols)), shape=(n_cells, n_cells))

    # --- Step 5 & 6: Store results ---
    adata.obsp["wnn_connectivities"] = combined
    adata.obsm["wnn_weights"] = cell_weights

    return adata
