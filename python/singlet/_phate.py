# SPDX-License-Identifier: MIT
"""PHATE embedding — Potential of Heat-diffusion for Affinity-based Trajectory Embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    pass


def phate(
    adata: AnnData,
    *,
    n_components: int = 2,
    knn: int = 5,
    decay: int | float = 40,
    t: int | str = "auto",
    random_state: int = 0,
    use_rep: str | None = None,
    copy: bool = False,
) -> AnnData:
    """PHATE embedding.

    Potential of Heat-diffusion for Affinity-based Trajectory Embedding.
    Captures both local and global nonlinear structure in data.

    Steps: kNN → adaptive Gaussian kernel → Markov diffusion →
    potential distance → metric MDS.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Number of embedding dimensions.
    knn
        Number of nearest neighbors for graph construction.
    decay
        Sets decay rate of adaptive kernel bandwidth. Higher values
        lead to sharper transitions.
    t
        Diffusion timescale. Use 'auto' to automatically select
        based on von Neumann entropy.
    random_state
        Random seed for reproducibility.
    use_rep
        Representation to use from ``adata.obsm``. If None, uses
        ``X_pca`` if available, otherwise raw X.
    copy
        Return a copy instead of modifying in place.

    Returns
    -------
    AnnData with ``adata.obsm['X_phate']`` set.
    """
    from sklearn.manifold import MDS
    from sklearn.neighbors import NearestNeighbors

    adata = adata.copy() if copy else adata
    n_cells = adata.n_obs

    # Get input representation
    if use_rep is not None:
        if use_rep not in adata.obsm:
            msg = f"'{use_rep}' not found in adata.obsm"
            raise KeyError(msg)
        X_input = np.asarray(adata.obsm[use_rep])
    elif "X_pca" in adata.obsm:
        X_input = np.asarray(adata.obsm["X_pca"])
    else:
        from scipy.sparse import issparse

        X_input = adata.X
        if issparse(X_input):
            X_input = X_input.toarray()
        X_input = np.asarray(X_input, dtype=np.float64)

    # Step 1: kNN graph
    knn_use = min(knn, n_cells - 1)
    nn = NearestNeighbors(n_neighbors=knn_use, metric="euclidean")
    nn.fit(X_input)
    distances, indices = nn.kneighbors(X_input)

    # Step 2: Adaptive Gaussian kernel with decay
    # Bandwidth: epsilon_i = distance to knn-th neighbor
    epsilon = distances[:, -1].copy()
    epsilon[epsilon == 0] = 1e-10

    # Build kernel matrix (dense for moderate-sized datasets)
    kernel = np.zeros((n_cells, n_cells))
    for idx in range(n_cells):
        for jj in range(knn_use):
            neighbor = indices[idx, jj]
            dist = distances[idx, jj]
            # Adaptive bandwidth
            eps_ij = (epsilon[idx] + epsilon[neighbor]) / 2.0
            # Kernel with decay parameter
            kernel[idx, neighbor] = np.exp(-((dist / eps_ij) ** decay))

    # Symmetrize kernel
    kernel = (kernel + kernel.T) / 2.0

    # Step 3: Markov normalization (row-stochastic)
    row_sums = kernel.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    markov = kernel / row_sums[:, np.newaxis]

    # Step 4: Diffusion — raise markov matrix to power t
    if t == "auto":
        # Von Neumann entropy criterion for automatic t selection
        t_val = _auto_select_t(markov)
    else:
        t_val = int(t)

    # Powered diffusion operator
    diff_op = np.linalg.matrix_power(markov, t_val)

    # Step 5: Potential distance — log transform diffusion operator
    # Clip to avoid log(0)
    diff_op = np.clip(diff_op, 1e-10, None)
    potential = np.log(diff_op)

    # Step 6: Compute pairwise potential distances
    # Use Euclidean distance in potential space
    # For efficiency, use MDS directly on the potential representation
    # Each row of 'potential' is a cell's representation in potential space

    # Step 7: Metric MDS on potential distances
    mds = MDS(
        n_components=n_components,
        dissimilarity="euclidean",
        random_state=random_state,
        normalized_stress="auto",
        max_iter=300,
        n_init=1,
    )

    embedding = mds.fit_transform(potential)

    # Store results
    adata.obsm["X_phate"] = embedding
    adata.uns["phate_params"] = {
        "n_components": n_components,
        "knn": knn,
        "decay": decay,
        "t": t_val,
        "random_state": random_state,
    }

    return adata


def _auto_select_t(markov: np.ndarray, max_t: int = 100) -> int:
    """Automatically select diffusion time via von Neumann entropy.

    Picks the knee point where entropy stabilizes.

    Parameters
    ----------
    markov
        Row-stochastic Markov matrix.
    max_t
        Maximum t to consider.

    Returns
    -------
    Optimal diffusion time.
    """
    powered = markov.copy()
    entropies = []

    for t_candidate in range(1, max_t + 1):
        if t_candidate > 1:
            powered = powered @ markov

        # Von Neumann entropy of the diffusion operator
        # Approximate via row-wise entropy
        row_entropy = np.zeros(powered.shape[0])
        for idx in range(powered.shape[0]):
            row = powered[idx]
            row_pos = row[row > 1e-10]
            if len(row_pos) > 0:
                row_pos = row_pos / row_pos.sum()
                row_entropy[idx] = -np.sum(row_pos * np.log(row_pos))

        entropies.append(float(np.mean(row_entropy)))

        # Early stopping: if entropy barely changes
        if len(entropies) >= 3:
            delta = abs(entropies[-1] - entropies[-2])
            prev_delta = abs(entropies[-2] - entropies[-3])
            if delta < 0.01 * prev_delta and t_candidate >= 5:
                return t_candidate

    # Default to knee point
    if len(entropies) >= 2:
        # Find where rate of change drops below threshold
        diffs = np.diff(entropies)
        if len(diffs) > 1:
            ratios = np.abs(diffs[1:]) / (np.abs(diffs[:-1]) + 1e-10)
            knee_candidates = np.where(ratios < 0.5)[0]
            if len(knee_candidates) > 0:
                return int(knee_candidates[0] + 2)

    return min(10, max_t)
