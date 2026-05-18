# SPDX-License-Identifier: MIT
"""Doublet detection using Scrublet-like algorithm."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def scrublet(
    adata: AnnData,
    *,
    sim_doublet_ratio: float = 2.0,
    n_neighbors: int = 30,
    expected_doublet_rate: float = 0.06,
    n_pcs: int = 30,
    random_state: int = 0,
    threshold: float | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Detect doublets using simulated doublet comparison.

    Simulates artificial doublets by averaging random pairs of cells,
    then scores each observed cell based on its neighborhood composition
    (fraction of neighbors that are simulated doublets).

    Parameters
    ----------
    adata
        Annotated data matrix. Should be normalized log-transformed data.
    sim_doublet_ratio
        Ratio of simulated doublets to observed cells.
    n_neighbors
        Number of neighbors for KNN graph.
    expected_doublet_rate
        Expected fraction of doublets (for automatic thresholding).
    n_pcs
        Number of PCs for dimensionality reduction.
    random_state
        Random seed.
    threshold
        Manual doublet score threshold. If None, uses automatic
        threshold based on bimodal distribution.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Adds to .obs:
        - 'doublet_score': Continuous doublet score [0, 1]
        - 'predicted_doublet': Boolean predicted doublet status
    """
    from scipy.sparse import issparse
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    adata = adata.copy() if copy else adata
    rng = np.random.default_rng(random_state)

    # Get expression matrix
    if issparse(adata.X):
        X = np.asarray(adata.X.todense())
    else:
        X = np.asarray(adata.X)

    n_obs, n_vars = X.shape
    n_sim = int(n_obs * sim_doublet_ratio)

    # Simulate doublets by averaging random pairs
    idx1 = rng.integers(0, n_obs, size=n_sim)
    idx2 = rng.integers(0, n_obs, size=n_sim)
    X_sim = (X[idx1] + X[idx2]) / 2.0

    # Combine observed + simulated
    X_combined = np.vstack([X, X_sim])

    # PCA
    n_pcs_use = min(n_pcs, min(X_combined.shape) - 1)
    pca = PCA(n_components=n_pcs_use, random_state=random_state)
    X_pca = pca.fit_transform(X_combined)

    # KNN
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_pca)

    # For each observed cell, compute fraction of neighbors that are simulated
    distances, indices = nn.kneighbors(X_pca[:n_obs])

    # Cells with index >= n_obs are simulated doublets
    doublet_scores = np.mean(indices >= n_obs, axis=1)

    # Also compute scores for simulated doublets (for threshold estimation)
    _, sim_indices = nn.kneighbors(X_pca[n_obs:])
    sim_scores = np.mean(sim_indices >= n_obs, axis=1)

    # Automatic threshold
    if threshold is None:
        # Use the expected doublet rate to set threshold
        sorted_scores = np.sort(doublet_scores)
        threshold_idx = int((1 - expected_doublet_rate) * n_obs)
        threshold = sorted_scores[min(threshold_idx, n_obs - 1)]

        # Alternative: use median of simulated doublet scores
        sim_median = np.median(sim_scores)
        # Use the more conservative (lower) threshold
        threshold = min(threshold, sim_median)

    adata.obs["doublet_score"] = doublet_scores.astype(np.float32)
    adata.obs["predicted_doublet"] = doublet_scores > threshold

    # Store threshold in uns
    adata.uns["scrublet"] = {"threshold": float(threshold)}

    return adata if copy else None
