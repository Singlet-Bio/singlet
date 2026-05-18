# SPDX-License-Identifier: MIT
"""Spectral clustering for single-cell data.

Provides singlet.spectral_clustering() — partition cells using spectral
graph theory via sklearn's SpectralClustering with kNN or RBF affinity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def spectral_clustering(
    adata: "AnnData",
    *,
    n_clusters: int = 10,
    use_rep: str = "X_pca",
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 10,
    random_state: int = 0,
) -> "AnnData":
    """Spectral clustering of cells.

    Uses spectral decomposition of a similarity graph (kNN or RBF) to
    partition cells into clusters. This captures non-convex cluster shapes
    better than k-means alone.

    Parameters
    ----------
    adata
        Annotated data matrix with a representation in adata.obsm[use_rep].
    n_clusters
        Number of clusters to find.
    use_rep
        Key in adata.obsm to use as cell representation.
        Default is 'X_pca' (PCA coordinates).
    affinity
        Affinity type for constructing the similarity graph:
        - 'nearest_neighbors': kNN graph (default, good for scRNA-seq).
        - 'rbf': radial basis function kernel.
    n_neighbors
        Number of neighbors when affinity='nearest_neighbors'.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        The input adata with adata.obs['spectral_clusters'] added,
        containing cluster labels as a categorical Series.

    Raises
    ------
    ValueError
        If use_rep is not found in adata.obsm.
        If affinity is not 'nearest_neighbors' or 'rbf'.
        If n_clusters < 2 or n_clusters > n_cells.
        If n_neighbors < 2.
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import SpectralClustering as _SpectralClustering

    # --- Validate inputs ---
    if affinity not in ("nearest_neighbors", "rbf"):
        msg = f"affinity must be 'nearest_neighbors' or 'rbf', got '{affinity}'"
        raise ValueError(msg)

    if use_rep not in adata.obsm:
        msg = (
            f"Representation '{use_rep}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
        raise ValueError(msg)

    n_cells = adata.n_obs
    if n_clusters < 2:
        msg = f"n_clusters must be >= 2, got {n_clusters}"
        raise ValueError(msg)
    if n_clusters > n_cells:
        msg = f"n_clusters ({n_clusters}) must be <= number of cells ({n_cells})"
        raise ValueError(msg)

    if n_neighbors < 2:
        msg = f"n_neighbors must be >= 2, got {n_neighbors}"
        raise ValueError(msg)

    # --- Extract representation ---
    X = np.asarray(adata.obsm[use_rep])

    # --- Run spectral clustering ---
    sc = _SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        random_state=random_state,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(X)

    # --- Store results ---
    adata.obs["spectral_clusters"] = pd.Categorical([str(x) for x in labels])

    return adata
