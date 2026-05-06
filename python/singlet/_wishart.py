"""Cluster membership confidence scoring."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def wishart_test(
    adata: AnnData,
    *,
    groupby: str = "leiden",
    use_rep: str = "X_pca",
    n_pcs: int | None = None,
    key_added: str = "cluster_confidence",
    copy: bool = False,
) -> AnnData | None:
    """Score per-cell cluster membership confidence.

    For each cell, computes the ratio of mean distance to its own cluster
    vs mean distance to the nearest other cluster. Low values indicate
    confident assignment, high values indicate borderline cells.

    Score = (mean_dist_own / mean_dist_nearest_other).
    Values < 1 mean well-assigned; values > 1 mean possibly misassigned.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Key in .obs with cluster labels.
    use_rep
        Key in .obsm.
    n_pcs
        Number of components to use.
    key_added
        Key for storing scores in .obs.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores confidence scores in .obs.
    """
    adata = adata.copy() if copy else adata

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")
    if use_rep not in adata.obsm:
        raise KeyError(f"'{use_rep}' not found in .obsm.")

    X = adata.obsm[use_rep]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    labels = adata.obs[groupby].values
    unique_labels = np.unique(labels)
    n_obs = X.shape[0]

    if len(unique_labels) < 2:
        adata.obs[key_added] = np.zeros(n_obs, dtype=np.float32)
        return adata if copy else None

    # Compute cluster centroids
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = X[mask].mean(axis=0)

    # For each cell: distance to own centroid / distance to nearest other centroid
    confidence = np.zeros(n_obs, dtype=np.float32)

    for i in range(n_obs):
        own_label = labels[i]
        own_dist = np.sqrt(np.sum((X[i] - centroids[own_label]) ** 2))

        # Find nearest other cluster centroid
        min_other_dist = np.inf
        for label in unique_labels:
            if label == own_label:
                continue
            dist = np.sqrt(np.sum((X[i] - centroids[label]) ** 2))
            if dist < min_other_dist:
                min_other_dist = dist

        if min_other_dist > 0:
            confidence[i] = own_dist / min_other_dist
        else:
            confidence[i] = 0.0

    adata.obs[key_added] = confidence

    return adata if copy else None
