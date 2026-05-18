# SPDX-License-Identifier: MIT
"""Mutual nearest neighbors batch correction."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def mnn_correct(
    adata: AnnData,
    *,
    batch_key: str = "batch",
    n_neighbors: int = 20,
    n_pcs: int | None = None,
    sigma: float = 1.0,
    cos_norm: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Correct batch effects using mutual nearest neighbors.

    Identifies MNN pairs between batches and computes correction vectors
    by Gaussian-weighted averaging of MNN differences.

    Parameters
    ----------
    adata
        Annotated data matrix with PCA computed.
    batch_key
        Key in .obs containing batch labels.
    n_neighbors
        Number of nearest neighbors for MNN detection.
    n_pcs
        Number of PCs to use. None uses all.
    sigma
        Bandwidth for Gaussian kernel smoothing of correction vectors.
    cos_norm
        Whether to use cosine normalization before finding MNNs.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores corrected embedding in
    `.obsm['X_mnn']`.
    """

    adata = adata.copy() if copy else adata

    if "X_pca" not in adata.obsm:
        raise KeyError("'X_pca' not found in .obsm. Run singlet.pca() first.")

    if batch_key not in adata.obs.columns:
        raise KeyError(f"'{batch_key}' not found in .obs.")

    X_pca = adata.obsm["X_pca"]
    if n_pcs is not None:
        X_pca = X_pca[:, :n_pcs]

    batches = adata.obs[batch_key].unique()
    if len(batches) < 2:
        # Nothing to correct
        adata.obsm["X_mnn"] = X_pca.copy()
        return adata if copy else None

    # Cosine normalize if requested
    if cos_norm:
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_pca / norms
    else:
        X_norm = X_pca

    # Order batches (use first as reference, correct others sequentially)
    batch_labels = adata.obs[batch_key].values
    batch_order = list(batches)

    corrected = X_pca.copy().astype(np.float64)

    ref_batch = batch_order[0]
    ref_mask = batch_labels == ref_batch

    for i in range(1, len(batch_order)):
        query_batch = batch_order[i]
        query_mask = batch_labels == query_batch

        ref_data = X_norm[ref_mask] if cos_norm else corrected[ref_mask]
        query_data = X_norm[query_mask] if cos_norm else corrected[query_mask]

        # Find mutual nearest neighbors
        mnn_pairs = _find_mnn(ref_data, query_data, n_neighbors)

        if len(mnn_pairs) == 0:
            # No MNN pairs found, skip correction
            continue

        # Compute correction vectors at MNN points
        ref_indices = np.where(ref_mask)[0]
        query_indices = np.where(query_mask)[0]

        mnn_diffs = []
        mnn_positions = []

        for ref_idx, query_idx in mnn_pairs:
            diff = corrected[ref_indices[ref_idx]] - corrected[query_indices[query_idx]]
            mnn_diffs.append(diff)
            mnn_positions.append(corrected[query_indices[query_idx]])

        mnn_diffs = np.array(mnn_diffs)
        mnn_positions = np.array(mnn_positions)

        # Smooth correction vectors using Gaussian kernel
        for j, qi in enumerate(np.where(query_mask)[0]):
            cell_pos = corrected[qi]

            # Compute distances to MNN query positions
            dists = np.sum((mnn_positions - cell_pos[None, :]) ** 2, axis=1)

            # Gaussian weights
            weights = np.exp(-dists / (2 * sigma**2))
            weight_sum = weights.sum()

            if weight_sum > 0:
                correction = (weights[:, None] * mnn_diffs).sum(axis=0) / weight_sum
                corrected[qi] += correction

        # Update reference to include corrected query cells
        ref_mask = ref_mask | query_mask

    adata.obsm["X_mnn"] = corrected.astype(np.float32)

    return adata if copy else None


def _find_mnn(data1: np.ndarray, data2: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Find mutual nearest neighbor pairs between two datasets."""
    from sklearn.neighbors import NearestNeighbors

    k1 = min(k, data1.shape[0])
    k2 = min(k, data2.shape[0])

    # data1 -> data2 nearest neighbors
    nn2 = NearestNeighbors(n_neighbors=k2, metric="euclidean")
    nn2.fit(data2)
    _, nn_1to2 = nn2.kneighbors(data1)

    # data2 -> data1 nearest neighbors
    nn1 = NearestNeighbors(n_neighbors=k1, metric="euclidean")
    nn1.fit(data1)
    _, nn_2to1 = nn1.kneighbors(data2)

    # Find mutual pairs
    mnn_pairs = []
    for i in range(data1.shape[0]):
        for j in nn_1to2[i]:
            if i in nn_2to1[j]:
                mnn_pairs.append((i, j))

    return mnn_pairs
