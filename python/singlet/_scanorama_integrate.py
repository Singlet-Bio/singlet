# SPDX-License-Identifier: MIT
"""Scanorama-style batch integration via mutual nearest neighbors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def scanorama_integrate(
    adata_list: AnnData | list[AnnData],
    *,
    batch_key: str | None = None,
    n_neighbors: int = 20,
    use_rep: str = "X_pca",
    n_components: int = 50,
    sigma: float = 15.0,
) -> AnnData:
    """Scanorama-style batch integration using mutual nearest neighbors.

    Finds mutual nearest neighbors across batches in PCA space, computes
    correction vectors from MNN pairs, and applies them to embed all
    batches in a shared space.

    Parameters
    ----------
    adata_list
        Either a list of AnnData objects (one per batch) or a single
        AnnData with ``batch_key`` specified.
    batch_key
        Key in ``.obs`` identifying batches when a single AnnData is
        provided. Ignored when a list is passed.
    n_neighbors
        Number of nearest neighbors for MNN detection.
    use_rep
        Key in ``.obsm`` for the embedding to integrate. Default is
        ``'X_pca'``.
    n_components
        Number of embedding dimensions to use.
    sigma
        Bandwidth for Gaussian kernel smoothing of correction vectors.

    Returns
    -------
    AnnData with corrected embedding stored in ``.obsm['X_scanorama']``.
    If a list was provided, returns the concatenated (merged) AnnData.
    If a single AnnData was provided, returns the same object (modified
    in-place).
    """
    from anndata import AnnData as _AnnData

    single_mode = isinstance(adata_list, _AnnData)

    if single_mode:
        adata = adata_list
        if batch_key is None:
            raise ValueError("When a single AnnData is provided, 'batch_key' must be specified.")
        if batch_key not in adata.obs.columns:
            raise KeyError(f"'{batch_key}' not found in .obs.")
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in .obsm. Run singlet.pca() first.")

        embedding = adata.obsm[use_rep][:, :n_components].copy().astype(np.float64)
        batch_labels = adata.obs[batch_key].values
        batches = list(dict.fromkeys(batch_labels))  # preserve order, deduplicate

        if len(batches) < 2:
            adata.obsm["X_scanorama"] = embedding.astype(np.float32)
            return adata

        corrected = _integrate_embeddings(embedding, batch_labels, batches, n_neighbors, sigma)
        adata.obsm["X_scanorama"] = corrected.astype(np.float32)
        return adata
    else:
        # List mode: validate and concatenate
        if not isinstance(adata_list, list) or len(adata_list) == 0:
            raise ValueError("adata_list must be a non-empty list of AnnData objects.")

        if len(adata_list) == 1:
            ad = adata_list[0]
            if use_rep not in ad.obsm:
                raise KeyError(f"'{use_rep}' not found in .obsm.")
            ad.obsm["X_scanorama"] = ad.obsm[use_rep][:, :n_components].copy().astype(np.float32)
            return ad

        # Validate all have the representation
        for idx, ad in enumerate(adata_list):
            if use_rep not in ad.obsm:
                raise KeyError(f"'{use_rep}' not found in .obsm of adata_list[{idx}].")

        # Build combined embedding and batch labels
        embeddings = []
        batch_labels_list = []
        batch_names = []

        for idx, ad in enumerate(adata_list):
            emb = ad.obsm[use_rep][:, :n_components].copy().astype(np.float64)
            embeddings.append(emb)
            batch_name = f"batch_{idx}"
            batch_names.append(batch_name)
            batch_labels_list.extend([batch_name] * ad.n_obs)

        combined_embedding = np.vstack(embeddings)
        batch_labels_arr = np.array(batch_labels_list)

        corrected = _integrate_embeddings(
            combined_embedding, batch_labels_arr, batch_names, n_neighbors, sigma
        )

        # Concatenate anndata objects
        merged = _concat_adatas(adata_list)
        merged.obsm["X_scanorama"] = corrected.astype(np.float32)
        return merged


def _integrate_embeddings(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    batch_order: list,
    n_neighbors: int,
    sigma: float,
) -> np.ndarray:
    """Apply sequential MNN-based correction across batches."""
    corrected = embedding.copy()

    # Use first batch as reference, sequentially correct others
    ref_batch = batch_order[0]
    integrated_mask = batch_labels == ref_batch

    for batch_idx in range(1, len(batch_order)):
        query_batch = batch_order[batch_idx]
        query_mask = batch_labels == query_batch

        ref_indices = np.where(integrated_mask)[0]
        query_indices = np.where(query_mask)[0]

        ref_data = corrected[ref_indices]
        query_data = corrected[query_indices]

        # Find mutual nearest neighbors
        mnn_pairs = _find_mnn_pairs(ref_data, query_data, n_neighbors)

        if len(mnn_pairs) == 0:
            # No MNN pairs — skip correction for this batch
            integrated_mask = integrated_mask | query_mask
            continue

        # Compute correction vectors at MNN query points
        mnn_diffs = np.array([ref_data[ri] - query_data[qi] for ri, qi in mnn_pairs])
        mnn_positions = np.array([query_data[qi] for _, qi in mnn_pairs])

        # Apply Gaussian-smoothed correction to all query cells
        for local_idx, global_idx in enumerate(query_indices):
            cell_pos = corrected[global_idx]
            dists_sq = np.sum((mnn_positions - cell_pos[None, :]) ** 2, axis=1)
            weights = np.exp(-dists_sq / (2 * sigma**2))
            weight_sum = weights.sum()

            if weight_sum > 0:
                correction = (weights[:, None] * mnn_diffs).sum(axis=0) / weight_sum
                corrected[global_idx] += correction

        integrated_mask = integrated_mask | query_mask

    return corrected


def _find_mnn_pairs(data1: np.ndarray, data2: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Find mutual nearest neighbor pairs between two datasets."""
    from sklearn.neighbors import NearestNeighbors

    k1 = min(k, data1.shape[0])
    k2 = min(k, data2.shape[0])

    if k1 == 0 or k2 == 0:
        return []

    # data1 cells' nearest neighbors in data2
    nn2 = NearestNeighbors(n_neighbors=k2, algorithm="auto")
    nn2.fit(data2)
    _, nn_1to2 = nn2.kneighbors(data1)

    # data2 cells' nearest neighbors in data1
    nn1 = NearestNeighbors(n_neighbors=k1, algorithm="auto")
    nn1.fit(data1)
    _, nn_2to1 = nn1.kneighbors(data2)

    # Find mutual pairs
    mnn_pairs = []
    for idx_in_1 in range(data1.shape[0]):
        for idx_in_2 in nn_1to2[idx_in_1]:
            if idx_in_1 in nn_2to1[idx_in_2]:
                mnn_pairs.append((idx_in_1, idx_in_2))

    return mnn_pairs


def _concat_adatas(adata_list: list[AnnData]) -> AnnData:
    """Concatenate a list of AnnData objects, preserving shared genes."""
    import pandas as pd
    from anndata import AnnData as _AnnData
    from scipy.sparse import issparse
    from scipy.sparse import vstack as sp_vstack

    # Find shared genes
    shared_vars = set(adata_list[0].var_names)
    for ad in adata_list[1:]:
        shared_vars &= set(ad.var_names)
    shared_vars = sorted(shared_vars)

    if len(shared_vars) == 0:
        raise ValueError("No shared genes found across batches.")

    # Subset each to shared genes and concatenate
    xs = []
    obs_frames = []
    for idx, ad in enumerate(adata_list):
        sub = ad[:, shared_vars]
        if issparse(sub.X):
            xs.append(sub.X)
        else:
            from scipy.sparse import csr_matrix

            xs.append(csr_matrix(sub.X))

        obs_df = sub.obs.copy()
        obs_df["_scanorama_batch"] = f"batch_{idx}"
        obs_frames.append(obs_df)

    merged_X = sp_vstack(xs)
    merged_obs = pd.concat(obs_frames, axis=0, ignore_index=True)
    merged_var = adata_list[0][:, shared_vars].var.copy()

    result = _AnnData(X=merged_X, obs=merged_obs, var=merged_var)
    result.obs_names = pd.Index([f"cell_{i}" for i in range(result.n_obs)])
    return result
