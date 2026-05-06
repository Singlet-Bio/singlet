"""Cellular reprogramming potential score.

Estimates how close each cell is to transitioning from a source cell type
to a target cell type, based on embedding distances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def reprogramming_score(
    adata: "AnnData",
    source_type: str,
    target_type: str,
    *,
    type_key: str = "cell_type",
    use_rep: str = "X_pca",
) -> "AnnData":
    """Estimate cellular reprogramming potential from source to target type.

    Computes a score for each cell indicating how close it is to transitioning
    from a source cell type to a target cell type, based on embedding distances.
    A score near 1.0 means the cell is close to the target; near 0.0 means it
    is close to the source.

    Parameters
    ----------
    adata
        Annotated data matrix with embedding in ``adata.obsm[use_rep]``.
    source_type
        Name of the source cell type in ``adata.obs[type_key]``.
    target_type
        Name of the target cell type in ``adata.obs[type_key]``.
    type_key
        Column in ``adata.obs`` containing cell type annotations.
    use_rep
        Key in ``adata.obsm`` for the embedding to use for distance
        computation.

    Returns
    -------
    AnnData
        The input ``adata`` with ``adata.obs['reprogramming_score']`` added.
        Score = 1 - (dist_to_target / (dist_to_target + dist_to_source)).

    Raises
    ------
    KeyError
        If ``type_key`` is not in ``adata.obs`` or ``use_rep`` not in
        ``adata.obsm``.
    ValueError
        If ``source_type`` or ``target_type`` is not found in the type column.
    """
    if type_key not in adata.obs.columns:
        msg = f"Column '{type_key}' not found in adata.obs"
        raise KeyError(msg)
    if use_rep not in adata.obsm:
        msg = f"Representation '{use_rep}' not found in adata.obsm"
        raise KeyError(msg)

    labels = adata.obs[type_key]
    unique_types = set(labels.unique())

    if source_type not in unique_types:
        msg = f"Source type '{source_type}' not found in adata.obs['{type_key}']"
        raise ValueError(msg)
    if target_type not in unique_types:
        msg = f"Target type '{target_type}' not found in adata.obs['{type_key}']"
        raise ValueError(msg)

    embedding = adata.obsm[use_rep]

    # Compute centroids for source and target
    source_mask = labels.values == source_type
    target_mask = labels.values == target_type

    source_centroid = np.mean(embedding[source_mask], axis=0)
    target_centroid = np.mean(embedding[target_mask], axis=0)

    # Compute distances from each cell to source and target centroids
    dist_to_source = np.linalg.norm(embedding - source_centroid, axis=1)
    dist_to_target = np.linalg.norm(embedding - target_centroid, axis=1)

    # Score: 1 means close to target, 0 means close to source
    total_dist = dist_to_target + dist_to_source
    # Avoid division by zero for cells exactly at both centroids
    score = np.where(total_dist > 0, 1.0 - (dist_to_target / total_dist), 0.5)

    adata.obs["reprogramming_score"] = score.astype(np.float64)

    return adata
