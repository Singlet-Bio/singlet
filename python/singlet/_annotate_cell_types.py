"""Automated cell type annotation from marker gene sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def annotate_cell_types(
    adata: AnnData,
    marker_dict: dict[str, list[str]],
    *,
    groupby: str = "leiden",
    min_score: float = 0.1,
) -> AnnData:
    """Annotate cell types based on marker gene overlap per cluster.

    For each cluster (defined by ``groupby``), scores overlap with known
    marker gene sets using a combination of:
    - Fraction of markers expressed in the cluster
    - Fold-enrichment of marker expression vs. background

    The cell type with the highest combined score is assigned to each
    cluster.

    Parameters
    ----------
    adata
        Annotated data matrix.
    marker_dict
        Dictionary mapping cell type names to lists of marker genes.
        Example: ``{"T cell": ["CD3D", "CD3E"], "B cell": ["CD19", "MS4A1"]}``
    groupby
        Key in ``.obs`` for cluster assignments. Default ``'leiden'``.
    min_score
        Minimum score threshold. Clusters scoring below this for all
        cell types are labeled ``'Unknown'``.

    Returns
    -------
    AnnData with ``adata.obs['cell_type']`` and
    ``adata.obs['cell_type_score']`` added.
    """
    from scipy.sparse import issparse

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")

    if not isinstance(marker_dict, dict) or len(marker_dict) == 0:
        raise ValueError("marker_dict must be a non-empty dictionary.")

    gene_names = list(adata.var_names)
    gene_set = set(gene_names)

    clusters = adata.obs[groupby].values
    unique_clusters = sorted(set(clusters), key=str)

    # Get expression matrix
    X = adata.X

    # Compute global mean expression per gene (background)
    if issparse(X):
        global_mean = np.asarray(X.mean(axis=0)).ravel()
    else:
        global_mean = np.mean(X, axis=0).ravel()

    # Avoid division by zero
    global_mean_safe = global_mean.copy()
    global_mean_safe[global_mean_safe == 0] = 1e-10

    # Score each cluster against each cell type
    cluster_annotations = {}
    cluster_scores = {}

    for cluster in unique_clusters:
        mask = clusters == cluster
        n_cells_cluster = mask.sum()

        if n_cells_cluster == 0:
            cluster_annotations[cluster] = "Unknown"
            cluster_scores[cluster] = 0.0
            continue

        # Compute cluster mean expression
        if issparse(X):
            cluster_mean = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            cluster_mean = np.mean(X[mask], axis=0).ravel()

        # Compute fraction of cells expressing each gene (> 0)
        if issparse(X):
            cluster_expressed = np.asarray((X[mask] > 0).mean(axis=0)).ravel()
        else:
            cluster_expressed = np.mean(np.asarray(X[mask]) > 0, axis=0).ravel()

        best_type = "Unknown"
        best_score = 0.0

        for cell_type, markers in marker_dict.items():
            # Filter to markers present in the dataset
            valid_markers = [g for g in markers if g in gene_set]
            if len(valid_markers) == 0:
                continue

            marker_indices = [gene_names.index(g) for g in valid_markers]

            # Score component 1: fraction of markers expressed in cluster
            frac_expressed = np.mean(cluster_expressed[marker_indices])

            # Score component 2: fold-enrichment of marker expression
            marker_cluster_mean = cluster_mean[marker_indices]
            marker_global_mean = global_mean_safe[marker_indices]
            fold_enrichment = np.mean(marker_cluster_mean / marker_global_mean)

            # Normalize fold enrichment with log transform for stability
            log_fold = np.log1p(fold_enrichment)

            # Combined score: geometric-like combination
            score = frac_expressed * log_fold

            if score > best_score:
                best_score = score
                best_type = cell_type

        if best_score < min_score:
            best_type = "Unknown"
            best_score = 0.0

        cluster_annotations[cluster] = best_type
        cluster_scores[cluster] = float(best_score)

    # Map cluster annotations to cells
    adata.obs["cell_type"] = [cluster_annotations[c] for c in clusters]
    adata.obs["cell_type_score"] = [cluster_scores[c] for c in clusters]

    return adata
