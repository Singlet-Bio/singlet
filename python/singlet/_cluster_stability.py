# SPDX-License-Identifier: MIT
"""Cluster stability analysis via bootstrap subsampling.

Provides singlet.cluster_stability() — assess clustering robustness by
repeatedly subsampling, reclustering, and comparing to the full clustering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def cluster_stability(
    adata: "AnnData",
    *,
    n_bootstraps: int = 20,
    resolution: float = 1.0,
    subsample_frac: float = 0.8,
    random_state: int = 0,
    method: str = "leiden",
    neighbors_key: str | None = None,
) -> dict:
    """Assess cluster stability via bootstrap subsampling.

    Repeatedly subsamples cells, reclusters, and computes agreement metrics
    (ARI and NMI) with the full-data clustering. Also computes per-cluster
    stability as the fraction of times cells in each cluster are assigned
    to the same cluster across bootstrap iterations.

    Parameters
    ----------
    adata
        Annotated data matrix. Must have a precomputed neighbor graph
        (run singlet.neighbors() first) and PCA in .obsm['X_pca'].
    n_bootstraps
        Number of bootstrap iterations.
    resolution
        Resolution parameter for Leiden/Louvain clustering.
    subsample_frac
        Fraction of cells to subsample in each iteration (0 < frac < 1).
    random_state
        Random seed for reproducibility.
    method
        Clustering method: 'leiden' or 'louvain'.
    neighbors_key
        Key for neighbors in .uns. If None, uses default 'neighbors'.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean_ari': Mean Adjusted Rand Index across bootstraps.
        - 'std_ari': Std of ARI.
        - 'mean_nmi': Mean Normalized Mutual Information.
        - 'std_nmi': Std of NMI.
        - 'per_cluster_stability': dict mapping cluster label → stability
          score (0-1).

    Also stores results in adata.uns['cluster_stability'].

    Raises
    ------
    ValueError
        If method is not 'leiden' or 'louvain'.
        If subsample_frac is not between 0 and 1 (exclusive).
        If PCA has not been computed.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> stability = singlet.cluster_stability(adata, n_bootstraps=10)
    >>> print(f"Mean ARI: {stability['mean_ari']:.3f}")
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    valid_methods = ("leiden", "louvain")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    if not (0 < subsample_frac < 1):
        msg = f"subsample_frac must be between 0 and 1 (exclusive), got {subsample_frac}"
        raise ValueError(msg)

    if "X_pca" not in adata.obsm:
        msg = "PCA not found in adata.obsm['X_pca']. Run singlet.pca() first."
        raise ValueError(msg)

    # Perform full clustering if not already done
    cluster_key = method
    if cluster_key not in adata.obs.columns:
        if method == "leiden":
            sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
        else:
            sc.tl.louvain(adata, resolution=resolution, key_added=cluster_key)

    full_labels = adata.obs[cluster_key].values.astype(str)
    n_cells = adata.n_obs
    n_subsample = int(n_cells * subsample_frac)

    rng = np.random.default_rng(random_state)

    ari_scores = []
    nmi_scores = []

    # Track per-cell cluster agreement for per-cluster stability
    cluster_agreement = np.zeros(n_cells, dtype=np.float64)
    cluster_counts = np.zeros(n_cells, dtype=np.float64)

    for bootstrap_idx in range(n_bootstraps):
        # Subsample cells
        indices = rng.choice(n_cells, size=n_subsample, replace=False)
        indices_sorted = np.sort(indices)

        # Create sub-AnnData with PCA
        sub_adata = sc.AnnData(
            X=adata.X[indices_sorted],
            obsm={"X_pca": adata.obsm["X_pca"][indices_sorted]},
        )
        sub_adata.obs_names = pd.Index([f"cell_{j}" for j in range(n_subsample)])

        # Compute neighbors on subsample
        sc.pp.neighbors(sub_adata, use_rep="X_pca")

        # Cluster subsample
        if method == "leiden":
            sc.tl.leiden(sub_adata, resolution=resolution, key_added="sub_cluster")
        else:
            sc.tl.louvain(sub_adata, resolution=resolution, key_added="sub_cluster")

        sub_labels = sub_adata.obs["sub_cluster"].values.astype(str)
        ref_labels = full_labels[indices_sorted]

        # Compute metrics
        ari = adjusted_rand_score(ref_labels, sub_labels)
        nmi = normalized_mutual_info_score(ref_labels, sub_labels)
        ari_scores.append(ari)
        nmi_scores.append(nmi)

        # Per-cell agreement: 1 if assigned to same majority cluster
        # Map sub_labels to ref_labels via contingency
        from collections import Counter

        # For each sub-cluster, find the most common reference cluster
        sub_to_ref = {}
        sub_unique = np.unique(sub_labels)
        for sc_label in sub_unique:
            mask = sub_labels == sc_label
            ref_in_sub = ref_labels[mask]
            most_common = Counter(ref_in_sub).most_common(1)[0][0]
            sub_to_ref[sc_label] = most_common

        # Check agreement
        for cell_pos, global_idx in enumerate(indices_sorted):
            predicted_ref = sub_to_ref[sub_labels[cell_pos]]
            if predicted_ref == full_labels[global_idx]:
                cluster_agreement[global_idx] += 1.0
            cluster_counts[global_idx] += 1.0

    # Per-cluster stability
    # For each cluster, average the per-cell stability of its members
    unique_clusters = sorted(set(full_labels))
    per_cluster_stability = {}
    for clust in unique_clusters:
        mask = full_labels == clust
        cells_in_cluster = cluster_counts[mask]
        agreement_in_cluster = cluster_agreement[mask]
        # Only count cells that were actually sampled
        sampled_mask = cells_in_cluster > 0
        if sampled_mask.sum() > 0:
            stability = (agreement_in_cluster[sampled_mask] / cells_in_cluster[sampled_mask]).mean()
        else:
            stability = 0.0
        per_cluster_stability[clust] = float(stability)

    result = {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "mean_nmi": float(np.mean(nmi_scores)),
        "std_nmi": float(np.std(nmi_scores)),
        "per_cluster_stability": per_cluster_stability,
    }

    adata.uns["cluster_stability"] = result
    return result
