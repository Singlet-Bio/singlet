"""Gene trend clustering along pseudotime.

Provides singlet.gene_trend_clustering() — bin cells by pseudotime, smooth
gene expression across bins, and cluster genes by similarity of their
temporal expression patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def gene_trend_clustering(
    adata: "AnnData",
    *,
    pseudotime_key: str = "dpt_pseudotime",
    n_clusters: int = 5,
    n_bins: int = 50,
    method: str = "kmeans",
    n_top_genes: int = 500,
) -> "pd.DataFrame":
    """Cluster genes by their expression trend along pseudotime.

    Bins cells by pseudotime, computes smoothed mean expression per bin for
    highly variable genes, then clusters genes by similarity of their
    temporal profiles.

    Parameters
    ----------
    adata
        Annotated data matrix. Must have pseudotime values in
        adata.obs[pseudotime_key].
    pseudotime_key
        Key in adata.obs containing pseudotime values.
    n_clusters
        Number of gene trend clusters.
    n_bins
        Number of pseudotime bins for smoothing.
    method
        Clustering method for gene trends: 'kmeans' or 'hierarchical'.
    n_top_genes
        Number of top variable genes to use. If adata has fewer genes,
        all genes are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'gene', 'trend_cluster', and one column per
        pseudotime bin ('bin_0', 'bin_1', ...) with smoothed expression values.

    Raises
    ------
    ValueError
        If pseudotime_key not found in adata.obs.
        If method is not 'kmeans' or 'hierarchical'.
        If n_bins < 3.
        If n_clusters < 2.

    Notes
    -----
    Results are also stored in:
    - adata.var['trend_cluster']: cluster assignment for each gene
    - adata.uns['gene_trends']: dict with 'trends' (genes × bins matrix),
      'bin_centers' (pseudotime bin centers), 'genes' (gene names used)
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.ndimage import uniform_filter1d
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.preprocessing import StandardScaler

    # --- Validate inputs ---
    if pseudotime_key not in adata.obs.columns:
        msg = (
            f"Pseudotime key '{pseudotime_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
        raise ValueError(msg)

    if method not in ("kmeans", "hierarchical"):
        msg = f"method must be 'kmeans' or 'hierarchical', got '{method}'"
        raise ValueError(msg)

    if n_bins < 3:
        msg = f"n_bins must be >= 3, got {n_bins}"
        raise ValueError(msg)

    if n_clusters < 2:
        msg = f"n_clusters must be >= 2, got {n_clusters}"
        raise ValueError(msg)

    # --- Select genes ---
    n_genes_use = min(n_top_genes, adata.n_vars)
    if "highly_variable" in adata.var.columns:
        hv_mask = adata.var["highly_variable"].values
        if hv_mask.sum() > 0:
            hv_genes = adata.var_names[hv_mask][:n_genes_use]
        else:
            # Fall back to variance-based selection
            hv_genes = _select_by_variance(adata, n_genes_use)
    else:
        hv_genes = _select_by_variance(adata, n_genes_use)

    gene_indices = [adata.var_names.get_loc(g) for g in hv_genes]

    # --- Get pseudotime and bin cells ---
    pseudotime = adata.obs[pseudotime_key].values.astype(np.float64)
    valid_mask = ~np.isnan(pseudotime)
    pseudotime_valid = pseudotime[valid_mask]

    pt_min, pt_max = pseudotime_valid.min(), pseudotime_valid.max()
    bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_assignments = np.digitize(pseudotime_valid, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    # --- Compute mean expression per bin ---
    X = adata.X[valid_mask]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    trends = np.zeros((len(gene_indices), n_bins), dtype=np.float64)
    for bin_idx in range(n_bins):
        mask = bin_assignments == bin_idx
        if mask.sum() > 0:
            trends[:, bin_idx] = X[mask][:, gene_indices].mean(axis=0)

    # --- Smooth trends ---
    kernel_size = max(3, n_bins // 10)
    for idx in range(len(gene_indices)):
        trends[idx] = uniform_filter1d(trends[idx], size=kernel_size)

    # --- Normalize and cluster genes ---
    scaler = StandardScaler()
    trends_scaled = scaler.fit_transform(trends)

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        gene_labels = clusterer.fit_predict(trends_scaled)
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        gene_labels = clusterer.fit_predict(trends_scaled)

    # --- Store results ---
    gene_names = list(hv_genes)
    adata.var["trend_cluster"] = pd.Series(dtype="object", index=adata.var_names)
    for idx, gene in enumerate(gene_names):
        adata.var.loc[gene, "trend_cluster"] = str(gene_labels[idx])

    adata.uns["gene_trends"] = {
        "trends": trends,
        "bin_centers": bin_centers,
        "genes": gene_names,
    }

    # --- Build result DataFrame ---
    bin_cols = {f"bin_{i}": trends[:, i] for i in range(n_bins)}
    result_df = pd.DataFrame(
        {
            "gene": gene_names,
            "trend_cluster": [str(x) for x in gene_labels],
            **bin_cols,
        }
    )

    return result_df


def _select_by_variance(adata: "AnnData", n_genes: int) -> list:
    """Select top genes by variance."""
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        variances = (
            np.asarray(X.power(2).mean(axis=0)).ravel() - np.asarray(X.mean(axis=0)).ravel() ** 2
        )
    else:
        variances = np.var(X, axis=0)

    top_idx = np.argsort(variances)[::-1][:n_genes]
    return list(adata.var_names[top_idx])
