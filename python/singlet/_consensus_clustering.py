"""Consensus clustering via multi-resolution Leiden.

Provides singlet.consensus_clustering() — run Leiden at multiple resolutions
and random seeds, build a consensus co-clustering matrix, and derive final
clusters via hierarchical clustering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def consensus_clustering(
    adata: "AnnData",
    *,
    n_runs: int = 20,
    resolution_range: tuple[float, float] = (0.5, 2.0),
    n_resolutions: int = 5,
    random_state: int = 0,
    method: str = "leiden",
) -> "AnnData":
    """Consensus clustering via multi-resolution repeated clustering.

    Runs Leiden (or Louvain) at multiple resolutions and random seeds,
    computes a consensus co-clustering matrix (fraction of runs in which
    each pair of cells is co-clustered), and derives final clusters via
    hierarchical clustering of the consensus matrix.

    Parameters
    ----------
    adata
        Annotated data matrix. Must have a precomputed neighbor graph
        (run singlet.neighbors() first).
    n_runs
        Number of clustering runs per resolution. Total runs = n_runs * n_resolutions.
    resolution_range
        Tuple of (min_resolution, max_resolution) to sample from.
    n_resolutions
        Number of evenly-spaced resolutions to use within the range.
    random_state
        Base random seed. Each run uses random_state + run_index.
    method
        Clustering method: 'leiden' or 'louvain'.

    Returns
    -------
    AnnData
        The input adata with:
        - adata.obs['consensus_clusters']: Final consensus cluster labels.
        - adata.uns['consensus_matrix']: Co-clustering frequency matrix (n_cells × n_cells).

    Raises
    ------
    ValueError
        If method is not 'leiden' or 'louvain'.
        If resolution_range is invalid.
        If neighbors have not been computed.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.consensus_clustering(adata, n_runs=10, n_resolutions=3)
    >>> adata.obs['consensus_clusters']
    """
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if method not in ("leiden", "louvain"):
        msg = f"method must be 'leiden' or 'louvain', got '{method}'"
        raise ValueError(msg)

    if resolution_range[0] >= resolution_range[1]:
        msg = f"resolution_range[0] must be < resolution_range[1], got {resolution_range}"
        raise ValueError(msg)

    if "connectivities" not in adata.obsp:
        msg = (
            "Neighbor graph not found. Run singlet.neighbors() first "
            "(adata.obsp['connectivities'] missing)."
        )
        raise ValueError(msg)

    n_cells = adata.n_obs
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    total_runs = n_runs * n_resolutions

    # Co-clustering matrix: counts how many times each pair is in the same cluster
    cocluster = np.zeros((n_cells, n_cells), dtype=np.float32)

    # Get adjacency for clustering
    adjacency = adata.obsp["connectivities"]

    run_idx = 0
    for res in resolutions:
        for run in range(n_runs):
            seed = random_state + run_idx
            labels = _run_clustering(adjacency, method, res, seed, n_cells)
            # Update co-clustering matrix
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                indices = np.where(mask)[0]
                # All pairs in the same cluster get +1
                idx_grid = np.ix_(indices, indices)
                cocluster[idx_grid] += 1.0
            run_idx += 1

    # Normalize to get frequency (0-1)
    consensus_matrix = cocluster / total_runs

    # Hierarchical clustering on the consensus matrix
    # Convert similarity to distance
    distance_matrix = 1.0 - consensus_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    # Ensure symmetry and non-negative
    distance_matrix = np.maximum(distance_matrix, 0.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    # Use condensed distance matrix for linkage
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    # Determine optimal number of clusters using the largest gap in merge distances
    n_clusters = _determine_n_clusters(linkage_matrix, max_k=min(30, n_cells // 2))
    final_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    # Convert to 0-indexed string labels
    final_labels = (final_labels - 1).astype(str)

    adata.obs["consensus_clusters"] = pd.Categorical(final_labels)
    adata.uns["consensus_matrix"] = consensus_matrix

    return adata


def _run_clustering(adjacency, method: str, resolution: float, seed: int, n_cells: int):
    """Run a single clustering iteration."""
    import numpy as np

    try:
        if method == "leiden":
            import igraph as ig
            import leidenalg

            sources, targets = adjacency.nonzero()
            weights = np.asarray(adjacency[sources, targets]).ravel()
            graph = ig.Graph(n=n_cells, edges=list(zip(sources, targets)), directed=False)
            graph.es["weight"] = weights

            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=seed,
            )
            return np.array(partition.membership)
        else:
            # Louvain fallback
            import community as community_louvain
            import networkx as nx

            graph = nx.from_scipy_sparse_array(adjacency)
            partition = community_louvain.best_partition(
                graph, resolution=resolution, random_state=seed
            )
            return np.array([partition[i] for i in range(n_cells)])
    except ImportError:
        # Spectral clustering fallback
        from sklearn.cluster import SpectralClustering

        n_clusters_est = max(2, int(resolution * 5))
        sc = SpectralClustering(
            n_clusters=n_clusters_est,
            affinity="precomputed",
            random_state=seed,
        )
        dense = adjacency.toarray() if hasattr(adjacency, "toarray") else adjacency
        labels = sc.fit_predict(np.abs(dense))
        return labels


def _determine_n_clusters(linkage_matrix, max_k: int = 30) -> int:
    """Determine optimal cluster count from linkage using the gap heuristic."""
    import numpy as np

    # Use merge distances to find natural gaps
    merge_distances = linkage_matrix[:, 2]
    n_merges = len(merge_distances)

    if n_merges < 2:
        return 2

    # Look at the last max_k merges (corresponding to k=2..max_k+1 clusters)
    check_range = min(max_k, n_merges - 1)
    # Gaps between successive merge distances (from end)
    gaps = np.diff(merge_distances[-check_range:])

    if len(gaps) == 0:
        return 2

    # The largest gap corresponds to the best cut
    best_gap_idx = np.argmax(gaps)
    # Number of clusters = total_merges - position_of_gap
    n_clusters = check_range - best_gap_idx

    return max(2, min(n_clusters, max_k))
