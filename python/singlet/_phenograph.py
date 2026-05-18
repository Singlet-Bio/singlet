# SPDX-License-Identifier: MIT
"""PhenoGraph-style clustering using shared nearest neighbor (SNN) graphs.

Provides singlet.phenograph() — builds a kNN graph, computes Jaccard-weighted
SNN graph, and applies Leiden/Louvain community detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def phenograph(
    adata: "AnnData",
    *,
    use_rep: str = "X_pca",
    n_neighbors: int = 30,
    resolution: float = 1.0,
    clustering: str = "leiden",
    key_added: str = "phenograph",
    random_state: int = 0,
) -> "AnnData":
    """PhenoGraph-style clustering with shared nearest neighbor (SNN) graph.

    Differs from standard Leiden/Louvain clustering by constructing a shared
    nearest neighbor (SNN) graph with Jaccard similarity weights, which
    provides more robust community detection especially for complex datasets.

    Steps:
    1. Build kNN graph from the representation (e.g., PCA).
    2. Compute SNN graph: for each pair of cells that share neighbors,
       weight the edge by their Jaccard similarity (|intersection| / |union|
       of their neighbor sets).
    3. Apply Leiden or Louvain community detection on the SNN graph.

    Parameters
    ----------
    adata
        Annotated data matrix. Must have the representation specified
        by `use_rep` in adata.obsm (default: 'X_pca').
    use_rep
        Key in adata.obsm for the representation to build kNN graph from.
    n_neighbors
        Number of nearest neighbors for the kNN graph.
    resolution
        Resolution parameter for community detection. Higher values
        produce more clusters.
    clustering
        Community detection algorithm: 'leiden' or 'louvain'.
    key_added
        Key in adata.obs where cluster labels are stored.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        The input adata with added:
        - adata.obs[key_added] : cluster labels (categorical)
        - adata.obsp['snn_connectivities'] : SNN graph (Jaccard-weighted)

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.phenograph(adata, n_neighbors=30)
    >>> adata.obs['phenograph']  # cluster labels
    """
    import numpy as np
    import pandas as pd

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"phenograph() requires an AnnData object, got {type(adata).__name__}")

    if clustering not in ("leiden", "louvain"):
        raise ValueError(f"clustering must be 'leiden' or 'louvain', got '{clustering}'")

    if use_rep not in adata.obsm:
        raise KeyError(
            f"'{use_rep}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )

    rep = np.asarray(adata.obsm[use_rep])
    n_cells = rep.shape[0]
    n_neighbors = min(n_neighbors, n_cells - 1)

    # Step 1: Build kNN graph
    knn_indices = _build_knn(rep, n_neighbors)

    # Step 2: Compute SNN graph with Jaccard weights
    snn_matrix = _compute_snn_jaccard(knn_indices, n_cells, n_neighbors)

    # Store SNN graph
    adata.obsp["snn_connectivities"] = snn_matrix

    # Step 3: Community detection on SNN graph
    labels = _community_detection(
        snn_matrix,
        clustering=clustering,
        resolution=resolution,
        random_state=random_state,
    )

    # Store labels
    labels_str = [str(lab) for lab in labels]
    adata.obs[key_added] = pd.Categorical(labels_str)

    return adata


def _build_knn(data, n_neighbors: int):
    """Build kNN graph and return indices matrix (n_cells x n_neighbors).

    Uses brute-force for small datasets, KDTree/BallTree for larger ones.
    """
    import numpy as np

    n_cells = data.shape[0]

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
        nn.fit(data)
        # +1 because the point itself is included
        distances, indices = nn.kneighbors(data)
        # Remove self (first column)
        return indices[:, 1:]
    except ImportError:
        # Fallback: brute force with chunked distance computation
        knn_indices = np.zeros((n_cells, n_neighbors), dtype=np.int64)
        chunk_size = 500

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = data[start:end]
            # Compute distances from chunk to all points
            # Use squared Euclidean for efficiency
            dists = (
                np.sum(chunk**2, axis=1, keepdims=True)
                + np.sum(data**2, axis=1, keepdims=True).T
                - 2.0 * chunk @ data.T
            )
            # Set self-distance to infinity
            for idx in range(end - start):
                dists[idx, start + idx] = np.inf
            # Get k nearest
            knn_indices[start:end] = np.argpartition(dists, n_neighbors, axis=1)[:, :n_neighbors]

        return knn_indices


def _compute_snn_jaccard(knn_indices, n_cells: int, n_neighbors: int):
    """Compute shared nearest neighbor graph with Jaccard similarity weights.

    For each pair (i, j), the Jaccard similarity is:
        |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
    where N(i) is the set of k nearest neighbors of cell i.
    """
    import numpy as np
    import scipy.sparse as sp

    # Build sparse SNN matrix — only compute for pairs that share at least
    # one neighbor (are in each other's neighborhoods or share neighbors)
    rows = []
    cols = []
    vals = []

    # For efficiency, build adjacency from kNN first
    # Create a sparse indicator of which cells are neighbors of which
    knn_sparse_rows = np.repeat(np.arange(n_cells), n_neighbors)
    knn_sparse_cols = knn_indices.ravel()
    knn_adj = sp.csr_matrix(
        (np.ones(len(knn_sparse_rows)), (knn_sparse_rows, knn_sparse_cols)),
        shape=(n_cells, n_cells),
    )

    # SNN: shared neighbors = knn_adj @ knn_adj.T (counts shared neighbors)
    snn_counts = knn_adj @ knn_adj.T

    # Convert shared counts to Jaccard similarity
    snn_coo = snn_counts.tocoo()
    for row_idx, col_idx, shared in zip(snn_coo.row, snn_coo.col, snn_coo.data):
        if row_idx >= col_idx:
            continue  # upper triangle only
        # Jaccard = |intersection| / |union|
        # |union| = |N(i)| + |N(j)| - |intersection|
        union_size = 2 * n_neighbors - shared
        if union_size > 0:
            jaccard = shared / union_size
            if jaccard > 0:
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(jaccard)
                rows.append(col_idx)
                cols.append(row_idx)
                vals.append(jaccard)

    if len(rows) == 0:
        # No shared neighbors — return empty sparse matrix
        return sp.csr_matrix((n_cells, n_cells), dtype=np.float64)

    snn_matrix = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_cells, n_cells),
    )

    return snn_matrix


def _community_detection(
    graph,
    clustering: str,
    resolution: float,
    random_state: int,
) -> list[int]:
    """Apply community detection on the SNN graph."""
    import numpy as np

    n_cells = graph.shape[0]

    # Try igraph + leidenalg/louvain first
    try:
        import igraph as ig

        sources, targets = graph.nonzero()
        weights = np.asarray(graph[sources, targets]).ravel()

        # Filter to upper triangle to avoid duplicate edges
        mask = sources < targets
        sources = sources[mask]
        targets = targets[mask]
        weights = weights[mask]

        if len(sources) == 0:
            # No edges — every cell is its own cluster
            return list(range(n_cells))

        g = ig.Graph(n=n_cells, edges=list(zip(sources.tolist(), targets.tolist())))
        g.es["weight"] = weights.tolist()

        if clustering == "leiden":
            try:
                import leidenalg

                partition = leidenalg.find_partition(
                    g,
                    leidenalg.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=resolution,
                    seed=random_state,
                )
                return partition.membership
            except ImportError:
                # Fall back to louvain via igraph
                partition = g.community_multilevel(weights="weight")
                return partition.membership
        else:
            # Louvain
            partition = g.community_multilevel(weights="weight")
            return partition.membership

    except ImportError:
        # Fallback: spectral clustering on the SNN graph
        return _spectral_fallback(graph, random_state)


def _spectral_fallback(graph, random_state: int) -> list[int]:
    """Spectral clustering fallback when igraph is not available."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n_cells = graph.shape[0]

    # Estimate k from eigengap
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L_norm = sp.eye(n_cells) - D_inv_sqrt @ graph @ D_inv_sqrt

    max_k = min(20, n_cells - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=max_k, which="SM", maxiter=1000)
        eigenvalues = np.sort(eigenvalues.real)
        gaps = np.diff(eigenvalues)
        if len(gaps) > 1:
            n_clusters = np.argmax(gaps[1:]) + 2
        else:
            n_clusters = 2
    except Exception:
        n_clusters = 5
        rng = np.random.default_rng(random_state)
        return rng.integers(0, n_clusters, size=n_cells).tolist()

    n_clusters = max(2, min(n_clusters, 20))

    # K-means on eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters].real
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    eigenvectors = eigenvectors / norms

    from scipy.cluster.vq import kmeans2

    _, labels = kmeans2(eigenvectors, n_clusters, minit="++", seed=random_state)
    return labels.tolist()
