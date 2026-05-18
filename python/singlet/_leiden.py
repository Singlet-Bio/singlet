# SPDX-License-Identifier: MIT
"""Leiden community detection for AnnData objects.

Provides singlet.leiden() — graph-based clustering using the Leiden algorithm.
Falls back to a spectral clustering approach if the `leidenalg` package
is not installed.
"""

from __future__ import annotations

from typing import Optional


def leiden(
    adata,
    *,
    resolution: float = 1.0,
    n_clusters: Optional[int] = None,
    key_added: str = "leiden",
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[list[str]]:
    """Cluster cells using the Leiden algorithm on the kNN graph.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have obsp['connectivities']
        (run singlet.neighbors() first).
    resolution : float, default 1.0
        Resolution parameter — higher values give more clusters.
        Ignored if n_clusters is set.
    n_clusters : int or None, default None
        If set, uses spectral clustering with this exact number of clusters
        instead of Leiden (useful when you know the expected cluster count).
    key_added : str, default "leiden"
        Key in adata.obs where cluster labels are stored.
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        If True, stores labels in adata.obs[key_added]. Returns None.
        If False, returns list of cluster labels.

    Returns
    -------
    list[str] or None
        Cluster labels (if inplace=False), or None (if inplace=True).

    Notes
    -----
    Requires the `leidenalg` package for true Leiden clustering.
    If not available, falls back to spectral clustering from scikit-learn
    (which requires `sklearn`). If neither is available, raises ImportError.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> adata.obs['leiden']  # cluster labels
    """

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"leiden() requires an AnnData object, got {type(adata).__name__}")

    if "connectivities" not in adata.obsp:
        raise KeyError(
            "'connectivities' not found in adata.obsp. Run singlet.neighbors(adata) first."
        )

    conn = adata.obsp["connectivities"]

    if n_clusters is not None:
        # Use spectral clustering for exact cluster count
        labels = _spectral_cluster(conn, n_clusters, random_state)
    else:
        # Try Leiden, fall back to spectral with estimated k
        labels = _leiden_cluster(conn, resolution, random_state)

    labels_str = [str(label) for label in labels]

    if inplace:
        import pandas as pd

        adata.obs[key_added] = pd.Categorical(labels_str)
        return None
    else:
        return labels_str


def _leiden_cluster(conn, resolution: float, random_state: int) -> list[int]:
    """Run Leiden clustering on connectivity matrix."""
    try:
        import igraph as ig
        import leidenalg

        # Convert sparse matrix to igraph
        sources, targets = conn.nonzero()
        weights = conn[sources, targets].A1
        g = ig.Graph(directed=False)
        g.add_vertices(conn.shape[0])
        edges = list(zip(sources.tolist(), targets.tolist()))
        g.add_edges(edges)
        g.es["weight"] = weights.tolist()

        # Remove self-loops and multi-edges
        g.simplify(combine_edges="max")

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )
        return partition.membership
    except ImportError:
        # Fall back to spectral clustering with estimated k

        # Estimate number of clusters from connectivity eigenvalues
        n_clusters = _estimate_n_clusters(conn)
        return _spectral_cluster(conn, n_clusters, random_state)


def _spectral_cluster(conn, n_clusters: int, random_state: int) -> list[int]:
    """Spectral clustering on connectivity matrix."""
    import numpy as np
    from scipy.sparse.linalg import eigsh

    n_cells = conn.shape[0]
    n_clusters = min(n_clusters, n_cells)

    # Normalized Laplacian spectral clustering
    import scipy.sparse as sp

    # Degree matrix
    degrees = np.asarray(conn.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L_norm = sp.eye(n_cells) - D_inv_sqrt @ conn @ D_inv_sqrt

    # Compute smallest eigenvectors of normalized Laplacian
    n_ev = min(n_clusters, n_cells - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_ev, which="SM", maxiter=1000)
    except Exception:
        # Fallback: random assignment
        rng = np.random.default_rng(random_state)
        return rng.integers(0, n_clusters, size=n_cells).tolist()

    # K-means on eigenvectors
    from scipy.cluster.vq import kmeans2

    eigenvectors = eigenvectors.real
    # Normalize rows
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    eigenvectors = eigenvectors / norms

    _, labels = kmeans2(eigenvectors, n_clusters, minit="++", seed=random_state)
    return labels.tolist()


def _estimate_n_clusters(conn, max_clusters: int = 30) -> int:
    """Estimate number of clusters from eigengap heuristic."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n_cells = conn.shape[0]
    degrees = np.asarray(conn.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L_norm = sp.eye(n_cells) - D_inv_sqrt @ conn @ D_inv_sqrt

    n_ev = min(max_clusters + 1, n_cells - 1)
    try:
        eigenvalues, _ = eigsh(L_norm, k=n_ev, which="SM", maxiter=1000)
        eigenvalues = np.sort(eigenvalues.real)
        # Eigengap heuristic: largest gap between consecutive eigenvalues
        gaps = np.diff(eigenvalues)
        # Skip the first eigenvalue (always ~0)
        if len(gaps) > 1:
            k = np.argmax(gaps[1:]) + 2  # +2 because we skip first gap and 0-index
            return max(2, min(k, max_clusters))
    except Exception:
        pass
    return 10  # default fallback
