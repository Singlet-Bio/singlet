"""UMAP embedding for AnnData objects.

Provides singlet.umap() — computes 2D/3D UMAP embedding from the kNN graph
for visualization. Uses a pure-scipy spectral initialization when the
`umap-learn` package is not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def umap(
    adata,
    *,
    n_components: int = 2,
    min_dist: float = 0.5,
    spread: float = 1.0,
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """Compute UMAP embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have obsp['connectivities']
        (run singlet.neighbors() first).
    n_components : int, default 2
        Dimension of the UMAP embedding (2 or 3).
    min_dist : float, default 0.5
        Minimum distance between points in the embedding.
    spread : float, default 1.0
        Scale of embedded points.
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        If True, stores embedding in adata.obsm['X_umap']. Returns None.
        If False, returns the embedding array.

    Returns
    -------
    numpy.ndarray or None
        UMAP embedding of shape (n_cells, n_components) if inplace=False.

    Notes
    -----
    Uses the `umap-learn` package if available. Otherwise falls back to
    a spectral embedding (first n eigenvectors of the graph Laplacian),
    which gives a reasonable 2D layout without UMAP's force-directed
    refinement.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.umap(adata)
    >>> adata.obsm['X_umap'].shape  # (n_cells, 2)
    """
    import numpy as np

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"umap() requires an AnnData object, got {type(adata).__name__}")

    if "connectivities" not in adata.obsp:
        raise KeyError(
            "'connectivities' not found in adata.obsp. Run singlet.neighbors(adata) first."
        )

    conn = adata.obsp["connectivities"]

    try:
        embedding = _umap_learn(conn, n_components, min_dist, spread, random_state)
    except ImportError:
        embedding = _spectral_embedding(conn, n_components, random_state)

    embedding = embedding.astype(np.float32)

    if inplace:
        adata.obsm["X_umap"] = embedding
        return None
    else:
        return embedding


def _umap_learn(conn, n_components, min_dist, spread, random_state):
    """UMAP via the umap-learn package."""
    import umap as umap_pkg

    reducer = umap_pkg.UMAP(
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        metric="precomputed",
    )
    # Convert connectivity to distance (1 - normalized_weight)
    import numpy as np

    conn_dense = conn.toarray()
    max_val = conn_dense.max()
    if max_val > 0:
        dist = 1.0 - conn_dense / max_val
    else:
        dist = np.ones_like(conn_dense)
    np.fill_diagonal(dist, 0)
    return reducer.fit_transform(dist)


def _spectral_embedding(conn, n_components, random_state):
    """Spectral embedding fallback (graph Laplacian eigenvectors)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n_cells = conn.shape[0]

    # Normalized Laplacian
    degrees = np.asarray(conn.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L_norm = sp.eye(n_cells) - D_inv_sqrt @ conn @ D_inv_sqrt

    # Get smallest eigenvectors (skip the trivial constant one)
    n_ev = min(n_components + 1, n_cells - 1)
    try:
        _, eigenvectors = eigsh(L_norm, k=n_ev, which="SM", maxiter=2000)
        # Skip the first eigenvector (constant)
        embedding = eigenvectors[:, 1 : n_components + 1].real
    except Exception:
        # Fallback to random if eigsh fails
        rng = np.random.default_rng(random_state)
        embedding = rng.standard_normal((n_cells, n_components))

    # Scale for nicer visualization
    embedding = embedding - embedding.mean(axis=0)
    scale = np.abs(embedding).max()
    if scale > 0:
        embedding = embedding / scale * 10

    return embedding
