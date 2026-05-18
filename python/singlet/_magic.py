# SPDX-License-Identifier: MIT
"""MAGIC imputation for AnnData objects.

Provides singlet.magic() — Markov Affinity-based Graph Imputation of Cells,
a diffusion-based method for denoising single-cell RNA-seq data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata


def magic(
    adata,
    *,
    n_neighbors: int = 30,
    t: int = 3,
    knn_dist: str = "euclidean",
    random_state: int = 0,
    use_rep: str | None = None,
    n_pcs: int | None = None,
) -> anndata.AnnData:
    """Denoise gene expression using MAGIC (diffusion-based imputation).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    n_neighbors : int, default 30
        Number of nearest neighbors for the kNN graph.
    t : int, default 3
        Diffusion time (power of the Markov transition matrix).
        Higher values produce more smoothing.
    knn_dist : str, default "euclidean"
        Distance metric for nearest neighbor search.
        Any metric supported by sklearn NearestNeighbors.
    random_state : int, default 0
        Random seed for reproducibility.
    use_rep : str or None, default None
        Key in ``adata.obsm`` to use as the representation for kNN.
        If None, uses 'X_pca' if available, else ``adata.X``.
    n_pcs : int or None, default None
        Number of columns to use from the representation.
        If None, uses all columns.

    Returns
    -------
    anndata.AnnData
        The input ``adata`` with ``adata.layers['magic']`` set to the
        imputed expression matrix (float32).

    Notes
    -----
    Implements the MAGIC algorithm (van Dijk et al., Cell 2018) with an
    adaptive Gaussian kernel. The affinity between cells i and j is:

        W[i,j] = exp(-d(i,j)^2 / (sigma_i * sigma_j))

    where sigma_i is the distance from cell i to its k-th neighbor.
    The resulting affinity matrix is row-normalized into a Markov
    transition matrix and raised to power ``t`` before being applied
    to the original expression matrix.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.pca(adata)
    >>> singlet.magic(adata, t=3)
    >>> adata.layers['magic'].shape  # (n_cells, n_genes)
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    if not hasattr(adata, "X") or not hasattr(adata, "obsm"):
        raise TypeError(f"magic() requires an AnnData object, got {type(adata).__name__}")

    n_cells = adata.shape[0]

    # --- Step 1: Get representation for kNN ---
    if use_rep is not None:
        rep = adata.obsm[use_rep]
    elif "X_pca" in adata.obsm:
        rep = adata.obsm["X_pca"]
    else:
        rep = adata.X
        if sp.issparse(rep):
            rep = rep.toarray()

    rep = np.asarray(rep, dtype=np.float64)

    if n_pcs is not None:
        rep = rep[:, :n_pcs]

    # --- Step 2: Compute kNN graph ---
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=knn_dist,
        algorithm="auto",
    )
    nn.set_params(random_state=random_state) if hasattr(nn, "random_state") else None
    nn.fit(rep)
    distances, indices = nn.kneighbors(rep)

    # --- Step 3: Build adaptive Gaussian affinity kernel ---
    # sigma[i] = distance to k-th neighbor (last column)
    sigma = distances[:, -1].copy()
    sigma[sigma == 0] = 1e-10  # avoid division by zero

    # Build sparse affinity matrix
    row_idx = np.repeat(np.arange(n_cells), n_neighbors)
    col_idx = indices.ravel()
    dists_flat = distances.ravel()

    # W[i,j] = exp(-d^2 / (sigma_i * sigma_j))
    sigma_i = sigma[row_idx]
    sigma_j = sigma[col_idx]
    affinities = np.exp(-(dists_flat**2) / (sigma_i * sigma_j))

    kernel = sp.csr_matrix(
        (affinities, (row_idx, col_idx)),
        shape=(n_cells, n_cells),
    )

    # Symmetrize: W = (W + W.T) / 2
    kernel = (kernel + kernel.T) / 2.0

    # --- Step 4: Row-normalize to Markov transition matrix ---
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv_row_sums = sp.diags(1.0 / row_sums)
    transition = inv_row_sums @ kernel

    # --- Step 5: Raise to power t ---
    powered = transition
    for _ in range(t - 1):
        powered = powered @ transition

    # --- Step 6: Apply imputed = T^t @ X ---
    expr = adata.X
    if sp.issparse(expr):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float64)

    imputed = powered @ expr

    # --- Step 7: Store result ---
    adata.layers["magic"] = np.asarray(imputed, dtype=np.float32)

    return adata
