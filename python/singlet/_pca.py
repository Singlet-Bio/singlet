"""Principal Component Analysis for AnnData objects.

Provides singlet.pca() — standard PCA for dimensionality reduction,
the step that typically follows normalization and HVG selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def pca(
    adata,
    *,
    n_comps: int = 50,
    use_highly_variable: bool = True,
    zero_center: bool = True,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """Compute PCA (Principal Component Analysis).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (normalized, log-transformed recommended).
    n_comps : int, default 50
        Number of principal components to compute.
    use_highly_variable : bool, default True
        If True and adata.var['highly_variable'] exists, only use HVGs.
        Falls back to all genes if column doesn't exist.
    zero_center : bool, default True
        If True, center the data before PCA (standard). Set False for
        very large sparse matrices where centering would densify.
    inplace : bool, default True
        If True, stores results in adata.obsm['X_pca'], adata.uns['pca'],
        and adata.varm['PCs']. Returns None.
        If False, returns the PCA embedding array.

    Returns
    -------
    numpy.ndarray or None
        PCA embedding of shape (n_cells, n_comps) if inplace=False.

    Notes
    -----
    Uses scipy.sparse.linalg.svds for sparse input (fast, memory-efficient)
    and numpy.linalg.svd for dense input.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> singlet.pca(adata)
    >>> adata.obsm['X_pca'].shape  # (n_cells, 50)
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"pca() requires an AnnData object, got {type(adata).__name__}")

    # Select genes
    if use_highly_variable and "highly_variable" in adata.var.columns:
        X = adata[:, adata.var["highly_variable"]].X
        var_mask = adata.var["highly_variable"].values
    else:
        X = adata.X
        var_mask = None

    n_cells, n_genes = X.shape
    n_comps = min(n_comps, n_cells - 1, n_genes - 1)

    if n_comps < 1:
        raise ValueError(
            f"Cannot compute PCA with n_comps={n_comps}. Need at least 2 cells and 2 genes."
        )

    if sp.issparse(X) and not zero_center:
        # Truncated SVD without centering (keeps sparse)
        from scipy.sparse.linalg import svds

        # svds returns components in ascending order
        U, S, Vt = svds(X.astype(np.float64), k=n_comps)
        # Reverse to descending order
        U = U[:, ::-1]
        S = S[::-1]
        Vt = Vt[::-1, :]
        X_pca = U * S
        components = Vt
        variance = S**2 / (n_cells - 1)
    else:
        # Center and compute SVD
        if sp.issparse(X):
            X_dense = X.toarray().astype(np.float64)
        else:
            X_dense = np.asarray(X, dtype=np.float64)

        mean = X_dense.mean(axis=0)
        X_centered = X_dense - mean

        # Use scipy's truncated SVD for efficiency
        from scipy.sparse.linalg import svds

        # svds works on dense arrays too
        X_sparse = sp.csr_matrix(X_centered)
        U, S, Vt = svds(X_sparse, k=n_comps)
        # Reverse to descending order
        U = U[:, ::-1]
        S = S[::-1]
        Vt = Vt[::-1, :]
        X_pca = U * S
        components = Vt
        variance = S**2 / (n_cells - 1)

    X_pca = X_pca.astype(np.float32)

    if inplace:
        adata.obsm["X_pca"] = X_pca
        adata.uns["pca"] = {
            "variance": variance.astype(np.float32),
            "variance_ratio": (variance / variance.sum()).astype(np.float32),
        }
        # Store loadings in varm if using all genes
        if var_mask is None:
            adata.varm["PCs"] = components.T.astype(np.float32)
        return None
    else:
        return X_pca
