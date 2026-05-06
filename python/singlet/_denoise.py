"""SVD-based denoising for AnnData objects.

Provides singlet.denoise() — reconstruct expression matrix using
truncated SVD to remove technical noise while preserving biological
signal. Stores denoised values in adata.layers['denoised'].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def denoise(
    adata: AnnData,
    *,
    n_components: int = 50,
    method: str = "svd",
    layer: str | None = None,
) -> AnnData:
    """Denoise expression matrix via truncated SVD reconstruction.

    Computes a low-rank approximation X ≈ U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
    which retains the top-k singular value components and discards noise
    in the remaining dimensions.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    n_components : int, default 50
        Number of SVD components to retain for reconstruction.
    method : str, default 'svd'
        Decomposition method:
        - 'svd': Truncated SVD directly on the matrix.
        - 'pca': Center the matrix first (mean-subtract), then SVD.
    layer : str or None, default None
        Layer to use as input. If None, uses adata.X.

    Returns
    -------
    anndata.AnnData
        The input adata with adata.layers['denoised'] set to the
        low-rank reconstructed matrix.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.denoise(adata, n_components=30)
    >>> adata.layers['denoised'].shape == adata.X.shape
    True
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.utils.extmath import randomized_svd

    if method not in ("svd", "pca"):
        msg = f"method must be 'svd' or 'pca', got {method!r}"
        raise ValueError(msg)

    # Get input matrix
    mat = adata.layers[layer] if layer is not None else adata.X
    if mat is None:
        msg = "adata.X is None; provide a layer or set adata.X"
        raise ValueError(msg)

    # Cap n_components to the matrix dimensions
    max_components = min(mat.shape[0], mat.shape[1]) - 1
    n_components = min(n_components, max(1, max_components))

    if method == "svd":
        # Truncated SVD without centering
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        transformed = svd.fit_transform(mat)  # U * S
        denoised = transformed @ svd.components_  # (U*S) @ Vt
    else:
        # PCA: center first, then SVD
        if sp.issparse(mat):
            dense = np.asarray(mat.todense()).astype(np.float32)
        else:
            dense = np.asarray(mat, dtype=np.float32)

        mean_vec = dense.mean(axis=0)
        centered = dense - mean_vec

        u_mat, sigma, vt = randomized_svd(centered, n_components=n_components, random_state=0)
        # Reconstruct: U @ diag(S) @ Vt + mean
        denoised = (u_mat * sigma) @ vt + mean_vec

    # Store as float32 dense
    if sp.issparse(denoised):
        denoised = np.asarray(denoised)
    denoised = np.asarray(denoised, dtype=np.float32)

    adata.layers["denoised"] = denoised
    return adata
