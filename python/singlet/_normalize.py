# SPDX-License-Identifier: MIT
"""Basic normalization for AnnData objects.

Provides singlet.normalize() — library-size normalization + log1p,
the most common preprocessing step in single-cell workflows. Keeps
raw counts in adata.layers["raw"] by default.
"""

from __future__ import annotations

from typing import Optional


def normalize(
    adata,
    *,
    target_sum: Optional[float] = 1e4,
    log: bool = True,
    keep_raw: bool = True,
    inplace: bool = True,
):
    """Normalize counts: library-size scaling + optional log1p.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with raw counts in X.
    target_sum : float or None, default 1e4
        Scale each cell to this total count. If None, scale to median
        library size (preserves relative depth differences).
    log : bool, default True
        Apply log1p(X) after scaling.
    keep_raw : bool, default True
        Store pre-normalization X in adata.layers["raw"].
    inplace : bool, default True
        If True, modifies adata.X in-place and returns None.
        If False, returns a normalized copy.

    Returns
    -------
    anndata.AnnData or None
        Normalized AnnData (if inplace=False), or None (if inplace=True).

    Notes
    -----
    Equivalent to scanpy's ``sc.pp.normalize_total`` + ``sc.pp.log1p``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)  # modifies in-place
    >>> adata.X.max()  # now log-normalized
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"normalize() requires an AnnData object, got {type(adata).__name__}")

    if not inplace:
        adata = adata.copy()

    X = adata.X

    # Save raw counts
    if keep_raw and "raw" not in adata.layers:
        adata.layers["raw"] = X.copy()

    # Compute library sizes (total counts per cell)
    if sp.issparse(X):
        lib_sizes = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    else:
        lib_sizes = X.sum(axis=1).astype(np.float64)

    # Determine scale factor
    if target_sum is None:
        target_sum = float(np.median(lib_sizes[lib_sizes > 0]))

    # Avoid division by zero
    lib_sizes[lib_sizes == 0] = 1.0
    scale_factors = target_sum / lib_sizes

    # Apply scaling
    if sp.issparse(X):
        # Efficient in-place scaling for sparse matrices
        X = X.astype(np.float32) if X.dtype != np.float32 else X.copy()
        # Multiply each row by its scale factor
        from scipy.sparse import diags

        scaling_matrix = diags(scale_factors)
        X = scaling_matrix @ X
        X = X.astype(np.float32)
    else:
        X = X.astype(np.float32)
        X *= scale_factors[:, np.newaxis].astype(np.float32)

    # Log transform
    if log:
        if sp.issparse(X):
            X.data = np.log1p(X.data)
        else:
            np.log1p(X, out=X)

    adata.X = X

    if not inplace:
        return adata
    return None
