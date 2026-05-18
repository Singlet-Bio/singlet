# SPDX-License-Identifier: MIT
"""Feature scaling (z-score normalization) for AnnData objects.

Provides singlet.scale() — standardizes each gene to zero mean and unit
variance, with optional clipping of extreme values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def scale(
    adata,
    *,
    max_value: Optional[float] = 10.0,
    zero_center: bool = True,
    inplace: bool = True,
) -> Optional["np.ndarray"]:
    """Scale gene expression to zero mean and unit variance.

    Standardizes each gene (column) independently. Optionally clips
    extreme values. Converts sparse matrices to dense (necessary for
    zero-centering).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should be log-normalized.
    max_value : float or None, default 10.0
        Clip values to [-max_value, max_value] after scaling.
        If None, no clipping is applied.
    zero_center : bool, default True
        If True, center each gene to zero mean. If False, only
        scale by standard deviation (preserves sparsity for
        non-negative data, though result is still dense).
    inplace : bool, default True
        If True, modifies adata.X in place.
        If False, returns scaled matrix without modifying adata.

    Returns
    -------
    numpy.ndarray or None
        Scaled matrix (n_cells × n_genes) if inplace=False.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.scale(adata)
    >>> adata.X.mean(axis=0)[:5]  # approximately zero
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"scale() requires an AnnData object, got {type(adata).__name__}")

    X = adata.X

    # Convert to dense float64 (zero-centering destroys sparsity)
    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float64)
    else:
        X = np.array(X, dtype=np.float64)

    # Compute per-gene mean and std
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Avoid division by zero for constant genes
    std[std == 0] = 1.0

    if zero_center:
        X = (X - mean) / std
    else:
        X = X / std

    # Clip extreme values
    if max_value is not None:
        np.clip(X, -max_value, max_value, out=X)

    if inplace:
        adata.X = X
        return None
    return X
