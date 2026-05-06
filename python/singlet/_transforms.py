"""Expression matrix transformations."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def log1p(
    adata: AnnData,
    *,
    layer: str | None = None,
    base: float | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Compute log1p (natural log of 1+x) of the expression matrix.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to transform. None uses .X.
    base
        Base of logarithm. None uses natural log.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True.
    """
    from scipy.sparse import issparse

    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data)
        if base is not None:
            X.data /= np.log(base)
    else:
        X = np.log1p(np.asarray(X, dtype=np.float64))
        if base is not None:
            X /= np.log(base)
        X = X.astype(np.float32)

    if layer is not None:
        adata.layers[layer] = X
    else:
        adata.X = X

    return adata if copy else None


def expm1(
    adata: AnnData,
    *,
    layer: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Compute expm1 (exp(x) - 1) — inverse of log1p.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to transform. None uses .X.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True.
    """
    from scipy.sparse import issparse

    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        X = X.copy()
        X.data = np.expm1(X.data)
    else:
        X = np.expm1(np.asarray(X, dtype=np.float64)).astype(np.float32)

    if layer is not None:
        adata.layers[layer] = X
    else:
        adata.X = X

    return adata if copy else None


def sqrt_transform(
    adata: AnnData,
    *,
    layer: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Compute square root transformation.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to transform. None uses .X.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True.
    """
    from scipy.sparse import issparse

    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        X = X.copy()
        X.data = np.sqrt(X.data)
    else:
        X = np.sqrt(np.asarray(X, dtype=np.float64)).astype(np.float32)

    if layer is not None:
        adata.layers[layer] = X
    else:
        adata.X = X

    return adata if copy else None
