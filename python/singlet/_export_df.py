"""DataFrame export utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def to_df(
    adata: AnnData,
    *,
    layer: str | None = None,
) -> pd.DataFrame:
    """Convert expression matrix to a DataFrame.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to export. None uses .X.

    Returns
    -------
    DataFrame with observations as rows and variables as columns.
    """
    from scipy.sparse import issparse

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        X = np.asarray(X.todense())
    else:
        X = np.asarray(X)

    return pd.DataFrame(
        X,
        index=adata.obs_names,
        columns=adata.var_names,
    )


def obs_df(
    adata: AnnData,
    keys: list[str],
    *,
    layer: str | None = None,
) -> pd.DataFrame:
    """Get a DataFrame with observation annotations and/or gene expression.

    Parameters
    ----------
    adata
        Annotated data matrix.
    keys
        List of keys. Can be:
        - Column names from .obs
        - Gene names from .var_names
        - Keys from .obsm (prefixed with 'obsm:')
    layer
        Layer to use for gene expression. None uses .X.

    Returns
    -------
    DataFrame with requested columns.
    """
    from scipy.sparse import issparse

    df = pd.DataFrame(index=adata.obs_names)

    for key in keys:
        if key.startswith("obsm:"):
            obsm_key = key[5:]
            if obsm_key in adata.obsm:
                arr = adata.obsm[obsm_key]
                for i in range(arr.shape[1]):
                    df[f"{obsm_key}_{i}"] = arr[:, i]
            else:
                raise KeyError(f"'{obsm_key}' not found in .obsm.")
        elif key in adata.obs.columns:
            df[key] = adata.obs[key].values
        elif key in adata.var_names:
            if layer is not None:
                X = adata.layers[layer]
            else:
                X = adata.X

            idx = list(adata.var_names).index(key)
            if issparse(X):
                df[key] = np.asarray(X[:, idx].todense()).flatten()
            else:
                df[key] = np.asarray(X[:, idx]).flatten()
        else:
            raise KeyError(f"'{key}' not found in .obs, .var_names, or .obsm.")

    return df


def var_df(
    adata: AnnData,
    keys: list[str],
    *,
    layer: str | None = None,
) -> pd.DataFrame:
    """Get a DataFrame with variable annotations and/or cell expression.

    Parameters
    ----------
    adata
        Annotated data matrix.
    keys
        List of keys from .var columns or cell names.
    layer
        Layer to use. None uses .X.

    Returns
    -------
    DataFrame with variables as rows.
    """
    from scipy.sparse import issparse

    df = pd.DataFrame(index=adata.var_names)

    for key in keys:
        if key in adata.var.columns:
            df[key] = adata.var[key].values
        elif key in adata.obs_names:
            if layer is not None:
                X = adata.layers[layer]
            else:
                X = adata.X

            idx = list(adata.obs_names).index(key)
            if issparse(X):
                df[key] = np.asarray(X[idx].todense()).flatten()
            else:
                df[key] = np.asarray(X[idx]).flatten()
        else:
            raise KeyError(f"'{key}' not found in .var columns or .obs_names.")

    return df
