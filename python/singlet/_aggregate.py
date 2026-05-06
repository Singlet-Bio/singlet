"""Pseudobulk aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def aggregate(
    adata: AnnData,
    *,
    groupby: str,
    layer: str | None = None,
    method: str = "sum",
) -> AnnData:
    """Aggregate expression by group (pseudobulk).

    Creates a new AnnData with one observation per group, where the
    expression is aggregated across cells within each group.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Key in .obs for grouping cells.
    layer
        Layer to aggregate. None uses .X.
    method
        Aggregation method: 'sum', 'mean', or 'median'.

    Returns
    -------
    New AnnData with groups as observations.
    """
    from scipy.sparse import issparse

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    groups = adata.obs[groupby]
    unique_groups = sorted(groups.unique(), key=str)

    n_groups = len(unique_groups)
    n_vars = adata.n_vars

    agg_X = np.zeros((n_groups, n_vars), dtype=np.float32)

    for i, group in enumerate(unique_groups):
        mask = (groups == group).values
        if issparse(X):
            sub = np.asarray(X[mask].todense())
        else:
            sub = np.asarray(X[mask])

        if method == "sum":
            agg_X[i] = sub.sum(axis=0)
        elif method == "mean":
            agg_X[i] = sub.mean(axis=0)
        elif method == "median":
            agg_X[i] = np.median(sub, axis=0)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'sum', 'mean', or 'median'.")

    result = AnnData(
        X=agg_X,
        var=adata.var.copy(),
    )
    result.obs_names = pd.Index([str(g) for g in unique_groups])
    result.obs["n_cells"] = [int((groups == g).sum()) for g in unique_groups]

    return result
