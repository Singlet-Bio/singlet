# SPDX-License-Identifier: MIT
"""Pseudobulk aggregation with multi-key grouping support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def pseudobulk(
    adata: AnnData,
    groupby: str | list[str],
    *,
    layer: str | None = None,
    agg: str = "mean",
) -> AnnData:
    """Aggregate single-cell data to pseudobulk profiles.

    Creates a new AnnData where each observation represents an aggregate
    (pseudobulk) of all cells within a group, suitable for downstream
    differential expression or visualization.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        One or more keys in ``.obs`` to group cells by. If a list,
        groups are defined by the combination of all keys.
    layer
        Layer to aggregate. ``None`` uses ``.X``.
    agg
        Aggregation function: ``'mean'``, ``'sum'``, or ``'median'``.

    Returns
    -------
    New AnnData with ``n_obs`` = number of unique groups. Preserves
    ``.var`` metadata from the input. ``.obs`` contains group labels
    and a ``'n_cells'`` column with cell counts per group.
    """
    import pandas as pd
    from anndata import AnnData as _AnnData
    from scipy.sparse import issparse

    valid_agg = ("mean", "sum", "median")
    if agg not in valid_agg:
        raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'.")

    # Normalize groupby to list
    if isinstance(groupby, str):
        groupby_keys = [groupby]
    else:
        groupby_keys = list(groupby)

    # Validate keys exist
    for key in groupby_keys:
        if key not in adata.obs.columns:
            raise KeyError(f"'{key}' not found in .obs.")

    # Get expression matrix
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in .layers.")
        X = adata.layers[layer]
    else:
        X = adata.X

    # Build group labels
    if len(groupby_keys) == 1:
        group_series = adata.obs[groupby_keys[0]].astype(str)
    else:
        group_series = adata.obs[groupby_keys[0]].astype(str)
        for key in groupby_keys[1:]:
            group_series = group_series + "|" + adata.obs[key].astype(str)

    group_values = group_series.values
    unique_groups = sorted(set(group_values), key=str)

    n_groups = len(unique_groups)
    n_vars = adata.n_vars

    agg_X = np.zeros((n_groups, n_vars), dtype=np.float32)
    cell_counts = np.zeros(n_groups, dtype=np.int64)

    for idx, group in enumerate(unique_groups):
        mask = group_values == group
        cell_counts[idx] = mask.sum()

        if issparse(X):
            sub = np.asarray(X[mask].todense())
        else:
            sub = np.asarray(X[mask])

        if agg == "sum":
            agg_X[idx] = sub.sum(axis=0)
        elif agg == "mean":
            agg_X[idx] = sub.mean(axis=0)
        elif agg == "median":
            agg_X[idx] = np.median(sub, axis=0)

    # Build obs DataFrame
    obs_data = {"n_cells": cell_counts}

    if len(groupby_keys) == 1:
        obs_data[groupby_keys[0]] = unique_groups
    else:
        # Split composite labels back into individual columns
        for col_idx, key in enumerate(groupby_keys):
            obs_data[key] = [g.split("|")[col_idx] for g in unique_groups]

    obs_df = pd.DataFrame(obs_data)
    obs_df.index = pd.Index([str(g) for g in unique_groups])

    result = _AnnData(
        X=agg_X,
        obs=obs_df,
        var=adata.var.copy(),
    )

    return result
