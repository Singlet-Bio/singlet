"""Rank genes by various criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def rank_genes(
    adata: AnnData,
    *,
    method: str = "variance",
    n_top: int | None = None,
    layer: str | None = None,
    groupby: str | None = None,
) -> pd.DataFrame:
    """Rank genes by various criteria.

    Parameters
    ----------
    adata
        Annotated data matrix.
    method
        Ranking criterion:
        - 'variance': Rank by variance across all cells
        - 'mean': Rank by mean expression
        - 'dispersion': Rank by dispersion (variance/mean)
        - 'dropout': Rank by dropout rate (fraction of zeros)
        - 'cv': Rank by coefficient of variation (std/mean)
    n_top
        Return only top N genes. None returns all.
    layer
        Layer to use.
    groupby
        If specified, ranks genes per group and returns group-specific rankings.

    Returns
    -------
    DataFrame with gene rankings and the computed metric.
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

    if groupby is not None and groupby in adata.obs.columns:
        # Per-group ranking
        groups = adata.obs[groupby].unique()
        frames = []
        for grp in sorted(groups, key=str):
            mask = (adata.obs[groupby] == grp).values
            X_grp = X[mask]
            df = _rank_single(X_grp, adata.var_names, method, n_top)
            df["group"] = grp
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    else:
        return _rank_single(X, adata.var_names, method, n_top)


def _rank_single(X: np.ndarray, var_names, method: str, n_top: int | None) -> pd.DataFrame:
    """Rank genes for a single group."""
    means = X.mean(axis=0)

    if method == "variance":
        metric = X.var(axis=0)
        metric_name = "variance"
    elif method == "mean":
        metric = means
        metric_name = "mean"
    elif method == "dispersion":
        var = X.var(axis=0)
        safe_mean = means.copy()
        safe_mean[safe_mean == 0] = 1
        metric = var / safe_mean
        metric_name = "dispersion"
    elif method == "dropout":
        metric = (X == 0).mean(axis=0)
        metric_name = "dropout_rate"
    elif method == "cv":
        std = X.std(axis=0)
        safe_mean = means.copy()
        safe_mean[safe_mean == 0] = 1
        metric = std / safe_mean
        metric_name = "cv"
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'variance', 'mean', 'dispersion', 'dropout', or 'cv'."
        )

    # Sort descending
    order = np.argsort(metric)[::-1]

    if n_top is not None:
        order = order[:n_top]

    df = pd.DataFrame(
        {
            "gene": [var_names[i] for i in order],
            metric_name: metric[order],
            "rank": range(1, len(order) + 1),
        }
    )

    return df
