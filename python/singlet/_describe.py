# SPDX-License-Identifier: MIT
"""Quick summary statistics for AnnData objects."""

from __future__ import annotations

from typing import Any


def describe(adata) -> dict[str, Any]:
    """Return summary statistics for an AnnData object.

    Gives a quick overview of dataset size, sparsity, count distribution,
    detected organism, and metadata columns — useful right after loading.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.

    Returns
    -------
    dict
        Summary with keys: n_cells, n_genes, sparsity, counts_per_cell
        (mean/median/min/max), genes_per_cell (mean/median/min/max),
        organism, obs_columns, var_columns, layers.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.describe(adata)
    {'n_cells': 5000, 'n_genes': 30000, 'sparsity': 0.95, ...}
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"describe() requires an AnnData object, got {type(adata).__name__}")

    X = adata.X
    n_cells, n_genes = X.shape

    # Sparsity
    if sp.issparse(X):
        nnz = X.nnz
        sparsity = 1.0 - nnz / (n_cells * n_genes) if n_cells * n_genes > 0 else 0.0
    else:
        nnz = int(np.count_nonzero(X))
        sparsity = 1.0 - nnz / (n_cells * n_genes) if n_cells * n_genes > 0 else 0.0

    # Counts per cell (total UMI/reads per cell)
    if n_cells == 0:
        counts_per_cell = np.array([], dtype=np.float64)
        genes_per_cell = np.array([], dtype=np.int64)
    elif sp.issparse(X):
        counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
        genes_per_cell = np.diff(X.tocsr().indptr)
    else:
        counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
        genes_per_cell = np.count_nonzero(X, axis=1)

    # Organism detection
    organism = adata.uns.get("organism", None) if hasattr(adata, "uns") else None

    # Metadata columns
    obs_columns = list(adata.obs.columns) if hasattr(adata, "obs") else []
    var_columns = list(adata.var.columns) if hasattr(adata, "var") else []
    layers = list(adata.layers.keys()) if hasattr(adata, "layers") else []

    def _stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
        return {
            "mean": round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "min": int(arr.min()),
            "max": int(arr.max()),
        }

    return {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "sparsity": round(sparsity, 4),
        "counts_per_cell": _stats(counts_per_cell),
        "genes_per_cell": _stats(genes_per_cell),
        "organism": organism,
        "obs_columns": obs_columns,
        "var_columns": var_columns,
        "layers": layers,
    }
