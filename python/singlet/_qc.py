# SPDX-License-Identifier: MIT
"""Quality control metric calculation for AnnData objects.

Provides singlet.calculate_qc_metrics() — computes per-cell and per-gene
quality metrics including counts, gene detection, and mitochondrial fraction.
"""

from __future__ import annotations

from typing import Optional


def calculate_qc_metrics(
    adata,
    *,
    qc_vars: Optional[list[str]] = None,
    percent_top: Optional[list[int]] = None,
    inplace: bool = True,
) -> Optional[tuple]:
    """Calculate quality control metrics.

    Computes per-cell and per-gene metrics and optionally stores them
    in adata.obs and adata.var.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (raw counts recommended).
    qc_vars : list[str] or None, default None
        Keys in adata.var for sets of genes to compute metrics for
        (e.g., if adata.var["mt"] marks mitochondrial genes, pass ["mt"]).
        If None and gene names start with "MT-" or "mt-", auto-detects
        mitochondrial genes.
    percent_top : list[int] or None, default None
        List of integers for computing pct_counts_in_top_N_genes.
        E.g., [50, 100, 200]. If None, uses [50, 100, 200, 500].
    inplace : bool, default True
        If True, stores metrics in adata.obs and adata.var.
        If False, returns (cell_metrics_df, gene_metrics_df).

    Returns
    -------
    tuple of pandas.DataFrame or None
        (obs_metrics, var_metrics) if inplace=False.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.calculate_qc_metrics(adata)
    >>> adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].head()
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"calculate_qc_metrics() requires an AnnData object, got {type(adata).__name__}"
        )

    X = adata.X
    n_cells, n_genes = X.shape

    if percent_top is None:
        percent_top = [50, 100, 200, 500]

    # Per-cell metrics
    if sp.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel()
        n_genes_by_counts = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        total_counts = X.sum(axis=1).ravel()
        n_genes_by_counts = (X > 0).sum(axis=1).ravel()

    obs_metrics = pd.DataFrame(
        {
            "n_genes_by_counts": n_genes_by_counts.astype(int),
            "total_counts": total_counts,
            "log1p_total_counts": np.log1p(total_counts),
        },
        index=adata.obs_names,
    )

    # Percent top genes
    for n in percent_top:
        if n > n_genes:
            continue
        if sp.issparse(X):
            # For sparse: convert rows to dense for topk
            pct = np.zeros(n_cells, dtype=np.float64)
            for i in range(n_cells):
                row = np.asarray(X[i].todense()).ravel()
                top_sum = np.sort(row)[-n:].sum()
                pct[i] = top_sum / (total_counts[i] + 1e-10) * 100
        else:
            sorted_x = np.sort(X, axis=1)[:, ::-1]
            top_sum = sorted_x[:, :n].sum(axis=1)
            pct = top_sum / (total_counts + 1e-10) * 100
        obs_metrics[f"pct_counts_in_top_{n}_genes"] = pct

    # QC variable metrics (e.g., mitochondrial)
    if qc_vars is None:
        # Auto-detect mitochondrial genes
        var_names_list = list(adata.var_names)
        mt_mask = np.array([g.startswith("MT-") or g.startswith("mt-") for g in var_names_list])
        if mt_mask.any():
            if sp.issparse(X):
                mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel()
            else:
                mt_counts = X[:, mt_mask].sum(axis=1).ravel()
            obs_metrics["total_counts_mt"] = mt_counts
            obs_metrics["pct_counts_mt"] = mt_counts / (total_counts + 1e-10) * 100
    else:
        for var_key in qc_vars:
            if var_key in adata.var.columns:
                mask = adata.var[var_key].values.astype(bool)
                if sp.issparse(X):
                    var_counts = np.asarray(X[:, mask].sum(axis=1)).ravel()
                else:
                    var_counts = X[:, mask].sum(axis=1).ravel()
                obs_metrics[f"total_counts_{var_key}"] = var_counts
                obs_metrics[f"pct_counts_{var_key}"] = var_counts / (total_counts + 1e-10) * 100

    # Per-gene metrics
    if sp.issparse(X):
        total_counts_gene = np.asarray(X.sum(axis=0)).ravel()
        n_cells_by_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        total_counts_gene = X.sum(axis=0).ravel()
        n_cells_by_counts = (X > 0).sum(axis=0).ravel()

    mean_counts = total_counts_gene / n_cells
    pct_dropout = (1 - n_cells_by_counts / n_cells) * 100

    var_metrics = pd.DataFrame(
        {
            "n_cells_by_counts": n_cells_by_counts.astype(int),
            "mean_counts": mean_counts,
            "log1p_mean_counts": np.log1p(mean_counts),
            "pct_dropout_by_counts": pct_dropout,
            "total_counts": total_counts_gene,
            "log1p_total_counts": np.log1p(total_counts_gene),
        },
        index=adata.var_names,
    )

    if inplace:
        for col in obs_metrics.columns:
            adata.obs[col] = obs_metrics[col].values
        for col in var_metrics.columns:
            adata.var[col] = var_metrics[col].values
        return None
    return obs_metrics, var_metrics
