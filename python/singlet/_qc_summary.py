# SPDX-License-Identifier: MIT
"""Comprehensive QC summary statistics for AnnData objects.

Provides singlet.qc_summary() — computes per-cell quality metrics and
returns a summary DataFrame with descriptive statistics.
"""

from __future__ import annotations


def qc_summary(
    adata,
    *,
    groupby: str | None = None,
    mito_prefix: str = "MT-",
    ribo_prefix: str = "RPS|RPL",
):
    """Compute comprehensive QC summary statistics.

    Calculates per-cell quality metrics and returns a summary DataFrame
    with descriptive statistics (median, mean, std, Q25, Q75).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (raw counts recommended).
    groupby : str or None, default None
        Key in adata.obs for grouping cells. If provided, computes
        summary statistics per group.
    mito_prefix : str, default 'MT-'
        Prefix for mitochondrial gene names.
    ribo_prefix : str, default 'RPS|RPL'
        Pipe-separated prefixes for ribosomal gene names.


    Returns
    -------
    pandas.DataFrame
        Summary statistics DataFrame. If groupby is None, returns overall
        statistics. If groupby is specified, returns per-group statistics.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> summary = singlet.qc_summary(adata)
    >>> summary
    >>> # Per-cluster summary
    >>> summary = singlet.qc_summary(adata, groupby='leiden')
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    # Validate inputs
    if groupby is not None and groupby not in adata.obs.columns:
        raise KeyError(
            f"groupby key '{groupby}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )

    X = adata.X
    n_cells, n_genes_total = X.shape
    gene_names = list(adata.var_names)

    # Compute per-cell metrics
    if sp.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)
    else:
        total_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)

    # Mitochondrial percentage
    mito_mask = np.array([g.startswith(mito_prefix) for g in gene_names], dtype=bool)
    if mito_mask.any():
        if sp.issparse(X):
            mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
        else:
            mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
        pct_mito = mito_counts / (total_counts + 1e-10) * 100
    else:
        pct_mito = np.zeros(n_cells, dtype=np.float64)

    # Ribosomal percentage
    ribo_prefixes = [p.strip() for p in ribo_prefix.split("|") if p.strip()]
    ribo_mask = np.array(
        [any(g.startswith(rp) for rp in ribo_prefixes) for g in gene_names],
        dtype=bool,
    )
    if ribo_mask.any():
        if sp.issparse(X):
            ribo_counts = np.asarray(X[:, ribo_mask].sum(axis=1)).ravel()
        else:
            ribo_counts = np.asarray(X[:, ribo_mask].sum(axis=1)).ravel()
        pct_ribo = ribo_counts / (total_counts + 1e-10) * 100
    else:
        pct_ribo = np.zeros(n_cells, dtype=np.float64)

    # Complexity: log(n_genes) / log(total_counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_genes = np.log(n_genes)
        log_counts = np.log(total_counts)
        complexity = np.where(log_counts > 0, log_genes / log_counts, 0.0)

    # Store in adata.obs
    adata.obs["qc_n_genes"] = n_genes.astype(int)
    adata.obs["qc_total_counts"] = total_counts
    adata.obs["qc_pct_mito"] = pct_mito
    adata.obs["qc_pct_ribo"] = pct_ribo
    adata.obs["qc_complexity"] = complexity

    # Include doublet_score if available
    has_doublet = "doublet_score" in adata.obs.columns
    if has_doublet:
        doublet_scores = adata.obs["doublet_score"].values.astype(np.float64)
    else:
        doublet_scores = None

    # Build metrics DataFrame for summary computation
    metrics_dict = {
        "n_genes": n_genes,
        "total_counts": total_counts,
        "pct_mito": pct_mito,
        "pct_ribo": pct_ribo,
        "complexity": complexity,
    }
    if has_doublet:
        metrics_dict["doublet_score"] = doublet_scores

    metrics_df = pd.DataFrame(metrics_dict, index=adata.obs_names)

    # Compute summary statistics
    def _summarize(df):
        """Compute summary stats for a DataFrame of metrics."""
        summary = {}
        for col in df.columns:
            vals = df[col].dropna()
            summary[col] = {
                "median": vals.median(),
                "mean": vals.mean(),
                "std": vals.std(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
                "min": vals.min(),
                "max": vals.max(),
            }
        return pd.DataFrame(summary).T

    if groupby is None:
        summary_df = _summarize(metrics_df)
        summary_df.index.name = "metric"
    else:
        groups = adata.obs[groupby]
        summaries = []
        for group_name, group_idx in metrics_df.groupby(groups):
            group_summary = _summarize(group_idx)
            group_summary["group"] = group_name
            summaries.append(group_summary)
        summary_df = pd.concat(summaries)
        summary_df.index.name = "metric"
        # Reorganize columns
        cols = ["group"] + [c for c in summary_df.columns if c != "group"]
        summary_df = summary_df[cols].reset_index()

    return summary_df
