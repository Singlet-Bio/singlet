# SPDX-License-Identifier: MIT
"""Marker gene specificity scoring.

Provides singlet.marker_specificity() — for each gene in each cluster,
compute AUROC, fold-change, detection rate, and specificity index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def marker_specificity(
    adata: "AnnData",
    *,
    groupby: str = "leiden",
    method: str = "auroc",
    n_top_genes: int = 50,
) -> "pd.DataFrame":
    """Compute marker gene specificity scores for each group.

    For each gene in each cluster, computes AUROC (or Cohen's d),
    log2 fold-change, detection rate, and specificity index (expression
    in cluster / max expression in any other cluster).

    Parameters
    ----------
    adata
        Annotated data matrix with log-normalized counts in .X.
    groupby
        Key in adata.obs to group cells by (e.g., 'leiden', 'cell_type').
    method
        Scoring method: 'auroc' for area under ROC curve, or 'cohen_d'
        for Cohen's d effect size.
    n_top_genes
        Number of top marker genes to report per group (ranked by method score).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene, group, auroc (or cohen_d), specificity,
        detection_rate, log2fc. Also stored in adata.uns['marker_specificity'].

    Raises
    ------
    ValueError
        If groupby key is not found in adata.obs.
        If method is not 'auroc' or 'cohen_d'.
    KeyError
        If groupby is not in adata.obs.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> markers = singlet.marker_specificity(adata, groupby='leiden')
    >>> markers.head()
    """
    import numpy as np
    import pandas as pd

    if method not in ("auroc", "cohen_d"):
        msg = f"method must be 'auroc' or 'cohen_d', got '{method}'"
        raise ValueError(msg)

    if groupby not in adata.obs.columns:
        msg = f"'{groupby}' not found in adata.obs. Available: {list(adata.obs.columns)}"
        raise KeyError(msg)

    # Get expression matrix as dense
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    groups = adata.obs[groupby].values
    unique_groups = np.unique(groups)

    # Precompute group masks and mean expressions
    group_masks = {}
    group_means = {}
    group_detection = {}
    for grp in unique_groups:
        mask = groups == grp
        group_masks[grp] = mask
        expr = X[mask]
        group_means[grp] = expr.mean(axis=0)
        group_detection[grp] = (expr > 0).mean(axis=0)

    results = []
    gene_names = np.asarray(adata.var_names)

    for grp in unique_groups:
        mask_in = group_masks[grp]
        mask_out = ~mask_in
        expr_in = X[mask_in]
        expr_out = X[mask_out]
        n_in = mask_in.sum()
        n_out = mask_out.sum()

        mean_in = group_means[grp]
        mean_out = expr_out.mean(axis=0)

        # Log2 fold-change (add pseudocount to avoid log(0))
        pseudocount = 1e-9
        log2fc = np.log2((mean_in + pseudocount) / (mean_out + pseudocount))

        # Detection rate in the group
        detection_rate = group_detection[grp]

        # Specificity index: mean_in / max(mean in any other group)
        other_means = np.array([group_means[g] for g in unique_groups if g != grp])
        if len(other_means) > 0:
            max_other = other_means.max(axis=0)
            specificity = (mean_in + pseudocount) / (max_other + pseudocount)
        else:
            specificity = np.ones(len(gene_names))

        # Primary score
        if method == "auroc":
            scores = _compute_auroc_vectorized(expr_in, expr_out, n_in, n_out)
        else:
            scores = _compute_cohen_d_vectorized(expr_in, expr_out, n_in, n_out)

        # Rank and select top genes
        top_idx = np.argsort(-scores)[:n_top_genes]

        for idx in top_idx:
            row = {
                "gene": gene_names[idx],
                "group": str(grp),
                method: float(scores[idx]),
                "specificity": float(specificity[idx]),
                "detection_rate": float(detection_rate[idx]),
                "log2fc": float(log2fc[idx]),
            }
            results.append(row)

    df = pd.DataFrame(results)
    adata.uns["marker_specificity"] = df

    return df


def _compute_auroc_vectorized(expr_in, expr_out, n_in: int, n_out: int):
    """Compute AUROC for each gene (vectorized via Mann-Whitney U)."""
    import numpy as np

    n_genes = expr_in.shape[1]
    aurocs = np.empty(n_genes, dtype=np.float64)

    for gene_idx in range(n_genes):
        vals_in = expr_in[:, gene_idx]
        vals_out = expr_out[:, gene_idx]
        # Mann-Whitney U statistic → AUROC
        # AUROC = U / (n_in * n_out)
        combined = np.concatenate([vals_in, vals_out])
        ranks = _rankdata(combined)
        rank_sum_in = ranks[:n_in].sum()
        u_stat = rank_sum_in - n_in * (n_in + 1) / 2
        aurocs[gene_idx] = u_stat / (n_in * n_out) if (n_in * n_out) > 0 else 0.5

    return aurocs


def _rankdata(arr):
    """Rank data with average tie-breaking."""
    import numpy as np

    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(arr))

    arr_sorted = arr[sorter]
    obs = np.concatenate(([True], arr_sorted[1:] != arr_sorted[:-1]))
    dense = np.cumsum(obs)[inv]

    # Average ranks for ties
    count = np.bincount(dense)
    cumcount = np.cumsum(count)
    # For each unique value, average rank = (start_rank + end_rank) / 2
    result = np.empty(len(arr), dtype=np.float64)
    for val in range(len(count)):
        mask = dense == val
        start = cumcount[val] - count[val]
        end = cumcount[val]
        result[mask] = (start + end + 1) / 2.0

    return result


def _compute_cohen_d_vectorized(expr_in, expr_out, n_in: int, n_out: int):
    """Compute Cohen's d for each gene."""
    import numpy as np

    mean_in = expr_in.mean(axis=0)
    mean_out = expr_out.mean(axis=0)
    var_in = expr_in.var(axis=0, ddof=1) if n_in > 1 else np.zeros(expr_in.shape[1])
    var_out = expr_out.var(axis=0, ddof=1) if n_out > 1 else np.zeros(expr_out.shape[1])

    # Pooled standard deviation
    pooled_var = ((n_in - 1) * var_in + (n_out - 1) * var_out) / max(n_in + n_out - 2, 1)
    pooled_std = np.sqrt(pooled_var + 1e-12)  # avoid division by zero

    cohen_d = (mean_in - mean_out) / pooled_std
    return cohen_d
