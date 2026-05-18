# SPDX-License-Identifier: MIT
"""Find marker genes for all clusters with filtering.

Provides singlet.find_all_markers() — comprehensive marker discovery
with fold change, percent expressing, and p-value filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def find_all_markers(
    adata: "AnnData",
    groupby: str = "leiden",
    *,
    method: str = "wilcoxon",
    min_fold_change: float = 1.5,
    min_pct: float = 0.1,
    max_pvalue: float = 0.05,
    n_top: Optional[int] = None,
    layer: Optional[str] = None,
) -> "pd.DataFrame":
    """Find marker genes for all clusters with comprehensive filtering.

    For each cluster/group, tests genes against all other cells (one-vs-rest),
    then filters by fold change, percent expressing, and statistical
    significance.

    Parameters
    ----------
    adata
        Annotated data matrix (log-normalized recommended).
    groupby
        Column in adata.obs to group cells by (e.g., 'leiden', 'cell_type').
    method
        Statistical test: 'wilcoxon' (rank-sum) or 't-test'.
    min_fold_change
        Minimum fold change (linear scale) to include a gene.
        A value of 1.5 means the gene must be 1.5x higher in the group.
    min_pct
        Minimum fraction of cells in the group expressing the gene
        (expression > 0).
    max_pvalue
        Maximum adjusted p-value (FDR) for significance.
    n_top
        If set, return only top N markers per group (ranked by score).
        If None, return all passing markers.
    layer
        Layer to use for expression. None uses .X.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - gene : str — gene name
        - group : str — cluster/group label
        - pvalue : float — raw p-value
        - fdr : float — adjusted p-value (Benjamini-Hochberg)
        - log2fc : float — log2 fold change (group vs rest)
        - pct_in : float — fraction expressing in group
        - pct_out : float — fraction expressing in rest
        - score : float — combined score (higher = better marker)

    Notes
    -----
    Results are also stored in adata.uns['all_markers'] as a dict of
    DataFrames keyed by group name.

    The score is computed as: -log10(fdr + 1e-300) * log2fc * pct_in,
    which rewards genes that are significant, highly upregulated, and
    broadly expressed in the group.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> markers = singlet.find_all_markers(adata, groupby='leiden')
    >>> markers.head(10)
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"find_all_markers() requires an AnnData object, got {type(adata).__name__}"
        )

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.columns")

    if method not in ("wilcoxon", "t-test"):
        raise ValueError(f"method must be 'wilcoxon' or 't-test', got '{method}'")

    if min_fold_change < 1.0:
        raise ValueError(
            f"min_fold_change must be >= 1.0, got {min_fold_change}. "
            "Use linear scale (1.5 = 50% increase)."
        )

    # Get expression matrix
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        mat = adata.layers[layer]
    else:
        mat = adata.X

    gene_names = np.array(adata.var_names)
    groups = sorted(adata.obs[groupby].unique().tolist(), key=str)

    all_results = []
    markers_by_group = {}

    for group in groups:
        mask_in = (adata.obs[groupby] == group).values
        mask_out = ~mask_in
        n_in = int(mask_in.sum())
        n_out = int(mask_out.sum())

        if n_in == 0 or n_out == 0:
            continue

        # Extract expression for group and rest
        if sp.issparse(mat):
            X_in = mat[mask_in]
            X_out = mat[mask_out]
            mean_in = np.asarray(X_in.mean(axis=0)).ravel()
            mean_out = np.asarray(X_out.mean(axis=0)).ravel()
            # Percent expressing (> 0)
            pct_in = np.asarray((X_in > 0).mean(axis=0)).ravel()
            pct_out = np.asarray((X_out > 0).mean(axis=0)).ravel()
        else:
            X_in = np.asarray(mat[mask_in])
            X_out = np.asarray(mat[mask_out])
            mean_in = X_in.mean(axis=0)
            mean_out = X_out.mean(axis=0)
            pct_in = (X_in > 0).mean(axis=0)
            pct_out = (X_out > 0).mean(axis=0)

        # Log2 fold change (pseudocount to avoid division by zero)
        log2fc = np.log2((mean_in + 1e-9) / (mean_out + 1e-9))

        # Pre-filter: min_pct and min_fold_change
        min_log2fc = np.log2(min_fold_change)
        pre_mask = (pct_in >= min_pct) & (log2fc >= min_log2fc)
        candidate_idx = np.where(pre_mask)[0]

        if len(candidate_idx) == 0:
            markers_by_group[str(group)] = pd.DataFrame(
                columns=["gene", "group", "pvalue", "fdr", "log2fc", "pct_in", "pct_out", "score"]
            )
            continue

        # Statistical tests on candidates only
        if method == "wilcoxon":
            pvals = _wilcoxon_test(mat, mask_in, mask_out, candidate_idx, n_in, n_out)
        else:
            pvals = _ttest(mat, mask_in, mask_out, candidate_idx, n_in, n_out)

        # FDR correction
        fdr = _benjamini_hochberg(pvals)

        # Filter by max_pvalue
        sig_mask = fdr <= max_pvalue
        final_idx = candidate_idx[sig_mask]
        final_pvals = pvals[sig_mask]
        final_fdr = fdr[sig_mask]
        final_log2fc = log2fc[final_idx]
        final_pct_in = pct_in[final_idx]
        final_pct_out = pct_out[final_idx]

        # Compute score: -log10(fdr) * log2fc * pct_in
        score = -np.log10(final_fdr + 1e-300) * final_log2fc * final_pct_in

        # Build per-group DataFrame
        group_df = pd.DataFrame(
            {
                "gene": gene_names[final_idx],
                "group": str(group),
                "pvalue": final_pvals,
                "fdr": final_fdr,
                "log2fc": final_log2fc,
                "pct_in": final_pct_in,
                "pct_out": final_pct_out,
                "score": score,
            }
        )

        # Sort by score descending
        group_df = group_df.sort_values("score", ascending=False).reset_index(drop=True)

        # Apply n_top filter
        if n_top is not None:
            group_df = group_df.head(n_top)

        markers_by_group[str(group)] = group_df
        all_results.append(group_df)

    # Combine all groups
    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
    else:
        result_df = pd.DataFrame(
            columns=["gene", "group", "pvalue", "fdr", "log2fc", "pct_in", "pct_out", "score"]
        )

    # Store in adata.uns
    adata.uns["all_markers"] = markers_by_group

    return result_df


def _wilcoxon_test(mat, mask_in, mask_out, candidate_idx, n_in, n_out):
    """Vectorized Wilcoxon rank-sum test for candidate genes."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.stats import norm, rankdata

    n_total = n_in + n_out
    n_candidates = len(candidate_idx)

    # Stack in-group on top
    if sp.issparse(mat):
        combined = sp.vstack([mat[mask_in], mat[mask_out]]).tocsc()
    else:
        combined = np.vstack([mat[mask_in], mat[mask_out]])

    pvals = np.ones(n_candidates, dtype=np.float64)
    # Under H0, E[R] = n_in * (n_total + 1) / 2
    mu_rank = n_in * (n_total + 1) / 2.0
    sigma = np.sqrt(n_in * n_out * (n_total + 1) / 12.0)

    if sigma == 0:
        return pvals

    chunk_size = 500
    for chunk_start in range(0, n_candidates, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_candidates)
        idx_chunk = candidate_idx[chunk_start:chunk_end]

        if sp.issparse(combined):
            chunk_data = combined[:, idx_chunk].toarray()
        else:
            chunk_data = combined[:, idx_chunk]

        # Rank each gene column
        ranks = np.apply_along_axis(rankdata, 0, chunk_data)

        # Sum of ranks for in-group
        rank_sum_in = ranks[:n_in, :].sum(axis=0)

        # Z-score (two-sided)
        z_scores = (rank_sum_in - mu_rank) / sigma
        chunk_pvals = 2.0 * norm.sf(np.abs(z_scores))
        pvals[chunk_start:chunk_end] = chunk_pvals

    return pvals


def _ttest(mat, mask_in, mask_out, candidate_idx, n_in, n_out):
    """Welch's t-test for candidate genes."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.stats import t as t_dist

    n_candidates = len(candidate_idx)
    pvals = np.ones(n_candidates, dtype=np.float64)

    if sp.issparse(mat):
        X_in = mat[mask_in][:, candidate_idx].toarray()
        X_out = mat[mask_out][:, candidate_idx].toarray()
    else:
        X_in = np.asarray(mat[mask_in][:, candidate_idx])
        X_out = np.asarray(mat[mask_out][:, candidate_idx])

    mean_in = X_in.mean(axis=0)
    mean_out = X_out.mean(axis=0)
    var_in = X_in.var(axis=0, ddof=1)
    var_out = X_out.var(axis=0, ddof=1)

    # Welch's t-statistic
    se = np.sqrt(var_in / n_in + var_out / n_out)
    # Avoid division by zero
    nonzero_se = se > 0
    t_stat = np.zeros(n_candidates)
    t_stat[nonzero_se] = (mean_in[nonzero_se] - mean_out[nonzero_se]) / se[nonzero_se]

    # Welch-Satterthwaite degrees of freedom
    numerator = (var_in / n_in + var_out / n_out) ** 2
    denominator = (var_in / n_in) ** 2 / (n_in - 1) + (var_out / n_out) ** 2 / (n_out - 1)
    denominator[denominator == 0] = 1
    df = numerator / denominator
    df = np.clip(df, 1, np.inf)

    # Two-sided p-value
    pvals[nonzero_se] = 2.0 * t_dist.sf(np.abs(t_stat[nonzero_se]), df[nonzero_se])

    return pvals


def _benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR correction."""
    import numpy as np

    n_val = len(pvals)
    if n_val == 0:
        return pvals.copy()

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    adjusted = sorted_pvals * n_val / np.arange(1, n_val + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n_val, dtype=np.float64)
    result[sorted_idx] = adjusted
    return result
