# SPDX-License-Identifier: MIT
"""Gene regulatory network inference for AnnData objects.

Provides singlet.infer_grn() — infers TF-target regulatory links from
expression correlation or mutual information.
"""

from __future__ import annotations


def infer_grn(
    adata,
    *,
    tf_list: list[str] | None = None,
    target_genes: list[str] | None = None,
    method: str = "correlation",
    n_top: int = 500,
):
    """Infer gene regulatory network from expression data.

    Computes regulatory links between transcription factors and target genes
    using correlation or mutual information.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    tf_list : list[str] or None, default None
        List of transcription factor gene names. If None, uses all genes
        as potential regulators.
    target_genes : list[str] or None, default None
        List of target gene names. If None, uses highly variable genes
        (if available in adata.var['highly_variable']), else all genes.
    method : str, default 'correlation'
        Method for inferring regulatory links:
        - 'correlation': Pearson correlation with t-test significance.
        - 'mutual_info': Discretized mutual information.
    n_top : int, default 500
        Number of top regulatory links to return.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: tf, target, weight, pvalue, fdr.
        Sorted by absolute weight descending.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> grn = singlet.infer_grn(adata, method='correlation', n_top=200)
    >>> grn.head()
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.stats import t as t_dist

    # Validate inputs
    if method not in ("correlation", "mutual_info"):
        raise ValueError(f"method must be 'correlation' or 'mutual_info', got '{method}'")

    if n_top < 1:
        raise ValueError(f"n_top must be >= 1, got {n_top}")

    gene_names = list(adata.var_names)
    gene_set = set(gene_names)

    # Determine TF list
    if tf_list is not None:
        tf_list = [g for g in tf_list if g in gene_set]
        if len(tf_list) == 0:
            raise ValueError("No genes from tf_list found in adata.var_names.")
    else:
        tf_list = gene_names

    # Determine target genes
    if target_genes is not None:
        target_genes = [g for g in target_genes if g in gene_set]
        if len(target_genes) == 0:
            raise ValueError("No genes from target_genes found in adata.var_names.")
    elif "highly_variable" in adata.var.columns:
        hv_mask = adata.var["highly_variable"].values.astype(bool)
        target_genes = [g for g, hv in zip(gene_names, hv_mask) if hv]
        if len(target_genes) == 0:
            target_genes = gene_names
    else:
        target_genes = gene_names

    # Get gene indices
    gene_to_idx = {g: idx for idx, g in enumerate(gene_names)}
    tf_indices = np.array([gene_to_idx[g] for g in tf_list])
    target_indices = np.array([gene_to_idx[g] for g in target_genes])

    # Extract expression matrices
    X = adata.X
    if sp.issparse(X):
        tf_expr = np.asarray(X[:, tf_indices].todense())
        target_expr = np.asarray(X[:, target_indices].todense())
    else:
        tf_expr = np.asarray(X[:, tf_indices])
        target_expr = np.asarray(X[:, target_indices])

    n_cells = tf_expr.shape[0]
    n_tfs = len(tf_list)
    n_targets = len(target_genes)

    if method == "correlation":
        # Pearson correlation between TFs and targets
        # Standardize
        tf_mean = tf_expr.mean(axis=0, keepdims=True)
        tf_std = tf_expr.std(axis=0, keepdims=True) + 1e-10
        tf_norm = (tf_expr - tf_mean) / tf_std

        tgt_mean = target_expr.mean(axis=0, keepdims=True)
        tgt_std = target_expr.std(axis=0, keepdims=True) + 1e-10
        tgt_norm = (target_expr - tgt_mean) / tgt_std

        # Correlation matrix: (n_tfs, n_targets)
        corr_matrix = (tf_norm.T @ tgt_norm) / n_cells

        # T-statistic for correlation significance
        # t = r * sqrt(n-2) / sqrt(1 - r^2)
        r_sq = corr_matrix**2
        r_sq = np.clip(r_sq, 0, 0.9999)
        t_stats = corr_matrix * np.sqrt(n_cells - 2) / np.sqrt(1 - r_sq)

        # Two-tailed p-value
        pval_matrix = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n_cells - 2))

        weight_matrix = corr_matrix

    else:
        # Mutual information (discretized)
        n_disc_bins = min(10, max(3, int(np.sqrt(n_cells / 5))))

        def _discretize(arr, num_bins):
            """Discretize continuous values into bins."""
            percentiles = np.linspace(0, 100, num_bins + 1)
            edges = np.percentile(arr, percentiles)
            edges[-1] += 1e-10
            return np.digitize(arr, edges[1:-1])

        weight_matrix = np.zeros((n_tfs, n_targets), dtype=np.float64)
        pval_matrix = np.ones((n_tfs, n_targets), dtype=np.float64)

        for tf_idx in range(n_tfs):
            tf_disc = _discretize(tf_expr[:, tf_idx], n_disc_bins)
            for tgt_idx in range(n_targets):
                tgt_disc = _discretize(target_expr[:, tgt_idx], n_disc_bins)

                # Compute MI
                # Joint distribution
                joint = np.zeros((n_disc_bins, n_disc_bins), dtype=np.float64)
                for cell_idx in range(n_cells):
                    ti = min(tf_disc[cell_idx], n_disc_bins - 1)
                    tj = min(tgt_disc[cell_idx], n_disc_bins - 1)
                    joint[ti, tj] += 1
                joint /= n_cells

                # Marginals
                px = joint.sum(axis=1)
                py = joint.sum(axis=0)

                # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
                mi_val = 0.0
                for xi in range(n_disc_bins):
                    for yi in range(n_disc_bins):
                        if joint[xi, yi] > 1e-10 and px[xi] > 1e-10 and py[yi] > 1e-10:
                            mi_val += joint[xi, yi] * np.log(joint[xi, yi] / (px[xi] * py[yi]))

                weight_matrix[tf_idx, tgt_idx] = mi_val

                # Approximate p-value: chi-square approximation
                # 2*N*MI ~ chi2(df=(bins-1)^2)
                from scipy.stats import chi2

                chi2_stat = 2 * n_cells * mi_val
                df_mi = (n_disc_bins - 1) ** 2
                pval_matrix[tf_idx, tgt_idx] = 1 - chi2.cdf(chi2_stat, df_mi)

    # Flatten and collect results (exclude self-links)
    results = []
    for tf_idx in range(n_tfs):
        for tgt_idx in range(n_targets):
            tf_name = tf_list[tf_idx]
            tgt_name = target_genes[tgt_idx]
            if tf_name == tgt_name:
                continue
            results.append(
                (
                    tf_name,
                    tgt_name,
                    weight_matrix[tf_idx, tgt_idx],
                    pval_matrix[tf_idx, tgt_idx],
                )
            )

    if not results:
        result_df = pd.DataFrame(columns=["tf", "target", "weight", "pvalue", "fdr"])
        adata.uns["grn"] = result_df
        return result_df

    result_df = pd.DataFrame(results, columns=["tf", "target", "weight", "pvalue"])

    # Sort by absolute weight, take top candidates for FDR
    result_df["abs_weight"] = result_df["weight"].abs()
    result_df = result_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)

    # Keep top n_top * 2 for FDR computation efficiency, then trim
    n_keep = min(len(result_df), n_top * 2)
    result_df = result_df.head(n_keep).reset_index(drop=True)

    # BH FDR correction
    n_tested = len(result_df)
    pvals = result_df["pvalue"].values
    sorted_idx = np.argsort(pvals)
    fdr = np.zeros(n_tested, dtype=np.float64)
    for rank, idx in enumerate(sorted_idx, 1):
        fdr[idx] = pvals[idx] * n_tested / rank
    # Ensure monotonicity
    for rank_idx in range(n_tested - 2, -1, -1):
        idx = sorted_idx[rank_idx]
        next_idx = sorted_idx[rank_idx + 1]
        if fdr[idx] > fdr[next_idx]:
            fdr[idx] = fdr[next_idx]
    fdr = np.clip(fdr, 0, 1)
    result_df["fdr"] = fdr

    # Final sort and trim
    result_df = result_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    result_df = result_df.head(n_top).reset_index(drop=True)
    result_df = result_df.drop(columns=["abs_weight"])

    # Store in adata.uns
    adata.uns["grn"] = result_df

    return result_df
