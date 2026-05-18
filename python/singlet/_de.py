# SPDX-License-Identifier: MIT
"""Differential expression (rank genes by groups) for AnnData objects.

Provides singlet.rank_genes_groups() — identifies marker genes for each
cluster/group using Wilcoxon rank-sum test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd


def rank_genes_groups(
    adata,
    groupby: str,
    *,
    groups: Optional[list[str]] = None,
    n_genes: int = 100,
    method: str = "wilcoxon",
    inplace: bool = True,
) -> Optional[dict]:
    """Find marker genes for each group (differential expression).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (log-normalized recommended).
    groupby : str
        Column in adata.obs to group cells by (e.g., "leiden").
    groups : list[str] or None, default None
        Specific groups to test. If None, tests all groups.
    n_genes : int, default 100
        Number of top genes to report per group.
    method : str, default "wilcoxon"
        Statistical test. Only "wilcoxon" (rank-sum) is supported.
    inplace : bool, default True
        If True, stores results in adata.uns['rank_genes_groups'].
        If False, returns dict with results.

    Returns
    -------
    dict or None
        Dict with keys: 'names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges'
        (each a dict mapping group → array). Or None if inplace=True.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> singlet.rank_genes_groups(adata, "leiden")
    >>> adata.uns['rank_genes_groups']['names']['0'][:5]  # top 5 markers
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"rank_genes_groups() requires an AnnData object, got {type(adata).__name__}"
        )

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.columns")

    if method != "wilcoxon":
        raise ValueError(f"Unsupported method '{method}'. Use 'wilcoxon'.")

    X = adata.X
    all_groups = adata.obs[groupby].unique().tolist()
    if groups is not None:
        test_groups = [g for g in groups if g in all_groups]
    else:
        test_groups = sorted(all_groups, key=str)

    gene_names = np.array(adata.var_names)
    n_genes = min(n_genes, len(gene_names))

    result = {"names": {}, "scores": {}, "pvals": {}, "pvals_adj": {}, "logfoldchanges": {}}

    for group in test_groups:
        mask_in = (adata.obs[groupby] == group).values
        mask_out = ~mask_in
        n_in = int(mask_in.sum())
        n_out = int(mask_out.sum())

        if n_in == 0 or n_out == 0:
            continue

        # Extract group and rest matrices
        if sp.issparse(X):
            X_in = X[mask_in]
            X_out = X[mask_out]
            mean_in = np.asarray(X_in.mean(axis=0)).ravel()
            mean_out = np.asarray(X_out.mean(axis=0)).ravel()
        else:
            X_in = X[mask_in]
            X_out = X[mask_out]
            mean_in = np.asarray(X_in.mean(axis=0)).ravel()
            mean_out = np.asarray(X_out.mean(axis=0)).ravel()

        # Log fold change (add pseudocount to avoid log(0))
        lfc = np.log2((mean_in + 1e-9) / (mean_out + 1e-9))

        # Vectorized Mann-Whitney U via rank sums
        scores, pvals = _vectorized_mannwhitney(X_in, X_out, n_in, n_out)

        # Benjamini-Hochberg FDR correction
        pvals_adj = _benjamini_hochberg(pvals)

        # Sort by score (descending)
        sorted_idx = np.argsort(-scores)[:n_genes]
        result["names"][str(group)] = gene_names[sorted_idx].tolist()
        result["scores"][str(group)] = scores[sorted_idx].tolist()
        result["pvals"][str(group)] = pvals[sorted_idx].tolist()
        result["pvals_adj"][str(group)] = pvals_adj[sorted_idx].tolist()
        result["logfoldchanges"][str(group)] = lfc[sorted_idx].tolist()

    if inplace:
        adata.uns["rank_genes_groups"] = result
        return None
    else:
        return result


def rank_genes_groups_df(
    adata,
    group: Optional[str] = None,
) -> "pd.DataFrame":
    """Get DE results as a tidy DataFrame.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have 'rank_genes_groups' in adata.uns.
    group : str or None, default None
        Specific group to return. If None, returns all groups.

    Returns
    -------
    pandas.DataFrame
        Columns: 'group', 'names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges'.
    """
    import pandas as pd

    if not hasattr(adata, "uns") or "rank_genes_groups" not in adata.uns:
        raise KeyError(
            "adata.uns['rank_genes_groups'] not found. Run singlet.rank_genes_groups() first."
        )

    result = adata.uns["rank_genes_groups"]
    groups_to_show = [group] if group is not None else list(result["names"].keys())

    rows = []
    for grp in groups_to_show:
        if grp not in result["names"]:
            raise KeyError(f"Group '{grp}' not found in rank_genes_groups results.")
        n = len(result["names"][grp])
        for i in range(n):
            row = {"group": grp, "names": result["names"][grp][i]}
            if "scores" in result:
                row["scores"] = result["scores"][grp][i]
            if "pvals" in result:
                row["pvals"] = result["pvals"][grp][i]
            if "pvals_adj" in result:
                row["pvals_adj"] = result["pvals_adj"][grp][i]
            if "logfoldchanges" in result:
                row["logfoldchanges"] = result["logfoldchanges"][grp][i]
            rows.append(row)

    return pd.DataFrame(rows)


def _vectorized_mannwhitney(X_in, X_out, n_in: int, n_out: int):
    """Vectorized Mann-Whitney U test across all genes simultaneously.

    Returns z-scores and p-values arrays of shape (n_genes,).
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.stats import norm, rankdata

    n_total = n_in + n_out
    n_genes = X_in.shape[1]

    # Stack in-group on top of out-group
    if sp.issparse(X_in):
        X_combined = sp.vstack([X_in, X_out]).tocsc()
    else:
        X_combined = np.vstack([X_in, X_out])

    # Compute rank sums for in-group across all genes
    scores = np.zeros(n_genes, dtype=np.float64)
    pvals = np.ones(n_genes, dtype=np.float64)

    # Process genes in chunks for memory efficiency
    chunk_size = 500
    mu = n_in * n_out / 2.0
    sigma = np.sqrt(n_in * n_out * (n_total + 1) / 12.0)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)

        if sp.issparse(X_combined):
            chunk = X_combined[:, start:end].toarray()
        else:
            chunk = X_combined[:, start:end]

        # Rank each column independently
        # rankdata along axis=0 ranks within each column
        ranks = np.apply_along_axis(rankdata, 0, chunk)

        # Sum of ranks for in-group (first n_in rows)
        R_in = ranks[:n_in, :].sum(axis=0)

        # U statistic
        U = R_in - n_in * (n_in + 1) / 2.0

        # Z-scores
        if sigma > 0:
            z = (U - mu) / sigma
            scores[start:end] = z
            pvals[start:end] = 2.0 * norm.sf(np.abs(z))

    return scores, pvals


def _benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values (same length as input).
    """
    import numpy as np

    n = len(pvals)
    if n == 0:
        return pvals.copy()

    # Sort p-values and track original order
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * n / rank
    adjusted = sorted_pvals * n / np.arange(1, n + 1)

    # Enforce monotonicity (cumulative minimum from the end)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(n, dtype=np.float64)
    result[sorted_idx] = adjusted
    return result
