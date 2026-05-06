"""Differential expression (rank genes by groups) for AnnData objects.

Provides singlet.rank_genes_groups() — identifies marker genes for each
cluster/group using Wilcoxon rank-sum test.
"""

from __future__ import annotations

from typing import Optional


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
        Dict with keys: 'names', 'scores', 'pvals', 'logfoldchanges'
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
    from scipy.stats import rankdata

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

    result = {"names": {}, "scores": {}, "pvals": {}, "logfoldchanges": {}}

    for group in test_groups:
        mask_in = (adata.obs[groupby] == group).values
        mask_out = ~mask_in
        n_in = mask_in.sum()
        n_out = mask_out.sum()

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
            mean_in = X_in.mean(axis=0)
            mean_out = X_out.mean(axis=0)

        # Log fold change (add pseudocount to avoid log(0))
        lfc = np.log2((mean_in + 1e-9) / (mean_out + 1e-9))

        # Wilcoxon rank-sum (Mann-Whitney U) — vectorized across genes
        scores = np.zeros(len(gene_names), dtype=np.float64)
        pvals = np.ones(len(gene_names), dtype=np.float64)

        for j in range(len(gene_names)):
            if sp.issparse(X):
                x_in = np.asarray(X_in[:, j].todense()).ravel()
                x_out = np.asarray(X_out[:, j].todense()).ravel()
            else:
                x_in = X_in[:, j]
                x_out = X_out[:, j]

            # Skip genes with no expression in either group
            if x_in.sum() == 0 and x_out.sum() == 0:
                continue

            # Mann-Whitney U statistic
            combined = np.concatenate([x_in, x_out])
            ranks = rankdata(combined)
            R_in = ranks[:n_in].sum()
            U = R_in - n_in * (n_in + 1) / 2
            # Normalize to z-score
            mu = n_in * n_out / 2
            sigma = np.sqrt(n_in * n_out * (n_in + n_out + 1) / 12)
            if sigma > 0:
                z = (U - mu) / sigma
                scores[j] = z
                # Two-sided p-value from normal approximation
                from scipy.stats import norm

                pvals[j] = 2 * norm.sf(abs(z))

        # Sort by score (descending)
        sorted_idx = np.argsort(-scores)[:n_genes]
        result["names"][str(group)] = gene_names[sorted_idx].tolist()
        result["scores"][str(group)] = scores[sorted_idx].tolist()
        result["pvals"][str(group)] = pvals[sorted_idx].tolist()
        result["logfoldchanges"][str(group)] = lfc[sorted_idx].tolist()

    if inplace:
        adata.uns["rank_genes_groups"] = result
        return None
    else:
        return result
