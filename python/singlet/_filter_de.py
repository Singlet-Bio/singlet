"""Filter DE results."""

from __future__ import annotations

from anndata import AnnData


def filter_rank_genes_groups(
    adata: AnnData,
    *,
    key: str = "rank_genes_groups",
    key_added: str = "rank_genes_groups_filtered",
    min_fold_change: float = 1.0,
    max_pval: float = 0.05,
    min_in_group_fraction: float = 0.0,
    max_out_group_fraction: float = 1.0,
) -> None:
    """Filter DE results by fold change, p-value, and expression fraction.

    Creates a new key in .uns with filtered results (genes not passing
    filters are replaced with empty string).

    Parameters
    ----------
    adata
        Annotated data matrix with DE results.
    key
        Key in .uns for DE results.
    key_added
        Key for storing filtered results.
    min_fold_change
        Minimum absolute log fold change.
    max_pval
        Maximum adjusted p-value.
    min_in_group_fraction
        Minimum fraction of cells expressing the gene within the group.
    max_out_group_fraction
        Maximum fraction of cells expressing outside the group.

    Returns
    -------
    None. Adds filtered results to .uns[key_added].
    """
    import numpy as np
    from scipy.sparse import issparse

    if key not in adata.uns:
        raise KeyError(f"'{key}' not found in .uns.")

    de_results = adata.uns[key]
    groups = list(de_results["names"].keys())

    # Get groupby info from params if available
    params = de_results.get("params", {})
    groupby = params.get("groupby")

    filtered_names: dict[str, list] = {}
    filtered_scores: dict[str, list] = {}

    for group in groups:
        names = de_results["names"][group]
        scores = de_results["scores"][group]

        has_pvals = "pvals_adj" in de_results
        has_lfc = "logfoldchanges" in de_results

        pvals = de_results["pvals_adj"][group] if has_pvals else None
        lfcs = de_results["logfoldchanges"][group] if has_lfc else None

        f_names = []
        f_scores = []

        for i in range(len(names)):
            gene = names[i]
            keep = True

            # p-value filter
            if has_pvals and pvals[i] > max_pval:
                keep = False

            # Fold change filter
            if has_lfc and abs(lfcs[i]) < min_fold_change:
                keep = False

            # Expression fraction filters
            if keep and (min_in_group_fraction > 0 or max_out_group_fraction < 1):
                if groupby and groupby in adata.obs.columns:
                    if gene in adata.var_names:
                        gene_idx = list(adata.var_names).index(gene)
                        group_mask = (adata.obs[groupby] == group).values

                        if issparse(adata.X):
                            expr = np.asarray(adata.X[:, gene_idx].todense()).flatten()
                        else:
                            expr = np.asarray(adata.X[:, gene_idx]).flatten()

                        in_frac = (expr[group_mask] > 0).mean()
                        out_frac = (expr[~group_mask] > 0).mean()

                        if in_frac < min_in_group_fraction:
                            keep = False
                        if out_frac > max_out_group_fraction:
                            keep = False

            if keep:
                f_names.append(gene)
                f_scores.append(scores[i])
            else:
                f_names.append("")
                f_scores.append(0.0)

        filtered_names[group] = f_names
        filtered_scores[group] = f_scores

    adata.uns[key_added] = {
        "names": filtered_names,
        "scores": filtered_scores,
        "params": params,
    }
