# SPDX-License-Identifier: MIT
"""Gene set enrichment / over-representation analysis.

Provides singlet.gene_set_enrichment() — tests whether marker genes for
each cluster are enriched in user-provided gene sets using Fisher's exact test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def gene_set_enrichment(
    adata,
    gene_sets: dict[str, list[str]],
    *,
    method: str = "fisher",
    groupby: str | None = None,
    n_top_genes: int = 200,
) -> "pd.DataFrame":
    """Gene set over-representation analysis on marker genes.

    For each gene set, tests whether marker genes (from DE results or top
    variable genes) are enriched using Fisher's exact test.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should have DE results in
        adata.uns['rank_genes_groups'] if groupby is provided.
    gene_sets : dict[str, list[str]]
        Dictionary mapping gene set names to lists of gene symbols.
    method : str, default 'fisher'
        Statistical test to use. Currently only 'fisher' (one-sided
        Fisher's exact test) is supported.
    groupby : str or None, default None
        If provided, test enrichment per group using DE results stored
        in adata.uns['rank_genes_groups']. If None, uses all variable
        genes or the full gene list as the test set.
    n_top_genes : int, default 200
        Number of top marker genes to use per group (or globally).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: gene_set, group (if groupby), pvalue,
        fdr, odds_ratio, overlap_genes, overlap_size.

    Raises
    ------
    ValueError
        If method is not 'fisher' or gene_sets is empty.
    KeyError
        If groupby is provided but DE results are missing.

    Examples
    --------
    >>> import singlet
    >>> gene_sets = {"apoptosis": ["BAX", "BCL2", "CASP3"],
    ...             "cell_cycle": ["CDK1", "CCNB1", "MKI67"]}
    >>> singlet.gene_set_enrichment(adata, gene_sets, groupby="leiden")
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact

    if method != "fisher":
        raise ValueError(f"Unsupported method '{method}'. Only 'fisher' is supported.")

    if not gene_sets:
        raise ValueError("gene_sets must be a non-empty dictionary.")

    # Get the universe of genes
    universe = set(adata.var_names)

    # Determine marker gene lists per group
    groups_markers: dict[str, list[str]] = {}

    if groupby is not None:
        # Extract markers from rank_genes_groups
        if "rank_genes_groups" not in adata.uns:
            raise KeyError(
                "adata.uns['rank_genes_groups'] not found. "
                "Run singlet.rank_genes_groups(adata, groupby=...) first."
            )
        rgg = adata.uns["rank_genes_groups"]
        rgg_names = rgg["names"]
        # Handle both dict format (singlet) and structured array format (scanpy)
        if isinstance(rgg_names, dict):
            group_names = list(rgg_names.keys())
            for group in group_names:
                gene_list = rgg_names[group][:n_top_genes]
                groups_markers[group] = [
                    g for g in gene_list if isinstance(g, str) and g in universe
                ]
        else:
            group_names = list(rgg_names.dtype.names)
            for group in group_names:
                gene_list = rgg_names[group][:n_top_genes]
                groups_markers[group] = [
                    g for g in gene_list if isinstance(g, str) and g in universe
                ]
    else:
        # Use highly variable genes or top expressed genes as markers
        if "highly_variable" in adata.var.columns:
            hvg = list(adata.var_names[adata.var["highly_variable"]])[:n_top_genes]
        else:
            hvg = list(adata.var_names[:n_top_genes])
        groups_markers["all"] = hvg

    # Run enrichment tests
    results = []
    n_universe = len(universe)

    for group_name, markers in groups_markers.items():
        marker_set = set(markers) & universe
        n_markers = len(marker_set)
        if n_markers == 0:
            continue

        for gs_name, gs_genes in gene_sets.items():
            gs_set = set(gs_genes) & universe
            n_gs = len(gs_set)
            if n_gs == 0:
                continue

            # Overlap
            overlap = marker_set & gs_set
            n_overlap = len(overlap)

            # 2x2 contingency table for Fisher's exact test
            # [[overlap, markers_not_in_gs], [gs_not_in_markers, neither]]
            a_val = n_overlap
            b_val = n_markers - n_overlap
            c_val = n_gs - n_overlap
            d_val = n_universe - n_markers - n_gs + n_overlap

            # One-sided (greater) Fisher's exact test
            table = np.array([[a_val, b_val], [c_val, d_val]])
            _, pvalue = fisher_exact(table, alternative="greater")

            # Odds ratio (with pseudocount to avoid division by zero)
            if b_val == 0 or c_val == 0:
                odds_ratio = float("inf") if a_val > 0 else 0.0
            else:
                odds_ratio = (a_val * d_val) / (b_val * c_val)

            row = {
                "gene_set": gs_name,
                "pvalue": pvalue,
                "odds_ratio": odds_ratio,
                "overlap_genes": ",".join(sorted(overlap)) if overlap else "",
                "overlap_size": n_overlap,
            }
            if groupby is not None:
                row["group"] = group_name
            results.append(row)

    if not results:
        cols = ["gene_set", "pvalue", "fdr", "odds_ratio", "overlap_genes", "overlap_size"]
        if groupby is not None:
            cols.insert(1, "group")
        df = pd.DataFrame(columns=cols)
        adata.uns["gene_set_enrichment"] = df
        return df

    df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests

    _, fdr, _, _ = multipletests(df["pvalue"].values, method="fdr_bh")
    df["fdr"] = fdr

    # Reorder columns
    if groupby is not None:
        col_order = [
            "gene_set",
            "group",
            "pvalue",
            "fdr",
            "odds_ratio",
            "overlap_genes",
            "overlap_size",
        ]
    else:
        col_order = [
            "gene_set",
            "pvalue",
            "fdr",
            "odds_ratio",
            "overlap_genes",
            "overlap_size",
        ]
    df = df[col_order].sort_values("pvalue").reset_index(drop=True)

    # Store in adata
    adata.uns["gene_set_enrichment"] = df

    return df
