# SPDX-License-Identifier: MIT
"""Marker gene dot plot visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def rank_genes_groups_dotplot(
    adata: "AnnData",
    *,
    n_genes: int = 5,
    groupby: str | None = None,
    groups: list[str] | None = None,
    var_names: list[str] | None = None,
    layer: str | None = None,
    standard_scale: str | None = "var",
    cmap: str = "Reds",
    size_title: str = "Fraction\nexpressing",
    color_title: str = "Mean expression",
    show: bool = True,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> "Axes | None":
    """Plot a dot plot of top DE marker genes per group.

    Requires rank_genes_groups() to have been run first, unless var_names is provided.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_genes
        Number of top genes per group to show.
    groupby
        Key in .obs used for grouping. If None, inferred from DE results.
    groups
        Subset of groups to show.
    var_names
        Explicit list of genes to show (overrides n_genes).
    layer
        Layer to use for expression values. None uses .X.
    standard_scale
        Scale across 'var' (genes) or None.
    cmap
        Colormap for mean expression.
    size_title
        Title for size legend.
    color_title
        Title for color legend.
    show
        Whether to display the plot.
    save
        File path to save figure.
    figsize
        Figure size (width, height).

    Returns
    -------
    Axes or None if show=True.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    # Get genes to plot
    if var_names is not None:
        genes_to_plot = list(var_names)
    else:
        if "rank_genes_groups" not in adata.uns:
            raise KeyError(
                "'rank_genes_groups' not found in .uns. "
                "Run singlet.rank_genes_groups() first or provide var_names."
            )
        de_results = adata.uns["rank_genes_groups"]
        names = de_results["names"]

        # Support both dict-of-lists and structured numpy array formats
        if isinstance(names, dict):
            de_groups = list(names.keys())
        else:
            de_groups = list(names.dtype.names)

        if groups is not None:
            de_groups = [g for g in de_groups if g in groups]

        genes_to_plot = []
        for g in de_groups:
            top = list(names[g][:n_genes])
            genes_to_plot.extend(top)
        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_genes: list[str] = []
        for g in genes_to_plot:
            if g not in seen:
                seen.add(g)
                unique_genes.append(g)
        genes_to_plot = unique_genes

    # Determine groupby
    if groupby is None:
        if "rank_genes_groups" in adata.uns:
            params = adata.uns["rank_genes_groups"].get("params")
            if isinstance(params, dict):
                groupby = params.get("groupby")
        if groupby is None:
            # Try to infer from obs columns matching DE group names
            de_names = adata.uns.get("rank_genes_groups", {}).get("names")
            if isinstance(de_names, dict):
                de_group_set = set(de_names.keys())
                for col in adata.obs.columns:
                    if set(adata.obs[col].unique()).issuperset(de_group_set):
                        groupby = col
                        break
        if groupby is None:
            raise ValueError("groupby must be specified or inferred from DE results.")

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs columns.")

    # Get expression matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Filter to genes present in var_names
    var_names_list = list(adata.var_names)
    gene_idx = [var_names_list.index(g) for g in genes_to_plot if g in var_names_list]
    found_genes = [adata.var_names[i] for i in gene_idx]

    if len(found_genes) == 0:
        raise ValueError("None of the specified genes found in adata.")

    # Compute mean expression and fraction expressing per group
    group_labels = adata.obs[groupby]
    if groups is not None:
        unique_groups = [g for g in groups if g in group_labels.values]
    else:
        unique_groups = sorted(group_labels.unique(), key=str)

    mean_expr = np.zeros((len(unique_groups), len(found_genes)))
    frac_expr = np.zeros((len(unique_groups), len(found_genes)))

    for i, grp in enumerate(unique_groups):
        mask = (group_labels == grp).values
        if issparse(X):
            sub = X[mask][:, gene_idx]
            sub_dense = np.asarray(sub.todense())
        else:
            sub_dense = np.asarray(X[mask][:, gene_idx])

        mean_expr[i] = sub_dense.mean(axis=0)
        frac_expr[i] = (sub_dense > 0).mean(axis=0)

    # Standard scale mean expression
    if standard_scale == "var":
        col_min = mean_expr.min(axis=0, keepdims=True)
        col_max = mean_expr.max(axis=0, keepdims=True)
        denom = col_max - col_min
        denom[denom == 0] = 1
        mean_expr_scaled = (mean_expr - col_min) / denom
    else:
        mean_expr_scaled = mean_expr

    # Plot
    n_groups = len(unique_groups)
    n_genes_plot = len(found_genes)

    if figsize is None:
        figsize = (max(5, n_genes_plot * 0.5 + 2), max(3, n_groups * 0.5 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Create dot plot
    for i in range(n_groups):
        for j in range(n_genes_plot):
            size = frac_expr[i, j] * 200
            color_val = mean_expr_scaled[i, j]
            ax.scatter(
                j,
                i,
                s=size,
                c=[color_val],
                cmap=cmap,
                vmin=0,
                vmax=1 if standard_scale == "var" else None,
                edgecolors="grey",
                linewidths=0.5,
            )

    ax.set_xticks(range(n_genes_plot))
    ax.set_xticklabels(found_genes, rotation=90, fontsize=8)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([str(g) for g in unique_groups], fontsize=9)

    ax.set_xlim(-0.5, n_genes_plot - 0.5)
    ax.set_ylim(-0.5, n_groups - 0.5)
    ax.invert_yaxis()

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return ax
