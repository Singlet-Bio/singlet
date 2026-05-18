# SPDX-License-Identifier: MIT
"""Matrix plot for ranked genes per group."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def rank_genes_groups_matrixplot(
    adata: "AnnData",
    *,
    n_genes: int = 5,
    groupby: str | None = None,
    groups: list[str] | None = None,
    var_names: list[str] | None = None,
    standard_scale: str | None = "var",
    cmap: str = "viridis",
    show: bool = True,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> "Axes | None":
    """Plot a matrix (colored grid) of mean expression per group.

    Each cell in the matrix shows the mean expression of a gene
    within a group, with color intensity representing expression level.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_genes
        Number of top DE genes per group.
    groupby
        Key in .obs for grouping.
    groups
        Subset of groups.
    var_names
        Explicit gene list (overrides DE results).
    standard_scale
        Scale across 'var' (genes) or 'group' or None.
    cmap
        Colormap.
    show
        Whether to display.
    save
        Save path.
    figsize
        Figure size.

    Returns
    -------
    Axes or None if show=True.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    # Determine genes
    if var_names is not None:
        genes = [g for g in var_names if g in adata.var_names]
    else:
        if "rank_genes_groups" not in adata.uns:
            raise KeyError(
                "'rank_genes_groups' not found in .uns. "
                "Run singlet.rank_genes_groups() first or provide var_names."
            )
        de_results = adata.uns["rank_genes_groups"]
        names = de_results["names"]

        # Support both dict format (singlet) and structured array (scanpy)
        if isinstance(names, dict):
            de_groups = list(names.keys())
        else:
            de_groups = list(names.dtype.names)

        if groups is not None:
            de_groups = [g for g in de_groups if g in groups]

        genes = []
        for g in de_groups:
            top = list(names[g][:n_genes])
            for gene in top:
                if gene in adata.var_names and gene not in genes:
                    genes.append(gene)

    if len(genes) == 0:
        raise ValueError("No valid genes to plot.")

    # Determine groupby
    if groupby is None:
        if "rank_genes_groups" in adata.uns:
            params = adata.uns["rank_genes_groups"].get("params", {})
            if isinstance(params, dict):
                groupby = params.get("groupby")
        if groupby is None:
            raise ValueError("groupby must be specified.")

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")

    group_labels = adata.obs[groupby]
    if groups is not None:
        unique_groups = [g for g in groups if g in group_labels.values]
    else:
        unique_groups = sorted(group_labels.unique(), key=str)

    # Compute mean expression per group
    gene_idx = [list(adata.var_names).index(g) for g in genes]

    X = adata.X
    mean_expr = np.zeros((len(unique_groups), len(genes)))

    for i, grp in enumerate(unique_groups):
        mask = (group_labels == grp).values
        if issparse(X):
            sub = np.asarray(X[mask][:, gene_idx].todense())
        else:
            sub = np.asarray(X[mask][:, gene_idx])
        mean_expr[i] = sub.mean(axis=0)

    # Standard scale
    if standard_scale == "var":
        col_min = mean_expr.min(axis=0, keepdims=True)
        col_max = mean_expr.max(axis=0, keepdims=True)
        denom = col_max - col_min
        denom[denom == 0] = 1
        mean_expr = (mean_expr - col_min) / denom
    elif standard_scale == "group":
        row_min = mean_expr.min(axis=1, keepdims=True)
        row_max = mean_expr.max(axis=1, keepdims=True)
        denom = row_max - row_min
        denom[denom == 0] = 1
        mean_expr = (mean_expr - row_min) / denom

    # Plot
    if figsize is None:
        figsize = (
            max(4, len(genes) * 0.4 + 2),
            max(3, len(unique_groups) * 0.5 + 1),
        )

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(mean_expr, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(unique_groups)))
    ax.set_yticklabels([str(g) for g in unique_groups], fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return ax
