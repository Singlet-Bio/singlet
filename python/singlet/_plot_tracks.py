"""Tracks plot for marker genes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.figure import Figure


def rank_genes_groups_tracksplot(
    adata: "AnnData",
    *,
    n_genes: int = 5,
    groupby: str | None = None,
    groups: list[str] | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save: str | None = None,
) -> "Figure | None":
    """Plot tracks of gene expression ordered by group.

    Shows top DE genes per group as horizontal expression tracks,
    with cells ordered by their group assignment.

    Parameters
    ----------
    adata
        Annotated data matrix with DE results.
    n_genes
        Number of top genes per group.
    groupby
        Key in .obs for grouping. Inferred from DE results if None.
    groups
        Subset of groups.
    cmap
        Colormap for expression.
    figsize
        Figure size.
    show
        Whether to display.
    save
        Save path.

    Returns
    -------
    Figure or None if show=True.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    if "rank_genes_groups" not in adata.uns:
        raise KeyError(
            "'rank_genes_groups' not found in .uns. Run singlet.rank_genes_groups() first."
        )

    de_results = adata.uns["rank_genes_groups"]
    names = de_results["names"]
    if hasattr(names, "dtype") and names.dtype.names:
        de_groups = list(names.dtype.names)
    else:
        de_groups = list(names.keys())

    if groupby is None:
        params = de_results.get("params")
        if isinstance(params, dict):
            groupby = params.get("groupby")
        # Try to infer groupby from obs columns matching DE group names
        if groupby is None:
            for col in adata.obs.columns:
                col_vals = set(adata.obs[col].astype(str).unique())
                if set(de_groups).issubset(col_vals):
                    groupby = col
                    break
    if groupby is None or groupby not in adata.obs.columns:
        raise ValueError("groupby must be specified or inferred from DE results.")

    if groups is not None:
        de_groups = [g for g in de_groups if g in groups]

    # Get top genes per group
    all_genes = []
    gene_group_labels = []
    for g in de_groups:
        top = list(names[g][:n_genes])
        for gene in top:
            if gene in adata.var_names and gene not in all_genes:
                all_genes.append(gene)
                gene_group_labels.append(g)

    if len(all_genes) == 0:
        raise ValueError("No genes found for plotting.")

    # Order cells by group
    group_labels = adata.obs[groupby]
    order = []
    group_boundaries = []
    pos = 0
    for g in de_groups:
        mask = (group_labels == g).values
        idx = np.where(mask)[0]
        order.extend(idx.tolist())
        pos += len(idx)
        group_boundaries.append(pos)

    # Add remaining cells not in de_groups
    remaining = set(range(adata.n_obs)) - set(order)
    order.extend(sorted(remaining))

    # Get expression
    gene_idx = [list(adata.var_names).index(g) for g in all_genes]

    if issparse(adata.X):
        expr = np.asarray(adata.X[order][:, gene_idx].todense())
    else:
        expr = np.asarray(adata.X[order][:, gene_idx])

    # Scale per gene to [0, 1]
    col_min = expr.min(axis=0, keepdims=True)
    col_max = expr.max(axis=0, keepdims=True)
    denom = col_max - col_min
    denom[denom == 0] = 1
    expr_scaled = (expr - col_min) / denom

    # Plot
    n_genes_plot = len(all_genes)
    n_cells = expr.shape[0]

    if figsize is None:
        figsize = (max(8, n_cells * 0.01), max(3, n_genes_plot * 0.3))

    fig, axes = plt.subplots(
        n_genes_plot,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    for i, (gene, ax_row) in enumerate(zip(all_genes, axes)):
        ax = ax_row[0]
        ax.imshow(
            expr_scaled[:, i : i + 1].T,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_yticks([0])
        ax.set_yticklabels([gene], fontsize=7, ha="right")
        ax.set_xticks([])

    # Add group boundaries
    for boundary in group_boundaries[:-1]:
        for ax_row in axes:
            ax_row[0].axvline(x=boundary - 0.5, color="white", linewidth=1)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return fig
