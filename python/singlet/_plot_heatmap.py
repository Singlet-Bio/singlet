"""Heatmap visualization for gene expression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def plot_heatmap(
    adata: "AnnData",
    var_names: list[str],
    *,
    groupby: str | None = None,
    layer: str | None = None,
    standard_scale: str | None = "var",
    cmap: str = "viridis",
    dendrogram: bool = False,
    swap_axes: bool = False,
    show: bool = True,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> "Axes | None":
    """Plot a heatmap of gene expression.

    Parameters
    ----------
    adata
        Annotated data matrix.
    var_names
        List of gene names to show.
    groupby
        Key in .obs to group cells by. Cells are ordered within groups.
    layer
        Layer to use. None uses .X.
    standard_scale
        Standardize across 'var' (genes) or 'obs' (cells) or None.
    cmap
        Colormap name.
    dendrogram
        Whether to show a dendrogram (requires scipy).
    swap_axes
        If True, genes on y-axis and cells on x-axis.
    show
        Whether to display the plot.
    save
        File path to save figure.
    figsize
        Figure size (width, height).
    vmin
        Minimum value for colormap.
    vmax
        Maximum value for colormap.

    Returns
    -------
    Axes or None if show=True.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    # Get expression data for selected genes
    gene_idx = [list(adata.var_names).index(g) for g in var_names if g in adata.var_names]

    if len(gene_idx) == 0:
        raise ValueError("No genes from var_names found in adata.var_names.")

    found_genes = [adata.var_names[i] for i in gene_idx]

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Extract gene expression matrix
    if issparse(X):
        expr = np.asarray(X[:, gene_idx].todense())
    else:
        expr = np.asarray(X[:, gene_idx])

    # Order cells by group
    if groupby is not None and groupby in adata.obs.columns:
        groups = adata.obs[groupby]
        order = np.argsort(groups.values)
        expr = expr[order]
        groups_ordered = groups.iloc[order]
    else:
        groups_ordered = None

    # Standard scale
    if standard_scale == "var":
        col_min = expr.min(axis=0, keepdims=True)
        col_max = expr.max(axis=0, keepdims=True)
        denom = col_max - col_min
        denom[denom == 0] = 1
        expr = (expr - col_min) / denom
    elif standard_scale == "obs":
        row_min = expr.min(axis=1, keepdims=True)
        row_max = expr.max(axis=1, keepdims=True)
        denom = row_max - row_min
        denom[denom == 0] = 1
        expr = (expr - row_min) / denom

    # Figure setup
    if figsize is None:
        n_genes = len(found_genes)
        n_cells = expr.shape[0]
        if swap_axes:
            figsize = (max(6, n_cells * 0.02), max(3, n_genes * 0.3))
        else:
            figsize = (max(4, n_genes * 0.4), max(4, n_cells * 0.01 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    if swap_axes:
        im = ax.imshow(
            expr.T,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(found_genes)))
        ax.set_yticklabels(found_genes, fontsize=8)
        ax.set_xlabel("Cells")
    else:
        im = ax.imshow(
            expr,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(found_genes)))
        ax.set_xticklabels(found_genes, rotation=90, fontsize=8)
        ax.set_ylabel("Cells")

    plt.colorbar(im, ax=ax, shrink=0.6)

    # Add group separators
    if groups_ordered is not None:
        cats = groups_ordered.unique()
        boundaries = []
        for cat in cats:
            mask = (groups_ordered == cat).values
            idx_arr = np.where(mask)[0]
            if len(idx_arr) > 0:
                boundaries.append(idx_arr[-1] + 0.5)

        for b in boundaries[:-1]:
            if swap_axes:
                ax.axvline(x=b, color="white", linewidth=0.5)
            else:
                ax.axhline(y=b, color="white", linewidth=0.5)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return ax
