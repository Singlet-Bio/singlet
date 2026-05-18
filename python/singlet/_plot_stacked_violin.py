# SPDX-License-Identifier: MIT
"""Stacked violin plot visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.figure import Figure


def plot_stacked_violin(
    adata: "AnnData",
    var_names: list[str],
    *,
    groupby: str,
    layer: str | None = None,
    scale: str = "width",
    stripplot: bool = False,
    jitter: float | bool = False,
    cmap: str = "tab20",
    figsize: tuple[float, float] | None = None,
    rotation: float = 90,
    show: bool = True,
    save: str | None = None,
) -> "Figure | None":
    """Plot stacked violin plots showing gene expression across groups.

    Creates one row per gene, showing distribution across groups.

    Parameters
    ----------
    adata
        Annotated data matrix.
    var_names
        List of genes to plot.
    groupby
        Key in .obs to group cells by.
    layer
        Layer to use. None uses .X.
    scale
        How to scale violins: 'width' (equal width), 'count', or 'area'.
    stripplot
        Whether to overlay strip/jitter plot.
    jitter
        Jitter amount for strip plot.
    cmap
        Colormap for group colors.
    figsize
        Figure size.
    rotation
        Rotation angle for x-axis labels.
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

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")

    groups = adata.obs[groupby]
    unique_groups = sorted(groups.unique(), key=str)
    n_groups = len(unique_groups)

    # Validate genes exist
    valid_genes = [g for g in var_names if g in adata.var_names]
    if len(valid_genes) == 0:
        raise ValueError("None of the specified genes found in adata.var_names.")

    # Get expression data
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    gene_indices = [list(adata.var_names).index(g) for g in valid_genes]

    # Figure setup
    if figsize is None:
        figsize = (max(4, n_groups * 0.8 + 1), max(3, len(valid_genes) * 1.2))

    fig, axes = plt.subplots(
        len(valid_genes),
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    # Get colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = [
        cmap_obj(i / max(1, n_groups - 1)) if n_groups > 1 else cmap_obj(0.0)
        for i in range(n_groups)
    ]

    for gene_i, (gene, gene_idx) in enumerate(zip(valid_genes, gene_indices)):
        ax = axes[gene_i, 0]

        # Get expression for this gene per group
        data_per_group = []
        for grp in unique_groups:
            mask = (groups == grp).values
            if issparse(X):
                expr = np.asarray(X[mask, gene_idx].todense()).flatten()
            else:
                expr = np.asarray(X[mask, gene_idx]).flatten()
            data_per_group.append(expr)

        # Plot violins
        parts = ax.violinplot(
            data_per_group,
            positions=range(n_groups),
            showmeans=False,
            showmedians=True,
            widths=0.8,
        )

        # Color violins
        for i, body in enumerate(parts.get("bodies", [])):
            body.set_facecolor(colors[i % len(colors)])
            body.set_alpha(0.7)

        # Style median lines
        for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
            if partname in parts:
                parts[partname].set_edgecolor("black")
                parts[partname].set_linewidth(0.5)

        # Strip plot overlay
        if stripplot:
            jitter_amount = jitter if isinstance(jitter, (int, float)) and jitter else 0.1
            rng = np.random.default_rng(42)
            for i, data in enumerate(data_per_group):
                if len(data) > 0:
                    x_jitter = rng.uniform(-jitter_amount, jitter_amount, size=len(data))
                    ax.scatter(
                        i + x_jitter,
                        data,
                        s=1,
                        alpha=0.3,
                        c=[colors[i]],
                        zorder=3,
                    )

        ax.set_ylabel(gene, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])

        if gene_i < len(valid_genes) - 1:
            ax.set_xticks([])

    # Bottom axis labels
    axes[-1, 0].set_xticks(range(n_groups))
    axes[-1, 0].set_xticklabels(
        [str(g) for g in unique_groups],
        rotation=rotation,
        ha="right" if rotation > 45 else "center",
        fontsize=8,
    )

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return fig
