# SPDX-License-Identifier: MIT
"""Plotting utilities for single-cell data visualization.

Provides singlet.plot_umap() and singlet.plot_violin() for quick,
publication-ready visualizations of AnnData objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import matplotlib.figure


def plot_umap(
    adata,
    color: Optional[str] = None,
    *,
    layer: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    palette: Optional[Union[str, list]] = None,
    size: float = 5.0,
    alpha: float = 0.8,
    figsize: tuple = (6, 5),
    legend_loc: str = "right margin",
    save: Optional[str] = None,
    show: bool = True,
    ax=None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot UMAP embedding colored by obs column or gene expression.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have 'X_umap' in obsm.
    color : str or None, default None
        Column in adata.obs (categorical/continuous) or gene name.
        If None, plots cells in gray.
    layer : str or None, default None
        Layer to use for gene expression values. If None, uses adata.X.
    title : str or None, default None
        Plot title. Defaults to `color` if provided.
    cmap : str, default "viridis"
        Colormap for continuous values.
    palette : str or list or None, default None
        Color palette for categorical values (matplotlib colormap name
        or list of colors). If None, uses "tab20".
    size : float, default 5.0
        Point size.
    alpha : float, default 0.8
        Point transparency.
    figsize : tuple, default (6, 5)
        Figure size in inches.
    legend_loc : str, default "right margin"
        Legend location. "right margin" places legend outside plot.
        Also accepts standard matplotlib positions.
    save : str or None, default None
        Path to save figure. If None, does not save.
    show : bool, default True
        Whether to display the plot (calls plt.show()).
    ax : matplotlib.axes.Axes or None, default None
        Pre-existing axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.umap(adata)
    >>> singlet.plot_umap(adata, color="leiden")
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not hasattr(adata, "obsm"):
        raise TypeError(f"plot_umap() requires an AnnData object, got {type(adata).__name__}")

    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] not found. Run singlet.umap() first.")

    coords = np.array(adata.obsm["X_umap"])
    if coords.shape[1] < 2:
        raise ValueError("X_umap must have at least 2 dimensions.")

    # Create figure/axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    x, y = coords[:, 0], coords[:, 1]

    if color is None:
        ax.scatter(x, y, s=size, alpha=alpha, c="lightgray", edgecolors="none")
    elif color in adata.obs.columns:
        values = adata.obs[color]
        if hasattr(values, "cat") or values.dtype == object:
            _plot_categorical(ax, x, y, values, size, alpha, palette, legend_loc)
        else:
            sc = ax.scatter(x, y, s=size, alpha=alpha, c=values, cmap=cmap, edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.8, label=color)
    elif color in adata.var_names:
        # Gene expression
        gene_idx = list(adata.var_names).index(color)
        import scipy.sparse as sp

        if layer is not None and layer in adata.layers:
            X_source = adata.layers[layer]
        else:
            X_source = adata.X
        if sp.issparse(X_source):
            values = np.asarray(X_source[:, gene_idx].todense()).ravel()
        else:
            values = X_source[:, gene_idx].ravel()
        sc = ax.scatter(x, y, s=size, alpha=alpha, c=values, cmap=cmap, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.8, label=color)
    else:
        raise KeyError(f"'{color}' not found in adata.obs or adata.var_names")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title if title is not None else (color or "UMAP"))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def _plot_categorical(ax, x, y, values, size, alpha, palette, legend_loc):
    """Plot categorical values with distinct colors."""
    import matplotlib.pyplot as plt
    import numpy as np

    categories = values.unique()
    if hasattr(categories, "categories"):
        categories = categories.categories.tolist()
    else:
        categories = sorted(set(values))

    n_cats = len(categories)

    # Get colors
    if palette is None:
        cmap_name = "tab20" if n_cats <= 20 else "tab20b"
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / max(n_cats - 1, 1)) for i in range(n_cats)]
    elif isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        colors = [cmap(i / max(n_cats - 1, 1)) for i in range(n_cats)]
    else:
        colors = list(palette)

    for i, cat in enumerate(categories):
        mask = np.array(values == cat)
        color = colors[i % len(colors)]
        ax.scatter(
            x[mask], y[mask], s=size, alpha=alpha, c=[color], label=str(cat), edgecolors="none"
        )

    if legend_loc == "right margin":
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=2, fontsize=8
        )
    else:
        ax.legend(loc=legend_loc, frameon=False, markerscale=2, fontsize=8)


def plot_violin(
    adata,
    keys: Union[str, list[str]],
    *,
    groupby: Optional[str] = None,
    layer: Optional[str] = None,
    log: bool = False,
    figsize: Optional[tuple] = None,
    rotation: float = 0,
    save: Optional[str] = None,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """Violin plot of gene expression or obs values across groups.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    keys : str or list[str]
        Gene names or adata.obs column names to plot.
    groupby : str or None, default None
        Column in adata.obs to group by on x-axis.
        If None, shows one violin per key across all cells.
    layer : str or None, default None
        Layer for gene expression. If None, uses adata.X.
    log : bool, default False
        Whether to log1p-transform values before plotting.
    figsize : tuple or None, default None
        Figure size. If None, auto-computed.
    rotation : float, default 0
        X-axis label rotation in degrees.
    save : str or None, default None
        Path to save figure.
    show : bool, default True
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure if show=False, otherwise None.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.plot_violin(adata, ["n_genes", "total_counts"], groupby="leiden")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "obs"):
        raise TypeError(f"plot_violin() requires an AnnData object, got {type(adata).__name__}")

    if isinstance(keys, str):
        keys = [keys]

    n_keys = len(keys)
    if figsize is None:
        figsize = (4 * n_keys, 4)

    fig, axes = plt.subplots(1, n_keys, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for idx, key in enumerate(keys):
        ax = axes[idx]

        # Get values
        if key in adata.obs.columns:
            all_values = np.array(adata.obs[key], dtype=np.float64)
        elif key in adata.var_names:
            gene_idx = list(adata.var_names).index(key)
            if layer is not None and layer in adata.layers:
                X_source = adata.layers[layer]
            else:
                X_source = adata.X
            if sp.issparse(X_source):
                all_values = np.asarray(X_source[:, gene_idx].todense()).ravel()
            else:
                all_values = np.array(X_source[:, gene_idx]).ravel()
        else:
            raise KeyError(f"'{key}' not found in adata.obs or adata.var_names")

        if log:
            all_values = np.log1p(all_values)

        if groupby is None:
            ax.violinplot([all_values], showmedians=True)
            ax.set_xticks([1])
            ax.set_xticklabels(["all"])
        else:
            if groupby not in adata.obs.columns:
                raise KeyError(f"'{groupby}' not found in adata.obs.columns")

            groups = adata.obs[groupby]
            categories = sorted(groups.unique(), key=str)
            data = [all_values[np.array(groups == cat)] for cat in categories]
            ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(categories) + 1))
            ax.set_xticklabels([str(c) for c in categories], rotation=rotation)
            ax.set_xlabel(groupby)

        ax.set_ylabel(key)
        ax.set_title(key)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_dotplot(
    adata,
    var_names: Union[list[str], dict[str, list[str]]],
    groupby: str,
    *,
    layer: Optional[str] = None,
    cmap: str = "Reds",
    figsize: Optional[tuple] = None,
    dendrogram: bool = False,
    save: Optional[str] = None,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """Dot plot showing gene expression by group.

    Dot size = fraction of cells expressing the gene.
    Dot color = mean expression among expressing cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    var_names : list[str] or dict[str, list[str]]
        Genes to plot. If dict, keys are category labels.
    groupby : str
        Column in adata.obs to group cells by.
    layer : str or None, default None
        Layer for expression values.
    cmap : str, default "Reds"
        Colormap for mean expression.
    figsize : tuple or None, default None
        Figure size. Auto-computed if None.
    dendrogram : bool, default False
        If True, order groups by dendrogram (requires prior dendrogram call).
    save : str or None, default None
        Path to save figure.
    show : bool, default True
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure if show=False.

    Examples
    --------
    >>> import singlet
    >>> singlet.plot_dotplot(adata, ["CD3D", "MS4A1", "NKG7"], groupby="leiden")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "obs"):
        raise TypeError(f"plot_dotplot() requires an AnnData object, got {type(adata).__name__}")

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.columns")

    # Flatten gene list
    if isinstance(var_names, dict):
        genes = []
        for gene_group in var_names.values():
            genes.extend(gene_group)
    else:
        genes = list(var_names)

    # Filter to available genes
    available = [g for g in genes if g in adata.var_names]
    if not available:
        raise ValueError("None of the specified genes are in adata.var_names")

    # Get groups order
    groups = adata.obs[groupby]
    if dendrogram and f"dendrogram_{groupby}" in adata.uns:
        categories = adata.uns[f"dendrogram_{groupby}"]["categories_ordered"]
    else:
        categories = sorted(groups.unique(), key=str)

    n_groups = len(categories)
    n_genes = len(available)

    # Compute fraction expressing and mean expression
    frac_expr = np.zeros((n_groups, n_genes), dtype=np.float64)
    mean_expr = np.zeros((n_groups, n_genes), dtype=np.float64)

    if layer is not None and layer in adata.layers:
        X_source = adata.layers[layer]
    else:
        X_source = adata.X

    for gi, gene in enumerate(available):
        gene_idx = list(adata.var_names).index(gene)
        if sp.issparse(X_source):
            col = np.asarray(X_source[:, gene_idx].todense()).ravel()
        else:
            col = np.array(X_source[:, gene_idx]).ravel()

        for ci, cat in enumerate(categories):
            mask = np.array(groups == cat)
            n_cells = mask.sum()
            if n_cells == 0:
                continue
            values = col[mask]
            expressing = values > 0
            frac_expr[ci, gi] = expressing.sum() / n_cells
            if expressing.any():
                mean_expr[ci, gi] = values[expressing].mean()

    # Plot
    if figsize is None:
        figsize = (max(4, n_genes * 0.6 + 2), max(3, n_groups * 0.5 + 1))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    max_size = 200
    for ci in range(n_groups):
        for gi in range(n_genes):
            size = frac_expr[ci, gi] * max_size
            if size > 0:
                ax.scatter(
                    gi,
                    ci,
                    s=size,
                    c=[mean_expr[ci, gi]],
                    cmap=cmap,
                    vmin=0,
                    vmax=mean_expr.max() if mean_expr.max() > 0 else 1,
                    edgecolors="gray",
                    linewidths=0.5,
                )

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(available, rotation=90)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([str(c) for c in categories])
    ax.set_xlabel("Genes")
    ax.set_ylabel(groupby)

    for frac in [0.25, 0.5, 0.75, 1.0]:
        ax.scatter([], [], s=frac * max_size, c="gray", alpha=0.5, label=f"{int(frac * 100)}%")
    ax.legend(
        title="% expressing",
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig
