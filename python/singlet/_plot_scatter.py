"""Generic scatter plot for any 2D embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def plot_scatter(
    adata: "AnnData",
    *,
    basis: str = "umap",
    color: str | None = None,
    layer: str | None = None,
    size: float | None = None,
    alpha: float = 0.8,
    title: str | None = None,
    cmap: str = "viridis",
    palette: str | None = None,
    legend_loc: str = "right margin",
    ax: "Axes | None" = None,
    show: bool = True,
    save: str | None = None,
) -> "Axes | None":
    """Plot a 2D scatter of any embedding stored in .obsm.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Key for the embedding in .obsm. Will look for 'X_{basis}'.
        E.g., 'umap', 'tsne', 'pca', 'diffmap'.
    color
        Key in .obs or gene name for coloring points.
    layer
        Layer to use for gene expression values. None uses .X.
    size
        Point size. Default: auto-scaled based on n_obs.
    alpha
        Point transparency.
    title
        Plot title. Default: the color key name.
    cmap
        Colormap for continuous data.
    palette
        Color palette for categorical data (e.g., 'tab20', 'Set2').
    legend_loc
        Location of legend for categorical data.
    ax
        Pre-existing matplotlib axes.
    show
        Whether to call plt.show().
    save
        File path to save figure.

    Returns
    -------
    Axes object if show=False, None otherwise.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get embedding coordinates
    obsm_key = f"X_{basis}"
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in .obsm. "
            f"Compute the embedding first (e.g., singlet.{basis}())."
        )

    coords = adata.obsm[obsm_key][:, :2]

    # Auto-scale point size
    if size is None:
        n = coords.shape[0]
        size = max(1, min(50, 120000 / n))

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    if color is None:
        # No coloring — just plot grey points
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=size,
            alpha=alpha,
            c="steelblue",
            edgecolors="none",
        )
    elif color in adata.obs.columns:
        values = adata.obs[color]
        if hasattr(values, "cat") or values.dtype == object:
            # Categorical
            _plot_categorical(ax, coords, values, size, alpha, palette, legend_loc)
        else:
            # Continuous obs column
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                s=size,
                alpha=alpha,
                cmap=cmap,
                edgecolors="none",
            )
            plt.colorbar(sc, ax=ax, shrink=0.8)
    elif color in adata.var_names:
        # Gene expression
        from scipy.sparse import issparse

        if layer is not None:
            X = adata.layers[layer]
        else:
            X = adata.X

        idx = list(adata.var_names).index(color)
        expr = (
            np.asarray(X[:, idx].todense()).flatten()
            if issparse(X)
            else np.asarray(X[:, idx]).flatten()
        )

        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=expr,
            s=size,
            alpha=alpha,
            cmap=cmap,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
    else:
        raise KeyError(f"'{color}' not found in .obs columns or .var_names.")

    # Labels and title
    ax.set_xlabel(f"{basis.upper()} 1")
    ax.set_ylabel(f"{basis.upper()} 2")
    if title is not None:
        ax.set_title(title)
    elif color is not None:
        ax.set_title(color)

    ax.set_xticks([])
    ax.set_yticks([])

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return ax


def _plot_categorical(ax, coords, values, size, alpha, palette, legend_loc):
    """Helper to plot categorical coloring."""
    import matplotlib.pyplot as plt

    cats = values.cat.categories if hasattr(values, "cat") else sorted(set(values))
    n_cats = len(cats)

    # Get colors
    if palette is not None:
        cmap = plt.get_cmap(palette)
    else:
        cmap = plt.get_cmap("tab20" if n_cats <= 20 else "tab20b")

    colors = [cmap(i / max(1, n_cats - 1)) if n_cats > 1 else cmap(0.0) for i in range(n_cats)]

    for i, cat in enumerate(cats):
        mask = values == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            c=[colors[i]],
            label=str(cat),
            edgecolors="none",
        )

    if legend_loc == "right margin":
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=8,
            markerscale=0.8,
        )
    elif legend_loc != "none":
        ax.legend(loc=legend_loc, frameon=False, fontsize=8)
