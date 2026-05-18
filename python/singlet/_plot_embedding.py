# SPDX-License-Identifier: MIT
"""Unified multi-panel embedding plot."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def plot_embedding(
    adata: "AnnData",
    *,
    basis: str = "X_umap",
    color: Optional[Union[str, list[str]]] = None,
    ncols: int = 4,
    size: Optional[float] = None,
    frameon: bool = True,
    title: Optional[Union[str, list[str]]] = None,
    cmap: str = "viridis",
    palette: Optional[Union[str, list]] = None,
    alpha: float = 0.8,
    figsize_single: tuple[float, float] = (4, 3.5),
    legend_loc: str = "right margin",
    save: Optional[str] = None,
    show: bool = False,
) -> tuple:
    """Plot 2D embeddings with one or more color overlays.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Key in .obsm for embedding coordinates.
    color
        Key(s) in .obs or gene name(s) for coloring panels.
    ncols
        Maximum columns in the grid.
    size
        Point size. Auto-computed if None.
    frameon
        Whether to draw axis frames.
    title
        Title(s) for panels.
    cmap
        Colormap for continuous data.
    palette
        Color palette for categorical data.
    alpha
        Point transparency.
    figsize_single
        Size of each individual panel (width, height).
    legend_loc
        Legend location for categorical panels.
    save
        File path to save figure.
    show
        Whether to call plt.show().

    Returns
    -------
    Tuple of (fig, axes).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    # Validate basis
    if basis not in adata.obsm:
        raise ValueError(
            f"'{basis}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )

    coords = adata.obsm[basis][:, :2]
    n_cells = coords.shape[0]

    # Auto-compute point size
    if size is None:
        size = min(120000 / n_cells, 20)

    # Normalize color to a list
    if color is None:
        color_keys: list[str] = []
    elif isinstance(color, str):
        color_keys = [color]
    else:
        color_keys = list(color)

    # Determine number of panels
    n_panels = max(len(color_keys), 1)
    actual_ncols = min(ncols, n_panels)
    nrows = ceil(n_panels / actual_ncols)

    fig, axes = plt.subplots(
        nrows,
        actual_ncols,
        figsize=(
            figsize_single[0] * actual_ncols,
            figsize_single[1] * nrows,
        ),
        squeeze=False,
    )

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Hide unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if not color_keys:
        # Single panel, all gray
        ax = axes_flat[0]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=size,
            alpha=alpha,
            c="lightgray",
            edgecolors="none",
        )
        _set_axis_labels(ax, basis)
        if title:
            panel_title = title if isinstance(title, str) else title[0]
            ax.set_title(panel_title)
        if not frameon:
            ax.axis("off")
    else:
        # Validate all color keys first
        for key in color_keys:
            if key not in adata.obs.columns and key not in adata.var_names:
                raise ValueError(f"'{key}' not found in adata.obs or adata.var_names")

        for panel_idx, key in enumerate(color_keys):
            ax = axes_flat[panel_idx]

            if key in adata.obs.columns:
                values = adata.obs[key]
                if hasattr(values, "cat") or values.dtype == object:
                    _plot_categorical_panel(
                        ax,
                        coords,
                        values,
                        size,
                        alpha,
                        palette,
                        legend_loc,
                    )
                else:
                    sc = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=values,
                        s=size,
                        alpha=alpha,
                        cmap=cmap,
                        edgecolors="none",
                    )
                    fig.colorbar(sc, ax=ax, shrink=0.8)
            else:
                # Gene expression
                expr = adata[:, key].X
                if issparse(expr):
                    expr = np.asarray(expr.todense()).flatten()
                else:
                    expr = np.asarray(expr).flatten()

                sc = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=expr,
                    s=size,
                    alpha=alpha,
                    cmap=cmap,
                    edgecolors="none",
                )
                fig.colorbar(sc, ax=ax, shrink=0.8)

            # Set title
            if title is not None:
                if isinstance(title, str):
                    panel_title = title
                elif panel_idx < len(title):
                    panel_title = title[panel_idx]
                else:
                    panel_title = key
            else:
                panel_title = key
            ax.set_title(panel_title)

            _set_axis_labels(ax, basis)

            if not frameon:
                ax.axis("off")

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)

    # Return consistent shape
    if n_panels == 1:
        return (fig, axes_flat[0])
    return (fig, axes)


def _set_axis_labels(ax, basis: str) -> None:
    """Set axis labels based on the basis name."""
    label = basis.replace("X_", "").upper()
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")


def _plot_categorical_panel(ax, coords, values, size, alpha, palette, legend_loc) -> None:
    """Plot a categorical variable on the given axes."""

    categories = (
        values.cat.categories.tolist() if hasattr(values, "cat") else sorted(values.unique())
    )
    n_categories = len(categories)

    # Get colors
    import matplotlib as _mpl

    if palette is None:
        if n_categories <= 10:
            cmap_cat = _mpl.colormaps.get_cmap("tab10")
        else:
            cmap_cat = _mpl.colormaps.get_cmap("tab20")
        colors = [cmap_cat(cat_idx % 20) for cat_idx in range(n_categories)]
    elif isinstance(palette, str):
        cmap_cat = _mpl.colormaps.get_cmap(palette)
        colors = [cmap_cat(cat_idx / max(n_categories - 1, 1)) for cat_idx in range(n_categories)]
    else:
        colors = palette[:n_categories]

    for cat_idx, cat in enumerate(categories):
        mask = values == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            c=[colors[cat_idx]],
            edgecolors="none",
            label=cat,
        )

    # Add legend only if not too many categories
    if n_categories <= 20:
        if legend_loc == "right margin":
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize="small",
                markerscale=0.8,
            )
        else:
            ax.legend(
                loc=legend_loc,
                frameon=False,
                fontsize="small",
            )
