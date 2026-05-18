# SPDX-License-Identifier: MIT
"""Ranking plot for DE genes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.figure import Figure


def plot_ranking(
    adata: "AnnData",
    *,
    n_genes: int = 20,
    groups: list[str] | None = None,
    key: str = "rank_genes_groups",
    fontsize: int = 8,
    ncols: int = 4,
    show: bool = True,
    save: str | None = None,
) -> "Figure | None":
    """Plot ranked DE genes as horizontal bar plots.

    Shows top N genes per group with their test statistics (scores).

    Parameters
    ----------
    adata
        Annotated data matrix with DE results.
    n_genes
        Number of top genes per group.
    groups
        Subset of groups to show. None shows all.
    key
        Key in .uns for DE results.
    fontsize
        Font size for gene labels.
    ncols
        Number of columns in grid.
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

    if key not in adata.uns:
        raise KeyError(f"'{key}' not found in .uns. Run singlet.rank_genes_groups() first.")

    de_results = adata.uns[key]

    # Support both dict-of-lists (singlet) and structured array (scanpy) formats
    names_data = de_results["names"]
    if isinstance(names_data, dict):
        all_groups = list(names_data.keys())
    else:
        all_groups = list(names_data.dtype.names)

    if groups is not None:
        plot_groups = [g for g in groups if g in all_groups]
    else:
        plot_groups = all_groups

    if len(plot_groups) == 0:
        raise ValueError("No groups to plot.")

    n_groups = len(plot_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * max(2, n_genes * 0.2)),
        squeeze=False,
    )

    for idx, group in enumerate(plot_groups):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        names = list(de_results["names"][group][:n_genes])
        scores = list(de_results["scores"][group][:n_genes])

        # Reverse for horizontal bar (top gene at top)
        names = names[::-1]
        scores = scores[::-1]

        colors = ["#d62728" if s > 0 else "#1f77b4" for s in scores]

        ax.barh(range(len(names)), scores, color=colors, height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=fontsize)
        ax.set_title(f"Group {group}", fontsize=fontsize + 2)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Score", fontsize=fontsize)

    # Hide empty subplots
    for idx in range(n_groups, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return fig
