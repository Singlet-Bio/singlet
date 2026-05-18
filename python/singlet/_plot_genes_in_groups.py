# SPDX-License-Identifier: MIT
"""Gene expression violin plots across groups.

Provides singlet.plot_genes_in_groups() — creates violin + strip plots
of gene expression split by categorical groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import matplotlib.figure
    import numpy as np


def plot_genes_in_groups(
    adata,
    var_names: list[str],
    groupby: str,
    *,
    use_raw: bool = False,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple["matplotlib.figure.Figure", "np.ndarray"]:
    """Plot gene expression as violin + strip plots split by groups.

    Creates one subplot per gene with violin plots overlaid with
    individual data points (strip), split by the categories in
    ``adata.obs[groupby]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    var_names : list[str]
        Gene names to plot (must be in adata.var_names or adata.raw.var_names).
    groupby : str
        Column in adata.obs used to group cells (must be categorical or object).
    use_raw : bool, default False
        If True, use adata.raw.X for expression values.
    figsize : tuple[float, float] or None, default None
        Figure size (width, height) in inches. If None, auto-computed as
        (max(4, 2.5 * n_groups), 3 * n_genes).

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Figure and array of axes objects.

    Raises
    ------
    KeyError
        If groupby column or gene names not found.
    ValueError
        If var_names is empty.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> fig, axes = singlet.plot_genes_in_groups(
    ...     adata, ["CD3D", "CD79A", "NKG7"], groupby="leiden"
    ... )
    """
    import matplotlib
    import numpy as np
    import scipy.sparse as sp

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Validation ---
    if not hasattr(adata, "obs"):
        raise TypeError(
            f"plot_genes_in_groups() requires an AnnData object, got {type(adata).__name__}"
        )

    if not var_names:
        raise ValueError("var_names must not be empty.")

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.columns")

    # Determine expression source
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None. Set use_raw=False or assign .raw first.")
        source_var_names = list(adata.raw.var_names)
        X_source = adata.raw.X
    else:
        source_var_names = list(adata.var_names)
        X_source = adata.X

    # Validate gene names
    for gene in var_names:
        if gene not in source_var_names:
            raise KeyError(
                f"Gene '{gene}' not found in {'adata.raw.var_names' if use_raw else 'adata.var_names'}"
            )

    # Get group labels
    groups = adata.obs[groupby]
    if hasattr(groups, "cat"):
        categories = list(groups.cat.categories)
    else:
        categories = sorted(groups.unique())
    n_groups = len(categories)
    n_genes = len(var_names)

    # Auto figure size
    if figsize is None:
        figsize = (max(4.0, 2.5 * n_groups), 3.0 * n_genes)

    fig, axes = plt.subplots(n_genes, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    rng = np.random.default_rng(0)

    for gene_idx, gene in enumerate(var_names):
        ax = axes[gene_idx]
        col_idx = source_var_names.index(gene)

        # Extract expression values for this gene
        if sp.issparse(X_source):
            expr = np.asarray(X_source[:, col_idx].todense()).ravel()
        else:
            expr = np.asarray(X_source[:, col_idx]).ravel()

        # Group expression by category
        group_data = []
        for cat in categories:
            mask = groups == cat
            group_data.append(expr[mask])

        # Violin plot
        positions = list(range(n_groups))
        parts = ax.violinplot(
            group_data,
            positions=positions,
            showextrema=False,
            showmedians=False,
        )

        # Style violins
        for pc in parts["bodies"]:
            pc.set_facecolor("#5e81ac")
            pc.set_alpha(0.6)

        # Strip overlay (jittered points)
        for pos, data in zip(positions, group_data):
            jitter = rng.uniform(-0.15, 0.15, size=len(data))
            ax.scatter(
                pos + jitter,
                data,
                s=3,
                alpha=0.4,
                c="#2e3440",
                edgecolors="none",
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("Expression")
        ax.set_title(gene)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    return fig, axes
