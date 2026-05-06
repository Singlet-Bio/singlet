"""PAGA graph visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def plot_paga(
    adata: "AnnData",
    *,
    threshold: float = 0.01,
    layout: str = "fr",
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    cmap: str = "tab20",
    labels: list[str] | None = None,
    fontsize: int = 9,
    show: bool = True,
    save: str | None = None,
    ax: "Axes | None" = None,
) -> "Axes | None":
    """Plot the PAGA graph.

    Parameters
    ----------
    adata
        Annotated data matrix with PAGA results in .uns['paga'].
    threshold
        Only show edges above this connectivity.
    layout
        Graph layout algorithm: 'fr' (Fruchterman-Reingold), 'circle', 'grid'.
    node_size_scale
        Scale factor for node sizes.
    edge_width_scale
        Scale factor for edge widths.
    cmap
        Colormap for node colors.
    labels
        Custom labels for nodes. None uses group names.
    fontsize
        Font size for labels.
    show
        Whether to display.
    save
        Save path.
    ax
        Pre-existing axes.

    Returns
    -------
    Axes or None if show=True.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    if "paga" not in adata.uns:
        raise KeyError("'paga' not found in .uns. Run singlet.paga() first.")

    paga_data = adata.uns["paga"]
    connectivity = paga_data["connectivities"]
    groups_key = paga_data["groups"]

    if issparse(connectivity):
        conn_dense = connectivity.toarray()
    else:
        conn_dense = np.asarray(connectivity)

    n_groups = conn_dense.shape[0]

    # Get group labels
    group_labels = adata.obs[groups_key]
    if hasattr(group_labels, "cat"):
        group_names = list(group_labels.cat.categories)
    else:
        group_names = sorted(group_labels.unique(), key=str)

    if labels is not None:
        group_names = labels

    # Compute node sizes (proportional to cell count)
    if hasattr(group_labels, "cat"):
        categories = group_labels.cat.categories
    else:
        categories = sorted(group_labels.unique(), key=str)

    cell_counts = np.array([(adata.obs[groups_key] == g).sum() for g in categories])
    node_sizes = (cell_counts / cell_counts.max()) * 300 * node_size_scale

    # Compute layout
    positions = _compute_layout(conn_dense, layout, n_groups)

    # Apply threshold to edges
    edges = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            w = conn_dense[i, j]
            if w > threshold:
                edges.append((i, j, w))

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Get colors
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i / max(1, n_groups - 1)) for i in range(n_groups)]

    # Draw edges
    if edges:
        max_weight = max(e[2] for e in edges)
        for i, j, w in edges:
            width = (w / max_weight) * 3 * edge_width_scale
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                "k-",
                linewidth=width,
                alpha=0.5,
                zorder=1,
            )

    # Draw nodes
    for i in range(n_groups):
        ax.scatter(
            positions[i, 0],
            positions[i, 1],
            s=node_sizes[i],
            c=[colors[i]],
            zorder=2,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            str(group_names[i]),
            (positions[i, 0], positions[i, 1]),
            fontsize=fontsize,
            ha="center",
            va="center",
            zorder=3,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.close(fig)
        return None

    return ax


def _compute_layout(adjacency: np.ndarray, method: str, n: int) -> np.ndarray:
    """Compute 2D layout for graph nodes."""
    if method == "circle":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    elif method == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        positions = np.zeros((n, 2))
        for i in range(n):
            positions[i] = [i % cols, i // cols]
        return positions
    else:
        # Fruchterman-Reingold force-directed layout
        return _fruchterman_reingold(adjacency, n)


def _fruchterman_reingold(
    adjacency: np.ndarray, n: int, iterations: int = 50, seed: int = 42
) -> np.ndarray:
    """Simple Fruchterman-Reingold force-directed layout."""
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 2))

    k = 1.0 / np.sqrt(n)  # Optimal distance
    temperature = 1.0

    for _ in range(iterations):
        # Repulsive forces (all pairs)
        disp = np.zeros((n, 2))
        for i in range(n):
            diff = pos[i] - pos
            dist = np.sqrt((diff**2).sum(axis=1)) + 1e-10
            force = (k**2) / dist
            disp[i] = (diff * force[:, None]).sum(axis=0)

        # Attractive forces (connected pairs)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    diff = pos[j] - pos[i]
                    dist = np.sqrt((diff**2).sum()) + 1e-10
                    force = (dist**2) / k * adjacency[i, j]
                    disp[i] += (diff / dist) * force
                    disp[j] -= (diff / dist) * force

        # Apply displacement with temperature cooling
        disp_norm = np.sqrt((disp**2).sum(axis=1)) + 1e-10
        pos += (disp / disp_norm[:, None]) * np.minimum(disp_norm, temperature)[:, None]
        temperature *= 0.9

    return pos
