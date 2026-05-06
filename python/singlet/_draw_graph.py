"""Force-directed graph layout."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def draw_graph(
    adata: AnnData,
    *,
    layout: str = "fa",
    random_state: int = 0,
    n_iterations: int = 500,
    key_added: str = "draw_graph_fa",
    adjacency: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Compute a force-directed graph layout.

    Provides an alternative to UMAP/t-SNE for visualizing the neighbor graph.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layout
        Layout algorithm: 'fa' (ForceAtlas2-like), 'fr' (Fruchterman-Reingold).
    random_state
        Random seed for initialization.
    n_iterations
        Number of iterations.
    key_added
        Key in .obsm for storing the layout.
    adjacency
        Key in .obsp. Default: 'connectivities'.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores layout in `.obsm['X_{key_added}']`.
    """
    from scipy.sparse import issparse

    adata = adata.copy() if copy else adata

    adj_key = adjacency or "connectivities"
    if adj_key not in adata.obsp:
        raise KeyError(f"'{adj_key}' not found in .obsp. Run singlet.neighbors() first.")

    W = adata.obsp[adj_key]
    n = W.shape[0]

    rng = np.random.default_rng(random_state)
    pos = rng.standard_normal((n, 2)).astype(np.float64)

    # Get edge list
    if issparse(W):
        W_coo = W.tocoo()
        sources = W_coo.row
        targets = W_coo.col
        weights = W_coo.data
    else:
        sources, targets = np.nonzero(W)
        weights = W[sources, targets]

    if layout == "fa":
        pos = _force_atlas2(pos, sources, targets, weights, n_iterations)
    else:
        pos = _fruchterman_reingold(pos, sources, targets, weights, n, n_iterations)

    obsm_key = f"X_{key_added}"
    adata.obsm[obsm_key] = pos.astype(np.float32)

    return adata if copy else None


def _force_atlas2(pos: np.ndarray, sources, targets, weights, iterations: int) -> np.ndarray:
    """Simplified ForceAtlas2 layout."""
    n = pos.shape[0]
    speed = 1.0
    gravity = 1.0

    for _ in range(iterations):
        forces = np.zeros_like(pos)

        # Repulsive forces
        if n <= 500:
            for i in range(n):
                diff = pos[i] - pos
                dist = np.sqrt((diff**2).sum(axis=1)) + 0.01
                repulsion = diff / dist[:, None]
                forces[i] += repulsion.sum(axis=0)
        else:
            # Grid-based approximation for large graphs
            center = pos.mean(axis=0)
            diff = pos - center
            dist = np.sqrt((diff**2).sum(axis=1)) + 0.01
            forces += diff / dist[:, None] * n * 0.1

        # Attractive forces (edges)
        for idx in range(len(sources)):
            i, j = sources[idx], targets[idx]
            w = weights[idx] if idx < len(weights) else 1.0
            diff = pos[j] - pos[i]
            attraction = diff * w * 0.01
            forces[i] += attraction
            forces[j] -= attraction

        # Gravity (pull toward center)
        center = pos.mean(axis=0)
        forces -= (pos - center) * gravity * 0.01

        # Apply forces
        force_mag = np.sqrt((forces**2).sum(axis=1)) + 0.01
        pos += forces / force_mag[:, None] * speed

        # Reduce speed over time
        speed *= 0.995

    return pos


def _fruchterman_reingold(
    pos: np.ndarray, sources, targets, weights, n: int, iterations: int
) -> np.ndarray:
    """Fruchterman-Reingold force-directed layout."""
    k = np.sqrt(1.0 / n)
    temperature = 1.0

    for _ in range(iterations):
        disp = np.zeros_like(pos)

        # Repulsive forces
        if n <= 300:
            for i in range(n):
                diff = pos[i] - pos
                dist = np.sqrt((diff**2).sum(axis=1)) + 1e-6
                force = (k**2) / dist
                disp[i] += (diff * force[:, None]).sum(axis=0)
        else:
            # Approximation for large graphs
            center = pos.mean(axis=0)
            diff = pos - center
            dist = np.sqrt((diff**2).sum(axis=1, keepdims=True)) + 1e-6
            disp += diff / dist * k * n * 0.01

        # Attractive forces
        for idx in range(len(sources)):
            i, j = int(sources[idx]), int(targets[idx])
            diff = pos[j] - pos[i]
            dist = np.sqrt((diff**2).sum()) + 1e-6
            w = weights[idx] if idx < len(weights) else 1.0
            force = (dist / k) * w
            disp[i] += (diff / dist) * force
            disp[j] -= (diff / dist) * force

        # Apply with temperature
        disp_norm = np.sqrt((disp**2).sum(axis=1)) + 1e-6
        pos += (disp / disp_norm[:, None]) * np.minimum(disp_norm, temperature)[:, None]
        temperature *= 0.95

    return pos
