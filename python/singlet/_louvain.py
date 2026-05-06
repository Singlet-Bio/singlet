"""Louvain community detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def louvain(
    adata: AnnData,
    *,
    resolution: float = 1.0,
    random_state: int = 0,
    key_added: str = "louvain",
    adjacency: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Cluster cells using Louvain community detection.

    Requires neighbors to be computed first.

    Parameters
    ----------
    adata
        Annotated data matrix.
    resolution
        Resolution parameter. Higher = more clusters.
    random_state
        Random seed for reproducibility.
    key_added
        Key in .obs to store cluster labels.
    adjacency
        Key in .obsp to use as adjacency matrix. Default: 'connectivities'.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores cluster labels in `.obs[key_added]`.
    """
    import igraph as ig

    adata = adata.copy() if copy else adata

    adj_key = adjacency or "connectivities"
    if adj_key not in adata.obsp:
        raise KeyError(f"'{adj_key}' not found in .obsp. Run singlet.neighbors() first.")

    adjacency_matrix = adata.obsp[adj_key]

    # Convert to igraph
    sources, targets = adjacency_matrix.nonzero()
    weights = np.asarray(adjacency_matrix[sources, targets]).flatten()

    g = ig.Graph(directed=False)
    g.add_vertices(adjacency_matrix.shape[0])
    edges = list(zip(sources.tolist(), targets.tolist()))
    g.add_edges(edges)
    g.es["weight"] = weights.tolist()

    # Remove self-loops and multi-edges
    g.simplify(combine_edges="max")

    # Run Louvain (community_multilevel in igraph)
    partition = g.community_multilevel(
        weights="weight",
        resolution=resolution,
    )

    labels = np.array(partition.membership)

    adata.obs[key_added] = pd.Categorical(
        values=labels.astype(str),
        categories=sorted(set(labels.astype(str)), key=lambda x: int(x)),
    )

    return adata if copy else None
