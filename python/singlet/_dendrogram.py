# SPDX-License-Identifier: MIT
"""Hierarchical clustering (dendrogram) for AnnData groups.

Provides singlet.dendrogram() — computes hierarchical clustering of cell
groups based on their mean expression profiles.
"""

from __future__ import annotations

from typing import Optional


def dendrogram(
    adata,
    groupby: str,
    *,
    use_rep: Optional[str] = "X_pca",
    method: str = "ward",
    optimal_ordering: bool = True,
    inplace: bool = True,
) -> Optional[dict]:
    """Compute hierarchical clustering of groups.

    Groups cells by `groupby` and computes a dendrogram based on
    average expression (or PCA) profiles. Useful for ordering groups
    in heatmaps and dot plots.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    groupby : str
        Column in adata.obs to group cells by.
    use_rep : str or None, default "X_pca"
        Representation to use. If "X_pca", uses PCA embeddings.
        If None, uses adata.X (mean expression).
    method : str, default "ward"
        Linkage method for scipy.cluster.hierarchy.linkage.
        Options: 'ward', 'single', 'complete', 'average', 'weighted'.
    optimal_ordering : bool, default True
        If True, reorders leaves for minimal distance between
        adjacent leaves.
    inplace : bool, default True
        If True, stores result in adata.uns[f'dendrogram_{groupby}'].
        If False, returns result dict.

    Returns
    -------
    dict or None
        Dict with keys: 'linkage', 'categories_ordered', 'dendrogram_info'.
        Or None if inplace=True.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.leiden(adata)
    >>> singlet.dendrogram(adata, "leiden")
    >>> adata.uns['dendrogram_leiden']['categories_ordered']
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    if not hasattr(adata, "obs"):
        raise TypeError(f"dendrogram() requires an AnnData object, got {type(adata).__name__}")

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.columns")

    groups = adata.obs[groupby]
    categories = sorted(groups.unique(), key=str)
    n_groups = len(categories)

    if n_groups < 2:
        result = {
            "linkage": np.array([]),
            "categories_ordered": categories,
            "dendrogram_info": {"method": method},
        }
        if inplace:
            adata.uns[f"dendrogram_{groupby}"] = result
            return None
        return result

    # Compute group centroids
    if use_rep is not None and use_rep in adata.obsm:
        X_rep = np.array(adata.obsm[use_rep])
    else:
        if sp.issparse(adata.X):
            X_rep = np.asarray(adata.X.todense())
        else:
            X_rep = np.array(adata.X)

    centroids = np.zeros((n_groups, X_rep.shape[1]), dtype=np.float64)
    for i, cat in enumerate(categories):
        mask = np.array(groups == cat)
        centroids[i] = X_rep[mask].mean(axis=0)

    # Compute pairwise distances and linkage
    if n_groups == 2:
        # pdist needs at least 2 points, linkage needs at least 2
        dists = pdist(centroids, metric="euclidean")
    else:
        dists = pdist(centroids, metric="euclidean")

    Z = linkage(dists, method=method, optimal_ordering=optimal_ordering)

    # Get leaf order
    leaf_order = leaves_list(Z)
    categories_ordered = [categories[i] for i in leaf_order]

    result = {
        "linkage": Z,
        "categories_ordered": categories_ordered,
        "dendrogram_info": {
            "method": method,
            "use_rep": use_rep,
            "optimal_ordering": optimal_ordering,
        },
    }

    if inplace:
        adata.uns[f"dendrogram_{groupby}"] = result
        return None
    return result
