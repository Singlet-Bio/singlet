# SPDX-License-Identifier: MIT
"""Metacell aggregation for AnnData objects.

Provides singlet.metacell() — group similar cells into metacells
via k-means clustering (or an existing grouping), then aggregate
gene expression to produce a compact representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def metacell(
    adata: AnnData,
    *,
    n_metacells: int = 100,
    groupby: str | None = None,
    method: str = "kmeans",
    use_rep: str = "X_pca",
) -> AnnData:
    """Aggregate cells into metacells by expression similarity.

    Groups cells via k-means (default) on a low-dimensional
    representation, then computes mean expression per group to
    produce a compact AnnData with one observation per metacell.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have ``adata.obsm[use_rep]``
        computed (e.g., via ``singlet.pca(adata)``) unless
        ``groupby`` is provided.
    n_metacells : int, default 100
        Number of metacells to generate (ignored if groupby is set).
    groupby : str or None, default None
        If provided, use this obs column as the grouping instead
        of computing k-means. Each unique value becomes one metacell.
    method : str, default 'kmeans'
        Clustering method for grouping cells. Currently only 'kmeans'
        is supported.
    use_rep : str, default 'X_pca'
        Representation in adata.obsm to use for k-means clustering.

    Returns
    -------
    anndata.AnnData
        New AnnData with shape (n_metacells, n_vars) containing mean
        expression. Includes:
        - ``.obs['metacell_size']``: number of cells per metacell.
        - ``.obs_names``: metacell identifiers.
        - ``.var``: copied from input.

    Side Effects
    ------------
    Sets ``adata.obs['metacell']`` in the input AnnData to the
    assigned metacell label for each cell.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> mc = singlet.metacell(adata, n_metacells=50)
    >>> mc.shape[0]
    50
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from anndata import AnnData as AnnDataClass

    if method not in ("kmeans",):
        msg = f"method must be 'kmeans', got {method!r}"
        raise ValueError(msg)

    if groupby is not None:
        # Use existing grouping
        if groupby not in adata.obs.columns:
            msg = f"Column {groupby!r} not found in adata.obs"
            raise KeyError(msg)
        labels = adata.obs[groupby].astype(str).values
    else:
        # K-means on low-dimensional representation
        if use_rep not in adata.obsm:
            msg = (
                f"Representation {use_rep!r} not found in adata.obsm. Run singlet.pca(adata) first."
            )
            raise KeyError(msg)

        from sklearn.cluster import MiniBatchKMeans

        n_metacells = min(n_metacells, adata.n_obs)
        kmeans = MiniBatchKMeans(
            n_clusters=n_metacells,
            random_state=0,
            batch_size=min(1024, adata.n_obs),
        )
        labels = kmeans.fit_predict(adata.obsm[use_rep]).astype(str)

    # Store assignments in original adata
    adata.obs["metacell"] = labels

    # Aggregate expression per metacell
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)
    n_genes = adata.n_vars

    # Get expression matrix
    mat = adata.X
    agg_mat = np.zeros((n_groups, n_genes), dtype=np.float32)
    sizes = np.zeros(n_groups, dtype=np.int64)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        sizes[idx] = mask.sum()
        subset = mat[mask]
        if sp.issparse(subset):
            agg_mat[idx] = np.asarray(subset.mean(axis=0)).ravel()
        else:
            agg_mat[idx] = subset.mean(axis=0)

    # Build new AnnData
    obs_df = pd.DataFrame(
        {"metacell_size": sizes},
        index=[f"metacell_{lab}" for lab in unique_labels],
    )
    var_df = adata.var.copy()

    mc_adata = AnnDataClass(X=agg_mat, obs=obs_df, var=var_df)
    return mc_adata
