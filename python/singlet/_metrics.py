"""Clustering metrics."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def silhouette_score(
    adata: AnnData,
    *,
    groupby: str = "leiden",
    use_rep: str = "X_pca",
    n_pcs: int | None = None,
    metric: str = "euclidean",
    sample_size: int | None = None,
    random_state: int = 0,
) -> float:
    """Compute mean silhouette score for clustering quality.

    Higher silhouette indicates better-separated clusters.
    Range: [-1, 1]. Above 0.5 is good, above 0.7 is excellent.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Key in .obs with cluster labels.
    use_rep
        Key in .obsm for distance computation.
    n_pcs
        Number of PCs to use.
    metric
        Distance metric.
    sample_size
        Sample size for faster computation. None uses all cells.
    random_state
        Random seed for sampling.

    Returns
    -------
    Mean silhouette score (float).
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")

    if use_rep not in adata.obsm:
        raise KeyError(f"'{use_rep}' not found in .obsm.")

    X = adata.obsm[use_rep]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    labels = adata.obs[groupby].values

    # Need at least 2 clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    score = sklearn_silhouette(
        X,
        labels,
        metric=metric,
        sample_size=sample_size,
        random_state=random_state,
    )

    return float(score)


def calinski_harabasz_score(
    adata: AnnData,
    *,
    groupby: str = "leiden",
    use_rep: str = "X_pca",
    n_pcs: int | None = None,
) -> float:
    """Compute Calinski-Harabasz index for clustering quality.

    Higher is better. Also known as Variance Ratio Criterion.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Key in .obs with cluster labels.
    use_rep
        Key in .obsm for distance computation.
    n_pcs
        Number of PCs to use.

    Returns
    -------
    Calinski-Harabasz score (float).
    """
    from sklearn.metrics import calinski_harabasz_score as sklearn_ch

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in .obs.")

    if use_rep not in adata.obsm:
        raise KeyError(f"'{use_rep}' not found in .obsm.")

    X = adata.obsm[use_rep]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    labels = adata.obs[groupby].values
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    return float(sklearn_ch(X, labels))
