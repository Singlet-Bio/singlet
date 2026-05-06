"""Cross-validate clustering resolution."""

from __future__ import annotations

import pandas as pd
from anndata import AnnData


def cross_validate_resolution(
    adata: AnnData,
    *,
    resolutions: list[float] | None = None,
    metric: str = "silhouette",
    n_pcs: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Find optimal Leiden resolution by evaluating cluster quality metrics.

    Runs Leiden at multiple resolutions and evaluates each using the
    specified metric.

    Parameters
    ----------
    adata
        Annotated data matrix with neighbors computed.
    resolutions
        List of resolutions to try. Default: [0.1, 0.2, ..., 2.0].
    metric
        Quality metric: 'silhouette' or 'calinski_harabasz'.
    n_pcs
        Number of PCs for metric computation.
    random_state
        Random seed.

    Returns
    -------
    DataFrame with columns: resolution, n_clusters, score.
    Best resolution has highest score.
    """
    import singlet

    if "connectivities" not in adata.obsp:
        raise KeyError("'connectivities' not in .obsp. Run singlet.neighbors() first.")

    if "X_pca" not in adata.obsm:
        raise KeyError("'X_pca' not in .obsm. Run singlet.pca() first.")

    if resolutions is None:
        resolutions = [round(x * 0.1, 1) for x in range(1, 21)]

    results = []

    for res in resolutions:
        # Cluster at this resolution
        key = f"_cv_leiden_{res}"
        singlet.leiden(adata, resolution=res, key_added=key, random_state=random_state)

        n_clusters = len(adata.obs[key].unique())

        # Compute metric
        if n_clusters < 2:
            score = 0.0
        elif metric == "silhouette":
            score = singlet.silhouette_score(
                adata,
                groupby=key,
                n_pcs=n_pcs,
            )
        elif metric == "calinski_harabasz":
            score = singlet.calinski_harabasz_score(
                adata,
                groupby=key,
                n_pcs=n_pcs,
            )
        else:
            raise ValueError(f"Unknown metric '{metric}'.")

        results.append(
            {
                "resolution": res,
                "n_clusters": n_clusters,
                "score": score,
            }
        )

        # Clean up temporary column
        del adata.obs[key]

    return pd.DataFrame(results)
