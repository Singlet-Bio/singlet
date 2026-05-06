"""Feature importance for cluster identity.

Provides singlet.gene_importance() — train a classifier to predict cluster
labels and extract gene importances for understanding cluster identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def gene_importance(
    adata: "AnnData",
    *,
    groupby: str = "leiden",
    method: str = "random_forest",
    n_top: int = 100,
    n_estimators: int = 100,
    random_state: int = 0,
    layer: str | None = None,
) -> pd.DataFrame:
    """Compute gene importances for distinguishing cell clusters.

    Trains a classifier (Random Forest or Gradient Boosting) to predict
    cluster labels from gene expression, then extracts feature importances
    to identify genes most important for cluster identity.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Key in adata.obs containing cluster labels to predict.
    method
        Classifier method:
        - 'random_forest': sklearn RandomForestClassifier
        - 'gradient_boosting': sklearn GradientBoostingClassifier
    n_top
        Number of top important genes to return.
    n_estimators
        Number of trees in the ensemble.
    random_state
        Random seed for reproducibility.
    layer
        Expression layer. None uses .X.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['gene', 'importance', 'rank'], sorted by
        importance (descending). Contains top n_top genes.

    Also stores results in adata.uns['gene_importance'] with keys:
        - 'importances': full importance array for all genes
        - 'top_genes': DataFrame of top n_top genes
        - 'params': dict of parameters used

    Raises
    ------
    ValueError
        If method is not recognized or groupby key not in adata.obs.
    KeyError
        If groupby column does not exist in adata.obs.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> top_genes = singlet.gene_importance(adata, groupby='leiden')
    >>> print(top_genes.head(10))
    """
    import numpy as np
    from scipy.sparse import issparse

    valid_methods = ("random_forest", "gradient_boosting")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    if groupby not in adata.obs.columns:
        msg = f"'{groupby}' not found in adata.obs. Available: {list(adata.obs.columns)}"
        raise KeyError(msg)

    # Get expression matrix
    X = adata.layers[layer] if layer is not None else adata.X
    if issparse(X):
        X = np.asarray(X.todense())
    else:
        X = np.asarray(X)

    # Get labels
    labels = adata.obs[groupby].values
    # Encode labels as integers
    unique_labels = np.unique(labels.astype(str))
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_to_int[str(lab)] for lab in labels])

    # Need at least 2 classes
    if len(unique_labels) < 2:
        msg = f"Need at least 2 classes in '{groupby}', got {len(unique_labels)}"
        raise ValueError(msg)

    # Train classifier
    if method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        # GradientBoosting doesn't support n_jobs, use fewer estimators
        clf = GradientBoostingClassifier(
            n_estimators=min(n_estimators, 50),
            random_state=random_state,
            max_features="sqrt",
            max_depth=5,
        )

    clf.fit(X, y)

    # Extract importances
    importances = clf.feature_importances_

    # Get top genes
    n_return = min(n_top, len(importances))
    top_indices = np.argsort(importances)[::-1][:n_return]

    result_df = pd.DataFrame(
        {
            "gene": [adata.var_names[i] for i in top_indices],
            "importance": importances[top_indices],
            "rank": np.arange(1, n_return + 1),
        }
    )

    # Store in adata
    adata.uns["gene_importance"] = {
        "importances": importances,
        "top_genes": result_df,
        "params": {
            "groupby": groupby,
            "method": method,
            "n_estimators": n_estimators,
            "n_top": n_top,
        },
    }

    return result_df
