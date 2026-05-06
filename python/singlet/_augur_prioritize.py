"""Augur-style cell type prioritization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

if TYPE_CHECKING:
    pass


def augur_prioritize(
    adata: AnnData,
    condition_key: str,
    *,
    cell_type_key: str = "cell_type",
    n_folds: int = 3,
    classifier: str = "rf",
    n_estimators: int = 100,
    random_state: int = 0,
    subsample_size: int | None = None,
    copy: bool = False,
) -> pd.DataFrame:
    """Augur-style cell type prioritization.

    For each cell type, trains a classifier to distinguish conditions (e.g.
    treatment vs control). Cell types with higher classification accuracy
    are more "perturbed" by the condition.

    Parameters
    ----------
    adata
        Annotated data matrix.
    condition_key
        Key in ``adata.obs`` identifying the condition/perturbation
        (must have exactly 2 unique values).
    cell_type_key
        Key in ``adata.obs`` identifying cell types.
    n_folds
        Number of cross-validation folds.
    classifier
        Classifier type. One of 'rf' (random forest) or 'lr' (logistic regression).
    n_estimators
        Number of trees for random forest classifier.
    random_state
        Random seed for reproducibility.
    subsample_size
        If set, subsample each cell type to this many cells per condition.
        Helps balance cell type representation.
    copy
        If True, does not store results in adata.uns.

    Returns
    -------
    pd.DataFrame with columns: cell_type, auc, mean_accuracy, n_cells.
    Also stores results in ``adata.uns['augur_results']`` unless copy=True.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    if condition_key not in adata.obs.columns:
        msg = f"'{condition_key}' not found in adata.obs"
        raise KeyError(msg)
    if cell_type_key not in adata.obs.columns:
        msg = f"'{cell_type_key}' not found in adata.obs"
        raise KeyError(msg)

    conditions = adata.obs[condition_key].unique()
    if len(conditions) != 2:
        msg = (
            f"condition_key '{condition_key}' must have exactly 2 unique values, "
            f"got {len(conditions)}: {list(conditions)}"
        )
        raise ValueError(msg)

    # Encode conditions as binary
    le = LabelEncoder()
    le.fit(conditions)

    cell_types = adata.obs[cell_type_key].unique()
    results = []

    rng = np.random.default_rng(random_state)

    for ct in cell_types:
        mask = adata.obs[cell_type_key] == ct
        ct_adata = adata[mask]
        n_cells = int(mask.sum())

        # Need enough cells for cross-validation
        if n_cells < n_folds * 2:
            results.append(
                {
                    "cell_type": ct,
                    "auc": np.nan,
                    "mean_accuracy": np.nan,
                    "n_cells": n_cells,
                }
            )
            continue

        # Get features
        from scipy.sparse import issparse

        X = ct_adata.X
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        y = le.transform(ct_adata.obs[condition_key].values)

        # Subsample if requested
        if subsample_size is not None and n_cells > subsample_size * 2:
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            n_take = min(subsample_size, len(idx0), len(idx1))
            chosen_idx0 = rng.choice(idx0, size=n_take, replace=False)
            chosen_idx1 = rng.choice(idx1, size=n_take, replace=False)
            chosen = np.concatenate([chosen_idx0, chosen_idx1])
            X = X[chosen]
            y = y[chosen]
            n_cells = len(chosen)

        # Check both classes present
        if len(np.unique(y)) < 2:
            results.append(
                {
                    "cell_type": ct,
                    "auc": np.nan,
                    "mean_accuracy": np.nan,
                    "n_cells": n_cells,
                }
            )
            continue

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_aucs = []
        fold_accs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if classifier == "rf":
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                )
            elif classifier == "lr":
                clf = LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    solver="lbfgs",
                )
            else:
                msg = f"Unknown classifier '{classifier}'. Use 'rf' or 'lr'."
                raise ValueError(msg)

            clf.fit(X_train, y_train)

            # Accuracy
            acc = clf.score(X_test, y_test)
            fold_accs.append(acc)

            # AUC
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = clf.decision_function(X_test)

            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = np.nan
            fold_aucs.append(auc)

        results.append(
            {
                "cell_type": ct,
                "auc": float(np.nanmean(fold_aucs)),
                "mean_accuracy": float(np.mean(fold_accs)),
                "n_cells": n_cells,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("auc", ascending=False).reset_index(drop=True)

    if not copy:
        adata.uns["augur_results"] = df

    return df
