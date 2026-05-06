"""Automated cell type prediction.

Provides singlet.predict_cell_type() — train on a labeled reference and
predict cell types on a query dataset with confidence scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


def predict_cell_type(
    adata: "AnnData",
    reference_adata: "AnnData",
    label_key: str,
    *,
    method: str = "knn",
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    threshold: float = 0.5,
) -> "AnnData":
    """Predict cell types from a labeled reference.

    Train a classifier on the reference dataset and predict labels on the
    query. Includes a rejection option: cells with low confidence are
    labeled as 'Unknown'.

    Parameters
    ----------
    adata
        Query annotated data matrix (cells × genes).
        Must have ``adata.obsm[use_rep]`` computed.
    reference_adata
        Reference annotated data matrix with known cell type labels.
        Must have ``reference_adata.obsm[use_rep]`` computed.
    label_key
        Key in ``reference_adata.obs`` containing cell type labels.
    method
        Classification method: 'knn' (k-nearest neighbors), 'svm'
        (support vector machine), or 'rf' (random forest).
    n_neighbors
        Number of neighbors for kNN method. Ignored for other methods.
    use_rep
        Key in ``.obsm`` for the embedding to use as features.
    threshold
        Minimum confidence for a prediction to be accepted. Cells below
        this threshold are labeled 'Unknown'.

    Returns
    -------
    AnnData
        Input ``adata`` with:
        - ``adata.obs['predicted_cell_type']``: predicted labels.
        - ``adata.obs['prediction_confidence']``: confidence scores [0, 1].
    """
    import numpy as np

    if method not in ("knn", "svm", "rf"):
        msg = f"method must be 'knn', 'svm', or 'rf', got {method!r}"
        raise ValueError(msg)

    if label_key not in reference_adata.obs.columns:
        msg = f"label_key {label_key!r} not found in reference_adata.obs"
        raise KeyError(msg)

    if use_rep not in reference_adata.obsm:
        msg = f"use_rep {use_rep!r} not found in reference_adata.obsm"
        raise KeyError(msg)

    if use_rep not in adata.obsm:
        msg = f"use_rep {use_rep!r} not found in adata.obsm"
        raise KeyError(msg)

    X_train = np.asarray(reference_adata.obsm[use_rep])
    X_query = np.asarray(adata.obsm[use_rep])
    labels = reference_adata.obs[label_key].values

    if method == "knn":
        predictions, confidences = _predict_knn(X_train, X_query, labels, n_neighbors)
    elif method == "svm":
        predictions, confidences = _predict_svm(X_train, X_query, labels)
    else:  # rf
        predictions, confidences = _predict_rf(X_train, X_query, labels)

    # Apply rejection threshold
    predictions = np.array(predictions, dtype=object)
    predictions[confidences < threshold] = "Unknown"

    adata.obs["predicted_cell_type"] = predictions
    adata.obs["prediction_confidence"] = confidences.astype(np.float32)

    return adata


def _predict_knn(
    X_train: "np.ndarray",
    X_query: "np.ndarray",
    labels: "np.ndarray",
    n_neighbors: int,
) -> tuple:
    """KNN-based prediction with per-class vote fraction as confidence."""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, labels)

    predictions = clf.predict(X_query)
    probabilities = clf.predict_proba(X_query)
    confidences = np.max(probabilities, axis=1)

    return predictions, confidences


def _predict_svm(
    X_train: "np.ndarray",
    X_query: "np.ndarray",
    labels: "np.ndarray",
) -> tuple:
    """SVM-based prediction with Platt scaling for probabilities."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_query_scaled = scaler.transform(X_query)

    clf = SVC(kernel="rbf", probability=True, random_state=0)
    clf.fit(X_train_scaled, labels)

    predictions = clf.predict(X_query_scaled)
    probabilities = clf.predict_proba(X_query_scaled)
    confidences = np.max(probabilities, axis=1)

    return predictions, confidences


def _predict_rf(
    X_train: "np.ndarray",
    X_query: "np.ndarray",
    labels: "np.ndarray",
) -> tuple:
    """Random forest prediction with out-of-bag-style probability."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, labels)

    predictions = clf.predict(X_query)
    probabilities = clf.predict_proba(X_query)
    confidences = np.max(probabilities, axis=1)

    return predictions, confidences
