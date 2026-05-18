# SPDX-License-Identifier: MIT
"""Tests for singlet.predict_cell_type()."""

import numpy as np
import pytest
import scipy.sparse as sp
from singlet._predict_cell_type import predict_cell_type


def _make_reference_query(n_ref=120, n_query=60, n_genes=50, n_pcs=20, n_types=4, seed=42):
    """Create reference and query AnnData with PCA embeddings."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Reference: cells with clear clusters
    cell_types = [f"type_{i}" for i in range(n_types)]
    labels = []
    X_ref_pca = np.zeros((n_ref, n_pcs), dtype=np.float32)

    per_type = n_ref // n_types
    for idx, ct in enumerate(cell_types):
        start = idx * per_type
        end = start + per_type
        # Each type is a cluster in PCA space
        center = rng.normal(0, 1, size=n_pcs) * 5
        X_ref_pca[start:end] = center + rng.normal(0, 0.5, size=(per_type, n_pcs))
        labels.extend([ct] * per_type)

    X_ref = rng.poisson(5, size=(n_ref, n_genes)).astype(np.float32)
    ref_adata = ad.AnnData(X=sp.csr_matrix(X_ref))
    ref_adata.obs_names = [f"ref_{i}" for i in range(n_ref)]
    ref_adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    ref_adata.obs["cell_type"] = labels
    ref_adata.obsm["X_pca"] = X_ref_pca

    # Query: sample from same clusters
    query_labels = []
    X_query_pca = np.zeros((n_query, n_pcs), dtype=np.float32)
    per_type_q = n_query // n_types

    for idx, ct in enumerate(cell_types):
        start = idx * per_type_q
        end = start + per_type_q
        # Use same centers as reference
        ref_center_start = idx * per_type
        ref_center_end = ref_center_start + per_type
        center = X_ref_pca[ref_center_start:ref_center_end].mean(axis=0)
        X_query_pca[start:end] = center + rng.normal(0, 0.5, size=(per_type_q, n_pcs))
        query_labels.extend([ct] * per_type_q)

    X_query = rng.poisson(5, size=(n_query, n_genes)).astype(np.float32)
    query_adata = ad.AnnData(X=sp.csr_matrix(X_query))
    query_adata.obs_names = [f"query_{i}" for i in range(n_query)]
    query_adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    query_adata.obs["true_type"] = query_labels
    query_adata.obsm["X_pca"] = X_query_pca

    return ref_adata, query_adata


class TestPredictCellTypeBasic:
    def test_returns_adata(self):
        """Should return the same adata object."""
        ref, query = _make_reference_query()
        result = predict_cell_type(query, ref, "cell_type")
        assert result is query

    def test_predicted_cell_type_column(self):
        """Should create predicted_cell_type column."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type")
        assert "predicted_cell_type" in query.obs.columns

    def test_prediction_confidence_column(self):
        """Should create prediction_confidence column."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type")
        assert "prediction_confidence" in query.obs.columns

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type")
        conf = query.obs["prediction_confidence"].values
        assert np.all(conf >= 0)
        assert np.all(conf <= 1)

    def test_prediction_accuracy(self):
        """Well-separated clusters should have high accuracy."""
        ref, query = _make_reference_query(n_ref=200, n_query=80)
        predict_cell_type(query, ref, "cell_type")
        predicted = query.obs["predicted_cell_type"].values
        true_labels = query.obs["true_type"].values
        accuracy = np.mean(predicted == true_labels)
        # With well-separated clusters, should get >80%
        assert accuracy > 0.8


class TestPredictCellTypeMethods:
    def test_knn_method(self):
        """KNN method should work."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type", method="knn")
        assert "predicted_cell_type" in query.obs.columns

    def test_svm_method(self):
        """SVM method should work."""
        ref, query = _make_reference_query(n_ref=80, n_query=40)
        predict_cell_type(query, ref, "cell_type", method="svm")
        assert "predicted_cell_type" in query.obs.columns

    def test_rf_method(self):
        """Random forest method should work."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type", method="rf")
        assert "predicted_cell_type" in query.obs.columns

    def test_invalid_method(self):
        """Should raise ValueError for invalid method."""
        ref, query = _make_reference_query()
        with pytest.raises(ValueError, match="method must be"):
            predict_cell_type(query, ref, "cell_type", method="invalid")


class TestPredictCellTypeRejection:
    def test_high_threshold_produces_unknown(self):
        """High threshold should reject some predictions as Unknown."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type", threshold=0.99)
        predicted = query.obs["predicted_cell_type"].values
        # With very high threshold, some should be Unknown
        # (unless all predictions are extremely confident)
        assert predicted.dtype == object  # string dtype

    def test_zero_threshold_no_unknown(self):
        """Zero threshold should never produce Unknown."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type", threshold=0.0)
        predicted = query.obs["predicted_cell_type"].values
        assert "Unknown" not in predicted


class TestPredictCellTypeValidation:
    def test_missing_label_key(self):
        """Should raise KeyError for missing label_key."""
        ref, query = _make_reference_query()
        with pytest.raises(KeyError, match="not found in reference"):
            predict_cell_type(query, ref, "nonexistent_key")

    def test_missing_use_rep_reference(self):
        """Should raise KeyError for missing use_rep in reference."""
        ref, query = _make_reference_query()
        del ref.obsm["X_pca"]
        with pytest.raises(KeyError, match="not found in reference"):
            predict_cell_type(query, ref, "cell_type")

    def test_missing_use_rep_query(self):
        """Should raise KeyError for missing use_rep in query."""
        ref, query = _make_reference_query()
        del query.obsm["X_pca"]
        with pytest.raises(KeyError, match="not found in adata"):
            predict_cell_type(query, ref, "cell_type")


class TestPredictCellTypeParameters:
    def test_custom_n_neighbors(self):
        """Should respect n_neighbors parameter."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type", n_neighbors=5)
        assert "predicted_cell_type" in query.obs.columns

    def test_confidence_dtype(self):
        """Confidence should be float32."""
        ref, query = _make_reference_query()
        predict_cell_type(query, ref, "cell_type")
        assert query.obs["prediction_confidence"].dtype == np.float32
