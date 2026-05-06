"""Tests for singlet.transfer_labels()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_ref_query(n_ref=100, n_query=50, n_genes=200, n_types=3, seed=42):
    """Create reference and query datasets for label transfer."""
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i}" for i in range(n_genes)]

    # Reference with distinct clusters
    X_ref = np.zeros((n_ref, n_genes), dtype=np.float32)
    labels = []
    for idx in range(n_ref):
        cluster = idx % n_types
        # Each cluster has high expression in different genes
        start = cluster * (n_genes // n_types)
        end = start + (n_genes // n_types)
        X_ref[idx, start:end] = rng.standard_normal(end - start).astype(np.float32) + 3
        X_ref[idx] += rng.standard_normal(n_genes).astype(np.float32) * 0.5
        labels.append(f"type_{cluster}")

    ref = AnnData(X=X_ref)
    ref.var_names = genes
    ref.obs_names = [f"ref_{i}" for i in range(n_ref)]
    ref.obs["celltype"] = labels
    singlet.pca(ref, n_comps=30, use_highly_variable=False)

    # Query from same distribution
    X_query = np.zeros((n_query, n_genes), dtype=np.float32)
    true_labels = []
    for idx in range(n_query):
        cluster = idx % n_types
        start = cluster * (n_genes // n_types)
        end = start + (n_genes // n_types)
        X_query[idx, start:end] = rng.standard_normal(end - start).astype(np.float32) + 3
        X_query[idx] += rng.standard_normal(n_genes).astype(np.float32) * 0.5
        true_labels.append(f"type_{cluster}")

    query = AnnData(X=X_query)
    query.var_names = genes
    query.obs_names = [f"query_{i}" for i in range(n_query)]
    query.obs["true_label"] = true_labels

    return ref, query


class TestTransferLabels:
    def test_basic_transfer(self):
        ref, query = _make_ref_query()
        result = singlet.transfer_labels(query, ref, "celltype")
        assert result is query  # returns same object
        assert "celltype_transferred" in query.obs.columns
        assert "celltype_confidence" in query.obs.columns

    def test_labels_are_valid(self):
        ref, query = _make_ref_query()
        singlet.transfer_labels(query, ref, "celltype")
        valid_labels = set(ref.obs["celltype"].unique())
        for label in query.obs["celltype_transferred"]:
            assert label in valid_labels

    def test_confidence_range(self):
        ref, query = _make_ref_query()
        singlet.transfer_labels(query, ref, "celltype")
        conf = query.obs["celltype_confidence"].values
        assert np.all(conf >= 0)
        assert np.all(conf <= 1)

    def test_accuracy_on_structured_data(self):
        """Labels should transfer accurately on well-separated clusters."""
        ref, query = _make_ref_query(n_ref=150, n_query=60, seed=123)
        singlet.transfer_labels(query, ref, "celltype")
        transferred = query.obs["celltype_transferred"].values
        true_labels = query.obs["true_label"].values
        accuracy = np.mean(
            [str(t) == str(p) for t, p in zip(true_labels, transferred)]
        )
        # Should get >60% on well-separated clusters
        assert accuracy > 0.6

    def test_n_neighbors_parameter(self):
        ref, query = _make_ref_query()
        singlet.transfer_labels(query, ref, "celltype", n_neighbors=5)
        assert "celltype_transferred" in query.obs.columns

    def test_with_precomputed_query_pca(self):
        ref, query = _make_ref_query()
        singlet.pca(query, n_comps=30, use_highly_variable=False)
        singlet.transfer_labels(query, ref, "celltype")
        assert "celltype_transferred" in query.obs.columns

    def test_missing_label_key_raises(self):
        ref, query = _make_ref_query()
        with pytest.raises(KeyError, match="nonexistent"):
            singlet.transfer_labels(query, ref, "nonexistent")

    def test_missing_ref_embedding_raises(self):
        rng = np.random.default_rng(42)
        ref = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
        query = AnnData(X=rng.standard_normal((20, 100)).astype(np.float32))
        ref.var_names = [f"g{i}" for i in range(100)]
        query.var_names = [f"g{i}" for i in range(100)]
        ref.obs["label"] = ["a"] * 50
        with pytest.raises(KeyError, match="X_pca"):
            singlet.transfer_labels(query, ref, "label")

    def test_categorical_output(self):
        ref, query = _make_ref_query()
        singlet.transfer_labels(query, ref, "celltype")
        assert hasattr(query.obs["celltype_transferred"], "cat")

    def test_different_n_neighbors(self):
        """Different k values should still produce valid results."""
        ref, query = _make_ref_query()
        singlet.transfer_labels(query, ref, "celltype", n_neighbors=3)
        assert "celltype_transferred" in query.obs.columns

        ref2, query2 = _make_ref_query()
        singlet.transfer_labels(query2, ref2, "celltype", n_neighbors=20)
        assert "celltype_transferred" in query2.obs.columns

    def test_public_api(self):
        assert hasattr(singlet, "transfer_labels")
        assert callable(singlet.transfer_labels)

    def test_confidence_higher_for_clear_assignments(self):
        """Well-separated cells should have higher confidence."""
        ref, query = _make_ref_query(n_ref=200, n_query=60, seed=99)
        singlet.transfer_labels(query, ref, "celltype")
        # Average confidence should be reasonable
        mean_conf = query.obs["celltype_confidence"].mean()
        assert mean_conf > 0.3
