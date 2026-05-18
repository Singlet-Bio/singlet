# SPDX-License-Identifier: MIT
"""Tests for singlet.topic_model()."""

import numpy as np
import pytest
import scipy.sparse as sp
from singlet._topic_model import topic_model


def _make_topic_adata(n_cells=100, n_genes=80, seed=42):
    """Create AnnData with count-like data."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Create data with clear topic structure: 3 cell groups
    group_size = n_cells // 3
    remainder = n_cells - 3 * group_size

    X_parts = []
    # Group 1: high expression in first third of genes
    X1 = rng.poisson(1, size=(group_size, n_genes)).astype(np.float32)
    X1[:, : n_genes // 3] += rng.poisson(10, size=(group_size, n_genes // 3))
    X_parts.append(X1)

    # Group 2: high expression in middle third
    X2 = rng.poisson(1, size=(group_size, n_genes)).astype(np.float32)
    mid_start = n_genes // 3
    mid_end = 2 * n_genes // 3
    X2[:, mid_start:mid_end] += rng.poisson(10, size=(group_size, mid_end - mid_start))
    X_parts.append(X2)

    # Group 3: high expression in last third
    n_group3 = group_size + remainder
    last_start = 2 * n_genes // 3
    X3 = rng.poisson(1, size=(n_group3, n_genes)).astype(np.float32)
    X3[:, last_start:] += rng.poisson(10, size=(n_group3, n_genes - last_start))
    X_parts.append(X3)

    X = np.vstack(X_parts)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    return adata


class TestTopicModelBasic:
    def test_returns_adata(self):
        """Should return the same adata object."""
        adata = _make_topic_adata()
        result = topic_model(adata, n_topics=5)
        assert result is adata

    def test_obsm_topics_created(self):
        """Should create obsm['topics'] with correct shape."""
        adata = _make_topic_adata(n_cells=60, n_genes=40)
        topic_model(adata, n_topics=5)
        assert "topics" in adata.obsm
        assert adata.obsm["topics"].shape == (60, 5)

    def test_uns_gene_weights_created(self):
        """Should create uns['topic_gene_weights'] with correct shape."""
        adata = _make_topic_adata(n_cells=60, n_genes=40)
        topic_model(adata, n_topics=5)
        assert "topic_gene_weights" in adata.uns
        assert adata.uns["topic_gene_weights"].shape == (5, 40)

    def test_topics_sum_to_one(self):
        """Topic proportions per cell should approximately sum to 1."""
        adata = _make_topic_adata(n_cells=50, n_genes=40)
        topic_model(adata, n_topics=3)
        row_sums = adata.obsm["topics"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_topics_nonnegative(self):
        """Topic proportions should be non-negative."""
        adata = _make_topic_adata(n_cells=50, n_genes=40)
        topic_model(adata, n_topics=3)
        assert np.all(adata.obsm["topics"] >= 0)

    def test_gene_weights_nonnegative(self):
        """Gene weights should be non-negative."""
        adata = _make_topic_adata(n_cells=50, n_genes=40)
        topic_model(adata, n_topics=3)
        assert np.all(adata.uns["topic_gene_weights"] >= 0)


class TestTopicModelMethods:
    def test_lda_method(self):
        """LDA method should work."""
        adata = _make_topic_adata(n_cells=50, n_genes=30)
        topic_model(adata, n_topics=3, method="lda")
        assert adata.obsm["topics"].shape == (50, 3)

    def test_nmf_method(self):
        """NMF method should work."""
        adata = _make_topic_adata(n_cells=50, n_genes=30)
        topic_model(adata, n_topics=3, method="nmf")
        assert adata.obsm["topics"].shape == (50, 3)

    def test_invalid_method(self):
        """Should raise ValueError for invalid method."""
        adata = _make_topic_adata(n_cells=30, n_genes=20)
        with pytest.raises(ValueError, match="method must be"):
            topic_model(adata, n_topics=3, method="invalid")


class TestTopicModelParameters:
    def test_custom_n_topics(self):
        """Should respect n_topics parameter."""
        adata = _make_topic_adata(n_cells=50, n_genes=40)
        topic_model(adata, n_topics=10)
        assert adata.obsm["topics"].shape[1] == 10
        assert adata.uns["topic_gene_weights"].shape[0] == 10

    def test_reproducibility(self):
        """Same random_state should give same results."""
        adata1 = _make_topic_adata(n_cells=40, n_genes=30)
        adata2 = _make_topic_adata(n_cells=40, n_genes=30)
        topic_model(adata1, n_topics=3, random_state=42)
        topic_model(adata2, n_topics=3, random_state=42)
        np.testing.assert_array_equal(adata1.obsm["topics"], adata2.obsm["topics"])

    def test_max_iter(self):
        """Should accept max_iter parameter."""
        adata = _make_topic_adata(n_cells=40, n_genes=30)
        # Just ensure it runs without error
        topic_model(adata, n_topics=3, max_iter=10)
        assert "topics" in adata.obsm


class TestTopicModelSparse:
    def test_sparse_input(self):
        """Should handle sparse matrix input."""
        adata = _make_topic_adata(n_cells=50, n_genes=30)
        assert sp.issparse(adata.X)
        topic_model(adata, n_topics=3)
        assert adata.obsm["topics"].shape == (50, 3)

    def test_dense_input(self):
        """Should handle dense matrix input."""
        adata = _make_topic_adata(n_cells=50, n_genes=30)
        adata.X = np.asarray(adata.X.toarray())
        topic_model(adata, n_topics=3)
        assert adata.obsm["topics"].shape == (50, 3)


class TestTopicModelDtype:
    def test_output_dtype(self):
        """Output should be float32."""
        adata = _make_topic_adata(n_cells=50, n_genes=30)
        topic_model(adata, n_topics=3)
        assert adata.obsm["topics"].dtype == np.float32
        assert adata.uns["topic_gene_weights"].dtype == np.float32
