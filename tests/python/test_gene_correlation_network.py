# SPDX-License-Identifier: MIT
"""Tests for singlet.gene_correlation_network()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._gene_correlation_network import gene_correlation_network


def _make_adata(n_cells=100, n_genes=200):
    """Create test AnnData with correlated genes."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # Create base expression
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)

    # Make first 5 genes correlated with each other
    base_signal = rng.standard_normal(n_cells).astype(np.float32)
    for idx in range(5):
        X[:, idx] = base_signal + 0.2 * rng.standard_normal(n_cells).astype(np.float32)

    # Make genes 5-10 correlated with each other (different group)
    base_signal2 = rng.standard_normal(n_cells).astype(np.float32)
    for idx in range(5, 10):
        X[:, idx] = base_signal2 + 0.2 * rng.standard_normal(n_cells).astype(np.float32)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]
    return adata


class TestGeneCorrelationNetwork:
    def test_basic(self):
        """Basic invocation with specified genes."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(20)]
        corr_df, adj_df = gene_correlation_network(adata, genes=genes, threshold=0.3)

        assert corr_df.shape == (20, 20)
        assert adj_df.shape == (20, 20)
        # Diagonal of correlation should be 1
        np.testing.assert_allclose(np.diag(corr_df.values), 1.0, atol=1e-5)
        # Diagonal of adjacency should be 0 (no self-loops)
        assert all(adj_df.values[idx, idx] == 0 for idx in range(20))

    def test_correlated_genes_detected(self):
        """Highly correlated genes should form network edges."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(10)]
        corr_df, adj_df = gene_correlation_network(adata, genes=genes, threshold=0.5)

        # First 5 genes are highly correlated — should have edges
        within_group1 = adj_df.iloc[:5, :5].values
        # At least some edges within group 1
        assert within_group1.sum() > 0

    def test_threshold_effect(self):
        """Higher threshold should produce fewer edges."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(20)]

        _, adj_low = gene_correlation_network(adata, genes=genes, threshold=0.1)
        _, adj_high = gene_correlation_network(adata, genes=genes, threshold=0.8)

        assert adj_low.values.sum() >= adj_high.values.sum()

    def test_pearson_vs_spearman(self):
        """Both methods should work."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(10)]

        corr_p, _ = gene_correlation_network(adata, genes=genes, method="pearson")
        corr_s, _ = gene_correlation_network(adata, genes=genes, method="spearman")

        assert corr_p.shape == corr_s.shape
        # Should be similar but not identical
        assert not np.allclose(corr_p.values, corr_s.values, atol=0.01)

    def test_hvg_fallback(self):
        """When genes=None, should use HVGs or top by variance."""
        adata = _make_adata(n_genes=50)
        corr_df, adj_df = gene_correlation_network(adata, n_top_genes=30)

        assert corr_df.shape[0] == 30
        assert adj_df.shape[0] == 30

    def test_hvg_annotated(self):
        """When HVGs are annotated, should use them."""
        adata = _make_adata(n_genes=50)
        adata.var["highly_variable"] = False
        adata.var.iloc[:15, adata.var.columns.get_loc("highly_variable")] = True

        corr_df, adj_df = gene_correlation_network(adata)

        assert corr_df.shape[0] == 15

    def test_stores_in_varp(self):
        """Should store results in adata.varp."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(10)]
        gene_correlation_network(adata, genes=genes)

        assert "gene_correlations" in adata.varp
        assert "gene_network" in adata.varp
        assert adata.varp["gene_correlations"].shape == (200, 200)
        assert adata.varp["gene_network"].shape == (200, 200)

    def test_symmetric(self):
        """Correlation and adjacency matrices should be symmetric."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(15)]
        corr_df, adj_df = gene_correlation_network(adata, genes=genes)

        np.testing.assert_allclose(corr_df.values, corr_df.values.T, atol=1e-10)
        np.testing.assert_allclose(adj_df.values, adj_df.values.T, atol=1e-10)

    def test_adjacency_binary(self):
        """Adjacency matrix should contain only 0s and 1s."""
        adata = _make_adata()
        genes = [f"GENE{idx}" for idx in range(10)]
        _, adj_df = gene_correlation_network(adata, genes=genes, threshold=0.3)

        unique_vals = set(adj_df.values.ravel())
        assert unique_vals <= {0.0, 1.0}

    def test_invalid_method(self):
        """Should raise on invalid method."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="method"):
            gene_correlation_network(adata, genes=["GENE0"], method="invalid")

    def test_invalid_threshold(self):
        """Should raise on invalid threshold."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="threshold"):
            gene_correlation_network(adata, genes=["GENE0"], threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            gene_correlation_network(adata, genes=["GENE0"], threshold=1.5)

    def test_no_genes_found(self):
        """Should raise when no specified genes are in the data."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="None of the specified genes"):
            gene_correlation_network(adata, genes=["NONEXISTENT1", "NONEXISTENT2"])

    def test_type_error(self):
        """Should raise on non-AnnData input."""
        with pytest.raises(TypeError, match="gene_correlation_network"):
            gene_correlation_network("not_adata")

    def test_layer(self):
        """Should use specified layer."""
        adata = _make_adata()
        adata.layers["raw"] = adata.X.copy()
        genes = [f"GENE{idx}" for idx in range(10)]
        corr_df, _ = gene_correlation_network(adata, genes=genes, layer="raw")
        assert corr_df.shape == (10, 10)

    def test_layer_not_found(self):
        """Should raise on missing layer."""
        adata = _make_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            gene_correlation_network(adata, genes=["GENE0"], layer="nonexistent")

    def test_dense_input(self):
        """Should work with dense matrices."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 30)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{idx}" for idx in range(30)]
        adata.obs_names = [f"c{idx}" for idx in range(50)]

        genes = [f"G{idx}" for idx in range(10)]
        corr_df, adj_df = gene_correlation_network(adata, genes=genes)
        assert corr_df.shape == (10, 10)

    def test_public_api(self):
        """Should be accessible via singlet.gene_correlation_network."""
        assert hasattr(singlet, "gene_correlation_network")
        assert callable(singlet.gene_correlation_network)
