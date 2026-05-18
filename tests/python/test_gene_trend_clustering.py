# SPDX-License-Identifier: MIT
"""Tests for singlet.gene_trend_clustering()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._gene_trend_clustering import gene_trend_clustering


def _make_adata_with_pseudotime(n_cells=150, n_genes=100, seed=42):
    """Create AnnData with pseudotime and gene expression trends."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Create pseudotime
    pseudotime = np.sort(rng.uniform(0, 1, n_cells))

    # Create genes with different temporal patterns
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    t = pseudotime

    # Group 1 (genes 0-24): increasing
    for g in range(25):
        X[:, g] = t * 5 + rng.normal(0, 0.3, n_cells)

    # Group 2 (genes 25-49): decreasing
    for g in range(25, 50):
        X[:, g] = (1 - t) * 5 + rng.normal(0, 0.3, n_cells)

    # Group 3 (genes 50-74): peak in middle
    for g in range(50, 75):
        X[:, g] = np.exp(-((t - 0.5) ** 2) / 0.05) * 5 + rng.normal(0, 0.3, n_cells)

    # Group 4 (genes 75-99): noisy/flat
    for g in range(75, 100):
        X[:, g] = rng.poisson(1, n_cells).astype(np.float32)

    X = np.clip(X, 0, None)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs["dpt_pseudotime"] = pseudotime

    return adata


class TestGeneTrendClusteringBasic:
    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=3, n_top_genes=50)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self):
        """DataFrame should have gene, trend_cluster, and bin columns."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=3, n_bins=20, n_top_genes=50)
        assert "gene" in result.columns
        assert "trend_cluster" in result.columns
        # Should have bin columns
        bin_cols = [c for c in result.columns if c.startswith("bin_")]
        assert len(bin_cols) == 20

    def test_stores_var_key(self):
        """Should store cluster labels in adata.var['trend_cluster']."""
        adata = _make_adata_with_pseudotime()
        gene_trend_clustering(adata, n_clusters=3, n_top_genes=50)
        assert "trend_cluster" in adata.var.columns

    def test_stores_uns_key(self):
        """Should store trends in adata.uns['gene_trends']."""
        adata = _make_adata_with_pseudotime()
        gene_trend_clustering(adata, n_clusters=3, n_top_genes=50)
        assert "gene_trends" in adata.uns
        gt = adata.uns["gene_trends"]
        assert "trends" in gt
        assert "bin_centers" in gt
        assert "genes" in gt

    def test_correct_n_clusters(self):
        """Should produce the requested number of clusters."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=4, n_top_genes=50)
        n_unique = result["trend_cluster"].nunique()
        assert n_unique == 4

    def test_n_genes_in_result(self):
        """Number of genes in result should match n_top_genes."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=3, n_top_genes=30)
        assert len(result) == 30


class TestGeneTrendClusteringMethods:
    def test_kmeans_method(self):
        """Should work with kmeans method."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=3, method="kmeans", n_top_genes=50)
        assert len(result) > 0

    def test_hierarchical_method(self):
        """Should work with hierarchical method."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=3, method="hierarchical", n_top_genes=50)
        assert len(result) > 0
        assert result["trend_cluster"].nunique() == 3


class TestGeneTrendClusteringParams:
    def test_custom_pseudotime_key(self):
        """Should accept custom pseudotime key."""
        adata = _make_adata_with_pseudotime()
        adata.obs["my_time"] = adata.obs["dpt_pseudotime"]
        result = gene_trend_clustering(
            adata, pseudotime_key="my_time", n_clusters=3, n_top_genes=50
        )
        assert len(result) > 0

    def test_different_n_bins(self):
        """Should work with different number of bins."""
        adata = _make_adata_with_pseudotime()
        result = gene_trend_clustering(adata, n_clusters=3, n_bins=10, n_top_genes=50)
        bin_cols = [c for c in result.columns if c.startswith("bin_")]
        assert len(bin_cols) == 10

    def test_uns_trends_shape(self):
        """Stored trends should match n_genes × n_bins."""
        adata = _make_adata_with_pseudotime()
        gene_trend_clustering(adata, n_clusters=3, n_bins=25, n_top_genes=40)
        trends = adata.uns["gene_trends"]["trends"]
        assert trends.shape == (40, 25)

    def test_bin_centers_length(self):
        """Bin centers should have length n_bins."""
        adata = _make_adata_with_pseudotime()
        gene_trend_clustering(adata, n_clusters=3, n_bins=30, n_top_genes=50)
        bin_centers = adata.uns["gene_trends"]["bin_centers"]
        assert len(bin_centers) == 30

    def test_highly_variable_used(self):
        """Should use highly_variable genes when available."""
        adata = _make_adata_with_pseudotime()
        adata.var["highly_variable"] = False
        adata.var.iloc[:30, adata.var.columns.get_loc("highly_variable")] = True
        result = gene_trend_clustering(adata, n_clusters=3, n_top_genes=50)
        # Should use HVGs (up to 30 in this case)
        assert len(result) == 30


class TestGeneTrendClusteringErrors:
    def test_missing_pseudotime(self):
        """Should raise ValueError if pseudotime key missing."""
        adata = _make_adata_with_pseudotime()
        with pytest.raises(ValueError, match="not found in adata.obs"):
            gene_trend_clustering(adata, pseudotime_key="nonexistent")

    def test_invalid_method(self):
        """Should raise ValueError for invalid method."""
        adata = _make_adata_with_pseudotime()
        with pytest.raises(ValueError, match="method must be"):
            gene_trend_clustering(adata, method="spectral")

    def test_n_bins_too_small(self):
        """Should raise ValueError if n_bins < 3."""
        adata = _make_adata_with_pseudotime()
        with pytest.raises(ValueError, match="n_bins must be >= 3"):
            gene_trend_clustering(adata, n_bins=2)

    def test_n_clusters_too_small(self):
        """Should raise ValueError if n_clusters < 2."""
        adata = _make_adata_with_pseudotime()
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            gene_trend_clustering(adata, n_clusters=1)
