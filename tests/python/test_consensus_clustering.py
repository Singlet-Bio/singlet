"""Tests for singlet.consensus_clustering()."""

import numpy as np
import pytest
import scipy.sparse as sp
from singlet._consensus_clustering import consensus_clustering


def _make_clustered_adata(n_cells=150, n_genes=100, n_clusters=3, seed=42):
    """Create AnnData with clear cluster structure and neighbors."""
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(seed)

    # Create data with clear clusters
    cells_per_cluster = n_cells // n_clusters
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    for k in range(n_clusters):
        start = k * cells_per_cluster
        end = start + cells_per_cluster
        gene_start = k * (n_genes // n_clusters)
        gene_end = gene_start + (n_genes // n_clusters)
        X[start:end, gene_start:gene_end] += 20.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    # Preprocess
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=20)
    sc.pp.neighbors(adata)

    return adata


class TestConsensusClusteringBasic:
    def test_returns_adata(self):
        """Should return the input AnnData."""
        adata = _make_clustered_adata()
        result = consensus_clustering(adata, n_runs=3, n_resolutions=2)
        assert result is adata

    def test_stores_obs_key(self):
        """Should store cluster labels in adata.obs['consensus_clusters']."""
        adata = _make_clustered_adata()
        consensus_clustering(adata, n_runs=3, n_resolutions=2)
        assert "consensus_clusters" in adata.obs.columns
        assert len(adata.obs["consensus_clusters"]) == adata.n_obs

    def test_stores_consensus_matrix(self):
        """Should store co-clustering matrix in adata.uns."""
        adata = _make_clustered_adata()
        consensus_clustering(adata, n_runs=3, n_resolutions=2)
        assert "consensus_matrix" in adata.uns
        matrix = adata.uns["consensus_matrix"]
        assert matrix.shape == (adata.n_obs, adata.n_obs)

    def test_consensus_matrix_values(self):
        """Consensus matrix should be in [0, 1] and symmetric."""
        adata = _make_clustered_adata()
        consensus_clustering(adata, n_runs=5, n_resolutions=2)
        matrix = adata.uns["consensus_matrix"]
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)
        # Symmetric
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
        # Diagonal should be 1 (cell always co-clusters with itself)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)

    def test_finds_clear_clusters(self):
        """With well-separated data, should find roughly the right clusters."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        consensus_clustering(adata, n_runs=5, n_resolutions=3)
        labels = adata.obs["consensus_clusters"]
        n_unique = labels.nunique()
        # Should find at least 2 clusters (data has 3 clear ones)
        assert n_unique >= 2

    def test_categorical_labels(self):
        """Cluster labels should be categorical."""
        adata = _make_clustered_adata()
        consensus_clustering(adata, n_runs=3, n_resolutions=2)
        assert hasattr(adata.obs["consensus_clusters"], "cat")


class TestConsensusClusteringParams:
    def test_different_random_state(self):
        """Different random_state should give potentially different results."""
        adata1 = _make_clustered_adata()
        adata2 = _make_clustered_adata()
        consensus_clustering(adata1, n_runs=3, n_resolutions=2, random_state=0)
        consensus_clustering(adata2, n_runs=3, n_resolutions=2, random_state=99)
        # Matrices may differ (not guaranteed, but structure should be similar)
        assert "consensus_matrix" in adata1.uns
        assert "consensus_matrix" in adata2.uns

    def test_single_resolution(self):
        """Should work with n_resolutions=1."""
        adata = _make_clustered_adata()
        consensus_clustering(adata, n_runs=3, n_resolutions=1)
        assert "consensus_clusters" in adata.obs.columns

    def test_many_runs(self):
        """Should work with more runs."""
        adata = _make_clustered_adata(n_cells=60, n_genes=50)
        consensus_clustering(adata, n_runs=10, n_resolutions=2)
        assert "consensus_clusters" in adata.obs.columns


class TestConsensusClusteringErrors:
    def test_invalid_method(self):
        """Should raise ValueError for unknown method."""
        adata = _make_clustered_adata()
        with pytest.raises(ValueError, match="method must be"):
            consensus_clustering(adata, method="invalid")

    def test_invalid_resolution_range(self):
        """Should raise ValueError for bad resolution range."""
        adata = _make_clustered_adata()
        with pytest.raises(ValueError, match="resolution_range"):
            consensus_clustering(adata, resolution_range=(2.0, 0.5))

    def test_missing_neighbors(self):
        """Should raise ValueError if neighbors not computed."""
        import anndata as ad

        adata = ad.AnnData(X=sp.csr_matrix(np.ones((10, 5))))
        with pytest.raises(ValueError, match="Neighbor graph not found"):
            consensus_clustering(adata)
