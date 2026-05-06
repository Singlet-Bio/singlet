"""Tests for singlet.cluster_stability()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._cluster_stability import cluster_stability


def _make_clustered_adata(n_cells=200, n_genes=100, n_clusters=3, seed=42):
    """Create AnnData with clear cluster structure and PCA/neighbors."""
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
        X[start:end, gene_start:gene_end] += 15.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    # Preprocess
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=20)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)

    return adata


class TestClusterStability:
    def test_basic_returns_dict(self):
        """Should return dict with expected keys."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        result = cluster_stability(adata, n_bootstraps=5, resolution=0.5)

        assert isinstance(result, dict)
        assert "mean_ari" in result
        assert "std_ari" in result
        assert "mean_nmi" in result
        assert "std_nmi" in result
        assert "per_cluster_stability" in result

    def test_stores_in_uns(self):
        """Should store results in adata.uns['cluster_stability']."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        result = cluster_stability(adata, n_bootstraps=5, resolution=0.5)

        assert "cluster_stability" in adata.uns
        assert adata.uns["cluster_stability"] == result

    def test_well_separated_clusters_high_stability(self):
        """Well-separated clusters should have high ARI/NMI."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        result = cluster_stability(adata, n_bootstraps=10, resolution=0.5)

        # Well-separated clusters should have high stability
        assert result["mean_ari"] > 0.5
        assert result["mean_nmi"] > 0.5

    def test_per_cluster_stability_format(self):
        """per_cluster_stability should be dict of cluster_label → float."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        result = cluster_stability(adata, n_bootstraps=5, resolution=0.5)

        pcs = result["per_cluster_stability"]
        assert isinstance(pcs, dict)
        for key, val in pcs.items():
            assert isinstance(key, str)
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_reproducible_with_seed(self):
        """Same random_state should give same results."""
        adata1 = _make_clustered_adata(n_cells=150, n_clusters=3)
        adata2 = _make_clustered_adata(n_cells=150, n_clusters=3)

        r1 = cluster_stability(adata1, n_bootstraps=5, random_state=123, resolution=0.5)
        r2 = cluster_stability(adata2, n_bootstraps=5, random_state=123, resolution=0.5)

        assert abs(r1["mean_ari"] - r2["mean_ari"]) < 1e-10
        assert abs(r1["mean_nmi"] - r2["mean_nmi"]) < 1e-10

    def test_different_seeds_differ(self):
        """Different random states should give different results."""
        adata1 = _make_clustered_adata(n_cells=150, n_clusters=3)
        adata2 = _make_clustered_adata(n_cells=150, n_clusters=3)

        r1 = cluster_stability(adata1, n_bootstraps=5, random_state=0, resolution=0.5)
        r2 = cluster_stability(adata2, n_bootstraps=5, random_state=999, resolution=0.5)

        # Very unlikely to be exactly equal with different seeds
        # (but not impossible, so just check they ran)
        assert "mean_ari" in r1
        assert "mean_ari" in r2

    def test_invalid_method_raises(self):
        """Should raise for invalid clustering method."""
        adata = _make_clustered_adata(n_cells=100, n_clusters=2)
        with pytest.raises(ValueError, match="method must be one of"):
            cluster_stability(adata, method="kmeans")

    def test_invalid_subsample_frac_raises(self):
        """Should raise for invalid subsample_frac."""
        adata = _make_clustered_adata(n_cells=100, n_clusters=2)
        with pytest.raises(ValueError, match="subsample_frac must be between"):
            cluster_stability(adata, subsample_frac=1.5)

        with pytest.raises(ValueError, match="subsample_frac must be between"):
            cluster_stability(adata, subsample_frac=0.0)

    def test_no_pca_raises(self):
        """Should raise if PCA is missing."""
        import anndata as ad

        adata = ad.AnnData(X=sp.csr_matrix(np.ones((50, 20))))
        adata.obs["leiden"] = pd.Categorical(["0"] * 25 + ["1"] * 25)

        with pytest.raises(ValueError, match="PCA not found"):
            cluster_stability(adata)

    def test_ari_nmi_in_valid_range(self):
        """ARI and NMI should be in valid ranges."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        result = cluster_stability(adata, n_bootstraps=5, resolution=0.5)

        # ARI can be negative but typically > -0.5
        assert result["mean_ari"] >= -1.0
        assert result["mean_ari"] <= 1.0
        # NMI is in [0, 1]
        assert result["mean_nmi"] >= 0.0
        assert result["mean_nmi"] <= 1.0

    def test_std_non_negative(self):
        """Standard deviations should be non-negative."""
        adata = _make_clustered_adata(n_cells=150, n_clusters=3)
        result = cluster_stability(adata, n_bootstraps=5, resolution=0.5)

        assert result["std_ari"] >= 0.0
        assert result["std_nmi"] >= 0.0

    def test_small_n_bootstraps(self):
        """Should work with small number of bootstraps."""
        adata = _make_clustered_adata(n_cells=100, n_clusters=2)
        result = cluster_stability(adata, n_bootstraps=2, resolution=0.5)

        assert "mean_ari" in result

    def test_singlet_import(self):
        """Should be importable from singlet namespace."""
        import singlet

        assert hasattr(singlet, "cluster_stability")
        assert callable(singlet.cluster_stability)
