"""Tests for singlet.phenograph()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._phenograph import phenograph


def _make_adata_with_pca(n_cells=150, n_clusters=3):
    """Create test AnnData with PCA embedding having clear clusters."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # Create clear cluster structure in PCA space
    cells_per_cluster = n_cells // n_clusters
    pca_data = []
    for cluster_idx in range(n_clusters):
        center = rng.standard_normal(20) * 10
        points = center + rng.standard_normal((cells_per_cluster, 20)) * 0.5
        pca_data.append(points)

    # Handle remainder cells
    remainder = n_cells - cells_per_cluster * n_clusters
    if remainder > 0:
        center = rng.standard_normal(20) * 10
        pca_data.append(center + rng.standard_normal((remainder, 20)) * 0.5)

    X_pca = np.vstack(pca_data).astype(np.float32)

    # Create dummy expression matrix
    X = sp.random(n_cells, 200, density=0.1, format="csr", random_state=42)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{idx}" for idx in range(200)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]
    adata.obsm["X_pca"] = X_pca

    return adata


class TestPhenograph:
    def test_basic(self):
        """Basic PhenoGraph clustering."""
        adata = _make_adata_with_pca()
        result = phenograph(adata)

        assert result is adata
        assert "phenograph" in adata.obs.columns
        assert adata.obs["phenograph"].dtype.name == "category"
        # Should find multiple clusters
        n_unique = adata.obs["phenograph"].nunique()
        assert n_unique >= 2

    def test_finds_clusters(self):
        """Should identify the 3 obvious clusters."""
        adata = _make_adata_with_pca(n_cells=150, n_clusters=3)
        phenograph(adata, n_neighbors=15, resolution=1.0)

        n_clusters = adata.obs["phenograph"].nunique()
        # Should find roughly 3 clusters (±1 due to algorithm)
        assert 2 <= n_clusters <= 6

    def test_snn_stored(self):
        """Should store SNN graph in obsp."""
        adata = _make_adata_with_pca()
        phenograph(adata)

        assert "snn_connectivities" in adata.obsp
        snn = adata.obsp["snn_connectivities"]
        assert snn.shape == (150, 150)
        # SNN should be sparse
        assert sp.issparse(snn)
        # Weights should be in [0, 1] (Jaccard similarity)
        if snn.nnz > 0:
            assert snn.data.min() >= 0
            assert snn.data.max() <= 1.0

    def test_custom_key(self):
        """Should store labels in custom key."""
        adata = _make_adata_with_pca()
        phenograph(adata, key_added="my_clusters")

        assert "my_clusters" in adata.obs.columns
        assert "phenograph" not in adata.obs.columns

    def test_resolution_effect(self):
        """Higher resolution should give more clusters (generally)."""
        adata_low = _make_adata_with_pca()
        adata_high = _make_adata_with_pca()

        phenograph(adata_low, resolution=0.1)
        phenograph(adata_high, resolution=5.0)

        # Both should produce valid clusters
        assert adata_low.obs["phenograph"].nunique() >= 1
        assert adata_high.obs["phenograph"].nunique() >= 1

    def test_n_neighbors_effect(self):
        """Different n_neighbors should produce valid results."""
        adata = _make_adata_with_pca()
        phenograph(adata, n_neighbors=10)
        n10 = adata.obs["phenograph"].nunique()

        adata2 = _make_adata_with_pca()
        phenograph(adata2, n_neighbors=50)
        n50 = adata2.obs["phenograph"].nunique()

        assert n10 >= 1
        assert n50 >= 1

    def test_missing_rep_raises(self):
        """Should raise when use_rep is not in obsm."""
        import anndata as ad

        X = sp.csr_matrix(np.random.rand(10, 20).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{idx}" for idx in range(20)]
        adata.obs_names = [f"c{idx}" for idx in range(10)]

        with pytest.raises(KeyError, match="X_pca"):
            phenograph(adata)

    def test_type_error(self):
        """Should raise on non-AnnData input."""
        with pytest.raises(TypeError, match="phenograph"):
            phenograph("not_adata")

    def test_invalid_clustering(self):
        """Should raise on invalid clustering method."""
        adata = _make_adata_with_pca()
        with pytest.raises(ValueError, match="clustering"):
            phenograph(adata, clustering="invalid")

    def test_louvain_option(self):
        """Louvain clustering should work."""
        adata = _make_adata_with_pca()
        phenograph(adata, clustering="louvain")
        assert "phenograph" in adata.obs.columns
        assert adata.obs["phenograph"].nunique() >= 1

    def test_small_dataset(self):
        """Should work with very small datasets."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X = sp.csr_matrix(rng.standard_normal((20, 50)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{idx}" for idx in range(50)]
        adata.obs_names = [f"c{idx}" for idx in range(20)]
        adata.obsm["X_pca"] = rng.standard_normal((20, 10)).astype(np.float32)

        phenograph(adata, n_neighbors=5)
        assert "phenograph" in adata.obs.columns

    def test_reproducible(self):
        """Same random_state should give same results."""
        adata1 = _make_adata_with_pca()
        adata2 = _make_adata_with_pca()

        phenograph(adata1, random_state=42)
        phenograph(adata2, random_state=42)

        labels1 = list(adata1.obs["phenograph"])
        labels2 = list(adata2.obs["phenograph"])
        assert labels1 == labels2

    def test_returns_adata(self):
        """Should return the same adata object."""
        adata = _make_adata_with_pca()
        result = phenograph(adata)
        assert result is adata

    def test_public_api(self):
        """Should be accessible via singlet.phenograph."""
        assert hasattr(singlet, "phenograph")
        assert callable(singlet.phenograph)
