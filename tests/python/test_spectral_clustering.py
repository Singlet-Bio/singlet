"""Tests for singlet.spectral_clustering()."""

import numpy as np
import pytest
import scipy.sparse as sp
from singlet._spectral_clustering import spectral_clustering


def _make_adata(n_cells=120, n_genes=80, n_clusters=3, seed=42):
    """Create AnnData with PCA and clear clusters."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Create data with distinct clusters
    cells_per = n_cells // n_clusters
    X = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)

    for k in range(n_clusters):
        start = k * cells_per
        end = start + cells_per
        gene_start = k * (n_genes // n_clusters)
        gene_end = gene_start + (n_genes // n_clusters)
        X[start:end, gene_start:gene_end] += 15.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    # Add PCA representation
    from sklearn.decomposition import PCA

    pca = PCA(n_components=20, random_state=0)
    adata.obsm["X_pca"] = pca.fit_transform(X)

    return adata


class TestSpectralClusteringBasic:
    def test_returns_adata(self):
        """Should return the input AnnData."""
        adata = _make_adata()
        result = spectral_clustering(adata, n_clusters=3)
        assert result is adata

    def test_stores_obs_key(self):
        """Should store cluster labels in adata.obs."""
        adata = _make_adata()
        spectral_clustering(adata, n_clusters=3)
        assert "spectral_clusters" in adata.obs.columns
        assert len(adata.obs["spectral_clusters"]) == adata.n_obs

    def test_correct_n_clusters(self):
        """Should produce the requested number of clusters."""
        adata = _make_adata()
        spectral_clustering(adata, n_clusters=4)
        n_unique = adata.obs["spectral_clusters"].nunique()
        assert n_unique == 4

    def test_categorical_output(self):
        """Cluster labels should be categorical."""
        adata = _make_adata()
        spectral_clustering(adata, n_clusters=3)
        assert hasattr(adata.obs["spectral_clusters"], "cat")

    def test_finds_true_clusters(self):
        """With clear clusters, should approximately recover them."""
        adata = _make_adata(n_cells=120, n_clusters=3)
        spectral_clustering(adata, n_clusters=3)
        labels = adata.obs["spectral_clusters"].values
        # Each ground-truth group should be mostly in one cluster
        for k in range(3):
            start = k * 40
            end = start + 40
            group_labels = labels[start:end]
            # Majority label should cover most cells
            most_common = max(set(group_labels), key=list(group_labels).count)
            purity = list(group_labels).count(most_common) / 40
            assert purity > 0.7


class TestSpectralClusteringAffinity:
    def test_rbf_affinity(self):
        """Should work with RBF affinity."""
        adata = _make_adata()
        spectral_clustering(adata, n_clusters=3, affinity="rbf")
        assert "spectral_clusters" in adata.obs.columns
        assert adata.obs["spectral_clusters"].nunique() == 3

    def test_nn_affinity(self):
        """Should work with nearest_neighbors affinity."""
        adata = _make_adata()
        spectral_clustering(adata, n_clusters=3, affinity="nearest_neighbors", n_neighbors=5)
        assert "spectral_clusters" in adata.obs.columns


class TestSpectralClusteringParams:
    def test_different_n_neighbors(self):
        """Should accept different n_neighbors values."""
        adata = _make_adata()
        spectral_clustering(adata, n_clusters=3, n_neighbors=5)
        labels_5 = adata.obs["spectral_clusters"].copy()

        spectral_clustering(adata, n_clusters=3, n_neighbors=20)
        labels_20 = adata.obs["spectral_clusters"].copy()

        # Results may differ with different n_neighbors
        assert len(labels_5) == len(labels_20)

    def test_reproducibility(self):
        """Same random_state should give same results."""
        adata1 = _make_adata()
        adata2 = _make_adata()
        spectral_clustering(adata1, n_clusters=3, random_state=42)
        spectral_clustering(adata2, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(
            adata1.obs["spectral_clusters"].values,
            adata2.obs["spectral_clusters"].values,
        )

    def test_custom_use_rep(self):
        """Should work with a custom representation key."""
        adata = _make_adata()
        adata.obsm["X_custom"] = adata.obsm["X_pca"][:, :5]
        spectral_clustering(adata, n_clusters=3, use_rep="X_custom")
        assert "spectral_clusters" in adata.obs.columns


class TestSpectralClusteringErrors:
    def test_invalid_affinity(self):
        """Should raise ValueError for invalid affinity."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="affinity must be"):
            spectral_clustering(adata, n_clusters=3, affinity="cosine")

    def test_missing_use_rep(self):
        """Should raise ValueError if use_rep not in obsm."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            spectral_clustering(adata, n_clusters=3, use_rep="X_umap")

    def test_n_clusters_too_small(self):
        """Should raise ValueError if n_clusters < 2."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            spectral_clustering(adata, n_clusters=1)

    def test_n_clusters_too_large(self):
        """Should raise ValueError if n_clusters > n_cells."""
        adata = _make_adata(n_cells=30)
        with pytest.raises(ValueError, match="n_clusters.*must be <="):
            spectral_clustering(adata, n_clusters=50)

    def test_n_neighbors_too_small(self):
        """Should raise ValueError if n_neighbors < 2."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="n_neighbors must be >= 2"):
            spectral_clustering(adata, n_clusters=3, n_neighbors=1)
