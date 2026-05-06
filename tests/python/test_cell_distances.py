"""Tests for singlet.cell_distances()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._cell_distances import cell_distances


def _make_clustered_adata(n_cells=120, n_dims=20, n_clusters=3, seed=42):
    """Create AnnData with PCA and cluster labels."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Create well-separated clusters in PCA space
    cells_per_cluster = n_cells // n_clusters
    pca = np.zeros((n_cells, n_dims), dtype=np.float32)

    for k in range(n_clusters):
        start = k * cells_per_cluster
        end = start + cells_per_cluster
        center = np.zeros(n_dims)
        center[k] = 10.0  # Separate clusters along different dimensions
        pca[start:end] = rng.normal(loc=center, scale=0.5, size=(cells_per_cluster, n_dims))

    adata = ad.AnnData(
        X=sp.csr_matrix(rng.poisson(2, (n_cells, 100)).astype(np.float32))
    )
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(100)]
    adata.obsm["X_pca"] = pca
    adata.obs["cluster"] = pd.Categorical(
        [f"C{i // cells_per_cluster}" for i in range(n_cells)]
    )

    return adata


class TestCellDistancesGroupby:
    """Test suite for cell_distances with groupby (centroid mode)."""

    def test_basic_output(self):
        """Test basic centroid distance output."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster")

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        assert list(result.index) == ["C0", "C1", "C2"]
        assert list(result.columns) == ["C0", "C1", "C2"]

    def test_diagonal_zero(self):
        """Test that self-distances are zero."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster")

        np.testing.assert_allclose(np.diag(result.values), 0.0, atol=1e-10)

    def test_symmetric(self):
        """Test that distance matrix is symmetric."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster")

        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)

    def test_positive_off_diagonal(self):
        """Test that off-diagonal distances are positive."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster")

        n_groups = result.shape[0]
        for idx in range(n_groups):
            for jdx in range(n_groups):
                if idx != jdx:
                    assert result.iloc[idx, jdx] > 0

    def test_euclidean_metric(self):
        """Test euclidean distances are correct."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster", metric="euclidean")

        # All clusters should be well-separated (center distance ~10)
        off_diag = result.values[np.triu_indices(3, k=1)]
        assert np.all(off_diag > 5.0)

    def test_cosine_metric(self):
        """Test cosine distances."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster", metric="cosine")

        assert isinstance(result, pd.DataFrame)
        # Cosine distances are bounded [0, 2]
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 2.0)

    def test_correlation_metric(self):
        """Test correlation distances."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, groupby="cluster", metric="correlation")

        assert isinstance(result, pd.DataFrame)
        assert np.all(result.values >= 0)

    def test_missing_groupby_raises(self):
        """Test that missing groupby key raises KeyError."""
        adata = _make_clustered_adata()

        with pytest.raises(KeyError, match="nonexistent"):
            cell_distances(adata, groupby="nonexistent")

    def test_missing_rep_raises(self):
        """Test that missing representation raises KeyError."""
        adata = _make_clustered_adata()

        with pytest.raises(KeyError, match="X_nonexistent"):
            cell_distances(adata, use_rep="X_nonexistent")


class TestCellDistancesCellLevel:
    """Test suite for cell_distances without groupby (kNN mode)."""

    def test_basic_output(self):
        """Test basic kNN distance output."""
        adata = _make_clustered_adata()

        result = cell_distances(adata)

        assert sp.issparse(result)
        assert result.shape == (120, 120)

    def test_stored_in_obsp(self):
        """Test that result is stored in adata.obsp."""
        adata = _make_clustered_adata()

        cell_distances(adata, metric="euclidean")

        assert "distances_euclidean" in adata.obsp

    def test_cosine_stored(self):
        """Test cosine metric stored correctly."""
        adata = _make_clustered_adata()

        cell_distances(adata, metric="cosine")

        assert "distances_cosine" in adata.obsp

    def test_n_neighbors(self):
        """Test n_neighbors controls sparsity."""
        adata = _make_clustered_adata()

        result = cell_distances(adata, n_neighbors=5)

        # Each row should have exactly 5 non-zero entries
        nnz_per_row = np.diff(result.indptr)
        assert np.all(nnz_per_row == 5)

    def test_distances_non_negative(self):
        """Test that all distances are non-negative."""
        adata = _make_clustered_adata()

        result = cell_distances(adata)

        assert np.all(result.data >= 0)

    def test_no_self_distances(self):
        """Test that diagonal is zero (no self-connections)."""
        adata = _make_clustered_adata()

        result = cell_distances(adata)

        # Diagonal should be zero (not stored in sparse)
        diag = result.diagonal()
        np.testing.assert_allclose(diag, 0.0, atol=1e-10)

    def test_small_dataset(self):
        """Test with very small dataset."""
        import anndata as ad

        rng = np.random.default_rng(7)
        adata = ad.AnnData(X=sp.csr_matrix(rng.poisson(3, (10, 50)).astype(np.float32)))
        adata.obsm["X_pca"] = rng.standard_normal((10, 5)).astype(np.float32)

        result = cell_distances(adata, n_neighbors=3)

        assert result.shape == (10, 10)
        nnz_per_row = np.diff(result.indptr)
        assert np.all(nnz_per_row == 3)

    def test_invalid_metric_raises(self):
        """Test that invalid metric raises ValueError."""
        adata = _make_clustered_adata()

        with pytest.raises(ValueError, match="metric must be"):
            cell_distances(adata, metric="manhattan")

    def test_different_use_rep(self):
        """Test with alternative representation."""
        adata = _make_clustered_adata()
        adata.obsm["X_custom"] = adata.obsm["X_pca"][:, :5]

        result = cell_distances(adata, use_rep="X_custom")

        assert result.shape == (120, 120)
