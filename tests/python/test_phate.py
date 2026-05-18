# SPDX-License-Identifier: MIT
"""Tests for singlet.phate."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import singlet


@pytest.fixture
def adata_with_pca():
    """Create test AnnData with PCA embedding."""
    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 80

    # Simulate branching trajectory
    branch1 = rng.normal(loc=[0, 0, 0], scale=0.5, size=(50, 3))
    branch1[:, 0] += np.linspace(0, 5, 50)

    branch2 = rng.normal(loc=[0, 0, 0], scale=0.5, size=(50, 3))
    branch2[:, 0] += np.linspace(0, 5, 50)
    branch2[:, 1] += np.linspace(0, 3, 50)

    latent = np.vstack([branch1, branch2])

    # Project to gene space
    projection = rng.normal(0, 1, size=(3, n_genes))
    X = (latent @ projection + rng.normal(0, 0.1, size=(n_cells, n_genes))).astype(
        np.float32
    )
    X = np.abs(X)  # Make non-negative

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=15, random_state=42)
    adata.obsm["X_pca"] = pca.fit_transform(X)

    return adata


@pytest.fixture
def adata_no_pca():
    """AnnData without PCA (will use raw X)."""
    rng = np.random.default_rng(7)
    n_cells = 60
    n_genes = 30
    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata_sparse():
    """AnnData with sparse matrix and no PCA."""
    rng = np.random.default_rng(12)
    n_cells = 60
    n_genes = 30
    X = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return AnnData(X=csr_matrix(X), obs=obs, var=var)


class TestPhate:
    def test_basic(self, adata_with_pca):
        """Test basic functionality."""
        result = singlet.phate(adata_with_pca)
        assert result is adata_with_pca
        assert "X_phate" in adata_with_pca.obsm

    def test_output_shape_2d(self, adata_with_pca):
        """Default is 2 components."""
        singlet.phate(adata_with_pca)
        embedding = adata_with_pca.obsm["X_phate"]
        assert embedding.shape == (adata_with_pca.n_obs, 2)

    def test_output_shape_3d(self, adata_with_pca):
        """Test 3 components."""
        singlet.phate(adata_with_pca, n_components=3)
        embedding = adata_with_pca.obsm["X_phate"]
        assert embedding.shape == (adata_with_pca.n_obs, 3)

    def test_no_nan(self, adata_with_pca):
        """Embedding should not contain NaN."""
        singlet.phate(adata_with_pca)
        embedding = adata_with_pca.obsm["X_phate"]
        assert not np.any(np.isnan(embedding))

    def test_no_inf(self, adata_with_pca):
        """Embedding should not contain Inf."""
        singlet.phate(adata_with_pca)
        embedding = adata_with_pca.obsm["X_phate"]
        assert not np.any(np.isinf(embedding))

    def test_params_stored(self, adata_with_pca):
        """Parameters should be stored in uns."""
        singlet.phate(adata_with_pca, knn=10, decay=20)
        params = adata_with_pca.uns["phate_params"]
        assert params["knn"] == 10
        assert params["decay"] == 20
        assert params["n_components"] == 2

    def test_copy_mode(self, adata_with_pca):
        """Test copy doesn't modify original."""
        result = singlet.phate(adata_with_pca, copy=True)
        assert result is not adata_with_pca
        assert "X_phate" in result.obsm
        assert "X_phate" not in adata_with_pca.obsm

    def test_no_pca_uses_raw(self, adata_no_pca):
        """Without PCA, should use raw X."""
        singlet.phate(adata_no_pca)
        assert "X_phate" in adata_no_pca.obsm
        assert adata_no_pca.obsm["X_phate"].shape == (adata_no_pca.n_obs, 2)

    def test_sparse_input(self, adata_sparse):
        """Test with sparse matrix input."""
        singlet.phate(adata_sparse)
        assert "X_phate" in adata_sparse.obsm
        assert not np.any(np.isnan(adata_sparse.obsm["X_phate"]))

    def test_custom_use_rep(self, adata_with_pca):
        """Test with explicit use_rep."""
        singlet.phate(adata_with_pca, use_rep="X_pca")
        assert "X_phate" in adata_with_pca.obsm

    def test_invalid_use_rep(self, adata_with_pca):
        """Test error on missing representation."""
        with pytest.raises(KeyError, match="not found"):
            singlet.phate(adata_with_pca, use_rep="X_nonexistent")

    def test_fixed_t(self, adata_with_pca):
        """Test with fixed diffusion time."""
        singlet.phate(adata_with_pca, t=5)
        params = adata_with_pca.uns["phate_params"]
        assert params["t"] == 5

    def test_auto_t(self, adata_with_pca):
        """Test auto diffusion time selection."""
        singlet.phate(adata_with_pca, t="auto")
        params = adata_with_pca.uns["phate_params"]
        assert isinstance(params["t"], int)
        assert params["t"] >= 1

    def test_different_knn(self, adata_with_pca):
        """Different knn should give different embeddings."""
        adata1 = adata_with_pca.copy()
        adata2 = adata_with_pca.copy()
        singlet.phate(adata1, knn=3, t=5, random_state=0)
        singlet.phate(adata2, knn=15, t=5, random_state=0)
        assert not np.allclose(adata1.obsm["X_phate"], adata2.obsm["X_phate"])

    def test_reproducibility(self, adata_with_pca):
        """Same random_state gives same results."""
        adata1 = adata_with_pca.copy()
        adata2 = adata_with_pca.copy()
        singlet.phate(adata1, random_state=42, t=5)
        singlet.phate(adata2, random_state=42, t=5)
        np.testing.assert_array_almost_equal(
            adata1.obsm["X_phate"], adata2.obsm["X_phate"]
        )
