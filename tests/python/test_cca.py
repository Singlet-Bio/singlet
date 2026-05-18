# SPDX-License-Identifier: MIT
"""Tests for singlet.cca() Canonical Correlation Analysis."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._cca import cca


def _make_cca_adata(n_cells=100, n_genes=200, n_batches=2):
    """Create AnnData with batches and shared correlation structure."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells_per_batch = n_cells // n_batches

    # Shared latent structure across batches
    latent_dim = 10
    shared_factors = rng.standard_normal((latent_dim, n_genes))

    all_X = []
    batch_labels = []
    obs_names = []
    cell_idx = 0

    for b in range(n_batches):
        # Each batch has shared structure + batch-specific noise
        loadings = rng.standard_normal((n_cells_per_batch, latent_dim))
        batch_effect = rng.standard_normal((1, n_genes)) * 2.0
        X_batch = loadings @ shared_factors + batch_effect + rng.standard_normal(
            (n_cells_per_batch, n_genes)
        ) * 0.5
        # Make non-negative (like count data)
        X_batch = np.abs(X_batch)
        all_X.append(X_batch)
        batch_labels.extend([f"batch_{b}"] * n_cells_per_batch)
        obs_names.extend([f"cell_{cell_idx + i}" for i in range(n_cells_per_batch)])
        cell_idx += n_cells_per_batch

    X = np.vstack(all_X).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index(obs_names)
    adata.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])
    adata.obs["batch"] = pd.Categorical(batch_labels)

    return adata


class TestCCA:
    def test_two_datasets_basic(self):
        """CCA between two AnnData objects works, stores X_cca."""
        adata = _make_cca_adata(n_cells=100, n_genes=200, n_batches=1)
        adata2 = _make_cca_adata(n_cells=100, n_genes=200, n_batches=1)
        # Ensure they share gene names
        adata2.var_names = adata.var_names.copy()
        adata2.obs_names = pd.Index([f"cell2_{i}" for i in range(100)])

        result = cca(adata, adata2)
        assert "X_cca" in result.obsm
        assert result.obsm["X_cca"].shape[0] == 200

    def test_batch_key_basic(self):
        """CCA using batch_key splits correctly and stores embedding."""
        adata = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)
        result = cca(adata, batch_key="batch")
        assert result is adata
        assert "X_cca" in adata.obsm

    def test_output_shape(self):
        """X_cca shape is (n_cells, n_components)."""
        adata = _make_cca_adata(n_cells=120, n_genes=200, n_batches=2)
        n_components = 20
        cca(adata, batch_key="batch", n_components=n_components)
        assert adata.obsm["X_cca"].shape == (120, n_components)

    def test_n_components_respected(self):
        """Smaller n_components gives smaller embedding."""
        adata1 = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)
        adata2 = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)

        cca(adata1, batch_key="batch", n_components=10)
        cca(adata2, batch_key="batch", n_components=25)

        assert adata1.obsm["X_cca"].shape[1] == 10
        assert adata2.obsm["X_cca"].shape[1] == 25

    def test_no_nans(self):
        """Output has no NaN/Inf values."""
        adata = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)
        cca(adata, batch_key="batch")
        Z = adata.obsm["X_cca"]
        assert np.all(np.isfinite(Z))

    def test_type_error(self):
        """Non-AnnData raises TypeError."""
        with pytest.raises(TypeError, match="cca"):
            cca("not_adata", batch_key="batch")

    def test_type_error_adata2(self):
        """Non-AnnData for adata2 raises TypeError."""
        adata = _make_cca_adata(n_cells=50, n_genes=100, n_batches=1)
        with pytest.raises(TypeError, match="adata2"):
            cca(adata, adata2="not_adata")

    def test_neither_adata2_nor_batch_key(self):
        """Raises ValueError when neither adata2 nor batch_key is provided."""
        adata = _make_cca_adata(n_cells=50, n_genes=100, n_batches=1)
        with pytest.raises(ValueError, match="adata2.*batch_key|batch_key.*adata2"):
            cca(adata)

    def test_missing_batch_key(self):
        """Raises ValueError for missing batch column."""
        adata = _make_cca_adata(n_cells=50, n_genes=100, n_batches=1)
        with pytest.raises(ValueError, match="nonexistent"):
            cca(adata, batch_key="nonexistent")

    def test_reproducible(self):
        """Same random_state gives same results."""
        adata1 = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)
        adata2 = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)

        cca(adata1, batch_key="batch", random_state=42)
        cca(adata2, batch_key="batch", random_state=42)

        np.testing.assert_allclose(
            adata1.obsm["X_cca"], adata2.obsm["X_cca"]
        )

    def test_stores_uns(self):
        """adata.uns['cca'] has params dict."""
        adata = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)
        cca(adata, batch_key="batch", n_components=15, n_features=100)
        assert "cca" in adata.uns
        params = adata.uns["cca"]["params"]
        assert params["n_components"] == 15
        assert params["n_features"] == 100
        assert params["batch_key"] == "batch"
        assert params["n_batches"] == 2

    def test_stores_uns_two_datasets(self):
        """Merged result .uns['cca'] has params when using adata2 mode."""
        adata = _make_cca_adata(n_cells=60, n_genes=200, n_batches=1)
        adata2 = _make_cca_adata(n_cells=60, n_genes=200, n_batches=1)
        adata2.var_names = adata.var_names.copy()
        adata2.obs_names = pd.Index([f"cell2_{i}" for i in range(60)])

        result = cca(adata, adata2, n_components=10, n_features=150)
        assert "cca" in result.uns
        params = result.uns["cca"]["params"]
        assert params["n_components"] == 10
        assert params["n_features"] == 150
        assert "n_shared_genes" in params
        assert "n_selected_genes" in params

    def test_public_api(self):
        """singlet.cca is callable from the public namespace."""
        assert hasattr(singlet, "cca")
        assert callable(singlet.cca)

    def test_sparse_input(self):
        """Works with sparse X matrix."""
        import anndata as ad

        rng = np.random.default_rng(7)
        n_cells, n_genes = 100, 200

        X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=7)
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index([f"cell_{i}" for i in range(n_cells)])
        adata.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])
        adata.obs["batch"] = pd.Categorical(
            ["A"] * (n_cells // 2) + ["B"] * (n_cells // 2)
        )

        cca(adata, batch_key="batch", n_components=10)
        assert "X_cca" in adata.obsm
        assert adata.obsm["X_cca"].shape == (n_cells, 10)
        assert np.all(np.isfinite(adata.obsm["X_cca"]))

    def test_merged_obs_names(self):
        """When merging two datasets, obs_names are preserved."""
        import anndata as ad

        rng = np.random.default_rng(99)
        n_genes = 150
        n_cells = 50

        adata1 = ad.AnnData(X=rng.random((n_cells, n_genes)).astype(np.float32))
        adata1.obs_names = pd.Index([f"sample_A_{i}" for i in range(n_cells)])
        adata1.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])

        adata2 = ad.AnnData(X=rng.random((n_cells, n_genes)).astype(np.float32))
        adata2.obs_names = pd.Index([f"sample_B_{i}" for i in range(n_cells)])
        adata2.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])

        result = cca(adata1, adata2, n_components=10)
        assert result.n_obs == 100
        # Check that obs_names from both datasets appear in the merged result
        for name in adata1.obs_names:
            assert name in result.obs_names
        for name in adata2.obs_names:
            assert name in result.obs_names

    def test_n_features(self):
        """n_features parameter limits gene selection."""
        adata_small = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)
        adata_large = _make_cca_adata(n_cells=100, n_genes=200, n_batches=2)

        # With very few features, still runs but may produce different results
        cca(adata_small, batch_key="batch", n_features=50, n_components=10)
        cca(adata_large, batch_key="batch", n_features=200, n_components=10)

        assert adata_small.obsm["X_cca"].shape == (100, 10)
        assert adata_large.obsm["X_cca"].shape == (100, 10)
        # Different n_features should give different embeddings
        assert not np.allclose(adata_small.obsm["X_cca"], adata_large.obsm["X_cca"])

    def test_single_batch_raises(self):
        """batch_key with only 1 unique value raises ValueError."""
        adata = _make_cca_adata(n_cells=50, n_genes=100, n_batches=1)
        adata.obs["batch"] = pd.Categorical(["only_one"] * 50)
        with pytest.raises(ValueError, match="fewer than 2"):
            cca(adata, batch_key="batch")

    def test_three_batches(self):
        """CCA works with more than 2 batches."""
        adata = _make_cca_adata(n_cells=150, n_genes=200, n_batches=3)
        cca(adata, batch_key="batch", n_components=10)
        assert adata.obsm["X_cca"].shape == (150, 10)
        assert np.all(np.isfinite(adata.obsm["X_cca"]))
