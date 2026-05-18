# SPDX-License-Identifier: MIT
"""Tests for singlet.weighted_nearest_neighbors()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData
from scipy import sparse


def _make_multimodal_adata(n_cells=100, n_pcs=20, n_protein=15, seed=42):
    """Create test AnnData with two modalities in obsm."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, (n_cells, 200)).astype(np.float32)
    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(200)]

    # Simulate PCA embeddings
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, n_pcs)).astype(
        np.float32
    )
    # Simulate protein (ADT) embeddings
    adata.obsm["X_protein"] = rng.standard_normal((n_cells, n_protein)).astype(
        np.float32
    )
    return adata


class TestWeightedNearestNeighbors:
    def test_basic_two_modalities(self):
        adata = _make_multimodal_adata()
        result = singlet.weighted_nearest_neighbors(
            adata, modalities=["X_pca", "X_protein"], n_neighbors=10
        )
        assert result is adata
        assert "wnn_connectivities" in adata.obsp
        assert "wnn_weights" in adata.obsm
        assert adata.obsm["wnn_weights"].shape == (100, 2)
        assert sparse.issparse(adata.obsp["wnn_connectivities"])
        # Weights should sum to 1 per cell
        weight_sums = adata.obsm["wnn_weights"].sum(axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-5)

    def test_fixed_weights(self):
        adata = _make_multimodal_adata()
        result = singlet.weighted_nearest_neighbors(
            adata,
            modalities=["X_pca", "X_protein"],
            weights=[0.7, 0.3],
            n_neighbors=10,
        )
        assert result is adata
        # With fixed weights, all cells should have same weights
        weights = adata.obsm["wnn_weights"]
        np.testing.assert_allclose(weights[:, 0], 0.7, atol=1e-5)
        np.testing.assert_allclose(weights[:, 1], 0.3, atol=1e-5)

    def test_single_modality(self):
        adata = _make_multimodal_adata()
        result = singlet.weighted_nearest_neighbors(
            adata, modalities=["X_pca"], n_neighbors=10
        )
        assert result is adata
        weights = adata.obsm["wnn_weights"]
        assert weights.shape == (100, 1)
        np.testing.assert_allclose(weights, 1.0)

    def test_connectivity_shape(self):
        adata = _make_multimodal_adata(n_cells=50)
        singlet.weighted_nearest_neighbors(
            adata, modalities=["X_pca", "X_protein"], n_neighbors=5
        )
        conn = adata.obsp["wnn_connectivities"]
        assert conn.shape == (50, 50)
        # Should be non-negative
        assert conn.min() >= 0

    def test_n_neighbors_capped(self):
        """n_neighbors larger than n_cells-1 should be capped."""
        adata = _make_multimodal_adata(n_cells=15)
        singlet.weighted_nearest_neighbors(
            adata, modalities=["X_pca", "X_protein"], n_neighbors=50
        )
        assert "wnn_connectivities" in adata.obsp

    def test_invalid_modality_raises(self):
        adata = _make_multimodal_adata()
        with pytest.raises((KeyError, ValueError)):
            singlet.weighted_nearest_neighbors(
                adata, modalities=["X_pca", "X_nonexistent"], n_neighbors=10
            )

    def test_three_modalities(self):
        adata = _make_multimodal_adata()
        rng = np.random.default_rng(123)
        adata.obsm["X_atac"] = rng.standard_normal((100, 10)).astype(
            np.float32
        )
        result = singlet.weighted_nearest_neighbors(
            adata,
            modalities=["X_pca", "X_protein", "X_atac"],
            n_neighbors=10,
        )
        assert result is adata
        assert adata.obsm["wnn_weights"].shape == (100, 3)
        weight_sums = adata.obsm["wnn_weights"].sum(axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-5)

    def test_reproducibility(self):
        adata1 = _make_multimodal_adata(seed=99)
        adata2 = _make_multimodal_adata(seed=99)
        singlet.weighted_nearest_neighbors(
            adata1, modalities=["X_pca", "X_protein"], n_neighbors=10
        )
        singlet.weighted_nearest_neighbors(
            adata2, modalities=["X_pca", "X_protein"], n_neighbors=10
        )
        np.testing.assert_allclose(
            adata1.obsm["wnn_weights"],
            adata2.obsm["wnn_weights"],
            atol=1e-6,
        )

    def test_weights_are_meaningful(self):
        """If one modality has no variance, it should get lower weight."""
        rng = np.random.default_rng(42)
        n_cells = 80
        X = rng.poisson(3, (n_cells, 100)).astype(np.float32)
        adata = AnnData(X=X)
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.var_names = [f"gene_{i}" for i in range(100)]

        # Good modality with clear structure
        adata.obsm["X_good"] = rng.standard_normal((n_cells, 20)).astype(
            np.float32
        )
        # Bad modality: all same value (no structure)
        adata.obsm["X_bad"] = np.ones((n_cells, 10), dtype=np.float32)
        # Add tiny noise to avoid degenerate kNN
        adata.obsm["X_bad"] += rng.standard_normal((n_cells, 10)).astype(
            np.float32
        ) * 1e-6

        singlet.weighted_nearest_neighbors(
            adata, modalities=["X_good", "X_bad"], n_neighbors=10
        )
        weights = adata.obsm["wnn_weights"]
        # Good modality should generally get higher weight
        assert weights[:, 0].mean() > weights[:, 1].mean()
