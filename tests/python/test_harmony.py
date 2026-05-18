# SPDX-License-Identifier: MIT
"""Tests for singlet.harmony() batch correction."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._harmony import harmony


def _make_batch_adata(n_cells_per_batch=100, n_genes=50, n_batches=3, n_comps=20):
    """Create AnnData with batch effects in PCA space."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells = n_cells_per_batch * n_batches

    # Generate PCA embeddings with batch effects
    # Shared biological signal (same structure across batches)
    Z_bio = rng.standard_normal((n_cells, n_comps)) * 0.5

    # Add batch-specific shifts (the "batch effect")
    batch_labels = []
    for b in range(n_batches):
        start = b * n_cells_per_batch
        end = (b + 1) * n_cells_per_batch
        shift = rng.standard_normal(n_comps) * 3.0  # strong batch effect
        Z_bio[start:end] += shift
        batch_labels.extend([f"batch_{b}"] * n_cells_per_batch)

    # Create minimal AnnData with PCA
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["batch"] = pd.Categorical(batch_labels)
    adata.obsm["X_pca"] = Z_bio.astype(np.float32)

    return adata


class TestHarmony:
    def test_basic_inplace(self):
        """Harmony should store corrected embeddings in obsm."""
        adata = _make_batch_adata()
        ret = harmony(adata, "batch")
        assert ret is None
        assert "X_pca_harmony" in adata.obsm
        assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape

    def test_not_inplace(self):
        """When inplace=False, return array without modifying adata."""
        adata = _make_batch_adata()
        result = harmony(adata, "batch", inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == adata.obsm["X_pca"].shape
        assert "X_pca_harmony" not in adata.obsm

    def test_reduces_batch_effect(self):
        """Corrected embeddings should have smaller batch separation."""
        adata = _make_batch_adata(n_cells_per_batch=150)
        Z_orig = adata.obsm["X_pca"].copy()

        harmony(adata, "batch")
        Z_corr = adata.obsm["X_pca_harmony"]

        # Compute batch centroid distances before and after
        batches = adata.obs["batch"].unique()
        orig_dists = []
        corr_dists = []

        for i, b1 in enumerate(batches):
            for b2 in batches[i + 1 :]:
                mask1 = (adata.obs["batch"] == b1).values
                mask2 = (adata.obs["batch"] == b2).values

                d_orig = np.linalg.norm(Z_orig[mask1].mean(axis=0) - Z_orig[mask2].mean(axis=0))
                d_corr = np.linalg.norm(Z_corr[mask1].mean(axis=0) - Z_corr[mask2].mean(axis=0))
                orig_dists.append(d_orig)
                corr_dists.append(d_corr)

        # After correction, batch centroids should be closer together
        assert np.mean(corr_dists) < np.mean(orig_dists)

    def test_n_comps(self):
        """Should respect n_comps parameter."""
        adata = _make_batch_adata(n_comps=30)
        result = harmony(adata, "batch", n_comps=10, inplace=False)
        assert result.shape == (adata.n_obs, 10)

    def test_single_batch_noop(self):
        """With only one batch, should return unchanged embeddings."""
        adata = _make_batch_adata(n_batches=1)
        Z_orig = adata.obsm["X_pca"].copy()
        harmony(adata, "batch")
        np.testing.assert_allclose(adata.obsm["X_pca_harmony"], Z_orig, atol=1e-10)

    def test_missing_pca_raises(self):
        """Should raise KeyError when X_pca is missing."""
        import anndata as ad

        adata = ad.AnnData(X=sp.random(50, 30, format="csr"))
        adata.obs["batch"] = ["a"] * 25 + ["b"] * 25
        with pytest.raises(KeyError, match="X_pca"):
            harmony(adata, "batch")

    def test_missing_key_raises(self):
        """Should raise KeyError for missing batch column."""
        adata = _make_batch_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            harmony(adata, "nonexistent")

    def test_type_error(self):
        """Should raise TypeError for non-AnnData input."""
        with pytest.raises(TypeError, match="harmony"):
            harmony("not_adata", "batch")

    def test_max_iter(self):
        """Should converge with default settings (no crash)."""
        adata = _make_batch_adata(n_cells_per_batch=50)
        harmony(adata, "batch", max_iter=5)
        assert "X_pca_harmony" in adata.obsm

    def test_theta_zero(self):
        """theta=0 should disable diversity penalty (still runs)."""
        adata = _make_batch_adata(n_cells_per_batch=50)
        harmony(adata, "batch", theta=0.0)
        assert "X_pca_harmony" in adata.obsm

    def test_reproducible(self):
        """Same random_state should give same results."""
        adata1 = _make_batch_adata()
        adata2 = _make_batch_adata()
        r1 = harmony(adata1, "batch", inplace=False, random_state=42)
        r2 = harmony(adata2, "batch", inplace=False, random_state=42)
        np.testing.assert_allclose(r1, r2)

    def test_different_seeds(self):
        """Different random_state may converge to similar solutions, but init differs."""
        adata = _make_batch_adata()
        # With very few iterations, different seeds should differ more
        r1 = harmony(adata, "batch", inplace=False, random_state=0, max_iter=1)
        adata2 = _make_batch_adata()
        r2 = harmony(adata2, "batch", inplace=False, random_state=99, max_iter=1)
        # At least some difference after just 1 iteration with different init
        assert not np.array_equal(r1, r2)

    def test_output_shape_preserved(self):
        """Output should have same shape as input PCA."""
        adata = _make_batch_adata(n_cells_per_batch=80, n_comps=15)
        harmony(adata, "batch")
        assert adata.obsm["X_pca_harmony"].shape == (240, 15)

    def test_no_nans(self):
        """Output should not contain NaN or Inf."""
        adata = _make_batch_adata()
        harmony(adata, "batch")
        Z = adata.obsm["X_pca_harmony"]
        assert np.all(np.isfinite(Z))

    def test_public_api(self):
        """harmony should be accessible from singlet namespace."""
        assert hasattr(singlet, "harmony")
        assert callable(singlet.harmony)

    def test_two_batches(self):
        """Should work with exactly 2 batches."""
        adata = _make_batch_adata(n_batches=2, n_cells_per_batch=80)
        harmony(adata, "batch")
        assert adata.obsm["X_pca_harmony"].shape == (160, 20)
        assert np.all(np.isfinite(adata.obsm["X_pca_harmony"]))
