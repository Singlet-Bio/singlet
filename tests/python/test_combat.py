# SPDX-License-Identifier: MIT
"""Tests for singlet.combat() batch correction."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._combat import combat


def _make_batch_adata(n_cells_per_batch=80, n_genes=100, n_batches=3):
    """Create AnnData with batch effects in expression space."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells = n_cells_per_batch * n_batches

    # Base expression (shared biology)
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float64)

    # Add batch-specific shifts (strong batch effect)
    batch_labels = []
    for b in range(n_batches):
        start = b * n_cells_per_batch
        end = (b + 1) * n_cells_per_batch
        shift = rng.standard_normal(n_genes) * 3.0
        X[start:end] += shift
        batch_labels.extend([f"batch_{b}"] * n_cells_per_batch)

    # Log1p to simulate normalized data
    X = np.maximum(X, 0)
    X = np.log1p(X)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["batch"] = pd.Categorical(batch_labels)
    adata.obs["condition"] = pd.Categorical(
        ["ctrl"] * (n_cells // 2) + ["treat"] * (n_cells - n_cells // 2)
    )

    return adata


class TestCombat:
    def test_basic_inplace(self):
        """ComBat should modify adata.X in place."""
        adata = _make_batch_adata()
        ret = combat(adata, "batch")
        assert ret is None
        # Should be dense after combat
        assert isinstance(adata.X, np.ndarray)

    def test_not_inplace(self):
        """inplace=False should return corrected matrix."""
        adata = _make_batch_adata()
        result = combat(adata, "batch", inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (adata.n_obs, adata.n_vars)
        # Original should still be sparse
        assert sp.issparse(adata.X)

    def test_reduces_batch_effect(self):
        """Corrected data should have smaller batch mean differences."""
        adata = _make_batch_adata(n_cells_per_batch=100)

        # Original batch mean differences
        X_orig = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
        batches = adata.obs["batch"].unique()
        orig_diffs = []
        for i, b1 in enumerate(batches):
            for b2 in batches[i + 1 :]:
                m1 = (adata.obs["batch"] == b1).values
                m2 = (adata.obs["batch"] == b2).values
                d = np.abs(X_orig[m1].mean(axis=0) - X_orig[m2].mean(axis=0)).mean()
                orig_diffs.append(d)

        combat(adata, "batch")
        X_corr = adata.X

        corr_diffs = []
        for i, b1 in enumerate(batches):
            for b2 in batches[i + 1 :]:
                m1 = (adata.obs["batch"] == b1).values
                m2 = (adata.obs["batch"] == b2).values
                d = np.abs(X_corr[m1].mean(axis=0) - X_corr[m2].mean(axis=0)).mean()
                corr_diffs.append(d)

        assert np.mean(corr_diffs) < np.mean(orig_diffs)

    def test_single_batch_noop(self):
        """Single batch should return data unchanged."""
        adata = _make_batch_adata(n_batches=1)
        X_orig = adata.X.toarray().copy()
        combat(adata, "batch")
        np.testing.assert_allclose(adata.X, X_orig, atol=1e-10)

    def test_with_covariates(self):
        """Should accept covariates parameter."""
        adata = _make_batch_adata()
        combat(adata, "batch", covariates=["condition"])
        assert isinstance(adata.X, np.ndarray)
        assert np.all(np.isfinite(adata.X))

    def test_missing_key_raises(self):
        """Should raise KeyError for missing batch column."""
        adata = _make_batch_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            combat(adata, "nonexistent")

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="combat"):
            combat("not_adata", "batch")

    def test_dense_input(self):
        """Should work with dense input matrix."""
        adata = _make_batch_adata()
        adata.X = adata.X.toarray()
        combat(adata, "batch")
        assert isinstance(adata.X, np.ndarray)
        assert np.all(np.isfinite(adata.X))

    def test_output_shape(self):
        """Output should preserve shape."""
        adata = _make_batch_adata(n_cells_per_batch=50, n_genes=80)
        combat(adata, "batch")
        assert adata.X.shape == (150, 80)

    def test_no_nans(self):
        """Output should not contain NaN or Inf."""
        adata = _make_batch_adata()
        combat(adata, "batch")
        assert np.all(np.isfinite(adata.X))

    def test_two_batches(self):
        """Should work with exactly 2 batches."""
        adata = _make_batch_adata(n_batches=2)
        combat(adata, "batch")
        assert adata.X.shape == (160, 100)
        assert np.all(np.isfinite(adata.X))

    def test_public_api(self):
        assert hasattr(singlet, "combat")
        assert callable(singlet.combat)
