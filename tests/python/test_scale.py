# SPDX-License-Identifier: MIT
"""Tests for singlet.scale()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._scale import scale


def _make_adata(n_cells=100, n_genes=50, sparse=True):
    """Create a test AnnData with known expression values."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    if sparse:
        X_data = sp.csr_matrix(X)
    else:
        X_data = X
    adata = ad.AnnData(X=X_data)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


class TestScale:
    def test_basic_inplace(self):
        """Scale should modify adata.X in place."""
        adata = _make_adata()
        ret = scale(adata)
        assert ret is None
        # Should be dense after scaling
        assert isinstance(adata.X, np.ndarray)

    def test_zero_mean(self):
        """After scaling, each gene should have approximately zero mean."""
        adata = _make_adata(n_cells=500)
        scale(adata)
        means = adata.X.mean(axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10)

    def test_unit_variance(self):
        """After scaling, each gene should have approximately unit variance."""
        adata = _make_adata(n_cells=500)
        scale(adata, max_value=None)  # no clipping for variance test
        stds = adata.X.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_max_value_clipping(self):
        """Values should be clipped to [-max_value, max_value]."""
        adata = _make_adata(n_cells=200)
        scale(adata, max_value=3.0)
        assert adata.X.max() <= 3.0
        assert adata.X.min() >= -3.0

    def test_max_value_none(self):
        """max_value=None should not clip."""
        adata = _make_adata(n_cells=500)
        scale(adata, max_value=None)
        # Some values might exceed 10
        # Just check no clipping was applied (could have values > 3)
        assert adata.X.max() > 3.0 or adata.X.min() < -3.0

    def test_not_inplace(self):
        """inplace=False should return array without modifying adata."""
        adata = _make_adata()
        result = scale(adata, inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (adata.n_obs, adata.n_vars)
        # Original should be unchanged (sparse)
        assert sp.issparse(adata.X)

    def test_sparse_input(self):
        """Should handle sparse input (converts to dense)."""
        adata = _make_adata(sparse=True)
        assert sp.issparse(adata.X)
        scale(adata)
        assert isinstance(adata.X, np.ndarray)

    def test_dense_input(self):
        """Should handle dense input."""
        adata = _make_adata(sparse=False)
        scale(adata)
        assert isinstance(adata.X, np.ndarray)

    def test_no_zero_center(self):
        """zero_center=False should only divide by std."""
        adata = _make_adata(n_cells=200)
        scale(adata, zero_center=False, max_value=None)
        # Mean should NOT be zero (was positive Poisson data)
        means = adata.X.mean(axis=0)
        assert np.any(means > 0.5)
        # But std should be approximately 1
        stds = adata.X.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=0.05)

    def test_constant_gene(self):
        """Should handle genes with zero variance (constant expression)."""
        import anndata as ad

        X = np.ones((50, 10), dtype=np.float32)
        X[:, 5] = 3.0  # constant gene
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(10)]
        scale(adata)
        # Constant gene should be 0 after centering (3-3)/1 = 0
        assert np.all(adata.X[:, 5] == 0.0)

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="scale"):
            scale("not_adata")

    def test_output_shape(self):
        """Output should preserve shape."""
        adata = _make_adata(n_cells=80, n_genes=30)
        scale(adata)
        assert adata.X.shape == (80, 30)

    def test_public_api(self):
        assert hasattr(singlet, "scale")
        assert callable(singlet.scale)
