# SPDX-License-Identifier: MIT
"""Tests for singlet.regress_out()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._regress import regress_out


def _make_adata(n_cells=100, n_genes=50):
    """Create AnnData with confounding variables."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # Expression correlated with total_counts
    total_counts = rng.uniform(1000, 10000, size=n_cells)
    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float64)

    # Add confounding: genes scale with total_counts
    for i in range(n_cells):
        X[i] *= total_counts[i] / 5000.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["total_counts"] = total_counts
    adata.obs["pct_mito"] = rng.uniform(0, 0.2, size=n_cells)

    return adata


class TestRegressOut:
    def test_basic_inplace(self):
        """Should modify adata.X in place."""
        adata = _make_adata()
        ret = regress_out(adata, ["total_counts"])
        assert ret is None
        assert isinstance(adata.X, np.ndarray)

    def test_not_inplace(self):
        """inplace=False should return array without modifying."""
        adata = _make_adata()
        result = regress_out(adata, ["total_counts"], inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (adata.n_obs, adata.n_vars)
        assert sp.issparse(adata.X)

    def test_removes_correlation(self):
        """After regression, gene expression should be less correlated with confound."""
        adata = _make_adata(n_cells=200)
        total_counts = adata.obs["total_counts"].values

        # Correlation before
        X_before = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        corr_before = np.abs(np.corrcoef(total_counts, X_before[:, 0])[0, 1])

        regress_out(adata, ["total_counts"])

        # Correlation after
        corr_after = np.abs(np.corrcoef(total_counts, adata.X[:, 0])[0, 1])

        assert corr_after < corr_before

    def test_multiple_keys(self):
        """Should regress out multiple variables simultaneously."""
        adata = _make_adata()
        regress_out(adata, ["total_counts", "pct_mito"])
        assert isinstance(adata.X, np.ndarray)
        assert np.all(np.isfinite(adata.X))

    def test_preserves_shape_and_finite(self):
        """Output should preserve shape and be finite."""
        adata = _make_adata(n_cells=200)
        regress_out(adata, ["total_counts"])
        assert adata.X.shape == (200, 50)
        assert np.all(np.isfinite(adata.X))

    def test_sparse_input(self):
        """Should handle sparse input."""
        adata = _make_adata()
        assert sp.issparse(adata.X)
        regress_out(adata, ["total_counts"])
        assert isinstance(adata.X, np.ndarray)

    def test_dense_input(self):
        """Should handle dense input."""
        adata = _make_adata()
        adata.X = adata.X.toarray()
        regress_out(adata, ["total_counts"])
        assert isinstance(adata.X, np.ndarray)

    def test_empty_keys_raises(self):
        """Should raise ValueError for empty keys."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="must not be empty"):
            regress_out(adata, [])

    def test_missing_key_raises(self):
        """Should raise KeyError for missing column."""
        adata = _make_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            regress_out(adata, ["nonexistent"])

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="regress_out"):
            regress_out("not_adata", ["x"])

    def test_output_shape(self):
        """Output should preserve shape."""
        adata = _make_adata(n_cells=80, n_genes=40)
        regress_out(adata, ["total_counts"])
        assert adata.X.shape == (80, 40)

    def test_no_nans(self):
        """Output should not contain NaN or Inf."""
        adata = _make_adata()
        regress_out(adata, ["total_counts", "pct_mito"])
        assert np.all(np.isfinite(adata.X))

    def test_public_api(self):
        assert hasattr(singlet, "regress_out")
        assert callable(singlet.regress_out)

    def test_layer_parameter(self):
        """Should regress out from a specific layer."""
        adata = _make_adata()
        adata.layers["raw_counts"] = adata.X.copy()
        regress_out(adata, ["total_counts"], layer="raw_counts")
        # Layer should be modified (dense)
        assert isinstance(adata.layers["raw_counts"], np.ndarray)
        # Original X should be untouched (still sparse)
        assert sp.issparse(adata.X)

    def test_layer_not_found_raises(self):
        """Should raise KeyError for missing layer."""
        adata = _make_adata()
        with pytest.raises(KeyError, match="nonexistent_layer"):
            regress_out(adata, ["total_counts"], layer="nonexistent_layer")

    def test_layer_inplace_false(self):
        """layer + inplace=False returns array without modifying layer."""
        adata = _make_adata()
        adata.layers["test_layer"] = adata.X.copy()
        result = regress_out(adata, ["total_counts"], layer="test_layer", inplace=False)
        assert isinstance(result, np.ndarray)
        # Layer should still be sparse
        assert sp.issparse(adata.layers["test_layer"])
