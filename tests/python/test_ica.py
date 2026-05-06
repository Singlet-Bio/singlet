"""Tests for singlet.ica()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._ica import ica


def _make_adata(n_cells=100, n_genes=200, sparse=True, with_hvg=False):
    """Create test AnnData for ICA."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
        X.data = rng.standard_normal(X.nnz).astype(np.float32)
    else:
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    if with_hvg:
        hvg = np.zeros(n_genes, dtype=bool)
        hvg[:100] = True
        adata.var["highly_variable"] = hvg

    return adata


class TestICA:
    def test_basic_sparse(self):
        adata = _make_adata(sparse=True)
        result = ica(adata, n_components=10)
        assert result is adata
        assert "X_ica" in adata.obsm
        assert adata.obsm["X_ica"].shape == (100, 10)
        assert adata.obsm["X_ica"].dtype == np.float32

    def test_basic_dense(self):
        adata = _make_adata(sparse=False)
        result = ica(adata, n_components=15)
        assert result is adata
        assert adata.obsm["X_ica"].shape == (100, 15)

    def test_varm_stored(self):
        adata = _make_adata(sparse=True)
        ica(adata, n_components=10, use_highly_variable=False)
        assert "ica_components" in adata.varm
        assert adata.varm["ica_components"].shape == (200, 10)
        assert adata.varm["ica_components"].dtype == np.float32

    def test_varm_with_hvg(self):
        adata = _make_adata(sparse=True, with_hvg=True)
        ica(adata, n_components=10)
        assert "ica_components" in adata.varm
        # Full-size: n_genes × n_components (zeros padded for non-HVG)
        assert adata.varm["ica_components"].shape == (200, 10)
        # Non-HVG genes should be all zeros
        assert np.all(adata.varm["ica_components"][100:] == 0)

    def test_uns_metadata(self):
        adata = _make_adata()
        ica(adata, n_components=10, random_state=42, max_iter=100)
        assert "ica" in adata.uns
        params = adata.uns["ica"]["params"]
        assert params["n_components"] == 10
        assert params["random_state"] == 42
        assert params["max_iter"] == 100
        assert params["whiten"] == "unit-variance"

    def test_n_components_clamped(self):
        """n_components larger than n_cells-1 gets clamped."""
        adata = _make_adata(n_cells=20, n_genes=50)
        ica(adata, n_components=100)
        # Should be clamped to min(100, 19, 49) = 19
        assert adata.obsm["X_ica"].shape == (20, 19)

    def test_reproducible(self):
        """Same random_state produces same result."""
        adata1 = _make_adata()
        adata2 = _make_adata()
        ica(adata1, n_components=10, random_state=123)
        ica(adata2, n_components=10, random_state=123)
        np.testing.assert_array_equal(
            adata1.obsm["X_ica"], adata2.obsm["X_ica"]
        )

    def test_different_seeds(self):
        """Different random_state produces different results."""
        adata1 = _make_adata()
        adata2 = _make_adata()
        ica(adata1, n_components=10, random_state=0)
        ica(adata2, n_components=10, random_state=99)
        assert not np.allclose(adata1.obsm["X_ica"], adata2.obsm["X_ica"])

    def test_type_error_non_adata(self):
        with pytest.raises(TypeError, match="requires an AnnData object"):
            ica(np.zeros((10, 5)))

    def test_value_error_too_small(self):
        """Single cell should raise."""
        import anndata as ad

        adata = ad.AnnData(X=np.ones((1, 5)))
        with pytest.raises(ValueError, match="Cannot compute ICA"):
            ica(adata, n_components=5)

    def test_registered_in_singlet(self):
        assert hasattr(singlet, "ica")
        assert "ica" in singlet.__all__
