"""Tests for singlet.normalize()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._normalize import normalize


def _make_adata(n_cells=50, n_genes=200, sparse=True):
    """Create test AnnData with integer counts."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(n_cells, n_genes, density=0.2, format="csr", random_state=42)
        X.data = rng.integers(1, 100, size=X.nnz).astype(np.float32)
    else:
        X = rng.integers(0, 50, size=(n_cells, n_genes)).astype(np.float32)
        mask = rng.random((n_cells, n_genes)) < 0.8
        X[mask] = 0

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestNormalize:
    def test_basic_sparse_inplace(self):
        adata = _make_adata(sparse=True)
        ret = normalize(adata)
        assert ret is None
        # X should now be log-normalized
        assert adata.X.max() < 15  # log1p(10000) ≈ 9.2

    def test_basic_dense_inplace(self):
        adata = _make_adata(sparse=False)
        ret = normalize(adata)
        assert ret is None
        assert adata.X.max() < 15

    def test_keeps_raw_layer(self):
        adata = _make_adata(sparse=True)
        raw_copy = adata.X.copy()
        normalize(adata)
        assert "raw" in adata.layers
        diff = np.abs(adata.layers["raw"] - raw_copy).sum()
        assert diff == 0

    def test_no_keep_raw(self):
        adata = _make_adata(sparse=True)
        normalize(adata, keep_raw=False)
        assert "raw" not in adata.layers

    def test_target_sum_changes_scale(self):
        adata1 = _make_adata(sparse=True)
        adata2 = _make_adata(sparse=True)
        normalize(adata1, target_sum=1e4, log=False)
        normalize(adata2, target_sum=1e6, log=False)
        # adata2 values should be ~100x larger
        max1 = adata1.X.max()
        max2 = adata2.X.max()
        assert max2 > max1 * 50

    def test_target_sum_none_uses_median(self):
        adata = _make_adata(sparse=True)
        normalize(adata, target_sum=None, log=False)
        # Should work without error; cells should have similar totals
        totals = np.asarray(adata.X.sum(axis=1)).ravel()
        cv = totals.std() / totals.mean()
        assert cv < 1.0  # less variable than raw

    def test_no_log(self):
        adata = _make_adata(sparse=True)
        normalize(adata, log=False)
        # Max should be close to target_sum (for highly expressed genes)
        assert adata.X.max() > 100  # not log-transformed

    def test_not_inplace_returns_copy(self):
        adata = _make_adata(sparse=True)
        result = normalize(adata, inplace=False)
        assert result is not None
        assert result is not adata
        # Original should be unchanged
        assert "raw" not in adata.layers

    def test_zero_cells_handled(self):
        """Cells with zero counts don't cause division by zero."""
        import anndata as ad

        X = sp.csr_matrix(np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = ["A", "B", "C"]
        adata.obs_names = ["c1", "c2", "c3"]
        normalize(adata)
        assert np.all(np.isfinite(adata.X.toarray()))

    def test_preserves_sparsity(self):
        adata = _make_adata(sparse=True)
        normalize(adata)
        assert sp.issparse(adata.X)

    def test_type_error(self):
        with pytest.raises(TypeError, match="normalize"):
            normalize("not_adata")

    def test_public_api(self):
        assert hasattr(singlet, "normalize")
        assert callable(singlet.normalize)

    def test_does_not_overwrite_existing_raw(self):
        """If layers['raw'] already exists, don't overwrite it."""
        adata = _make_adata(sparse=True)
        adata.layers["raw"] = adata.X.copy()
        marker = adata.layers["raw"].sum()
        normalize(adata)
        assert adata.layers["raw"].sum() == marker

    def test_float32_output(self):
        adata = _make_adata(sparse=True)
        normalize(adata)
        assert adata.X.dtype == np.float32
