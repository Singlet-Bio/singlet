"""Tests for singlet.pca()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._pca import pca


def _make_adata(n_cells=200, n_genes=500, sparse=True, with_hvg=False):
    """Create test AnnData for PCA."""
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
        hvg[:200] = True
        adata.var["highly_variable"] = hvg

    return adata


class TestPCA:
    def test_basic_inplace_sparse(self):
        adata = _make_adata(sparse=True)
        ret = pca(adata, n_comps=20)
        assert ret is None
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (200, 20)
        assert adata.obsm["X_pca"].dtype == np.float32

    def test_basic_inplace_dense(self):
        adata = _make_adata(sparse=False)
        ret = pca(adata, n_comps=10)
        assert ret is None
        assert adata.obsm["X_pca"].shape == (200, 10)

    def test_uses_hvg(self):
        adata = _make_adata(sparse=True, with_hvg=True)
        pca(adata, n_comps=20)
        assert adata.obsm["X_pca"].shape == (200, 20)
        # PCs should NOT be in varm when using HVGs (subset of genes)
        assert "PCs" not in adata.varm

    def test_stores_pcs_when_all_genes(self):
        adata = _make_adata(sparse=True)
        pca(adata, n_comps=20, use_highly_variable=False)
        assert "PCs" in adata.varm
        assert adata.varm["PCs"].shape == (500, 20)

    def test_variance_info(self):
        adata = _make_adata(sparse=True)
        pca(adata, n_comps=20)
        assert "pca" in adata.uns
        assert "variance" in adata.uns["pca"]
        assert "variance_ratio" in adata.uns["pca"]
        assert len(adata.uns["pca"]["variance"]) == 20
        # Variance ratio should sum to ~1
        vr = adata.uns["pca"]["variance_ratio"]
        assert abs(vr.sum() - 1.0) < 0.01 or vr.sum() <= 1.0

    def test_variance_descending(self):
        adata = _make_adata(sparse=True)
        pca(adata, n_comps=20)
        var = adata.uns["pca"]["variance"]
        assert np.all(var[:-1] >= var[1:])

    def test_not_inplace(self):
        adata = _make_adata(sparse=True)
        result = pca(adata, n_comps=15, inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (200, 15)
        assert "X_pca" not in adata.obsm

    def test_no_zero_center(self):
        adata = _make_adata(sparse=True)
        pca(adata, n_comps=10, zero_center=False)
        assert adata.obsm["X_pca"].shape == (200, 10)

    def test_n_comps_clamped(self):
        """n_comps is reduced if larger than n_cells - 1."""
        adata = _make_adata(n_cells=20, n_genes=500, sparse=True)
        pca(adata, n_comps=50)
        assert adata.obsm["X_pca"].shape[1] <= 19

    def test_type_error(self):
        with pytest.raises(TypeError, match="pca"):
            pca("not_adata")

    def test_too_few_cells(self):
        """Raises on degenerate input."""
        import anndata as ad

        X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = ["A", "B", "C"]
        adata.obs_names = ["c1"]
        with pytest.raises(ValueError, match="Cannot compute PCA"):
            pca(adata)

    def test_public_api(self):
        assert hasattr(singlet, "pca")
        assert callable(singlet.pca)

    def test_orthogonal_components(self):
        """PCA components should be approximately orthogonal."""
        adata = _make_adata(n_cells=100, n_genes=200, sparse=True)
        pca(adata, n_comps=10)
        X_pca = adata.obsm["X_pca"]
        # Columns should be uncorrelated
        corr = np.corrcoef(X_pca.T)
        np.fill_diagonal(corr, 0)
        assert np.abs(corr).max() < 0.1
