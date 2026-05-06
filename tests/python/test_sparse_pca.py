"""Tests for singlet.sparse_pca()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n_cells=80, n_genes=100, sparse=True, with_hvg=False, seed=42):
    """Create test AnnData for sparse PCA."""
    rng = np.random.default_rng(seed)
    if sparse:
        X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=seed)
        X.data = rng.standard_normal(X.nnz).astype(np.float32)
    else:
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)

    adata = AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    if with_hvg:
        hvg = np.zeros(n_genes, dtype=bool)
        hvg[:50] = True
        adata.var["highly_variable"] = hvg

    return adata


class TestSparsePCA:
    def test_basic_dense(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=60)
        result = singlet.sparse_pca(adata, n_components=10)
        assert result is adata
        assert "X_sparse_pca" in adata.obsm
        assert adata.obsm["X_sparse_pca"].shape == (50, 10)

    def test_basic_sparse(self):
        adata = _make_adata(sparse=True, n_cells=50, n_genes=60)
        singlet.sparse_pca(adata, n_components=10)
        assert "X_sparse_pca" in adata.obsm
        assert adata.obsm["X_sparse_pca"].shape == (50, 10)

    def test_loadings_stored(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=60)
        singlet.sparse_pca(adata, n_components=5)
        assert "sparse_pca_loadings" in adata.varm
        assert adata.varm["sparse_pca_loadings"].shape == (60, 5)

    def test_loadings_with_hvg(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=100, with_hvg=True)
        singlet.sparse_pca(adata, n_components=5)
        # Loadings should be full gene space
        assert adata.varm["sparse_pca_loadings"].shape == (100, 5)
        # Non-HVG genes should have zero loadings
        non_hvg_loadings = adata.varm["sparse_pca_loadings"][50:, :]
        assert np.allclose(non_hvg_loadings, 0)

    def test_uns_metadata(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=60)
        singlet.sparse_pca(adata, n_components=5, alpha=2.0)
        assert "sparse_pca" in adata.uns
        assert adata.uns["sparse_pca"]["n_components"] == 5
        assert adata.uns["sparse_pca"]["alpha"] == 2.0

    def test_sparsity_increases_with_alpha(self):
        """Higher alpha should produce sparser loadings."""
        adata_low = _make_adata(sparse=False, n_cells=50, n_genes=60, seed=1)
        singlet.sparse_pca(adata_low, n_components=5, alpha=0.1)
        sparsity_low = np.mean(adata_low.varm["sparse_pca_loadings"] == 0)

        adata_high = _make_adata(sparse=False, n_cells=50, n_genes=60, seed=1)
        singlet.sparse_pca(adata_high, n_components=5, alpha=5.0)
        sparsity_high = np.mean(adata_high.varm["sparse_pca_loadings"] == 0)

        assert sparsity_high >= sparsity_low

    def test_dtype_float32(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=60)
        singlet.sparse_pca(adata, n_components=5)
        assert adata.obsm["X_sparse_pca"].dtype == np.float32
        assert adata.varm["sparse_pca_loadings"].dtype == np.float32

    def test_n_components_clamped(self):
        """n_components is clamped to min(n_cells, n_genes)."""
        adata = _make_adata(sparse=False, n_cells=20, n_genes=30)
        singlet.sparse_pca(adata, n_components=100)
        assert adata.obsm["X_sparse_pca"].shape[1] == 20

    def test_type_error(self):
        with pytest.raises(TypeError, match="sparse_pca"):
            singlet.sparse_pca("not_adata")

    def test_invalid_n_components(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=60)
        with pytest.raises(ValueError, match="n_components"):
            singlet.sparse_pca(adata, n_components=0)

    def test_reproducibility(self):
        """Same random_state should give same results."""
        adata1 = _make_adata(sparse=False, n_cells=40, n_genes=50, seed=10)
        singlet.sparse_pca(adata1, n_components=5, random_state=42)

        adata2 = _make_adata(sparse=False, n_cells=40, n_genes=50, seed=10)
        singlet.sparse_pca(adata2, n_components=5, random_state=42)

        np.testing.assert_array_almost_equal(
            adata1.obsm["X_sparse_pca"], adata2.obsm["X_sparse_pca"]
        )

    def test_public_api(self):
        assert hasattr(singlet, "sparse_pca")
        assert callable(singlet.sparse_pca)

    def test_embedding_finite(self):
        adata = _make_adata(sparse=False, n_cells=50, n_genes=60)
        singlet.sparse_pca(adata, n_components=5)
        assert np.all(np.isfinite(adata.obsm["X_sparse_pca"]))
