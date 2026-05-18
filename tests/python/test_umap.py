# SPDX-License-Identifier: MIT
"""Tests for singlet.umap()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._umap import umap


def _make_adata_with_graph(n_cells=100):
    """Create test AnnData with a connectivity graph."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.random(n_cells, 200, density=0.2, format="csr", random_state=42)
    X.data = rng.standard_normal(X.nnz).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(200)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Build a simple kNN graph
    pca_embed = rng.standard_normal((n_cells, 20)).astype(np.float32)
    adata.obsm["X_pca"] = pca_embed

    # Compute neighbors
    from singlet._neighbors import neighbors

    neighbors(adata, n_neighbors=10)
    return adata


class TestUMAP:
    def test_basic_inplace(self):
        adata = _make_adata_with_graph()
        ret = umap(adata)
        assert ret is None
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (100, 2)
        assert adata.obsm["X_umap"].dtype == np.float32

    def test_3d(self):
        adata = _make_adata_with_graph()
        umap(adata, n_components=3)
        assert adata.obsm["X_umap"].shape == (100, 3)

    def test_not_inplace(self):
        adata = _make_adata_with_graph()
        result = umap(adata, inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 2)
        assert "X_umap" not in adata.obsm

    def test_missing_connectivities_raises(self):
        import anndata as ad

        X = sp.csr_matrix(np.random.rand(10, 20).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{i}" for i in range(20)]
        adata.obs_names = [f"c{i}" for i in range(10)]
        with pytest.raises(KeyError, match="connectivities"):
            umap(adata)

    def test_type_error(self):
        with pytest.raises(TypeError, match="umap"):
            umap("not_adata")

    def test_finite_values(self):
        adata = _make_adata_with_graph()
        umap(adata)
        assert np.all(np.isfinite(adata.obsm["X_umap"]))

    def test_reproducible(self):
        adata1 = _make_adata_with_graph()
        adata2 = _make_adata_with_graph()
        umap(adata1, random_state=42)
        umap(adata2, random_state=42)
        np.testing.assert_array_equal(adata1.obsm["X_umap"], adata2.obsm["X_umap"])

    def test_public_api(self):
        assert hasattr(singlet, "umap")
        assert callable(singlet.umap)
