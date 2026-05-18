# SPDX-License-Identifier: MIT
"""Tests for singlet.neighbors()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._neighbors import neighbors


def _make_adata_with_pca(n_cells=100, n_pcs=30):
    """Create test AnnData with PCA embedding."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.random(n_cells, 500, density=0.3, format="csr", random_state=42)
    X.data = rng.standard_normal(X.nnz).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(500)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    # Add PCA embedding
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, n_pcs)).astype(np.float32)
    return adata


class TestNeighbors:
    def test_basic_inplace(self):
        adata = _make_adata_with_pca()
        ret = neighbors(adata)
        assert ret is None
        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp
        assert sp.issparse(adata.obsp["connectivities"])
        assert sp.issparse(adata.obsp["distances"])

    def test_shape(self):
        adata = _make_adata_with_pca(n_cells=50)
        neighbors(adata, n_neighbors=10)
        assert adata.obsp["connectivities"].shape == (50, 50)
        assert adata.obsp["distances"].shape == (50, 50)

    def test_stores_params(self):
        adata = _make_adata_with_pca()
        neighbors(adata, n_neighbors=20, metric="cosine")
        assert adata.uns["neighbors"]["n_neighbors"] == 20
        assert adata.uns["neighbors"]["metric"] == "cosine"

    def test_not_inplace(self):
        adata = _make_adata_with_pca()
        result = neighbors(adata, inplace=False)
        assert isinstance(result, dict)
        assert "connectivities" in result
        assert "distances" in result
        assert "connectivities" not in adata.obsp

    def test_n_pcs(self):
        adata = _make_adata_with_pca(n_pcs=50)
        neighbors(adata, n_pcs=10)
        assert adata.uns["neighbors"]["n_pcs"] == 10

    def test_cosine_metric(self):
        adata = _make_adata_with_pca()
        neighbors(adata, metric="cosine")
        assert adata.obsp["distances"].nnz > 0

    def test_cityblock_metric(self):
        adata = _make_adata_with_pca()
        neighbors(adata, metric="cityblock")
        assert adata.obsp["distances"].nnz > 0

    def test_missing_pca_raises(self):
        import anndata as ad

        X = sp.csr_matrix(np.random.rand(10, 20).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{i}" for i in range(20)]
        adata.obs_names = [f"c{i}" for i in range(10)]
        with pytest.raises(KeyError, match="X_pca"):
            neighbors(adata)

    def test_type_error(self):
        with pytest.raises(TypeError, match="neighbors"):
            neighbors("not_adata")

    def test_symmetric_graph(self):
        """Connectivities should be symmetric."""
        adata = _make_adata_with_pca(n_cells=50)
        neighbors(adata, n_neighbors=10)
        C = adata.obsp["connectivities"]
        diff = np.abs(C - C.T).max()
        assert diff < 1e-10

    def test_distances_non_negative(self):
        adata = _make_adata_with_pca()
        neighbors(adata)
        assert adata.obsp["distances"].data.min() >= 0

    def test_public_api(self):
        assert hasattr(singlet, "neighbors")
        assert callable(singlet.neighbors)
