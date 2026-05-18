# SPDX-License-Identifier: MIT
"""Tests for singlet.filter_cells() and singlet.filter_genes()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._filter import filter_cells, filter_genes


def _make_adata(n_cells=100, n_genes=500, sparse=True):
    """Create test AnnData with known structure."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(n_cells, n_genes, density=0.1, format="csr", random_state=42)
        X.data = rng.integers(1, 100, size=X.nnz).astype(np.float32)
    else:
        X = rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float32)
        mask = rng.random((n_cells, n_genes)) < 0.9
        X[mask] = 0

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestFilterCells:
    def test_min_genes_sparse(self):
        adata = _make_adata(sparse=True)
        result = filter_cells(adata, min_genes=60)
        assert result.shape[0] < adata.shape[0]
        # Verify all remaining cells have >= 60 genes
        genes_per_cell = np.diff(result.X.tocsr().indptr)
        assert genes_per_cell.min() >= 60

    def test_min_genes_dense(self):
        adata = _make_adata(sparse=False)
        result = filter_cells(adata, min_genes=5)
        assert result.shape[0] <= adata.shape[0]

    def test_max_genes(self):
        adata = _make_adata(sparse=True)
        result = filter_cells(adata, max_genes=40)
        assert result.shape[0] < adata.shape[0]
        genes_per_cell = np.diff(result.X.tocsr().indptr)
        assert genes_per_cell.max() <= 40

    def test_min_counts(self):
        adata = _make_adata(sparse=True)
        result = filter_cells(adata, min_counts=3000)
        counts = np.asarray(result.X.sum(axis=1)).ravel()
        assert counts.min() >= 3000

    def test_max_counts(self):
        adata = _make_adata(sparse=True)
        result = filter_cells(adata, max_counts=2000)
        counts = np.asarray(result.X.sum(axis=1)).ravel()
        assert counts.max() <= 2000

    def test_combined_filters(self):
        adata = _make_adata(sparse=True)
        result = filter_cells(adata, min_genes=30, max_counts=5000)
        assert result.shape[0] <= adata.shape[0]

    def test_no_filter_returns_copy(self):
        adata = _make_adata(sparse=True)
        result = filter_cells(adata)
        assert result.shape == adata.shape
        assert result is not adata

    def test_inplace(self):
        adata = _make_adata(sparse=True)
        original_n = adata.shape[0]
        ret = filter_cells(adata, min_genes=60, inplace=True)
        assert ret is None
        assert adata.shape[0] < original_n

    def test_type_error(self):
        with pytest.raises(TypeError, match="filter_cells"):
            filter_cells("not_adata", min_genes=100)

    def test_preserves_obs(self):
        adata = _make_adata(sparse=True)
        adata.obs["batch"] = "A"
        result = filter_cells(adata, min_genes=60)
        assert "batch" in result.obs.columns


class TestFilterGenes:
    def test_min_cells_sparse(self):
        adata = _make_adata(sparse=True)
        result = filter_genes(adata, min_cells=15)
        assert result.shape[1] < adata.shape[1]
        cells_per_gene = np.diff(result.X.tocsc().indptr)
        assert cells_per_gene.min() >= 15

    def test_min_cells_dense(self):
        adata = _make_adata(sparse=False)
        result = filter_genes(adata, min_cells=2)
        assert result.shape[1] <= adata.shape[1]

    def test_max_cells(self):
        adata = _make_adata(sparse=True)
        result = filter_genes(adata, max_cells=8)
        assert result.shape[1] < adata.shape[1]
        cells_per_gene = np.diff(result.X.tocsc().indptr)
        assert cells_per_gene.max() <= 8

    def test_min_counts(self):
        adata = _make_adata(sparse=True)
        result = filter_genes(adata, min_counts=500)
        counts = np.asarray(result.X.sum(axis=0)).ravel()
        assert counts.min() >= 500

    def test_max_counts(self):
        adata = _make_adata(sparse=True)
        result = filter_genes(adata, max_counts=300)
        counts = np.asarray(result.X.sum(axis=0)).ravel()
        assert counts.max() <= 300

    def test_combined_filters(self):
        adata = _make_adata(sparse=True)
        result = filter_genes(adata, min_cells=5, max_counts=1000)
        assert result.shape[1] <= adata.shape[1]

    def test_no_filter_returns_copy(self):
        adata = _make_adata(sparse=True)
        result = filter_genes(adata)
        assert result.shape == adata.shape
        assert result is not adata

    def test_inplace(self):
        adata = _make_adata(sparse=True)
        original_n_genes = adata.shape[1]
        ret = filter_genes(adata, min_cells=15, inplace=True)
        assert ret is None
        assert adata.shape[1] < original_n_genes

    def test_type_error(self):
        with pytest.raises(TypeError, match="filter_genes"):
            filter_genes(42, min_cells=3)

    def test_preserves_var(self):
        adata = _make_adata(sparse=True)
        adata.var["highly_variable"] = True
        result = filter_genes(adata, min_cells=15)
        assert "highly_variable" in result.var.columns


class TestPublicAPI:
    def test_filter_cells_accessible(self):
        assert hasattr(singlet, "filter_cells")
        assert callable(singlet.filter_cells)

    def test_filter_genes_accessible(self):
        assert hasattr(singlet, "filter_genes")
        assert callable(singlet.filter_genes)
