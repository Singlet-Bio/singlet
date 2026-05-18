# SPDX-License-Identifier: MIT
"""Tests for singlet.highly_variable_genes()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._hvg import highly_variable_genes


def _make_normalized_adata(n_cells=200, n_genes=1000, sparse=True):
    """Create test AnnData with log-normalized-like values."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
        X.data = rng.exponential(1.0, size=X.nnz).astype(np.float32)
    else:
        X = rng.exponential(0.3, size=(n_cells, n_genes)).astype(np.float32)
        mask = rng.random((n_cells, n_genes)) < 0.7
        X[mask] = 0

    # Make some genes highly variable (bimodal expression)
    for i in range(50):
        expressing_cells = rng.choice(n_cells, size=n_cells // 3, replace=False)
        if sparse:
            X = X.tolil()
            X[expressing_cells, i] = rng.exponential(5.0, size=len(expressing_cells))
            X = X.tocsr()
        else:
            X[expressing_cells, i] = rng.exponential(5.0, size=len(expressing_cells))

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestHighlyVariableGenes:
    def test_basic_inplace(self):
        adata = _make_normalized_adata(sparse=True)
        ret = highly_variable_genes(adata)
        assert ret is None
        assert "highly_variable" in adata.var.columns
        assert "means" in adata.var.columns
        assert "dispersions" in adata.var.columns
        assert "dispersions_norm" in adata.var.columns

    def test_selects_n_top_genes(self):
        adata = _make_normalized_adata(sparse=True)
        highly_variable_genes(adata, n_top_genes=500)
        assert adata.var["highly_variable"].sum() == 500

    def test_default_2000(self):
        adata = _make_normalized_adata(sparse=True, n_genes=3000)
        highly_variable_genes(adata)
        assert adata.var["highly_variable"].sum() == 2000

    def test_dense_input(self):
        adata = _make_normalized_adata(sparse=False)
        highly_variable_genes(adata, n_top_genes=100)
        assert adata.var["highly_variable"].sum() == 100

    def test_not_inplace_returns_list(self):
        adata = _make_normalized_adata(sparse=True)
        result = highly_variable_genes(adata, n_top_genes=200, inplace=False)
        assert isinstance(result, list)
        assert len(result) == 200
        assert all(isinstance(g, str) for g in result)
        # Should not modify adata
        assert "highly_variable" not in adata.var.columns

    def test_n_top_genes_exceeds_n_genes(self):
        """Gracefully handles n_top_genes > n_genes."""
        adata = _make_normalized_adata(sparse=True, n_genes=100)
        highly_variable_genes(adata, n_top_genes=500)
        assert adata.var["highly_variable"].sum() <= 100

    def test_type_error(self):
        with pytest.raises(TypeError, match="highly_variable_genes"):
            highly_variable_genes("not_adata")

    def test_bad_flavor(self):
        adata = _make_normalized_adata(sparse=True)
        with pytest.raises(ValueError, match="flavor"):
            highly_variable_genes(adata, flavor="unknown")

    def test_variable_genes_have_high_dispersion(self):
        """HVGs should have higher normalized dispersion than non-HVGs."""
        adata = _make_normalized_adata(sparse=True, n_genes=2000)
        highly_variable_genes(adata, n_top_genes=500)
        hvg_disp = adata.var.loc[adata.var["highly_variable"], "dispersions_norm"].mean()
        non_hvg_disp = adata.var.loc[~adata.var["highly_variable"], "dispersions_norm"].mean()
        assert hvg_disp > non_hvg_disp

    def test_public_api(self):
        assert hasattr(singlet, "highly_variable_genes")
        assert callable(singlet.highly_variable_genes)

    def test_means_are_positive(self):
        adata = _make_normalized_adata(sparse=True)
        highly_variable_genes(adata)
        assert (adata.var["means"] >= 0).all()
        assert (adata.var["dispersions"] >= 0).all()
