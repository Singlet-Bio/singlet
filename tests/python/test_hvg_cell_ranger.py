"""Tests for singlet.highly_variable_genes_cell_ranger()."""

import numpy as np
import pytest
import scipy.sparse as sp


def _make_adata(n_cells=200, n_genes=3000, sparse=True):
    """Create test AnnData for HVG Cell Ranger."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(
            n_cells, n_genes, density=0.3,
            format="csr", random_state=42,
        )
        X.data = np.abs(
            rng.standard_normal(X.nnz).astype(np.float32)
        ) + 0.01
    else:
        X = np.abs(
            rng.standard_normal(
                (n_cells, n_genes)
            ).astype(np.float32)
        ) + 0.01

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestHighlyVariableGenesCellRanger:
    def test_basic_inplace(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata()
        result = highly_variable_genes_cell_ranger(adata)
        assert result is None
        assert "highly_variable" in adata.var.columns
        assert "means" in adata.var.columns
        assert "dispersions" in adata.var.columns
        assert "dispersions_norm" in adata.var.columns
        assert adata.var["highly_variable"].sum() == 2000

    def test_custom_n_top_genes(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata()
        highly_variable_genes_cell_ranger(
            adata, n_top_genes=500
        )
        assert adata.var["highly_variable"].sum() == 500

    def test_not_inplace(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata()
        result = highly_variable_genes_cell_ranger(
            adata, inplace=False
        )
        assert isinstance(result, list)
        assert len(result) == 2000
        assert all(isinstance(g, str) for g in result)
        # Should NOT store in adata.var when inplace=False
        assert "highly_variable" not in adata.var.columns

    def test_dense_input(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata(sparse=False)
        highly_variable_genes_cell_ranger(adata)
        assert adata.var["highly_variable"].sum() == 2000

    def test_n_bins_parameter(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata()
        highly_variable_genes_cell_ranger(adata, n_bins=10)
        assert adata.var["highly_variable"].sum() == 2000

    def test_means_positive(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata()
        highly_variable_genes_cell_ranger(adata)
        assert (adata.var["means"] >= 0).all()

    def test_fewer_genes_than_requested(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata(n_genes=100)
        highly_variable_genes_cell_ranger(
            adata, n_top_genes=200
        )
        # Should select all genes when fewer than requested
        assert adata.var["highly_variable"].sum() == 100

    def test_type_error(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        with pytest.raises(TypeError, match="AnnData"):
            highly_variable_genes_cell_ranger("not_adata")

    def test_dispersions_dtype(self):
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata = _make_adata()
        highly_variable_genes_cell_ranger(adata)
        assert adata.var["means"].dtype == np.float32
        assert adata.var["dispersions"].dtype == np.float32
        assert (
            adata.var["dispersions_norm"].dtype == np.float32
        )

    def test_public_api(self):
        import singlet

        assert hasattr(
            singlet, "highly_variable_genes_cell_ranger"
        )
        assert callable(
            singlet.highly_variable_genes_cell_ranger
        )

    def test_different_from_seurat(self):
        """Cell Ranger (median/MAD) differs from Seurat (mean/std)."""
        from singlet._hvg import highly_variable_genes
        from singlet._hvg_cell_ranger import (
            highly_variable_genes_cell_ranger,
        )

        adata1 = _make_adata()
        adata2 = _make_adata()
        highly_variable_genes_cell_ranger(adata1)
        highly_variable_genes(adata2)
        # The normalized dispersions should differ
        assert not np.allclose(
            adata1.var["dispersions_norm"].values,
            adata2.var["dispersions_norm"].values,
        )
