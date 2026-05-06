"""Tests for singlet.describe()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._describe import describe


def _make_adata(n_cells=100, n_genes=500, sparse=True, organism=None):
    """Create a test AnnData object."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(n_cells, n_genes, density=0.1, format="csr", random_state=42)
        X.data = rng.integers(1, 100, size=X.nnz).astype(np.float32)
    else:
        X = rng.integers(0, 100, size=(n_cells, n_genes)).astype(np.float32)
        # Make ~90% zeros to simulate sparsity
        mask = rng.random((n_cells, n_genes)) < 0.9
        X[mask] = 0

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    if organism:
        adata.uns["organism"] = organism
    return adata


class TestDescribe:
    def test_basic_sparse(self):
        adata = _make_adata(sparse=True)
        result = describe(adata)
        assert result["n_cells"] == 100
        assert result["n_genes"] == 500
        assert 0 < result["sparsity"] < 1
        assert result["counts_per_cell"]["min"] >= 0
        assert result["counts_per_cell"]["max"] > 0
        assert result["genes_per_cell"]["min"] >= 0
        assert result["genes_per_cell"]["max"] > 0

    def test_basic_dense(self):
        adata = _make_adata(sparse=False)
        result = describe(adata)
        assert result["n_cells"] == 100
        assert result["n_genes"] == 500
        assert 0 < result["sparsity"] < 1

    def test_organism_detected(self):
        adata = _make_adata(organism="Homo sapiens")
        result = describe(adata)
        assert result["organism"] == "Homo sapiens"

    def test_organism_none_when_absent(self):
        adata = _make_adata()
        result = describe(adata)
        assert result["organism"] is None

    def test_obs_var_columns(self):
        adata = _make_adata()
        adata.obs["batch"] = "A"
        adata.var["highly_variable"] = True
        result = describe(adata)
        assert "batch" in result["obs_columns"]
        assert "highly_variable" in result["var_columns"]

    def test_layers_listed(self):
        adata = _make_adata()
        adata.layers["raw"] = adata.X.copy()
        result = describe(adata)
        assert "raw" in result["layers"]

    def test_empty_adata(self):
        """Handles 0-cell AnnData gracefully."""
        import anndata as ad

        adata = ad.AnnData(X=sp.csr_matrix((0, 100)))
        adata.var_names = [f"GENE{i}" for i in range(100)]
        result = describe(adata)
        assert result["n_cells"] == 0
        assert result["n_genes"] == 100
        assert result["sparsity"] == 0.0

    def test_type_error_on_non_adata(self):
        with pytest.raises(TypeError, match="describe"):
            describe("not_adata")

    def test_public_api(self):
        """describe is accessible from singlet namespace."""
        assert hasattr(singlet, "describe")
        assert callable(singlet.describe)

    def test_return_types(self):
        """All values have expected types."""
        adata = _make_adata()
        result = describe(adata)
        assert isinstance(result["n_cells"], int)
        assert isinstance(result["n_genes"], int)
        assert isinstance(result["sparsity"], float)
        assert isinstance(result["counts_per_cell"], dict)
        assert isinstance(result["genes_per_cell"], dict)
        assert isinstance(result["obs_columns"], list)
        assert isinstance(result["var_columns"], list)
        assert isinstance(result["layers"], list)
