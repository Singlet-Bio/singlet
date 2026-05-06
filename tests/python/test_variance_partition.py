"""Tests for singlet.variance_partition()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._variance_partition import variance_partition


def _make_structured_adata(n_cells=200, n_genes=100, seed=42):
    """Create AnnData with known variance structure."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Assign cells to batches and cell types
    n_batches = 3
    n_types = 4
    batch = np.array([f"batch_{i % n_batches}" for i in range(n_cells)])
    cell_type = np.array([f"type_{i % n_types}" for i in range(n_cells)])

    # Base expression
    X = rng.normal(5, 1, size=(n_cells, n_genes)).astype(np.float32)

    # Add batch effects to genes 0-19
    for b_idx in range(n_batches):
        mask = batch == f"batch_{b_idx}"
        X[mask, :20] += (b_idx - 1) * 5.0

    # Add cell type effects to genes 20-49
    for ct_idx in range(n_types):
        mask = cell_type == f"type_{ct_idx}"
        X[mask, 20:50] += (ct_idx - 1.5) * 5.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["cell_type"] = pd.Categorical(cell_type)

    return adata


class TestVariancePartition:
    """Test suite for variance_partition."""

    def test_basic_output(self):
        """Test basic output structure."""
        adata = _make_structured_adata()

        result = variance_partition(adata, keys=["batch", "cell_type"])

        assert isinstance(result, pd.DataFrame)
        assert "batch" in result.columns
        assert "cell_type" in result.columns
        assert "residual" in result.columns
        assert len(result) > 0

    def test_stored_in_uns(self):
        """Test that result is stored in adata.uns."""
        adata = _make_structured_adata()

        result = variance_partition(adata, keys=["batch", "cell_type"])

        assert "variance_partition" in adata.uns
        pd.testing.assert_frame_equal(adata.uns["variance_partition"], result)

    def test_values_sum_to_one(self):
        """Test that variance fractions sum to approximately 1."""
        adata = _make_structured_adata()

        result = variance_partition(adata, keys=["batch", "cell_type"])

        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_values_non_negative(self):
        """Test that all values are non-negative."""
        adata = _make_structured_adata()

        result = variance_partition(adata, keys=["batch", "cell_type"])

        assert (result >= 0).all().all()

    def test_batch_genes_high_batch_variance(self):
        """Test that batch-affected genes show high batch variance."""
        adata = _make_structured_adata()

        result = variance_partition(
            adata, keys=["batch", "cell_type"], genes=[f"GENE{i}" for i in range(20)]
        )

        # Genes 0-19 have strong batch effects
        mean_batch_var = result["batch"].mean()
        assert mean_batch_var > 0.3

    def test_celltype_genes_high_type_variance(self):
        """Test that cell type-affected genes show high cell_type variance."""
        adata = _make_structured_adata()

        result = variance_partition(
            adata, keys=["batch", "cell_type"], genes=[f"GENE{i}" for i in range(20, 50)]
        )

        # Genes 20-49 have strong cell type effects
        mean_type_var = result["cell_type"].mean()
        assert mean_type_var > 0.3

    def test_n_top_genes(self):
        """Test n_top_genes parameter limits output size."""
        adata = _make_structured_adata()

        result = variance_partition(adata, keys=["batch"], n_top_genes=30)

        assert len(result) == 30

    def test_explicit_genes(self):
        """Test explicit gene list."""
        adata = _make_structured_adata()
        gene_list = ["GENE0", "GENE10", "GENE50"]

        result = variance_partition(adata, keys=["batch"], genes=gene_list)

        assert len(result) == 3
        assert set(result.index) == set(gene_list)

    def test_single_key(self):
        """Test with single key."""
        adata = _make_structured_adata()

        result = variance_partition(adata, keys=["batch"], n_top_genes=50)

        assert "batch" in result.columns
        assert "residual" in result.columns
        assert len(result.columns) == 2

    def test_highly_variable_genes_used(self):
        """Test that highly_variable genes are used when available."""
        adata = _make_structured_adata()
        adata.var["highly_variable"] = False
        adata.var.iloc[:25, adata.var.columns.get_loc("highly_variable")] = True

        result = variance_partition(adata, keys=["batch"], n_top_genes=50)

        # Should use at most 25 genes (all HVGs)
        assert len(result) == 25

    def test_missing_key_raises(self):
        """Test that missing key raises KeyError."""
        adata = _make_structured_adata()

        with pytest.raises(KeyError, match="nonexistent"):
            variance_partition(adata, keys=["nonexistent"])

    def test_empty_keys_raises(self):
        """Test that empty keys raises ValueError."""
        adata = _make_structured_adata()

        with pytest.raises(ValueError, match="non-empty"):
            variance_partition(adata, keys=[])

    def test_dense_input(self):
        """Test with dense matrix input."""
        import anndata as ad

        rng = np.random.default_rng(99)
        X = rng.normal(5, 2, size=(100, 50)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{i}" for i in range(50)]
        adata.obs["group"] = pd.Categorical([f"g{i % 3}" for i in range(100)])

        result = variance_partition(adata, keys=["group"], n_top_genes=20)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_layer_parameter(self):
        """Test using a specific layer."""
        adata = _make_structured_adata()
        adata.layers["normalized"] = adata.X.copy()

        result = variance_partition(adata, keys=["batch"], layer="normalized", n_top_genes=20)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_residual_dominates_noise_genes(self):
        """Test that genes without structure have high residual."""
        adata = _make_structured_adata()

        # Genes 50-99 have no batch or cell type effect
        result = variance_partition(
            adata, keys=["batch", "cell_type"], genes=[f"GENE{i}" for i in range(50, 70)]
        )

        mean_residual = result["residual"].mean()
        assert mean_residual > 0.5
