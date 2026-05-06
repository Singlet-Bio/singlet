"""Tests for singlet.cell_type_proportions()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from singlet._cell_type_proportions import cell_type_proportions


def _make_adata_with_types():
    """Create test AnnData with cell type and condition annotations."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells = 100
    X = rng.standard_normal((n_cells, 20)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(20)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Cell types: 40 T cells, 30 B cells, 30 Monocytes
    cell_types = ["T cell"] * 40 + ["B cell"] * 30 + ["Monocyte"] * 30
    adata.obs["cell_type"] = pd.Categorical(cell_types)

    # Conditions: sample_1 (first 50 cells), sample_2 (last 50 cells)
    conditions = ["sample_1"] * 50 + ["sample_2"] * 50
    adata.obs["sample"] = pd.Categorical(conditions)

    return adata


class TestCellTypeProportions:
    def test_basic_with_condition(self):
        adata = _make_adata_with_types()
        result = cell_type_proportions(adata, "cell_type", condition_key="sample")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)  # 2 samples × 3 cell types
        assert set(result.columns) == {"B cell", "Monocyte", "T cell"}
        assert set(result.index) == {"sample_1", "sample_2"}

    def test_proportions_sum_to_one(self):
        adata = _make_adata_with_types()
        result = cell_type_proportions(adata, "cell_type", condition_key="sample")
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_no_condition_key(self):
        adata = _make_adata_with_types()
        result = cell_type_proportions(adata, "cell_type")
        assert result.shape == (1, 3)  # single row
        assert result.index[0] == "all"
        # 40/100, 30/100, 30/100
        assert result["T cell"].iloc[0] == pytest.approx(0.4)
        assert result["B cell"].iloc[0] == pytest.approx(0.3)
        assert result["Monocyte"].iloc[0] == pytest.approx(0.3)

    def test_normalize_false(self):
        adata = _make_adata_with_types()
        result = cell_type_proportions(
            adata, "cell_type", condition_key="sample", normalize=False
        )
        # Raw counts
        total = result.sum(axis=1)
        assert total["sample_1"] == 50
        assert total["sample_2"] == 50

    def test_stored_in_uns(self):
        adata = _make_adata_with_types()
        result = cell_type_proportions(adata, "cell_type", condition_key="sample")
        assert "cell_type_proportions" in adata.uns
        pd.testing.assert_frame_equal(adata.uns["cell_type_proportions"], result)

    def test_missing_groupby_raises(self):
        adata = _make_adata_with_types()
        with pytest.raises(KeyError, match="groupby key"):
            cell_type_proportions(adata, "nonexistent")

    def test_missing_condition_key_raises(self):
        adata = _make_adata_with_types()
        with pytest.raises(KeyError, match="condition_key"):
            cell_type_proportions(adata, "cell_type", condition_key="nonexistent")

    def test_type_error_non_adata(self):
        with pytest.raises(TypeError, match="requires an AnnData"):
            cell_type_proportions(np.zeros((10, 5)), "cell_type")

    def test_columns_sorted_alphabetically(self):
        adata = _make_adata_with_types()
        result = cell_type_proportions(adata, "cell_type", condition_key="sample")
        assert list(result.columns) == sorted(result.columns)

    def test_uneven_distribution(self):
        """Test with uneven cell type distribution per sample."""
        import anndata as ad

        rng = np.random.default_rng(0)
        n_cells = 60
        X = rng.standard_normal((n_cells, 10)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(10)]
        adata.obs_names = [f"c{i}" for i in range(n_cells)]

        # Sample A: 20 type1, 10 type2
        # Sample B: 5 type1, 25 type2
        types = ["type1"] * 20 + ["type2"] * 10 + ["type1"] * 5 + ["type2"] * 25
        samples = ["A"] * 30 + ["B"] * 30
        adata.obs["ct"] = pd.Categorical(types)
        adata.obs["sample"] = pd.Categorical(samples)

        result = cell_type_proportions(adata, "ct", condition_key="sample")
        # Sample A: type1=20/30, type2=10/30
        assert result.loc["A", "type1"] == pytest.approx(20 / 30)
        assert result.loc["A", "type2"] == pytest.approx(10 / 30)
        # Sample B: type1=5/30, type2=25/30
        assert result.loc["B", "type1"] == pytest.approx(5 / 30)
        assert result.loc["B", "type2"] == pytest.approx(25 / 30)

    def test_many_conditions(self):
        """Test with many samples."""
        import anndata as ad

        rng = np.random.default_rng(1)
        n_cells = 200
        X = rng.standard_normal((n_cells, 10)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(10)]
        adata.obs_names = [f"c{i}" for i in range(n_cells)]

        # 4 samples of 50 cells each, 2 cell types
        samples = [f"s{i // 50}" for i in range(n_cells)]
        types = rng.choice(["A", "B"], size=n_cells)
        adata.obs["cell_type"] = pd.Categorical(types)
        adata.obs["batch"] = pd.Categorical(samples)

        result = cell_type_proportions(adata, "cell_type", condition_key="batch")
        assert result.shape == (4, 2)
        np.testing.assert_allclose(result.sum(axis=1).values, 1.0, atol=1e-10)

    def test_registered_in_singlet(self):
        assert hasattr(singlet, "cell_type_proportions")
        assert "cell_type_proportions" in singlet.__all__
