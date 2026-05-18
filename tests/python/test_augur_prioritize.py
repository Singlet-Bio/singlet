# SPDX-License-Identifier: MIT
"""Tests for singlet.augur_prioritize."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import singlet


@pytest.fixture
def adata_two_conditions():
    """Create test AnnData with two conditions and multiple cell types."""
    rng = np.random.default_rng(42)
    n_cells = 200
    n_genes = 100

    # Create count matrix with different profiles per condition/cell type
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Make cell types respond differently to condition
    # Type A: strongly perturbed (big expression difference)
    # Type B: weakly perturbed
    cell_types = np.array(["TypeA"] * 100 + ["TypeB"] * 100)
    conditions = np.array(
        ["ctrl"] * 50 + ["treated"] * 50 + ["ctrl"] * 50 + ["treated"] * 50
    )

    # Add strong signal for TypeA
    X[50:100, :20] += 10  # TypeA treated cells have high expression in first 20 genes
    # Add weak signal for TypeB
    X[150:200, :5] += 2  # TypeB treated cells have slight change in 5 genes

    obs = pd.DataFrame(
        {"condition": conditions, "cell_type": cell_types},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata_sparse(adata_two_conditions):
    """Same data but with sparse matrix."""
    adata = adata_two_conditions.copy()
    adata.X = csr_matrix(adata.X)
    return adata


class TestAugurPrioritize:
    def test_basic(self, adata_two_conditions):
        """Test basic functionality."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition"
        )
        assert isinstance(result, pd.DataFrame)
        assert "cell_type" in result.columns
        assert "auc" in result.columns
        assert "mean_accuracy" in result.columns
        assert "n_cells" in result.columns
        assert len(result) == 2  # Two cell types

    def test_stored_in_uns(self, adata_two_conditions):
        """Test results stored in adata.uns."""
        singlet.augur_prioritize(adata_two_conditions, condition_key="condition")
        assert "augur_results" in adata_two_conditions.uns
        stored = adata_two_conditions.uns["augur_results"]
        assert isinstance(stored, pd.DataFrame)
        assert len(stored) == 2

    def test_copy_mode(self, adata_two_conditions):
        """Test copy=True doesn't modify original."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition", copy=True
        )
        assert "augur_results" not in adata_two_conditions.uns
        assert isinstance(result, pd.DataFrame)

    def test_type_a_higher_auc(self, adata_two_conditions):
        """TypeA should have higher AUC than TypeB (more perturbed)."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition"
        )
        auc_a = result.loc[result["cell_type"] == "TypeA", "auc"].values[0]
        auc_b = result.loc[result["cell_type"] == "TypeB", "auc"].values[0]
        assert auc_a > auc_b

    def test_auc_range(self, adata_two_conditions):
        """AUC should be between 0 and 1."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition"
        )
        assert (result["auc"] >= 0).all()
        assert (result["auc"] <= 1).all()

    def test_sparse_input(self, adata_sparse):
        """Test with sparse matrix input."""
        result = singlet.augur_prioritize(adata_sparse, condition_key="condition")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert not result["auc"].isna().all()

    def test_logistic_regression_classifier(self, adata_two_conditions):
        """Test with logistic regression classifier."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition", classifier="lr"
        )
        assert isinstance(result, pd.DataFrame)
        assert not result["auc"].isna().all()

    def test_custom_n_folds(self, adata_two_conditions):
        """Test with different number of CV folds."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition", n_folds=5
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_subsample(self, adata_two_conditions):
        """Test with subsample_size."""
        result = singlet.augur_prioritize(
            adata_two_conditions,
            condition_key="condition",
            subsample_size=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_missing_condition_key(self, adata_two_conditions):
        """Test error on missing condition key."""
        with pytest.raises(KeyError, match="not found"):
            singlet.augur_prioritize(
                adata_two_conditions, condition_key="nonexistent"
            )

    def test_missing_cell_type_key(self, adata_two_conditions):
        """Test error on missing cell type key."""
        with pytest.raises(KeyError, match="not found"):
            singlet.augur_prioritize(
                adata_two_conditions,
                condition_key="condition",
                cell_type_key="nonexistent",
            )

    def test_non_binary_condition(self, adata_two_conditions):
        """Test error when condition has != 2 values."""
        adata_two_conditions.obs["condition"] = np.random.choice(
            ["a", "b", "c"], size=adata_two_conditions.n_obs
        )
        with pytest.raises(ValueError, match="exactly 2 unique values"):
            singlet.augur_prioritize(
                adata_two_conditions, condition_key="condition"
            )

    def test_invalid_classifier(self, adata_two_conditions):
        """Test error on invalid classifier."""
        with pytest.raises(ValueError, match="Unknown classifier"):
            singlet.augur_prioritize(
                adata_two_conditions,
                condition_key="condition",
                classifier="svm",
            )

    def test_sorted_by_auc(self, adata_two_conditions):
        """Test results are sorted by AUC descending."""
        result = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition"
        )
        assert result["auc"].is_monotonic_decreasing

    def test_reproducibility(self, adata_two_conditions):
        """Test same random_state gives same results."""
        r1 = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition", random_state=42, copy=True
        )
        r2 = singlet.augur_prioritize(
            adata_two_conditions, condition_key="condition", random_state=42, copy=True
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_too_few_cells(self):
        """Test cell type with too few cells gets NaN."""
        rng = np.random.default_rng(0)
        X = rng.poisson(2, size=(54, 50)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "condition": ["ctrl"] * 25 + ["treated"] * 25 + ["ctrl", "treated"] * 2,
                "cell_type": ["big"] * 50 + ["tiny"] * 4,
            },
            index=[f"c{i}" for i in range(54)],
        )
        adata = AnnData(X=X, obs=obs)
        result = singlet.augur_prioritize(adata, condition_key="condition", n_folds=3)
        tiny_row = result[result["cell_type"] == "tiny"]
        assert tiny_row["auc"].isna().values[0]
