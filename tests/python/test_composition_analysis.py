# SPDX-License-Identifier: MIT
"""Tests for singlet.composition_analysis()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._composition_analysis import composition_analysis


def _make_composition_adata(n_cells=300, n_genes=50):
    """Create test AnnData with cell types and conditions."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.poisson(2, (n_cells, n_genes)).astype(np.float32))

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

    # Create cell types with different proportions in disease vs control
    # Control: 50% TypeA, 30% TypeB, 20% TypeC
    # Disease: 30% TypeA, 50% TypeB, 20% TypeC (TypeB enriched)
    half = n_cells // 2
    control_types = rng.choice(
        ["TypeA", "TypeB", "TypeC"], size=half, p=[0.5, 0.3, 0.2]
    )
    disease_types = rng.choice(
        ["TypeA", "TypeB", "TypeC"], size=n_cells - half, p=[0.3, 0.5, 0.2]
    )

    cell_types = np.concatenate([control_types, disease_types])
    conditions = np.array(["control"] * half + ["disease"] * (n_cells - half))

    adata.obs["cell_type"] = pd.Categorical(cell_types)
    adata.obs["condition"] = pd.Categorical(conditions)

    return adata


class TestCompositionAnalysis:
    def test_basic(self):
        """Basic composition analysis invocation."""
        adata = _make_composition_adata()
        result = composition_analysis(adata, "cell_type", "condition")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        expected_cols = ["group", "condition", "proportion", "fold_change", "pvalue", "fdr"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_groups_present(self):
        """Result should contain all cell types and conditions."""
        adata = _make_composition_adata()
        result = composition_analysis(adata, "cell_type", "condition")

        assert set(result["group"].unique()) == {"TypeA", "TypeB", "TypeC"}
        assert set(result["condition"].unique()) == {"control", "disease"}

    def test_proportions_sum_to_one(self):
        """Proportions within each condition should sum to ~1."""
        adata = _make_composition_adata()
        result = composition_analysis(adata, "cell_type", "condition")

        for cond in result["condition"].unique():
            cond_df = result[result["condition"] == cond]
            total = cond_df["proportion"].sum()
            assert abs(total - 1.0) < 1e-6, f"Proportions sum to {total} for {cond}"

    def test_fold_change_reference(self):
        """Fold change for reference condition should be 1.0."""
        adata = _make_composition_adata()
        result = composition_analysis(adata, "cell_type", "condition")

        # Default reference is first alphabetically = 'control'
        ref_rows = result[result["condition"] == "control"]
        assert (ref_rows["fold_change"] == 1.0).all()

    def test_custom_reference(self):
        """Custom reference condition should work."""
        adata = _make_composition_adata()
        result = composition_analysis(
            adata, "cell_type", "condition", reference="disease"
        )

        ref_rows = result[result["condition"] == "disease"]
        assert (ref_rows["fold_change"] == 1.0).all()

    def test_dirichlet_method(self):
        """Dirichlet method should produce p-values."""
        adata = _make_composition_adata()
        result = composition_analysis(
            adata, "cell_type", "condition", method="dirichlet"
        )

        non_ref = result[result["condition"] != "control"]
        assert (non_ref["pvalue"] >= 0).all()
        assert (non_ref["pvalue"] <= 1).all()

    def test_prop_test_method(self):
        """Proportion z-test method should work."""
        adata = _make_composition_adata()
        result = composition_analysis(
            adata, "cell_type", "condition", method="prop_test"
        )

        assert len(result) > 0
        non_ref = result[result["condition"] != "control"]
        assert (non_ref["pvalue"] >= 0).all()
        assert (non_ref["pvalue"] <= 1).all()

    def test_fdr_correction(self):
        """FDR values should be >= p-values."""
        adata = _make_composition_adata()
        result = composition_analysis(adata, "cell_type", "condition")

        non_ref = result[result["condition"] != "control"]
        assert (non_ref["fdr"] >= non_ref["pvalue"] - 1e-10).all()
        assert (non_ref["fdr"] <= 1.0).all()

    def test_stores_in_uns(self):
        """Should store results in adata.uns['composition_analysis']."""
        adata = _make_composition_adata()
        composition_analysis(adata, "cell_type", "condition")

        assert "composition_analysis" in adata.uns
        assert isinstance(adata.uns["composition_analysis"], pd.DataFrame)

    def test_detects_enrichment(self):
        """Should detect TypeB enrichment in disease with large sample."""
        import anndata as ad

        rng = np.random.default_rng(42)
        n_cells = 1000
        n_genes = 20
        X = sp.csr_matrix(rng.poisson(2, (n_cells, n_genes)).astype(np.float32))

        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{idx}" for idx in range(n_genes)]
        adata.obs_names = [f"c{idx}" for idx in range(n_cells)]

        half = n_cells // 2
        # Strong composition shift
        control_types = rng.choice(["A", "B"], size=half, p=[0.8, 0.2])
        disease_types = rng.choice(["A", "B"], size=half, p=[0.2, 0.8])

        adata.obs["cell_type"] = pd.Categorical(
            np.concatenate([control_types, disease_types])
        )
        adata.obs["condition"] = pd.Categorical(
            ["control"] * half + ["disease"] * half
        )

        result = composition_analysis(adata, "cell_type", "condition")
        # TypeB in disease should have fold_change > 1 and significant p-value
        typeb_disease = result[
            (result["group"] == "B") & (result["condition"] == "disease")
        ]
        assert typeb_disease["fold_change"].values[0] > 2.0
        assert typeb_disease["pvalue"].values[0] < 0.05

    def test_missing_groupby_raises(self):
        """Should raise KeyError when groupby column missing."""
        adata = _make_composition_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            composition_analysis(adata, "nonexistent", "condition")

    def test_missing_condition_raises(self):
        """Should raise KeyError when condition_key missing."""
        adata = _make_composition_adata()
        with pytest.raises(KeyError, match="missing_cond"):
            composition_analysis(adata, "cell_type", "missing_cond")

    def test_invalid_method_raises(self):
        """Should raise ValueError on invalid method."""
        adata = _make_composition_adata()
        with pytest.raises(ValueError, match="method"):
            composition_analysis(adata, "cell_type", "condition", method="invalid")

    def test_single_condition_raises(self):
        """Should raise ValueError with fewer than 2 conditions."""
        adata = _make_composition_adata()
        adata.obs["single_cond"] = "only_one"
        with pytest.raises(ValueError, match="at least 2"):
            composition_analysis(adata, "cell_type", "single_cond")

    def test_invalid_reference_raises(self):
        """Should raise ValueError when reference not in conditions."""
        adata = _make_composition_adata()
        with pytest.raises(ValueError, match="reference"):
            composition_analysis(
                adata, "cell_type", "condition", reference="nonexistent"
            )

    def test_type_error(self):
        """Should raise TypeError on non-AnnData input."""
        with pytest.raises(TypeError, match="composition_analysis"):
            composition_analysis("not_adata", "cell_type", "condition")

    def test_multiple_conditions(self):
        """Should work with more than 2 conditions."""
        import anndata as ad

        rng = np.random.default_rng(42)
        n_cells = 300
        n_genes = 20
        X = sp.csr_matrix(rng.poisson(2, (n_cells, n_genes)).astype(np.float32))

        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{idx}" for idx in range(n_genes)]
        adata.obs_names = [f"c{idx}" for idx in range(n_cells)]

        third = n_cells // 3
        adata.obs["cell_type"] = pd.Categorical(
            rng.choice(["A", "B"], size=n_cells, p=[0.6, 0.4])
        )
        adata.obs["condition"] = pd.Categorical(
            ["cond1"] * third + ["cond2"] * third + ["cond3"] * (n_cells - 2 * third)
        )

        result = composition_analysis(adata, "cell_type", "condition")
        assert set(result["condition"].unique()) == {"cond1", "cond2", "cond3"}

    def test_public_api(self):
        """Should be accessible via singlet.composition_analysis."""
        assert hasattr(singlet, "composition_analysis")
        assert callable(singlet.composition_analysis)
