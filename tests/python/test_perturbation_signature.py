"""Tests for singlet.perturbation_signature()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._perturbation_signature import perturbation_signature


def _make_perturb_adata(n_cells_per_cond=60, n_genes=80, n_conditions=3, seed=42):
    """Create AnnData with conditions including a control."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    n_cells = n_cells_per_cond * (n_conditions + 1)  # +1 for control

    # Control: baseline expression
    X_control = rng.poisson(5, size=(n_cells_per_cond, n_genes)).astype(np.float32)

    # Treatments: shift some genes
    X_parts = [X_control]
    conditions = ["control"] * n_cells_per_cond

    for cond_idx in range(n_conditions):
        X_treat = rng.poisson(5, size=(n_cells_per_cond, n_genes)).astype(np.float32)
        # Upregulate first few genes in each treatment
        n_de = 10 + cond_idx * 5
        X_treat[:, :n_de] += (cond_idx + 1) * 3.0
        X_parts.append(X_treat)
        conditions.extend([f"treatment_{cond_idx + 1}"] * n_cells_per_cond)

    X = np.vstack(X_parts)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs["condition"] = conditions

    return adata


class TestPerturbationSignatureBasic:
    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        adata = _make_perturb_adata()
        result = perturbation_signature(adata, "condition")
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self):
        """DataFrame should have required columns."""
        adata = _make_perturb_adata()
        result = perturbation_signature(adata, "condition")
        assert "gene" in result.columns
        assert "condition" in result.columns
        assert "effect_size" in result.columns
        assert "pvalue" in result.columns
        assert "fdr" in result.columns

    def test_correct_n_rows(self):
        """Should have n_genes × n_conditions rows."""
        adata = _make_perturb_adata(n_genes=80, n_conditions=3)
        result = perturbation_signature(adata, "condition")
        # 3 non-control conditions × 80 genes = 240 rows
        assert len(result) == 80 * 3

    def test_stores_uns_key(self):
        """Should store results in adata.uns."""
        adata = _make_perturb_adata()
        perturbation_signature(adata, "condition")
        assert "perturbation_signature" in adata.uns
        assert isinstance(adata.uns["perturbation_signature"], dict)

    def test_uns_has_condition_keys(self):
        """uns dict should have one key per non-reference condition."""
        adata = _make_perturb_adata(n_conditions=2)
        perturbation_signature(adata, "condition")
        sig = adata.uns["perturbation_signature"]
        assert "treatment_1" in sig
        assert "treatment_2" in sig
        assert "control" not in sig

    def test_detects_upregulated_genes(self):
        """Upregulated genes should have positive effect sizes."""
        adata = _make_perturb_adata(n_conditions=1, n_cells_per_cond=100)
        result = perturbation_signature(adata, "condition")
        # First 10 genes should be significantly upregulated
        top_genes = result.nsmallest(10, "pvalue")
        assert (top_genes["effect_size"] > 0).all()


class TestPerturbationSignatureMethods:
    def test_mean_shift(self):
        """mean_shift should produce positive values for upregulated genes."""
        adata = _make_perturb_adata(n_conditions=1, n_cells_per_cond=100)
        result = perturbation_signature(adata, "condition", method="mean_shift")
        # First 10 genes are shifted up by ~3 counts
        gene0_effect = result[result["gene"] == "GENE0"]["effect_size"].iloc[0]
        assert gene0_effect > 1.0

    def test_fold_change(self):
        """fold_change method should work and return log2 values."""
        adata = _make_perturb_adata(n_conditions=1, n_cells_per_cond=100)
        result = perturbation_signature(adata, "condition", method="fold_change")
        # Upregulated genes should have positive log2FC
        gene0_effect = result[result["gene"] == "GENE0"]["effect_size"].iloc[0]
        assert gene0_effect > 0

    def test_cohen_d(self):
        """cohen_d method should produce standardized effect sizes."""
        adata = _make_perturb_adata(n_conditions=1, n_cells_per_cond=100)
        result = perturbation_signature(adata, "condition", method="cohen_d")
        # Strong effect should have Cohen's d > 0.5
        gene0_effect = result[result["gene"] == "GENE0"]["effect_size"].iloc[0]
        assert gene0_effect > 0.5

    def test_methods_rank_similarly(self):
        """Different methods should roughly agree on top DE genes."""
        adata = _make_perturb_adata(n_conditions=1, n_cells_per_cond=100)
        result_ms = perturbation_signature(adata, "condition", method="mean_shift")
        result_cd = perturbation_signature(adata, "condition", method="cohen_d")
        top_ms = set(result_ms.nsmallest(10, "pvalue")["gene"])
        top_cd = set(result_cd.nsmallest(10, "pvalue")["gene"])
        # At least 5 of top 10 should overlap
        assert len(top_ms & top_cd) >= 5


class TestPerturbationSignatureStats:
    def test_pvalues_range(self):
        """P-values should be between 0 and 1."""
        adata = _make_perturb_adata()
        result = perturbation_signature(adata, "condition")
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()

    def test_fdr_range(self):
        """FDR values should be between 0 and 1."""
        adata = _make_perturb_adata()
        result = perturbation_signature(adata, "condition")
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_fdr_geq_pvalue(self):
        """FDR should be >= p-value (multiple testing correction)."""
        adata = _make_perturb_adata()
        result = perturbation_signature(adata, "condition")
        assert (result["fdr"] >= result["pvalue"] - 1e-10).all()


class TestPerturbationSignatureErrors:
    def test_missing_condition_key(self):
        """Should raise ValueError if condition_key missing."""
        adata = _make_perturb_adata()
        with pytest.raises(ValueError, match="not found in adata.obs"):
            perturbation_signature(adata, "nonexistent")

    def test_missing_reference(self):
        """Should raise ValueError if reference not in conditions."""
        adata = _make_perturb_adata()
        with pytest.raises(ValueError, match="Reference.*not found"):
            perturbation_signature(adata, "condition", reference="missing_ctrl")

    def test_invalid_method(self):
        """Should raise ValueError for invalid method."""
        adata = _make_perturb_adata()
        with pytest.raises(ValueError, match="method must be one of"):
            perturbation_signature(adata, "condition", method="wilcox")

    def test_single_condition(self):
        """Should raise ValueError with only one condition."""
        import anndata as ad

        X = np.ones((20, 10), dtype=np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        adata.obs["condition"] = "control"
        adata.obs_names = [f"c{i}" for i in range(20)]
        adata.var_names = [f"g{i}" for i in range(10)]
        with pytest.raises(ValueError, match="at least 2 conditions"):
            perturbation_signature(adata, "condition")
