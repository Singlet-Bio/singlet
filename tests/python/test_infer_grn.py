"""Tests for singlet.infer_grn()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._infer_grn import infer_grn


def _make_grn_adata(n_cells=100, n_genes=80):
    """Create test AnnData with correlated TF-target pairs."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # Create expression matrix
    X = np.zeros((n_cells, n_genes), dtype=np.float32)

    # TF genes: first 10 are TFs (or fewer if n_genes is small)
    n_tfs = min(10, n_genes // 4)
    for g_idx in range(n_tfs):
        X[:, g_idx] = rng.poisson(5, n_cells).astype(np.float32)

    # Target genes: some correlated with TFs
    mid = min(n_tfs + 20, n_genes)
    for g_idx in range(n_tfs, mid):
        # Correlated with TF0
        X[:, g_idx] = X[:, 0] * rng.uniform(0.5, 2.0) + rng.normal(0, 0.5, n_cells)

    end = min(mid + 20, n_genes)
    for g_idx in range(mid, end):
        # Correlated with TF1 (if exists)
        tf_idx = min(1, n_tfs - 1)
        X[:, g_idx] = X[:, tf_idx] * rng.uniform(0.3, 1.5) + rng.normal(0, 0.5, n_cells)

    # Random genes (no strong correlation)
    for g_idx in range(end, n_genes):
        X[:, g_idx] = rng.poisson(3, n_cells).astype(np.float32)

    X = np.maximum(X, 0)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    n_tfs_for_names = min(10, n_genes // 4)
    adata.var_names = [f"TF{idx}" if idx < n_tfs_for_names else f"TARGET{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

    return adata


class TestInferGRN:
    def test_basic_correlation(self):
        """Basic GRN inference with correlation method."""
        adata = _make_grn_adata()
        result = infer_grn(adata, method="correlation", n_top=100)

        assert isinstance(result, pd.DataFrame)
        assert "tf" in result.columns
        assert "target" in result.columns
        assert "weight" in result.columns
        assert "pvalue" in result.columns
        assert "fdr" in result.columns

    def test_stores_in_uns(self):
        """Result stored in adata.uns."""
        adata = _make_grn_adata()
        infer_grn(adata, n_top=50)

        assert "grn" in adata.uns
        assert isinstance(adata.uns["grn"], pd.DataFrame)

    def test_n_top_limits_output(self):
        """Returns at most n_top rows."""
        adata = _make_grn_adata()
        result = infer_grn(adata, n_top=20)
        assert len(result) <= 20

    def test_no_self_links(self):
        """No gene should regulate itself."""
        adata = _make_grn_adata()
        result = infer_grn(adata, n_top=200)
        self_links = result[result["tf"] == result["target"]]
        assert len(self_links) == 0

    def test_tf_list_filter(self):
        """Only specified TFs appear as regulators."""
        adata = _make_grn_adata()
        tf_list = ["TF0", "TF1", "TF2"]
        result = infer_grn(adata, tf_list=tf_list, n_top=100)
        assert set(result["tf"].unique()).issubset(set(tf_list))

    def test_target_genes_filter(self):
        """Only specified targets appear."""
        adata = _make_grn_adata()
        targets = ["TARGET10", "TARGET11", "TARGET12"]
        result = infer_grn(adata, target_genes=targets, n_top=50)
        # Targets in the result should be subset of specified + TFs (since TFs excluded from self)
        assert set(result["target"].unique()).issubset(set(targets))

    def test_highly_variable_used(self):
        """If highly_variable in var, uses those as targets."""
        adata = _make_grn_adata()
        # Mark first 30 genes as highly variable
        adata.var["highly_variable"] = False
        adata.var.iloc[:30, adata.var.columns.get_loc("highly_variable")] = True
        result = infer_grn(adata, n_top=50)
        # Target genes should be from highly variable set
        hv_genes = set(adata.var_names[adata.var["highly_variable"]])
        assert set(result["target"].unique()).issubset(hv_genes)

    def test_mutual_info_method(self):
        """Mutual information method works."""
        adata = _make_grn_adata(n_cells=50, n_genes=20)
        result = infer_grn(adata, method="mutual_info", n_top=30)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # MI weights should be non-negative
        assert (result["weight"] >= 0).all()

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        adata = _make_grn_adata()
        with pytest.raises(ValueError, match="method must be"):
            infer_grn(adata, method="invalid")

    def test_empty_tf_list_raises(self):
        """Empty effective tf_list raises ValueError."""
        adata = _make_grn_adata()
        with pytest.raises(ValueError, match="No genes from tf_list"):
            infer_grn(adata, tf_list=["NONEXISTENT_GENE"])

    def test_empty_target_list_raises(self):
        """Empty effective target_genes raises ValueError."""
        adata = _make_grn_adata()
        with pytest.raises(ValueError, match="No genes from target_genes"):
            infer_grn(adata, target_genes=["NONEXISTENT_GENE"])

    def test_correlated_genes_rank_high(self):
        """Strongly correlated TF-target pairs should rank high."""
        adata = _make_grn_adata()
        tf_list = ["TF0", "TF1"]
        result = infer_grn(adata, tf_list=tf_list, n_top=50)

        # Top results should include targets correlated with TF0 and TF1
        top_targets = set(result.head(20)["target"].values)
        expected_targets = {f"TARGET{idx}" for idx in range(10, 50)}
        overlap = top_targets & expected_targets
        assert len(overlap) >= 5

    def test_pvalues_in_range(self):
        """P-values and FDR should be in [0, 1]."""
        adata = _make_grn_adata()
        result = infer_grn(adata, n_top=50)
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_dense_matrix(self):
        """Works with dense X matrix."""
        adata = _make_grn_adata()
        adata.X = np.asarray(adata.X.todense())
        result = infer_grn(adata, n_top=50)
        assert len(result) > 0

    def test_sorted_by_abs_weight(self):
        """Results sorted by absolute weight descending."""
        adata = _make_grn_adata()
        result = infer_grn(adata, n_top=50)
        abs_weights = result["weight"].abs().values
        assert np.all(abs_weights[:-1] >= abs_weights[1:])

    def test_singlet_api(self):
        """Function accessible via singlet namespace."""
        adata = _make_grn_adata()
        result = singlet.infer_grn(adata, n_top=20)
        assert isinstance(result, pd.DataFrame)
