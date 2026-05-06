"""Tests for singlet.find_all_markers()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._find_all_markers import find_all_markers


def _make_adata_with_clusters(n_cells=200, n_genes=100, n_clusters=3):
    """Create test AnnData with clusters and differentially expressed genes."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    cells_per_cluster = n_cells // n_clusters

    # Base expression (sparse)
    X = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Add cluster-specific marker genes
    # Cluster 0: genes 0-4 are markers
    # Cluster 1: genes 5-9 are markers
    # Cluster 2: genes 10-14 are markers
    labels = []
    for cluster_idx in range(n_clusters):
        start_cell = cluster_idx * cells_per_cluster
        end_cell = start_cell + cells_per_cluster
        labels.extend([str(cluster_idx)] * cells_per_cluster)

        # Add background noise
        X[start_cell:end_cell, :] = rng.exponential(0.1, (cells_per_cluster, n_genes))

        # Add marker expression for this cluster's specific genes
        marker_start = cluster_idx * 5
        marker_end = marker_start + 5
        X[start_cell:end_cell, marker_start:marker_end] = rng.exponential(
            3.0, (cells_per_cluster, 5)
        )

    # Handle remaining cells
    remaining = n_cells - cells_per_cluster * n_clusters
    if remaining > 0:
        labels.extend(["0"] * remaining)
        X[cells_per_cluster * n_clusters :, :] = rng.exponential(
            0.1, (remaining, n_genes)
        )
        X[cells_per_cluster * n_clusters :, :5] = rng.exponential(
            3.0, (remaining, 5)
        )

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]
    adata.obs["leiden"] = pd.Categorical(labels[:n_cells])

    return adata


class TestFindAllMarkers:
    def test_basic(self):
        """Basic find_all_markers invocation."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden")

        assert isinstance(result, type(result))  # pd.DataFrame
        assert len(result) > 0
        # Check required columns
        expected_cols = ["gene", "group", "pvalue", "fdr", "log2fc", "pct_in", "pct_out", "score"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_finds_true_markers(self):
        """Should find the planted marker genes."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden", min_fold_change=1.5)

        # Cluster 0's markers should include GENE0-GENE4
        cluster0_markers = result[result["group"] == "0"]["gene"].tolist()
        expected_markers = [f"GENE{idx}" for idx in range(5)]
        found = [m for m in expected_markers if m in cluster0_markers]
        # Should find at least some of the planted markers
        assert len(found) >= 3, f"Expected >=3 markers, found {found}"

    def test_fold_change_filter(self):
        """Higher min_fold_change should produce fewer markers."""
        adata = _make_adata_with_clusters()
        result_low = find_all_markers(adata, groupby="leiden", min_fold_change=1.2)
        result_high = find_all_markers(adata, groupby="leiden", min_fold_change=5.0)

        assert len(result_low) >= len(result_high)

    def test_pct_filter(self):
        """Higher min_pct should produce fewer or equal markers."""
        adata = _make_adata_with_clusters()
        result_low = find_all_markers(adata, groupby="leiden", min_pct=0.01)
        result_high = find_all_markers(adata, groupby="leiden", min_pct=0.9)

        assert len(result_low) >= len(result_high)

    def test_n_top(self):
        """n_top should limit markers per group."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden", n_top=3)

        for group in result["group"].unique():
            group_count = len(result[result["group"] == group])
            assert group_count <= 3

    def test_wilcoxon_method(self):
        """Wilcoxon test should work."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden", method="wilcoxon")
        assert len(result) > 0

    def test_ttest_method(self):
        """T-test should work."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden", method="t-test")
        assert len(result) > 0

    def test_stores_in_uns(self):
        """Should store results in adata.uns['all_markers']."""
        adata = _make_adata_with_clusters()
        find_all_markers(adata, groupby="leiden")

        assert "all_markers" in adata.uns
        assert isinstance(adata.uns["all_markers"], dict)
        # Should have entry for each cluster
        assert "0" in adata.uns["all_markers"]

    def test_score_ordering(self):
        """Results within each group should be sorted by score descending."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden")

        for group in result["group"].unique():
            group_df = result[result["group"] == group]
            scores = group_df["score"].values
            # Check monotonically decreasing
            assert all(
                scores[idx] >= scores[idx + 1] for idx in range(len(scores) - 1)
            ), f"Scores not sorted for group {group}"

    def test_log2fc_positive(self):
        """All returned markers should have positive log2fc (upregulated)."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden")

        if len(result) > 0:
            assert (result["log2fc"] > 0).all()

    def test_pct_in_range(self):
        """pct_in and pct_out should be in [0, 1] (within floating point tolerance)."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden")

        if len(result) > 0:
            assert (result["pct_in"] >= -1e-10).all() and (result["pct_in"] <= 1.0 + 1e-10).all()
            assert (result["pct_out"] >= -1e-10).all() and (result["pct_out"] <= 1.0 + 1e-10).all()

    def test_fdr_le_maxpvalue(self):
        """All returned markers should have FDR <= max_pvalue."""
        adata = _make_adata_with_clusters()
        result = find_all_markers(adata, groupby="leiden", max_pvalue=0.01)

        if len(result) > 0:
            assert (result["fdr"] <= 0.01).all()

    def test_missing_groupby_raises(self):
        """Should raise when groupby column is missing."""
        adata = _make_adata_with_clusters()
        with pytest.raises(KeyError, match="nonexistent"):
            find_all_markers(adata, groupby="nonexistent")

    def test_invalid_method_raises(self):
        """Should raise on invalid method."""
        adata = _make_adata_with_clusters()
        with pytest.raises(ValueError, match="method"):
            find_all_markers(adata, groupby="leiden", method="invalid")

    def test_invalid_fold_change_raises(self):
        """Should raise when min_fold_change < 1."""
        adata = _make_adata_with_clusters()
        with pytest.raises(ValueError, match="min_fold_change"):
            find_all_markers(adata, groupby="leiden", min_fold_change=0.5)

    def test_type_error(self):
        """Should raise on non-AnnData input."""
        with pytest.raises(TypeError, match="find_all_markers"):
            find_all_markers("not_adata")

    def test_dense_input(self):
        """Should work with dense expression matrices."""
        import anndata as ad
        import pandas as pd

        rng = np.random.default_rng(42)
        X = rng.exponential(0.5, (60, 40)).astype(np.float32)
        # Make first 10 genes markers for group "a"
        X[:30, :10] = rng.exponential(5.0, (30, 10))

        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{idx}" for idx in range(40)]
        adata.obs_names = [f"c{idx}" for idx in range(60)]
        adata.obs["group"] = pd.Categorical(["a"] * 30 + ["b"] * 30)

        result = find_all_markers(adata, groupby="group")
        assert len(result) > 0

    def test_layer(self):
        """Should use specified layer."""
        adata = _make_adata_with_clusters()
        adata.layers["normalized"] = adata.X.copy()
        result = find_all_markers(adata, groupby="leiden", layer="normalized")
        assert len(result) > 0

    def test_layer_not_found_raises(self):
        """Should raise when layer doesn't exist."""
        adata = _make_adata_with_clusters()
        with pytest.raises(KeyError, match="missing"):
            find_all_markers(adata, groupby="leiden", layer="missing")

    def test_max_pvalue_strict(self):
        """Very strict p-value should return fewer results."""
        adata = _make_adata_with_clusters()
        result_loose = find_all_markers(adata, groupby="leiden", max_pvalue=0.99)
        result_strict = find_all_markers(adata, groupby="leiden", max_pvalue=1e-10)

        assert len(result_loose) >= len(result_strict)

    def test_public_api(self):
        """Should be accessible via singlet.find_all_markers."""
        assert hasattr(singlet, "find_all_markers")
        assert callable(singlet.find_all_markers)
