# SPDX-License-Identifier: MIT
"""Tests for singlet.marker_specificity()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._marker_specificity import marker_specificity


def _make_marker_adata(n_cells=200, n_genes=100, n_clusters=3, seed=42):
    """Create AnnData with clear markers per cluster."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    cells_per_cluster = n_cells // n_clusters
    X = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)

    # Add strong markers: each cluster has specific genes highly expressed
    genes_per_cluster = n_genes // n_clusters
    labels = []
    for k in range(n_clusters):
        start = k * cells_per_cluster
        end = start + cells_per_cluster
        gene_start = k * genes_per_cluster
        gene_end = gene_start + genes_per_cluster
        X[start:end, gene_start:gene_end] += 10.0
        labels.extend([str(k)] * cells_per_cluster)

    # Handle remaining cells
    remaining = n_cells - n_clusters * cells_per_cluster
    labels.extend([str(n_clusters - 1)] * remaining)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs["leiden"] = pd.Categorical(labels)

    # Log-normalize
    import scanpy as sc

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    return adata


class TestMarkerSpecificityBasic:
    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        """DataFrame should have expected columns."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden")
        expected_cols = {"gene", "group", "auroc", "specificity", "detection_rate", "log2fc"}
        assert expected_cols.issubset(set(result.columns))

    def test_stores_in_uns(self):
        """Should store results in adata.uns['marker_specificity']."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden")
        assert "marker_specificity" in adata.uns
        pd.testing.assert_frame_equal(adata.uns["marker_specificity"], result)

    def test_n_top_genes_respected(self):
        """Should return at most n_top_genes per group."""
        adata = _make_marker_adata(n_clusters=3)
        result = marker_specificity(adata, groupby="leiden", n_top_genes=10)
        for grp in result["group"].unique():
            grp_rows = result[result["group"] == grp]
            assert len(grp_rows) <= 10

    def test_all_groups_present(self):
        """Each group should have markers reported."""
        adata = _make_marker_adata(n_clusters=4)
        result = marker_specificity(adata, groupby="leiden")
        groups_in_result = set(result["group"].unique())
        groups_in_obs = set(adata.obs["leiden"].unique())
        assert groups_in_result == groups_in_obs

    def test_auroc_range(self):
        """AUROC values should be in [0, 1]."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden", method="auroc")
        assert result["auroc"].min() >= 0.0
        assert result["auroc"].max() <= 1.0

    def test_top_markers_are_cluster_specific(self):
        """Top markers per cluster should have high AUROC for well-separated data."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden", n_top_genes=5)
        # Top markers should have AUROC > 0.7 for well-separated clusters
        top_aurocs = result.groupby("group")["auroc"].max()
        assert all(top_aurocs > 0.7)

    def test_specificity_positive(self):
        """Specificity values should be positive."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden")
        assert all(result["specificity"] > 0)


class TestMarkerSpecificityMethods:
    def test_cohen_d_method(self):
        """Cohen's d method should work and have different column."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden", method="cohen_d")
        assert "cohen_d" in result.columns
        assert "auroc" not in result.columns

    def test_cohen_d_positive_for_markers(self):
        """Cohen's d should be positive for true markers."""
        adata = _make_marker_adata()
        result = marker_specificity(adata, groupby="leiden", method="cohen_d")
        # Top markers should have positive Cohen's d
        top_scores = result.groupby("group")["cohen_d"].max()
        assert all(top_scores > 0)


class TestMarkerSpecificityEdgeCases:
    def test_dense_matrix(self):
        """Should work with dense matrix."""
        adata = _make_marker_adata()
        adata.X = adata.X.toarray()
        result = marker_specificity(adata, groupby="leiden")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_two_groups(self):
        """Should work with only two groups."""
        adata = _make_marker_adata(n_clusters=2)
        result = marker_specificity(adata, groupby="leiden")
        assert len(result["group"].unique()) == 2

    def test_n_top_genes_larger_than_available(self):
        """If n_top_genes > n_genes, should return all genes."""
        adata = _make_marker_adata(n_genes=20)
        result = marker_specificity(adata, groupby="leiden", n_top_genes=100)
        for grp in result["group"].unique():
            grp_rows = result[result["group"] == grp]
            assert len(grp_rows) <= 20


class TestMarkerSpecificityErrors:
    def test_invalid_method(self):
        """Should raise ValueError for unknown method."""
        adata = _make_marker_adata()
        with pytest.raises(ValueError, match="method must be"):
            marker_specificity(adata, method="invalid")

    def test_missing_groupby(self):
        """Should raise KeyError if groupby key not in obs."""
        adata = _make_marker_adata()
        with pytest.raises(KeyError, match="not found"):
            marker_specificity(adata, groupby="nonexistent")
