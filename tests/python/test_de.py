"""Tests for singlet.rank_genes_groups() and rank_genes_groups_df()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._de import rank_genes_groups, rank_genes_groups_df


def _make_clustered_adata(n_cells=60, n_genes=100):
    """Create AnnData with known cluster structure and marker genes."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # 3 clusters of 20 cells each
    # Cluster 0: genes 0-9 are highly expressed
    # Cluster 1: genes 10-19 are highly expressed
    # Cluster 2: genes 20-29 are highly expressed
    X = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)

    for i in range(20):
        X[i, :10] += rng.poisson(20, size=10)  # cluster 0 markers
    for i in range(20, 40):
        X[i, 10:20] += rng.poisson(20, size=10)  # cluster 1 markers
    for i in range(40, 60):
        X[i, 20:30] += rng.poisson(20, size=10)  # cluster 2 markers

    X_sparse = sp.csr_matrix(X)
    adata = ad.AnnData(X=X_sparse)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["cluster"] = pd.Categorical(["0"] * 20 + ["1"] * 20 + ["2"] * 20)
    return adata


class TestRankGenesGroups:
    def test_basic_inplace(self):
        adata = _make_clustered_adata()
        ret = rank_genes_groups(adata, "cluster")
        assert ret is None
        assert "rank_genes_groups" in adata.uns
        result = adata.uns["rank_genes_groups"]
        assert "0" in result["names"]
        assert "1" in result["names"]
        assert "2" in result["names"]

    def test_finds_correct_markers(self):
        """Top markers for each cluster should be the expected genes."""
        adata = _make_clustered_adata()
        rank_genes_groups(adata, "cluster", n_genes=20)
        result = adata.uns["rank_genes_groups"]

        # Cluster 0 markers should be GENE0-GENE9
        top_0 = set(result["names"]["0"][:10])
        expected_0 = {f"GENE{i}" for i in range(10)}
        assert len(top_0 & expected_0) >= 7  # at least 7/10 correct

        # Cluster 1 markers should be GENE10-GENE19
        top_1 = set(result["names"]["1"][:10])
        expected_1 = {f"GENE{i}" for i in range(10, 20)}
        assert len(top_1 & expected_1) >= 7

    def test_not_inplace(self):
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", inplace=False)
        assert isinstance(result, dict)
        assert "names" in result
        assert "scores" in result
        assert "pvals" in result
        assert "pvals_adj" in result
        assert "logfoldchanges" in result
        assert "rank_genes_groups" not in adata.uns

    def test_specific_groups(self):
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", groups=["0", "2"], inplace=False)
        assert "0" in result["names"]
        assert "2" in result["names"]
        assert "1" not in result["names"]

    def test_n_genes(self):
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", n_genes=5, inplace=False)
        assert len(result["names"]["0"]) == 5

    def test_scores_descending(self):
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", inplace=False)
        for group in result["scores"]:
            scores = result["scores"][group]
            assert scores == sorted(scores, reverse=True)

    def test_pvals_bounded(self):
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", inplace=False)
        for group in result["pvals"]:
            pvals = result["pvals"][group]
            assert all(0 <= p <= 1 for p in pvals)

    def test_missing_groupby_raises(self):
        adata = _make_clustered_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            rank_genes_groups(adata, "nonexistent")

    def test_bad_method_raises(self):
        adata = _make_clustered_adata()
        with pytest.raises(ValueError, match="method"):
            rank_genes_groups(adata, "cluster", method="ttest")

    def test_type_error(self):
        with pytest.raises(TypeError, match="rank_genes_groups"):
            rank_genes_groups("not_adata", "cluster")

    def test_dense_input(self):
        adata = _make_clustered_adata()
        adata.X = adata.X.toarray()
        result = rank_genes_groups(adata, "cluster", n_genes=10, inplace=False)
        assert len(result["names"]["0"]) == 10

    def test_public_api(self):
        assert hasattr(singlet, "rank_genes_groups")
        assert callable(singlet.rank_genes_groups)

    def test_pvals_adj_present(self):
        """Adjusted p-values should be in the result dict."""
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", inplace=False)
        assert "pvals_adj" in result
        for group in result["pvals_adj"]:
            pvals_adj = result["pvals_adj"][group]
            assert all(0 <= p <= 1 for p in pvals_adj)

    def test_pvals_adj_geq_pvals(self):
        """Adjusted p-values should be >= raw p-values (BH only inflates)."""
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", inplace=False)
        for group in result["pvals"]:
            for raw, adj in zip(result["pvals"][group], result["pvals_adj"][group]):
                assert adj >= raw - 1e-15  # numerical tolerance

    def test_pvals_adj_markers_significant(self):
        """Known marker genes should have small adjusted p-values."""
        adata = _make_clustered_adata()
        result = rank_genes_groups(adata, "cluster", inplace=False)
        # Top markers for cluster 0 should be significant after correction
        top_pvals = result["pvals_adj"]["0"][:5]
        assert all(p < 0.05 for p in top_pvals)


class TestRankGenesGroupsDf:
    def test_basic(self):
        """Should return a DataFrame with expected columns."""
        adata = _make_clustered_adata()
        rank_genes_groups(adata, "cluster")
        df = rank_genes_groups_df(adata)
        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        assert "names" in df.columns
        assert "scores" in df.columns
        assert "pvals" in df.columns
        assert "pvals_adj" in df.columns
        assert "logfoldchanges" in df.columns

    def test_single_group(self):
        """Should filter to specific group."""
        adata = _make_clustered_adata()
        rank_genes_groups(adata, "cluster", n_genes=10)
        df = rank_genes_groups_df(adata, group="0")
        assert len(df) == 10
        assert all(df["group"] == "0")

    def test_all_groups(self):
        """group=None should return all groups."""
        adata = _make_clustered_adata()
        rank_genes_groups(adata, "cluster", n_genes=10)
        df = rank_genes_groups_df(adata)
        groups = df["group"].unique()
        assert len(groups) == 3  # 3 clusters

    def test_missing_de_raises(self):
        """Should raise KeyError when DE not run."""
        adata = _make_clustered_adata()
        with pytest.raises(KeyError, match="rank_genes_groups"):
            rank_genes_groups_df(adata)

    def test_invalid_group_raises(self):
        """Should raise KeyError for nonexistent group."""
        adata = _make_clustered_adata()
        rank_genes_groups(adata, "cluster")
        with pytest.raises(KeyError, match="nonexistent"):
            rank_genes_groups_df(adata, group="nonexistent")

    def test_public_api(self):
        assert hasattr(singlet, "rank_genes_groups_df")
        assert callable(singlet.rank_genes_groups_df)
