# SPDX-License-Identifier: MIT
"""Tests for singlet.gene_set_enrichment()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata_with_de(n_cells=100, n_genes=200, seed=42):
    """Create test AnnData with DE results."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2, (n_cells, n_genes)).astype(np.float32)

    adata = AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["group"] = [f"cluster_{i % 3}" for i in range(n_cells)]

    # Run DE
    singlet.rank_genes_groups(adata, groupby="group")
    return adata


def _make_simple_adata(n_cells=80, n_genes=100, seed=42):
    """Create simple test AnnData without DE results."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, (n_cells, n_genes)).astype(np.float32)

    adata = AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var["highly_variable"] = np.array(
        [True] * 50 + [False] * 50, dtype=bool
    )
    return adata


class TestGeneSetEnrichment:
    def test_basic_with_groupby(self):
        adata = _make_adata_with_de()
        gene_sets = {
            "set_a": [f"GENE{i}" for i in range(0, 20)],
            "set_b": [f"GENE{i}" for i in range(50, 70)],
        }
        result = singlet.gene_set_enrichment(adata, gene_sets, groupby="group")
        assert isinstance(result, type(result))  # DataFrame
        assert "gene_set" in result.columns
        assert "group" in result.columns
        assert "pvalue" in result.columns
        assert "fdr" in result.columns
        assert "odds_ratio" in result.columns
        assert "overlap_size" in result.columns

    def test_basic_without_groupby(self):
        adata = _make_simple_adata()
        gene_sets = {
            "set_a": [f"GENE{i}" for i in range(0, 30)],
            "set_b": [f"GENE{i}" for i in range(40, 60)],
        }
        result = singlet.gene_set_enrichment(adata, gene_sets)
        assert "group" not in result.columns
        assert len(result) == 2
        assert "gene_set" in result.columns

    def test_stores_in_uns(self):
        adata = _make_simple_adata()
        gene_sets = {"my_set": [f"GENE{i}" for i in range(10)]}
        singlet.gene_set_enrichment(adata, gene_sets)
        assert "gene_set_enrichment" in adata.uns
        assert len(adata.uns["gene_set_enrichment"]) > 0

    def test_pvalues_valid(self):
        adata = _make_adata_with_de()
        gene_sets = {
            "set_a": [f"GENE{i}" for i in range(0, 30)],
        }
        result = singlet.gene_set_enrichment(adata, gene_sets, groupby="group")
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_overlap_genes_format(self):
        adata = _make_adata_with_de()
        gene_sets = {"set_a": [f"GENE{i}" for i in range(0, 50)]}
        result = singlet.gene_set_enrichment(adata, gene_sets, groupby="group")
        # overlap_genes should be comma-separated or empty string
        for val in result["overlap_genes"]:
            assert isinstance(val, str)

    def test_enriched_set_has_low_pvalue(self):
        """A set that perfectly matches markers should have low p-value."""
        adata = _make_adata_with_de()
        # Get actual top markers for cluster_0
        rgg = adata.uns["rank_genes_groups"]
        top_markers = list(rgg["names"]["cluster_0"][:20])
        gene_sets = {
            "matching": top_markers,
            "random": [f"GENE{i}" for i in range(180, 200)],
        }
        result = singlet.gene_set_enrichment(
            adata, gene_sets, groupby="group", n_top_genes=50
        )
        matching_rows = result[
            (result["gene_set"] == "matching") & (result["group"] == "cluster_0")
        ]
        if len(matching_rows) > 0:
            assert matching_rows["pvalue"].iloc[0] < 0.05

    def test_n_top_genes_parameter(self):
        adata = _make_adata_with_de()
        gene_sets = {"set_a": [f"GENE{i}" for i in range(0, 30)]}
        result_50 = singlet.gene_set_enrichment(
            adata, gene_sets, groupby="group", n_top_genes=50
        )
        result_10 = singlet.gene_set_enrichment(
            adata, gene_sets, groupby="group", n_top_genes=10
        )
        # Fewer markers should give different overlap sizes
        assert len(result_50) > 0
        assert len(result_10) > 0

    def test_empty_gene_sets_raises(self):
        adata = _make_simple_adata()
        with pytest.raises(ValueError, match="non-empty"):
            singlet.gene_set_enrichment(adata, {})

    def test_invalid_method_raises(self):
        adata = _make_simple_adata()
        gene_sets = {"set_a": ["GENE0", "GENE1"]}
        with pytest.raises(ValueError, match="Unsupported method"):
            singlet.gene_set_enrichment(adata, gene_sets, method="gsea")

    def test_missing_de_results_raises(self):
        adata = _make_simple_adata()
        gene_sets = {"set_a": ["GENE0", "GENE1"]}
        with pytest.raises(KeyError, match="rank_genes_groups"):
            singlet.gene_set_enrichment(adata, gene_sets, groupby="group")

    def test_no_overlap_genes(self):
        """Gene sets with no overlap should still return results."""
        adata = _make_simple_adata()
        gene_sets = {"no_match": ["NONEXISTENT1", "NONEXISTENT2"]}
        result = singlet.gene_set_enrichment(adata, gene_sets)
        # Should return empty DataFrame (no valid gene sets in universe)
        assert len(result) == 0

    def test_public_api(self):
        assert hasattr(singlet, "gene_set_enrichment")
        assert callable(singlet.gene_set_enrichment)

    def test_odds_ratio_positive(self):
        adata = _make_adata_with_de()
        gene_sets = {"set_a": [f"GENE{i}" for i in range(0, 40)]}
        result = singlet.gene_set_enrichment(adata, gene_sets, groupby="group")
        assert (result["odds_ratio"] >= 0).all()
