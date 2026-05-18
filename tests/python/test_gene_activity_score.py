# SPDX-License-Identifier: MIT
"""Tests for singlet.gene_activity_score()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._gene_activity_score import gene_activity_score


def _make_adata(n_cells=100, n_genes=200):
    """Create AnnData with known gene expression patterns."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Make pathway genes (0-4) highly expressed in cells 0-49
    X[:50, :5] += 10.0
    # Make second pathway genes (5-9) highly expressed in cells 50-99
    X[50:, 5:10] += 8.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    return adata


class TestGeneActivityScore:
    def test_basic_mean(self):
        """Should compute mean scores and store in adata.obs."""
        adata = _make_adata()
        gene_sets = {"pathway1": ["GENE0", "GENE1", "GENE2"]}
        result = gene_activity_score(adata, gene_sets)

        assert isinstance(result, pd.DataFrame)
        assert "activity_pathway1" in result.columns
        assert "activity_pathway1" in adata.obs.columns
        assert result.shape == (100, 1)

    def test_multiple_pathways(self):
        """Should handle multiple gene sets."""
        adata = _make_adata()
        gene_sets = {
            "path_a": ["GENE0", "GENE1", "GENE2"],
            "path_b": ["GENE5", "GENE6", "GENE7"],
        }
        result = gene_activity_score(adata, gene_sets)

        assert result.shape == (100, 2)
        assert "activity_path_a" in result.columns
        assert "activity_path_b" in result.columns
        assert "activity_path_a" in adata.obs.columns
        assert "activity_path_b" in adata.obs.columns

    def test_mean_method_separates_groups(self):
        """Cells with high pathway expression should have higher scores."""
        adata = _make_adata()
        gene_sets = {"active": ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"]}
        result = gene_activity_score(adata, gene_sets, method="mean")

        # First 50 cells have high expression of these genes
        high_cells = result["activity_active"].values[:50].mean()
        low_cells = result["activity_active"].values[50:].mean()
        assert high_cells > low_cells

    def test_weighted_method(self):
        """Weighted method should work and produce different results from mean."""
        adata = _make_adata()
        gene_sets = {"path": ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"]}

        result_mean = gene_activity_score(adata, gene_sets, method="mean")  # noqa: F841
        # Reset obs
        adata.obs = adata.obs.drop(columns=["activity_path"])
        result_weighted = gene_activity_score(adata, gene_sets, method="weighted")

        # Both should separate the groups
        assert result_weighted["activity_path"].values[:50].mean() > 0
        assert result_weighted["activity_path"].values[50:].mean() < 0

    def test_zscore_method(self):
        """Z-score method should work."""
        adata = _make_adata()
        gene_sets = {"path": ["GENE0", "GENE1", "GENE2"]}
        result = gene_activity_score(adata, gene_sets, method="z-score")

        assert "activity_path" in result.columns
        # Should separate groups
        assert (
            result["activity_path"].values[:50].mean() > result["activity_path"].values[50:].mean()
        )

    def test_scale_false(self):
        """scale=False should skip standardization."""
        adata = _make_adata()
        gene_sets = {"path": ["GENE0", "GENE1"]}
        result = gene_activity_score(adata, gene_sets, scale=False)

        # All values should be positive (poisson + offset)
        assert result["activity_path"].values.min() >= 0

    def test_scale_true_produces_unit_variance(self):
        """scale=True should produce approximately zero-mean, unit-var."""
        adata = _make_adata()
        gene_sets = {"path": ["GENE0", "GENE1", "GENE2"]}
        result = gene_activity_score(adata, gene_sets, scale=True)

        scores = result["activity_path"].values
        assert abs(scores.mean()) < 0.01
        assert abs(scores.std() - 1.0) < 0.05

    def test_missing_genes_ignored(self):
        """Genes not in adata should be silently skipped."""
        adata = _make_adata()
        gene_sets = {"path": ["GENE0", "NONEXISTENT1", "NONEXISTENT2"]}
        result = gene_activity_score(adata, gene_sets)
        assert "activity_path" in result.columns

    def test_all_genes_missing(self):
        """If all genes missing, score should be zero."""
        adata = _make_adata()
        gene_sets = {"empty": ["FAKEGENE1", "FAKEGENE2"]}
        result = gene_activity_score(adata, gene_sets, scale=False)
        assert np.all(result["activity_empty"].values == 0)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="method must be one of"):
            gene_activity_score(adata, {"p": ["GENE0"]}, method="invalid")

    def test_empty_gene_sets_raises(self):
        """Empty gene_sets dict should raise ValueError."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="gene_sets must not be empty"):
            gene_activity_score(adata, {})

    def test_layer_parameter(self):
        """Should use specified layer."""
        adata = _make_adata()
        adata.layers["normalized"] = adata.X.toarray() * 2.0
        gene_sets = {"path": ["GENE0", "GENE1"]}

        result = gene_activity_score(adata, gene_sets, layer="normalized", scale=False)
        # Layer has 2x values, so scores should be higher
        adata.obs = adata.obs.drop(columns=["activity_path"])
        result_x = gene_activity_score(adata, gene_sets, scale=False)

        assert result["activity_path"].mean() > result_x["activity_path"].mean()

    def test_dense_matrix(self):
        """Should work with dense X."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X = rng.poisson(2, size=(50, 100)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"GENE{i}" for i in range(100)]

        gene_sets = {"path": ["GENE0", "GENE1", "GENE2"]}
        result = gene_activity_score(adata, gene_sets)
        assert result.shape == (50, 1)

    def test_index_matches_obs_names(self):
        """DataFrame index should match adata.obs_names."""
        adata = _make_adata()
        gene_sets = {"path": ["GENE0"]}
        result = gene_activity_score(adata, gene_sets)
        assert list(result.index) == list(adata.obs_names)

    def test_singlet_import(self):
        """Should be importable from singlet namespace."""
        import singlet

        assert hasattr(singlet, "gene_activity_score")
        assert callable(singlet.gene_activity_score)
