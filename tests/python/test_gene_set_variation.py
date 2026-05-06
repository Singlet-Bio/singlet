"""Tests for singlet.gene_set_variation()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData
from scipy import sparse


def _make_adata(n_cells=80, n_genes=200, seed=42):
    """Create test AnnData with expression data."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, (n_cells, n_genes)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


def _make_sparse_adata(n_cells=60, n_genes=150, seed=42):
    """Create test AnnData with sparse expression data."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(1, (n_cells, n_genes)).astype(np.float32)
    adata = AnnData(X=sparse.csr_matrix(X))
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestGeneSetVariation:
    def test_gsva_basic(self):
        adata = _make_adata()
        gene_sets = {
            "pathway_a": [f"GENE{i}" for i in range(0, 30)],
            "pathway_b": [f"GENE{i}" for i in range(50, 80)],
        }
        result = singlet.gene_set_variation(adata, gene_sets, method="gsva")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (80, 2)
        assert "pathway_a" in result.columns
        assert "pathway_b" in result.columns
        assert "gsva_scores" in adata.obsm

    def test_ssgsea_basic(self):
        adata = _make_adata()
        gene_sets = {
            "set_1": [f"GENE{i}" for i in range(10, 40)],
            "set_2": [f"GENE{i}" for i in range(100, 130)],
        }
        result = singlet.gene_set_variation(adata, gene_sets, method="ssgsea")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (80, 2)
        assert "set_1" in result.columns
        assert "set_2" in result.columns

    def test_gsva_stores_in_obsm(self):
        adata = _make_adata()
        gene_sets = {"path1": [f"GENE{i}" for i in range(20)]}
        singlet.gene_set_variation(adata, gene_sets)
        assert "gsva_scores" in adata.obsm
        scores = adata.obsm["gsva_scores"]
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape[0] == adata.n_obs

    def test_gsva_gaussian_kcdf(self):
        adata = _make_adata()
        gene_sets = {"path1": [f"GENE{i}" for i in range(0, 25)]}
        result = singlet.gene_set_variation(
            adata, gene_sets, method="gsva", kcdf="Gaussian"
        )
        assert not result.isna().all().all()

    def test_gsva_poisson_kcdf(self):
        adata = _make_adata()
        gene_sets = {"path1": [f"GENE{i}" for i in range(0, 25)]}
        result = singlet.gene_set_variation(
            adata, gene_sets, method="gsva", kcdf="Poisson"
        )
        assert not result.isna().all().all()

    def test_sparse_input(self):
        adata = _make_sparse_adata()
        gene_sets = {
            "pathway_a": [f"GENE{i}" for i in range(0, 20)],
            "pathway_b": [f"GENE{i}" for i in range(50, 70)],
        }
        result = singlet.gene_set_variation(adata, gene_sets, method="gsva")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (60, 2)

    def test_missing_genes_filtered(self):
        adata = _make_adata(n_genes=100)
        gene_sets = {
            "mixed": [f"GENE{i}" for i in range(90, 120)],  # 10 present, 20 missing
        }
        result = singlet.gene_set_variation(adata, gene_sets)
        assert "mixed" in result.columns
        # Should still produce values (10 genes present)
        assert not result["mixed"].isna().all()

    def test_too_few_genes_returns_nan(self):
        adata = _make_adata(n_genes=100)
        gene_sets = {
            "tiny": ["NONEXISTENT1", "NONEXISTENT2"],  # 0 present
        }
        result = singlet.gene_set_variation(adata, gene_sets)
        assert "tiny" in result.columns
        assert result["tiny"].isna().all()

    def test_invalid_method_raises(self):
        adata = _make_adata()
        gene_sets = {"path1": [f"GENE{i}" for i in range(10)]}
        with pytest.raises(ValueError):
            singlet.gene_set_variation(adata, gene_sets, method="invalid")

    def test_multiple_gene_sets(self):
        adata = _make_adata(n_cells=50, n_genes=300)
        gene_sets = {
            f"pathway_{k}": [f"GENE{i}" for i in range(k * 20, k * 20 + 20)]
            for k in range(10)
        }
        result = singlet.gene_set_variation(adata, gene_sets, method="gsva")
        assert result.shape == (50, 10)

    def test_scores_vary_across_cells(self):
        """Scores should not be constant across cells."""
        rng = np.random.default_rng(42)
        n_cells = 100
        n_genes = 200
        # Create structured data where some cells overexpress pathway genes
        X = rng.poisson(2, (n_cells, n_genes)).astype(np.float32)
        # Boost pathway genes in first 50 cells
        X[:50, :30] += 5
        adata = AnnData(X=X)
        adata.var_names = [f"GENE{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

        gene_sets = {"boosted": [f"GENE{i}" for i in range(30)]}
        result = singlet.gene_set_variation(adata, gene_sets, method="gsva")
        # Cells with boosted expression should have higher scores
        assert result["boosted"].std() > 0
        mean_boosted = result["boosted"].iloc[:50].mean()
        mean_normal = result["boosted"].iloc[50:].mean()
        assert mean_boosted > mean_normal

    def test_ssgsea_scores_vary(self):
        """ssGSEA should also produce variable scores."""
        adata = _make_adata()
        gene_sets = {"path1": [f"GENE{i}" for i in range(0, 30)]}
        result = singlet.gene_set_variation(adata, gene_sets, method="ssgsea")
        assert result["path1"].std() > 0
