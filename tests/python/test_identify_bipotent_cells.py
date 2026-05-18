# SPDX-License-Identifier: MIT
"""Tests for singlet.identify_bipotent_cells()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._identify_bipotent_cells import identify_bipotent_cells


def _make_bipotent_adata(n_cells=150, n_genes=200, seed=42):
    """Create AnnData with cells expressing lineage markers at different levels."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Base expression
    X = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)

    # Define lineage marker indices
    # Lineage A markers: genes 0-9
    # Lineage B markers: genes 10-19
    # Lineage C markers: genes 20-29

    # First 50 cells: strong lineage A
    X[:50, :10] += 10.0

    # Next 50 cells: strong lineage B
    X[50:100, 10:20] += 10.0

    # Last 50 cells: bipotent (co-express A and B)
    X[100:150, :10] += 7.0
    X[100:150, 10:20] += 7.0

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    # Name the lineage genes meaningfully
    for i in range(10):
        gene_names[i] = f"LINEAGE_A_{i}"
    for i in range(10, 20):
        gene_names[i] = f"LINEAGE_B_{i}"
    for i in range(20, 30):
        gene_names[i] = f"LINEAGE_C_{i}"

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = gene_names

    return adata


def _lineage_genes():
    """Return lineage gene dictionary."""
    return {
        "lineage_A": [f"LINEAGE_A_{i}" for i in range(10)],
        "lineage_B": [f"LINEAGE_B_{i}" for i in range(10, 20)],
        "lineage_C": [f"LINEAGE_C_{i}" for i in range(20, 30)],
    }


class TestIdentifyBipotentCells:
    """Test suite for identify_bipotent_cells."""

    def test_basic_entropy(self):
        """Test basic entropy method identifies bipotent cells."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        result = identify_bipotent_cells(adata, lineages, method="entropy")

        assert result is adata
        assert "bipotent_score" in adata.obs.columns
        assert "top_lineage" in adata.obs.columns
        assert "is_bipotent" in adata.obs.columns

        # Bipotent cells should have higher scores
        bipotent_scores = adata.obs["bipotent_score"].values[100:150]
        committed_scores = adata.obs["bipotent_score"].values[:100]
        assert np.mean(bipotent_scores) > np.mean(committed_scores)

    def test_basic_score_ratio(self):
        """Test score_ratio method identifies bipotent cells."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        result = identify_bipotent_cells(adata, lineages, method="score_ratio")

        assert result is adata
        assert "bipotent_score" in adata.obs.columns

        # Bipotent cells should have ratio closer to 1
        bipotent_scores = adata.obs["bipotent_score"].values[100:150]
        committed_scores = adata.obs["bipotent_score"].values[:100]
        assert np.mean(bipotent_scores) > np.mean(committed_scores)

    def test_threshold(self):
        """Test threshold parameter controls classification."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        # Low threshold = more bipotent calls
        identify_bipotent_cells(adata, lineages, threshold=0.1)
        n_low = adata.obs["is_bipotent"].sum()

        # High threshold = fewer bipotent calls
        identify_bipotent_cells(adata, lineages, threshold=0.9)
        n_high = adata.obs["is_bipotent"].sum()

        assert n_low >= n_high

    def test_top_lineage_assignment(self):
        """Test that top_lineage is correctly assigned."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        identify_bipotent_cells(adata, lineages)

        # Cells 0-49 should be assigned to lineage_A
        top_a = adata.obs["top_lineage"].values[:50]
        assert (top_a == "lineage_A").sum() > 40  # allow some noise

        # Cells 50-99 should be assigned to lineage_B
        top_b = adata.obs["top_lineage"].values[50:100]
        assert (top_b == "lineage_B").sum() > 40

    def test_score_range(self):
        """Test that scores are in valid range [0, 1]."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        identify_bipotent_cells(adata, lineages, method="entropy")
        scores = adata.obs["bipotent_score"].values
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

        identify_bipotent_cells(adata, lineages, method="score_ratio")
        scores = adata.obs["bipotent_score"].values
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_dense_input(self):
        """Test with dense matrix input."""
        import anndata as ad

        rng = np.random.default_rng(123)
        X = rng.poisson(2, size=(80, 50)).astype(np.float32)
        X[:40, :5] += 8.0  # lineage A
        X[40:, 5:10] += 8.0  # lineage B

        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{i}" for i in range(50)]

        lineages = {"A": [f"G{i}" for i in range(5)], "B": [f"G{i}" for i in range(5, 10)]}

        result = identify_bipotent_cells(adata, lineages)
        assert "bipotent_score" in result.obs.columns

    def test_missing_genes_handled(self):
        """Test that missing genes in lineage dict are gracefully handled."""
        adata = _make_bipotent_adata()
        lineages = {
            "lineage_A": ["LINEAGE_A_0", "NONEXISTENT_GENE"],
            "lineage_B": ["LINEAGE_B_10", "ANOTHER_MISSING"],
        }

        # Should not raise
        result = identify_bipotent_cells(adata, lineages)
        assert "bipotent_score" in result.obs.columns

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        with pytest.raises(ValueError, match="method must be"):
            identify_bipotent_cells(adata, lineages, method="invalid")

    def test_empty_lineage_genes_raises(self):
        """Test that empty lineage_genes raises ValueError."""
        adata = _make_bipotent_adata()

        with pytest.raises(ValueError, match="non-empty"):
            identify_bipotent_cells(adata, {})

    def test_single_lineage_raises(self):
        """Test that single lineage raises ValueError."""
        adata = _make_bipotent_adata()

        with pytest.raises(ValueError, match="at least 2"):
            identify_bipotent_cells(adata, {"A": ["LINEAGE_A_0"]})

    def test_layer_parameter(self):
        """Test using a specific layer."""
        adata = _make_bipotent_adata()
        adata.layers["raw"] = adata.X.copy()
        lineages = _lineage_genes()

        result = identify_bipotent_cells(adata, lineages, layer="raw")
        assert "bipotent_score" in result.obs.columns

    def test_categorical_dtype(self):
        """Test that top_lineage is categorical."""
        adata = _make_bipotent_adata()
        lineages = _lineage_genes()

        identify_bipotent_cells(adata, lineages)
        assert hasattr(adata.obs["top_lineage"], "cat")

    def test_two_lineages(self):
        """Test with exactly two lineages."""
        adata = _make_bipotent_adata()
        lineages = {
            "lineage_A": [f"LINEAGE_A_{i}" for i in range(10)],
            "lineage_B": [f"LINEAGE_B_{i}" for i in range(10, 20)],
        }

        result = identify_bipotent_cells(adata, lineages)
        assert "bipotent_score" in result.obs.columns
        assert set(adata.obs["top_lineage"].cat.categories) == {"lineage_A", "lineage_B"}
