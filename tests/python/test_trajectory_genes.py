# SPDX-License-Identifier: MIT
"""Tests for singlet.trajectory_genes()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._trajectory_genes import trajectory_genes


def _make_trajectory_adata(n_cells=150, n_genes=100):
    """Create test AnnData with pseudotime."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # Create pseudotime that goes from 0 to 1
    pseudotime = np.sort(rng.uniform(0, 1, n_cells))

    # Create expression matrix where some genes vary with pseudotime
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g_idx in range(n_genes):
        if g_idx < 20:
            # Genes that increase with pseudotime
            X[:, g_idx] = pseudotime * 10 + rng.normal(0, 0.5, n_cells)
        elif g_idx < 40:
            # Genes that peak in the middle
            X[:, g_idx] = (
                np.sin(pseudotime * np.pi) * 8 + rng.normal(0, 0.5, n_cells)
            )
        else:
            # Random genes (no trajectory signal)
            X[:, g_idx] = rng.poisson(3, n_cells).astype(np.float32)

    X = np.maximum(X, 0)  # no negative counts

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]
    adata.obs["dpt_pseudotime"] = pseudotime

    return adata


class TestTrajectoryGenes:
    def test_basic(self):
        """Basic trajectory gene detection."""
        adata = _make_trajectory_adata()
        result = trajectory_genes(adata)

        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "pvalue" in result.columns
        assert "fdr" in result.columns
        assert "trend_score" in result.columns
        assert "max_pseudotime" in result.columns

    def test_stores_in_uns(self):
        """Result stored in adata.uns."""
        adata = _make_trajectory_adata()
        trajectory_genes(adata)

        assert "trajectory_genes" in adata.uns
        assert isinstance(adata.uns["trajectory_genes"], pd.DataFrame)

    def test_stores_smoothed_layer(self):
        """Smoothed expression stored in adata.layers."""
        adata = _make_trajectory_adata()
        trajectory_genes(adata)

        assert "trajectory_smoothed" in adata.layers
        smoothed = adata.layers["trajectory_smoothed"]
        assert smoothed.shape == adata.X.shape

    def test_returns_correct_number(self):
        """Returns n_top_genes rows."""
        adata = _make_trajectory_adata()
        result = trajectory_genes(adata, n_top_genes=20)
        assert len(result) == 20

    def test_n_top_larger_than_genes(self):
        """If n_top > n_genes, return all genes."""
        adata = _make_trajectory_adata(n_genes=50)
        result = trajectory_genes(adata, n_top_genes=200)
        assert len(result) == 50

    def test_trajectory_genes_ranked_first(self):
        """Genes with trajectory signal should rank higher."""
        adata = _make_trajectory_adata()
        result = trajectory_genes(adata, n_top_genes=50)

        # The top genes should include more from the first 40 (signal genes)
        top_genes = set(result["gene"].values)
        signal_genes = {f"GENE{idx}" for idx in range(40)}
        overlap = top_genes & signal_genes
        # At least half the top 50 should be signal genes
        assert len(overlap) >= 20

    def test_spline_method(self):
        """Spline method works."""
        adata = _make_trajectory_adata()
        result = trajectory_genes(adata, method="spline")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_custom_pseudotime_key(self):
        """Custom pseudotime key works."""
        adata = _make_trajectory_adata()
        adata.obs["my_pt"] = adata.obs["dpt_pseudotime"]
        result = trajectory_genes(adata, pseudotime_key="my_pt")
        assert len(result) > 0

    def test_missing_pseudotime_key_raises(self):
        """Missing pseudotime key raises KeyError."""
        adata = _make_trajectory_adata()
        with pytest.raises(KeyError, match="not_a_key"):
            trajectory_genes(adata, pseudotime_key="not_a_key")

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        adata = _make_trajectory_adata()
        with pytest.raises(ValueError, match="method must be"):
            trajectory_genes(adata, method="invalid")

    def test_too_few_bins_raises(self):
        """n_bins < 3 raises ValueError."""
        adata = _make_trajectory_adata()
        with pytest.raises(ValueError, match="n_bins must be"):
            trajectory_genes(adata, n_bins=2)

    def test_nan_pseudotime_cells_handled(self):
        """Cells with NaN pseudotime are excluded from analysis."""
        adata = _make_trajectory_adata()
        # Set some cells to NaN pseudotime
        # .loc on the frame, not chained .iloc on the column: chained assignment
        # is a silent no-op under pandas >= 3 (Copy-on-Write).
        adata.obs.loc[adata.obs.index[:10], "dpt_pseudotime"] = np.nan
        result = trajectory_genes(adata)
        assert len(result) > 0
        # Smoothed layer should have 0 for NaN cells
        smoothed = adata.layers["trajectory_smoothed"]
        assert (smoothed[:10] == 0).all()

    def test_pvalues_in_range(self):
        """P-values and FDR should be in [0, 1]."""
        adata = _make_trajectory_adata()
        result = trajectory_genes(adata)
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_dense_matrix(self):
        """Works with dense X matrix."""
        adata = _make_trajectory_adata()
        adata.X = np.asarray(adata.X.todense())
        result = trajectory_genes(adata)
        assert len(result) > 0

    def test_few_bins(self):
        """Works with small number of bins."""
        adata = _make_trajectory_adata()
        result = trajectory_genes(adata, n_bins=5)
        assert len(result) > 0

    def test_singlet_api(self):
        """Function accessible via singlet namespace."""
        adata = _make_trajectory_adata()
        result = singlet.trajectory_genes(adata)
        assert isinstance(result, pd.DataFrame)
