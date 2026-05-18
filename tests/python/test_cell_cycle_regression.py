# SPDX-License-Identifier: MIT
"""Tests for singlet.cell_cycle_regression()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata_with_cc(n_obs=100, seed=42):
    """Create AnnData with some cell cycle genes and cycling signal."""
    rng = np.random.default_rng(seed)

    # Include S and G2M genes from Tirosh list
    s_genes = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2"]
    g2m_genes = ["CDK1", "TOP2A", "MKI67", "NUSAP1", "UBE2C"]
    other_genes = [f"gene_{i}" for i in range(90)]
    all_genes = s_genes + g2m_genes + other_genes
    n_vars = len(all_genes)

    X = rng.poisson(3, size=(n_obs, n_vars)).astype(np.float32)

    # Simulate cycling cells: first 30 in S, next 30 in G2M
    X[:30, :5] += 8.0  # S genes elevated
    X[30:60, 5:10] += 8.0  # G2M genes elevated

    adata = AnnData(X=np.log1p(X))
    adata.var_names = all_genes
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return adata


def test_cell_cycle_regression_basic():
    """Basic invocation adds expected columns."""
    adata = _make_adata_with_cc()
    result = singlet.cell_cycle_regression(adata)

    assert result is adata  # Returns same object
    assert "cell_cycle_phase" in adata.obs.columns
    assert "S_score" in adata.obs.columns
    assert "G2M_score" in adata.obs.columns


def test_cell_cycle_regression_phases():
    """Phase labels are valid categories."""
    adata = _make_adata_with_cc()
    singlet.cell_cycle_regression(adata)

    phases = adata.obs["cell_cycle_phase"].unique()
    for phase in phases:
        assert phase in ["G1", "S", "G2M"]


def test_cell_cycle_regression_scores_finite():
    """Scores are finite values."""
    adata = _make_adata_with_cc()
    singlet.cell_cycle_regression(adata)

    assert np.all(np.isfinite(adata.obs["S_score"].values))
    assert np.all(np.isfinite(adata.obs["G2M_score"].values))


def test_cell_cycle_regression_detects_cycling():
    """S cells should have high S_score, G2M cells high G2M_score."""
    adata = _make_adata_with_cc()
    singlet.cell_cycle_regression(adata, regress=False)

    # First 30 cells should tend toward S phase
    s_scores = adata.obs["S_score"].values
    mean_s_score_cycling = s_scores[:30].mean()
    mean_s_score_rest = s_scores[60:].mean()
    assert mean_s_score_cycling > mean_s_score_rest

    # Cells 30-60 should tend toward G2M
    g2m_scores = adata.obs["G2M_score"].values
    mean_g2m_cycling = g2m_scores[30:60].mean()
    mean_g2m_rest = g2m_scores[60:].mean()
    assert mean_g2m_cycling > mean_g2m_rest


def test_cell_cycle_regression_regress_modifies_x():
    """With regress=True, X is modified."""
    adata = _make_adata_with_cc()
    X_before = adata.X.copy()
    singlet.cell_cycle_regression(adata, regress=True)

    # X should be different after regression
    assert not np.allclose(adata.X, X_before)


def test_cell_cycle_regression_no_regress():
    """With regress=False, X is not modified."""
    adata = _make_adata_with_cc()
    X_before = adata.X.copy()
    singlet.cell_cycle_regression(adata, regress=False)

    np.testing.assert_array_equal(adata.X, X_before)


def test_cell_cycle_regression_reduces_cc_variance():
    """After regression, CC gene variance should be reduced."""
    adata = _make_adata_with_cc()
    # Variance of S genes before regression
    s_idx = [list(adata.var_names).index(g) for g in ["MCM5", "PCNA", "TYMS"]]
    var_before = np.var(adata.X[:, s_idx], axis=0).mean()

    singlet.cell_cycle_regression(adata, regress=True)
    var_after = np.var(adata.X[:, s_idx], axis=0).mean()

    # Variance should decrease after removing CC signal
    assert var_after < var_before


def test_cell_cycle_regression_custom_genes():
    """Custom gene lists work."""
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 30)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(30)]

    singlet.cell_cycle_regression(
        adata,
        s_genes=["g0", "g1", "g2", "g3", "g4"],
        g2m_genes=["g10", "g11", "g12", "g13", "g14"],
    )
    assert "cell_cycle_phase" in adata.obs.columns


def test_cell_cycle_regression_no_cc_genes():
    """When no CC genes are present, scores are zero and X unchanged."""
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((40, 20)).astype(np.float32))
    adata.var_names = [f"random_{i}" for i in range(20)]
    X_before = adata.X.copy()

    singlet.cell_cycle_regression(adata, regress=True)

    # All phases should be G1
    assert (adata.obs["cell_cycle_phase"] == "G1").all()
    # Scores should be 0
    assert np.allclose(adata.obs["S_score"].values, 0.0)
    assert np.allclose(adata.obs["G2M_score"].values, 0.0)
    # X should be unchanged (regression of zeros does nothing)
    np.testing.assert_allclose(adata.X, X_before, atol=1e-5)


def test_cell_cycle_regression_x_finite():
    """After regression, X should contain only finite values."""
    adata = _make_adata_with_cc()
    singlet.cell_cycle_regression(adata, regress=True)
    assert np.all(np.isfinite(adata.X))


def test_cell_cycle_regression_sparse_input():
    """Works with sparse input matrix."""
    from scipy.sparse import csr_matrix

    adata = _make_adata_with_cc()
    adata.X = csr_matrix(adata.X)
    singlet.cell_cycle_regression(adata, regress=True)

    assert "cell_cycle_phase" in adata.obs.columns
    # After regression X should be dense (since we modified it)
    assert np.all(np.isfinite(np.asarray(adata.X)))


def test_cell_cycle_regression_return_type():
    """Returns adata."""
    adata = _make_adata_with_cc()
    result = singlet.cell_cycle_regression(adata)
    assert result is adata
