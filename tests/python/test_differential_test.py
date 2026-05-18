# SPDX-License-Identifier: MIT
"""Tests for singlet.differential_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import singlet


@pytest.fixture()
def adata_groups():
    """Create AnnData with 3 distinct groups in PCA space."""
    import anndata as ad

    rng = np.random.default_rng(123)
    n_cells = 120
    n_genes = 80
    n_pcs = 10

    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)

    # Create 3 well-separated clusters
    pca = np.zeros((n_cells, n_pcs))
    # Group A: centered at (5, 0, ...)
    pca[:40, 0] = rng.normal(5, 0.3, 40)
    pca[:40, 1] = rng.normal(0, 0.3, 40)
    # Group B: centered at (-5, 0, ...)
    pca[40:80, 0] = rng.normal(-5, 0.3, 40)
    pca[40:80, 1] = rng.normal(0, 0.3, 40)
    # Group C: centered at (0, 5, ...)
    pca[80:120, 0] = rng.normal(0, 0.3, 40)
    pca[80:120, 1] = rng.normal(5, 0.3, 40)

    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["A"] * 40 + ["B"] * 40 + ["C"] * 40
    adata.obsm["X_pca"] = pca

    return adata


@pytest.fixture()
def adata_same_dist():
    """Create AnnData where groups come from same distribution."""
    import anndata as ad

    rng = np.random.default_rng(999)
    n_cells = 100
    n_genes = 50
    n_pcs = 5

    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    pca = rng.normal(0, 1, size=(n_cells, n_pcs))

    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["X"] * 50 + ["Y"] * 50
    adata.obsm["X_pca"] = pca

    return adata


def test_ks_basic(adata_groups):
    """Test KS method returns expected structure."""
    result = singlet.differential_test(adata_groups, groupby="group", method="ks")

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"group1", "group2", "statistic", "pvalue"}
    # 3 groups → 3 pairwise comparisons
    assert len(result) == 3
    assert "differential_test" in adata_groups.uns


def test_ks_detects_difference(adata_groups):
    """KS test should detect well-separated groups."""
    result = singlet.differential_test(adata_groups, groupby="group", method="ks")
    # All p-values should be very small
    assert np.all(result["pvalue"].values < 0.05)


def test_ks_no_difference(adata_same_dist):
    """KS test should not reject H0 for same distribution."""
    result = singlet.differential_test(
        adata_same_dist, groupby="group", method="ks"
    )
    # p-value should generally be non-significant (> 0.01)
    # Use a lenient threshold since it's random
    assert result["pvalue"].values[0] > 0.001


def test_energy_basic(adata_groups):
    """Test energy method returns expected structure."""
    result = singlet.differential_test(
        adata_groups, groupby="group", method="energy"
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert np.all(result["statistic"].values > 0)


def test_energy_detects_difference(adata_groups):
    """Energy test should detect well-separated groups."""
    result = singlet.differential_test(
        adata_groups, groupby="group", method="energy"
    )
    assert np.all(result["pvalue"].values < 0.05)


def test_mmd_basic(adata_groups):
    """Test MMD method returns expected structure."""
    result = singlet.differential_test(
        adata_groups, groupby="group", method="mmd"
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert np.all(result["statistic"].values >= 0)


def test_mmd_detects_difference(adata_groups):
    """MMD test should detect well-separated groups."""
    result = singlet.differential_test(
        adata_groups, groupby="group", method="mmd"
    )
    assert np.all(result["pvalue"].values < 0.05)


def test_specific_groups(adata_groups):
    """Test selecting specific groups to compare."""
    result = singlet.differential_test(
        adata_groups, groupby="group", groups=["A", "B"], method="ks"
    )
    assert len(result) == 1
    assert result.iloc[0]["group1"] == "A"
    assert result.iloc[0]["group2"] == "B"


def test_custom_use_rep(adata_groups):
    """Test using a custom embedding."""
    adata_groups.obsm["X_custom"] = adata_groups.obsm["X_pca"][:, :3]
    result = singlet.differential_test(
        adata_groups, groupby="group", use_rep="X_custom", method="ks"
    )
    assert len(result) == 3


def test_missing_groupby_raises(adata_groups):
    """Should raise KeyError if groupby column doesn't exist."""
    with pytest.raises(KeyError, match="not found in adata.obs"):
        singlet.differential_test(adata_groups, groupby="nonexistent")


def test_missing_use_rep_raises(adata_groups):
    """Should raise KeyError if use_rep doesn't exist."""
    with pytest.raises(KeyError, match="not found in adata.obsm"):
        singlet.differential_test(
            adata_groups, groupby="group", use_rep="X_fake"
        )


def test_invalid_method_raises(adata_groups):
    """Should raise ValueError for invalid method."""
    with pytest.raises(ValueError, match="Method must be"):
        singlet.differential_test(
            adata_groups, groupby="group", method="invalid"
        )


def test_invalid_group_raises(adata_groups):
    """Should raise ValueError if specified group not found."""
    with pytest.raises(ValueError, match="Group 'Z' not found"):
        singlet.differential_test(
            adata_groups, groupby="group", groups=["A", "Z"]
        )


def test_result_stored_in_uns(adata_groups):
    """Result should be stored in adata.uns."""
    result = singlet.differential_test(adata_groups, groupby="group")
    stored = adata_groups.uns["differential_test"]
    pd.testing.assert_frame_equal(result, stored)
