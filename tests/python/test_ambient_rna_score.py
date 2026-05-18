# SPDX-License-Identifier: MIT
"""Tests for singlet.ambient_rna_score()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n=100, m=200, seed=42, sparse=True):
    """Create test AnnData with count-like data."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2, size=(n, m)).astype(np.float32)
    # Add some zeros to simulate dropout
    mask = rng.random((n, m)) < 0.3
    X[mask] = 0
    if sparse:
        X = sp.csr_matrix(X)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    return adata


def test_ambient_score_basic_cosine():
    """Basic ambient RNA scoring with cosine method."""
    adata = _make_adata()
    result = singlet.ambient_rna_score(adata, method="cosine")
    assert result is adata
    assert "ambient_rna_score" in adata.obs.columns
    assert len(adata.obs["ambient_rna_score"]) == 100


def test_ambient_score_basic_correlation():
    """Basic ambient RNA scoring with correlation method."""
    adata = _make_adata()
    result = singlet.ambient_rna_score(adata, method="correlation")
    assert result is adata
    assert "ambient_rna_score" in adata.obs.columns


def test_ambient_score_cosine_bounded():
    """Cosine scores should be in [0, 1]."""
    adata = _make_adata()
    singlet.ambient_rna_score(adata, method="cosine")
    scores = adata.obs["ambient_rna_score"].values
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_ambient_score_with_profile():
    """Test with a provided empty droplet profile."""
    adata = _make_adata(m=50)
    rng = np.random.default_rng(99)
    profile = rng.exponential(1, size=50)
    singlet.ambient_rna_score(adata, empty_droplet_profile=profile)
    assert "ambient_rna_score" in adata.obs.columns
    scores = adata.obs["ambient_rna_score"].values
    assert np.all(np.isfinite(scores))


def test_ambient_score_profile_mismatch():
    """Should raise ValueError when profile length doesn't match genes."""
    adata = _make_adata(m=50)
    profile = np.ones(30)  # wrong length
    with pytest.raises(ValueError, match="does not match"):
        singlet.ambient_rna_score(adata, empty_droplet_profile=profile)


def test_ambient_score_invalid_method():
    """Should raise ValueError for invalid method."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="method must be"):
        singlet.ambient_rna_score(adata, method="invalid")


def test_ambient_score_dense_input():
    """Works with dense matrix input."""
    adata = _make_adata(sparse=False)
    singlet.ambient_rna_score(adata)
    assert "ambient_rna_score" in adata.obs.columns
    scores = adata.obs["ambient_rna_score"].values
    assert np.all(np.isfinite(scores))


def test_ambient_score_high_contamination():
    """Cells that look like ambient should have high scores."""
    rng = np.random.default_rng(55)
    n, m = 80, 100

    # Create ambient profile (concentrated in first 20 genes)
    ambient_profile = np.zeros(m)
    ambient_profile[:20] = rng.exponential(5, size=20)

    # Half cells are "clean" with diverse expression
    X = np.zeros((n, m), dtype=np.float32)
    X[:40, :] = rng.poisson(3, size=(40, m)).astype(np.float32)

    # Half cells are "contaminated" (look like ambient)
    for cell_idx in range(40, n):
        X[cell_idx, :] = ambient_profile * rng.uniform(0.5, 2.0) + rng.poisson(
            0.1, size=m
        )

    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(m)]
    adata.obs_names = [f"c{i}" for i in range(n)]

    singlet.ambient_rna_score(adata, empty_droplet_profile=ambient_profile)
    scores = adata.obs["ambient_rna_score"].values

    # Contaminated cells should have higher scores on average
    clean_mean = scores[:40].mean()
    contaminated_mean = scores[40:].mean()
    assert contaminated_mean > clean_mean


def test_ambient_score_n_top_ambient():
    """Varying n_top_ambient produces valid results."""
    adata = _make_adata(m=50)
    singlet.ambient_rna_score(adata, n_top_ambient=10)
    scores_10 = adata.obs["ambient_rna_score"].values.copy()

    singlet.ambient_rna_score(adata, n_top_ambient=40)
    scores_40 = adata.obs["ambient_rna_score"].values.copy()

    # Both should be valid
    assert np.all(np.isfinite(scores_10))
    assert np.all(np.isfinite(scores_40))
    # They should differ (different gene sets)
    assert not np.allclose(scores_10, scores_40)


def test_ambient_score_zero_expression():
    """Handles cells with zero expression gracefully."""
    rng = np.random.default_rng(77)
    X = rng.poisson(2, size=(50, 30)).astype(np.float32)
    X[0, :] = 0  # First cell has no expression
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.obs_names = [f"c{i}" for i in range(50)]

    singlet.ambient_rna_score(adata, method="cosine")
    scores = adata.obs["ambient_rna_score"].values
    assert np.all(np.isfinite(scores))


def test_ambient_score_correlation_range():
    """Correlation scores should be finite."""
    adata = _make_adata()
    singlet.ambient_rna_score(adata, method="correlation")
    scores = adata.obs["ambient_rna_score"].values
    assert np.all(np.isfinite(scores))
