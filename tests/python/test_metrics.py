# SPDX-License-Identifier: MIT
"""Tests for singlet.silhouette_score() and calinski_harabasz_score()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=100, m=150, seed=42):
    rng = np.random.default_rng(seed)
    # Create well-separated clusters
    X = np.vstack(
        [
            rng.standard_normal((n // 3, m)) + np.array([5] * m),
            rng.standard_normal((n // 3, m)) + np.array([-5] * m),
            rng.standard_normal((n - 2 * (n // 3), m)),
        ]
    ).astype(np.float32)
    adata = AnnData(X=X)
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.leiden(adata)
    return adata


def test_silhouette_basic():
    adata = _make_adata()
    score = singlet.silhouette_score(adata)
    assert isinstance(score, float)
    assert -1 <= score <= 1


def test_silhouette_well_separated():
    adata = _make_adata()
    score = singlet.silhouette_score(adata)
    # Well-separated clusters should have positive silhouette
    assert score > 0


def test_silhouette_no_groupby_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.silhouette_score(adata, groupby="nonexistent")


def test_silhouette_no_rep_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.silhouette_score(adata, use_rep="nonexistent")


def test_calinski_basic():
    adata = _make_adata()
    score = singlet.calinski_harabasz_score(adata)
    assert isinstance(score, float)
    assert score > 0


def test_calinski_well_separated():
    adata = _make_adata()
    score = singlet.calinski_harabasz_score(adata)
    assert score > 10  # Well-separated should have high CH


def test_silhouette_n_pcs():
    adata = _make_adata()
    score = singlet.silhouette_score(adata, n_pcs=5)
    assert -1 <= score <= 1
