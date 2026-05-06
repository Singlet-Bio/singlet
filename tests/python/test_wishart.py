"""Tests for singlet.wishart_test()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=90, m=100, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.leiden(adata)
    return adata


def test_wishart_basic():
    adata = _make_adata()
    singlet.wishart_test(adata)
    assert "cluster_confidence" in adata.obs.columns
    assert len(adata.obs["cluster_confidence"]) == adata.n_obs


def test_wishart_values_nonneg():
    adata = _make_adata()
    singlet.wishart_test(adata)
    assert (adata.obs["cluster_confidence"] >= 0).all()


def test_wishart_copy():
    adata = _make_adata()
    result = singlet.wishart_test(adata, copy=True)
    assert result is not None
    assert "cluster_confidence" not in adata.obs.columns
    assert "cluster_confidence" in result.obs.columns


def test_wishart_key_added():
    adata = _make_adata()
    singlet.wishart_test(adata, key_added="my_conf")
    assert "my_conf" in adata.obs.columns


def test_wishart_no_groupby_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.wishart_test(adata, groupby="nonexistent")


def test_wishart_finite():
    adata = _make_adata()
    singlet.wishart_test(adata)
    assert np.all(np.isfinite(adata.obs["cluster_confidence"].values))


def test_wishart_well_assigned_low():
    # Create clearly separated clusters
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.standard_normal((30, 50)) + 10,
            rng.standard_normal((30, 50)) - 10,
        ]
    ).astype(np.float32)
    adata = AnnData(X=X)
    adata.obs["cluster"] = ["A"] * 30 + ["B"] * 30
    singlet.pca(adata)
    singlet.wishart_test(adata, groupby="cluster")
    # Well-separated should have low confidence scores (< 1)
    assert adata.obs["cluster_confidence"].mean() < 1.0
