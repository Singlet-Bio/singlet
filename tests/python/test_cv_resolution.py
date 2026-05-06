"""Tests for singlet.cross_validate_resolution()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=100, m=150, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    return adata


def test_cv_resolution_basic():
    adata = _make_adata()
    result = singlet.cross_validate_resolution(adata, resolutions=[0.5, 1.0, 1.5])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "resolution" in result.columns
    assert "n_clusters" in result.columns
    assert "score" in result.columns


def test_cv_resolution_scores_reasonable():
    adata = _make_adata()
    result = singlet.cross_validate_resolution(adata, resolutions=[0.5, 1.0])
    # Silhouette scores should be in [-1, 1]
    assert (result["score"] >= -1).all()
    assert (result["score"] <= 1).all()


def test_cv_resolution_calinski():
    adata = _make_adata()
    result = singlet.cross_validate_resolution(
        adata,
        resolutions=[0.5, 1.0],
        metric="calinski_harabasz",
    )
    assert (result["score"] >= 0).all()


def test_cv_resolution_no_neighbors_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    singlet.pca(adata)
    with pytest.raises(KeyError):
        singlet.cross_validate_resolution(adata, resolutions=[1.0])


def test_cv_resolution_n_clusters_increases():
    adata = _make_adata()
    result = singlet.cross_validate_resolution(adata, resolutions=[0.1, 2.0])
    # Higher resolution should give more clusters
    assert result.iloc[1]["n_clusters"] >= result.iloc[0]["n_clusters"]


def test_cv_resolution_no_temp_columns():
    adata = _make_adata()
    singlet.cross_validate_resolution(adata, resolutions=[0.5, 1.0])
    # Temporary columns should be cleaned up
    temp_cols = [c for c in adata.obs.columns if c.startswith("_cv_leiden")]
    assert len(temp_cols) == 0
