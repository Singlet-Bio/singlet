"""Tests for singlet.nmf() — local NMF decomposition."""

import numpy as np
import pytest
import singlet
from anndata import AnnData
from scipy.sparse import csr_matrix


def _make_nonneg_adata(n=80, m=100, seed=42):
    rng = np.random.default_rng(seed)
    # Non-negative data (like counts or normalized)
    X = rng.poisson(5, size=(n, m)).astype(np.float32)
    return AnnData(X=X)


def test_nmf_basic():
    adata = _make_nonneg_adata()
    singlet.nmf(adata, n_components=5)
    assert "X_nmf" in adata.obsm
    assert adata.obsm["X_nmf"].shape == (80, 5)


def test_nmf_loadings():
    adata = _make_nonneg_adata()
    singlet.nmf(adata, n_components=5)
    assert "nmf_loadings" in adata.varm
    assert adata.varm["nmf_loadings"].shape == (100, 5)


def test_nmf_uns():
    adata = _make_nonneg_adata()
    singlet.nmf(adata, n_components=5)
    assert "nmf" in adata.uns
    assert "components" in adata.uns["nmf"]
    assert "reconstruction_err" in adata.uns["nmf"]


def test_nmf_nonneg_output():
    adata = _make_nonneg_adata()
    singlet.nmf(adata, n_components=5)
    assert (adata.obsm["X_nmf"] >= 0).all()
    assert (adata.varm["nmf_loadings"] >= 0).all()


def test_nmf_copy():
    adata = _make_nonneg_adata()
    result = singlet.nmf(adata, n_components=5, copy=True)
    assert result is not None
    assert "X_nmf" not in adata.obsm
    assert "X_nmf" in result.obsm


def test_nmf_negative_raises():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 30)).astype(np.float32)  # Has negatives
    adata = AnnData(X=X)
    with pytest.raises(ValueError):
        singlet.nmf(adata)


def test_nmf_sparse():
    rng = np.random.default_rng(42)
    X = csr_matrix(rng.poisson(3, size=(60, 80)).astype(np.float32))
    adata = AnnData(X=X)
    singlet.nmf(adata, n_components=5)
    assert adata.obsm["X_nmf"].shape == (60, 5)


def test_nmf_layer():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 40)).astype(np.float32))  # Negative X
    adata.layers["counts"] = rng.poisson(5, size=(50, 40)).astype(np.float32)
    singlet.nmf(adata, layer="counts", n_components=3)
    assert "X_nmf" in adata.obsm


def test_nmf_n_components():
    adata = _make_nonneg_adata()
    singlet.nmf(adata, n_components=10)
    assert adata.obsm["X_nmf"].shape[1] == 10


def test_nmf_reconstruction_err_decreases():
    adata = _make_nonneg_adata()
    singlet.nmf(adata, n_components=5, max_iter=10)
    err_5 = adata.uns["nmf"]["reconstruction_err"]
    singlet.nmf(adata, n_components=20, max_iter=10)
    err_20 = adata.uns["nmf"]["reconstruction_err"]
    # More components should give lower error
    assert err_20 <= err_5
