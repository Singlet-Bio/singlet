"""Tests for singlet.log1p(), expm1(), sqrt_transform()."""

import numpy as np
import singlet
from anndata import AnnData
from scipy.sparse import csr_matrix


def _make_adata(n=50, m=30, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n, m)).astype(np.float32)
    return AnnData(X=X)


def test_log1p_basic():
    adata = _make_adata()
    orig = adata.X.copy()
    singlet.log1p(adata)
    expected = np.log1p(orig)
    assert np.allclose(adata.X, expected, atol=1e-5)


def test_log1p_copy():
    adata = _make_adata()
    orig_sum = adata.X.sum()
    result = singlet.log1p(adata, copy=True)
    assert result is not None
    assert np.isclose(adata.X.sum(), orig_sum)


def test_log1p_base():
    adata = _make_adata()
    singlet.log1p(adata, base=2)
    # log2(1+x) = log(1+x) / log(2)
    adata2 = _make_adata()
    expected = np.log1p(adata2.X) / np.log(2)
    assert np.allclose(adata.X, expected, atol=1e-5)


def test_log1p_sparse():
    rng = np.random.default_rng(42)
    X = csr_matrix(rng.poisson(3, size=(30, 20)).astype(np.float32))
    adata = AnnData(X=X)
    singlet.log1p(adata)
    from scipy.sparse import issparse

    assert issparse(adata.X)


def test_expm1_basic():
    adata = _make_adata()
    singlet.log1p(adata)
    singlet.expm1(adata)
    # Should recover original (approximately)
    orig = _make_adata()
    assert np.allclose(adata.X, orig.X, atol=1e-4)


def test_expm1_copy():
    adata = _make_adata()
    singlet.log1p(adata)
    logged_sum = adata.X.sum()
    result = singlet.expm1(adata, copy=True)
    assert result is not None
    assert np.isclose(adata.X.sum(), logged_sum, atol=1e-5)


def test_sqrt_basic():
    adata = _make_adata()
    orig = adata.X.copy()
    singlet.sqrt_transform(adata)
    expected = np.sqrt(orig)
    assert np.allclose(adata.X, expected, atol=1e-5)


def test_sqrt_copy():
    adata = _make_adata()
    result = singlet.sqrt_transform(adata, copy=True)
    assert result is not None


def test_sqrt_sparse():
    rng = np.random.default_rng(42)
    X = csr_matrix(rng.poisson(5, size=(30, 20)).astype(np.float32))
    adata = AnnData(X=X)
    singlet.sqrt_transform(adata)
    from scipy.sparse import issparse

    assert issparse(adata.X)


def test_log1p_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy()
    singlet.log1p(adata, layer="raw")
    # X should be unchanged
    orig = _make_adata()
    assert np.allclose(adata.X, orig.X)
    # Layer should be transformed
    assert not np.allclose(np.asarray(adata.layers["raw"]), orig.X)
