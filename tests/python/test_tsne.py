"""Tests for singlet.tsne()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=50, m=100, seed=42):
    """Create a small test AnnData."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    return AnnData(X=X)


def test_tsne_basic():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    singlet.tsne(adata)
    assert "X_tsne" in adata.obsm
    assert adata.obsm["X_tsne"].shape == (50, 2)


def test_tsne_copy():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    result = singlet.tsne(adata, copy=True)
    assert result is not None
    assert isinstance(result, AnnData)
    assert "X_tsne" in result.obsm
    assert "X_tsne" not in adata.obsm


def test_tsne_n_pcs():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    singlet.tsne(adata, n_pcs=5)
    assert adata.obsm["X_tsne"].shape == (50, 2)


def test_tsne_perplexity():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    singlet.tsne(adata, perplexity=10)
    assert adata.obsm["X_tsne"].shape == (50, 2)


def test_tsne_use_rep():
    adata = _make_adata(50, 100)
    rng = np.random.default_rng(0)
    adata.obsm["X_test"] = rng.standard_normal((50, 10)).astype(np.float32)
    singlet.tsne(adata, use_rep="X_test")
    assert adata.obsm["X_tsne"].shape == (50, 2)


def test_tsne_no_pca_raises():
    adata = _make_adata(50, 100)
    with pytest.raises(KeyError, match="X_pca"):
        singlet.tsne(adata)


def test_tsne_custom_learning_rate():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    singlet.tsne(adata, learning_rate=200)
    assert adata.obsm["X_tsne"].shape == (50, 2)


def test_tsne_deterministic():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    adata1 = adata.copy()
    adata2 = adata.copy()
    singlet.tsne(adata1, random_state=7)
    singlet.tsne(adata2, random_state=7)
    np.testing.assert_array_equal(adata1.obsm["X_tsne"], adata2.obsm["X_tsne"])


def test_tsne_values_finite():
    adata = _make_adata(50, 100)
    singlet.pca(adata)
    singlet.tsne(adata)
    assert np.all(np.isfinite(adata.obsm["X_tsne"]))
