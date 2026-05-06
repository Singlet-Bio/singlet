"""Tests for singlet.draw_graph()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=80, m=150, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    return adata


def test_draw_graph_basic():
    adata = _make_adata()
    singlet.draw_graph(adata, n_iterations=50)
    assert "X_draw_graph_fa" in adata.obsm
    assert adata.obsm["X_draw_graph_fa"].shape == (80, 2)


def test_draw_graph_fr():
    adata = _make_adata()
    singlet.draw_graph(adata, layout="fr", key_added="draw_graph_fr", n_iterations=50)
    assert "X_draw_graph_fr" in adata.obsm


def test_draw_graph_copy():
    adata = _make_adata()
    result = singlet.draw_graph(adata, copy=True, n_iterations=50)
    assert result is not None
    assert "X_draw_graph_fa" not in adata.obsm
    assert "X_draw_graph_fa" in result.obsm


def test_draw_graph_no_neighbors_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.draw_graph(adata)


def test_draw_graph_finite():
    adata = _make_adata()
    singlet.draw_graph(adata, n_iterations=50)
    assert np.all(np.isfinite(adata.obsm["X_draw_graph_fa"]))


def test_draw_graph_custom_key():
    adata = _make_adata()
    singlet.draw_graph(adata, key_added="my_layout", n_iterations=50)
    assert "X_my_layout" in adata.obsm


def test_draw_graph_iterations():
    adata = _make_adata()
    singlet.draw_graph(adata, n_iterations=10)
    assert "X_draw_graph_fa" in adata.obsm
