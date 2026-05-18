# SPDX-License-Identifier: MIT
"""Tests for singlet.plot_scatter()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=80, m=150, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.umap(adata)
    singlet.leiden(adata)
    adata.obs["continuous_val"] = rng.standard_normal(n)
    return adata


def test_plot_scatter_basic():
    adata = _make_adata()
    result = singlet.plot_scatter(adata, basis="umap")
    assert result is None  # show=True returns None


def test_plot_scatter_no_color():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="umap", show=False)
    assert ax is not None


def test_plot_scatter_categorical_color():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="umap", color="leiden", show=False)
    assert ax is not None


def test_plot_scatter_continuous_color():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="umap", color="continuous_val", show=False)
    assert ax is not None


def test_plot_scatter_gene_color():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="umap", color="gene_0", show=False)
    assert ax is not None


def test_plot_scatter_pca_basis():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="pca", show=False)
    assert ax is not None


def test_plot_scatter_missing_basis_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.plot_scatter(adata, basis="nonexistent")


def test_plot_scatter_missing_color_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.plot_scatter(adata, basis="umap", color="nonexistent_col")


def test_plot_scatter_title():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="umap", title="My Plot", show=False)
    assert ax.get_title() == "My Plot"


def test_plot_scatter_save(tmp_path):
    adata = _make_adata()
    path = str(tmp_path / "test.png")
    singlet.plot_scatter(adata, basis="umap", save=path)
    import os

    assert os.path.exists(path)


def test_plot_scatter_custom_size():
    adata = _make_adata()
    ax = singlet.plot_scatter(adata, basis="umap", size=5, show=False)
    assert ax is not None
