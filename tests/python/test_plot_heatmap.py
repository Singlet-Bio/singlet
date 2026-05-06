"""Tests for singlet.plot_heatmap()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=60, m=50, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    return adata


def test_plot_heatmap_basic():
    adata = _make_adata()
    result = singlet.plot_heatmap(adata, ["gene_0", "gene_1", "gene_2"])
    assert result is None  # show=True


def test_plot_heatmap_show_false():
    adata = _make_adata()
    ax = singlet.plot_heatmap(adata, ["gene_0", "gene_1"], show=False)
    assert ax is not None


def test_plot_heatmap_groupby():
    adata = _make_adata()
    ax = singlet.plot_heatmap(
        adata,
        ["gene_0", "gene_1", "gene_2"],
        groupby="group",
        show=False,
    )
    assert ax is not None


def test_plot_heatmap_no_scale():
    adata = _make_adata()
    ax = singlet.plot_heatmap(
        adata,
        ["gene_0"],
        standard_scale=None,
        show=False,
    )
    assert ax is not None


def test_plot_heatmap_obs_scale():
    adata = _make_adata()
    ax = singlet.plot_heatmap(
        adata,
        ["gene_0", "gene_1"],
        standard_scale="obs",
        show=False,
    )
    assert ax is not None


def test_plot_heatmap_swap_axes():
    adata = _make_adata()
    ax = singlet.plot_heatmap(
        adata,
        ["gene_0", "gene_1", "gene_2"],
        swap_axes=True,
        show=False,
    )
    assert ax is not None


def test_plot_heatmap_save(tmp_path):
    adata = _make_adata()
    path = str(tmp_path / "heatmap.png")
    singlet.plot_heatmap(adata, ["gene_0", "gene_1"], save=path)
    import os

    assert os.path.exists(path)


def test_plot_heatmap_no_genes_raises():
    adata = _make_adata()
    with pytest.raises(ValueError):
        singlet.plot_heatmap(adata, ["nonexistent_gene"])


def test_plot_heatmap_custom_cmap():
    adata = _make_adata()
    ax = singlet.plot_heatmap(
        adata,
        ["gene_0", "gene_1"],
        cmap="hot",
        show=False,
    )
    assert ax is not None


def test_plot_heatmap_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy()
    ax = singlet.plot_heatmap(
        adata,
        ["gene_0", "gene_1"],
        layer="raw",
        show=False,
    )
    assert ax is not None
