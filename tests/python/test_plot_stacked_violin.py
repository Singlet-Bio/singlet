# SPDX-License-Identifier: MIT
"""Tests for singlet.plot_stacked_violin()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=80, m=50, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    return adata


def test_stacked_violin_basic():
    adata = _make_adata()
    result = singlet.plot_stacked_violin(
        adata,
        ["gene_0", "gene_1", "gene_2"],
        groupby="group",
    )
    assert result is None


def test_stacked_violin_show_false():
    adata = _make_adata()
    fig = singlet.plot_stacked_violin(
        adata,
        ["gene_0", "gene_1"],
        groupby="group",
        show=False,
    )
    assert fig is not None


def test_stacked_violin_stripplot():
    adata = _make_adata()
    fig = singlet.plot_stacked_violin(
        adata,
        ["gene_0", "gene_1"],
        groupby="group",
        stripplot=True,
        jitter=0.2,
        show=False,
    )
    assert fig is not None


def test_stacked_violin_single_gene():
    adata = _make_adata()
    fig = singlet.plot_stacked_violin(
        adata,
        ["gene_0"],
        groupby="group",
        show=False,
    )
    assert fig is not None


def test_stacked_violin_save(tmp_path):
    adata = _make_adata()
    path = str(tmp_path / "violin.png")
    singlet.plot_stacked_violin(
        adata,
        ["gene_0", "gene_1"],
        groupby="group",
        save=path,
    )
    import os

    assert os.path.exists(path)


def test_stacked_violin_missing_groupby_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.plot_stacked_violin(
            adata,
            ["gene_0"],
            groupby="nonexistent",
        )


def test_stacked_violin_no_genes_raises():
    adata = _make_adata()
    with pytest.raises(ValueError):
        singlet.plot_stacked_violin(
            adata,
            ["fake_gene_xyz"],
            groupby="group",
        )


def test_stacked_violin_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy()
    fig = singlet.plot_stacked_violin(
        adata,
        ["gene_0"],
        groupby="group",
        layer="raw",
        show=False,
    )
    assert fig is not None
