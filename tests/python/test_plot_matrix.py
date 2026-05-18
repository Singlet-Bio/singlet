# SPDX-License-Identifier: MIT
"""Tests for singlet.rank_genes_groups_matrixplot()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_de_adata(n=90, m=60, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    for i in range(n):
        g = i % 3
        X[i, g * 10 : (g + 1) * 10] += 3.0
    adata.X = X
    singlet.rank_genes_groups(adata, groupby="group")
    return adata


def test_matrixplot_basic():
    adata = _make_de_adata()
    result = singlet.rank_genes_groups_matrixplot(adata, groupby="group")
    assert result is None


def test_matrixplot_show_false():
    adata = _make_de_adata()
    ax = singlet.rank_genes_groups_matrixplot(adata, groupby="group", show=False)
    assert ax is not None


def test_matrixplot_n_genes():
    adata = _make_de_adata()
    ax = singlet.rank_genes_groups_matrixplot(
        adata, groupby="group", n_genes=3, show=False
    )
    assert ax is not None


def test_matrixplot_var_names():
    adata = _make_de_adata()
    ax = singlet.rank_genes_groups_matrixplot(
        adata,
        groupby="group",
        var_names=["gene_0", "gene_10", "gene_20"],
        show=False,
    )
    assert ax is not None


def test_matrixplot_groups():
    adata = _make_de_adata()
    ax = singlet.rank_genes_groups_matrixplot(
        adata, groupby="group", groups=["g0", "g1"], show=False
    )
    assert ax is not None


def test_matrixplot_no_de_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((30, 20)).astype(np.float32))
    adata.obs["g"] = ["a"] * 15 + ["b"] * 15
    with pytest.raises(KeyError):
        singlet.rank_genes_groups_matrixplot(adata, groupby="g")


def test_matrixplot_save(tmp_path):
    adata = _make_de_adata()
    path = str(tmp_path / "matrix.png")
    singlet.rank_genes_groups_matrixplot(adata, groupby="group", save=path)
    import os

    assert os.path.exists(path)


def test_matrixplot_no_scale():
    adata = _make_de_adata()
    ax = singlet.rank_genes_groups_matrixplot(
        adata, groupby="group", standard_scale=None, show=False
    )
    assert ax is not None
