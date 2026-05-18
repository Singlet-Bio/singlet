# SPDX-License-Identifier: MIT
"""Tests for singlet preprocessing recipes."""

import numpy as np
import singlet
from anndata import AnnData


def _make_count_adata(n=200, m=500, seed=42):
    rng = np.random.default_rng(seed)
    # Simulate count data
    X = rng.poisson(5, size=(n, m)).astype(np.float32)
    # Ensure some cells have enough genes and some genes enough cells
    X[:, :300] += 1  # Most genes expressed
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    return adata


def test_recipe_seurat_basic():
    adata = _make_count_adata()
    singlet.recipe_seurat(adata)
    assert "highly_variable" in adata.var.columns


def test_recipe_seurat_copy():
    adata = _make_count_adata()
    result = singlet.recipe_seurat(adata, copy=True)
    assert result is not None
    assert "highly_variable" in result.var.columns


def test_recipe_seurat_filters():
    adata = _make_count_adata()
    n_orig = adata.n_obs
    singlet.recipe_seurat(adata, min_genes=50)
    # Should have same or fewer cells
    assert adata.n_obs <= n_orig


def test_recipe_seurat_n_top_genes():
    adata = _make_count_adata()
    singlet.recipe_seurat(adata, n_top_genes=100)
    assert adata.var["highly_variable"].sum() <= 100


def test_recipe_zheng17_basic():
    adata = _make_count_adata()
    singlet.recipe_zheng17(adata, n_top_genes=100)
    # Should be subset to HVGs
    assert adata.n_vars <= 100


def test_recipe_zheng17_copy():
    adata = _make_count_adata()
    n_vars_orig = adata.n_vars
    result = singlet.recipe_zheng17(adata, n_top_genes=100, copy=True)
    assert result is not None
    assert result.n_vars <= 100
    assert adata.n_vars == n_vars_orig  # Original unchanged


def test_recipe_zheng17_normalized():
    adata = _make_count_adata()
    singlet.recipe_zheng17(adata, n_top_genes=50)
    # After normalize+log, values should not be raw counts
    assert adata.X.max() < 20  # log-transformed
