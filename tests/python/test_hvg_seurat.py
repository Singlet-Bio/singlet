# SPDX-License-Identifier: MIT
"""Tests for singlet.highly_variable_genes_seurat_v3()."""

import numpy as np
import singlet
from anndata import AnnData


def _make_count_adata(n=200, m=500, seed=42):
    rng = np.random.default_rng(seed)
    # Simulate count-like data (Poisson)
    lam = rng.exponential(5, size=m)
    X = rng.poisson(lam[None, :], size=(n, m)).astype(np.float32)
    # Make some genes highly variable
    X[:, :20] = rng.poisson(50, size=(n, 20)).astype(np.float32)
    X[: n // 2, :10] += 100  # Strong differential in first 10 genes
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    return adata


def test_hvg_seurat_basic():
    adata = _make_count_adata()
    singlet.highly_variable_genes_seurat_v3(adata, n_top_genes=100)
    assert "highly_variable_seurat_v3" in adata.var.columns
    assert adata.var["highly_variable_seurat_v3"].sum() == 100


def test_hvg_seurat_var_columns():
    adata = _make_count_adata()
    singlet.highly_variable_genes_seurat_v3(adata)
    assert "means_seurat_v3" in adata.var.columns
    assert "variances_seurat_v3" in adata.var.columns
    assert "variances_norm_seurat_v3" in adata.var.columns


def test_hvg_seurat_copy():
    adata = _make_count_adata()
    result = singlet.highly_variable_genes_seurat_v3(adata, copy=True)
    assert result is not None
    assert "highly_variable_seurat_v3" not in adata.var.columns
    assert "highly_variable_seurat_v3" in result.var.columns


def test_hvg_seurat_n_top_genes():
    adata = _make_count_adata()
    singlet.highly_variable_genes_seurat_v3(adata, n_top_genes=50)
    assert adata.var["highly_variable_seurat_v3"].sum() == 50


def test_hvg_seurat_variable_genes_are_variable():
    adata = _make_count_adata()
    singlet.highly_variable_genes_seurat_v3(adata, n_top_genes=20)
    hvg_mask = adata.var["highly_variable_seurat_v3"].values
    # HVG should have higher normalized variance than non-HVG
    norm_var = adata.var["variances_norm_seurat_v3"].values
    mean_norm_var_hvg = norm_var[hvg_mask].mean()
    mean_norm_var_non_hvg = norm_var[~hvg_mask].mean()
    assert mean_norm_var_hvg > mean_norm_var_non_hvg


def test_hvg_seurat_batch():
    adata = _make_count_adata()
    adata.obs["batch"] = ["b0"] * 100 + ["b1"] * 100
    singlet.highly_variable_genes_seurat_v3(adata, batch_key="batch", n_top_genes=50)
    assert adata.var["highly_variable_seurat_v3"].sum() == 50


def test_hvg_seurat_means_positive():
    adata = _make_count_adata()
    singlet.highly_variable_genes_seurat_v3(adata)
    # Means should be non-negative for count data
    assert (adata.var["means_seurat_v3"] >= 0).all()


def test_hvg_seurat_layer():
    adata = _make_count_adata()
    adata.layers["counts"] = adata.X.copy()
    adata.X = np.zeros_like(adata.X)  # Zero out main X
    singlet.highly_variable_genes_seurat_v3(adata, layer="counts", n_top_genes=50)
    assert adata.var["highly_variable_seurat_v3"].sum() == 50
