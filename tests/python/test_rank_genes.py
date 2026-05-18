# SPDX-License-Identifier: MIT
"""Tests for singlet.rank_genes()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=60, m=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    return adata


def test_rank_variance():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="variance")
    assert isinstance(result, pd.DataFrame)
    assert "gene" in result.columns
    assert "variance" in result.columns
    assert len(result) == adata.n_vars


def test_rank_mean():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="mean")
    assert "mean" in result.columns


def test_rank_dispersion():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="dispersion")
    assert "dispersion" in result.columns


def test_rank_dropout():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="dropout")
    assert "dropout_rate" in result.columns


def test_rank_cv():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="cv")
    assert "cv" in result.columns


def test_rank_n_top():
    adata = _make_adata()
    result = singlet.rank_genes(adata, n_top=10)
    assert len(result) == 10


def test_rank_sorted():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="variance")
    variances = result["variance"].values
    assert all(variances[i] >= variances[i + 1] for i in range(len(variances) - 1))


def test_rank_groupby():
    adata = _make_adata()
    result = singlet.rank_genes(adata, method="variance", groupby="group", n_top=5)
    assert "group" in result.columns
    # Should have entries for each group
    assert len(result["group"].unique()) == 3


def test_rank_invalid_method_raises():
    adata = _make_adata()
    with pytest.raises(ValueError):
        singlet.rank_genes(adata, method="invalid")


def test_rank_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy()
    result = singlet.rank_genes(adata, layer="raw")
    assert len(result) == adata.n_vars
