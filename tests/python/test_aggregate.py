# SPDX-License-Identifier: MIT
"""Tests for singlet.aggregate()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=60, m=30, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.poisson(5, size=(n, m)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    return adata


def test_aggregate_sum():
    adata = _make_adata()
    result = singlet.aggregate(adata, groupby="group", method="sum")
    assert result.n_obs == 3
    assert result.n_vars == 30


def test_aggregate_mean():
    adata = _make_adata()
    result = singlet.aggregate(adata, groupby="group", method="mean")
    assert result.n_obs == 3


def test_aggregate_median():
    adata = _make_adata()
    result = singlet.aggregate(adata, groupby="group", method="median")
    assert result.n_obs == 3


def test_aggregate_n_cells():
    adata = _make_adata()
    result = singlet.aggregate(adata, groupby="group")
    assert "n_cells" in result.obs.columns
    assert result.obs["n_cells"].sum() == adata.n_obs


def test_aggregate_sum_correct():
    adata = _make_adata()
    result = singlet.aggregate(adata, groupby="group", method="sum")
    # Check one group manually
    mask = adata.obs["group"] == "g0"
    expected = np.asarray(adata.X[mask.values]).sum(axis=0)
    assert np.allclose(result.X[0], expected, atol=1e-5)


def test_aggregate_no_groupby_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.aggregate(adata, groupby="nonexistent")


def test_aggregate_invalid_method_raises():
    adata = _make_adata()
    with pytest.raises(ValueError):
        singlet.aggregate(adata, groupby="group", method="invalid")


def test_aggregate_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy() * 2
    result = singlet.aggregate(adata, groupby="group", layer="raw")
    result_x = singlet.aggregate(adata, groupby="group")
    assert np.allclose(result.X, result_x.X * 2, atol=1e-4)
