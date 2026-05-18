# SPDX-License-Identifier: MIT
"""Tests for singlet.to_df(), obs_df(), var_df()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData
from scipy.sparse import csr_matrix


def _make_adata(n=30, m=20, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    adata.obs["cluster"] = [f"c{i % 3}" for i in range(n)]
    adata.obs["score"] = rng.standard_normal(n)
    return adata


# to_df tests


def test_to_df_basic():
    adata = _make_adata()
    df = singlet.to_df(adata)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (30, 20)


def test_to_df_index():
    adata = _make_adata()
    df = singlet.to_df(adata)
    assert list(df.index) == list(adata.obs_names)
    assert list(df.columns) == list(adata.var_names)


def test_to_df_values():
    adata = _make_adata()
    df = singlet.to_df(adata)
    assert np.allclose(df.values, adata.X, atol=1e-6)


def test_to_df_sparse():
    rng = np.random.default_rng(42)
    X = csr_matrix(rng.standard_normal((20, 15)).astype(np.float32))
    adata = AnnData(X=X)
    df = singlet.to_df(adata)
    assert df.shape == (20, 15)


def test_to_df_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X * 2
    df = singlet.to_df(adata, layer="raw")
    assert np.allclose(df.values, adata.X * 2, atol=1e-5)


# obs_df tests


def test_obs_df_obs_column():
    adata = _make_adata()
    df = singlet.obs_df(adata, ["cluster"])
    assert "cluster" in df.columns
    assert len(df) == adata.n_obs


def test_obs_df_gene():
    adata = _make_adata()
    df = singlet.obs_df(adata, ["gene_0"])
    assert "gene_0" in df.columns
    assert len(df) == adata.n_obs


def test_obs_df_mixed():
    adata = _make_adata()
    df = singlet.obs_df(adata, ["cluster", "gene_0", "score"])
    assert df.shape == (30, 3)


def test_obs_df_obsm():
    adata = _make_adata()
    singlet.pca(adata)
    df = singlet.obs_df(adata, ["obsm:X_pca"])
    assert df.shape[0] == adata.n_obs
    assert df.shape[1] > 0


def test_obs_df_missing_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.obs_df(adata, ["nonexistent"])


# var_df tests


def test_var_df_var_column():
    adata = _make_adata()
    adata.var["mean_expr"] = np.mean(adata.X, axis=0)
    df = singlet.var_df(adata, ["mean_expr"])
    assert "mean_expr" in df.columns
    assert len(df) == adata.n_vars


def test_var_df_cell():
    adata = _make_adata()
    df = singlet.var_df(adata, ["cell_0"])
    assert "cell_0" in df.columns
    assert len(df) == adata.n_vars


def test_var_df_missing_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.var_df(adata, ["nonexistent"])
