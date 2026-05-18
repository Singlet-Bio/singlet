# SPDX-License-Identifier: MIT
"""Tests for singlet.metacell()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n=200, m=100, seed=42, sparse=False):
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, size=(n, m)).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    # Add PCA representation
    adata.obsm["X_pca"] = rng.standard_normal((n, 20)).astype(np.float32)
    return adata


def test_metacell_basic():
    adata = _make_adata()
    mc = singlet.metacell(adata, n_metacells=20)
    assert mc.shape[0] == 20
    assert mc.shape[1] == adata.n_vars


def test_metacell_obs_columns():
    adata = _make_adata()
    mc = singlet.metacell(adata, n_metacells=20)
    assert "metacell_size" in mc.obs.columns
    # Sizes should sum to total cells
    assert mc.obs["metacell_size"].sum() == adata.n_obs


def test_metacell_assigns_labels():
    adata = _make_adata()
    singlet.metacell(adata, n_metacells=20)
    assert "metacell" in adata.obs.columns
    assert adata.obs["metacell"].nunique() == 20


def test_metacell_groupby():
    adata = _make_adata(n=100, m=50)
    adata.obs["cluster"] = [f"c{i % 5}" for i in range(100)]
    mc = singlet.metacell(adata, groupby="cluster")
    assert mc.shape[0] == 5
    assert mc.obs["metacell_size"].sum() == 100


def test_metacell_groupby_preserves_var():
    adata = _make_adata(n=80, m=40)
    adata.obs["batch"] = [f"b{i % 3}" for i in range(80)]
    adata.var["feature_type"] = "Gene Expression"
    mc = singlet.metacell(adata, groupby="batch")
    assert "feature_type" in mc.var.columns
    assert list(mc.var_names) == list(adata.var_names)


def test_metacell_sparse_input():
    adata = _make_adata(sparse=True)
    mc = singlet.metacell(adata, n_metacells=15)
    assert mc.shape[0] == 15
    assert not sp.issparse(mc.X)


def test_metacell_mean_expression():
    """Metacell expression should be the mean of member cells."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((10, 5)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs_names = [f"c{i}" for i in range(10)]
    adata.obs["group"] = ["A"] * 5 + ["B"] * 5

    mc = singlet.metacell(adata, groupby="group")
    # Check that metacell A is the mean of first 5 cells
    expected_a = X[:5].mean(axis=0)
    expected_b = X[5:].mean(axis=0)

    # Find which row is A vs B
    a_idx = [i for i, name in enumerate(mc.obs_names) if "A" in name][0]
    b_idx = [i for i, name in enumerate(mc.obs_names) if "B" in name][0]

    np.testing.assert_allclose(mc.X[a_idx], expected_a, rtol=1e-5)
    np.testing.assert_allclose(mc.X[b_idx], expected_b, rtol=1e-5)


def test_metacell_n_exceeds_cells():
    """If n_metacells > n_cells, cap to n_cells."""
    adata = _make_adata(n=30, m=20)
    mc = singlet.metacell(adata, n_metacells=100)
    assert mc.shape[0] <= 30


def test_metacell_invalid_method():
    adata = _make_adata()
    with pytest.raises(ValueError, match="method must be"):
        singlet.metacell(adata, method="invalid")


def test_metacell_missing_groupby():
    adata = _make_adata()
    with pytest.raises(KeyError, match="not found"):
        singlet.metacell(adata, groupby="nonexistent")


def test_metacell_missing_rep():
    adata = _make_adata()
    del adata.obsm["X_pca"]
    with pytest.raises(KeyError, match="not found"):
        singlet.metacell(adata, n_metacells=10)


def test_metacell_single_cell():
    """Single cell should produce 1 metacell."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1, 10)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(10)]
    adata.obs_names = ["c0"]
    adata.obsm["X_pca"] = rng.standard_normal((1, 5)).astype(np.float32)
    mc = singlet.metacell(adata, n_metacells=1)
    assert mc.shape[0] == 1
