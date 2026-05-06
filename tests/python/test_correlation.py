"""Tests for singlet.correlation_matrix()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=80, m=50, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    return adata


def test_correlation_basic():
    adata = _make_adata()
    result = singlet.correlation_matrix(adata, var_names=["gene_0", "gene_1", "gene_2"])
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)


def test_correlation_symmetric():
    adata = _make_adata()
    result = singlet.correlation_matrix(adata, var_names=["gene_0", "gene_1", "gene_2"])
    assert np.allclose(result.values, result.values.T, atol=1e-10)


def test_correlation_diagonal_one():
    adata = _make_adata()
    result = singlet.correlation_matrix(adata, var_names=["gene_0", "gene_1"])
    assert np.allclose(np.diag(result.values), 1.0, atol=1e-6)


def test_correlation_spearman():
    adata = _make_adata()
    result = singlet.correlation_matrix(
        adata,
        var_names=["gene_0", "gene_1"],
        method="spearman",
    )
    assert result.shape == (2, 2)


def test_correlation_range():
    adata = _make_adata()
    result = singlet.correlation_matrix(adata, var_names=["gene_0", "gene_1", "gene_2"])
    assert (result.values >= -1.0 - 1e-10).all()
    assert (result.values <= 1.0 + 1e-10).all()


def test_correlation_groupby():
    adata = _make_adata()
    result = singlet.correlation_matrix(
        adata,
        var_names=["gene_0", "gene_1"],
        groupby="group",
    )
    assert result.shape == (2, 2)


def test_correlation_no_var_names():
    adata = _make_adata()
    result = singlet.correlation_matrix(adata)
    # Should use top 50 by variance
    assert result.shape[0] <= 50


def test_correlation_obs():
    adata = _make_adata(n=20, m=50)
    singlet.pca(adata)
    result = singlet.correlation_matrix(adata, use="obs")
    assert result.shape == (20, 20)


def test_correlation_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy()
    result = singlet.correlation_matrix(
        adata,
        var_names=["gene_0", "gene_1"],
        layer="raw",
    )
    assert result.shape == (2, 2)


def test_correlation_invalid_use_raises():
    adata = _make_adata()
    with pytest.raises(ValueError):
        singlet.correlation_matrix(adata, use="invalid")
