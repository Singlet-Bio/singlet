"""Tests for singlet.downsample_counts()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData
from scipy.sparse import csr_matrix


def _make_count_adata(n=50, m=100, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.poisson(10, size=(n, m)).astype(np.float32)
    return AnnData(X=X)


def test_downsample_per_cell():
    adata = _make_count_adata()
    singlet.downsample_counts(adata, counts_per_cell=200)
    cell_totals = np.asarray(adata.X.sum(axis=1)).flatten()
    assert (cell_totals <= 200 + 1).all()  # +1 for float precision


def test_downsample_total():
    adata = _make_count_adata()
    singlet.downsample_counts(adata, total_counts=1000)
    total = adata.X.sum()
    assert abs(total - 1000) < 2  # Should be exactly 1000


def test_downsample_copy():
    adata = _make_count_adata()
    orig_sum = adata.X.sum()
    result = singlet.downsample_counts(adata, total_counts=1000, copy=True)
    assert result is not None
    assert abs(adata.X.sum() - orig_sum) < 1  # Original unchanged
    assert abs(result.X.sum() - 1000) < 2


def test_downsample_no_args_raises():
    adata = _make_count_adata()
    with pytest.raises(ValueError):
        singlet.downsample_counts(adata)


def test_downsample_both_args_raises():
    adata = _make_count_adata()
    with pytest.raises(ValueError):
        singlet.downsample_counts(adata, total_counts=100, counts_per_cell=50)


def test_downsample_nonnegative():
    adata = _make_count_adata()
    singlet.downsample_counts(adata, counts_per_cell=100)
    assert (np.asarray(adata.X) >= 0).all()


def test_downsample_sparse():
    rng = np.random.default_rng(42)
    X = csr_matrix(rng.poisson(5, size=(30, 50)).astype(np.float32))
    adata = AnnData(X=X)
    singlet.downsample_counts(adata, counts_per_cell=50)
    from scipy.sparse import issparse

    assert issparse(adata.X)


def test_downsample_low_count_unchanged():
    adata = _make_count_adata()
    # Set first cell to have very few counts
    adata.X[0] = 0
    adata.X[0, 0] = 5
    singlet.downsample_counts(adata, counts_per_cell=1000)
    # Cell with 5 counts should be unchanged
    assert adata.X[0].sum() == 5


def test_downsample_deterministic():
    adata1 = _make_count_adata()
    adata2 = _make_count_adata()
    singlet.downsample_counts(adata1, counts_per_cell=100, random_state=42)
    singlet.downsample_counts(adata2, counts_per_cell=100, random_state=42)
    assert np.allclose(np.asarray(adata1.X), np.asarray(adata2.X))
