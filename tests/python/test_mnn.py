"""Tests for singlet.mnn_correct()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_batch_adata(n=120, m=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    # Add batch effect
    X[n // 2 :] += 2.0  # Batch shift
    adata = AnnData(X=X)
    adata.obs["batch"] = ["b0"] * (n // 2) + ["b1"] * (n // 2)
    singlet.pca(adata)
    return adata


def test_mnn_basic():
    adata = _make_batch_adata()
    singlet.mnn_correct(adata, batch_key="batch")
    assert "X_mnn" in adata.obsm
    assert adata.obsm["X_mnn"].shape[0] == adata.n_obs


def test_mnn_reduces_batch_effect():
    adata = _make_batch_adata()
    singlet.mnn_correct(adata, batch_key="batch")

    # After correction, batch means should be closer
    b0_mask = adata.obs["batch"] == "b0"
    b1_mask = adata.obs["batch"] == "b1"

    # Original PCA batch difference
    orig_diff = np.abs(
        adata.obsm["X_pca"][b0_mask].mean(axis=0) - adata.obsm["X_pca"][b1_mask].mean(axis=0)
    ).mean()

    # Corrected batch difference
    corr_diff = np.abs(
        adata.obsm["X_mnn"][b0_mask].mean(axis=0) - adata.obsm["X_mnn"][b1_mask].mean(axis=0)
    ).mean()

    assert corr_diff < orig_diff


def test_mnn_copy():
    adata = _make_batch_adata()
    result = singlet.mnn_correct(adata, batch_key="batch", copy=True)
    assert result is not None
    assert "X_mnn" not in adata.obsm
    assert "X_mnn" in result.obsm


def test_mnn_no_pca_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    adata.obs["batch"] = ["a"] * 25 + ["b"] * 25
    with pytest.raises(KeyError):
        singlet.mnn_correct(adata, batch_key="batch")


def test_mnn_no_batch_raises():
    adata = _make_batch_adata()
    with pytest.raises(KeyError):
        singlet.mnn_correct(adata, batch_key="nonexistent")


def test_mnn_single_batch():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    adata.obs["batch"] = "single"
    singlet.pca(adata)
    singlet.mnn_correct(adata, batch_key="batch")
    # Single batch: X_mnn should equal X_pca
    assert np.allclose(adata.obsm["X_mnn"], adata.obsm["X_pca"])


def test_mnn_n_pcs():
    adata = _make_batch_adata()
    singlet.mnn_correct(adata, batch_key="batch", n_pcs=10)
    assert adata.obsm["X_mnn"].shape[1] == 10


def test_mnn_values_finite():
    adata = _make_batch_adata()
    singlet.mnn_correct(adata, batch_key="batch")
    assert np.all(np.isfinite(adata.obsm["X_mnn"]))


def test_mnn_shape_preserved():
    adata = _make_batch_adata()
    n_pcs = adata.obsm["X_pca"].shape[1]
    singlet.mnn_correct(adata, batch_key="batch")
    assert adata.obsm["X_mnn"].shape == (adata.n_obs, n_pcs)
