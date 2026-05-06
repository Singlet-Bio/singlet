"""Tests for singlet.scanorama_integrate()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_batch_adata(n=120, m=200, seed=42):
    """Create a single AnnData with two batches and a batch effect."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    # Add batch effect to second half
    X[n // 2 :] += 3.0
    adata = AnnData(X=X)
    adata.obs["batch"] = ["b0"] * (n // 2) + ["b1"] * (n // 2)
    singlet.pca(adata)
    return adata


def _make_two_adatas(n=60, m=200, seed=42):
    """Create two separate AnnData objects with a batch effect."""
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n, m)).astype(np.float32)
    X2 = rng.standard_normal((n, m)).astype(np.float32) + 3.0
    genes = [f"gene_{i}" for i in range(m)]

    adata1 = AnnData(X=X1)
    adata1.var_names = genes
    singlet.pca(adata1)

    adata2 = AnnData(X=X2)
    adata2.var_names = genes
    singlet.pca(adata2)

    return [adata1, adata2]


def test_scanorama_single_adata_basic():
    adata = _make_batch_adata()
    result = singlet.scanorama_integrate(adata, batch_key="batch")
    assert result is adata
    assert "X_scanorama" in adata.obsm
    assert adata.obsm["X_scanorama"].shape[0] == adata.n_obs


def test_scanorama_reduces_batch_effect():
    adata = _make_batch_adata()
    singlet.scanorama_integrate(adata, batch_key="batch")

    b0_mask = adata.obs["batch"] == "b0"
    b1_mask = adata.obs["batch"] == "b1"

    # Original PCA batch difference
    orig_diff = np.abs(
        adata.obsm["X_pca"][b0_mask].mean(axis=0)
        - adata.obsm["X_pca"][b1_mask].mean(axis=0)
    ).mean()

    # Corrected batch difference
    corr_diff = np.abs(
        adata.obsm["X_scanorama"][b0_mask].mean(axis=0)
        - adata.obsm["X_scanorama"][b1_mask].mean(axis=0)
    ).mean()

    assert corr_diff < orig_diff


def test_scanorama_list_mode():
    adata_list = _make_two_adatas()
    result = singlet.scanorama_integrate(adata_list)
    assert "X_scanorama" in result.obsm
    assert result.n_obs == 120  # 60 + 60


def test_scanorama_list_reduces_batch_effect():
    adata_list = _make_two_adatas()
    result = singlet.scanorama_integrate(adata_list)

    # Check the merged result has less batch difference
    batch_col = result.obs["_scanorama_batch"]
    b0_mask = (batch_col == "batch_0").values
    b1_mask = (batch_col == "batch_1").values

    corr_diff = np.abs(
        result.obsm["X_scanorama"][b0_mask].mean(axis=0)
        - result.obsm["X_scanorama"][b1_mask].mean(axis=0)
    ).mean()

    # The correction should reduce the mean difference substantially
    # (original diff is ~3.0 in PCA space)
    assert corr_diff < 3.0


def test_scanorama_single_batch():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    adata.obs["batch"] = "single"
    singlet.pca(adata)
    singlet.scanorama_integrate(adata, batch_key="batch")
    # Single batch: X_scanorama should equal X_pca (truncated)
    n_comps = min(50, adata.obsm["X_pca"].shape[1])
    assert np.allclose(
        adata.obsm["X_scanorama"], adata.obsm["X_pca"][:, :n_comps]
    )


def test_scanorama_no_pca_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    adata.obs["batch"] = ["a"] * 25 + ["b"] * 25
    with pytest.raises(KeyError, match="X_pca"):
        singlet.scanorama_integrate(adata, batch_key="batch")


def test_scanorama_no_batch_key_raises():
    adata = _make_batch_adata()
    with pytest.raises(KeyError, match="nonexistent"):
        singlet.scanorama_integrate(adata, batch_key="nonexistent")


def test_scanorama_single_adata_no_batch_key_raises():
    adata = _make_batch_adata()
    with pytest.raises(ValueError, match="batch_key"):
        singlet.scanorama_integrate(adata)


def test_scanorama_values_finite():
    adata = _make_batch_adata()
    singlet.scanorama_integrate(adata, batch_key="batch")
    assert np.all(np.isfinite(adata.obsm["X_scanorama"]))


def test_scanorama_n_components():
    adata = _make_batch_adata()
    singlet.scanorama_integrate(adata, batch_key="batch", n_components=10)
    assert adata.obsm["X_scanorama"].shape[1] == 10


def test_scanorama_custom_sigma():
    adata = _make_batch_adata()
    singlet.scanorama_integrate(adata, batch_key="batch", sigma=5.0)
    assert "X_scanorama" in adata.obsm
    assert np.all(np.isfinite(adata.obsm["X_scanorama"]))


def test_scanorama_n_neighbors():
    adata = _make_batch_adata()
    singlet.scanorama_integrate(adata, batch_key="batch", n_neighbors=5)
    assert "X_scanorama" in adata.obsm


def test_scanorama_list_single_element():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    singlet.pca(adata)
    result = singlet.scanorama_integrate([adata])
    assert "X_scanorama" in result.obsm


def test_scanorama_empty_list_raises():
    with pytest.raises(ValueError):
        singlet.scanorama_integrate([])


def test_scanorama_preserves_obs():
    adata = _make_batch_adata()
    adata.obs["extra"] = "hello"
    singlet.scanorama_integrate(adata, batch_key="batch")
    assert "extra" in adata.obs.columns
    assert (adata.obs["extra"] == "hello").all()
