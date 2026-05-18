# SPDX-License-Identifier: MIT
"""Tests for singlet.knn_impute()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n=80, m=150, seed=42, sparse=False):
    """Create test AnnData with PCA for kNN imputation tests."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, size=(n, m)).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    # Add PCA embedding
    adata.obsm["X_pca"] = rng.standard_normal((n, 20)).astype(np.float32)
    return adata


def test_knn_impute_basic():
    """Basic imputation creates output layer."""
    adata = _make_adata()
    result = singlet.knn_impute(adata, n_neighbors=10)
    assert result is adata
    assert "knn_imputed" in adata.layers
    assert adata.layers["knn_imputed"].shape == adata.X.shape


def test_knn_impute_dtype():
    """Output should be float32."""
    adata = _make_adata()
    singlet.knn_impute(adata, n_neighbors=10)
    assert adata.layers["knn_imputed"].dtype == np.float32


def test_knn_impute_smooths_noise():
    """Imputed values should be smoother (lower variance across neighbors)."""
    rng = np.random.default_rng(123)
    n, m = 100, 50
    # Create data with clear structure + noise
    signal = np.repeat(rng.standard_normal((10, m)), 10, axis=0)
    noise = rng.standard_normal((n, m)) * 2
    X = (signal + noise).astype(np.float32)
    # Make non-negative
    X = np.maximum(X, 0)

    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(m)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    # Use the signal as PCA (cells in same group are close)
    adata.obsm["X_pca"] = signal.astype(np.float32)

    singlet.knn_impute(adata, n_neighbors=8)
    imputed = adata.layers["knn_imputed"]

    # Imputed should be closer to signal than noisy X
    err_original = np.mean((X - signal) ** 2)
    err_imputed = np.mean((imputed - signal) ** 2)
    assert err_imputed < err_original


def test_knn_impute_uniform_weights():
    """Uniform weighting should work."""
    adata = _make_adata()
    singlet.knn_impute(adata, n_neighbors=10, weights="uniform")
    assert "knn_imputed" in adata.layers
    assert adata.layers["knn_imputed"].shape == adata.X.shape


def test_knn_impute_sparse_input():
    """Should handle sparse X matrix."""
    adata = _make_adata(sparse=True)
    singlet.knn_impute(adata, n_neighbors=10)
    assert "knn_imputed" in adata.layers
    assert not sp.issparse(adata.layers["knn_imputed"])


def test_knn_impute_layer_input():
    """Should impute from specified layer."""
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy() * 2.0
    singlet.knn_impute(adata, n_neighbors=10, layer="raw")
    # Imputed values should be larger (since raw = 2*X)
    assert "knn_imputed" in adata.layers
    mean_imputed = adata.layers["knn_imputed"].mean()
    mean_x = np.asarray(adata.X).mean()
    assert mean_imputed > mean_x * 1.5


def test_knn_impute_invalid_weights():
    """Should raise ValueError for invalid weights."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="weights must be"):
        singlet.knn_impute(adata, weights="invalid")


def test_knn_impute_missing_rep():
    """Should raise KeyError for missing representation."""
    adata = _make_adata()
    del adata.obsm["X_pca"]
    with pytest.raises(KeyError, match="X_pca"):
        singlet.knn_impute(adata)


def test_knn_impute_small_dataset():
    """Works with very few cells."""
    adata = _make_adata(n=5, m=10)
    singlet.knn_impute(adata, n_neighbors=20)
    assert "knn_imputed" in adata.layers
    assert adata.layers["knn_imputed"].shape == (5, 10)


def test_knn_impute_single_cell():
    """Single cell should just return its own values."""
    rng = np.random.default_rng(55)
    X = rng.poisson(3, size=(1, 20)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(20)]
    adata.obs_names = ["cell_0"]
    adata.obsm["X_pca"] = rng.standard_normal((1, 5)).astype(np.float32)

    singlet.knn_impute(adata, n_neighbors=10)
    assert "knn_imputed" in adata.layers


def test_knn_impute_distance_vs_uniform():
    """Distance and uniform weighting should give different results."""
    adata = _make_adata(n=50, seed=88)
    singlet.knn_impute(adata, n_neighbors=10, weights="distance")
    dist_imputed = adata.layers["knn_imputed"].copy()

    singlet.knn_impute(adata, n_neighbors=10, weights="uniform")
    unif_imputed = adata.layers["knn_imputed"]

    # They should differ
    assert not np.allclose(dist_imputed, unif_imputed)


def test_knn_impute_deterministic():
    """Same input should give same output."""
    adata = _make_adata()
    singlet.knn_impute(adata, n_neighbors=10)
    result1 = adata.layers["knn_imputed"].copy()

    singlet.knn_impute(adata, n_neighbors=10)
    result2 = adata.layers["knn_imputed"]

    np.testing.assert_array_equal(result1, result2)
