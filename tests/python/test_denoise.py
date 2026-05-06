"""Tests for singlet.denoise()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n=100, m=200, seed=42, sparse=False):
    rng = np.random.default_rng(seed)
    X = rng.poisson(2, size=(n, m)).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    return adata


def test_denoise_basic():
    adata = _make_adata()
    result = singlet.denoise(adata, n_components=10)
    assert result is adata
    assert "denoised" in adata.layers
    assert adata.layers["denoised"].shape == adata.X.shape


def test_denoise_dtype():
    adata = _make_adata()
    singlet.denoise(adata, n_components=10)
    assert adata.layers["denoised"].dtype == np.float32


def test_denoise_reduces_noise():
    """Denoised matrix should have lower rank (less variance in residuals)."""
    rng = np.random.default_rng(123)
    # Create low-rank signal + noise
    signal = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 200))
    noise = rng.standard_normal((100, 200)) * 0.5
    X = (signal + noise).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(200)]
    adata.obs_names = [f"c{i}" for i in range(100)]

    singlet.denoise(adata, n_components=5, method="svd")
    denoised = adata.layers["denoised"]

    # Denoised should be closer to signal than the noisy input
    residual_noisy = np.linalg.norm(X - signal)
    residual_denoised = np.linalg.norm(denoised - signal)
    assert residual_denoised < residual_noisy


def test_denoise_pca_method():
    adata = _make_adata()
    singlet.denoise(adata, n_components=10, method="pca")
    assert "denoised" in adata.layers
    assert adata.layers["denoised"].shape == adata.X.shape


def test_denoise_sparse_input():
    adata = _make_adata(sparse=True)
    singlet.denoise(adata, n_components=10)
    assert "denoised" in adata.layers
    assert not sp.issparse(adata.layers["denoised"])


def test_denoise_sparse_pca():
    adata = _make_adata(sparse=True)
    singlet.denoise(adata, n_components=10, method="pca")
    assert "denoised" in adata.layers
    assert adata.layers["denoised"].shape == adata.X.shape


def test_denoise_layer_input():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy() * 2
    singlet.denoise(adata, n_components=10, layer="raw")
    # Denoised should be based on the layer, not X
    assert "denoised" in adata.layers


def test_denoise_n_components_capped():
    """n_components larger than matrix dims should not error."""
    adata = _make_adata(n=20, m=30)
    singlet.denoise(adata, n_components=500)
    assert "denoised" in adata.layers


def test_denoise_invalid_method():
    adata = _make_adata()
    with pytest.raises(ValueError, match="method must be"):
        singlet.denoise(adata, method="invalid")


def test_denoise_shape_preserved():
    adata = _make_adata(n=50, m=80)
    singlet.denoise(adata, n_components=20)
    assert adata.layers["denoised"].shape == (50, 80)


def test_denoise_one_component():
    adata = _make_adata()
    singlet.denoise(adata, n_components=1)
    assert "denoised" in adata.layers
    # Rank-1 approximation
    denoised = adata.layers["denoised"]
    # Check it's approximately rank 1 (second singular value ~ 0)
    _, sigma, _ = np.linalg.svd(denoised, full_matrices=False)
    assert sigma[1] / sigma[0] < 1e-5
