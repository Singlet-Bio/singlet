"""Tests for singlet.diffmap()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata_with_neighbors(n=80, m=200, seed=42):
    """Create test AnnData with neighbors computed."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    adata = AnnData(X=X)
    singlet.pca(adata)
    singlet.neighbors(adata)
    return adata


def test_diffmap_basic():
    adata = _make_adata_with_neighbors(80, 200)
    singlet.diffmap(adata)
    assert "X_diffmap" in adata.obsm
    # Default n_comps=15, shape should be (80, 15)
    assert adata.obsm["X_diffmap"].shape[0] == 80
    assert adata.obsm["X_diffmap"].shape[1] > 0


def test_diffmap_copy():
    adata = _make_adata_with_neighbors(80, 200)
    result = singlet.diffmap(adata, copy=True)
    assert result is not None
    assert isinstance(result, AnnData)
    assert "X_diffmap" in result.obsm
    assert "X_diffmap" not in adata.obsm


def test_diffmap_n_comps():
    adata = _make_adata_with_neighbors(80, 200)
    singlet.diffmap(adata, n_comps=5)
    assert adata.obsm["X_diffmap"].shape == (80, 5)


def test_diffmap_no_neighbors_raises():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 200)).astype(np.float32)
    adata = AnnData(X=X)
    with pytest.raises(KeyError, match="connectivities"):
        singlet.diffmap(adata)


def test_diffmap_eigenvalues_stored():
    adata = _make_adata_with_neighbors(80, 200)
    singlet.diffmap(adata, n_comps=5)
    assert "diffmap_evals" in adata.uns
    evals = adata.uns["diffmap_evals"]
    assert len(evals) == 5
    # Check sorted descending
    for i in range(len(evals) - 1):
        assert evals[i] >= evals[i + 1]


def test_diffmap_eigenvalues_bounded():
    adata = _make_adata_with_neighbors(80, 200)
    singlet.diffmap(adata, n_comps=5)
    evals = adata.uns["diffmap_evals"]
    assert np.all(evals <= 1.0 + 1e-6)


def test_diffmap_values_finite():
    adata = _make_adata_with_neighbors(80, 200)
    singlet.diffmap(adata)
    assert np.all(np.isfinite(adata.obsm["X_diffmap"]))


def test_diffmap_deterministic():
    adata1 = _make_adata_with_neighbors(80, 200, seed=42)
    adata2 = _make_adata_with_neighbors(80, 200, seed=42)
    singlet.diffmap(adata1, n_comps=5)
    singlet.diffmap(adata2, n_comps=5)
    # Eigenvectors may have sign ambiguity, so compare absolute values
    np.testing.assert_allclose(
        np.abs(adata1.obsm["X_diffmap"]),
        np.abs(adata2.obsm["X_diffmap"]),
        atol=1e-5,
    )
