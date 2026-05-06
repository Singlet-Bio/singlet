"""Tests for singlet.spatial_neighbors()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_spatial_adata(n=100, m=50, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    # Random 2D spatial coordinates
    adata.obsm["spatial"] = rng.standard_normal((n, 2)).astype(np.float32)
    return adata


def test_spatial_basic():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata)
    assert "spatial_connectivities" in adata.obsp
    assert "spatial_distances" in adata.obsp


def test_spatial_connectivity_shape():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata)
    assert adata.obsp["spatial_connectivities"].shape == (100, 100)


def test_spatial_n_neighbors():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata, n_neighbors=10)
    conn = adata.obsp["spatial_connectivities"]
    # Each cell should have at least some neighbors
    assert conn.nnz > 0


def test_spatial_radius():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata, radius=1.0)
    assert "spatial_connectivities" in adata.obsp


def test_spatial_copy():
    adata = _make_spatial_adata()
    result = singlet.spatial_neighbors(adata, copy=True)
    assert result is not None
    assert "spatial_connectivities" not in adata.obsp
    assert "spatial_connectivities" in result.obsp


def test_spatial_no_coords_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 30)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.spatial_neighbors(adata)


def test_spatial_key_added():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata, key_added="my_spatial")
    assert "my_spatial_connectivities" in adata.obsp
    assert "my_spatial_distances" in adata.obsp


def test_spatial_uns_stored():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata)
    assert "spatial_neighbors" in adata.uns
    assert "params" in adata.uns["spatial_neighbors"]


def test_spatial_distances_nonneg():
    adata = _make_spatial_adata()
    singlet.spatial_neighbors(adata)
    assert (adata.obsp["spatial_distances"].data >= 0).all()


def test_spatial_3d_coords():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 30)).astype(np.float32))
    adata.obsm["spatial"] = rng.standard_normal((50, 3)).astype(np.float32)
    singlet.spatial_neighbors(adata)
    assert "spatial_connectivities" in adata.obsp
