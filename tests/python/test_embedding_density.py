"""Tests for singlet.embedding_density()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=100, m=200, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.umap(adata)
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    return adata


def test_density_basic():
    adata = _make_adata()
    singlet.embedding_density(adata)
    assert "umap_density" in adata.obs.columns
    assert len(adata.obs["umap_density"]) == adata.n_obs


def test_density_values_positive():
    adata = _make_adata()
    singlet.embedding_density(adata)
    assert (adata.obs["umap_density"] >= 0).all()


def test_density_groupby():
    adata = _make_adata()
    singlet.embedding_density(adata, groupby="group")
    for g in ["g0", "g1", "g2"]:
        assert f"umap_density_{g}" in adata.obs.columns


def test_density_specific_group():
    adata = _make_adata()
    singlet.embedding_density(adata, groupby="group", group="g0")
    assert "umap_density_g0" in adata.obs.columns
    assert "umap_density_g1" not in adata.obs.columns


def test_density_copy():
    adata = _make_adata()
    result = singlet.embedding_density(adata, copy=True)
    assert result is not None
    assert "umap_density" not in adata.obs.columns
    assert "umap_density" in result.obs.columns


def test_density_custom_key():
    adata = _make_adata()
    singlet.embedding_density(adata, key_added="my_density")
    assert "my_density" in adata.obs.columns


def test_density_no_embedding_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.embedding_density(adata)


def test_density_pca_basis():
    adata = _make_adata()
    singlet.embedding_density(adata, basis="pca")
    assert "pca_density" in adata.obs.columns


def test_density_finite_values():
    adata = _make_adata()
    singlet.embedding_density(adata)
    assert np.all(np.isfinite(adata.obs["umap_density"].values))
