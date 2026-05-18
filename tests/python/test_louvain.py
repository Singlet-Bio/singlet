# SPDX-License-Identifier: MIT
"""Tests for singlet.louvain()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=100, m=200, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    return adata


def test_louvain_basic():
    adata = _make_adata()
    singlet.louvain(adata)
    assert "louvain" in adata.obs.columns
    assert adata.obs["louvain"].dtype.name == "category"
    assert len(adata.obs["louvain"].cat.categories) >= 1


def test_louvain_resolution():
    adata = _make_adata()
    singlet.louvain(adata, resolution=0.1, key_added="low")
    singlet.louvain(adata, resolution=5.0, key_added="high")
    assert len(adata.obs["low"].cat.categories) <= len(adata.obs["high"].cat.categories)


def test_louvain_copy():
    adata = _make_adata()
    result = singlet.louvain(adata, copy=True)
    assert result is not None
    assert "louvain" not in adata.obs.columns
    assert "louvain" in result.obs.columns


def test_louvain_key_added():
    adata = _make_adata()
    singlet.louvain(adata, key_added="my_clusters")
    assert "my_clusters" in adata.obs.columns


def test_louvain_no_neighbors_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.louvain(adata)


def test_louvain_all_cells_assigned():
    adata = _make_adata()
    singlet.louvain(adata)
    assert adata.obs["louvain"].notna().all()
    assert len(adata.obs["louvain"]) == adata.n_obs


def test_louvain_categories_are_string_ints():
    adata = _make_adata()
    singlet.louvain(adata)
    for cat in adata.obs["louvain"].cat.categories:
        int(cat)  # Should not raise
