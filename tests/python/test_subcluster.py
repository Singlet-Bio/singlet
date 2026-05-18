# SPDX-License-Identifier: MIT
"""Tests for singlet.leiden_subclustering()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=120, m=200, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.leiden(adata)
    return adata


def test_subcluster_basic():
    adata = _make_adata()
    cats = list(adata.obs["leiden"].cat.categories[:2])
    singlet.leiden_subclustering(adata, restrict_to=("leiden", cats))
    assert "leiden_sub" in adata.obs.columns


def test_subcluster_key_added():
    adata = _make_adata()
    cats = list(adata.obs["leiden"].cat.categories[:1])
    singlet.leiden_subclustering(adata, restrict_to=("leiden", cats), key_added="my_sub")
    assert "my_sub" in adata.obs.columns


def test_subcluster_labels_contain_comma():
    adata = _make_adata()
    cats = list(adata.obs["leiden"].cat.categories[:2])
    singlet.leiden_subclustering(adata, restrict_to=("leiden", cats))
    # Subclustered cells should have comma-separated labels
    sub_labels = adata.obs["leiden_sub"]
    has_comma = any("," in str(label) for label in sub_labels)
    assert has_comma


def test_subcluster_copy():
    adata = _make_adata()
    cats = list(adata.obs["leiden"].cat.categories[:1])
    result = singlet.leiden_subclustering(adata, restrict_to=("leiden", cats), copy=True)
    assert result is not None
    assert "leiden_sub" not in adata.obs.columns
    assert "leiden_sub" in result.obs.columns


def test_subcluster_no_obs_key_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.leiden_subclustering(adata, restrict_to=("nonexistent", ["0"]))


def test_subcluster_all_cells_assigned():
    adata = _make_adata()
    cats = list(adata.obs["leiden"].cat.categories[:2])
    singlet.leiden_subclustering(adata, restrict_to=("leiden", cats))
    assert adata.obs["leiden_sub"].notna().all()
    assert len(adata.obs["leiden_sub"]) == adata.n_obs
