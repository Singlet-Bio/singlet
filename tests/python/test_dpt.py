# SPDX-License-Identifier: MIT
"""Tests for singlet.dpt()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=80, m=200, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.diffmap(adata)
    return adata


def test_dpt_basic():
    adata = _make_adata()
    singlet.dpt(adata)
    assert "dpt_pseudotime" in adata.obs.columns
    assert len(adata.obs["dpt_pseudotime"]) == adata.n_obs


def test_dpt_range():
    adata = _make_adata()
    singlet.dpt(adata)
    pt = adata.obs["dpt_pseudotime"].values
    assert pt.min() >= 0.0
    assert pt.max() <= 1.0 + 1e-6


def test_dpt_root_cell():
    adata = _make_adata()
    singlet.dpt(adata, root_cell=5)
    assert adata.uns["iroot"] == 5
    assert adata.obs["dpt_pseudotime"].iloc[5] == 0.0


def test_dpt_root_is_zero():
    adata = _make_adata()
    singlet.dpt(adata)
    root = adata.uns["iroot"]
    assert adata.obs["dpt_pseudotime"].iloc[root] == 0.0


def test_dpt_copy():
    adata = _make_adata()
    result = singlet.dpt(adata, copy=True)
    assert result is not None
    assert "dpt_pseudotime" not in adata.obs.columns
    assert "dpt_pseudotime" in result.obs.columns


def test_dpt_no_diffmap_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.dpt(adata)


def test_dpt_n_dcs():
    adata = _make_adata()
    singlet.dpt(adata, n_dcs=3)
    assert "dpt_pseudotime" in adata.obs.columns


def test_dpt_values_finite():
    adata = _make_adata()
    singlet.dpt(adata)
    assert np.all(np.isfinite(adata.obs["dpt_pseudotime"].values))


def test_dpt_max_is_one():
    adata = _make_adata()
    singlet.dpt(adata)
    assert np.isclose(adata.obs["dpt_pseudotime"].max(), 1.0, atol=1e-5)
