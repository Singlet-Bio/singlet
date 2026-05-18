# SPDX-License-Identifier: MIT
"""Tests for singlet.scrublet()."""

import numpy as np
import singlet
from anndata import AnnData


def _make_adata(n=150, m=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    # Simulate some doublets (averaged pairs)
    for i in range(n - 10, n):
        j, k = rng.integers(0, n - 10, size=2)
        X[i] = (X[j] + X[k]) / 2
    adata = AnnData(X=X)
    return adata


def test_scrublet_basic():
    adata = _make_adata()
    singlet.scrublet(adata)
    assert "doublet_score" in adata.obs.columns
    assert "predicted_doublet" in adata.obs.columns


def test_scrublet_scores_range():
    adata = _make_adata()
    singlet.scrublet(adata)
    scores = adata.obs["doublet_score"].values
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_scrublet_copy():
    adata = _make_adata()
    result = singlet.scrublet(adata, copy=True)
    assert result is not None
    assert "doublet_score" not in adata.obs.columns
    assert "doublet_score" in result.obs.columns


def test_scrublet_threshold_stored():
    adata = _make_adata()
    singlet.scrublet(adata)
    assert "scrublet" in adata.uns
    assert "threshold" in adata.uns["scrublet"]


def test_scrublet_manual_threshold():
    adata = _make_adata()
    singlet.scrublet(adata, threshold=0.3)
    # All cells above 0.3 should be predicted doublets
    mask = adata.obs["doublet_score"] > 0.3
    assert (adata.obs["predicted_doublet"][mask]).all()


def test_scrublet_predicted_is_bool():
    adata = _make_adata()
    singlet.scrublet(adata)
    assert adata.obs["predicted_doublet"].dtype == bool


def test_scrublet_sim_ratio():
    adata = _make_adata()
    singlet.scrublet(adata, sim_doublet_ratio=1.0)
    assert "doublet_score" in adata.obs.columns


def test_scrublet_n_pcs():
    adata = _make_adata()
    singlet.scrublet(adata, n_pcs=10)
    assert "doublet_score" in adata.obs.columns


def test_scrublet_deterministic():
    adata1 = _make_adata()
    adata2 = _make_adata()
    singlet.scrublet(adata1, random_state=0)
    singlet.scrublet(adata2, random_state=0)
    assert np.allclose(
        adata1.obs["doublet_score"].values,
        adata2.obs["doublet_score"].values,
    )
