# SPDX-License-Identifier: MIT
"""Tests for singlet.batch_evaluation()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n=120, m=100, seed=42, n_batches=3, n_labels=4):
    """Create test AnnData with batch and label annotations."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2, size=(n, m)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]

    # PCA embedding
    adata.obsm["X_pca"] = rng.standard_normal((n, 20)).astype(np.float32)

    # Batch labels
    batches = [f"batch_{i % n_batches}" for i in range(n)]
    adata.obs["batch"] = batches
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Cell type labels
    labels = [f"type_{i % n_labels}" for i in range(n)]
    adata.obs["celltype"] = labels
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    return adata


def test_batch_eval_basic():
    """Basic batch evaluation returns expected keys."""
    adata = _make_adata()
    result = singlet.batch_evaluation(adata, batch_key="batch")
    assert isinstance(result, dict)
    assert "kbet_acceptance_rate" in result
    assert "batch_lisi" in result
    assert "batch_asw" in result
    assert result["label_lisi"] is None  # no label_key provided


def test_batch_eval_stored_in_uns():
    """Result is stored in adata.uns['batch_evaluation']."""
    adata = _make_adata()
    result = singlet.batch_evaluation(adata, batch_key="batch")
    assert "batch_evaluation" in adata.uns
    assert adata.uns["batch_evaluation"] is result


def test_batch_eval_with_label_key():
    """Providing label_key computes label_lisi."""
    adata = _make_adata()
    result = singlet.batch_evaluation(
        adata, batch_key="batch", label_key="celltype"
    )
    assert result["label_lisi"] is not None
    assert result["label_lisi"] >= 1.0


def test_batch_eval_kbet_range():
    """kBET acceptance rate should be in [0, 1]."""
    adata = _make_adata()
    result = singlet.batch_evaluation(adata, batch_key="batch")
    assert 0.0 <= result["kbet_acceptance_rate"] <= 1.0


def test_batch_eval_lisi_range():
    """Batch LISI should be >= 1 and <= n_batches."""
    adata = _make_adata(n_batches=3)
    result = singlet.batch_evaluation(adata, batch_key="batch")
    assert result["batch_lisi"] >= 1.0
    assert result["batch_lisi"] <= 3.0 + 0.01  # small tolerance


def test_batch_eval_well_mixed():
    """Well-mixed batches should have high LISI and kBET."""
    rng = np.random.default_rng(42)
    n = 150

    # All cells from same distribution (perfect mixing)
    pca = rng.standard_normal((n, 20)).astype(np.float32)
    X = rng.poisson(2, size=(n, 50)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(50)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = pca

    # Random batch assignment (well-mixed)
    adata.obs["batch"] = [f"B{i % 3}" for i in range(n)]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    result = singlet.batch_evaluation(adata, batch_key="batch", n_neighbors=20)
    # Well-mixed should have high acceptance and LISI close to n_batches
    assert result["kbet_acceptance_rate"] > 0.3
    assert result["batch_lisi"] > 1.5


def test_batch_eval_poorly_mixed():
    """Poorly mixed batches should have low LISI."""
    rng = np.random.default_rng(77)
    n = 120

    # Create batches that are well-separated in PCA space
    pca = np.zeros((n, 20), dtype=np.float32)
    # Batch A: cells 0-39, centered at +10
    pca[:40, 0] = 10.0 + rng.standard_normal(40).astype(np.float32) * 0.1
    # Batch B: cells 40-79, centered at -10
    pca[40:80, 0] = -10.0 + rng.standard_normal(40).astype(np.float32) * 0.1
    # Batch C: cells 80-119, centered at +20
    pca[80:, 0] = 20.0 + rng.standard_normal(40).astype(np.float32) * 0.1

    X = rng.poisson(2, size=(n, 50)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(50)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = pca
    adata.obs["batch"] = ["A"] * 40 + ["B"] * 40 + ["C"] * 40
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    result = singlet.batch_evaluation(adata, batch_key="batch", n_neighbors=15)
    # Poorly mixed should have LISI close to 1
    assert result["batch_lisi"] < 1.5


def test_batch_eval_missing_batch_key():
    """Should raise KeyError for missing batch_key."""
    adata = _make_adata()
    with pytest.raises(KeyError, match="not_a_key"):
        singlet.batch_evaluation(adata, batch_key="not_a_key")


def test_batch_eval_missing_label_key():
    """Should raise KeyError for missing label_key."""
    adata = _make_adata()
    with pytest.raises(KeyError, match="not_a_key"):
        singlet.batch_evaluation(adata, batch_key="batch", label_key="not_a_key")


def test_batch_eval_missing_rep():
    """Should raise KeyError for missing representation."""
    adata = _make_adata()
    with pytest.raises(KeyError, match="X_harmony"):
        singlet.batch_evaluation(adata, batch_key="batch", use_rep="X_harmony")


def test_batch_eval_custom_rep():
    """Works with custom representation key."""
    adata = _make_adata()
    rng = np.random.default_rng(42)
    adata.obsm["X_harmony"] = rng.standard_normal((120, 10)).astype(np.float32)
    result = singlet.batch_evaluation(
        adata, batch_key="batch", use_rep="X_harmony"
    )
    assert "kbet_acceptance_rate" in result


def test_batch_eval_two_batches():
    """Works with only 2 batches."""
    rng = np.random.default_rng(42)
    n = 80
    X = rng.poisson(2, size=(n, 50)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(50)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = rng.standard_normal((n, 20)).astype(np.float32)
    adata.obs["batch"] = ["A" if i < 40 else "B" for i in range(n)]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    result = singlet.batch_evaluation(adata, batch_key="batch", n_neighbors=15)
    assert result["batch_lisi"] >= 1.0
    assert result["batch_lisi"] <= 2.0 + 0.01


def test_batch_eval_asw_type():
    """batch_asw should be a float."""
    adata = _make_adata()
    result = singlet.batch_evaluation(adata, batch_key="batch")
    assert isinstance(result["batch_asw"], float)
    assert -1.0 <= result["batch_asw"] <= 1.0


def test_batch_eval_small_dataset():
    """Works with very small datasets."""
    rng = np.random.default_rng(42)
    n = 10
    X = rng.poisson(2, size=(n, 20)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(20)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = rng.standard_normal((n, 5)).astype(np.float32)
    adata.obs["batch"] = ["A"] * 5 + ["B"] * 5
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    result = singlet.batch_evaluation(adata, batch_key="batch", n_neighbors=5)
    assert "kbet_acceptance_rate" in result
