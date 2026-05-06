"""Tests for singlet.entropy_score()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata(n=100, m=200, seed=42, sparse=False, n_clusters=3):
    """Create test AnnData with PCA and cluster labels."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2, size=(n, m)).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    # Add PCA embedding
    adata.obsm["X_pca"] = rng.standard_normal((n, 20)).astype(np.float32)
    # Add cluster labels
    labels = [f"cluster_{i % n_clusters}" for i in range(n)]
    adata.obs["leiden"] = labels
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    return adata


def test_entropy_score_basic():
    """Basic neighborhood entropy computation."""
    adata = _make_adata()
    result = singlet.entropy_score(adata, groupby="leiden", n_neighbors=10)
    assert result is adata
    assert "neighborhood_entropy" in adata.obs.columns
    assert len(adata.obs["neighborhood_entropy"]) == 100


def test_entropy_score_values_bounded():
    """Entropy should be >= 0 and <= log2(n_clusters)."""
    adata = _make_adata(n_clusters=4)
    singlet.entropy_score(adata, groupby="leiden", n_neighbors=15)
    vals = adata.obs["neighborhood_entropy"].values
    assert np.all(vals >= 0)
    # Max entropy is log2(n_categories)
    assert np.all(vals <= np.log2(4) + 1e-10)


def test_entropy_score_pure_neighborhoods():
    """Cells with same-label neighbors should have entropy ~0."""
    rng = np.random.default_rng(99)
    n = 60
    # Create two well-separated clusters
    X = np.zeros((n, 50), dtype=np.float32)
    pca = np.zeros((n, 10), dtype=np.float32)
    # Cluster A: cells 0-29, centered at +5
    pca[:30, 0] = 5.0 + rng.standard_normal(30) * 0.1
    # Cluster B: cells 30-59, centered at -5
    pca[30:, 0] = -5.0 + rng.standard_normal(30) * 0.1

    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(50)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = pca
    adata.obs["cluster"] = ["A"] * 30 + ["B"] * 30
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    singlet.entropy_score(adata, groupby="cluster", n_neighbors=10)
    vals = adata.obs["neighborhood_entropy"].values
    # All cells should have near-zero entropy (pure neighborhoods)
    assert np.all(vals < 0.1)


def test_entropy_score_mixed_neighborhoods():
    """Cells with equal mix of labels should have high entropy."""
    rng = np.random.default_rng(77)
    n = 80
    # All cells in same region but alternating labels
    pca = rng.standard_normal((n, 10)).astype(np.float32) * 0.01
    X = np.zeros((n, 50), dtype=np.float32)

    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(50)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = pca
    # Alternating labels: A, B, A, B...
    adata.obs["cluster"] = ["A" if i % 2 == 0 else "B" for i in range(n)]
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    singlet.entropy_score(adata, groupby="cluster", n_neighbors=20)
    vals = adata.obs["neighborhood_entropy"].values
    # Should be close to log2(2) = 1.0
    assert np.mean(vals) > 0.8


def test_entropy_score_expression_mode():
    """When groupby=None, compute expression entropy."""
    adata = _make_adata()
    result = singlet.entropy_score(adata)
    assert result is adata
    assert "expression_entropy" in adata.obs.columns
    vals = adata.obs["expression_entropy"].values
    assert np.all(vals >= 0)


def test_entropy_score_expression_sparse():
    """Expression entropy works with sparse input."""
    adata = _make_adata(sparse=True)
    singlet.entropy_score(adata)
    assert "expression_entropy" in adata.obs.columns
    vals = adata.obs["expression_entropy"].values
    assert np.all(np.isfinite(vals))


def test_entropy_score_missing_groupby():
    """Should raise KeyError for invalid groupby key."""
    adata = _make_adata()
    with pytest.raises(KeyError, match="not_a_key"):
        singlet.entropy_score(adata, groupby="not_a_key")


def test_entropy_score_missing_rep():
    """Should raise KeyError for missing representation."""
    adata = _make_adata()
    del adata.obsm["X_pca"]
    with pytest.raises(KeyError, match="X_pca"):
        singlet.entropy_score(adata, groupby="leiden")


def test_entropy_score_small_dataset():
    """Works with very small datasets."""
    adata = _make_adata(n=5, m=10)
    singlet.entropy_score(adata, groupby="leiden", n_neighbors=10)
    assert "neighborhood_entropy" in adata.obs.columns
    assert len(adata.obs["neighborhood_entropy"]) == 5


def test_entropy_score_single_cluster():
    """All same label → entropy should be 0."""
    adata = _make_adata(n=50)
    adata.obs["uniform"] = "A"
    adata.obs["uniform"] = adata.obs["uniform"].astype("category")
    singlet.entropy_score(adata, groupby="uniform", n_neighbors=10)
    vals = adata.obs["neighborhood_entropy"].values
    assert np.allclose(vals, 0.0)
