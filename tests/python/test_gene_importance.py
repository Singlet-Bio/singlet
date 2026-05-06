"""Tests for singlet.gene_importance()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData


def _make_adata_with_clusters(n_obs=150, n_vars=100, n_clusters=3, seed=42):
    """Create AnnData with clearly separable clusters."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, n_vars)).astype(np.float32)

    # Create cluster labels
    labels = np.array([str(i % n_clusters) for i in range(n_obs)])

    # Make some genes highly discriminative
    for cluster_id in range(n_clusters):
        mask = labels == str(cluster_id)
        # Each cluster has 5 marker genes with elevated expression
        marker_start = cluster_id * 10
        X[mask, marker_start : marker_start + 5] += 5.0

    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.obs["leiden"] = pd.Categorical(labels)
    return adata


def test_gene_importance_basic():
    """Basic invocation returns DataFrame."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(adata, groupby="leiden")

    assert isinstance(result, pd.DataFrame)
    assert "gene" in result.columns
    assert "importance" in result.columns
    assert "rank" in result.columns


def test_gene_importance_sorted():
    """Results are sorted by importance (descending)."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(adata, groupby="leiden")

    importances = result["importance"].values
    assert np.all(importances[:-1] >= importances[1:])


def test_gene_importance_n_top():
    """n_top limits the number of returned genes."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(adata, groupby="leiden", n_top=20)
    assert len(result) == 20


def test_gene_importance_n_top_exceeds_genes():
    """n_top > n_genes returns all genes."""
    adata = _make_adata_with_clusters(n_vars=50)
    result = singlet.gene_importance(adata, groupby="leiden", n_top=200)
    assert len(result) == 50


def test_gene_importance_random_forest():
    """Random forest method works."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(
        adata, groupby="leiden", method="random_forest", n_estimators=20
    )
    assert len(result) > 0
    assert result["importance"].sum() > 0


def test_gene_importance_gradient_boosting():
    """Gradient boosting method works."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(
        adata, groupby="leiden", method="gradient_boosting", n_estimators=20
    )
    assert len(result) > 0
    assert result["importance"].sum() > 0


def test_gene_importance_finds_markers():
    """Top genes should include discriminative marker genes."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(adata, groupby="leiden", n_top=30)

    top_genes = set(result["gene"].values)
    # Marker genes are gene_0-4, gene_10-14, gene_20-24
    expected_markers = {f"gene_{i}" for i in list(range(5)) + list(range(10, 15)) + list(range(20, 25))}
    overlap = top_genes & expected_markers
    # At least half of the markers should be in top 30
    assert len(overlap) >= 7


def test_gene_importance_stored_in_uns():
    """Results are stored in adata.uns."""
    adata = _make_adata_with_clusters()
    singlet.gene_importance(adata, groupby="leiden")

    assert "gene_importance" in adata.uns
    assert "importances" in adata.uns["gene_importance"]
    assert "top_genes" in adata.uns["gene_importance"]
    assert "params" in adata.uns["gene_importance"]


def test_gene_importance_invalid_method():
    """Invalid method raises ValueError."""
    adata = _make_adata_with_clusters()
    with pytest.raises(ValueError, match="method must be one of"):
        singlet.gene_importance(adata, groupby="leiden", method="invalid")


def test_gene_importance_missing_groupby():
    """Missing groupby column raises KeyError."""
    adata = _make_adata_with_clusters()
    with pytest.raises(KeyError, match="nonexistent"):
        singlet.gene_importance(adata, groupby="nonexistent")


def test_gene_importance_rank_column():
    """Rank column is sequential 1..n."""
    adata = _make_adata_with_clusters()
    result = singlet.gene_importance(adata, groupby="leiden", n_top=50)
    expected_ranks = np.arange(1, len(result) + 1)
    np.testing.assert_array_equal(result["rank"].values, expected_ranks)


def test_gene_importance_reproducible():
    """Same random_state gives same results."""
    adata = _make_adata_with_clusters()
    r1 = singlet.gene_importance(adata, groupby="leiden", random_state=0)
    r2 = singlet.gene_importance(adata, groupby="leiden", random_state=0)
    pd.testing.assert_frame_equal(r1, r2)


def test_gene_importance_sparse_input():
    """Works with sparse input matrix."""
    from scipy.sparse import csr_matrix

    adata = _make_adata_with_clusters()
    adata.X = csr_matrix(adata.X)
    result = singlet.gene_importance(adata, groupby="leiden")
    assert len(result) > 0


def test_gene_importance_single_class():
    """Raises error with only one class."""
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 20)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(20)]
    adata.obs["group"] = "A"
    with pytest.raises(ValueError, match="at least 2 classes"):
        singlet.gene_importance(adata, groupby="group")
