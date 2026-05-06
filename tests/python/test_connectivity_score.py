"""Tests for singlet.connectivity_score()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata_with_graph(n=150, m=50, n_groups=3, seed=42):
    """Create AnnData with a precomputed kNN graph and cluster labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    adata.obs["cluster"] = [f"c{i % n_groups}" for i in range(n)]

    # Build a simple kNN-like connectivity graph
    # Connect cells that are close in index (simulates neighborhood)
    k = 10
    rows, cols, vals = [], [], []
    for cell_idx in range(n):
        neighbors_list = rng.choice(
            [j for j in range(n) if j != cell_idx], size=k, replace=False
        )
        for nb in neighbors_list:
            rows.append(cell_idx)
            cols.append(nb)
            vals.append(1.0)

    conn = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # Symmetrize
    conn = conn + conn.T
    conn.data[:] = 1.0
    adata.obsp["connectivities"] = conn
    return adata


def test_connectivity_basic():
    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster")
    assert result.shape == (3, 3)


def test_connectivity_returns_dataframe():
    import pandas as pd

    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster")
    assert isinstance(result, pd.DataFrame)


def test_connectivity_diagonal_one():
    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster")
    np.testing.assert_allclose(np.diag(result.values), 1.0)


def test_connectivity_symmetric():
    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster")
    np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)


def test_connectivity_stored_in_uns():
    adata = _make_adata_with_graph()
    singlet.connectivity_score(adata, "cluster")
    assert "cluster_connectivity" in adata.uns
    assert adata.uns["cluster_connectivity"].shape == (3, 3)


def test_connectivity_range():
    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster")
    assert (result.values >= 0.0 - 1e-10).all()
    assert (result.values <= 1.0 + 1e-10).all()


def test_connectivity_overlap_method():
    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster", method="overlap")
    assert result.shape == (3, 3)
    np.testing.assert_allclose(np.diag(result.values), 1.0)


def test_connectivity_cosine_method():
    adata = _make_adata_with_graph()
    result = singlet.connectivity_score(adata, "cluster", method="cosine")
    assert result.shape == (3, 3)
    np.testing.assert_allclose(np.diag(result.values), 1.0)


def test_connectivity_invalid_method():
    adata = _make_adata_with_graph()
    with pytest.raises(ValueError, match="method must be"):
        singlet.connectivity_score(adata, "cluster", method="bad")


def test_connectivity_missing_groupby():
    adata = _make_adata_with_graph()
    with pytest.raises(KeyError, match="not found"):
        singlet.connectivity_score(adata, "nonexistent")


def test_connectivity_missing_graph():
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.standard_normal((20, 10)).astype(np.float32))
    adata.obs["group"] = ["a", "b"] * 10
    adata.var_names = [f"g{i}" for i in range(10)]
    adata.obs_names = [f"c{i}" for i in range(20)]
    with pytest.raises(KeyError, match="No kNN graph"):
        singlet.connectivity_score(adata, "group")


def test_connectivity_two_disconnected_clusters():
    """Two clusters with no edges between them should have connectivity 0."""
    n = 20
    X = np.zeros((n, 5), dtype=np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obs["group"] = ["A"] * 10 + ["B"] * 10

    # Build graph with edges only within clusters
    rows, cols, vals = [], [], []
    for cell_idx in range(10):
        for nb in range(10):
            if cell_idx != nb:
                rows.append(cell_idx)
                cols.append(nb)
                vals.append(1.0)
    for cell_idx in range(10, 20):
        for nb in range(10, 20):
            if cell_idx != nb:
                rows.append(cell_idx)
                cols.append(nb)
                vals.append(1.0)

    adata.obsp["connectivities"] = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n, n)
    )

    result = singlet.connectivity_score(adata, "group")
    # Off-diagonal should be 0
    assert result.loc["A", "B"] == 0.0
    assert result.loc["B", "A"] == 0.0


def test_connectivity_many_clusters():
    adata = _make_adata_with_graph(n=200, n_groups=10)
    result = singlet.connectivity_score(adata, "cluster")
    assert result.shape == (10, 10)


def test_connectivity_labels_match():
    adata = _make_adata_with_graph(n_groups=4)
    result = singlet.connectivity_score(adata, "cluster")
    expected_labels = sorted(adata.obs["cluster"].unique())
    assert list(result.index) == expected_labels
    assert list(result.columns) == expected_labels
