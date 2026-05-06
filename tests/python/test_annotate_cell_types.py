"""Tests for singlet.annotate_cell_types()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata_with_clusters(n=150, m=50, n_clusters=3, seed=42):
    """Create AnnData with clusters that have distinct marker expression."""
    rng = np.random.default_rng(seed)

    # Base expression (low for all)
    X = rng.poisson(1, size=(n, m)).astype(np.float32)

    cells_per_cluster = n // n_clusters
    genes_per_cluster = 5

    # Cluster 0: genes 0-4 are markers (high expression)
    X[:cells_per_cluster, :genes_per_cluster] += 10.0

    # Cluster 1: genes 5-9 are markers
    X[cells_per_cluster : 2 * cells_per_cluster, genes_per_cluster : 2 * genes_per_cluster] += 10.0

    # Cluster 2: genes 10-14 are markers
    X[2 * cells_per_cluster :, 2 * genes_per_cluster : 3 * genes_per_cluster] += 10.0

    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["leiden"] = (
        ["0"] * cells_per_cluster
        + ["1"] * cells_per_cluster
        + ["2"] * (n - 2 * cells_per_cluster)
    )
    return adata


def _marker_dict():
    return {
        "T cell": [f"gene_{i}" for i in range(5)],
        "B cell": [f"gene_{i}" for i in range(5, 10)],
        "Monocyte": [f"gene_{i}" for i in range(10, 15)],
    }


def test_annotate_basic():
    adata = _make_adata_with_clusters()
    markers = _marker_dict()
    result = singlet.annotate_cell_types(adata, markers)
    assert result is adata
    assert "cell_type" in adata.obs.columns
    assert "cell_type_score" in adata.obs.columns


def test_annotate_correct_assignments():
    adata = _make_adata_with_clusters()
    markers = _marker_dict()
    singlet.annotate_cell_types(adata, markers)

    # Cluster 0 should be T cell (genes 0-4 are high)
    cluster_0_type = adata.obs.loc[adata.obs["leiden"] == "0", "cell_type"].iloc[0]
    assert cluster_0_type == "T cell"

    # Cluster 1 should be B cell
    cluster_1_type = adata.obs.loc[adata.obs["leiden"] == "1", "cell_type"].iloc[0]
    assert cluster_1_type == "B cell"

    # Cluster 2 should be Monocyte
    cluster_2_type = adata.obs.loc[adata.obs["leiden"] == "2", "cell_type"].iloc[0]
    assert cluster_2_type == "Monocyte"


def test_annotate_scores_positive():
    adata = _make_adata_with_clusters()
    markers = _marker_dict()
    singlet.annotate_cell_types(adata, markers)

    scores = adata.obs["cell_type_score"].values
    # All clusters should have positive scores (markers are clearly enriched)
    assert all(s > 0 for s in scores)


def test_annotate_unknown_when_no_markers():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.poisson(1, size=(60, 30)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(30)]
    adata.obs["leiden"] = ["0"] * 30 + ["1"] * 30

    # Markers that don't exist in dataset
    markers = {"Type_A": ["missing_gene_1", "missing_gene_2"]}
    singlet.annotate_cell_types(adata, markers)
    assert (adata.obs["cell_type"] == "Unknown").all()


def test_annotate_min_score_threshold():
    rng = np.random.default_rng(42)
    # Uniform expression — no enrichment
    adata = AnnData(X=np.ones((60, 30), dtype=np.float32))
    adata.var_names = [f"gene_{i}" for i in range(30)]
    adata.obs["leiden"] = ["0"] * 30 + ["1"] * 30

    markers = {"Type_A": ["gene_0", "gene_1"]}
    singlet.annotate_cell_types(adata, markers, min_score=5.0)
    assert (adata.obs["cell_type"] == "Unknown").all()


def test_annotate_custom_groupby():
    adata = _make_adata_with_clusters()
    adata.obs["custom_clusters"] = adata.obs["leiden"]
    markers = _marker_dict()
    singlet.annotate_cell_types(adata, markers, groupby="custom_clusters")
    assert "cell_type" in adata.obs.columns


def test_annotate_missing_groupby_raises():
    adata = _make_adata_with_clusters()
    markers = _marker_dict()
    with pytest.raises(KeyError, match="nonexistent"):
        singlet.annotate_cell_types(adata, markers, groupby="nonexistent")


def test_annotate_empty_marker_dict_raises():
    adata = _make_adata_with_clusters()
    with pytest.raises(ValueError, match="non-empty"):
        singlet.annotate_cell_types(adata, {})


def test_annotate_sparse_matrix():
    from scipy.sparse import csr_matrix

    adata = _make_adata_with_clusters()
    adata.X = csr_matrix(adata.X)
    markers = _marker_dict()
    singlet.annotate_cell_types(adata, markers)
    assert "cell_type" in adata.obs.columns

    # Should still assign correctly
    cluster_0_type = adata.obs.loc[adata.obs["leiden"] == "0", "cell_type"].iloc[0]
    assert cluster_0_type == "T cell"


def test_annotate_partial_markers():
    """Test when only some markers are present in dataset."""
    adata = _make_adata_with_clusters()
    markers = {
        "T cell": ["gene_0", "gene_1", "nonexistent_1", "nonexistent_2"],
        "B cell": ["gene_5", "gene_6"],
    }
    singlet.annotate_cell_types(adata, markers)
    # Should still work with partial markers
    cluster_0_type = adata.obs.loc[adata.obs["leiden"] == "0", "cell_type"].iloc[0]
    assert cluster_0_type == "T cell"


def test_annotate_consistent_per_cluster():
    """All cells in same cluster should have same annotation."""
    adata = _make_adata_with_clusters()
    markers = _marker_dict()
    singlet.annotate_cell_types(adata, markers)

    for cluster in adata.obs["leiden"].unique():
        mask = adata.obs["leiden"] == cluster
        types = adata.obs.loc[mask, "cell_type"].unique()
        assert len(types) == 1
