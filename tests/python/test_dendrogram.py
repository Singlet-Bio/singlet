# SPDX-License-Identifier: MIT
"""Tests for singlet.dendrogram()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._dendrogram import dendrogram


def _make_clustered_adata(n_cells=120, n_genes=50, n_clusters=4):
    """Create AnnData with clusters and PCA."""
    import anndata as ad

    rng = np.random.default_rng(42)
    cells_per_cluster = n_cells // n_clusters

    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    labels = []
    for c in range(n_clusters):
        start = c * cells_per_cluster
        end = (c + 1) * cells_per_cluster
        X[start:end, c * 5 : (c + 1) * 5] += 10
        labels.extend([f"cluster_{c}"] * cells_per_cluster)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["cluster"] = pd.Categorical(labels)

    # Add PCA
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 20)).astype(np.float32)
    # Make PCA reflect cluster structure
    for c in range(n_clusters):
        start = c * cells_per_cluster
        end = (c + 1) * cells_per_cluster
        adata.obsm["X_pca"][start:end, :3] += c * 5

    return adata


class TestDendrogram:
    def test_basic_inplace(self):
        """Should store result in adata.uns."""
        adata = _make_clustered_adata()
        ret = dendrogram(adata, "cluster")
        assert ret is None
        assert "dendrogram_cluster" in adata.uns
        result = adata.uns["dendrogram_cluster"]
        assert "linkage" in result
        assert "categories_ordered" in result

    def test_not_inplace(self):
        """inplace=False should return dict."""
        adata = _make_clustered_adata()
        result = dendrogram(adata, "cluster", inplace=False)
        assert isinstance(result, dict)
        assert "categories_ordered" in result
        assert "dendrogram_cluster" not in adata.uns

    def test_categories_ordered(self):
        """Should return all categories in some order."""
        adata = _make_clustered_adata()
        result = dendrogram(adata, "cluster", inplace=False)
        ordered = result["categories_ordered"]
        assert len(ordered) == 4
        assert set(ordered) == {f"cluster_{i}" for i in range(4)}

    def test_linkage_shape(self):
        """Linkage matrix should have correct shape (n-1 × 4)."""
        adata = _make_clustered_adata(n_clusters=5)
        result = dendrogram(adata, "cluster", inplace=False)
        Z = result["linkage"]
        assert Z.shape == (4, 4)  # 5 clusters → 4 merges

    def test_use_rep_none(self):
        """use_rep=None should use adata.X directly."""
        adata = _make_clustered_adata()
        result = dendrogram(adata, "cluster", use_rep=None, inplace=False)
        assert "categories_ordered" in result
        assert len(result["categories_ordered"]) == 4

    def test_method_options(self):
        """Should work with different linkage methods."""
        adata = _make_clustered_adata()
        for method in ["ward", "average", "complete", "single"]:
            result = dendrogram(adata, "cluster", method=method, inplace=False)
            assert len(result["categories_ordered"]) == 4

    def test_single_group(self):
        """Should handle single group gracefully."""
        import anndata as ad

        adata = ad.AnnData(X=sp.random(50, 30, format="csr"))
        adata.obs["group"] = pd.Categorical(["a"] * 50)
        result = dendrogram(adata, "group", inplace=False)
        assert result["categories_ordered"] == ["a"]

    def test_two_groups(self):
        """Should work with exactly 2 groups."""
        adata = _make_clustered_adata(n_clusters=2, n_cells=60)
        result = dendrogram(adata, "cluster", inplace=False)
        assert len(result["categories_ordered"]) == 2

    def test_missing_groupby_raises(self):
        """Should raise KeyError for missing column."""
        adata = _make_clustered_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            dendrogram(adata, "nonexistent")

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="dendrogram"):
            dendrogram("not_adata", "cluster")

    def test_public_api(self):
        assert hasattr(singlet, "dendrogram")
        assert callable(singlet.dendrogram)
