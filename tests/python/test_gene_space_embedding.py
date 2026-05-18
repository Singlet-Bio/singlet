# SPDX-License-Identifier: MIT
"""Tests for singlet.gene_space_embedding()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._gene_space_embedding import gene_space_embedding


def _make_gene_adata(n_cells=100, n_genes=200, seed=42):
    """Create AnnData with gene expression patterns."""
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(seed)

    # Create data with correlated gene modules
    n_modules = 4
    genes_per_module = n_genes // n_modules
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Add correlated expression within modules
    for mod in range(n_modules):
        gene_start = mod * genes_per_module
        gene_end = gene_start + genes_per_module
        # Cells in certain range express this module highly
        cell_start = mod * (n_cells // n_modules)
        cell_end = cell_start + (n_cells // n_modules)
        X[cell_start:cell_end, gene_start:gene_end] += 10.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    # Normalize
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    return adata


class TestGeneSpaceEmbeddingBasic:
    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert isinstance(result, pd.DataFrame)

    def test_has_gene_column(self):
        """DataFrame should have a 'gene' column."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert "gene" in result.columns

    def test_has_dim_columns(self):
        """DataFrame should have dim_0, dim_1 columns."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert "dim_0" in result.columns
        assert "dim_1" in result.columns

    def test_correct_n_rows(self):
        """Number of rows should equal n_top_genes (or n_genes if fewer)."""
        adata = _make_gene_adata(n_genes=200)
        result = gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert len(result) == 50

    def test_stores_in_varm(self):
        """Should store full embedding matrix in adata.varm."""
        adata = _make_gene_adata(n_genes=100)
        gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert "gene_embedding" in adata.varm
        assert adata.varm["gene_embedding"].shape == (100, 2)

    def test_varm_has_nans_for_nonselected(self):
        """Non-selected genes should have NaN in varm."""
        adata = _make_gene_adata(n_genes=100)
        gene_space_embedding(adata, n_top_genes=30, method="pca")
        embedding = adata.varm["gene_embedding"]
        # Some entries should be NaN (non-selected genes)
        assert np.any(np.isnan(embedding))
        # Selected genes should not have NaN
        non_nan_count = np.sum(~np.isnan(embedding[:, 0]))
        assert non_nan_count == 30

    def test_n_components_3(self):
        """Should work with n_components=3."""
        adata = _make_gene_adata()
        result = gene_space_embedding(
            adata, n_top_genes=50, n_components=3, method="pca"
        )
        assert "dim_0" in result.columns
        assert "dim_1" in result.columns
        assert "dim_2" in result.columns
        assert adata.varm["gene_embedding"].shape[1] == 3


class TestGeneSpaceEmbeddingMethods:
    def test_pca_method(self):
        """PCA method should work."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert len(result) == 50
        assert not result["dim_0"].isna().any()

    def test_umap_method(self):
        """UMAP method should work."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="umap")
        assert len(result) == 50
        assert not result["dim_0"].isna().any()

    def test_tsne_method(self):
        """t-SNE method should work."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="tsne")
        assert len(result) == 50
        assert not result["dim_0"].isna().any()


class TestGeneSpaceEmbeddingEdgeCases:
    def test_fewer_genes_than_requested(self):
        """If fewer genes available than n_top_genes, use all."""
        adata = _make_gene_adata(n_genes=30)
        result = gene_space_embedding(adata, n_top_genes=100, method="pca")
        assert len(result) == 30

    def test_dense_matrix(self):
        """Should work with dense expression matrix."""
        adata = _make_gene_adata(n_genes=50)
        adata.X = adata.X.toarray()
        result = gene_space_embedding(adata, n_top_genes=30, method="pca")
        assert len(result) == 30

    def test_small_dataset(self):
        """Should work with very small dataset."""
        adata = _make_gene_adata(n_cells=20, n_genes=10)
        result = gene_space_embedding(adata, n_top_genes=5, method="pca")
        assert len(result) == 5

    def test_gene_names_match(self):
        """Gene names in result should be valid gene names from adata."""
        adata = _make_gene_adata()
        result = gene_space_embedding(adata, n_top_genes=50, method="pca")
        assert all(g in adata.var_names for g in result["gene"])

    def test_reproducible_with_same_seed(self):
        """Same random_state should give same results."""
        adata1 = _make_gene_adata()
        adata2 = _make_gene_adata()
        r1 = gene_space_embedding(adata1, n_top_genes=50, method="pca", random_state=0)
        r2 = gene_space_embedding(adata2, n_top_genes=50, method="pca", random_state=0)
        pd.testing.assert_frame_equal(r1, r2)


class TestGeneSpaceEmbeddingErrors:
    def test_invalid_method(self):
        """Should raise ValueError for unknown method."""
        adata = _make_gene_adata()
        with pytest.raises(ValueError, match="method must be"):
            gene_space_embedding(adata, method="invalid")

    def test_too_few_genes(self):
        """Should raise ValueError if fewer than 3 genes."""
        import anndata as ad

        adata = ad.AnnData(X=sp.csr_matrix(np.ones((10, 2))))
        adata.var_names = ["A", "B"]
        with pytest.raises(ValueError, match="at least 3 genes"):
            gene_space_embedding(adata)
