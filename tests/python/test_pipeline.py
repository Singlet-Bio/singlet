"""End-to-end integration test: full singlet analysis pipeline.

Tests the complete workflow from data creation through differential expression,
verifying that all pipeline steps chain together correctly.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet


@pytest.fixture
def raw_adata():
    """Create a realistic raw count matrix with cluster structure."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells = 300
    n_genes = 2000

    # Simulate 3 cell types with distinct marker genes
    X = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Background expression (Poisson, low rate)
    X += rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Cell type 1 (cells 0-99): genes 0-49 highly expressed
    X[:100, :50] += rng.poisson(30, size=(100, 50)).astype(np.float32)
    # Cell type 2 (cells 100-199): genes 50-99 highly expressed
    X[100:200, 50:100] += rng.poisson(25, size=(100, 50)).astype(np.float32)
    # Cell type 3 (cells 200-299): genes 100-149 highly expressed
    X[200:300, 100:150] += rng.poisson(20, size=(100, 50)).astype(np.float32)

    # Add some cells with very low gene counts (should be filtered)
    X[295:300, :] = 0
    X[295:300, 0] = 1  # just 1 gene detected

    X_sparse = sp.csr_matrix(X)
    adata = ad.AnnData(X=X_sparse)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.uns["organism"] = "Homo sapiens"
    return adata


class TestFullPipeline:
    """Integration test running the complete analysis pipeline."""

    def test_full_pipeline(self, raw_adata):
        """Run the complete pipeline: QC → normalize → HVG → PCA → neighbors → leiden → umap → DE."""
        adata = raw_adata

        # Step 1: Describe
        stats = singlet.describe(adata)
        assert stats["n_cells"] == 300
        assert stats["n_genes"] == 2000
        assert stats["organism"] == "Homo sapiens"
        assert 0 < stats["sparsity"] < 1

        # Step 2: Filter cells
        singlet.filter_cells(adata, min_genes=10, inplace=True)
        assert adata.shape[0] < 300  # some low-quality cells removed
        assert adata.shape[0] >= 290  # but most kept

        # Step 3: Filter genes
        singlet.filter_genes(adata, min_cells=5, inplace=True)
        assert adata.shape[1] <= 2000  # some rarely-detected genes may be removed
        assert adata.shape[1] >= 150  # marker genes kept

        n_cells_post_filter = adata.shape[0]
        n_genes_post_filter = adata.shape[1]

        # Step 4: Normalize
        singlet.normalize(adata)
        assert "raw" in adata.layers
        assert sp.issparse(adata.X)
        assert adata.X.dtype == np.float32
        assert adata.X.max() < 15  # log-transformed

        # Step 5: Highly variable genes
        singlet.highly_variable_genes(adata, n_top_genes=500)
        assert "highly_variable" in adata.var.columns
        n_hvg = adata.var["highly_variable"].sum()
        assert n_hvg == min(500, n_genes_post_filter)

        # Step 6: PCA
        singlet.pca(adata, n_comps=30)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (n_cells_post_filter, 30)
        assert "pca" in adata.uns
        var = adata.uns["pca"]["variance"]
        assert np.all(var[:-1] >= var[1:])

        # Step 7: Neighbors
        singlet.neighbors(adata, n_neighbors=15)
        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp
        assert adata.obsp["connectivities"].shape == (
            n_cells_post_filter,
            n_cells_post_filter,
        )

        # Step 8: Leiden clustering
        singlet.leiden(adata)
        assert "leiden" in adata.obs.columns
        n_clusters = adata.obs["leiden"].nunique()
        assert n_clusters >= 2

        # Step 9: UMAP
        singlet.umap(adata)
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (n_cells_post_filter, 2)
        assert np.all(np.isfinite(adata.obsm["X_umap"]))

        # Step 10: Differential expression
        singlet.rank_genes_groups(adata, "leiden")
        assert "rank_genes_groups" in adata.uns
        rgg = adata.uns["rank_genes_groups"]
        assert len(rgg["names"]) == n_clusters
        first_group = list(rgg["pvals"].keys())[0]
        assert rgg["pvals"][first_group][0] < 0.05

    def test_pipeline_copy_mode(self, raw_adata):
        """Pipeline works in copy (non-inplace) mode."""
        adata = raw_adata.copy()

        # Filter (copy mode)
        filtered = singlet.filter_cells(adata, min_genes=10)
        assert filtered is not adata
        assert filtered.shape[0] < adata.shape[0]

        filtered = singlet.filter_genes(filtered, min_cells=5)

        # Normalize (copy mode)
        normalized = singlet.normalize(filtered, inplace=False)
        assert normalized is not filtered
        assert "raw" in normalized.layers

        # HVG (copy mode returns list)
        hvg_list = singlet.highly_variable_genes(normalized, n_top_genes=200, inplace=False)
        assert isinstance(hvg_list, list)
        assert len(hvg_list) == 200

        # PCA (copy mode)
        singlet.highly_variable_genes(normalized, n_top_genes=200)
        X_pca = singlet.pca(normalized, n_comps=20, inplace=False)
        assert isinstance(X_pca, np.ndarray)
        assert X_pca.shape == (normalized.shape[0], 20)

    def test_pipeline_dense_matrix(self, raw_adata):
        """Pipeline works with dense matrices too."""
        adata = raw_adata.copy()
        adata.X = adata.X.toarray()

        singlet.filter_cells(adata, min_genes=10, inplace=True)
        singlet.filter_genes(adata, min_cells=5, inplace=True)
        singlet.normalize(adata)
        singlet.highly_variable_genes(adata, n_top_genes=300)
        singlet.pca(adata, n_comps=20)
        singlet.neighbors(adata, n_neighbors=10)
        singlet.leiden(adata, n_clusters=3)

        assert adata.obs["leiden"].nunique() == 3

    def test_pipeline_small_dataset(self):
        """Pipeline handles very small datasets gracefully."""
        import anndata as ad

        rng = np.random.default_rng(123)
        X = sp.random(20, 100, density=0.3, format="csr", random_state=123)
        X.data = rng.poisson(5, size=X.nnz).astype(np.float32)

        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{i}" for i in range(100)]
        adata.obs_names = [f"c{i}" for i in range(20)]

        singlet.normalize(adata)
        singlet.highly_variable_genes(adata, n_top_genes=50)
        singlet.pca(adata, n_comps=10)
        singlet.neighbors(adata, n_neighbors=5)
        singlet.leiden(adata, n_clusters=2)
        singlet.umap(adata)

        assert adata.obsm["X_umap"].shape == (20, 2)
        assert adata.obs["leiden"].nunique() == 2
