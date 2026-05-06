"""Tests for singlet.hotspot_genes()."""

import numpy as np
import pytest

import singlet


@pytest.fixture
def adata_with_hotspots():
    """Create test AnnData with genes that have local hotspots."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(42)
    n_cells = 150
    n_genes = 100

    # Create base expression
    X = rng.poisson(1, size=(n_cells, n_genes)).astype(np.float32)

    # Create a hotspot gene: cells 0-20 have high expression (cluster)
    X[:20, 0] = rng.poisson(15, size=20).astype(np.float32)

    # Create another hotspot gene: cells 50-70 have high expression
    X[50:70, 1] = rng.poisson(12, size=20).astype(np.float32)

    # Gene 2 is uniformly expressed (no hotspot)
    X[:, 2] = rng.poisson(5, size=n_cells).astype(np.float32)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Create PCA where hotspot cells are neighbors
    pca_rep = rng.standard_normal((n_cells, 20))
    # Make cells 0-20 cluster together
    pca_rep[:20, :3] += 5.0
    # Make cells 50-70 cluster together
    pca_rep[50:70, :3] -= 5.0
    adata.obsm["X_pca"] = pca_rep

    return adata


class TestHotspotGenes:
    """Tests for hotspot_genes function."""

    def test_basic_gi_star(self, adata_with_hotspots):
        """Test basic Getis-Ord Gi* computation."""
        import pandas as pd

        adata = adata_with_hotspots
        result = singlet.hotspot_genes(
            adata, n_neighbors=15, method="gi_star"
        )

        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "statistic" in result.columns
        assert "pvalue" in result.columns
        assert "fdr" in result.columns
        assert "n_hotspot_cells" in result.columns
        assert len(result) == adata.n_vars

    def test_basic_local_morans(self, adata_with_hotspots):
        """Test local Moran's I computation."""
        import pandas as pd

        adata = adata_with_hotspots
        result = singlet.hotspot_genes(
            adata, n_neighbors=15, method="local_morans"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == adata.n_vars

    def test_hotspot_gene_detection(self, adata_with_hotspots):
        """Test that genes with true hotspots rank higher."""
        adata = adata_with_hotspots
        result = singlet.hotspot_genes(
            adata, n_neighbors=15, method="gi_star"
        )

        # Gene_0 and Gene_1 should have high statistics (they have hotspots)
        top_genes = result.head(10)["gene"].tolist()
        assert "Gene_0" in top_genes or "Gene_1" in top_genes

    def test_hotspot_cells_counted(self, adata_with_hotspots):
        """Test that hotspot cell counts are reasonable."""
        adata = adata_with_hotspots
        result = singlet.hotspot_genes(
            adata, n_neighbors=15, method="gi_star"
        )

        # Hotspot genes should have some hotspot cells
        gene0_row = result[result["gene"] == "Gene_0"]
        assert gene0_row["n_hotspot_cells"].values[0] > 0

    def test_stored_in_uns(self, adata_with_hotspots):
        """Test results stored in adata.uns."""
        adata = adata_with_hotspots
        result = singlet.hotspot_genes(adata, n_neighbors=15)

        assert "hotspot_genes" in adata.uns
        assert len(adata.uns["hotspot_genes"]) == len(result)

    def test_fdr_threshold(self, adata_with_hotspots):
        """Test that FDR values are between 0 and 1."""
        adata = adata_with_hotspots
        result = singlet.hotspot_genes(adata, n_neighbors=15)

        assert np.all(result["fdr"] >= 0)
        assert np.all(result["fdr"] <= 1)
        assert np.all(result["pvalue"] >= 0)
        assert np.all(result["pvalue"] <= 1)

    def test_gene_subset(self, adata_with_hotspots):
        """Test with specific gene subset."""
        adata = adata_with_hotspots
        selected = ["Gene_0", "Gene_1", "Gene_5"]
        result = singlet.hotspot_genes(
            adata, n_neighbors=15, genes=selected
        )

        assert len(result) == 3
        assert set(result["gene"].tolist()) == set(selected)

    def test_hvg_selection(self, adata_with_hotspots):
        """Test automatic HVG selection."""
        adata = adata_with_hotspots
        # Mark some genes as HVG
        adata.var["highly_variable"] = False
        adata.var.loc[["Gene_0", "Gene_1", "Gene_2", "Gene_3"], "highly_variable"] = (
            True
        )

        result = singlet.hotspot_genes(adata, n_neighbors=15)
        assert len(result) == 4

    def test_invalid_method(self, adata_with_hotspots):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            singlet.hotspot_genes(
                adata_with_hotspots, method="invalid"
            )

    def test_missing_representation(self, adata_with_hotspots):
        """Test error on missing representation."""
        with pytest.raises(KeyError, match="not found in adata.obsm"):
            singlet.hotspot_genes(
                adata_with_hotspots, use_rep="X_nonexistent"
            )

    def test_invalid_n_neighbors(self, adata_with_hotspots):
        """Test error on invalid n_neighbors."""
        with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
            singlet.hotspot_genes(
                adata_with_hotspots, n_neighbors=0
            )

    def test_no_genes_found(self, adata_with_hotspots):
        """Test error when specified genes don't exist."""
        with pytest.raises(ValueError, match="None of the specified genes"):
            singlet.hotspot_genes(
                adata_with_hotspots, genes=["NonexistentGene"]
            )

    def test_sorted_descending(self, adata_with_hotspots):
        """Test that results are sorted by statistic descending."""
        adata = adata_with_hotspots
        result = singlet.hotspot_genes(adata, n_neighbors=15)

        stats = result["statistic"].values
        assert np.all(stats[:-1] >= stats[1:])

    def test_small_dataset(self):
        """Test with very small dataset."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(55)
        X = sp.csr_matrix(rng.poisson(3, size=(20, 10)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G{i}" for i in range(10)]
        adata.obs_names = [f"C{i}" for i in range(20)]
        adata.obsm["X_pca"] = rng.standard_normal((20, 5))

        # Should work with small n_neighbors
        result = singlet.hotspot_genes(adata, n_neighbors=5)
        assert len(result) == 10
