"""Tests for singlet.cell_level_de()."""

import numpy as np
import pytest

import singlet


@pytest.fixture
def adata_two_conditions():
    """Create test AnnData with two conditions and treatment effect."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 150

    # Base expression
    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)

    # Create treatment effect: first 20 genes upregulated in condition B
    # for a subset of cells (heterogeneous response)
    # Condition B cells (50-99) have upregulated genes 0-19
    X[50:80, :20] += rng.poisson(8, size=(30, 20)).astype(np.float32)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Assign conditions
    adata.obs["treatment"] = ["control"] * 50 + ["treated"] * 50

    # PCA representation
    pca_rep = rng.standard_normal((n_cells, 15))
    adata.obsm["X_pca"] = pca_rep

    return adata


class TestCellLevelDE:
    """Tests for cell_level_de function."""

    def test_basic_output(self, adata_two_conditions):
        """Test basic cell-level DE computation."""
        adata = adata_two_conditions
        result = singlet.cell_level_de(
            adata, condition_key="treatment", n_top_genes=30
        )

        # Should return adata
        assert result is adata

        # Check layers
        assert "cell_level_lfc" in adata.layers
        assert adata.layers["cell_level_lfc"].shape == (100, 150)

        # Check uns
        assert "cell_level_de_genes" in adata.uns
        assert len(adata.uns["cell_level_de_genes"]) == 30
        assert "cell_level_de_params" in adata.uns

    def test_lfc_values(self, adata_two_conditions):
        """Test that LFC values are reasonable."""
        import scipy.sparse as sp

        adata = adata_two_conditions
        singlet.cell_level_de(
            adata, condition_key="treatment", n_top_genes=30
        )

        lfc = adata.layers["cell_level_lfc"]
        if sp.issparse(lfc):
            lfc_dense = lfc.toarray()
        else:
            lfc_dense = np.asarray(lfc)

        # LFC should have finite values
        assert np.all(np.isfinite(lfc_dense))

        # Some cells should have non-zero LFC
        assert np.any(lfc_dense != 0)

    def test_params_stored(self, adata_two_conditions):
        """Test parameters stored correctly."""
        adata = adata_two_conditions
        singlet.cell_level_de(
            adata,
            condition_key="treatment",
            n_neighbors=20,
            n_top_genes=25,
        )

        params = adata.uns["cell_level_de_params"]
        assert params["condition_key"] == "treatment"
        assert params["n_neighbors"] == 20
        assert params["n_top_genes"] == 25
        assert params["use_rep"] == "X_pca"
        assert len(params["conditions"]) == 2

    def test_gene_names(self, adata_two_conditions):
        """Test that gene names are valid."""
        adata = adata_two_conditions
        singlet.cell_level_de(
            adata, condition_key="treatment", n_top_genes=20
        )

        gene_list = adata.uns["cell_level_de_genes"]
        assert all(g in adata.var_names for g in gene_list)

    def test_n_top_genes_limits(self, adata_two_conditions):
        """Test n_top_genes larger than total genes."""
        adata = adata_two_conditions
        singlet.cell_level_de(
            adata, condition_key="treatment", n_top_genes=500
        )

        # Should cap at actual number of genes
        assert len(adata.uns["cell_level_de_genes"]) == 150

    def test_invalid_condition_key(self, adata_two_conditions):
        """Test error on missing condition_key."""
        with pytest.raises(KeyError, match="not found in adata.obs"):
            singlet.cell_level_de(
                adata_two_conditions, condition_key="nonexistent"
            )

    def test_invalid_representation(self, adata_two_conditions):
        """Test error on missing representation."""
        with pytest.raises(KeyError, match="not found in adata.obsm"):
            singlet.cell_level_de(
                adata_two_conditions,
                condition_key="treatment",
                use_rep="X_nonexistent",
            )

    def test_more_than_two_conditions(self):
        """Test error when more than two conditions."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(77)
        X = sp.csr_matrix(rng.poisson(2, size=(60, 50)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["cond"] = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
        adata.obsm["X_pca"] = rng.standard_normal((60, 10))

        with pytest.raises(ValueError, match="exactly 2 unique values"):
            singlet.cell_level_de(adata, condition_key="cond")

    def test_single_condition(self):
        """Test error when only one condition."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(88)
        X = sp.csr_matrix(rng.poisson(2, size=(40, 50)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["cond"] = ["A"] * 40
        adata.obsm["X_pca"] = rng.standard_normal((40, 10))

        with pytest.raises(ValueError, match="exactly 2 unique values"):
            singlet.cell_level_de(adata, condition_key="cond")

    def test_invalid_n_neighbors(self):
        """Test error on invalid n_neighbors."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(11)
        X = sp.csr_matrix(rng.poisson(2, size=(40, 50)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["cond"] = ["A"] * 20 + ["B"] * 20
        adata.obsm["X_pca"] = rng.standard_normal((40, 10))

        with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
            singlet.cell_level_de(adata, condition_key="cond", n_neighbors=0)

    def test_invalid_n_top_genes(self):
        """Test error on invalid n_top_genes."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(22)
        X = sp.csr_matrix(rng.poisson(2, size=(40, 50)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["cond"] = ["A"] * 20 + ["B"] * 20
        adata.obsm["X_pca"] = rng.standard_normal((40, 10))

        with pytest.raises(ValueError, match="n_top_genes must be >= 1"):
            singlet.cell_level_de(
                adata, condition_key="cond", n_top_genes=0
            )

    def test_sparse_output(self, adata_two_conditions):
        """Test that output layer is sparse."""
        import scipy.sparse as sp

        adata = adata_two_conditions
        singlet.cell_level_de(
            adata, condition_key="treatment", n_top_genes=20
        )

        assert sp.issparse(adata.layers["cell_level_lfc"])

    def test_small_neighborhood(self):
        """Test with very small neighborhood."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(33)
        X = sp.csr_matrix(rng.poisson(3, size=(50, 30)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["cond"] = ["A"] * 25 + ["B"] * 25
        adata.obsm["X_pca"] = rng.standard_normal((50, 8))

        result = singlet.cell_level_de(
            adata, condition_key="cond", n_neighbors=5, n_top_genes=10
        )
        assert result is adata
        assert "cell_level_lfc" in adata.layers

    def test_dense_input(self):
        """Test with dense expression matrix."""
        import anndata as ad

        rng = np.random.default_rng(44)
        X = rng.poisson(3, size=(60, 40)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obs["group"] = ["ctrl"] * 30 + ["stim"] * 30
        adata.obsm["X_pca"] = rng.standard_normal((60, 10))

        singlet.cell_level_de(
            adata, condition_key="group", n_top_genes=15
        )
        assert "cell_level_lfc" in adata.layers
        assert len(adata.uns["cell_level_de_genes"]) == 15
