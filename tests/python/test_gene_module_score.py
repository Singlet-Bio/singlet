"""Tests for singlet.gene_module_score() module scoring."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._gene_module_score import gene_module_score


def _make_module_adata(n_cells=100, n_genes=200):
    """Create AnnData with named genes and signal in module genes.

    Genes 0-9 are designated as "module" genes and receive higher
    expression values so that module scoring produces a clear signal.
    """
    import anndata as ad

    rng = np.random.default_rng(42)

    # Background expression (low)
    X = rng.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Boost module genes (indices 0-9) so they have higher mean expression
    X[:, :10] += rng.poisson(lam=5, size=(n_cells, 10)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


class TestGeneModuleScore:
    def test_basic_scoring(self):
        """Basic usage returns a DataFrame with module scores."""
        adata = _make_module_adata()
        modules = {"my_module": [f"gene_{i}" for i in range(10)]}
        result = gene_module_score(adata, modules)
        assert isinstance(result, pd.DataFrame)
        assert "my_module" in result.columns

    def test_stores_in_obs(self):
        """Scores are stored in adata.obs['score_{name}']."""
        adata = _make_module_adata()
        modules = {"cell_cycle": [f"gene_{i}" for i in range(5)]}
        gene_module_score(adata, modules)
        assert "score_cell_cycle" in adata.obs.columns

    def test_output_shape(self):
        """DataFrame shape is (n_cells, n_modules)."""
        adata = _make_module_adata(n_cells=80, n_genes=150)
        modules = {
            "mod_a": [f"gene_{i}" for i in range(5)],
            "mod_b": [f"gene_{i}" for i in range(5, 10)],
        }
        result = gene_module_score(adata, modules)
        assert result.shape == (80, 2)

    def test_multiple_modules(self):
        """Handles multiple modules simultaneously."""
        adata = _make_module_adata(n_cells=50, n_genes=300)
        modules = {
            "module_1": [f"gene_{i}" for i in range(10)],
            "module_2": [f"gene_{i}" for i in range(50, 60)],
            "module_3": [f"gene_{i}" for i in range(100, 110)],
        }
        result = gene_module_score(adata, modules)
        assert list(result.columns) == ["module_1", "module_2", "module_3"]
        assert result.shape == (50, 3)

    def test_missing_genes_warning(self):
        """Warns about missing genes but does not error."""
        adata = _make_module_adata()
        modules = {"mixed": ["gene_0", "gene_1", "NONEXISTENT_A", "NONEXISTENT_B"]}
        with pytest.warns(UserWarning, match="not found"):
            result = gene_module_score(adata, modules)
        assert result.shape == (100, 1)

    def test_all_genes_missing_raises(self):
        """Raises ValueError if all module genes are missing."""
        adata = _make_module_adata()
        modules = {"bad": ["FAKE_GENE_1", "FAKE_GENE_2"]}
        with pytest.raises(ValueError, match="no valid genes"):
            gene_module_score(adata, modules)

    def test_type_error_adata(self):
        """Non-AnnData input raises TypeError."""
        with pytest.raises(TypeError, match="AnnData"):
            gene_module_score("not_adata", {"mod": ["gene_0"]})

    def test_type_error_modules(self):
        """Non-dict gene_modules raises TypeError."""
        adata = _make_module_adata()
        with pytest.raises(TypeError, match="dict"):
            gene_module_score(adata, [["gene_0", "gene_1"]])

    def test_empty_modules_raises(self):
        """Empty dict raises ValueError."""
        adata = _make_module_adata()
        with pytest.raises(ValueError, match="must not be empty"):
            gene_module_score(adata, {})

    def test_reproducible(self):
        """Same random_state gives identical results."""
        adata1 = _make_module_adata()
        adata2 = _make_module_adata()
        modules = {"mod": [f"gene_{i}" for i in range(10)]}
        r1 = gene_module_score(adata1, modules, random_state=7)
        r2 = gene_module_score(adata2, modules, random_state=7)
        pd.testing.assert_frame_equal(r1, r2)

    def test_no_nans(self):
        """Output contains no NaN values."""
        adata = _make_module_adata(n_cells=120, n_genes=400)
        modules = {"mod": [f"gene_{i}" for i in range(20)]}
        result = gene_module_score(adata, modules)
        assert not result.isna().any().any()
        assert np.all(np.isfinite(result.values))

    def test_ctrl_size(self):
        """Different ctrl_size values produce valid results."""
        adata = _make_module_adata()
        modules = {"mod": [f"gene_{i}" for i in range(10)]}
        r1 = gene_module_score(adata, modules, ctrl_size=10)
        r2 = gene_module_score(adata, modules, ctrl_size=100)
        # Both should be valid DataFrames
        assert r1.shape == r2.shape
        assert not r1.isna().any().any()
        assert not r2.isna().any().any()

    def test_sparse_input(self):
        """Works with sparse X matrix."""
        import anndata as ad

        rng = np.random.default_rng(42)
        n_cells, n_genes = 100, 200
        X_sparse = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
        adata = ad.AnnData(X=X_sparse)
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        modules = {"mod": [f"gene_{i}" for i in range(10)]}
        result = gene_module_score(adata, modules)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 1)
        assert not result.isna().any().any()

    def test_public_api(self):
        """gene_module_score is accessible from singlet namespace."""
        assert hasattr(singlet, "gene_module_score")
        assert callable(singlet.gene_module_score)

    def test_stores_uns(self):
        """adata.uns has gene_module_score metadata."""
        adata = _make_module_adata()
        modules = {"alpha": [f"gene_{i}" for i in range(5)]}
        gene_module_score(adata, modules, ctrl_size=30, n_bins=10, random_state=99)
        assert "gene_module_score" in adata.uns
        params = adata.uns["gene_module_score"]["params"]
        assert params["ctrl_size"] == 30
        assert params["n_bins"] == 10
        assert params["random_state"] == 99
        assert adata.uns["gene_module_score"]["modules"] == ["alpha"]

    def test_index_matches_obs_names(self):
        """Returned DataFrame index matches adata.obs_names."""
        adata = _make_module_adata(n_cells=60)
        modules = {"mod": [f"gene_{i}" for i in range(5)]}
        result = gene_module_score(adata, modules)
        assert list(result.index) == list(adata.obs_names)

    def test_module_genes_score_higher(self):
        """Module genes with boosted expression should yield positive scores."""
        adata = _make_module_adata(n_cells=200, n_genes=500)
        # Genes 0-9 have boosted expression in _make_module_adata
        modules = {"boosted": [f"gene_{i}" for i in range(10)]}
        result = gene_module_score(adata, modules)
        # Mean score should be positive since module genes are upregulated
        assert result["boosted"].mean() > 0
