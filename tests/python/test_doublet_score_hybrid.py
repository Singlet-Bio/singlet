"""Tests for singlet.doublet_score_hybrid()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._doublet_score_hybrid import doublet_score_hybrid


def _make_doublet_adata(n_cells=200, n_genes=100):
    """Create test AnnData with some synthetic doublets."""
    import anndata as ad

    rng = np.random.default_rng(42)

    # Normal singlets: moderate expression
    n_singlets = int(n_cells * 0.9)
    n_doublets = n_cells - n_singlets

    singlet_expr = rng.poisson(3, (n_singlets, n_genes)).astype(np.float32)

    # Doublets: higher library size and complexity (sum of two singlets)
    idx1 = rng.integers(0, n_singlets, size=n_doublets)
    idx2 = rng.integers(0, n_singlets, size=n_doublets)
    doublet_expr = (singlet_expr[idx1] + singlet_expr[idx2]).astype(np.float32)

    X = np.vstack([singlet_expr, doublet_expr])
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

    # Ground truth labels
    adata.obs["is_doublet"] = [False] * n_singlets + [True] * n_doublets

    return adata


class TestDoubletScoreHybrid:
    def test_basic(self):
        """Basic hybrid doublet score computation."""
        adata = _make_doublet_adata()
        result = doublet_score_hybrid(adata)

        assert result is adata
        assert "doublet_score_hybrid" in adata.obs.columns
        assert "predicted_doublet_hybrid" in adata.obs.columns

    def test_scores_in_range(self):
        """Doublet scores should be in [0, 1]."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata)

        scores = adata.obs["doublet_score_hybrid"].values
        assert (scores >= 0).all()
        assert (scores <= 1.0 + 1e-6).all()

    def test_predictions_boolean(self):
        """Predictions should be boolean."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata)

        preds = adata.obs["predicted_doublet_hybrid"]
        assert preds.dtype == bool

    def test_expected_doublet_rate(self):
        """Predicted doublet fraction should approximate expected_doublet_rate."""
        adata = _make_doublet_adata(n_cells=500)
        rate = 0.1
        doublet_score_hybrid(adata, expected_doublet_rate=rate)

        predicted_rate = adata.obs["predicted_doublet_hybrid"].mean()
        # Should be approximately the expected rate (threshold-based)
        assert abs(predicted_rate - rate) < 0.05

    def test_all_methods(self):
        """Default (all methods) should work."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata)
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_simulation_only(self):
        """Using only simulation method should work."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata, methods=["simulation"])
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_library_size_only(self):
        """Using only library_size method should work."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata, methods=["library_size"])
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_complexity_only(self):
        """Using only complexity method should work."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata, methods=["complexity"])
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_two_methods(self):
        """Using two methods should work."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata, methods=["simulation", "library_size"])
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_doublets_score_higher(self):
        """Synthetic doublets should generally score higher than singlets."""
        adata = _make_doublet_adata(n_cells=300, n_genes=200)
        doublet_score_hybrid(adata)

        singlet_scores = adata.obs.loc[~adata.obs["is_doublet"], "doublet_score_hybrid"]
        doublet_scores = adata.obs.loc[adata.obs["is_doublet"], "doublet_score_hybrid"]

        # Doublets should have higher mean score
        assert doublet_scores.mean() > singlet_scores.mean()

    def test_reproducible(self):
        """Same random_state should give same results."""
        adata1 = _make_doublet_adata()
        adata2 = _make_doublet_adata()

        doublet_score_hybrid(adata1, random_state=42)
        doublet_score_hybrid(adata2, random_state=42)

        np.testing.assert_array_almost_equal(
            adata1.obs["doublet_score_hybrid"].values,
            adata2.obs["doublet_score_hybrid"].values,
        )

    def test_different_seeds(self):
        """Different random states should give different results."""
        adata1 = _make_doublet_adata()
        adata2 = _make_doublet_adata()

        doublet_score_hybrid(adata1, random_state=0)
        doublet_score_hybrid(adata2, random_state=99)

        # Scores should not be identical (very unlikely with different seeds)
        scores1 = adata1.obs["doublet_score_hybrid"].values
        scores2 = adata2.obs["doublet_score_hybrid"].values
        assert not np.allclose(scores1, scores2)

    def test_n_neighbors(self):
        """Custom n_neighbors should work."""
        adata = _make_doublet_adata()
        doublet_score_hybrid(adata, n_neighbors=10)
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_sparse_input(self):
        """Should work with sparse expression matrix."""
        adata = _make_doublet_adata()
        assert sp.issparse(adata.X)
        doublet_score_hybrid(adata)
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_dense_input(self):
        """Should work with dense expression matrix."""
        adata = _make_doublet_adata()
        adata.X = adata.X.toarray()
        doublet_score_hybrid(adata)
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_invalid_method_raises(self):
        """Should raise ValueError on invalid method."""
        adata = _make_doublet_adata()
        with pytest.raises(ValueError, match="Invalid method"):
            doublet_score_hybrid(adata, methods=["invalid_method"])

    def test_invalid_n_neighbors_raises(self):
        """Should raise ValueError on n_neighbors < 1."""
        adata = _make_doublet_adata()
        with pytest.raises(ValueError, match="n_neighbors"):
            doublet_score_hybrid(adata, n_neighbors=0)

    def test_invalid_rate_raises(self):
        """Should raise ValueError on invalid expected_doublet_rate."""
        adata = _make_doublet_adata()
        with pytest.raises(ValueError, match="expected_doublet_rate"):
            doublet_score_hybrid(adata, expected_doublet_rate=0.0)
        with pytest.raises(ValueError, match="expected_doublet_rate"):
            doublet_score_hybrid(adata, expected_doublet_rate=1.0)

    def test_type_error(self):
        """Should raise TypeError on non-AnnData input."""
        with pytest.raises(TypeError, match="doublet_score_hybrid"):
            doublet_score_hybrid("not_adata")

    def test_small_dataset(self):
        """Should work with very small datasets."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X = rng.poisson(5, (20, 30)).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        adata.var_names = [f"G{idx}" for idx in range(30)]
        adata.obs_names = [f"c{idx}" for idx in range(20)]

        doublet_score_hybrid(adata, n_neighbors=5)
        assert "doublet_score_hybrid" in adata.obs.columns

    def test_public_api(self):
        """Should be accessible via singlet.doublet_score_hybrid."""
        assert hasattr(singlet, "doublet_score_hybrid")
        assert callable(singlet.doublet_score_hybrid)
