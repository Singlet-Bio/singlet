"""Tests for singlet.score_genes()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._score import score_genes


def _make_adata(n_cells=100, n_genes=200):
    """Create AnnData with some 'marker' genes highly expressed in a subset."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Make genes 0-4 highly expressed in cells 0-49
    X[:50, :5] += 10.0

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    return adata


class TestScoreGenes:
    def test_basic_inplace(self):
        """Should store score in adata.obs."""
        adata = _make_adata()
        ret = score_genes(adata, ["GENE0", "GENE1", "GENE2"])
        assert ret is None
        assert "score" in adata.obs.columns

    def test_custom_score_name(self):
        """Should use custom score_name."""
        adata = _make_adata()
        score_genes(adata, ["GENE0"], score_name="my_score")
        assert "my_score" in adata.obs.columns

    def test_not_inplace(self):
        """inplace=False should return array."""
        adata = _make_adata()
        result = score_genes(adata, ["GENE0", "GENE1"], inplace=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (adata.n_obs,)
        assert "score" not in adata.obs.columns

    def test_high_score_for_expressing_cells(self):
        """Cells expressing the gene set highly should have higher scores."""
        adata = _make_adata()
        scores = score_genes(adata, ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"], inplace=False)
        # Cells 0-49 express these genes highly
        mean_high = scores[:50].mean()
        mean_low = scores[50:].mean()
        assert mean_high > mean_low

    def test_missing_genes_filtered(self):
        """Should work when some genes are missing (filters to valid ones)."""
        adata = _make_adata()
        scores = score_genes(adata, ["GENE0", "NOTEXIST1", "NOTEXIST2"], inplace=False)
        assert scores.shape == (adata.n_obs,)

    def test_all_missing_raises(self):
        """Should raise ValueError when no genes are found."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="None of the genes"):
            score_genes(adata, ["FAKE1", "FAKE2"])

    def test_empty_gene_list_raises(self):
        """Should raise ValueError for empty gene list."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="must not be empty"):
            score_genes(adata, [])

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="score_genes"):
            score_genes("not_adata", ["GENE0"])

    def test_dense_input(self):
        """Should work with dense matrix."""
        adata = _make_adata()
        adata.X = adata.X.toarray()
        scores = score_genes(adata, ["GENE0", "GENE1"], inplace=False)
        assert scores.shape == (adata.n_obs,)

    def test_layer(self):
        """Should use specified layer."""
        adata = _make_adata()
        adata.layers["raw"] = adata.X.copy()
        # Zero out X
        adata.X = sp.csr_matrix(adata.X.shape)
        scores_x = score_genes(adata, ["GENE0"], inplace=False)
        scores_layer = score_genes(adata, ["GENE0"], layer="raw", inplace=False)
        # Scores from layer should be higher (X is zeroed)
        assert scores_layer.mean() > scores_x.mean()

    def test_score_is_finite(self):
        """Scores should not contain NaN or Inf."""
        adata = _make_adata()
        scores = score_genes(adata, ["GENE0", "GENE1", "GENE2"], inplace=False)
        assert np.all(np.isfinite(scores))

    def test_public_api(self):
        assert hasattr(singlet, "score_genes")
        assert callable(singlet.score_genes)
