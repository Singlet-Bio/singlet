"""Tests for singlet._annotate (gene_programs, project, annotate)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._annotate import _detect_organism, _models_dir, annotate, project

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_adata():
    """Create a small AnnData with 10 cells × 200 genes."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.random(10, 200, density=0.3, format="csr", random_state=rng)
    X.data = np.abs(X.data) * 100  # ensure non-negative counts
    gene_names = [f"GENE{i}" for i in range(200)]
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    adata.uns["organism"] = "Homo sapiens"
    return adata


@pytest.fixture
def mock_W():
    """A fake W matrix (200 genes × 5 programs)."""
    rng = np.random.default_rng(7)
    W = rng.random((200, 5)).astype(np.float64)
    gene_names = [f"GENE{i}" for i in range(200)]
    program_names = [f"P{i + 1:03d}" for i in range(5)]
    return pd.DataFrame(W, index=gene_names, columns=program_names)


@pytest.fixture
def mock_labels():
    """Fake program → cell type label mapping."""
    return {
        "P001": "T cell",
        "P002": "B cell",
        "P003": "Monocyte",
        "P004": "NK cell",
        "P005": "Fibroblast",
    }


# ---------------------------------------------------------------------------
# _detect_organism
# ---------------------------------------------------------------------------


class TestDetectOrganism:
    def test_from_uns(self, small_adata):
        assert _detect_organism(small_adata) == "Homo sapiens"

    def test_from_obs_column(self, small_adata):
        del small_adata.uns["organism"]
        small_adata.obs["organism"] = "Mus musculus"
        assert _detect_organism(small_adata) == "Mus musculus"

    def test_from_species_column(self, small_adata):
        del small_adata.uns["organism"]
        small_adata.obs["species"] = "Danio rerio"
        assert _detect_organism(small_adata) == "Danio rerio"

    def test_raises_when_missing(self, small_adata):
        del small_adata.uns["organism"]
        with pytest.raises(ValueError, match="Could not detect organism"):
            _detect_organism(small_adata)


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


class TestProject:
    @patch("singlet._annotate.gene_programs")
    def test_basic_shape(self, mock_gp, small_adata, mock_W):
        mock_gp.return_value = mock_W
        H = project(small_adata, organism="Homo sapiens", k=5)
        assert H.shape == (10, 5)
        assert H.dtype == np.float64

    @patch("singlet._annotate.gene_programs")
    def test_non_negative(self, mock_gp, small_adata, mock_W):
        mock_gp.return_value = mock_W
        H = project(small_adata, organism="Homo sapiens", k=5)
        assert np.all(H >= 0)

    @patch("singlet._annotate.gene_programs")
    def test_auto_detect_organism(self, mock_gp, small_adata, mock_W):
        mock_gp.return_value = mock_W
        H = project(small_adata, k=5)  # organism auto-detected from uns
        mock_gp.assert_called_once_with("Homo sapiens", 5)
        assert H.shape == (10, 5)

    @patch("singlet._annotate.gene_programs")
    def test_too_few_genes_raises(self, mock_gp, small_adata):
        # W has only 50 genes, none overlap with adata's 200
        W_small = pd.DataFrame(
            np.random.rand(50, 5),
            index=[f"OTHER{i}" for i in range(50)],
            columns=[f"P{i + 1:03d}" for i in range(5)],
        )
        mock_gp.return_value = W_small
        with pytest.raises(ValueError, match="genes overlap"):
            project(small_adata, organism="Homo sapiens", k=5)

    @patch("singlet._annotate.gene_programs")
    def test_dense_input(self, mock_gp, mock_W):
        """Works with dense X matrix."""
        import anndata as ad

        mock_gp.return_value = mock_W
        X_dense = np.random.rand(5, 200).astype(np.float32)
        adata = ad.AnnData(X=X_dense)
        adata.var_names = [f"GENE{i}" for i in range(200)]
        adata.obs_names = [f"cell_{i}" for i in range(5)]
        adata.uns["organism"] = "Homo sapiens"
        H = project(adata, k=5)
        assert H.shape == (5, 5)
        assert np.all(H >= 0)


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


class TestAnnotate:
    @patch("singlet._annotate._load_program_labels")
    @patch("singlet._annotate.gene_programs")
    def test_basic_output(self, mock_gp, mock_labels_fn, small_adata, mock_W, mock_labels):
        mock_gp.return_value = mock_W
        mock_labels_fn.return_value = mock_labels

        result = annotate(small_adata, organism="Homo sapiens", k=5)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 10
        assert set(result.columns) == {
            "cell_type",
            "confidence",
            "top_program",
            "top_program_loading",
        }

    @patch("singlet._annotate._load_program_labels")
    @patch("singlet._annotate.gene_programs")
    def test_confidence_range(self, mock_gp, mock_labels_fn, small_adata, mock_W, mock_labels):
        mock_gp.return_value = mock_W
        mock_labels_fn.return_value = mock_labels

        result = annotate(small_adata, organism="Homo sapiens", k=5)
        assert (result["confidence"] >= 0).all()
        assert (result["confidence"] <= 1).all()

    @patch("singlet._annotate._load_program_labels")
    @patch("singlet._annotate.gene_programs")
    def test_cell_types_from_labels(
        self, mock_gp, mock_labels_fn, small_adata, mock_W, mock_labels
    ):
        mock_gp.return_value = mock_W
        mock_labels_fn.return_value = mock_labels

        result = annotate(small_adata, organism="Homo sapiens", k=5)
        # All cell types should come from the mock labels
        valid_types = set(mock_labels.values())
        for ct in result["cell_type"]:
            assert ct in valid_types

    @patch("singlet._annotate._load_program_labels")
    @patch("singlet._annotate.gene_programs")
    def test_missing_labels_unknown(self, mock_gp, mock_labels_fn, small_adata, mock_W):
        mock_gp.return_value = mock_W
        mock_labels_fn.return_value = {}  # No labels → all "unknown"

        result = annotate(small_adata, organism="Homo sapiens", k=5)
        assert (result["cell_type"] == "unknown").all()

    @patch("singlet._annotate._load_program_labels")
    @patch("singlet._annotate.gene_programs")
    def test_index_matches_adata(self, mock_gp, mock_labels_fn, small_adata, mock_W, mock_labels):
        mock_gp.return_value = mock_W
        mock_labels_fn.return_value = mock_labels

        result = annotate(small_adata, organism="Homo sapiens", k=5)
        assert list(result.index) == list(small_adata.obs_names)


# ---------------------------------------------------------------------------
# _models_dir
# ---------------------------------------------------------------------------


def test_models_dir_exists():
    d = _models_dir()
    assert d.exists()
    assert d.is_dir()
