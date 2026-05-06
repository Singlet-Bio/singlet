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

    def test_raises_when_missing(self):
        import anndata as ad

        # Use ambiguous gene names that won't trigger the capitalization heuristic
        mixed_genes = [f"gene{i}" for i in range(200)]  # all lowercase → no match
        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((3, 200))),
            var=pd.DataFrame(index=mixed_genes),
        )
        with pytest.raises(ValueError, match="Could not detect organism"):
            _detect_organism(adata)


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


# ---------------------------------------------------------------------------
# _download_model
# ---------------------------------------------------------------------------


class TestDownloadModel:
    def test_returns_cached_path(self, tmp_path, monkeypatch):
        """Returns local path if model file already exists."""
        from singlet._annotate import _download_model

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)
        cached = tmp_path / "homo_sapiens_k100.1pz"
        cached.write_bytes(b"cached model")

        result = _download_model("Homo sapiens", 100)
        assert result == cached

    def test_downloads_from_r2(self, tmp_path, monkeypatch):
        """Downloads W matrix from R2 when not cached."""
        from unittest.mock import MagicMock

        from singlet._annotate import _download_model

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)

        mock_resp = MagicMock()
        mock_resp.content = b"model data bytes"
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = _download_model("Mus musculus", 50)

        assert result == tmp_path / "mus_musculus_k50.1pz"
        assert result.read_bytes() == b"model data bytes"
        assert "gene_programs/mus_musculus_k50.1pz" in mock_get.call_args[0][0]

    def test_connection_error_gives_guidance(self, tmp_path, monkeypatch):
        """ConnectionError gives clear guidance about network."""
        import requests
        from singlet._annotate import _download_model

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)

        with patch("requests.get", side_effect=requests.ConnectionError("no network")):
            with pytest.raises(RuntimeError, match="Cannot reach model server"):
                _download_model("Homo sapiens", 100)


# ---------------------------------------------------------------------------
# gene_programs
# ---------------------------------------------------------------------------


class TestGenePrograms:
    def test_returns_dataframe(self, tmp_path, monkeypatch, small_adata):
        """gene_programs returns a DataFrame."""
        from singlet._annotate import gene_programs
        from singlet._io import write_1pz

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)

        # Use equal dims to avoid shape mismatch from the internal name swap
        import anndata as ad

        k = 10
        # After read_1pz: shape = (k, k) so the name assignment doesn't matter
        W = np.random.default_rng(0).random((k, k)).astype(np.float32)
        model_adata = ad.AnnData(X=sp.csr_matrix(W))
        model_adata.obs_names = pd.Index([f"G{i}" for i in range(k)])
        model_adata.var_names = pd.Index([f"P{i + 1:03d}" for i in range(k)])
        write_1pz(model_adata, tmp_path / "homo_sapiens_k10.1pz")

        result = gene_programs("Homo sapiens", k=10)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (k, k)


# ---------------------------------------------------------------------------
# _load_program_labels
# ---------------------------------------------------------------------------


class TestLoadProgramLabels:
    def test_returns_cached_labels(self, tmp_path, monkeypatch):
        """Returns labels from local cache if file exists."""
        import json

        from singlet._annotate import _load_program_labels

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)
        labels = {"P001": "T cell", "P002": "B cell"}
        (tmp_path / "homo_sapiens_k100_labels.json").write_text(json.dumps(labels))

        result = _load_program_labels("Homo sapiens", 100)
        assert result == labels

    def test_downloads_labels(self, tmp_path, monkeypatch):
        """Downloads labels from R2 when not cached."""
        from unittest.mock import MagicMock

        from singlet._annotate import _load_program_labels

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)

        labels = {"P001": "Macrophage", "P002": "Neuron"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = labels
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = _load_program_labels("Homo sapiens", 100)

        assert result == labels
        # Should be cached locally
        cached = tmp_path / "homo_sapiens_k100_labels.json"
        assert cached.exists()

    def test_returns_empty_on_network_error(self, tmp_path, monkeypatch):
        """Returns empty dict if download fails."""
        from singlet._annotate import _load_program_labels

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)

        with patch("requests.get", side_effect=Exception("timeout")):
            result = _load_program_labels("Homo sapiens", 100)

        assert result == {}


class TestGeneProgramsNoNames:
    """Test gene_programs fallback when model has no obs/var names."""

    def test_generates_default_names(self, tmp_path, monkeypatch):
        """gene_programs generates default names when model has empty names."""
        from singlet._annotate import gene_programs

        monkeypatch.setattr("singlet._annotate._models_dir", lambda: tmp_path)

        k = 5
        W = np.random.default_rng(42).random((k, k)).astype(np.float32)

        class FakeAdata:
            X = sp.csr_matrix(W)
            obs_names = pd.Index([])
            var_names = pd.Index([])

        with patch("singlet._io.read_1pz", return_value=FakeAdata()):
            with patch(
                "singlet._annotate._download_model",
                return_value=tmp_path / "model.1pz",
            ):
                result = gene_programs("Homo sapiens", k=k)

        assert result.shape == (k, k)
        assert list(result.columns) == [f"P{i + 1:03d}" for i in range(k)]
        assert list(result.index) == [f"gene_{i}" for i in range(k)]


class TestAnnotateDetectOrganism:
    """Test annotate() auto-detecting organism via _detect_organism."""

    @patch("singlet._annotate._load_program_labels")
    @patch("singlet._annotate.project")
    def test_annotate_without_organism(self, mock_proj, mock_labels_fn, small_adata):
        """annotate() calls _detect_organism when organism is None."""
        k = 5
        H = np.random.default_rng(0).random((10, k)).astype(np.float32)
        mock_proj.return_value = H
        mock_labels_fn.return_value = {f"P{i + 1:03d}": f"type_{i}" for i in range(k)}

        result = annotate(small_adata, k=5)  # no organism → triggers _detect_organism
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 10
        # Verify project was called with detected organism
        mock_proj.assert_called_once()
        call_kwargs = mock_proj.call_args
        assert call_kwargs[1]["organism"] is not None  # organism was detected


class TestDetectOrganismFromGenes:
    """Test _detect_organism gene-name heuristic."""

    def test_human_genes_detected(self):
        """ALL-CAPS gene names → Homo sapiens."""
        import anndata as ad

        human_genes = [f"GENE{i}" for i in range(200)]
        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((3, 200))),
            var=pd.DataFrame(index=human_genes),
        )
        assert _detect_organism(adata) == "Homo sapiens"

    def test_mouse_genes_detected(self):
        """Title-case gene names → Mus musculus."""
        import anndata as ad

        mouse_genes = [f"Gene{i}" for i in range(200)]
        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((3, 200))),
            var=pd.DataFrame(index=mouse_genes),
        )
        assert _detect_organism(adata) == "Mus musculus"

    def test_obs_organism_takes_priority(self):
        """obs['organism'] takes priority over gene heuristic."""
        import anndata as ad

        human_genes = [f"GENE{i}" for i in range(200)]
        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((3, 200))),
            obs=pd.DataFrame({"organism": ["Mus musculus"] * 3}, index=["c0", "c1", "c2"]),
            var=pd.DataFrame(index=human_genes),
        )
        assert _detect_organism(adata) == "Mus musculus"

    def test_too_few_genes_raises(self):
        """Too few gene names → ValueError (can't infer)."""
        import anndata as ad

        adata = ad.AnnData(
            X=sp.csr_matrix(np.ones((3, 10))),
            var=pd.DataFrame(index=[f"GENE{i}" for i in range(10)]),
        )
        with pytest.raises(ValueError, match="Could not detect organism"):
            _detect_organism(adata)


class TestProjectAnnotateTypeValidation:
    """project() and annotate() reject non-AnnData inputs."""

    def test_project_rejects_non_adata(self):
        with pytest.raises(TypeError, match="project\\(\\) requires an AnnData"):
            project("not_an_adata", organism="Homo sapiens")

    def test_annotate_rejects_non_adata(self):
        from singlet._annotate import annotate

        with pytest.raises(TypeError, match="annotate\\(\\) requires an AnnData"):
            annotate({"X": [1, 2, 3]}, organism="Homo sapiens")
