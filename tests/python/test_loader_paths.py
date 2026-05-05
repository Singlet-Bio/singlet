"""Tests for singlet._loader: _resolve_gse_path, download, load, load_sample."""

from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import requests
import scipy.sparse as sp


@pytest.fixture(autouse=True)
def reset_catalog(monkeypatch):
    """Reset catalog state before each test."""
    import singlet._catalog as cat_mod

    monkeypatch.setattr(cat_mod, "_CATALOG_CACHE", None)
    monkeypatch.setattr(cat_mod, "_SAMPLE_INDEX_CACHE", None)
    monkeypatch.setattr(cat_mod, "_CATALOG_DIR", None)
    monkeypatch.delenv("SINGLET_CATALOG_DIR", raising=False)


class TestResolveGsePath:
    """Test _resolve_gse_path catalog path resolution."""

    def test_returns_none_no_catalog_dir(self, monkeypatch):
        """Returns None when no catalog dir set."""
        from singlet._loader import _resolve_gse_path

        assert _resolve_gse_path("GSE999999") is None

    def test_returns_none_accession_not_found(self, tmp_path, monkeypatch):
        """Returns None if accession not in catalog."""
        import singlet._catalog as cat_mod
        from singlet._loader import _resolve_gse_path

        df = pd.DataFrame({"gse_id": ["GSE001"], "path": ["pipeline/quant/GSE001"]})
        df.to_parquet(tmp_path / "catalog_v1.parquet")
        cat_mod.set_catalog_dir(tmp_path)

        assert _resolve_gse_path("GSE999") is None

    def test_returns_path_when_counts_exists(self, tmp_path, monkeypatch):
        """Returns path to counts.1pz when it exists."""
        import singlet._catalog as cat_mod
        from singlet._loader import _resolve_gse_path

        # Setup: catalog dir is tmp_path/catalog/ → parent is tmp_path/
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        df = pd.DataFrame({"gse_id": ["GSE100"], "path": ["pipeline/quant/GSE100"]})
        df.to_parquet(catalog_dir / "catalog_v1.parquet")
        cat_mod.set_catalog_dir(catalog_dir)

        # Create the counts file
        counts_dir = tmp_path / "pipeline" / "quant" / "GSE100"
        counts_dir.mkdir(parents=True)
        counts_file = counts_dir / "counts.1pz"
        counts_file.write_bytes(b"dummy")

        result = _resolve_gse_path("GSE100")
        assert result == counts_file

    def test_returns_none_empty_path_field(self, tmp_path, monkeypatch):
        """Returns None if path field is empty string."""
        import singlet._catalog as cat_mod
        from singlet._loader import _resolve_gse_path

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        df = pd.DataFrame({"gse_id": ["GSE100"], "path": [""]})
        df.to_parquet(catalog_dir / "catalog_v1.parquet")
        cat_mod.set_catalog_dir(catalog_dir)

        assert _resolve_gse_path("GSE100") is None

    def test_returns_none_no_counts_file(self, tmp_path, monkeypatch):
        """Returns None when path exists but counts.1pz is missing."""
        import singlet._catalog as cat_mod
        from singlet._loader import _resolve_gse_path

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        df = pd.DataFrame({"gse_id": ["GSE100"], "path": ["pipeline/quant/GSE100"]})
        df.to_parquet(catalog_dir / "catalog_v1.parquet")
        cat_mod.set_catalog_dir(catalog_dir)

        # Create directory but NOT counts.1pz
        (tmp_path / "pipeline" / "quant" / "GSE100").mkdir(parents=True)

        assert _resolve_gse_path("GSE100") is None


class TestDownload:
    """Test download() function."""

    def test_invalid_source_raises(self):
        """Raises ValueError for invalid source."""
        from singlet._loader import download

        with pytest.raises(ValueError, match="source must be"):
            download("GSE001", source="invalid")

    def test_returns_cached_if_exists(self, tmp_path):
        """Returns existing file without downloading if not force."""
        from singlet._loader import download

        dest = tmp_path / "GSE001.1pz"
        dest.write_bytes(b"cached data")

        result = download("GSE001", output_dir=tmp_path)
        assert result == dest

    def test_force_redownloads(self, tmp_path, monkeypatch):
        """force=True triggers download even if file exists."""
        from singlet._loader import download

        dest = tmp_path / "GSE001.1pz"
        dest.write_bytes(b"old data")

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "4"}
        mock_resp.iter_content.return_value = [b"new!"]
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            with patch(
                "tqdm.tqdm",
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(update=MagicMock())),
                    __exit__=MagicMock(return_value=False),
                ),
            ):
                result = download("GSE001", output_dir=tmp_path, force=True)

        assert result == dest
        assert dest.read_bytes() == b"new!"

    def test_404_raises_file_not_found(self, tmp_path):
        """404 response gives clear FileNotFoundError with guidance."""
        from singlet._loader import download

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(FileNotFoundError, match="not found on zenodo"):
                download("GSE_FAKE", output_dir=tmp_path, force=True)

    def test_500_raises_runtime_error(self, tmp_path):
        """Non-404 HTTP errors give RuntimeError with context."""
        from singlet._loader import download

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error", response=mock_resp
        )

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Failed to download"):
                download("GSE_FAKE", output_dir=tmp_path, force=True)


class TestLoad:
    """Test load() with local file paths."""

    @pytest.fixture
    def h5ad_file(self, tmp_path):
        """Create a test h5ad file."""
        mat = sp.random(5, 10, density=0.4, format="csr", dtype=np.float32)
        adata = ad.AnnData(X=mat)
        adata.var_names = pd.Index([f"GENE{i}" for i in range(10)])
        adata.obs_names = pd.Index([f"CELL{i}" for i in range(5)])
        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)
        return path

    def test_load_h5ad(self, h5ad_file):
        """Loads .h5ad file directly."""
        from singlet._loader import load

        adata = load(h5ad_file)
        assert adata.shape == (5, 10)

    def test_load_1pz_file(self, tmp_path):
        """Loads .1pz file directly."""
        from singlet._io import write_1pz
        from singlet._loader import load

        mat = sp.random(8, 12, density=0.3, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 100).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.var_names = pd.Index([f"G{i:02d}" for i in range(12)])
        adata.obs_names = pd.Index([f"C{i:02d}" for i in range(8)])
        path = tmp_path / "counts.1pz"
        write_1pz(adata, path)

        loaded = load(path)
        assert loaded.shape == (8, 12)

    def test_load_with_gene_filter(self, h5ad_file):
        """Gene filter subsets columns."""
        from singlet._loader import load

        adata = load(h5ad_file, genes=["GENE0", "GENE5"])
        assert adata.shape[1] == 2
        assert set(adata.var_names) == {"GENE0", "GENE5"}

    def test_load_with_obs_filter(self, tmp_path):
        """obs_filter subsets rows."""
        from singlet._loader import load

        mat = sp.random(10, 5, density=0.5, format="csr", dtype=np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs["celltype"] = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        loaded = load(path, obs_filter={"celltype": "A"})
        assert loaded.shape[0] == 5

    def test_load_with_obs_filter_list(self, tmp_path):
        """obs_filter with list does isin check."""
        from singlet._loader import load

        mat = sp.random(6, 3, density=0.5, format="csr", dtype=np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs["tissue"] = ["lung", "brain", "lung", "heart", "brain", "lung"]
        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        loaded = load(path, obs_filter={"tissue": ["lung", "brain"]})
        assert loaded.shape[0] == 5

    def test_load_gse_from_catalog(self, tmp_path, monkeypatch):
        """Resolves GSE accession from local catalog."""
        import singlet._catalog as cat_mod
        from singlet._io import write_1pz
        from singlet._loader import load

        # Setup catalog
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        df = pd.DataFrame({"gse_id": ["GSE100"], "path": ["pipeline/quant/GSE100"]})
        df.to_parquet(catalog_dir / "catalog_v1.parquet")
        cat_mod.set_catalog_dir(catalog_dir)

        # Create counts file
        counts_dir = tmp_path / "pipeline" / "quant" / "GSE100"
        counts_dir.mkdir(parents=True)
        mat = sp.random(4, 6, density=0.5, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 100).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.var_names = pd.Index([f"G{i}" for i in range(6)])
        adata.obs_names = pd.Index([f"C{i}" for i in range(4)])
        write_1pz(adata, counts_dir / "counts.1pz")

        loaded = load("GSE100")
        assert loaded.shape == (4, 6)


class TestLoadSample:
    """Test load_sample() with mocked catalog."""

    def test_missing_sample_raises(self, tmp_path, monkeypatch):
        """Raises KeyError for unknown GSM."""
        import singlet._catalog as cat_mod
        from singlet._loader import load_sample

        idx_df = pd.DataFrame({"gsm_id": ["GSM001"], "gse_id": ["GSE001"]})
        idx_df.to_parquet(tmp_path / "sample_index.parquet")
        cat_mod.set_catalog_dir(tmp_path)

        with pytest.raises(KeyError, match="GSM999"):
            load_sample("GSM999")

    def test_no_catalog_dir_raises(self, tmp_path, monkeypatch):
        """Raises RuntimeError if no catalog dir set."""
        import singlet._catalog as cat_mod
        from singlet._loader import load_sample

        # Set catalog dir just to load the index, then unset
        idx_df = pd.DataFrame(
            {
                "gsm_id": ["GSM001"],
                "gse_id": ["GSE001"],
                "organism": ["human"],
                "n_cells": [100],
                "species_subdir": [""],
                "col_offset": [0],
                "col_count": [100],
            }
        )
        idx_df.to_parquet(tmp_path / "sample_index.parquet")
        cat_mod.set_catalog_dir(tmp_path)
        # Load index into cache, then clear catalog dir
        cat_mod._load_sample_index()
        cat_mod._CATALOG_DIR = None
        monkeypatch.delenv("SINGLET_CATALOG_DIR", raising=False)

        with pytest.raises(RuntimeError, match="load_sample requires a local catalog"):
            load_sample("GSM001")

    def test_load_gse_download_fallback(self, tmp_path, monkeypatch):
        """Falls back to download when GSE not in local catalog."""
        import singlet._catalog as cat_mod
        from singlet._io import write_1pz
        from singlet._loader import load

        # No catalog dir → resolve returns None → triggers download
        cat_mod._CATALOG_DIR = None

        # Create a .1pz to be "downloaded"
        mat = sp.random(3, 4, density=0.5, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 100).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.var_names = pd.Index([f"G{i}" for i in range(4)])
        adata.obs_names = pd.Index([f"C{i}" for i in range(3)])
        download_path = tmp_path / "GSE999.1pz"
        write_1pz(adata, download_path)

        # Mock download to return our file
        with patch("singlet._loader.download", return_value=download_path):
            loaded = load("GSE999")

        assert loaded.shape == (3, 4)

    def test_load_zarr(self, tmp_path):
        """Loads .zarr path via anndata.read_zarr."""
        zarr = pytest.importorskip("zarr")  # noqa: F841
        from singlet._loader import load

        adata = ad.AnnData(X=sp.random(4, 6, density=0.3, format="csr"))
        adata.obs_names = pd.Index([f"C{i}" for i in range(4)])
        adata.var_names = pd.Index([f"G{i}" for i in range(6)])
        zarr_path = tmp_path / "data.zarr"
        adata.write_zarr(zarr_path)

        loaded = load(zarr_path)
        assert loaded.shape == (4, 6)

    def test_load_unknown_extension_falls_through(self, tmp_path):
        """Unknown extension falls through to read_matrix."""
        from singlet._io import write_spz
        from singlet._loader import load

        mat = sp.random(5, 7, density=0.3, format="csr", dtype=np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i}" for i in range(5)])
        adata.var_names = pd.Index([f"G{i}" for i in range(7)])
        # Write as .spz but give it a weird extension
        real_path = tmp_path / "data.spz"
        write_spz(adata, real_path)
        weird_path = tmp_path / "data.blob"
        weird_path.write_bytes(real_path.read_bytes())

        loaded = load(weird_path)
        assert loaded.shape == (5, 7)

    def test_load_sample_success(self, tmp_path, monkeypatch):
        """load_sample reads sample via singlepress.read_1pz_columns."""
        from unittest.mock import MagicMock

        import singlet._catalog as cat_mod
        from singlet._loader import load_sample

        # Setup sample index
        idx_df = pd.DataFrame(
            {
                "gsm_id": ["GSM001"],
                "gse_id": ["GSE001"],
                "organism": ["human"],
                "species_subdir": [""],
                "col_offset": [10],
                "col_count": [5],
            }
        )
        idx_df.to_parquet(tmp_path / "sample_index.parquet")
        cat_mod.set_catalog_dir(tmp_path)

        # Create mock matrix returned by singlepress
        mock_mat = sp.random(3, 5, density=0.5, format="csc", dtype=np.float32)
        mock_mat.rownames = ["G1", "G2", "G3"]
        mock_mat.colnames = None

        mock_sp = MagicMock()
        mock_sp.read_1pz_columns.return_value = mock_mat
        monkeypatch.setitem(__import__("sys").modules, "singlepress", mock_sp)

        adata = load_sample("GSM001")
        assert adata.shape == (5, 3)  # transposed: 5 cells × 3 genes
        assert list(adata.var_names) == ["G1", "G2", "G3"]
        assert adata.obs["gsm_id"].iloc[0] == "GSM001"
        assert adata.obs["gse_id"].iloc[0] == "GSE001"

    def test_load_sample_with_gene_filter(self, tmp_path, monkeypatch):
        """load_sample filters to requested genes."""
        from unittest.mock import MagicMock

        import singlet._catalog as cat_mod
        from singlet._loader import load_sample

        idx_df = pd.DataFrame(
            {
                "gsm_id": ["GSM002"],
                "gse_id": ["GSE002"],
                "organism": ["mouse"],
                "species_subdir": ["mus"],
                "col_offset": [0],
                "col_count": [3],
            }
        )
        idx_df.to_parquet(tmp_path / "sample_index.parquet")
        cat_mod.set_catalog_dir(tmp_path)

        mock_mat = sp.random(4, 3, density=0.5, format="csc", dtype=np.float32)
        mock_mat.rownames = ["GeneA", "GeneB", "GeneC", "GeneD"]
        mock_mat.colnames = None

        mock_sp = MagicMock()
        mock_sp.read_1pz_columns.return_value = mock_mat
        monkeypatch.setitem(__import__("sys").modules, "singlepress", mock_sp)

        adata = load_sample("GSM002", genes=["GeneA", "GeneC"])
        assert adata.shape == (3, 2)  # 3 cells × 2 genes
        assert set(adata.var_names) == {"GeneA", "GeneC"}
