"""Tests for singlet._loader (download, load, load_sample) — unit tests with mocks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


class TestDownload:
    def test_invalid_source_raises(self, tmp_path):
        from singlet._loader import download

        with pytest.raises(ValueError, match="source must be"):
            download("GSE123456", output_dir=tmp_path, source="ftp")

    def test_cached_returns_immediately(self, tmp_path):
        from singlet._loader import download

        cached = tmp_path / "GSE123456.1pz"
        cached.write_bytes(b"fake")
        result = download("GSE123456", output_dir=tmp_path)
        assert result == cached

    def test_force_re_downloads(self, tmp_path):
        from singlet._loader import download

        cached = tmp_path / "GSE123456.1pz"
        cached.write_bytes(b"old data")

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "4"}
        mock_resp.iter_content.return_value = [b"new!"]

        with patch("requests.get", return_value=mock_resp):
            result = download("GSE123456", output_dir=tmp_path, force=True)

        assert result == cached
        assert cached.read_bytes() == b"new!"

    @patch("requests.get")
    def test_zenodo_url_format(self, mock_get, tmp_path):
        from singlet._loader import download

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "0"}
        mock_resp.iter_content.return_value = []
        mock_get.return_value = mock_resp

        download("GSE999999", output_dir=tmp_path, source="zenodo")

        call_args = mock_get.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "GSE999999.1pz" in url

    @patch("requests.get")
    def test_aws_requires_auth_headers(self, mock_get, tmp_path):
        import singlet._auth as auth
        from singlet._loader import download

        auth._API_KEY = "sk-test"
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "0"}
        mock_resp.iter_content.return_value = []
        mock_get.return_value = mock_resp

        download("GSE999999", output_dir=tmp_path, source="aws")

        call_kwargs = mock_get.call_args.kwargs or {}
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers
        auth._API_KEY = None

    @patch("requests.get")
    def test_interrupted_download_no_corrupt_file(self, mock_get, tmp_path):
        """Interrupted download does not leave a corrupt .1pz file."""
        from singlet._loader import download

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "100"}
        mock_resp.iter_content.side_effect = ConnectionError("network dropped")
        mock_get.return_value = mock_resp

        with pytest.raises(ConnectionError):
            download("GSE_INTERRUPT", output_dir=tmp_path, source="zenodo")

        # The .1pz file should NOT exist (only .part was written and cleaned up)
        assert not (tmp_path / "GSE_INTERRUPT.1pz").exists()
        assert not (tmp_path / "GSE_INTERRUPT.1pz.part").exists()


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_local_h5ad(self, tmp_path):
        """load() reads .h5ad files directly."""
        import anndata as ad
        from singlet._loader import load

        adata = ad.AnnData(X=sp.random(5, 10, format="csr", density=0.5))
        adata.var_names = [f"gene_{i}" for i in range(10)]
        adata.obs_names = [f"cell_{i}" for i in range(5)]
        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        result = load(path)
        assert result.shape == (5, 10)

    def test_load_with_gene_subset(self, tmp_path):
        """load() with genes= filters to specific genes."""
        import anndata as ad
        from singlet._loader import load

        adata = ad.AnnData(X=sp.random(5, 10, format="csr", density=0.5))
        adata.var_names = [f"gene_{i}" for i in range(10)]
        adata.obs_names = [f"cell_{i}" for i in range(5)]
        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        result = load(path, genes=["gene_0", "gene_5", "gene_9"])
        assert result.n_vars == 3
        assert set(result.var_names) == {"gene_0", "gene_5", "gene_9"}

    def test_load_with_obs_filter(self, tmp_path):
        """load() with obs_filter= subsets cells."""
        import anndata as ad
        from singlet._loader import load

        X = sp.random(10, 5, format="csr", density=0.5)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(5)]
        adata.obs_names = [f"cell_{i}" for i in range(10)]
        adata.obs["tissue"] = ["lung"] * 5 + ["liver"] * 5

        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        result = load(path, obs_filter={"tissue": "lung"})
        assert result.n_obs == 5
        assert all(result.obs["tissue"] == "lung")

    def test_load_with_obs_filter_list(self, tmp_path):
        """obs_filter with list value uses isin()."""
        import anndata as ad
        from singlet._loader import load

        X = sp.random(9, 3, format="csr", density=0.5)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(3)]
        adata.obs_names = [f"c{i}" for i in range(9)]
        adata.obs["type"] = ["A", "B", "C"] * 3

        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        result = load(path, obs_filter={"type": ["A", "C"]})
        assert result.n_obs == 6

    @patch("singlet._io.read_1pz")
    @patch("singlet._loader._resolve_gse_path")
    def test_load_accession_uses_local_first(self, mock_resolve, mock_read):
        """When local catalog has the file, download is not called."""
        import anndata as ad
        from singlet._loader import load

        # Mock resolve returns a path, mock read_1pz returns AnnData
        mock_resolve.return_value = Path("/fake/local.1pz")
        mock_read.return_value = ad.AnnData(X=sp.random(2, 3, format="csr"))

        with patch("singlet._loader.download") as mock_dl:
            _adata = load("GSE000001")
            mock_dl.assert_not_called()

        mock_read.assert_called_once_with(Path("/fake/local.1pz"))

    def test_load_nonexistent_file_triggers_accession_path(self):
        """Non-existent path treated as accession."""
        from singlet._loader import load

        with patch("singlet._loader._resolve_gse_path", return_value=None):
            with patch("singlet._loader.download") as mock_dl:
                mock_dl.side_effect = Exception("No network")
                with pytest.raises(Exception, match="No network"):
                    load("GSE999999")
                mock_dl.assert_called_once()


# ---------------------------------------------------------------------------
# load_sample()
# ---------------------------------------------------------------------------


class TestLoadSample:
    @patch("singlet._loader._resolve_gse_path")
    def test_load_sample_missing_raises(self, mock_resolve):
        from singlet._loader import load_sample

        mock_resolve.return_value = None
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError, KeyError)):
            load_sample("GSM0000000")


# ---------------------------------------------------------------------------
# _cache_dir()
# ---------------------------------------------------------------------------


def test_cache_dir_exists():
    from singlet._loader import _cache_dir

    d = _cache_dir()
    assert d.exists()
    assert d.is_dir()
    assert ".singlet" in str(d)
