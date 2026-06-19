# SPDX-License-Identifier: MIT
"""Tests for singlet._loader (download, load, load_sample) — unit tests with mocks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


class _FakeResp:
    """Minimal urlopen() context-manager / reader stub."""

    def __init__(self, chunks, content_length=None):
        self._chunks = list(chunks)
        self.headers = {"content-length": str(content_length or 0)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *a):
        return self._chunks.pop(0) if self._chunks else b""


class TestDownload:
    def test_cached_returns_immediately(self, tmp_path):
        from singlet._loader import download

        cached = tmp_path / "GSE123456.singlet"
        cached.write_bytes(b"fake")
        result = download("GSE123456", output_dir=tmp_path)
        assert result == cached

    def test_force_re_downloads(self, tmp_path):
        from singlet import _loader

        cached = tmp_path / "GSE123456.singlet"
        cached.write_bytes(b"old data")

        with patch.object(
            _loader.urllib.request, "urlopen", return_value=_FakeResp([b"new!"])
        ):
            result = _loader.download("GSE123456", output_dir=tmp_path, force=True)

        assert result == cached
        assert cached.read_bytes() == b"new!"

    def test_r2_url_format(self, tmp_path):
        from singlet import _loader

        captured = {}

        def fake_urlopen(req, *a, **k):
            captured["url"] = req.full_url
            return _FakeResp([])

        with patch.object(_loader.urllib.request, "urlopen", side_effect=fake_urlopen):
            _loader.download("GSE999999", output_dir=tmp_path)

        assert captured["url"] == (
            "https://data.singlet.bio/data/GSE999999/GSE999999.singlet"
        )

    def test_data_base_override(self, tmp_path, monkeypatch):
        from singlet import _loader

        monkeypatch.setenv("SINGLET_DATA_BASE", "https://example.test")
        captured = {}

        def fake_urlopen(req, *a, **k):
            captured["url"] = req.full_url
            return _FakeResp([])

        with patch.object(_loader.urllib.request, "urlopen", side_effect=fake_urlopen):
            _loader.download("GSE1", output_dir=tmp_path)

        assert captured["url"] == "https://example.test/data/GSE1/GSE1.singlet"

    def test_interrupted_download_no_corrupt_file(self, tmp_path):
        """Interrupted download does not leave a corrupt bundle file."""
        from singlet import _loader

        def boom(req, *a, **k):
            raise ConnectionError("network dropped")

        with patch.object(_loader.urllib.request, "urlopen", side_effect=boom):
            with pytest.raises(ConnectionError):
                _loader.download("GSE_INTERRUPT", output_dir=tmp_path)

        assert not (tmp_path / "GSE_INTERRUPT.singlet").exists()
        assert not (tmp_path / "GSE_INTERRUPT.singlet.part").exists()


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
