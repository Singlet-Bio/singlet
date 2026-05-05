"""Tests for catalog internal paths: _download_parquet, refresh, failure inference."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch):
    """Clear catalog caches between tests."""
    import singlet._catalog as cat_mod

    monkeypatch.setattr(cat_mod, "_CATALOG_CACHE", None)
    monkeypatch.setattr(cat_mod, "_SAMPLE_INDEX_CACHE", None)
    monkeypatch.setattr(cat_mod, "_CATALOG_DIR", None)
    monkeypatch.delenv("SINGLET_CATALOG_DIR", raising=False)


class TestDownloadParquet:
    """Test _download_parquet downloads and caches."""

    def test_downloads_and_saves(self, tmp_path):
        """Downloads parquet content and saves to cache path."""
        import singlet._catalog as cat_mod

        df = pd.DataFrame({"gse_id": ["GSE001"], "organism": ["human"]})
        content = df.to_parquet()

        mock_resp = MagicMock()
        mock_resp.content = content
        mock_resp.raise_for_status = MagicMock()

        cache_path = tmp_path / "subdir" / "catalog.parquet"
        with patch("requests.get", return_value=mock_resp):
            result = cat_mod._download_parquet("http://example.com/f.parquet", cache_path)

        assert cache_path.exists()
        assert len(result) == 1
        assert result.iloc[0]["gse_id"] == "GSE001"


class TestLoadCatalogFallback:
    """Test _load_catalog download fallback path."""

    def test_falls_back_to_download(self, tmp_path, monkeypatch):
        """Downloads catalog when no local or bundled file available."""
        import singlet._catalog as cat_mod

        # Patch bundled path to not exist
        monkeypatch.setattr(cat_mod, "_get_catalog_dir", lambda: None)

        df = pd.DataFrame({"gse_id": ["GSE002"]})
        monkeypatch.setattr(cat_mod, "_download_parquet", lambda url, path: df)

        # Patch Path.exists for both bundled and cache checks
        orig_exists = Path.exists

        def fake_exists(self):
            if "catalog_v1" in str(self) or "sample_index" in str(self):
                return False
            return orig_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)

        result = cat_mod._load_catalog()
        assert len(result) == 1

    def test_raises_on_download_failure(self, tmp_path, monkeypatch):
        """Raises RuntimeError when download fails."""
        import singlet._catalog as cat_mod

        monkeypatch.setattr(cat_mod, "_get_catalog_dir", lambda: None)

        def fail_download(url, path):
            raise Exception("network error")

        monkeypatch.setattr(cat_mod, "_download_parquet", fail_download)

        orig_exists = Path.exists

        def fake_exists(self):
            if "catalog_v1" in str(self) or "sample_index" in str(self):
                return False
            return orig_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)

        with pytest.raises(RuntimeError, match="Could not load catalog"):
            cat_mod._load_catalog()


class TestRefresh:
    """Test refresh() function."""

    def test_refresh_clears_and_redownloads(self, tmp_path, monkeypatch):
        """refresh() clears cache and downloads fresh data."""
        import singlet._catalog as cat_mod

        cat_df = pd.DataFrame({"gse_id": ["GSE999"]})
        idx_df = pd.DataFrame({"gsm_id": ["GSM999"]})

        downloads = []

        def mock_download(url, path):
            downloads.append(url)
            if "catalog_v1" in url:
                return cat_df
            return idx_df

        monkeypatch.setattr(cat_mod, "_download_parquet", mock_download)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        cat_mod.refresh()
        assert len(downloads) == 2
        assert cat_mod._CATALOG_CACHE is cat_df
        assert cat_mod._SAMPLE_INDEX_CACHE is idx_df


class TestFailureCategoriesInference:
    """Test failure_categories with no failure_category column."""

    def test_infers_from_metrics(self, monkeypatch):
        """Infers failure categories from mapping_rate and cells_called."""
        import singlet._catalog as cat_mod

        idx = pd.DataFrame(
            {
                "gsm_id": ["A", "B", "C", "D", "E"],
                "gse_id": ["G"] * 5,
                "status": ["HARD_FAIL", "FAIL", "FAIL", "FAIL", "FAIL"],
                "mapping_rate": [0.0, 0.0, 0.05, 0.3, 0.5],
                "cells_called": [0, 0, 10, 20, 200],
            }
        )
        # No failure_category column → triggers inference
        monkeypatch.setattr(cat_mod, "_load_sample_index", lambda: idx)

        result = cat_mod.failure_categories()
        assert len(result) > 0
        categories = set(result["category"])
        # Should include some of: download_fail, pipeline_crash, align_low_map, cells_below_threshold
        assert len(categories) >= 2
        assert result["count"].sum() == 5

    def test_empty_failures(self, monkeypatch):
        """Returns empty DataFrame if no failures exist."""
        import singlet._catalog as cat_mod

        idx = pd.DataFrame({"gsm_id": ["A"], "gse_id": ["G"], "status": ["SUCCESS"]})
        monkeypatch.setattr(cat_mod, "_load_sample_index", lambda: idx)

        result = cat_mod.failure_categories()
        assert len(result) == 0
        assert "category" in result.columns


class TestSummaryHelper:
    """Test summary() _fmt helper and edge cases."""

    def test_summary_with_no_tissue_column(self, monkeypatch):
        """summary() handles missing tissue column gracefully."""
        import singlet._catalog as cat_mod

        idx = pd.DataFrame(
            {
                "gsm_id": ["A", "B"],
                "gse_id": ["G1", "G2"],
                "status": ["SUCCESS", "SUCCESS"],
                "cells_called": [1000, 2000],
                "organism": ["human", "human"],
                "protocol": ["10xv3", "10xv2"],
            }
        )
        monkeypatch.setattr(cat_mod, "_load_sample_index", lambda: idx)
        monkeypatch.setattr(cat_mod, "species", lambda: ["human"])

        result = cat_mod.summary()
        assert "2 samples" in result
        assert "2 series" in result

    def test_fmt_millions(self, monkeypatch):
        """summary() formats large cell counts with M suffix."""
        import singlet._catalog as cat_mod

        idx = pd.DataFrame(
            {
                "gsm_id": [f"G{i}" for i in range(10)],
                "gse_id": [f"S{i}" for i in range(10)],
                "status": ["SUCCESS"] * 10,
                "cells_called": [500_000] * 10,  # 5M total
                "organism": ["human"] * 10,
                "protocol": ["10xv3"] * 10,
            }
        )
        monkeypatch.setattr(cat_mod, "_load_sample_index", lambda: idx)
        monkeypatch.setattr(cat_mod, "species", lambda: ["human"])

        result = cat_mod.summary()
        assert "5.0M" in result

    def test_fmt_thousands(self, monkeypatch):
        """summary() formats mid-range cell counts with K suffix."""
        import singlet._catalog as cat_mod

        idx = pd.DataFrame(
            {
                "gsm_id": ["A"],
                "gse_id": ["G1"],
                "status": ["SUCCESS"],
                "cells_called": [5000],
                "organism": ["human"],
                "protocol": ["10xv3"],
            }
        )
        monkeypatch.setattr(cat_mod, "_load_sample_index", lambda: idx)
        monkeypatch.setattr(cat_mod, "species", lambda: ["human"])

        result = cat_mod.summary()
        assert "5.0K" in result
