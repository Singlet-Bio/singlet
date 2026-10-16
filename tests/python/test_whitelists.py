# SPDX-License-Identifier: MIT
"""Whitelists: bare names only, large ones fetched on demand, downloads checked."""

from __future__ import annotations

import gzip
import io
import urllib.error

import pytest
from singlet import whitelists as wl


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(wl, "_PKG_DATA", tmp_path / "bundled")  # nothing bundled
    monkeypatch.setattr(wl, "_CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\b", "", ".hidden", "x" * 80, "a..b", "x y"])
def test_names_with_separators_or_traversal_are_refused(bad, cache):
    with pytest.raises(ValueError):
        wl.ensure_whitelist(bad)
    with pytest.raises(ValueError):
        wl.load_whitelist(bad)


def test_a_missing_large_whitelist_is_downloaded_once_and_checked(cache, monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return _Resp(gzip.compress(b"AAACCTGAGAAACCAT-1\nAAACCTGAGAAACCGC\n"))

    monkeypatch.setattr(wl.urllib.request, "urlopen", fake_urlopen)
    assert wl.load_whitelist("737K-august-2016") == {"AAACCTGAGAAACCAT-1", "AAACCTGAGAAACCGC"}
    assert wl.load_whitelist("737K-august-2016")  # second call reads the cache
    assert len(calls) == 1 and calls[0].endswith("737K-august-2016.txt.gz")


@pytest.mark.parametrize("body", [b"<html>not found</html>\n", b"", b"ACGT\nnot a barcode\n"])
def test_a_download_that_is_not_a_barcode_list_is_rejected_and_not_cached(cache, monkeypatch, body):
    monkeypatch.setattr(
        wl.urllib.request, "urlopen", lambda req, timeout=0: _Resp(gzip.compress(body))
    )
    with pytest.raises(ValueError):
        wl.ensure_whitelist("737K-august-2016")
    assert not any(cache.glob("*.txt")) if cache.exists() else True


def test_a_whitelist_that_is_not_on_the_cdn_must_be_bundled(cache):
    with pytest.raises(FileNotFoundError):
        wl.ensure_whitelist("737K-april-2014")


def test_http_errors_become_file_not_found(cache, monkeypatch):
    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

    monkeypatch.setattr(wl.urllib.request, "urlopen", boom)
    with pytest.raises(FileNotFoundError):
        wl.ensure_whitelist("3M-february-2018")


def test_listing_includes_names_that_download_on_demand(cache):
    names = wl.list_available_whitelists()
    for n in ("3M-february-2018", "737K-august-2016", "737K-cratac-v1", "gex_737K-arc-v1"):
        assert n in names
