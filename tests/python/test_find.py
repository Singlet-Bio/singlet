# SPDX-License-Identifier: MIT
"""Tests for singlet.find: natural-language search (network mocked)."""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import urllib.error
import urllib.parse
from unittest.mock import patch

import pytest

# `import singlet.find as m` would bind the *function* singlet.find (the
# package re-exports it under the module's name), so fetch the module itself.
find_mod = importlib.import_module("singlet.find")


def _json_resp(payload):
    """A urlopen() stand-in: BytesIO is a context manager with read()."""
    return io.BytesIO(json.dumps(payload).encode())


def _capture(payload):
    """urlopen() side effect that records the request and returns *payload*."""
    seen = {}

    def fake_urlopen(req, *a, **k):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        return _json_resp(payload)

    return seen, fake_urlopen


def _query_params(url: str) -> dict:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv("SINGLET_API_KEY", raising=False)
    monkeypatch.delenv("SINGLET_API_BASE", raising=False)
    find_mod.set_api_key(None)
    yield
    find_mod.set_api_key(None)


class TestFindDefaults:
    def test_default_level_is_gse(self):
        seen, fake = _capture({"accessions": ["GSE138867", "GSE146974"]})
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=fake):
            out = find_mod.find("human PBMC 10x")
        assert out == ["GSE138867", "GSE146974"]
        params = _query_params(seen["url"])
        assert params["level"] == "gse"
        assert params["q"] == "human PBMC 10x"
        assert seen["url"].startswith("https://singlet.bio/api/nl-search?")

    def test_gsm_level_is_passed_through(self):
        seen, fake = _capture({"accessions": ["GSM4120733"]})
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=fake):
            out = find_mod.find("pbmc", level="gsm", limit=5)
        assert out == ["GSM4120733"]
        assert _query_params(seen["url"])["level"] == "gsm"
        assert _query_params(seen["url"])["limit"] == "5"

    def test_bad_level_is_rejected(self):
        with pytest.raises(ValueError, match="level"):
            find_mod.find("pbmc", level="study")

    def test_empty_query_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            find_mod.find("   ")

    def test_limit_truncates_and_results_shape_is_tolerated(self):
        _, fake = _capture({"results": ["GSE1", "GSE2", "GSE3"]})
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=fake):
            assert find_mod.find("x", limit=2) == ["GSE1", "GSE2"]

    def test_api_base_override(self, monkeypatch):
        monkeypatch.setenv("SINGLET_API_BASE", "https://example.test/api/")
        seen, fake = _capture({"accessions": []})
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=fake):
            assert find_mod.find("x") == []
        assert seen["url"].startswith("https://example.test/api/nl-search?")

    def test_api_key_is_sent_as_bearer(self):
        find_mod.set_api_key("  sk-test  ")
        seen, fake = _capture({"accessions": []})
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=fake):
            find_mod.find("x")
        assert seen["headers"].get("Authorization") == "Bearer sk-test"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SINGLET_API_KEY", "sk-env")
        assert find_mod._api_key() == "sk-env"


class TestFindErrors:
    @staticmethod
    def _http_error(code, body=b""):
        return urllib.error.HTTPError(
            "https://singlet.bio/api/nl-search", code, "err", {}, io.BytesIO(body)
        )

    def test_rate_limit_mentions_api_key(self):
        err = self._http_error(429, b'{"message": "daily limit"}')
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(find_mod.SingletSearchError, match="daily limit.*API key"):
                find_mod.find("x")

    def test_rejected_key(self):
        err = self._http_error(401, b'{"error": "bad key"}')
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(find_mod.SingletSearchError, match="rejected.*bad key"):
                find_mod.find("x")

    def test_other_http_error(self):
        err = self._http_error(500, b"not json")
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(find_mod.SingletSearchError, match="HTTP 500"):
                find_mod.find("x")

    def test_unreachable(self):
        err = urllib.error.URLError("no route")
        with patch.object(find_mod.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(find_mod.SingletSearchError, match="Could not reach"):
                find_mod.find("x")

    def test_invalid_json(self):
        with patch.object(find_mod.urllib.request, "urlopen", return_value=io.BytesIO(b"<html>")):
            with pytest.raises(find_mod.SingletSearchError, match="invalid response"):
                find_mod.find("x")


class TestFindLoad:
    def test_defaults_to_top_three_studies(self):
        with patch.object(find_mod, "find", return_value=["GSE1", "GSE2", "GSE3"]) as f:
            with patch("singlet._loader.load", return_value="ADATA") as load:
                out = find_mod.find_load("pbmc", genes=["CD3E"])
        f.assert_called_once_with("pbmc", level="gse", limit=3)
        load.assert_called_once_with(["GSE1", "GSE2", "GSE3"], genes=["CD3E"])
        assert out == "ADATA"

    def test_no_match_raises(self):
        with patch.object(find_mod, "find", return_value=[]):
            with pytest.raises(LookupError, match="No datasets matched"):
                find_mod.find_load("nothing")


class TestOffline:
    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_truthy(self, monkeypatch, value):
        monkeypatch.setenv("SINGLET_OFFLINE", value)
        assert find_mod._offline()

    @pytest.mark.parametrize("value", ["", "0", "false"])
    def test_falsy(self, monkeypatch, value):
        monkeypatch.setenv("SINGLET_OFFLINE", value)
        assert not find_mod._offline()


def _versions(**installed):
    """importlib.metadata.version stand-in that knows only *installed*."""

    def fake(dist):
        if dist in installed:
            return installed[dist]
        raise importlib.metadata.PackageNotFoundError(dist)

    return fake


class TestPackageVersion:
    """The distribution is "singlet-bio"; "singlet" is the pre-rename name."""

    def test_prefers_the_singlet_bio_distribution(self):
        fake = _versions(**{"singlet-bio": "9.9.9", "singlet": "1.0.0"})
        with patch("importlib.metadata.version", side_effect=fake):
            assert find_mod._package_version() == "9.9.9"
            assert find_mod._user_agent() == "singlet-python/9.9.9"

    def test_falls_back_to_the_pre_rename_distribution(self):
        with patch("importlib.metadata.version", side_effect=_versions(singlet="1.2.3")):
            assert find_mod._package_version() == "1.2.3"

    def test_source_checkout_uses_the_package_version(self):
        import singlet

        with patch("importlib.metadata.version", side_effect=_versions()):
            assert find_mod._package_version() == singlet.__version__
