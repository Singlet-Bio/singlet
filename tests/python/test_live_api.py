# SPDX-License-Identifier: MIT
"""Tests for the live-API fallbacks: singlet.info() and python -m singlet.

The offline catalog snapshot lags the live catalog, so info() falls back to
GET /api/gse/<id> or /api/gsm/<id>, and ``python -m singlet`` prints live
numbers from /api/stats. Network calls are mocked.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pandas as pd
import pytest
import singlet._catalog as cat_mod
from singlet import __main__ as main_mod


def _resp(payload):
    return io.BytesIO(json.dumps(payload).encode())


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    monkeypatch.delenv("SINGLET_OFFLINE", raising=False)
    monkeypatch.delenv("SINGLET_API_BASE", raising=False)
    monkeypatch.setattr(cat_mod, "_CATALOG_CACHE", pd.DataFrame({"gse_id": ["GSE1"]}))
    monkeypatch.setattr(cat_mod, "_SAMPLE_INDEX_CACHE", pd.DataFrame({"gsm_id": ["GSM1"]}))


class TestLiveInfo:
    def test_snapshot_hit_does_not_touch_the_network(self):
        with patch("urllib.request.urlopen") as urlopen:
            assert cat_mod.info("gse1")["gse_id"] == "GSE1"
        urlopen.assert_not_called()

    def test_gse_falls_back_to_live_api(self):
        payload = {
            "series": {"id": "GSE138867", "title": "PBMC", "n_cells": 46741},
            "meta": {"has_bundle": 1},
            "samples": [{"gsm_id": "GSM4120733"}],
            "conditions": [],
        }
        seen = {}

        def fake(req, *a, **k):
            seen["url"] = req.full_url
            return _resp(payload)

        with patch("urllib.request.urlopen", side_effect=fake):
            rec = cat_mod.info("GSE138867")
        assert seen["url"] == "https://singlet.bio/api/gse/GSE138867"
        assert rec["source"] == "live"
        assert rec["gse_id"] == "GSE138867"
        assert rec["n_cells"] == 46741
        assert rec["samples"][0]["gsm_id"] == "GSM4120733"
        assert "conditions" not in rec

    def test_gsm_falls_back_to_live_api(self):
        payload = {
            "sample": {"gsm_id": "GSM4120733", "gse_id": "GSE138867", "n_cells": 5000},
            "series": {"id": "GSE138867"},
            "siblings": [],
        }
        with patch("urllib.request.urlopen", return_value=_resp(payload)) as urlopen:
            rec = cat_mod.info("GSM4120733")
        assert urlopen.call_args[0][0].full_url.endswith("/api/gsm/GSM4120733")
        assert rec["gse_id"] == "GSE138867"
        assert rec["series"] == {"id": "GSE138867"}
        assert rec["source"] == "live"

    @pytest.mark.parametrize(
        "error",
        [
            urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b"")),
            urllib.error.URLError("offline"),
            TimeoutError("slow"),
        ],
    )
    def test_unknown_or_unreachable_raises_keyerror(self, error):
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(KeyError, match="GSE999999"):
                cat_mod.info("GSE999999")

    def test_unexpected_payload_shape(self):
        with patch("urllib.request.urlopen", return_value=_resp({"error": "nope"})):
            assert cat_mod._live_info("GSE999999") is None
        with patch("urllib.request.urlopen", return_value=_resp(["not", "a", "dict"])):
            assert cat_mod._live_info("GSE999999") is None

    def test_offline_env_disables_the_fallback(self, monkeypatch):
        monkeypatch.setenv("SINGLET_OFFLINE", "1")
        with patch("urllib.request.urlopen") as urlopen:
            assert cat_mod._live_info("GSE138867") is None
        urlopen.assert_not_called()


class TestLiveStats:
    STATS = {"studies_with_files": 7841, "samples_in_files": 57748, "cells_in_files": 250666906}

    def test_main_prints_live_numbers(self, capsys):
        with patch("urllib.request.urlopen", return_value=_resp(self.STATS)) as urlopen:
            main_mod.main()
        assert urlopen.call_args[0][0].full_url == "https://singlet.bio/api/stats"
        out = capsys.readouterr().out
        assert "live" in out
        assert "7,841 studies" in out
        assert "250,666,906 cells" in out

    def test_main_falls_back_to_the_snapshot(self, capsys, monkeypatch):
        monkeypatch.setattr(main_mod.singlet, "summary", lambda: "singlet atlas: snapshot")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            main_mod.main()
        out = capsys.readouterr().out
        assert "offline" in out
        assert "singlet atlas: snapshot" in out

    def test_missing_fields_are_shown_as_unknown(self):
        assert "? studies" in main_mod._format_live({})

    def test_non_dict_payload_is_ignored(self):
        with patch("urllib.request.urlopen", return_value=_resp([1, 2])):
            assert main_mod._live_stats() is None
