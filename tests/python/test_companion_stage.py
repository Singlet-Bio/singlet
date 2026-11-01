# SPDX-License-Identifier: MIT
"""Companion stage: a companion that failed, is unsupported or raises is recorded in status.json and never raises."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import companion_stage as CS  # noqa: E402


def _claim(tmp_path, organism, companions):
    claim = tmp_path / "claim"
    claim.mkdir()
    (claim / "manifest.json").write_text(json.dumps({"organism": organism, "companions": companions}))
    return str(claim)


def test_failed_missing_and_raising_companions_are_recorded(tmp_path, monkeypatch):
    comp = [
        {"lib_type": "adt", "lib_gsm": "L1", "runs": ["R1"], "reads": 5, "status": "ok"},
        {"lib_type": "crispr", "lib_gsm": "L2", "runs": ["R2"], "reads": 9, "status": "failed", "reason": "all_sources_failed"},
        {"lib_type": "vdj", "lib_gsm": "L3", "runs": ["R3"], "reads": 2_000_000, "status": "ok"},
    ]
    out = tmp_path / "out"
    out.mkdir()
    st = CS.run_stage(str(out), _claim(tmp_path, "Homo sapiens", comp), "/nowhere", log=lambda *a: None)
    by = {s["lib_type"]: s for s in st}
    assert by["adt"]["status"] == "missing" and "not implemented" in by["adt"]["reason"]
    assert by["crispr"]["status"] == "failed" and by["crispr"]["reason"] == "fetch failed"
    assert by["vdj"]["status"] == "failed" and "no GEX columns" in by["vdj"]["reason"]   # exon_counts.1pz absent
    assert by["vdj"]["reads"] == 2_000_000 and by["vdj"]["wall_s"] >= 0
    assert json.load(open(out / "companions" / "status.json")) == st


def test_unsupported_organism_is_missing(tmp_path):
    comp = [{"lib_type": "vdj", "lib_gsm": "L3", "runs": ["R3"], "reads": 1, "status": "ok"}]
    out = tmp_path / "out"
    out.mkdir()
    st = CS.run_stage(str(out), _claim(tmp_path, "Danio rerio", comp), "/nowhere", log=lambda *a: None)
    assert st[0]["status"] == "missing" and "Danio rerio" in st[0]["reason"]
