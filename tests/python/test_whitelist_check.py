"""Second whitelist guard (scripts/atlas_rerun/whitelist_check.py): the engine's log lines and pileup_stats.json decide whether a sample ran
with its barcode whitelist, and drain_worker.finish_phase fails a sample that did not, before anything is published."""

import json
import os
import subprocess
import sys
import threading
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "..", "scripts", "atlas_rerun")
sys.path.insert(0, SCRIPTS)

import drain_worker as D  # noqa: E402
import whitelist_check as W  # noqa: E402

RESOLVED = "[singlet] Auto-resolved whitelist: 10x-3p-v3 → 3M-february-2018.txt\n"
NOT_FOUND = "[singlet] Whitelist 3M-february-2018.txt not found for 10x-3p-v3 — using barcode auto-discovery\n"
NO_HEADER = "[singlet] WARNING: cannot read .1fq header for whitelist auto-resolution — using barcode auto-discovery\n"
TENX_AUTO = "[singlet] Protocol 10x-5p-v2 (id=4) uses barcode auto-discovery — using barcode auto-discovery\n"
WL_FREE = "[singlet] Protocol drop-seq (id=9) is whitelist-free — using barcode auto-discovery\n"
CB_WARN = ("[singlet] WARNING: the protocol is a barcode-whitelist type (CB_UMI_Simple) but no whitelist file was found; the pileup takes "
           "the sorted-BAM path\n")


def sample(tmp_path, log, missing=False, stats=True):
    """-> (log path, out dir) of a fake finished sample."""
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "singlet.log").write_text("[singlet] start\n" + log + "[singlet] done\n", encoding="utf-8")
    if stats is not None:
        body = {"provenance": {}, "pileup_threads": 1}
        if stats:
            body["whitelist_missing"] = missing
        (out / "pileup_stats.json").write_text(json.dumps(body))
    return str(tmp_path / "singlet.log"), str(out)


def test_a_resolved_10x_whitelist_passes(tmp_path):
    r = W.check(*sample(tmp_path, RESOLVED))
    assert r["ok"] is True and "3M-february-2018.txt" in r["resolved"]


def test_a_whitelist_free_protocol_passes(tmp_path):
    r = W.check(*sample(tmp_path, WL_FREE))
    assert r["ok"] is True and "whitelist-free" in r["why"]


@pytest.mark.parametrize(
    "line, why",
    [
        (NOT_FOUND, "without its whitelist file"),
        (NO_HEADER, ".1fq header"),
        (TENX_AUTO, "10x protocol used barcode auto-discovery"),
        (CB_WARN, "CB_UMI_Simple"),
    ],
)
def test_log_lines_that_fail(tmp_path, line, why):
    r = W.check(*sample(tmp_path, line))
    assert r["ok"] is False and why in r["why"] and r["line"]


def test_whitelist_missing_true_fails(tmp_path):
    r = W.check(*sample(tmp_path, RESOLVED, missing=True))
    assert r["ok"] is False and "whitelist_missing = true" in r["why"]


def test_no_whitelist_missing_key_fails(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert W.check(*sample(a, RESOLVED, stats=False))["ok"] is False  # pileup_stats.json without the key
    assert W.check(*sample(b, RESOLVED, stats=None))["ok"] is False  # no pileup_stats.json at all


def test_unreadable_log(tmp_path):
    r = W.check(str(tmp_path / "absent.log"), str(tmp_path))
    assert r["ok"] is None


def test_cli_exit_codes(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    cli = [sys.executable, os.path.join(SCRIPTS, "whitelist_check.py")]
    assert subprocess.run(cli + list(sample(ok, RESOLVED)), capture_output=True).returncode == 0
    assert subprocess.run(cli + list(sample(bad, NOT_FOUND, missing=True)), capture_output=True).returncode == 1
    assert subprocess.run(cli + [str(tmp_path / "absent.log"), str(tmp_path)], capture_output=True).returncode == 2


def test_worker_check_bypass(tmp_path, monkeypatch):
    lp, od = sample(tmp_path, NOT_FOUND)
    monkeypatch.setenv("S9_NO_WL_CHECK", "1")
    assert W.worker_check(lp, od) is None
    monkeypatch.delenv("S9_NO_WL_CHECK")
    assert W.worker_check(lp, od)["ok"] is False


def finish(tmp_path, monkeypatch, log, missing):
    """Runs drain_worker.finish_phase on a fake finished sample. -> (result, events, published, released)."""
    monkeypatch.delenv("S9_NO_WL_CHECK", raising=False)
    scr = tmp_path / "scr"
    scr.mkdir()
    sample(scr, log, missing=missing)
    claim = tmp_path / "claim"
    claim.mkdir()
    events, published, released = [], [], []
    monkeypatch.setattr(D.C, "emit", lambda root, ev, gsm, **kw: events.append(dict(kw, event=ev, gsm=gsm)))
    monkeypatch.setattr(D, "publish", lambda *a, **kw: published.append(a) or "none")
    monkeypatch.setattr(D.Q, "release", lambda stage, gsm, claim, failed=False: released.append((gsm, failed)))
    monkeypatch.setattr(D.Q, "finish", lambda stage, gsm, claim: None)
    monkeypatch.setattr(D.C, "species_check", lambda p: None)
    a = types.SimpleNamespace(no_private_sidecar=True, stage=str(tmp_path / "stage"), engine="gpu")
    job = types.SimpleNamespace(gsm="GSM1", claim=str(claim), man={"attempts": 1}, scr=str(scr), t0=time.time(), compute_s=1.0,
                                heart=types.SimpleNamespace(stop=threading.Event(), lost=False))
    res = D.finish_phase(a, {}, {"root": str(tmp_path / "root")}, job)
    return res, events, published, released


def test_finish_phase_never_publishes_a_sample_without_its_whitelist(tmp_path, monkeypatch):
    res, events, published, released = finish(tmp_path, monkeypatch, NOT_FOUND, True)
    assert res.startswith("failed: whitelist check")
    assert published == []
    assert [(e["event"], e["reason"]) for e in events] == [("failed", "whitelist_missing")]
    assert "not found for 10x-3p-v3" in events[0]["detail"]
    assert released == [("GSM1", False)]  # requeued for another attempt, not dropped


def test_finish_phase_publishes_a_sample_with_its_whitelist(tmp_path, monkeypatch):
    res, events, published, released = finish(tmp_path, monkeypatch, RESOLVED, False)
    assert res.startswith("done") and len(published) == 1
    assert [e["event"] for e in events] == ["done"] and events[0]["whitelist_check"]["ok"] is True
