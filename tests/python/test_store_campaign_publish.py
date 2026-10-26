# SPDX-License-Identifier: MIT
"""scripts/store_publish.py: a new study from pipeline output to store objects, with a stand-in packer and transcoder."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_p = Path(__file__).with_name("test_store_campaign_common.py")
if "test_store_campaign_common" not in sys.modules:
    _s = importlib.util.spec_from_file_location("test_store_campaign_common", _p)
    _m = importlib.util.module_from_spec(_s)
    sys.modules["test_store_campaign_common"] = _m
    _s.loader.exec_module(_m)
C = sys.modules["test_store_campaign_common"]
camp = C.load_script("store_campaign")
pub = C.load_script("store_publish")

PUB, PRIV = "singlet-data", "singlet-restricted"

FAKE_PACKER = """
import contextlib, json, zipfile
from pathlib import Path

LEAK = False

def pack_gse(gse_id, results_dir, catalog_path, out_path, *, verbose=True, metadata_enriched_path=None,
             publications_enriched_path=None, series_meta_path=None, strict=False, private_out_path=None, _catalog_df=None):
    gsms = [l.strip() for l in Path(catalog_path).read_text().splitlines() if l.strip()]
    done = [g for g in gsms if (Path(results_dir) / g / "out" / "summary.json").exists()]
    if not done:
        raise ValueError(f"No done GSMs found for {gse_id} under {results_dir}")
    with zipfile.ZipFile(out_path, "w") as z, (zipfile.ZipFile(private_out_path, "w") if private_out_path else contextlib.nullcontext()) as zr:
        z.writestr("manifest.json", json.dumps({"gse_id": gse_id, "gsm_ids": done}))
        z.writestr("study_meta.json", "{}")
        for g in done:
            out = Path(results_dir) / g / "out"
            for f in sorted(out.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(out).as_posix()
                arc = f"samples/{g}/{rel}"
                if rel.startswith("donor/snp_") and not LEAK:
                    if zr is not None:
                        zr.writestr(arc, f.read_bytes())
                    continue
                z.writestr(arc, f.read_bytes())
    if series_meta_path:
        Path(series_meta_path).write_text("{}")
    return Path(out_path)
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    class E:
        pass

    e = E()
    e.tmp, e.mp = tmp_path, monkeypatch
    e.root = tmp_path / "r2"
    (e.root / PUB).mkdir(parents=True)
    e.tool = C.write_fake_tool(tmp_path)
    (tmp_path / "fakepack").mkdir()
    (tmp_path / "fakepack" / "fake_packer.py").write_text(FAKE_PACKER)
    e.results = tmp_path / "results"
    for g in ("GSM70001", "GSM70002"):
        out = e.results / g / "out"
        (out / "donor").mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({"n_cells_called": 3}))
        (out / "exon_counts.1pz").write_bytes(g.encode() * 50)
        (out / "donor" / "snp_ad.1pz").write_bytes(b"germline" * 30)
        (out / "donor" / "donor_assignments.tsv").write_bytes(b"AAAC\tD1\n")
    e.catalog = tmp_path / "catalog.txt"
    e.catalog.write_text("GSM70001\nGSM70002\nGSM70003\n")  # the third has no result: not done
    monkeypatch.setenv("FAKE_COUNT_FILE", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("FAKE_PAD", "2500")
    for k in ("FAKE_MIGRATE_FAIL", "FAKE_VERIFY_FAIL", "INGEST_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    sys.modules.pop("fake_packer", None)

    def run(*extra, gse="GSE700001", apply=True, base=True):
        argv = [
            gse,
            "--results",
            str(e.results),
            "--catalog",
            str(e.catalog),
            "--packer-module",
            "fake_packer",
            "--packer-path",
            str(tmp_path / "fakepack"),
            "--local-root",
            str(e.root),
            "--scratch",
            str(tmp_path / "scratch"),
            "--part-size-mib",
            "0.001",
            "--retry-sleep",
            "0",
            "--json-out",
            str(tmp_path / "result.json"),
        ]
        argv += C.tool_cmds(e.tool)
        if apply:
            argv.append("--apply")
        return pub.main(argv + list(extra))

    e.run = run
    e.result = lambda: json.loads((tmp_path / "result.json").read_text())
    return e


def keys(root, bucket=PUB):
    return sorted(
        p.relative_to(root / bucket).as_posix() for p in (root / bucket).rglob("*") if p.is_file()
    )


def test_dry_run_packs_migrates_verifies_and_writes_nothing(env):
    assert env.run(apply=False) == 0
    assert (
        env.result()["status"] == "dry_ok" and env.result()["uploads"]["would_upload"] == 4
    )  # 2 samples + 2 packs
    assert keys(env.root) == []
    assert not (env.tmp / "scratch" / "work").exists()  # the temporary legacy bundle is gone


def test_publish_writes_store_objects_manifest_last_and_no_legacy_bundle(env):
    assert env.run() == 0
    r = env.result()
    assert (
        r["status"] == "ok"
        and r["n_samples"] == 2
        and r.get("legacy_bytes") is None
        and "legacy_key" not in r
    )
    ks = keys(env.root)
    assert not [k for k in ks if k.startswith("data/")]  # nothing in the legacy key space
    assert "store/m/GSE700001.json" in ks and "store/ledger/GSE700001.jsonl" in ks
    assert (
        len([k for k in ks if k.startswith("store/s/")]) == 2
        and len([k for k in ks if k.startswith("store/ref/")]) == 2
    )
    m = json.loads((env.root / PUB / "store" / "m" / "GSE700001.json").read_text())
    assert "legacy" not in m or m["legacy"] is None  # a new study has no legacy bundle
    led = [
        json.loads(x)
        for x in (env.root / PUB / "store" / "ledger" / "GSE700001.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in led][:2] == ["start", "packed"] and led[-1]["event"] == "committed"
    rows = json.loads(Path(r["rows_file"]).read_text())
    assert (
        rows["store_migration"]["status"] == "retired"
        and rows["store_migration"]["verify_ok"] == 1
        and rows["store_migration"]["legacy_key"] is None
    )
    assert len(rows["store_sample"]) == 2 and rows["store_study"]["gse_id"] == "GSE700001"
    assert led[-1]["summary"]["stage"] == "done"


def test_the_manifest_is_written_after_every_object(env, monkeypatch):
    order = []
    orig = camp.LocalStore.put_file, camp.LocalStore.put_bytes

    def pf(self, bucket, key, *a, **kw):
        order.append(key)
        return orig[0](self, bucket, key, *a, **kw)

    def pb(self, bucket, key, *a, **kw):
        order.append(key)
        return orig[1](self, bucket, key, *a, **kw)

    monkeypatch.setattr(camp.LocalStore, "put_file", pf)
    monkeypatch.setattr(camp.LocalStore, "put_bytes", pb)
    assert env.run() == 0
    data = [k for k in order if not k.startswith("store/ledger/")]
    assert data[-1] == "store/m/GSE700001.json" and len(data) == 5
    assert order.index("store/ledger/GSE700001.jsonl") < order.index(
        "store/m/GSE700001.json"
    )  # ledger with the objects, then the manifest
    assert order[-1] == "store/ledger/GSE700001.jsonl"  # and the committed record after it


def test_restricted_members_go_to_the_private_bucket_only_when_asked(env):
    assert env.run() == 0
    assert not (env.root / PRIV).exists()
    env2 = env.root.parent / "r2b"
    (env2 / PUB).mkdir(parents=True)
    argv = [
        "GSE700002",
        "--results",
        str(env.results),
        "--catalog",
        str(env.catalog),
        "--packer-module",
        "fake_packer",
        "--packer-path",
        str(env.tmp / "fakepack"),
        "--local-root",
        str(env2),
        "--scratch",
        str(env.tmp / "s2"),
        "--part-size-mib",
        "0.001",
        "--retry-sleep",
        "0",
        "--apply",
        "--private",
        PRIV,
    ] + C.tool_cmds(env.tool)
    assert pub.main(argv) == 0
    side = env2 / PRIV / "restricted" / "GSE700002" / "GSE700002.restricted.singlet"
    with zipfile.ZipFile(side) as z:
        assert sorted(n for n in z.namelist() if n.startswith("samples/")) == [
            "samples/GSM70001/donor/snp_ad.1pz",
            "samples/GSM70002/donor/snp_ad.1pz",
        ]
    assert keys(env2, PRIV) == ["restricted/GSE700002/GSE700002.restricted.singlet"]
    assert not [k for k in keys(env2) if not k.startswith("store/")]


def test_a_packer_that_leaks_restricted_members_is_refused(env):
    sys.path.insert(0, str(env.tmp / "fakepack"))
    try:
        import fake_packer

        fake_packer.LEAK = True
        assert env.run() == 1
    finally:
        sys.path.remove(str(env.tmp / "fakepack"))
        sys.modules.pop("fake_packer", None)
    r = env.result()
    assert r["status"] == "blocked_restricted"
    assert all(k.startswith("store/ledger/") for k in keys(env.root))


def test_verify_failure_publishes_nothing(env, monkeypatch):
    monkeypatch.setenv("FAKE_VERIFY_FAIL", "GSE700001")
    # the fake verifier names the study after the bundle file, which is <GSE>.singlet
    assert env.run() == 1
    assert env.result()["status"] == "verify_failed"
    assert all(k.startswith("store/ledger/") for k in keys(env.root))


def test_a_study_with_no_finished_samples_fails_at_the_packer(env):
    env.catalog.write_text("GSM70003\n")
    assert env.run() == 1
    r = env.result()
    assert r["status"] == "failed" and r["stage"] == "pack" and "No done GSMs" in r["error"]
    assert keys(env.root) == ["store/ledger/GSE700001.jsonl"]  # only the error record


def test_republishing_uploads_nothing_new(env, monkeypatch):
    assert env.run() == 0
    calls = []
    orig = camp.LocalStore.put_file
    monkeypatch.setattr(
        camp.LocalStore,
        "put_file",
        lambda self, b, k, *a, **kw: (calls.append(k), orig(self, b, k, *a, **kw))[1],
    )
    assert env.run() == 0
    assert calls == []  # packed bundles are deterministic here: every object is already in R2
    assert env.result()["uploads"]["skipped"] == 4


class _Ingest(BaseHTTPRequestHandler):
    status = 200
    log = []

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Ingest.log.append(
            (self.path, self.headers.get("X-Ingest-Token"), json.loads(self.rfile.read(n)))
        )
        body = b'{"ok":true}'
        self.send_response(_Ingest.status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_catalog_rows_are_posted_and_a_failed_post_does_not_unpublish(env, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setenv("INGEST_TOKEN", "tok")
    try:
        _Ingest.status, _Ingest.log = 200, []
        assert env.run("--ingest-url", url) == 0
        assert env.result()["catalog"] == "posted"
        tables = [p.rsplit("/", 1)[1] for p, _t, _b in _Ingest.log]
        assert tables == ["store-study"] and all(t == "tok" for _p, t, _b in _Ingest.log)
        assert (
            _Ingest.log[0][2]["final"] is True
            and _Ingest.log[0][2]["manifest"]["gse"] == "GSE700001"
            and "study_files" not in _Ingest.log[0][2]["manifest"]
        )
        monkeypatch.setattr(camp.time, "sleep", lambda s: None)
        _Ingest.status, _Ingest.log = 400, []
        assert env.run("--ingest-url", url) == 3  # live, rows not accepted
        assert env.result()["status"] == "ok" and env.result()["catalog"].startswith("not posted")
        assert (env.root / PUB / "store" / "m" / "GSE700001.json").exists()
        monkeypatch.delenv("INGEST_TOKEN")
        assert env.run("--ingest-url", url) == 3 and "INGEST_TOKEN" in env.result()["catalog"]
    finally:
        srv.shutdown()


def test_private_bucket_must_differ(env):
    with pytest.raises(SystemExit):
        env.run("--private", PUB)
