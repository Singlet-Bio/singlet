# SPDX-License-Identifier: MIT
"""scripts/v2_campaign.py against directory-backed buckets (and an in-memory S3 for the boto3 paths).

Covers: manifest written last, idempotent reruns, one failing study does not stop the shard, nothing outside v2/ is ever
written, resume, dry mode, restricted members refused, sharding, catalog SQL (executed in SQLite) and POST rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_p = Path(__file__).with_name("test_v2_campaign_common.py")
if "test_v2_campaign_common" not in sys.modules:
    _s = importlib.util.spec_from_file_location("test_v2_campaign_common", _p)
    _m = importlib.util.module_from_spec(_s)
    sys.modules["test_v2_campaign_common"] = _m
    _s.loader.exec_module(_m)
C = sys.modules["test_v2_campaign_common"]
camp = C.load_script("v2_campaign")

PUB = "singlet-data"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "r2"
    tool = C.write_fake_tool(tmp_path)
    counts = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_COUNT_FILE", str(counts))
    for k in ("FAKE_MIGRATE_FAIL", "FAKE_VERIFY_FAIL", "FAKE_BAD_NAME", "FAKE_EXPECT_LEGACY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FAKE_PAD", "2500")  # sample objects of ~2.6 kB: three parts at the 1048-byte test part size
    store = camp.LocalStore(root)
    monkeypatch.setattr(camp, "make_store", lambda a: store)

    class E:
        pass

    e = E()
    e.tmp, e.root, e.tool, e.counts, e.store, e.mp = tmp_path, root, tool, counts, store, monkeypatch
    e.results = tmp_path / "results.jsonl"
    e.rows = tmp_path / "rows"
    e.ledger = tmp_path / "ledger"

    def run(*extra, mode="full", workers=1, base=True):
        args = ["run", "--mode", mode, "--local-root", str(root), "--scratch", str(tmp_path / "scratch"),
                "--results", str(e.results), "--rows-dir", str(e.rows), "--ledger-dir", str(e.ledger),
                "--workers", str(workers), "--io-threads", "4", "--retry-sleep", "0", "--part-size-mib", "0.001"]
        args += C.tool_cmds(tool) + list(extra)
        if base and not any(x in extra for x in ("--gse", "--gse-list", "--shard", "--limit")):
            args += ["--shard", "0/1"]
        return camp.main(args)

    e.run = run
    e.rows_of = lambda: {r["gse"]: r for r in (json.loads(ln) for ln in e.results.read_text().splitlines())}
    e.ncalls = lambda what: sum(1 for ln in (counts.read_text().splitlines() if counts.exists() else []) if ln.startswith(what))
    return e


# ── ETag arithmetic ───────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [0, 1, 99, 100, 101, 250, 300])
def test_etag_rule(tmp_path, size):
    data = bytes(range(256)) * 2
    data = (data * 2)[:size]
    p = tmp_path / "f"
    p.write_bytes(data)
    d = camp.hash_file(p, 100)
    assert d.size == size and d.sha256 == hashlib.sha256(data).hexdigest() and d.md5 == hashlib.md5(data).hexdigest()
    if size <= 100:
        assert d.etag == hashlib.md5(data).hexdigest() and "-" not in d.etag
    else:
        parts = [hashlib.md5(data[i : i + 100]).digest() for i in range(0, size, 100)]
        assert d.etag == hashlib.md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"
    assert camp.digest_bytes(data, 100).etag == d.etag


def test_headers_per_key_class():
    ct, cc = camp.object_headers("v2/s/GSM1/" + "a" * 32 + ".s2")
    assert ct == "application/octet-stream" and "immutable" in cc and "31536000" in cc
    assert camp.object_headers("v2/ref/" + "b" * 32 + ".pack")[1] == cc
    assert camp.object_headers("v2/m/GSE1.json") == ("application/json", "public, max-age=300")
    assert camp.object_headers("v2/ref/index.json") == ("application/json", "public, max-age=300")  # mutable, not immutable
    assert camp.object_headers("v2/ledger/GSE1.jsonl")[1] == "no-store"


# ── the happy path ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_full_run_commits_manifest_last_and_never_touches_legacy(env):
    gses = C.seed(env.root, 3)
    before = {g: (env.root / PUB / "data" / g / f"{g}.singlet").read_bytes() for g in gses}
    assert env.run() == 0
    rows = env.rows_of()
    assert [rows[g]["status"] for g in gses] == ["ok"] * 3
    for g in gses:
        r = rows[g]
        assert r["n_samples"] == 2 and r["verify"]["ok"] and r["legacy_bytes"] == len(before[g])
        assert r["legacy_sha256"] == hashlib.sha256(before[g]).hexdigest()
        assert r["members_by_kind"]["pz2"] == 2 and r["members_by_kind"]["cells"] == 2 and r["members_by_kind"]["derived"] == 2
        assert r["uploads"]["uploaded"] == r["uploads"]["objects"] == 3 and r["uploads"]["skipped"] == 0  # 2 samples + 1 pack
        assert r["report"] == {"fallbacks": 1}
        m = json.loads((env.root / PUB / "v2" / "m" / f"{g}.json").read_text())
        assert m["legacy"]["etag"] == r["legacy_etag"] and m["legacy"]["bytes"] == r["legacy_bytes"]
        assert m["legacy"]["sha256"] == r["legacy_sha256"] and m["legacy"]["ledger"] == f"v2/ledger/{g}.jsonl"
        for s in m["samples"]:
            blob = (env.root / PUB / s["key"]).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == s["sha256"] and s["key"].endswith(s["sha256"][:32] + ".s2")
        led = [json.loads(x) for x in (env.root / PUB / "v2" / "ledger" / f"{g}.jsonl").read_text().splitlines()]
        assert [e["event"] for e in led][0] == "start" and led[-1]["event"] == "committed"
        assert led[-1]["summary"]["status"] == "ok"
        assert sum(1 for e in led if e["event"] == "object") == r["uploads"]["objects"]
    # legacy bytes untouched, and every write was under v2/
    for g in gses:
        assert (env.root / PUB / "data" / g / f"{g}.singlet").read_bytes() == before[g]
    writes = [(op, b, k) for op, b, k in env.store.ops if op in ("put", "delete", "copy", "create_bucket")]
    assert writes and all(k.startswith("v2/") for _op, _b, k in writes) and not any(op == "delete" for op, _b, _k in writes)
    # manifest is the last data write of its study (the ledger objects are bookkeeping)
    for g in gses:
        puts = [k for op, _b, k in env.store.ops if op == "put" and not k.startswith("v2/ledger/")
                and (g[3:].lstrip("0") in k or k == f"v2/m/{g}.json")]
        study_puts = [k for k in puts if f"GSM{int(g[3:])}000" in k or k == f"v2/m/{g}.json"]
        assert study_puts[-1] == f"v2/m/{g}.json" and study_puts.count(f"v2/m/{g}.json") == 1
    i_manifest = [i for i, (op, _b, k) in enumerate(env.store.ops) if op == "put" and k == "v2/m/GSE000001.json"][0]
    later = [k for op, _b, k in env.store.ops[i_manifest + 1 :] if op == "put" and "GSE000001" in k or op == "put" and "GSM1000" in k]
    assert all(k.startswith("v2/ledger/") for k in later)  # only the committed ledger line follows the manifest
    # headers
    s0 = json.loads((env.root / PUB / "v2" / "m" / f"{gses[0]}.json").read_text())["samples"][0]["key"]
    h = env.store.head(PUB, s0)
    assert h.content_type == "application/octet-stream" and h.cache_control == "public, max-age=31536000, immutable"
    assert h.meta["sha256"] == s0.rsplit("/", 1)[1][:32] + h.meta["sha256"][32:]
    hm = env.store.head(PUB, f"v2/m/{gses[0]}.json")
    assert hm.content_type == "application/json" and hm.cache_control == "public, max-age=300"
    assert "-" in h.etag  # tiny test part size: the sample object was uploaded multipart
    assert (env.rows / f"{gses[0]}.json").exists() and (env.ledger / f"{gses[0]}.jsonl").exists()


def test_rerun_uploads_no_object_twice(env):
    gses = C.seed(env.root, 2)
    assert env.run() == 0
    env.store.ops.clear()
    env.results.unlink()
    assert env.run() == 0
    rows = env.rows_of()
    for g in gses:
        assert rows[g]["uploads"]["uploaded"] == 0 and rows[g]["uploads"]["skipped"] == rows[g]["uploads"]["objects"] == 3
    puts = [k for op, _b, k in env.store.ops if op == "put"]
    assert sorted(set(puts)) == sorted({f"v2/m/{g}.json" for g in gses} | {f"v2/ledger/{g}.jsonl" for g in gses})
    led = (env.root / PUB / "v2" / "ledger" / f"{gses[0]}.jsonl").read_text().splitlines()
    assert sum(1 for x in led if json.loads(x)["event"] == "committed") == 2  # history is kept across runs


def test_resume_skips_done_and_redoes_a_changed_legacy_bundle(env):
    gses = C.seed(env.root, 3)
    assert env.run() == 0 and env.ncalls("migrate") == 3
    env.results.unlink()
    assert env.run("--resume") == 0
    assert env.ncalls("migrate") == 3  # nothing transcoded again
    assert {r["status"] for r in env.rows_of().values()} == {"skipped_done"}
    assert all(r["v2_bytes"] and r["n_samples"] == 2 for r in env.rows_of().values())
    # the rows file for a skipped study still describes the committed migration
    assert json.loads((env.rows / f"{gses[0]}.json").read_text())["v2_migration"]["status"] == "ok"
    # the legacy bundle of one study is replaced: only that one is migrated again
    C.put_legacy(env.root, gses[1], ["GSM20001", "GSM20002", "GSM20003"], salt="changed")
    env.results.unlink()
    assert env.run("--resume") == 0
    rows = env.rows_of()
    assert rows[gses[1]]["status"] == "ok" and rows[gses[1]]["n_samples"] == 3
    assert rows[gses[0]]["status"] == rows[gses[2]]["status"] == "skipped_done"
    assert env.ncalls("migrate") == 4


# ── failures ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_failing_study_is_recorded_and_the_shard_continues(env):
    gses = C.seed(env.root, 3)
    env.mp.setenv("FAKE_MIGRATE_FAIL", gses[1])
    assert env.run() == 1
    rows = env.rows_of()
    assert rows[gses[1]]["status"] == "migrate_failed" and "refusing" in rows[gses[1]]["tail"]
    assert rows[gses[0]]["status"] == rows[gses[2]]["status"] == "ok"
    assert not (env.root / PUB / "v2" / "m" / f"{gses[1]}.json").exists()
    assert not list((env.root / PUB / "v2" / "s").glob("GSM20*"))
    led = [json.loads(x) for x in (env.root / PUB / "v2" / "ledger" / f"{gses[1]}.jsonl").read_text().splitlines()]
    assert led[-1]["event"] == "error" and led[-1]["status"] == "migrate_failed"
    # the failure row only carries the migration record
    assert set(json.loads((env.rows / f"{gses[1]}.json").read_text())) == {"v2_migration"}


def test_verify_failure_uploads_nothing_for_that_study(env):
    gses = C.seed(env.root, 2)
    env.mp.setenv("FAKE_VERIFY_FAIL", gses[0])
    assert env.run() == 1
    rows = env.rows_of()
    assert rows[gses[0]]["status"] == "verify_failed" and rows[gses[0]]["verify"]["ok"] is False
    assert rows[gses[1]]["status"] == "ok"
    keys = [k for op, _b, k in env.store.ops if op == "put"]
    assert not any("GSM10001" in k or "GSM10002" in k or k == f"v2/m/{gses[0]}.json" for k in keys)


def test_verify_sees_the_manifest_that_will_be_published(env):
    C.seed(env.root, 1)
    env.mp.setenv("FAKE_EXPECT_LEGACY", "1")  # the fake verifier fails if the manifest has no legacy block yet
    assert env.run() == 0


def test_upload_failure_leaves_no_manifest_and_does_not_stop_the_shard(env):
    gses = C.seed(env.root, 3)

    class Flaky(camp.LocalStore):
        def put_file(self, bucket, key, path, **kw):
            if "GSM20002" in key:
                raise OSError("disk on fire")
            return super().put_file(bucket, key, path, **kw)

    store = Flaky(env.root)
    env.mp.setattr(camp, "make_store", lambda a: store)
    assert env.run() == 1
    rows = env.rows_of()
    assert rows[gses[1]]["status"] == "upload_failed" and "disk on fire" in rows[gses[1]]["error"]
    assert rows[gses[0]]["status"] == rows[gses[2]]["status"] == "ok"
    assert not (env.root / PUB / "v2" / "m" / f"{gses[1]}.json").exists()  # invisible: no manifest


def test_transient_upload_error_is_retried(env):
    gses = C.seed(env.root, 1)
    state = {"n": 0}

    class Flaky(camp.LocalStore):
        def put_file(self, bucket, key, path, **kw):
            if "GSM10001" in key and state["n"] < 2:
                state["n"] += 1
                raise OSError("connection reset")
            return super().put_file(bucket, key, path, **kw)

    env.mp.setattr(camp, "make_store", lambda a: Flaky(env.root))
    assert env.run() == 0 and state["n"] == 2
    assert env.rows_of()[gses[0]]["status"] == "ok"


def test_legacy_object_replaced_during_the_read(env):
    gses = C.seed(env.root, 2)

    class Racy(camp.LocalStore):
        def download(self, bucket, key, path, **kw):
            if gses[0] in key:
                raise camp.LegacyChanged("replaced")
            return super().download(bucket, key, path, **kw)

    env.mp.setattr(camp, "make_store", lambda a: Racy(env.root))
    assert env.run() == 1
    rows = env.rows_of()
    assert rows[gses[0]]["status"] == "legacy_changed" and rows[gses[1]]["status"] == "ok"


def test_legacy_object_replaced_after_the_read(env):
    gses = C.seed(env.root, 1)

    class Racy(camp.LocalStore):
        n = 0

        def head(self, bucket, key):
            h = super().head(bucket, key)
            if h is not None and key == camp.legacy_key(gses[0]):
                Racy.n += 1
                if Racy.n >= 2:
                    h.etag = "changed"
            return h

    env.mp.setattr(camp, "make_store", lambda a: Racy(env.root))
    assert env.run() == 1
    assert env.rows_of()[gses[0]]["status"] == "legacy_changed"


def test_object_name_that_is_not_its_digest_is_refused(env):
    gses = C.seed(env.root, 1)
    env.mp.setenv("FAKE_BAD_NAME", "1")
    assert env.run() == 1
    r = env.rows_of()[gses[0]]
    assert r["status"] == "collect_failed" and "digest16" in r["error"]
    assert all(k.startswith("v2/ledger/") for op, _b, k in env.store.ops if op == "put")  # only the error record


def test_restricted_members_block_the_study(env):
    gses = C.seed(env.root, 2, restricted="GSE000002")
    assert env.run() == 0  # blocked is reported, not a failure of the campaign
    rows = env.rows_of()
    assert rows[gses[1]]["status"] == "blocked_restricted" and rows[gses[1]]["restricted_members"] == 2
    assert rows[gses[0]]["status"] == "ok"
    assert env.ncalls("migrate") == 1


def test_missing_study(env):
    C.seed(env.root, 1)
    assert env.run("--gse", "GSE000001", "GSE999999") == 0
    assert env.rows_of()["GSE999999"]["status"] == "missing"


# ── modes ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_dry_mode_writes_nothing(env):
    gses = C.seed(env.root, 2)
    before = sorted(str(p) for p in env.root.rglob("*"))
    assert env.run(mode="dry") == 0
    assert sorted(str(p) for p in env.root.rglob("*")) == before
    assert not [op for op, _b, _k in env.store.ops if op in ("put", "delete", "copy")]
    r = env.rows_of()[gses[0]]
    assert r["status"] == "dry_ok" and r["uploads"]["would_upload"] == r["uploads"]["objects"] and r["v2_bytes"] > 0
    assert not env.rows.exists()  # no catalog rows for a dry run
    # after a real run a dry run reports everything as already there
    assert env.run() == 0
    env.results.unlink()
    assert env.run(mode="dry") == 0
    r = env.rows_of()[gses[0]]
    assert r["uploads"]["would_upload"] == 0 and r["uploads"]["would_skip"] == r["uploads"]["objects"]


def test_list_only(env):
    gses = C.seed(env.root, 2, restricted="GSE000001")
    assert env.run("--list-only", mode="dry") == 0
    rows = env.rows_of()
    assert rows[gses[0]]["status"] == "blocked_restricted" and rows[gses[1]]["status"] == "dry_listed"
    assert env.ncalls("migrate") == 0


def test_pilot_picks_across_sizes_and_adds_named_studies(env):
    for i in range(1, 8):
        C.put_legacy(env.root, f"GSE{i:06d}", [f"GSM{i}0001"] * 1 + [f"GSM{i}{j:04d}" for j in range(2, 2 + i)], salt=str(i))
    assert env.run("--pilot-n", "3", "--gse", "GSE000004", mode="pilot", base=False) == 0
    got = set(env.rows_of())
    assert "GSE000004" in got and len(got) == 3  # smallest, largest-in-range, plus the named one is among the picks
    assert all(r["status"] == "ok" for r in env.rows_of().values())
    # pilot reads every object back
    some = next(iter(env.rows_of().values()))
    assert any(e["event"] == "object" and e.get("readback") == "sha256" or e.get("action") == "skipped"
               for e in map(json.loads, (env.root / PUB / "v2" / "ledger" / f"{some['gse']}.jsonl").read_text().splitlines()))


def test_selection_sharding_and_order():
    assert camp.accession_order(["GSE10", "GSE9", "GSE100", "GSE2"]) == ["GSE2", "GSE9", "GSE10", "GSE100"]
    picks = camp.pilot_pick([(f"GSE{i}", i * 10) for i in range(1, 11)], 3, 1e9)
    assert picks[0] == "GSE1" and picks[-1] == "GSE10" and len(picks) == 3
    assert camp.pilot_pick([("GSE1", 5), ("GSE2", 9_000_000_000)], 2, 8e9) == ["GSE1"]  # the very large one is excluded
    with pytest.raises(Exception):
        camp.parse_shard("4/4")


def test_shards_partition_the_studies(env):
    gses = C.seed(env.root, 7)
    seen = []
    for i in range(3):
        if env.results.exists():
            env.results.unlink()
        assert env.run("--list-only", "--shard", f"{i}/3", mode="dry", base=False) == 0
        seen.append(sorted(env.rows_of()))
    assert sorted(sum(seen, [])) == gses and len(set(sum(seen, []))) == 7
    assert seen[0] == gses[0::3]


def test_exclude_list_and_gse_list(env, tmp_path):
    gses = C.seed(env.root, 4)
    ex = tmp_path / "ex.txt"
    ex.write_text("GSE000002  # big\n")
    lst = tmp_path / "l.txt"
    lst.write_text("\n".join(gses[:3]))
    assert env.run("--gse-list", str(lst), "--exclude-list", str(ex), "--list-only", mode="dry") == 0
    assert sorted(env.rows_of()) == [gses[0], gses[2]]


def test_deadline_defers_studies_that_cannot_finish(env):
    gses = C.seed(env.root, 2)
    import time as _t

    assert env.run("--deadline-epoch", str(_t.time() + 0.001), "--min-rate-mbs", "0.0000001") == 0
    assert {r["status"] for r in env.rows_of().values()} == {"deferred"}
    assert env.ncalls("migrate") == 0


def test_concurrent_workers(env):
    C.seed(env.root, 6)
    assert env.run(workers=3) == 0
    assert {r["status"] for r in env.rows_of().values()} == {"ok"}


# ── the write guard ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_guard_refuses_anything_outside_v2(tmp_path):
    store = camp.LocalStore(tmp_path)
    g = camp.GuardedStore(store, {PUB: ("v2/",)})
    g.put_bytes(PUB, "v2/m/GSE1.json", b"{}", content_type="application/json", cache_control="x")
    for bad in ("data/GSE1/GSE1.singlet", "restricted/GSE1/x", "v1/x"):
        with pytest.raises(camp.WriteRefused):
            g.put_bytes(PUB, bad, b"x", content_type="a", cache_control="b")
    with pytest.raises(camp.WriteRefused):
        g.put_bytes("other-bucket", "v2/x", b"x", content_type="a", cache_control="b")
    with pytest.raises(camp.WriteRefused):
        g.delete(PUB, "v2/m/GSE1.json")  # deletes are not allowed at all
    with pytest.raises(camp.WriteRefused):
        g.copy(PUB, "v2/m/GSE1.json", PUB, "data/GSE1/GSE1.singlet", size=2)
    with pytest.raises(camp.WriteRefused):
        g.create_bucket("new")
    assert g.head(PUB, "v2/m/GSE1.json").size == 2  # reads pass through


# ── S3Store (the boto3 code paths, against the in-memory fake) ────────────────────────────────────────────────────────


def _s3_env(env, fake):
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    env.mp.setattr(camp, "make_store", lambda a: store)
    return store


def _legacy_into(fake, gse="GSE000001", gsms=("GSM10001", "GSM10002")):
    p = C.make_bundle(Path(fake_tmp := Path(__import__("tempfile").mkdtemp())) / "b.singlet", gse, list(gsms), salt=gse)
    fake.put_raw(PUB, f"data/{gse}/{gse}.singlet", p.read_bytes())
    return p.read_bytes()


def test_s3store_run_end_to_end(env):
    fake = C.FakeS3()
    blob = _legacy_into(fake)
    _s3_env(env, fake)
    assert env.run() == 0
    r = env.rows_of()["GSE000001"]
    assert r["status"] == "ok" and r["legacy_sha256"] == hashlib.sha256(blob).hexdigest()
    m = json.loads(fake.objs[(PUB, "v2/m/GSE000001.json")]["data"])
    k = m["samples"][0]["key"]
    o = fake.objs[(PUB, k)]
    assert o["CacheControl"] == "public, max-age=31536000, immutable" and o["ContentType"] == "application/octet-stream"
    assert "-" in o["etag"]  # multipart at the tiny test part size
    assert ("upload_file", PUB, k, "multipart") in fake.calls
    assert any(c[0] == "get" and c[1] == PUB and c[2].startswith("data/") and c[3] for c in fake.calls)  # ranged legacy read
    assert not any(c[0] in ("delete", "copy_object") for c in fake.calls)
    assert not any(c[0] in ("put_object", "upload_file") and not c[2].startswith("v2/") for c in fake.calls)
    assert "etag_note" not in json.dumps([json.loads(x) for x in fake.objs[(PUB, "v2/ledger/GSE000001.jsonl")]["data"].decode().splitlines()])


def test_s3store_download_is_pinned_to_the_etag(env):
    fake = C.FakeS3()
    _legacy_into(fake)
    store = _s3_env(env, fake)
    h = store.head(PUB, "data/GSE000001/GSE000001.singlet")
    store.download(PUB, "data/GSE000001/GSE000001.singlet", env.tmp / "x", if_match=h.etag, size=h.size, part=1000, threads=3)
    assert (env.tmp / "x").read_bytes() == fake.objs[(PUB, "data/GSE000001/GSE000001.singlet")]["data"]
    with pytest.raises(camp.LegacyChanged):
        store.download(PUB, "data/GSE000001/GSE000001.singlet", env.tmp / "y", if_match="0" * 32, size=h.size)


def test_s3store_accepts_an_etag_rule_difference_only_after_a_sha256_readback(env):
    class OddEtag(C.FakeS3):
        def upload_file(self, path, Bucket, Key, ExtraArgs=None, Config=None):
            super().upload_file(path, Bucket, Key, ExtraArgs, Config)
            self.objs[(Bucket, Key)]["etag"] = "feedface-7"

    fake = OddEtag()
    _legacy_into(fake)
    _s3_env(env, fake)
    assert env.run() == 0
    led = [json.loads(x) for x in fake.objs[(PUB, "v2/ledger/GSE000001.jsonl")]["data"].decode().splitlines()]
    objs = [e for e in led if e["event"] == "object" and e["action"] == "uploaded"]
    assert objs and all(e["readback"] == "sha256" and "etag_note" in e for e in objs)


def test_s3store_corrupted_object_fails_the_study(env):
    class Corrupting(C.FakeS3):
        def upload_file(self, path, Bucket, Key, ExtraArgs=None, Config=None):
            super().upload_file(path, Bucket, Key, ExtraArgs, Config)
            o = self.objs[(Bucket, Key)]
            if Key.endswith(".s2"):
                o["data"] = o["data"][:-1] + b"\x00"
                o["etag"] = "0" * 32

    fake = Corrupting()
    _legacy_into(fake)
    _s3_env(env, fake)
    assert env.run() == 1
    assert env.rows_of()["GSE000001"]["status"] == "upload_failed"
    assert (PUB, "v2/m/GSE000001.json") not in fake.objs


# ── catalog ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def _sqlite_with_schema():
    db = sqlite3.connect(":memory:")
    for t, spec in camp.CATALOG_SCHEMA.items():
        sql = {"text": "TEXT", "int": "INTEGER", "real": "REAL"}
        cols = ", ".join(f'"{c}" {sql[k]}' for c, k in spec["cols"].items())
        db.execute(f'CREATE TABLE "{t}" ({cols}, "{spec["stamp"]}" TEXT, PRIMARY KEY ({", ".join(chr(34) + p + chr(34) for p in spec["pk"])}))')
    return db


def test_catalog_sql_executes_and_is_idempotent(env):
    gses = C.seed(env.root, 3)
    assert env.run() == 0
    out = env.tmp / "sql"
    argv = ["catalog-sql", "--rows-dir", str(env.rows), "--out", str(out), "--max-rows", "4"]
    assert camp.main(argv) == 0
    files = sorted(p.name for p in out.glob("*.sql"))
    assert files[0].endswith("v2_sample.sql") and files[-1].endswith("v2_study.sql")  # study rows last
    assert len([f for f in files if "v2_sample" in f]) == 2  # 6 sample rows, 4 per file
    assert "wrangler d1 execute singlet-catalog --remote" in (out / "APPLY.txt").read_text()
    db = _sqlite_with_schema()
    for f in files:
        db.executescript((out / f).read_text())
    assert db.execute("select count(*) from v2_sample").fetchone()[0] == 6
    assert db.execute("select count(*) from v2_study").fetchone()[0] == 3
    assert db.execute("select count(*) from v2_migration").fetchone()[0] == 3
    row = db.execute("select status, legacy_bytes, verify_ok, members_by_kind from v2_migration where gse_id='GSE000001'").fetchone()
    assert row[0] == "ok" and row[1] > 0 and row[2] == 1 and json.loads(row[3])["pz2"] == 2
    s = db.execute("select key, members, stats, n_called from v2_sample where gsm_id='GSM10001'").fetchone()
    assert s[0].startswith("v2/s/GSM10001/") and json.loads(s[1])["exon_counts.1pz"] == [1, 128, 99]
    assert json.loads(s[2])["median_umi"] == 1500 and s[3] == 3
    # idempotent: applying again changes nothing
    for f in files:
        db.executescript((out / f).read_text())
    assert db.execute("select count(*) from v2_sample").fetchone()[0] == 6
    # a failed rerun updates the migration row, keeps the study row
    env.mp.setenv("FAKE_MIGRATE_FAIL", "GSE000001")
    assert env.run() == 1
    assert camp.main(["catalog-sql", "--rows-dir", str(env.rows), "--out", str(out)]) == 0
    for f in sorted(p.name for p in out.glob("*.sql")):
        db.executescript((out / f).read_text())
    assert db.execute("select status from v2_migration where gse_id='GSE000001'").fetchone()[0] == "migrate_failed"
    assert db.execute("select count(*) from v2_study where gse_id='GSE000001'").fetchone()[0] == 1


def test_catalog_sql_from_r2_matches_rows_dir(env):
    C.seed(env.root, 2)
    assert env.run() == 0
    a, b = env.tmp / "a", env.tmp / "b"
    assert camp.main(["catalog-sql", "--rows-dir", str(env.rows), "--out", str(a), "--no-stamp"]) == 0
    assert camp.main(["catalog-sql", "--from-r2", "--local-root", str(env.root), "--out", str(b), "--no-stamp"]) == 0
    dbs = []
    for d in (a, b):
        db = sqlite3.connect(":memory:")
        for t, spec in camp.CATALOG_SCHEMA.items():
            sql = {"text": "TEXT", "int": "INTEGER", "real": "REAL"}
            db.execute(f'CREATE TABLE "{t}" ({", ".join(chr(34) + c + chr(34) + " " + sql[k] for c, k in spec["cols"].items())}, '
                       f'"{spec["stamp"]}" TEXT, PRIMARY KEY ({", ".join(chr(34) + p + chr(34) for p in spec["pk"])}))')
        for f in sorted(d.glob("*.sql")):
            db.executescript(f.read_text())
        dbs.append(db)
    for q in ("select gsm_id, gse_id, key, bytes, sha256, n_columns, n_called, members from v2_sample order by 1",
              "select gse_id, manifest_key, rev, bytes, n_samples, legacy_etag, legacy_bytes from v2_study order by 1"):
        assert dbs[0].execute(q).fetchall() == dbs[1].execute(q).fetchall()


def test_sql_quoting():
    assert camp.sql_literal("it's") == "'it''s'" and camp.sql_literal(None) == "NULL" and camp.sql_literal(float("nan")) == "NULL"
    assert camp.sql_literal("a\x00b") == "'ab'" and camp.sql_literal(True) == "1" and camp.sql_literal(2.5) == "2.5"
    st = camp.upsert_sql("v2_study", camp.clean_row("v2_study", {"gse_id": "GSE1", "rev": "x'; DROP TABLE v2_study; --"}), None, False)
    db = _sqlite_with_schema()
    db.executescript(st)
    assert db.execute("select rev from v2_study").fetchone()[0] == "x'; DROP TABLE v2_study; --"


class _Ingest(BaseHTTPRequestHandler):
    log = []
    fail_first = False

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        _Ingest.log.append((self.path, self.headers.get("X-Ingest-Token"), self.headers.get("User-Agent"), body))
        if _Ingest.fail_first:
            _Ingest.fail_first = False
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"slow down"}')
            return
        ok = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(ok)))
        self.end_headers()
        self.wfile.write(ok)

    def log_message(self, *a):
        pass


def test_catalog_post_orders_tables_and_retries(env, monkeypatch):
    C.seed(env.root, 2)
    assert env.run() == 0
    srv = HTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Ingest.log = []
    _Ingest.fail_first = True
    monkeypatch.setattr(camp.time, "sleep", lambda s: None)
    monkeypatch.setenv("INGEST_TOKEN", "t0ken")
    try:
        assert camp.main(["catalog-post", "--rows-dir", str(env.rows), "--base-url", f"http://127.0.0.1:{srv.server_port}", "--batch", "3"]) == 0
    finally:
        srv.shutdown()
    paths = [p for p, *_ in _Ingest.log]
    assert paths[0] == "/api/ingest/v2_sample" and paths[1] == "/api/ingest/v2_sample"  # first POST got 429 and was repeated
    order = [p.rsplit("/", 1)[1] for p in paths]
    assert order.index("v2_study") > order.index("v2_migration") > order.index("v2_sample")
    assert all(tok == "t0ken" and "singlet-v2-campaign" in ua for _p, tok, ua, _b in _Ingest.log)
    rows = [r for p, _t, _u, b in _Ingest.log if p.endswith("v2_sample") for r in b["rows"]]
    assert len({r["gsm_id"] for r in rows}) == 4 and len(rows) == 4 + 3  # 4 samples; the 429'd batch of 3 was sent again
    assert all(isinstance(r["bytes"], int) and isinstance(r["members"], str) for r in rows)
    assert camp.main(["catalog-post", "--rows-dir", str(env.rows), "--dry-run"]) == 0


def test_catalog_post_needs_a_token(env, monkeypatch, capsys):
    C.seed(env.root, 1)
    assert env.run() == 0
    monkeypatch.delenv("INGEST_TOKEN", raising=False)
    assert camp.main(["catalog-post", "--rows-dir", str(env.rows)]) == 2


# ── reference index, plan, summarize ─────────────────────────────────────────────────────────────────────────────────


def test_refs_index(env):
    C.seed(env.root, 2)
    assert env.run() == 0
    assert camp.main(["refs-index", "--local-root", str(env.root)]) == 0
    assert not (env.root / PUB / "v2" / "ref" / "index.json").exists()  # no --apply
    assert camp.main(["refs-index", "--local-root", str(env.root), "--apply"]) == 0
    idx = json.loads((env.root / PUB / "v2" / "ref" / "index.json").read_text())
    assert len(idx["packs"]) == 2 and all(p["sha256"][:32] == p["id"] and p["kind"] == "vocab" for p in idx["packs"])
    assert env.store.head(PUB, "v2/ref/index.json").cache_control == "public, max-age=300"
    assert camp.main(["refs-index", "--local-root", str(env.root), "--apply"]) == 0  # the index is not listed as a pack


def test_plan_and_summarize(env, capsys):
    gses = C.seed(env.root, 3)
    assert env.run() == 0
    capsys.readouterr()
    assert camp.main(["plan", "--local-root", str(env.root), "--shards", "4", "--out", str(env.tmp / "plan.tsv")]) == 0
    out = capsys.readouterr().out
    assert "studies=3" in out and "node-h total" in out
    assert (env.tmp / "plan.tsv").read_text().splitlines()[1].startswith(gses[0])
    assert camp.main(["summarize", str(env.results)]) == 0
    out = capsys.readouterr().out
    assert "ok=3" in out and "members by kind" in out and "end to end" in out


def test_projection_is_bound_by_the_slower_side():
    p = camp.project(2.7e12, 8202, 32, rate_dl=150, rate_cpu_core=5, cores=8)
    assert p["bound"] == "compute" and p["node_mbs"] == pytest.approx(30.0)
    q = camp.project(2.7e12, 8202, 32, rate_dl=150, rate_cpu_core=50, cores=8)
    assert q["bound"] == "download" and q["node_mbs"] == 150
    assert q["task_hours"] == pytest.approx(2.7e12 / 32 / 1e6 / 150 / 3600 + 0.1)


def test_frozen_listing_gives_every_shard_the_same_studies(env):
    gses = C.seed(env.root, 4)
    assert camp.main(["plan", "--local-root", str(env.root), "--shards", "2", "--out", str(env.tmp / "plan.tsv")]) == 0
    C.seed(env.root, 6)  # two new studies appear in the bucket after the listing was frozen
    seen = []
    for i in range(2):
        if env.results.exists():
            env.results.unlink()
        assert env.run("--list-only", "--listing", str(env.tmp / "plan.tsv"), "--shard", f"{i}/2", mode="dry", base=False) == 0
        seen += sorted(env.rows_of())
    assert sorted(seen) == gses  # not the new ones, none twice
    env.results.unlink()
    assert env.run("--list-only", "--shard", "0/1", mode="dry", base=False) == 0
    assert len(env.rows_of()) == 6  # a fresh listing sees them (the sweep after the campaign)
