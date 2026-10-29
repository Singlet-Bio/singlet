# SPDX-License-Identifier: MIT
"""scripts/s1_retire.py: archive (A), the catalog flip (B, s1_campaign.py retire-sql), delete (C), restore, audit.

Directory-backed buckets, an in-memory S3 for the boto3 paths, SQLite for the catalog. The point of the three steps: the site never
offers a legacy URL that is dead, so the legacy object is deleted only after the catalog says `retired` and the caches had time to expire.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

_p = Path(__file__).with_name("test_s1_campaign_common.py")
if "test_s1_campaign_common" not in sys.modules:
    _s = importlib.util.spec_from_file_location("test_s1_campaign_common", _p)
    _m = importlib.util.module_from_spec(_s)
    sys.modules["test_s1_campaign_common"] = _m
    _s.loader.exec_module(_m)
C = sys.modules["test_s1_campaign_common"]
camp = C.load_script("s1_campaign")
ret = C.load_script("s1_retire")

PUB, ARC = "singlet-data", "singlet-archive"
NOW = dt.datetime(2026, 10, 25, 12, 0, 0, tzinfo=dt.timezone.utc)
NOW_S = "2026-10-25T12:00:00Z"


def ago(hours: float = 0, minutes: float = 0) -> str:
    return (NOW - dt.timedelta(hours=hours, minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    """Three studies migrated with the campaign (stand-in transcoder); copied per test."""
    base = tmp_path_factory.mktemp("migrated")
    root = base / "r2"
    tool = C.write_fake_tool(base)
    gses = C.seed(root, 3)
    args = [
        "run",
        "--mode",
        "full",
        "--local-root",
        str(root),
        "--scratch",
        str(base / "scratch"),
        "--shard",
        "0/1",
        "--workers",
        "2",
        "--retry-sleep",
        "0",
        "--part-size-mib",
        "0.001",
    ] + C.tool_cmds(tool)
    import os

    os.environ["FAKE_PAD"] = "2500"
    try:
        assert camp.main(args) == 0
    finally:
        os.environ.pop("FAKE_PAD", None)
    return root, gses


@pytest.fixture
def r2(migrated, tmp_path):
    src, gses = migrated
    root = tmp_path / "r2"
    shutil.copytree(src, root)

    class E:
        pass

    e = E()
    e.root, e.gses, e.tmp = root, gses, tmp_path
    e.store = camp.LocalStore(root)
    e.list = tmp_path / "archived.jsonl"
    e.legacy = lambda g: root / PUB / "data" / g / f"{g}.singlet"
    e.arch = lambda g: root / ARC / "data" / g / f"{g}.singlet"
    e.orig = {g: e.legacy(g).read_bytes() for g in gses}

    def go(*argv, store=None, local=True):
        args = list(argv) + (["--local-root", str(root)] if local else [])
        if store is not None:
            old = camp.LocalStore
            ret.vc.LocalStore = lambda r: store
            try:
                return ret.main(args)
            finally:
                ret.vc.LocalStore = old
        return ret.main(args)

    def archive(*extra, store=None, lst=None):
        return go("archive", "--all", "--apply", "--list", str(lst or e.list), *extra, store=store)

    def ages(rows, name="ages.csv"):
        p = tmp_path / name
        p.write_text(
            "gse_id,verified_at,updated_at\n" + "".join(f"{g},{v},{u}\n" for g, v, u in rows)
        )
        return p

    def delete(ages_file, *extra, store=None, lst=None, apply=True):
        tail = ["--apply", "--confirm", "RETIRE"] if apply else []
        return go(
            "delete",
            "--list",
            str(lst or e.list),
            "--ages",
            str(ages_file),
            "--now",
            NOW_S,
            *tail,
            *extra,
            store=store,
        )

    e.go, e.archive, e.ages, e.delete = go, archive, ages, delete
    return e


def tree(root):
    return sorted(
        (p.relative_to(root).as_posix(), p.stat().st_size)
        for p in Path(root).rglob("*")
        if p.is_file()
    )


def entries(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def good_ages(r2, gses=None, verified_h=30, retired_min=10):
    return r2.ages(
        [(g, ago(hours=verified_h), ago(minutes=retired_min)) for g in (gses or r2.gses)]
    )


# ── (A) archive ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_dry_run_changes_nothing(r2, capsys):
    before = tree(r2.root)
    assert r2.go("archive", "--all") == 0
    out = capsys.readouterr().out
    assert out.count("would-archive") == 3 and "does not exist; --apply would create it" in out
    assert tree(r2.root) == before and not r2.list.exists()


def test_archive_apply_needs_the_list(r2):
    before = tree(r2.root)
    assert r2.go("archive", "--all", "--apply") == 2
    assert tree(r2.root) == before


def test_archive_copies_verifies_lists_and_deletes_nothing(r2):
    store = camp.LocalStore(r2.root)
    store_before = [t for t in tree(r2.root) if t[0].startswith(f"{PUB}/store/")]
    assert r2.archive(store=store) == 0
    for g in r2.gses:
        assert r2.legacy(g).read_bytes() == r2.orig[g]  # the legacy object is still there
        assert r2.arch(g).read_bytes() == r2.orig[g]  # same key, same bytes
        meta = json.loads((r2.root / ".meta" / ARC / "data" / g / f"{g}.singlet.json").read_text())
        assert meta["storage_class"] == "STANDARD_IA"
    assert [op for op, _b, _k in store.ops if op == "delete"] == []
    lst = entries(r2.list)
    assert sorted(e["gse"] for e in lst) == r2.gses  # (written in completion order by the workers)
    e0 = sorted(lst, key=lambda e: e["gse"])[0]
    assert (
        e0["event"] == "archived"
        and e0["key"] == f"data/{e0['gse']}/{e0['gse']}.singlet"
        and e0["bytes"] == len(r2.orig[e0["gse"]])
    )
    assert (
        e0["etag"] == hashlib.md5(r2.orig[e0["gse"]]).hexdigest() == e0["archive_etag"]
        and e0["verified_by"] == "etag"
        and e0["archive_bucket"] == ARC
    )
    assert len(e0["sha256"]) == 64 and e0["at"]
    led = [
        json.loads(x)
        for x in (r2.root / PUB / "s1" / "ledger" / f"{r2.gses[0]}.jsonl").read_text().splitlines()
    ]
    assert led[-1]["event"] == "archived"
    # .s1 objects untouched (only the ledgers gained a line)
    store_after = [t for t in tree(r2.root) if t[0].startswith(f"{PUB}/store/")]
    assert {k for k, _s in set(store_before) ^ set(store_after)} <= {
        f"{PUB}/s1/ledger/{g}.jsonl" for g in r2.gses
    }


def test_archive_is_idempotent_and_reuses_a_verified_copy(r2):
    assert r2.archive() == 0
    store = camp.LocalStore(r2.root)
    assert r2.archive(store=store) == 0
    assert [op for op, _b, _k in store.ops if op == "copy"] == []
    assert (
        len(entries(r2.list)) == 6 and sorted(ret.load_list([str(r2.list)])) == r2.gses
    )  # deduplicated by the reader


def test_archive_refuses_a_study_rebuilt_after_it_was_migrated(r2, capsys):
    g = r2.gses[1]
    C.put_legacy(
        r2.root, g, ["GSM20001", "GSM20002", "GSM20003"], salt="rebuilt"
    )  # new bundle, different ETag
    assert r2.archive() == 1
    assert f"{g}\tREFUSED: legacy object changed since it was migrated" in capsys.readouterr().out
    assert not r2.arch(g).exists() and g not in [e["gse"] for e in entries(r2.list)]
    assert r2.arch(r2.gses[0]).exists()


def test_archive_refuses_without_a_manifest_missing_samples_or_a_verified_ledger(r2, capsys):
    g0, g1, g2 = r2.gses
    (r2.root / PUB / "s1" / "m" / f"{g0}.json").unlink()
    m1 = json.loads((r2.root / PUB / "s1" / "m" / f"{g1}.json").read_text())
    (r2.root / PUB / m1["samples"][0]["key"]).unlink()
    (r2.root / PUB / "s1" / "ledger" / f"{g2}.jsonl").unlink()
    assert r2.go("archive", "--gse", g0, g1, g2, "--apply", "--list", str(r2.list)) == 1
    out = capsys.readouterr().out
    assert (
        "no readable .s1 manifest" in out
        and "sample objects of the manifest are missing" in out
        and "no committed, verified migration" in out
    )
    assert not any((r2.root / ARC).rglob("*.singlet")) and not r2.list.exists()


def test_a_full_archive_waits_for_every_manifest(r2):
    (r2.root / PUB / "s1" / "m" / f"{r2.gses[2]}.json").unlink()
    with pytest.raises(SystemExit) as e:
        r2.archive()
    assert "no .s1 manifest yet" in str(e.value) and r2.gses[2] in str(e.value)
    assert not (r2.root / ARC).exists()
    assert r2.archive("--allow-partial") == 1  # the third is REFUSED, the others are archived
    assert sorted(e["gse"] for e in entries(r2.list)) == r2.gses[:2]


def test_source_changing_during_the_copy_is_not_listed(r2, capsys):
    g = r2.gses[0]

    class Racy(camp.LocalStore):
        def copy(self, *a, **kw):
            out = super().copy(*a, **kw)
            if a[1].endswith(f"{g}.singlet") and a[0] == PUB:
                self._p(PUB, a[1]).write_bytes(b"replaced meanwhile")
                self._write_meta(PUB, a[1], {"etag": "changed"})
            return out

    assert r2.go("archive", "--gse", g, "--apply", "--list", str(r2.list), store=Racy(r2.root)) == 1
    assert "changed while it was being archived" in capsys.readouterr().out
    assert not r2.list.exists()


def test_multipart_copy_is_verified_by_sha256_against_the_migrated_bundle(r2):
    store = camp.LocalStore(r2.root, copy_single_max=300, copy_part=256)
    g = r2.gses[0]
    assert len(r2.orig[g]) > 300
    assert r2.go("archive", "--gse", g, "--apply", "--list", str(r2.list), store=store) == 0
    assert r2.arch(g).read_bytes() == r2.orig[g]
    e = entries(r2.list)[0]
    assert (
        e["verified_by"] == "sha256"
        and e["multipart"] is True
        and e["archive_etag"].endswith("-" + str(-(-len(r2.orig[g]) // 256)))
    )
    assert e["archive_etag"] != e["etag"]


def test_a_corrupt_copy_is_never_listed(r2, capsys):
    class Bad(camp.LocalStore):
        def copy(self, src_bucket, src_key, dst_bucket, dst_key, **kw):
            out = super().copy(src_bucket, src_key, dst_bucket, dst_key, **kw)
            p = self._p(dst_bucket, dst_key)
            raw = p.read_bytes()
            p.write_bytes(bytes([raw[0] ^ 0xFF]) + raw[1:])  # same size, first byte wrong
            m = json.loads(self._mp(dst_bucket, dst_key).read_text())
            m["etag"] = "0" * 32
            self._write_meta(dst_bucket, dst_key, m)
            return out

    g = r2.gses[0]
    assert r2.go("archive", "--gse", g, "--apply", "--list", str(r2.list), store=Bad(r2.root)) == 1
    assert "differs from the source" in capsys.readouterr().out
    assert r2.legacy(g).exists() and not r2.list.exists()


def test_shards_keep_meaning_the_same_studies(r2):
    assert r2.go("archive", "--all", "--shard", "0/2", "--apply", "--list", str(r2.list)) == 0
    assert sorted(e["gse"] for e in entries(r2.list)) == [r2.gses[0], r2.gses[2]]
    assert r2.go("archive", "--all", "--shard", "1/2", "--apply", "--list", str(r2.list)) == 0
    assert sorted(ret.load_list([str(r2.list)])) == r2.gses and all(
        r2.legacy(g).exists() for g in r2.gses
    )


# ── (B) the catalog flip ──────────────────────────────────────────────────────────────────────────────────────────────


def _migration_db(gses, extra=()):
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE s1_migration (gse_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK (status IN ('pending','running','uploaded','verified','failed','retired')), "
        "verified_at TEXT, updated_at TEXT)"
    )
    db.executemany(
        "INSERT INTO s1_migration VALUES (?, 'verified', ?, ?)",
        [(g, ago(hours=30), ago(hours=30)) for g in gses],
    )
    db.executemany("INSERT INTO s1_migration VALUES (?, ?, ?, ?)", list(extra))
    return db


def _apply_sql(db, out):
    for n in [
        ln.split("--file=")[1]
        for ln in (out / "APPLY.txt").read_text().splitlines()
        if ln.startswith("wrangler")
    ]:
        db.executescript((out / n).read_text())


def test_retire_sql_flips_the_archived_studies_only_and_undoes_a_restore(r2, tmp_path):
    assert (
        r2.go("archive", "--gse", r2.gses[0], r2.gses[1], "--apply", "--list", str(r2.list)) == 0
    )  # the third is not archived
    out = tmp_path / "sql"
    assert camp.main(["retire-sql", "--list", str(r2.list), "--out", str(out)]) == 0
    db = _migration_db(r2.gses, extra=[("GSE999", "failed", None, None)])
    _apply_sql(db, out)
    _apply_sql(db, out)  # twice: harmless
    st = dict(db.execute("SELECT gse_id, status FROM s1_migration"))
    assert (
        st[r2.gses[0]] == st[r2.gses[1]] == "retired"
        and st[r2.gses[2]] == "verified"
        and st["GSE999"] == "failed"
    )
    assert "6 D1 row writes" in (out / "APPLY.txt").read_text()
    # a restore later: the status goes back to verified, the other stays retired
    restored = tmp_path / "restored.jsonl"
    assert (
        r2.go(
            "delete",
            "--list",
            str(r2.list),
            "--ages",
            str(good_ages(r2)),
            "--now",
            NOW_S,
            "--apply",
            "--confirm",
            "RETIRE",
        )
        == 0
    )  # delete the two
    assert r2.go("restore", "--gse", r2.gses[1], "--apply", "--list", str(restored)) == 0
    assert camp.main(["retire-sql", "--list", str(r2.list), str(restored), "--out", str(out)]) == 0
    _apply_sql(db, out)
    st = dict(db.execute("SELECT gse_id, status FROM s1_migration"))
    assert st[r2.gses[1]] == "verified" and st[r2.gses[0]] == "retired"
    assert entries(restored)[0]["event"] == "restored"


# ── (C) delete ────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_delete_is_a_dry_run_until_applied_and_confirmed(r2, capsys):
    assert r2.archive() == 0
    capsys.readouterr()
    before = tree(r2.root)
    assert r2.delete(good_ages(r2), apply=False) == 0
    out = capsys.readouterr().out
    assert out.count("would-delete") == 3 and "verified 30.0 h, retired 10.0 min ago" in out
    assert tree(r2.root) == before
    assert (
        r2.go(
            "delete",
            "--list",
            str(r2.list),
            "--ages",
            str(good_ages(r2)),
            "--now",
            NOW_S,
            "--apply",
        )
        == 2
    )  # no --confirm
    assert (
        r2.go(
            "delete",
            "--list",
            str(r2.list),
            "--ages",
            str(good_ages(r2)),
            "--now",
            NOW_S,
            "--apply",
            "--confirm",
            "no",
        )
        == 2
    )
    assert (
        r2.go("delete", "--ages", str(good_ages(r2))) == 2
        and r2.go("delete", "--list", str(r2.list)) == 2
    )  # both inputs are required
    assert tree(r2.root) == before and all(r2.legacy(g).exists() for g in r2.gses)


def test_delete_waits_for_the_catalog_and_for_both_ages(r2, capsys):
    assert r2.archive() == 0
    g0, g1, g2 = r2.gses
    capsys.readouterr()
    ages = r2.ages(
        [
            (g1, ago(hours=23.9), ago(minutes=60)),  # verified too recently
            (g2, ago(hours=48), ago(minutes=4.9)),
        ]
    )  # retired too recently; g0 is not in the file at all
    assert r2.delete(ages) == 0  # waiting is not a failure
    out = capsys.readouterr().out
    assert f"{g0}\tNOT-YET: not 'retired' in the catalog yet" in out
    assert f"{g1}\tNOT-YET: verified 23.9 h ago, needs 24 h" in out
    assert f"{g2}\tNOT-YET: retired 4.9 min ago, needs 5 min" in out
    assert all(r2.legacy(g).exists() for g in r2.gses)
    # exactly at the thresholds it goes through (>=), and the thresholds are options
    ages = r2.ages(
        [
            (g0, ago(hours=24), ago(minutes=5)),
            (g1, ago(hours=1), ago(minutes=1)),
            (g2, ago(hours=1), ago(minutes=1)),
        ]
    )
    assert r2.delete(ages, "--min-verified-age-hours", "1", "--min-retired-age-minutes", "1") == 0
    assert not any(r2.legacy(g).exists() for g in r2.gses)


def test_delete_removes_only_archived_retired_studies_and_keeps_everything_else(r2):
    assert (
        r2.go("archive", "--gse", r2.gses[0], r2.gses[1], "--apply", "--list", str(r2.list)) == 0
    )  # the third is not in the list
    store = camp.LocalStore(r2.root)
    ages = good_ages(r2)  # the catalog says all three are retired, but only two are archived
    before_store = [t for t in tree(r2.root) if t[0].startswith(f"{PUB}/store/")]
    assert r2.delete(ages, store=store) == 0
    assert (
        not r2.legacy(r2.gses[0]).exists()
        and not r2.legacy(r2.gses[1]).exists()
        and r2.legacy(r2.gses[2]).exists()
    )
    assert sorted(k for op, _b, k in store.ops if op == "delete") == sorted(
        f"data/{g}/{g}.singlet" for g in r2.gses[:2]
    )
    assert all(r2.arch(g).read_bytes() == r2.orig[g] for g in r2.gses[:2])  # the archive keeps them
    after_store = [t for t in tree(r2.root) if t[0].startswith(f"{PUB}/store/")]
    assert {k for k, _s in set(before_store) ^ set(after_store)} <= {
        f"{PUB}/s1/ledger/{g}.jsonl" for g in r2.gses
    }  # .s1 objects untouched
    led = [
        json.loads(x)
        for x in (r2.root / PUB / "s1" / "ledger" / f"{r2.gses[0]}.jsonl").read_text().splitlines()
    ]
    assert led[-1]["event"] == "legacy-deleted" and led[-1]["verified_age_h"] == 30.0
    assert r2.delete(ages) == 0  # again: nothing left to do
    assert r2.go("audit") == 0


def test_delete_checks_the_archive_the_legacy_object_and_the_store_objects_again(r2, capsys):
    assert r2.archive() == 0
    g0, g1, g2 = r2.gses
    r2.arch(g0).write_bytes(b"x" * len(r2.orig[g0]))  # the archive copy was altered (same size)
    meta = r2.root / ".meta" / ARC / "data" / g0 / f"{g0}.singlet.json"
    m = json.loads(meta.read_text())
    m["etag"] = "f" * 32
    meta.write_text(json.dumps(m))
    C.put_legacy(
        r2.root, g1, ["GSM20001", "GSM20002", "GSM20003"], salt="rebuilt"
    )  # the legacy object changed after it was archived
    (r2.root / PUB / "s1" / "m" / f"{g2}.json").unlink()  # the .s1 manifest of the third is gone
    capsys.readouterr()
    assert r2.delete(good_ages(r2)) == 1
    out = capsys.readouterr().out
    assert "the archive copy is not the one in the list" in out
    assert "not the one that was archived" in out
    assert "no readable .s1 manifest" in out
    assert all(r2.legacy(g).exists() for g in r2.gses)
    # an archive copy that is gone
    shutil.rmtree(r2.root / ARC)
    assert r2.delete(good_ages(r2, [g1])) == 1 and "no archive copy" in capsys.readouterr().out


def test_delete_refuses_when_a_sample_object_of_store_is_missing(r2, capsys):
    assert r2.archive() == 0
    m = json.loads((r2.root / PUB / "s1" / "m" / f"{r2.gses[0]}.json").read_text())
    (r2.root / PUB / m["samples"][0]["key"]).unlink()
    capsys.readouterr()
    assert r2.delete(good_ages(r2)) == 1
    assert (
        "sample objects of the manifest are missing" in capsys.readouterr().out
        and r2.legacy(r2.gses[0]).exists()
    )


def test_a_restore_after_the_archive_cancels_the_delete(r2, capsys):
    assert r2.archive() == 0
    g = r2.gses[0]
    assert (
        r2.go("restore", "--gse", g, "--apply", "--list", str(r2.list)) == 0
    )  # appended after the archived line
    capsys.readouterr()
    assert r2.delete(good_ages(r2)) == 0
    assert (
        f"{g}\tNOT-YET: restored after it was archived" in capsys.readouterr().out
        and r2.legacy(g).exists()
    )
    assert not r2.legacy(r2.gses[1]).exists()


def test_delete_input_may_be_a_bare_list_of_studies(r2, tmp_path):
    assert r2.archive() == 0
    bare = tmp_path / "bare.txt"
    bare.write_text("# archived studies\n" + "\n".join(r2.gses[:2]) + "\n")
    assert (
        r2.delete(good_ages(r2), lst=bare) == 0
    )  # no etags in the list: the copy is verified from scratch
    assert (
        not r2.legacy(r2.gses[0]).exists()
        and not r2.legacy(r2.gses[1]).exists()
        and r2.legacy(r2.gses[2]).exists()
    )


def test_ages_file_formats(tmp_path):
    rows = [
        ("GSE1", "2026-10-24T08:00:00Z", "2026-10-25T11:00:00Z"),
        ("GSE2", "2026-10-24 08:00:00", "2026-10-25 11:00:00"),
    ]
    csvf = tmp_path / "a.csv"
    csvf.write_text("gse_id,verified_at,updated_at\n" + "\n".join(",".join(r) for r in rows) + "\n")
    js = tmp_path / "a.json"
    js.write_text(json.dumps([dict(gse_id=g, verified_at=v, updated_at=u) for g, v, u in rows]))
    wr = tmp_path / "w.json"  # wrangler d1 execute --json
    wr.write_text(
        json.dumps(
            [
                {
                    "results": [dict(gse_id=g, verified_at=v, updated_at=u) for g, v, u in rows],
                    "success": True,
                    "meta": {"changes": 0},
                }
            ]
        )
    )
    st = tmp_path / "s.csv"  # an export that carries the status: only the retired rows count
    st.write_text(
        "gse_id,status,verified_at,updated_at\nGSE1,retired,2026-10-24T08:00:00Z,2026-10-25T11:00:00Z\nGSE2,verified,2026-10-24T08:00:00Z,2026-10-25T11:00:00Z\n"
    )
    for p in (csvf, js, wr):
        ages = ret.load_ages(str(p))
        assert sorted(ages) == ["GSE1", "GSE2"]
        assert (
            ages["GSE1"][0]
            == dt.datetime(2026, 10, 24, 8, tzinfo=dt.timezone.utc)
            == ages["GSE2"][0]
        )
        assert ages["GSE2"][1] == dt.datetime(2026, 10, 25, 11, tzinfo=dt.timezone.utc)
    assert sorted(ret.load_ages(str(st))) == ["GSE1"]
    assert ret.parse_ts("2026-10-24T08:00:00+02:00") == dt.datetime(
        2026, 10, 24, 6, tzinfo=dt.timezone.utc
    )
    assert (
        ret.parse_ts("") is None and ret.parse_ts("garbage") is None and ret.parse_ts(None) is None
    )


def test_missing_timestamps_in_the_ages_file_are_refused(r2, capsys):
    assert r2.archive() == 0
    p = r2.tmp / "bad.csv"
    p.write_text("gse_id,verified_at,updated_at\n" + f"{r2.gses[0]},,{ago(minutes=30)}\n")
    capsys.readouterr()
    assert (
        r2.delete(p) == 1
        and "no verified_at / updated_at" in capsys.readouterr().out
        and r2.legacy(r2.gses[0]).exists()
    )


def test_the_three_steps_in_order_with_the_catalog_in_sqlite(r2, tmp_path, capsys):
    """A, then B applied to SQLite, then C with the ages read back from SQLite: nothing is deleted before the flip and the waits."""
    assert r2.archive() == 0
    db = _migration_db(r2.gses)

    def export():
        return [
            (g, v, u)
            for g, v, u in db.execute(
                "SELECT gse_id, verified_at, updated_at FROM s1_migration WHERE status = 'retired'"
            )
        ]

    assert (
        export() == [] and r2.delete(r2.ages(export(), "none.csv"), apply=True) == 0
    )  # (C) before (B): everything NOT-YET
    assert all(r2.legacy(g).exists() for g in r2.gses)
    out = tmp_path / "sql"
    assert camp.main(["retire-sql", "--list", str(r2.list), "--out", str(out)]) == 0
    _apply_sql(db, out)
    db.execute("UPDATE s1_migration SET updated_at = ?", (ago(minutes=2),))  # flipped 2 minutes ago
    capsys.readouterr()
    assert (
        r2.delete(r2.ages(export(), "t2.csv")) == 0
        and "retired 2.0 min ago, needs 5 min" in capsys.readouterr().out
    )
    assert all(r2.legacy(g).exists() for g in r2.gses)
    db.execute("UPDATE s1_migration SET updated_at = ?", (ago(minutes=6),))
    assert r2.delete(r2.ages(export(), "t6.csv")) == 0  # verified 30 h ago, retired 6 min ago
    assert not any(r2.legacy(g).exists() for g in r2.gses) and all(
        r2.arch(g).exists() for g in r2.gses
    )


# ── restore, audit ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_restore_puts_back_identical_bytes(r2, capsys):
    assert r2.archive() == 0
    assert r2.delete(good_ages(r2)) == 0
    capsys.readouterr()
    assert r2.go("restore", "--all") == 0  # dry run
    assert (
        capsys.readouterr().out.count("would-restore") == 3 and not r2.legacy(r2.gses[0]).exists()
    )
    restored = r2.tmp / "restored.jsonl"
    assert (
        r2.go(
            "restore",
            "--all",
            "--apply",
            "--list",
            str(restored),
            "--ledger",
            str(r2.tmp / "l.jsonl"),
        )
        == 0
    )
    for g in r2.gses:
        assert r2.legacy(g).read_bytes() == r2.orig[g]
        assert r2.arch(g).exists()  # the archive copy stays until the grace period ends
    assert [e["event"] for e in entries(restored)] == ["restored"] * 3
    capsys.readouterr()
    assert r2.go("restore", "--all", "--apply", "--list", str(restored)) == 0
    assert capsys.readouterr().out.count("already-present") == 3
    led = [
        json.loads(x)
        for x in (r2.root / PUB / "s1" / "ledger" / f"{r2.gses[0]}.jsonl").read_text().splitlines()
    ]
    assert [
        e["event"] for e in led if e["event"] in ("archived", "legacy-deleted", "restored")
    ] == ["archived", "legacy-deleted", "restored"]


def test_restore_does_not_overwrite_a_different_object_unless_forced(r2, capsys):
    g = r2.gses[0]
    assert r2.go("archive", "--gse", g, "--apply", "--list", str(r2.list)) == 0
    C.put_legacy(r2.root, g, ["GSM1"], salt="newer")
    newer = r2.legacy(g).read_bytes()
    assert r2.go("restore", "--gse", g, "--apply") == 1
    assert r2.legacy(g).read_bytes() == newer and "SKIPPED" in capsys.readouterr().out
    assert r2.go("restore", "--gse", g, "--apply", "--force") == 0
    assert r2.legacy(g).read_bytes() == r2.orig[g]


def test_audit_lists_what_remains(r2, capsys):
    assert r2.go("audit") == 0
    out = capsys.readouterr().out
    assert out.count("\t1\t0") == 3 and "3 legacy bundles" in out
    assert r2.go("audit", "--expect-none") == 1
    assert r2.archive() == 0
    assert r2.delete(good_ages(r2, [r2.gses[0]])) == 0
    capsys.readouterr()
    assert r2.go("audit") == 0
    out = capsys.readouterr().out
    assert "2 legacy bundles" in out and "3 archived" in out
    assert r2.delete(good_ages(r2)) == 0
    capsys.readouterr()
    assert r2.go("audit", "--expect-none") == 0
    assert "0 legacy bundles" in capsys.readouterr().out
    (r2.root / PUB / "data" / "GSE9").mkdir(parents=True)
    (r2.root / PUB / "data" / "GSE9" / "stray.singlet").write_bytes(b"x")
    assert r2.go("audit", "--expect-none") == 1  # a stray .singlet key is reported too


# ── S3Store.copy (boto3 code path) ────────────────────────────────────────────────────────────────────────────────────


def _fake_with_object(size=1000):
    fake = C.FakeS3()
    data = bytes(range(256)) * (size // 256 + 1)
    fake.put_raw(
        PUB, "data/GSE1/GSE1.singlet", data[:size], ContentType="application/zip", CacheControl="x"
    )
    fake.buckets.add(ARC)
    return fake, data[:size]


def test_s3_single_copy_keeps_etag_and_sets_storage_class():
    fake, data = _fake_with_object()
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    etag, mp = store.copy(
        PUB,
        "data/GSE1/GSE1.singlet",
        ARC,
        "data/GSE1/GSE1.singlet",
        size=len(data),
        storage_class="STANDARD_IA",
    )
    assert not mp and etag == hashlib.md5(data).hexdigest()
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["StorageClass"] == "STANDARD_IA"
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["ContentType"] == "application/zip"


def test_s3_copy_falls_back_when_the_storage_class_is_not_accepted():
    fake, data = _fake_with_object()
    fake.reject_storage_class = True
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    store.copy(
        PUB,
        "data/GSE1/GSE1.singlet",
        ARC,
        "data/GSE1/GSE1.singlet",
        size=len(data),
        storage_class="STANDARD_IA",
    )
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["StorageClass"] is None
    assert [c for c in fake.calls if c[0] == "copy_object"] == [
        ("copy_object", ARC, "data/GSE1/GSE1.singlet", "STANDARD_IA"),
        ("copy_object", ARC, "data/GSE1/GSE1.singlet", None),
    ]


def test_s3_multipart_copy_uses_fixed_ranges_and_aborts_on_failure():
    fake, data = _fake_with_object(1000)
    store = camp.S3Store(client=fake, transfer_config=C.TC, copy_single_max=300, copy_part=256)
    etag, mp = store.copy(
        PUB,
        "data/GSE1/GSE1.singlet",
        ARC,
        "data/GSE1/GSE1.singlet",
        size=1000,
        storage_class="STANDARD_IA",
    )
    assert mp and etag.endswith("-4")
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["data"] == data
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["ContentType"] == "application/zip"
    assert sorted(c[3] for c in fake.calls if c[0] == "upload_part_copy") == [1, 2, 3, 4]
    fake2, _ = _fake_with_object(1000)
    orig = fake2.upload_part_copy

    def boom(**kw):
        if kw["PartNumber"] == 3:
            raise C.FakeClientError("InternalError", 500)
        return orig(**kw)

    fake2.upload_part_copy = boom
    import time as _t

    real_sleep = _t.sleep
    _t.sleep = lambda s: None
    try:
        with pytest.raises(C.FakeClientError):
            camp.S3Store(
                client=fake2, transfer_config=C.TC, copy_single_max=300, copy_part=256
            ).copy(PUB, "data/GSE1/GSE1.singlet", ARC, "data/GSE1/GSE1.singlet", size=1000)
    finally:
        _t.sleep = real_sleep
    assert (
        any(c[0] == "abort_mpu" for c in fake2.calls)
        and (ARC, "data/GSE1/GSE1.singlet") not in fake2.objs
    )


def test_the_whole_sequence_on_the_s3_store_creates_the_archive_bucket(
    migrated, tmp_path, monkeypatch
):
    """A and C through S3Store + FakeS3, from a campaign-written tree loaded into the fake."""
    src, gses = migrated
    fake = C.FakeS3()
    fake.buckets.add(PUB)
    root = src / PUB
    for p in sorted(root.rglob("*")):
        if p.is_file():
            key = p.relative_to(root).as_posix()
            meta_p = src / ".meta" / PUB / (key + ".json")
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            fake.put_raw(
                PUB, key, p.read_bytes(), etag=meta.get("etag"), Metadata=meta.get("metadata", {})
            )
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    monkeypatch.setattr(ret.vc, "S3Store", lambda: store)
    lst = tmp_path / "archived.jsonl"
    assert ARC not in fake.buckets
    assert ret.main(["archive", "--all", "--apply", "--list", str(lst), "--workers", "2"]) == 0
    assert ARC in fake.buckets and ("create_bucket", ARC) in fake.calls
    assert not any(c[0] == "delete" for c in fake.calls)  # step A deletes nothing
    for g in gses:
        assert (PUB, f"data/{g}/{g}.singlet") in fake.objs
        assert fake.objs[(ARC, f"data/{g}/{g}.singlet")]["StorageClass"] == "STANDARD_IA"
    ages = tmp_path / "ages.json"
    ages.write_text(
        json.dumps(
            [
                {"gse_id": g, "verified_at": ago(hours=30), "updated_at": ago(minutes=10)}
                for g in gses
            ]
        )
    )
    assert (
        ret.main(
            [
                "delete",
                "--list",
                str(lst),
                "--ages",
                str(ages),
                "--now",
                NOW_S,
                "--apply",
                "--confirm",
                "RETIRE",
            ]
        )
        == 0
    )
    assert sorted(c[2] for c in fake.calls if c[0] == "delete") == sorted(
        f"data/{g}/{g}.singlet" for g in gses
    )  # only legacy keys
    assert ret.main(["audit", "--expect-none"]) == 0
