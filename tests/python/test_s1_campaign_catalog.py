# SPDX-License-Identifier: MIT
"""The catalog side of the campaign: rows for D1 schema 032/033, the SQL files (applied here in SQLite), the day budget, the order of
application, the POST bodies, the reference packs, and the whole chain with the real transcoder on two samples of the corpus bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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

PUB = "singlet-data"
REPO = Path(__file__).resolve().parents[2]
WEBSITE = Path(
    os.environ.get("SINGLET_WEBSITE_DIR", "C:/Users/zacha/Projects/singlet-website")
)  # a singlet-website checkout: used when present (drift checks), never required

# Schema 032 + 033 as written by W5 (schema/032_v2_catalog.sql, schema/033_v2_sample_rev_key.sql): the final shape of the three tables.
DDL_033 = """
CREATE TABLE s1_study (gse_id TEXT PRIMARY KEY, manifest_key TEXT NOT NULL, rev TEXT NOT NULL, bytes INTEGER, n_samples INTEGER NOT NULL,
  created_at TEXT, legacy_bytes INTEGER, manifest_bytes INTEGER, manifest_sha256 TEXT, n_cells_called INTEGER, reference_build TEXT,
  packs_json TEXT, pipeline_json TEXT, updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')));
CREATE TABLE s1_sample (gse_id TEXT NOT NULL, gsm_id TEXT NOT NULL, key TEXT NOT NULL, bytes INTEGER, sha256 TEXT, n_columns INTEGER,
  n_called INTEGER, n_members INTEGER, median_umi REAL, median_genes REAL, modalities TEXT, members_json TEXT, cell_stats_json TEXT,
  rev TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), PRIMARY KEY (gse_id, rev, gsm_id));
CREATE INDEX idx_s1_sample_gsm ON s1_sample (gsm_id);
CREATE TABLE s1_migration (gse_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'uploaded', 'verified', 'failed', 'retired')),
  legacy_key TEXT, legacy_etag TEXT, legacy_bytes INTEGER, s1_bytes INTEGER, n_samples INTEGER, members_by_kind TEXT, verify_ok INTEGER,
  commit_sha TEXT, migrated_at TEXT, verified_at TEXT, ledger_key TEXT, error TEXT,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')));
CREATE INDEX idx_s1_migration_status ON s1_migration (status);
"""
DDL_032 = DDL_033.replace("PRIMARY KEY (gse_id, rev, gsm_id)", "PRIMARY KEY (gse_id, gsm_id)")


def website_ddl(which: str) -> str:
    d = WEBSITE / "schema"
    text = (d / "032_v2_catalog.sql").read_text(encoding="utf-8")
    if which == "033":
        text += (d / "033_v2_sample_rev_key.sql").read_text(encoding="utf-8")
        text += (d / "036_s1_rename.sql").read_text(encoding="utf-8")
    return text


def new_db(which: str = "033") -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(DDL_033 if which == "033" else DDL_032)
    return db


def apply_files(db, day_dir: Path, only=None):
    names = [
        ln.split("--file=")[1]
        for ln in (day_dir / "APPLY.txt").read_text().splitlines()
        if ln.startswith("wrangler")
    ]
    for n in names:
        if only is None or only(n):
            db.executescript((day_dir.parent / n).read_text(encoding="utf-8"))
    return names


def dump(db, table):
    return [
        dict(r)
        for r in db.execute(
            f"SELECT * FROM {table} ORDER BY 1, 2, 3"
            if table == "s1_sample"
            else f"SELECT * FROM {table} ORDER BY 1"
        )
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return C.build_env(tmp_path, monkeypatch, camp)


def synth(gse, n, rev="1" * 16, extra=None, status="ok", build=True):
    """Catalog rows from a synthetic manifest of n samples (no transcoder)."""
    samples = []
    for i in range(n):
        gsm = f"GSM{gse[3:]}{i:03d}"
        h = hashlib.sha256(f"{gse}{i}".encode()).hexdigest()
        members = {
            "exon_counts.1pz": {
                "kind": 1,
                "off": 4096,
                "len": 500,
                "digest": "c" * 32,
                "chunk_cols": 256,
                "chunk_index": [100, 64],
                "nnz": 5,
            },
            "cells": {"kind": 2, "off": 64, "len": 100},
            "summary.json": {"kind": 3, "off": 900, "len": 9},
        }
        members.update(extra or {})
        samples.append(
            {
                "gsm": gsm,
                "key": f"s1/s/{gsm}/{h[:32]}.s1",
                "bytes": 1000 + i,
                "sha256": h,
                "n_columns": 50,
                "n_called": 7,
                "members": members,
                "stats": {"median_umi": 1500.5, "median_genes": 700},
            }
        )
    man = {
        "format": "singlet-s1",
        "manifest_version": 1,
        "gse": gse,
        "rev": rev,
        "created_at": "2026-10-22T00:00:00Z",
        "reference_build": "GRCh38-2024-A",
        "pipeline": {"v": 1},
        "packs": {"exon": "a" * 32, "whitelist": None},
        "samples": samples,
        "legacy": {
            "key": f"data/{gse}/{gse}.singlet",
            "etag": "abc-3",
            "bytes": 5000,
            "sha256": "d" * 64,
            "migrated_at": "2026-10-22T00:00:00Z",
        },
    }
    row = {
        "gse": gse,
        "status": status,
        "at": "2026-10-22T01:00:00Z",
        "verified_at": "2026-10-22T01:00:00Z",
        "commit": "deadbeef",
        "legacy_key": man["legacy"]["key"],
        "legacy_etag": "abc-3",
        "legacy_bytes": 5000,
        "s1_bytes": 9999,
        "n_samples": n,
        "members_by_kind": {"pz2": n, "cells": n, "stored": n},
        "verify": {"ok": True, "rc": 0, "secs": 1.5},
        "manifest_bytes": 1234,
        "manifest_sha256": "e" * 64,
    }
    return man, row, (camp.build_catalog_rows(man, row) if build else None)


# ── the rows ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_rows_follow_the_ingest_code_for_columns_and_types():
    man, _row, rows = synth("GSE1", 2)
    s = rows["s1_sample"][0]
    assert list(s) == list(camp.SAMPLE_COLS)
    assert (
        s["n_members"] == 3
        and s["median_umi"] == 1500.5
        and s["median_genes"] == 700.0
        and isinstance(s["median_genes"], float)
    )
    assert json.loads(s["modalities"]) == [
        "exon_counts",
        "summary",
    ]  # registry order; `cells` is not a modality
    assert (
        json.loads(s["members_json"]) == man["samples"][0]["members"]
        and " " not in s["members_json"]
    )  # JSON.stringify form
    assert json.loads(s["cell_stats_json"]) == {"median_umi": 1500.5, "median_genes": 700}
    st = rows["s1_study"]
    assert (
        list(st) == list(camp.STUDY_COLS)
        and st["bytes"] == 1000 + 1001
        and st["n_samples"] == 2
        and st["n_cells_called"] == 14
    )
    assert (
        st["legacy_bytes"] == 5000
        and st["manifest_key"] == "s1/m/GSE1.json"
        and json.loads(st["packs_json"]) == {"exon": "a" * 32, "whitelist": None}
    )
    mg = rows["s1_migration"]
    assert (
        list(mg) == list(camp.MIGRATION_COLS)
        and mg["status"] == "verified"
        and mg["verify_ok"] == 1
        and mg["error"] is None
    )
    assert json.loads(mg["members_by_kind"]) == {
        "matrix": 2,
        "cells": 2,
        "stored": 2,
    }  # the catalog's names for the treatments
    assert (
        mg["verified_at"] == "2026-10-22T01:00:00Z" and mg["ledger_key"] == "s1/ledger/GSE1.jsonl"
    )


def test_a_study_that_never_had_a_legacy_file_is_posted_as_retired():
    man, row, _ = synth("GSE7", 2, build=False)
    man["legacy"] = None
    for k in ("legacy_key", "legacy_etag", "legacy_bytes", "legacy_sha256"):
        row.pop(k, None)
    rows = camp.build_catalog_rows(man, row)
    mg = rows["s1_migration"]
    assert (
        mg["status"] == "retired"
        and mg["verify_ok"] == 1
        and mg["legacy_key"] is None
        and mg["legacy_etag"] is None
    )
    assert rows["s1_study"]["legacy_bytes"] is None
    # a migrated study is verified until the retirement flips it
    assert synth("GSE8", 1)[2]["s1_migration"]["status"] == "verified"
    # in the SQL it is applied directly as retired (the site then never offers a legacy file for it)
    db = new_db()
    for st in camp.study_statements(rows)["sample"] + camp.study_statements(rows)["flip"]:
        db.executescript(st)
    assert db.execute("select status, verify_ok from s1_migration").fetchone()[:] == (
        "retired",
        1,
    )
    body = camp.post_payloads(man, rows)[-1]
    assert body["migration"]["status"] == "retired" and body["migration"]["verify_ok"] is True


def test_a_failed_study_has_only_a_failed_migration_row():
    rows = camp.build_catalog_rows(
        None,
        {
            "gse": "GSE9",
            "status": "verify_failed",
            "error": "verify exited 1" + "x" * 3000,
            "legacy_key": "data/GSE9/GSE9.singlet",
            "legacy_etag": "e",
            "legacy_bytes": 5,
            "verify": {"ok": False, "rc": 1},
            "at": "t",
        },
    )
    assert set(rows) == {"s1_migration"}
    m = rows["s1_migration"]
    assert (
        m["status"] == "failed"
        and m["verify_ok"] == 0
        and len(m["error"]) == 2000
        and m["verified_at"] is None
    )


def test_restricted_members_and_bad_samples_are_refused_like_the_endpoint_does():
    man, row, _ = synth(
        "GSE1", 1, extra={"donor/snp_ad.1pz": {"kind": 3, "off": 1, "len": 1}}, build=False
    )
    with pytest.raises(camp.CatalogError, match="restricted"):
        camp.build_catalog_rows(man, row)
    man, row, _ = synth("GSE1", 1, build=False)
    man["samples"][0]["members"]["x"] = {"kind": 9, "off": 0, "len": 0}
    with pytest.raises(camp.CatalogError, match="integer kind"):
        camp.build_catalog_rows(man, row)
    man, row, _ = synth("GSE1", 1, build=False)
    man["samples"][0]["key"] = "s1/s/GSM999/" + "a" * 32 + ".s1"
    with pytest.raises(camp.CatalogError, match="key must be"):
        camp.build_catalog_rows(man, row)
    man, row, _ = synth("GSE1", 1, build=False)
    man["rev"] = "xyz"
    with pytest.raises(camp.CatalogError, match="16-hex rev"):
        camp.build_catalog_rows(man, row)


def test_modality_registry_matches_python_and_typescript():
    mine = {n: tuple(m) for n, m in camp.MODALITY_MEMBERS}
    sys.path.insert(0, str(REPO / "python"))
    from singlet import bundle  # noqa: E402
    from singlet._restricted import is_restricted_member

    # the Python registry also lists the per-donor layers (ase_counts, donor_snp_*, mt_events); the catalog registry does not
    py = {
        n: tuple(m.members)
        for n, m in bundle.MODALITIES.items()
        if not any(is_restricted_member(x) for x in m.members)
    }
    assert mine == py and [n for n, _ in camp.MODALITY_MEMBERS] == [
        n for n in bundle.MODALITIES if n in py
    ]
    ts = WEBSITE / "functions" / "_shared" / "modalities.ts"
    if not ts.exists():
        pytest.skip("website worktree not present")
    import re

    found = re.findall(
        r'M\("(\w+)", "[^"]+", "[^"]+", \[([^\]]*)\]', ts.read_text(encoding="utf-8")
    )
    web = {n: tuple(re.findall(r'"([^"]+)"', members)) for n, members in found}
    assert mine == web and [n for n, _ in found] == [n for n, _ in camp.MODALITY_MEMBERS]


def test_embedded_ddl_matches_the_website_schema():
    if not (WEBSITE / "schema" / "036_s1_rename.sql").exists():
        pytest.skip("website checkout with schema 036 not present")
    for which, mine in (("033", DDL_033),):
        a, b = new_db(which), sqlite3.connect(":memory:")
        b.executescript(website_ddl(which))
        for t in ("s1_study", "s1_sample", "s1_migration"):

            def cols(db):
                return [(r[1], r[2], r[3], r[5]) for r in db.execute(f"PRAGMA table_info({t})")]

            assert cols(a) == cols(b), t


# ── the SQL ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_sql_quoting():
    assert (
        camp.sql_literal("it's") == "'it''s'"
        and camp.sql_literal(None) == "NULL"
        and camp.sql_literal(float("nan")) == "NULL"
    )
    assert (
        camp.sql_literal("a\x00b") == "'ab'"
        and camp.sql_literal(True) == "1"
        and camp.sql_literal(2.5) == "2.5"
    )
    man, row, rows = synth("GSE1", 1)
    rows["s1_study"]["reference_build"] = "x'; DROP TABLE s1_study; --"
    db = new_db()
    for st in camp.study_statements(rows)["sample"] + camp.study_statements(rows)["flip"]:
        db.executescript(st)
    assert (
        db.execute("select reference_build from s1_study").fetchone()[0]
        == "x'; DROP TABLE s1_study; --"
    )


def test_apply_order_study_turns_verified_only_after_its_samples(tmp_path):
    rows = {g: synth(g, n)[2] for g, n in (("GSE1", 3), ("GSE2", 2))}
    out = tmp_path / "sql"
    plan = camp.write_sql_files(
        rows, out, day_budget=10_000, max_rows=4, max_bytes=10**7, database="singlet-catalog"
    )
    assert len(plan["days"]) == 1
    day = out / "day01"
    names = [
        ln.split("--file=")[1]
        for ln in (day / "APPLY.txt").read_text().splitlines()
        if ln.startswith("wrangler")
    ]
    kinds = [n.split("/")[1].split("_")[1] for n in names]
    assert kinds == sorted(kinds, key=["sample", "flip"].index) and set(kinds) == {
        "sample",
        "flip",
    }  # every sample file before any flip file
    db = new_db()
    for n in names:
        db.executescript((out / n).read_text(encoding="utf-8"))
        if n.split("/")[1].startswith("1_sample"):
            # while samples load: no study row, no migration row, so nothing is served from the .s1 objects
            assert (
                db.execute("select count(*) from s1_study").fetchone()[0] == 0
                and db.execute("select count(*) from s1_migration").fetchone()[0] == 0
            )
    assert (
        dump(db, "s1_migration")[0]["status"] == "verified"
        and db.execute("select count(*) from s1_sample").fetchone()[0] == 5
    )
    # a study's three flip statements stay in one file: study upsert, delete of other revs, migration upsert
    for n in names:
        if "2_flip" in n:
            stmts = [s for s in (out / n).read_text(encoding="utf-8").splitlines() if s]
            assert len(stmts) % 3 == 0
            for i in range(0, len(stmts), 3):
                assert (
                    stmts[i].startswith("INSERT INTO s1_study")
                    and stmts[i + 1].startswith("DELETE FROM s1_sample")
                    and stmts[i + 2].startswith("INSERT INTO s1_migration")
                )


def test_the_flip_is_applied_with_the_file_not_before_even_when_the_study_has_many_sample_files(
    tmp_path,
):
    rows = {"GSE1": synth("GSE1", 9)[2]}
    out = tmp_path / "sql"
    camp.write_sql_files(rows, out, day_budget=10_000, max_rows=2, max_bytes=10**7, database="d")
    day = out / "day01"
    names = [
        ln.split("--file=")[1].split("/")[1]
        for ln in (day / "APPLY.txt").read_text().splitlines()
        if ln.startswith("wrangler")
    ]
    assert names == [f"1_sample_{i:04d}.sql" for i in range(5)] + ["2_flip_0000.sql"]


def test_sql_is_idempotent_and_a_new_rev_replaces_the_old_one(tmp_path):
    out = tmp_path / "sql"
    db = new_db()
    for rev, n in (("1" * 16, 3), ("2" * 16, 2)):
        rows = {"GSE1": synth("GSE1", n, rev=rev)[2]}
        camp.write_sql_files(
            rows, out, day_budget=10_000, max_rows=100, max_bytes=10**7, database="d"
        )
        apply_files(db, out / "day01")
        apply_files(db, out / "day01")  # twice
        got = dump(db, "s1_sample")
        assert len(got) == n and {r["rev"] for r in got} == {
            rev
        }  # the older rev's rows are gone after the flip
        assert dump(db, "s1_study")[0]["rev"] == rev and dump(db, "s1_study")[0]["n_samples"] == n
    # during the load of rev 2 the rows of rev 1 were still there beside them (033): check the intermediate state
    db = new_db()
    for rev, n in (("1" * 16, 3), ("2" * 16, 2)):
        rows = {"GSE1": synth("GSE1", n, rev=rev)[2]}
        camp.write_sql_files(
            rows, out, day_budget=10_000, max_rows=100, max_bytes=10**7, database="d"
        )
        apply_files(db, out / "day01", only=lambda n_: "1_sample" in n_)
        revs = {r["rev"] for r in dump(db, "s1_sample")}
        assert (
            dump(db, "s1_study")[:1] == [] or dump(db, "s1_study")[0]["rev"] == "1" * 16
        )  # readers still see rev 1 until the flip
        if rev == "2" * 16:
            assert revs == {"1" * 16, "2" * 16}
        apply_files(db, out / "day01", only=lambda n_: "2_flip" in n_)


def test_schema_032_variant_and_the_loud_failure_of_the_wrong_key(tmp_path):
    rows = {"GSE1": synth("GSE1", 2)[2]}
    out = tmp_path / "sql32"
    camp.write_sql_files(
        rows, out, day_budget=10_000, max_rows=100, max_bytes=10**7, database="d", schema="032"
    )
    db = new_db("032")
    apply_files(db, out / "day01")
    assert db.execute("select count(*) from s1_sample").fetchone()[0] == 2
    out33 = tmp_path / "sql33"
    camp.write_sql_files(
        rows, out33, day_budget=10_000, max_rows=100, max_bytes=10**7, database="d"
    )  # default 033
    with pytest.raises(sqlite3.OperationalError, match="ON CONFLICT"):
        apply_files(
            new_db("032"), out33 / "day01"
        )  # 033 SQL against a 032 table fails, it does not write wrongly


def test_day_budget_keeps_whole_studies_together(tmp_path):
    rows = {
        f"GSE{i}": synth(f"GSE{i}", n)[2]
        for i, n in ((1, 10), (2, 10), (3, 10), (4, 10), (5, 40), (6, 1))
    }
    out = tmp_path / "sql"
    plan = camp.write_sql_files(
        rows, out, day_budget=110, max_rows=1000, max_bytes=10**7, database="d"
    )
    per = {g: camp.est_writes(r) for g, r in rows.items()}
    assert per["GSE1"] == 10 * 3 + 2 + 2
    days = [(out / f"day{d['day']:02d}" / "STUDIES.txt").read_text().split() for d in plan["days"]]
    assert days == [
        ["GSE1", "GSE2", "GSE3"],
        ["GSE4"],
        ["GSE5"],
        ["GSE6"],
    ]  # GSE5 alone exceeds the budget (3 x 34 = 102 fit, 124 does not): a day of its own
    assert sorted(sum(days, [])) == sorted(rows) and all(
        sum(per[g] for g in d) <= 110 or len(d) == 1 for d in days
    )
    assert plan["est_writes"] == sum(per.values())
    assert "4 day(s)" in (out / "PLAN.txt").read_text()
    # each day directory is self-contained: applying day 1 alone leaves a consistent catalog
    db = new_db()
    apply_files(db, out / "day01")
    assert {r["gse_id"] for r in dump(db, "s1_study")} == {"GSE1", "GSE2", "GSE3"} and {
        r["gse_id"] for r in dump(db, "s1_migration")
    } == {"GSE1", "GSE2", "GSE3"}


def test_a_study_whose_sample_row_cannot_fit_one_statement_is_skipped_loudly_and_a_big_one_is_shrunk(
    tmp_path,
):
    big_names = {"names": {"rows": {"pack": "a" * 32, "member": "exon"}, "pad": "x" * 120_000}}
    man, row, rows = synth(
        "GSE1",
        1,
        extra={
            "exon2.1pz": dict({"kind": 1, "off": 1, "len": 1, "chunk_index": [1, 2]}, **big_names)
        },
    )
    st = camp.study_statements(rows)
    assert len(st["sample"][0]) < camp.MAX_STATEMENT
    kept = json.loads(rows["s1_sample"][0]["members_json"])["exon2.1pz"]
    assert kept["chunk_index"] == [1, 2] and "off" in kept
    # shrunk member json no longer carries `names` in the SQL
    assert "x" * 100 not in st["sample"][0]
    man, row, rows = synth(
        "GSE2", 1, extra={f"m{i}": {"kind": 3, "off": i, "len": 1} for i in range(3000)}
    )
    with pytest.raises(camp.CatalogError):
        camp.study_statements(rows)
    out = tmp_path / "sql"
    plan = camp.write_sql_files(
        {"GSE2": rows, "GSE3": synth("GSE3", 1)[2]},
        out,
        day_budget=10_000,
        max_rows=10,
        max_bytes=10**7,
        database="d",
        log=lambda *_: None,
    )
    assert (
        list(plan["skipped"]) == ["GSE2"]
        and "SKIPPED GSE2" in (out / "PLAN.txt").read_text()
        and plan["n_studies"] == 1
    )


def test_the_whole_atlas_fits_the_planned_days():
    n_studies, per_study = 8202, 6.4
    est = camp.WRITES_PER_ROW
    total = int(n_studies * per_study) * est["s1_sample"] + n_studies * (
        est["s1_study"] + est["s1_migration"]
    )
    days = -(-total // camp.DAY_BUDGET)
    assert camp.DAY_BUDGET == 80_000 and 2 <= days <= 3  # about 52k samples: 2 to 3 days


# ── through the campaign ──────────────────────────────────────────────────────────────────────────────────────────────


def test_campaign_rows_to_sql_in_sqlite_and_from_r2_agree(env, tmp_path):
    C.seed(env.root, 3)
    assert env.run() == 0
    a, b = tmp_path / "a", tmp_path / "b"
    assert (
        camp.main(["catalog-sql", "--rows-dir", str(env.rows), "--out", str(a), "--max-rows", "4"])
        == 0
    )
    assert (
        camp.main(
            [
                "catalog-sql",
                "--from-r2",
                "--local-root",
                str(env.root),
                "--out",
                str(b),
                "--max-rows",
                "4",
            ]
        )
        == 0
    )
    dbs = []
    for d in (a, b):
        db = new_db()
        apply_files(db, d / "day01")
        dbs.append(db)
    for t in ("s1_study", "s1_sample", "s1_migration"):

        def strip(rs):
            return [{k: v for k, v in r.items() if k != "updated_at"} for r in rs]

        assert strip(dump(dbs[0], t)) == strip(dump(dbs[1], t)), t
    db = dbs[0]
    assert (
        db.execute("select count(*) from s1_sample").fetchone()[0] == 6
        and db.execute("select count(*) from s1_study").fetchone()[0] == 3
    )
    m = dump(db, "s1_migration")[0]
    assert (
        m["status"] == "verified"
        and m["legacy_etag"]
        and m["legacy_bytes"] > 0
        and json.loads(m["members_by_kind"])["matrix"] == 2
    )
    s = dump(db, "s1_sample")[0]
    assert s["key"].startswith("s1/s/GSM") and json.loads(s["members_json"])["exon_counts.1pz"][
        "chunk_index"
    ] == [200, 48]
    assert json.loads(s["modalities"]) == ["exon_counts", "splice_psi"] + [
        x for x in json.loads(s["modalities"]) if x not in ("exon_counts", "splice_psi")
    ]
    # a failed rerun does not downgrade a published study: its rows file is left alone
    env.mp.setenv("FAKE_MIGRATE_FAIL", "GSE000001")
    assert env.run() == 1
    assert "s1_sample" in json.loads((env.rows / "GSE000001.json").read_text())


def test_a_study_that_never_succeeded_gets_a_failed_migration_row(env, tmp_path):
    C.seed(env.root, 2)
    env.mp.setenv("FAKE_VERIFY_FAIL", "GSE000002")
    assert env.run() == 1
    out = tmp_path / "sql"
    assert camp.main(["catalog-sql", "--rows-dir", str(env.rows), "--out", str(out)]) == 0
    db = new_db()
    apply_files(db, out / "day01")
    st = {r["gse_id"]: r for r in dump(db, "s1_migration")}
    assert (
        st["GSE000001"]["status"] == "verified"
        and st["GSE000002"]["status"] == "failed"
        and "verify exited" in st["GSE000002"]["error"]
    )
    assert [r["gse_id"] for r in dump(db, "s1_study")] == ["GSE000001"]


def test_a_restricted_member_in_the_store_object_stops_the_study_before_any_upload(env):
    C.seed(env.root, 1)
    env.mp.setenv("FAKE_RESTRICTED_MEMBER", "1")
    assert env.run() == 1
    r = env.rows_of()["GSE000001"]
    assert r["status"] == "collect_failed" and "restricted members" in r["error"]
    assert all(k.startswith("s1/ledger/") for op, _b, k in env.store.ops if op == "put")


def test_pack_lists_in_the_manifest_are_all_uploaded(env):
    C.seed(env.root, 1)
    env.mp.setenv(
        "FAKE_PACK_LIST", "1"
    )  # packs.whitelist is a list of two ids, as W2 writes it when a sample uses several whitelists
    assert env.run() == 0
    assert (
        env.rows_of()["GSE000001"]["uploads"]["objects"] == 4
    )  # two samples + the exon pack + the whitelist pack (both ids are the same two files)


# ── POST ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_post_bodies_are_slices_that_share_rev_and_totals():
    man, _row, rows = synth("GSE1", 5)
    bodies = camp.post_payloads(man, rows, slice_size=2)
    assert [len(b["manifest"]["samples"]) for b in bodies] == [2, 2, 1] and [
        b["final"] for b in bodies
    ] == [False, False, True]
    for b in bodies:
        m = b["manifest"]
        assert (
            "study_files" not in m
            and m["rev"] == man["rev"]
            and m["n_samples"] == 5
            and m["bytes"] == rows["s1_study"]["bytes"]
        )
        assert ("migration" in b) == b["final"]
    mg = bodies[-1]["migration"]
    assert (
        mg["status"] == "verified"
        and mg["verify_ok"] is True
        and mg["members_by_kind"] == {"matrix": 5, "cells": 5, "stored": 5}
    )
    assert bodies[-1]["manifest_sha256"] == "e" * 64 and bodies[-1]["manifest_bytes"] == 1234
    assert "gse_id" not in mg and all(v is not None for v in mg.values())
    # a body over the limit is cut into more slices
    big = camp.post_payloads(man, rows, slice_size=400, max_body=2500)
    assert (
        len(big) >= 2
        and all(len(json.dumps(b)) <= 2500 for b in big)
        and sum(len(b["manifest"]["samples"]) for b in big) == 5
    )


class _Ingest(BaseHTTPRequestHandler):
    log = []
    fail_first = False

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Ingest.log.append(
            (
                self.path,
                self.headers.get("X-Ingest-Token"),
                self.headers.get("User-Agent"),
                json.loads(self.rfile.read(n)),
            )
        )
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


def test_catalog_post_sends_manifests_with_the_token_and_retries(env, monkeypatch):
    C.seed(env.root, 2)
    assert env.run() == 0
    srv = HTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Ingest.log, _Ingest.fail_first = [], True
    monkeypatch.setattr(camp.time, "sleep", lambda s: None)
    monkeypatch.setenv("INGEST_TOKEN", "t0ken")
    try:
        assert (
            camp.main(
                [
                    "catalog-post",
                    "--local-root",
                    str(env.root),
                    "--base-url",
                    f"http://127.0.0.1:{srv.server_port}",
                ]
            )
            == 0
        )
    finally:
        srv.shutdown()
    assert [p for p, *_ in _Ingest.log] == [
        "/api/ingest/s1-study"
    ] * 3  # the first got 429 and was sent again, then one body per study
    assert all(tok == "t0ken" and "singlet-campaign" in ua for _p, tok, ua, _b in _Ingest.log)
    bodies = [b for *_x, b in _Ingest.log]
    assert (
        bodies[0]["manifest"]["gse"] == bodies[1]["manifest"]["gse"] == "GSE000001"
        and bodies[2]["manifest"]["gse"] == "GSE000002"
    )
    assert all(
        "study_files" not in b["manifest"] and b["migration"]["status"] == "verified"
        for b in bodies
    )
    assert camp.main(["catalog-post", "--local-root", str(env.root), "--dry-run"]) == 0


def test_catalog_post_needs_a_token(env, monkeypatch):
    C.seed(env.root, 1)
    assert env.run() == 0
    monkeypatch.delenv("INGEST_TOKEN", raising=False)
    assert camp.main(["catalog-post", "--local-root", str(env.root)]) == 2


def _node_tools():
    node = shutil.which("node")
    esb = (
        WEBSITE
        / "node_modules"
        / ".bin"
        / ("esbuild.cmd" if sys.platform == "win32" else "esbuild")
    )
    if (
        not node
        or not esb.exists()
        or not (WEBSITE / "functions" / "_shared" / "store-catalog.ts").exists()
    ):
        return None
    return node, str(esb)


def test_rows_equal_what_the_website_endpoint_writes_from_the_same_manifest(env, tmp_path):
    """Runs the website's own parseIngestBody + ingestStatements (TypeScript, bundled with its esbuild) on the POST body this
    script builds, in SQLite with the real schema files, and compares every column with the rows of the SQL path."""
    tools = _node_tools()
    if tools is None:
        pytest.skip("node / the website worktree are not available")
    node, esb = tools
    C.seed(env.root, 2)
    assert env.run() == 0
    items = {}
    for g in ("GSE000001", "GSE000002"):
        m = json.loads((env.root / PUB / "s1" / "m" / f"{g}.json").read_text())
        rows = json.loads((env.rows / f"{g}.json").read_text())
        items[g] = (m, rows)
    bodies = [camp.post_payloads(m, r)[-1] for m, r in items.values()]
    (tmp_path / "bodies.json").write_text(json.dumps(bodies))
    ts = tmp_path / "ingest.ts"
    root = str(WEBSITE).replace("\\", "/")
    ts.write_text(
        f'''
import {{ readFileSync, writeFileSync }} from "node:fs";
import {{ parseIngestBody, ingestStatements }} from "{root}/functions/_shared/store-catalog";
import {{ createTestD1 }} from "{root}/src/test/helpers/d1-sqlite";
const dir = process.argv[2];
const schema = readFileSync("{root}/schema/032_v2_catalog.sql", "utf8") + readFileSync("{root}/schema/033_v2_sample_rev_key.sql", "utf8") + readFileSync("{root}/schema/036_s1_rename.sql", "utf8");
const db: any = createTestD1(schema);
for (const body of JSON.parse(readFileSync(dir + "/bodies.json", "utf8"))) {{
  const {{ batches, tail }} = ingestStatements(db, parseIngestBody(body), "STAMP", null);
  for (const b of batches) await db.batch(b);
  if (tail.length) await db.batch(tail);
}}
const out: any = {{}};
for (const t of ["s1_study", "s1_sample", "s1_migration"]) out[t] = db.raw.prepare(`SELECT * FROM ${{t}} ORDER BY 1, 2, 3`).all();
writeFileSync(dir + "/w5.json", JSON.stringify(out));
''',
        encoding="utf-8",
    )
    r = subprocess.run(
        [
            esb,
            str(ts),
            "--bundle",
            "--platform=node",
            "--format=esm",
            f"--outfile={tmp_path / 'ingest.mjs'}",
            "--log-level=error",
        ],
        capture_output=True,
        text=True,
        cwd=str(WEBSITE),
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [node, "--no-warnings", str(tmp_path / "ingest.mjs"), str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(WEBSITE),
    )
    assert r.returncode == 0, r.stderr
    theirs = json.loads((tmp_path / "w5.json").read_text())
    out = tmp_path / "sql"
    assert camp.main(["catalog-sql", "--rows-dir", str(env.rows), "--out", str(out)]) == 0
    db = new_db()
    apply_files(db, out / "day01")
    for t in ("s1_study", "s1_sample", "s1_migration"):
        mine = dump(db, t)
        assert len(mine) == len(theirs[t]) > 0
        for a, b in zip(mine, theirs[t]):
            for k, v in a.items():
                if k != "updated_at":
                    assert v == b[k] or (
                        isinstance(v, str) and json.loads(v) == json.loads(b[k])
                    ), (t, k, v, b[k])


# ── reference packs ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_refs_index_publishes_the_packs_and_the_index_once(env):
    packs = REPO / "python" / "singlet" / "s1" / "packs"
    if not (packs / "index.json").exists():
        pytest.skip("no packs directory in this checkout")
    n = len(json.loads((packs / "index.json").read_text())["packs"])
    argv = ["refs-index", "--local-root", str(env.root), "--packs-dir", str(packs)]
    assert camp.main(argv) == 0
    assert not (env.root / PUB / "s1").exists()  # no --apply: nothing written
    assert camp.main(argv + ["--apply"]) == 0
    keys = sorted(k for k, _ in env.store.list_keys(PUB, "s1/ref/"))
    assert len(keys) == n + 1 and "s1/ref/index.json" in keys
    h = env.store.head(PUB, "s1/ref/index.json")
    assert h.content_type == "application/json" and h.cache_control == "public, max-age=300"
    pk = next(k for k in keys if k.endswith(".pack"))
    assert env.store.head(PUB, pk).cache_control == "public, max-age=31536000, immutable"
    env.store.ops.clear()
    assert camp.main(argv + ["--apply"]) == 0
    assert [o for o in env.store.ops if o[0] == "put"] == []  # second run: everything already there


def test_refs_index_refuses_a_pack_that_is_not_its_name(env, tmp_path):
    packs = REPO / "python" / "singlet" / "s1" / "packs"
    if not (packs / "index.json").exists():
        pytest.skip("no packs directory in this checkout")
    d = tmp_path / "packs"
    shutil.copytree(packs, d)
    victim = next(d.glob("*.pack"))
    raw = bytearray(victim.read_bytes())
    raw[-1] ^= 0xFF
    victim.write_bytes(bytes(raw))
    assert (
        camp.main(["refs-index", "--local-root", str(env.root), "--packs-dir", str(d), "--apply"])
        == 1
    )
    assert not (env.root / PUB / "s1").exists()  # nothing is uploaded when any pack is wrong
    victim.unlink()
    assert (
        camp.main(["refs-index", "--local-root", str(env.root), "--packs-dir", str(d), "--apply"])
        == 1
    )


# ── the real transcoder and verifier ──────────────────────────────────────────────────────────────────────────────────

CORPUS = REPO / "bench" / "compression" / "corpus" / "bundles" / "GSE145370.singlet"


def test_real_transcoder_and_verifier_end_to_end_on_two_samples(tmp_path):
    """No stand-ins: the bundle of the corpus, `python -m singlet.s1.migrate` and `verify` on two of its samples, upload to a
    directory-backed R2, the catalog SQL, applied to the real schema. Skipped without the corpus bundle."""
    if not CORPUS.exists():
        pytest.skip("corpus bundle bench/compression/corpus/bundles/GSE145370.singlet not present")
    try:
        sys.path.insert(0, str(REPO / "python"))
        from singlet import _pz2  # noqa: F401
    except ImportError:
        pytest.skip("native codec singlet._pz2 not built")
    root = tmp_path / "r2"
    dst = root / PUB / "data" / "GSE145370" / "GSE145370.singlet"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(CORPUS, dst)
    samples = "GSM4317428,GSM4317431"
    args = [
        "run",
        "--mode",
        "full",
        "--local-root",
        str(root),
        "--gse",
        "GSE145370",
        "--scratch",
        str(tmp_path / "scratch"),
        "--results",
        str(tmp_path / "res.jsonl"),
        "--rows-dir",
        str(tmp_path / "rows"),
        "--workers",
        "1",
        "--readback",
        "sha256",
        "--migrate-args",
        json.dumps(["--samples", samples]),
        "--verify-args",
        json.dumps(["--samples", samples]),
    ]
    assert camp.main(args) == 0
    r = json.loads((tmp_path / "res.jsonl").read_text().splitlines()[-1])
    assert (
        r["status"] == "ok"
        and r["n_samples"] == 2
        and r["verify"]["ok"]
        and r["s1_sample_bytes"] < 0.3 * 24_000_000
    )
    assert r["members_by_kind"]["pz2"] == 8 and r["migrate_ledger_lines"] > 20
    m = json.loads((root / PUB / "s1" / "m" / "GSE145370.json").read_text())
    assert (
        m["legacy"]["bytes"] == CORPUS.stat().st_size
        and m["legacy"]["etag"] == r["legacy_etag"]
        and m["legacy"]["sha256"] == r["legacy_sha256"]
    )
    out = tmp_path / "sql"
    assert camp.main(["catalog-sql", "--rows-dir", str(tmp_path / "rows"), "--out", str(out)]) == 0
    db = new_db()
    apply_files(db, out / "day01")
    assert db.execute("select count(*) from s1_sample").fetchone()[0] == 2
    mg = dump(db, "s1_migration")[0]
    assert mg["status"] == "verified" and mg["verify_ok"] == 1 and mg["s1_bytes"] == r["s1_bytes"]
    if (
        WEBSITE / "schema" / "033_v2_sample_rev_key.sql"
    ).exists():  # and in the website's own schema files
        db2 = sqlite3.connect(":memory:")
        db2.executescript(website_ddl("033"))
        apply_files(db2, out / "day01")
        assert db2.execute("select count(*) from s1_sample").fetchone()[0] == 2
