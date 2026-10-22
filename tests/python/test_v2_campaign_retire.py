# SPDX-License-Identifier: MIT
"""scripts/v2_retire.py: archive, verify, delete; restore; audit. Directory-backed buckets and an in-memory S3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
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
ret = C.load_script("v2_retire")

PUB, ARC = "singlet-data", "singlet-archive"


@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    """Three studies migrated with the campaign (stand-in transcoder); copied per test."""
    base = tmp_path_factory.mktemp("migrated")
    root = base / "r2"
    tool = C.write_fake_tool(base)
    gses = C.seed(root, 3)
    args = ["run", "--mode", "full", "--local-root", str(root), "--scratch", str(base / "scratch"), "--shard", "0/1",
            "--workers", "2", "--retry-sleep", "0", "--part-size-mib", "0.001"] + C.tool_cmds(tool)
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
    e.ledger = tmp_path / "retire.jsonl"
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

    e.go = go
    return e


def tree(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in Path(root).rglob("*") if p.is_file())


# ── retire ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_dry_run_changes_nothing(r2, capsys):
    before = tree(r2.root)
    assert r2.go("retire", "--all") == 0
    out = capsys.readouterr().out
    assert out.count("would-retire") == 3 and "does not exist; --apply would create it" in out
    assert tree(r2.root) == before


def test_apply_needs_the_confirmation_phrase(r2, capsys):
    before = tree(r2.root)
    assert r2.go("retire", "--all", "--apply") == 2
    assert r2.go("retire", "--all", "--apply", "--confirm", "yes") == 2
    assert tree(r2.root) == before


def test_retire_archives_verifies_then_deletes(r2, capsys):
    v2_before = [t for t in tree(r2.root) if t[0].startswith(f"{PUB}/v2/")]
    assert r2.go("retire", "--all", "--apply", "--confirm", "RETIRE", "--ledger", str(r2.ledger)) == 0
    for g in r2.gses:
        assert not r2.legacy(g).exists()
        assert r2.arch(g).read_bytes() == r2.orig[g]  # same key, same bytes
        h = r2.store.head(ARC, f"data/{g}/{g}.singlet")
        assert h.etag == hashlib.md5(r2.orig[g]).hexdigest()
        meta = json.loads((r2.root / ".meta" / ARC / "data" / g / f"{g}.singlet.json").read_text())
        assert meta["storage_class"] == "STANDARD_IA"
    recs = [json.loads(x) for x in r2.ledger.read_text().splitlines()]
    assert {r["stage"] for r in recs} == {"archived", "retired"} and all(r["ok"] for r in recs)
    # v2 objects are untouched (only the ledgers gained a line)
    v2_after = [t for t in tree(r2.root) if t[0].startswith(f"{PUB}/v2/")]
    changed = {k for (k, s) in set(v2_after) ^ set(v2_before)}
    assert all("ledger" in k for k in changed)
    led = [json.loads(x) for x in (r2.root / PUB / "v2" / "ledger" / f"{r2.gses[0]}.jsonl").read_text().splitlines()]
    assert led[-1]["event"] == "retired" and led[-1]["verified_by"] == "etag"
    capsys.readouterr()
    assert r2.go("retire", "--gse", *r2.gses, "--apply", "--confirm", "RETIRE") == 0
    assert capsys.readouterr().out.count("already-retired") == 3


def test_every_delete_is_a_legacy_key_and_follows_a_verified_archive_copy(r2):
    store = camp.LocalStore(r2.root)
    assert r2.go("retire", "--all", "--apply", "--confirm", "RETIRE", store=store) == 0
    ops = store.ops
    deletes = [(i, k) for i, (op, _b, k) in enumerate(ops) if op == "delete"]
    assert sorted(k for _i, k in deletes) == sorted(f"data/{g}/{g}.singlet" for g in r2.gses)
    for i, k in deletes:
        copies = [j for j, (op, b, kk) in enumerate(ops) if op == "copy" and b == ARC and kk == k]
        assert copies and copies[0] < i  # archived before deleted


def test_refuses_a_study_rebuilt_after_it_was_migrated(r2, capsys):
    g = r2.gses[1]
    C.put_legacy(r2.root, g, ["GSM20001", "GSM20002", "GSM20003"], salt="rebuilt")  # new bundle, different ETag
    assert r2.go("retire", "--all", "--apply", "--confirm", "RETIRE") == 1
    out = capsys.readouterr().out
    assert f"{g}\tREFUSED: legacy object changed since it was migrated" in out
    assert r2.legacy(g).exists() and not r2.arch(g).exists()
    assert not r2.legacy(r2.gses[0]).exists() and r2.arch(r2.gses[0]).exists()  # the others went through


def test_refuses_without_a_manifest_missing_samples_or_a_verified_ledger(r2, capsys):
    g0, g1, g2 = r2.gses
    (r2.root / PUB / "v2" / "m" / f"{g0}.json").unlink()
    m1 = json.loads((r2.root / PUB / "v2" / "m" / f"{g1}.json").read_text())
    (r2.root / PUB / m1["samples"][0]["key"]).unlink()
    (r2.root / PUB / "v2" / "ledger" / f"{g2}.jsonl").unlink()
    assert r2.go("retire", "--gse", g0, g1, g2, "--apply", "--confirm", "RETIRE") == 1
    out = capsys.readouterr().out
    assert "no readable v2 manifest" in out and "sample objects of the manifest are missing" in out and "no committed, verified migration" in out
    assert all(r2.legacy(g).exists() for g in r2.gses) and not any((r2.root / ARC).rglob("*.singlet"))


def test_a_full_retire_waits_for_every_manifest(r2):
    (r2.root / PUB / "v2" / "m" / f"{r2.gses[2]}.json").unlink()
    with pytest.raises(SystemExit) as e:
        r2.go("retire", "--all", "--apply", "--confirm", "RETIRE")
    assert "no v2 manifest yet" in str(e.value) and r2.gses[2] in str(e.value)
    assert all(r2.legacy(g).exists() for g in r2.gses)
    assert r2.go("retire", "--all", "--allow-partial", "--apply", "--confirm", "RETIRE") == 1  # the third is REFUSED
    assert not r2.legacy(r2.gses[0]).exists() and r2.legacy(r2.gses[2]).exists()


def test_source_changing_during_the_archive_step_is_not_deleted(r2, capsys):
    g = r2.gses[0]

    class Racy(camp.LocalStore):
        def copy(self, *a, **kw):
            out = super().copy(*a, **kw)
            if a[1].endswith(f"{g}.singlet") and a[0] == PUB:
                self._p(PUB, a[1]).write_bytes(b"replaced meanwhile")
                self._write_meta(PUB, a[1], {"etag": "changed"})
            return out

    store = Racy(r2.root)
    assert r2.go("retire", "--gse", g, "--apply", "--confirm", "RETIRE", store=store) == 1
    assert "changed while it was being archived" in capsys.readouterr().out
    assert r2.legacy(g).exists()


def test_multipart_copy_is_verified_by_sha256_against_the_migrated_bundle(r2, capsys):
    store = camp.LocalStore(r2.root, copy_single_max=300, copy_part=256)
    g = r2.gses[0]
    assert len(r2.orig[g]) > 300
    assert r2.go("retire", "--gse", g, "--apply", "--confirm", "RETIRE", "--ledger", str(r2.ledger), store=store) == 0
    assert r2.arch(g).read_bytes() == r2.orig[g]
    assert store.head(ARC, f"data/{g}/{g}.singlet").etag.endswith("-" + str(-(-len(r2.orig[g]) // 256)))  # composite ETag
    recs = [json.loads(x) for x in r2.ledger.read_text().splitlines()]
    assert next(r for r in recs if r["stage"] == "archived")["verified_by"] == "sha256"
    assert not r2.legacy(g).exists()


def test_a_corrupt_copy_is_never_accepted(r2, capsys):
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
    assert r2.go("retire", "--gse", g, "--apply", "--confirm", "RETIRE", store=Bad(r2.root)) == 1
    assert "differs from the source" in capsys.readouterr().out
    assert r2.legacy(g).exists()


def test_a_rerun_after_a_crash_between_archive_and_delete(r2):
    g = r2.gses[0]

    class Dies(camp.LocalStore):
        def delete(self, bucket, key):
            raise OSError("node died")

    assert r2.go("retire", "--gse", g, "--apply", "--confirm", "RETIRE", store=Dies(r2.root)) == 1
    assert r2.legacy(g).exists() and r2.arch(g).exists()
    store = camp.LocalStore(r2.root)
    assert r2.go("retire", "--gse", g, "--apply", "--confirm", "RETIRE", store=store) == 0
    assert [op for op, _b, _k in store.ops if op == "copy"] == []  # the verified archive copy was reused
    assert not r2.legacy(g).exists()


# ── restore ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_restore_puts_back_identical_bytes(r2, capsys):
    assert r2.go("retire", "--all", "--apply", "--confirm", "RETIRE") == 0
    capsys.readouterr()
    assert r2.go("restore", "--all") == 0  # dry run
    assert capsys.readouterr().out.count("would-restore") == 3 and not r2.legacy(r2.gses[0]).exists()
    assert r2.go("restore", "--all", "--apply", "--ledger", str(r2.ledger)) == 0
    for g in r2.gses:
        assert r2.legacy(g).read_bytes() == r2.orig[g]
        assert r2.arch(g).exists()  # the archive copy stays until the grace period ends
    capsys.readouterr()
    assert r2.go("restore", "--all", "--apply") == 0
    assert capsys.readouterr().out.count("already-present") == 3
    led = [json.loads(x) for x in (r2.root / PUB / "v2" / "ledger" / f"{r2.gses[0]}.jsonl").read_text().splitlines()]
    assert [e["event"] for e in led if e["event"] in ("retired", "restored")] == ["retired", "restored"]


def test_restore_does_not_overwrite_a_different_object_unless_forced(r2, capsys):
    g = r2.gses[0]
    assert r2.go("retire", "--gse", g, "--apply", "--confirm", "RETIRE") == 0
    C.put_legacy(r2.root, g, ["GSM1"], salt="newer")
    newer = r2.legacy(g).read_bytes()
    assert r2.go("restore", "--gse", g, "--apply") == 1
    assert r2.legacy(g).read_bytes() == newer and "SKIPPED" in capsys.readouterr().out
    assert r2.go("restore", "--gse", g, "--apply", "--force") == 0
    assert r2.legacy(g).read_bytes() == r2.orig[g]


# ── audit ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_audit_lists_what_remains(r2, capsys):
    assert r2.go("audit") == 0
    out = capsys.readouterr().out
    assert out.count("\t1\t0") == 3 and "3 legacy bundles" in out
    assert r2.go("audit", "--expect-none") == 1
    assert r2.go("retire", "--gse", r2.gses[0], "--apply", "--confirm", "RETIRE") == 0
    capsys.readouterr()
    assert r2.go("audit") == 0
    out = capsys.readouterr().out
    assert "2 legacy bundles" in out and "1 archived" in out
    assert r2.go("retire", "--all", "--apply", "--confirm", "RETIRE") == 0
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
    fake.put_raw(PUB, "data/GSE1/GSE1.singlet", data[:size], ContentType="application/zip", CacheControl="x")
    fake.buckets.add(ARC)
    return fake, data[:size]


def test_s3_single_copy_keeps_etag_and_sets_storage_class():
    fake, data = _fake_with_object()
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    etag, mp = store.copy(PUB, "data/GSE1/GSE1.singlet", ARC, "data/GSE1/GSE1.singlet", size=len(data), storage_class="STANDARD_IA")
    assert not mp and etag == hashlib.md5(data).hexdigest()
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["StorageClass"] == "STANDARD_IA"
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["ContentType"] == "application/zip"


def test_s3_copy_falls_back_when_the_storage_class_is_not_accepted():
    fake, data = _fake_with_object()
    fake.reject_storage_class = True
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    etag, mp = store.copy(PUB, "data/GSE1/GSE1.singlet", ARC, "data/GSE1/GSE1.singlet", size=len(data), storage_class="STANDARD_IA")
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["StorageClass"] is None
    assert [c for c in fake.calls if c[0] == "copy_object"] == [("copy_object", ARC, "data/GSE1/GSE1.singlet", "STANDARD_IA"),
                                                                 ("copy_object", ARC, "data/GSE1/GSE1.singlet", None)]


def test_s3_multipart_copy_uses_fixed_ranges_and_aborts_on_failure():
    fake, data = _fake_with_object(1000)
    store = camp.S3Store(client=fake, transfer_config=C.TC, copy_single_max=300, copy_part=256)
    etag, mp = store.copy(PUB, "data/GSE1/GSE1.singlet", ARC, "data/GSE1/GSE1.singlet", size=1000, storage_class="STANDARD_IA")
    assert mp and etag.endswith("-4")
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["data"] == data
    assert fake.objs[(ARC, "data/GSE1/GSE1.singlet")]["ContentType"] == "application/zip"
    assert sorted(c[3] for c in fake.calls if c[0] == "upload_part_copy") == [1, 2, 3, 4]
    # a failing part aborts the upload
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
            camp.S3Store(client=fake2, transfer_config=C.TC, copy_single_max=300, copy_part=256).copy(
                PUB, "data/GSE1/GSE1.singlet", ARC, "data/GSE1/GSE1.singlet", size=1000)
    finally:
        _t.sleep = real_sleep
    assert any(c[0] == "abort_mpu" for c in fake2.calls) and (ARC, "data/GSE1/GSE1.singlet") not in fake2.objs


def test_retire_on_the_s3_store_creates_the_archive_bucket(migrated, tmp_path, monkeypatch):
    """The whole retire flow through S3Store + FakeS3, from a campaign-written tree loaded into the fake."""
    src, gses = migrated
    fake = C.FakeS3()
    fake.buckets.add(PUB)
    root = src / PUB
    for p in sorted(root.rglob("*")):
        if p.is_file():
            key = p.relative_to(root).as_posix()
            meta_p = src / ".meta" / PUB / (key + ".json")
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            fake.put_raw(PUB, key, p.read_bytes(), etag=meta.get("etag"), Metadata=meta.get("metadata", {}))
    store = camp.S3Store(client=fake, transfer_config=C.TC)
    monkeypatch.setattr(ret.vc, "S3Store", lambda: store)
    assert ARC not in fake.buckets
    assert ret.main(["retire", "--all", "--apply", "--confirm", "RETIRE", "--workers", "2"]) == 0
    assert ARC in fake.buckets and ("create_bucket", ARC) in fake.calls
    for g in gses:
        assert (PUB, f"data/{g}/{g}.singlet") not in fake.objs
        assert fake.objs[(ARC, f"data/{g}/{g}.singlet")]["StorageClass"] == "STANDARD_IA"
    # every delete the client saw was a legacy key
    assert sorted(c[2] for c in fake.calls if c[0] == "delete") == sorted(f"data/{g}/{g}.singlet" for g in gses)
    assert ret.main(["audit", "--expect-none"]) == 0


def test_shards_keep_meaning_the_same_studies_while_other_shards_delete(r2):
    """The study list comes from the manifests, not from the legacy keys that the other shards are removing."""
    assert r2.go("retire", "--all", "--shard", "0/2", "--apply", "--confirm", "RETIRE") == 0
    assert not r2.legacy(r2.gses[0]).exists() and not r2.legacy(r2.gses[2]).exists() and r2.legacy(r2.gses[1]).exists()
    assert r2.go("retire", "--all", "--shard", "1/2", "--apply", "--confirm", "RETIRE") == 0
    assert not any(r2.legacy(g).exists() for g in r2.gses) and all(r2.arch(g).exists() for g in r2.gses)
