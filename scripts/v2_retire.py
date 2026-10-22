#!/usr/bin/env python3
"""Retire legacy `.singlet` bundles once their v2 migration is verified: archive first, delete from the public bucket last.

For every study with a v2 manifest, in this order (stopping at the first failed check; nothing is deleted before the
archive copy is verified):

  1. refuse unless `v2/m/<GSE>.json` exists, its `legacy` block (written by the campaign) names the object that is live now
     (same ETag and size: the bundle was not rebuilt after it was migrated), every sample object in the manifest exists
     with the right size, and the study's ledger ends in a committed, verified migration;
  2. server-side copy `singlet-data/data/<GSE>/<GSE>.singlet` to `singlet-archive/<same key>` (storage class
     STANDARD_IA = Infrequent Access if the API takes it; objects over 4 GiB are copied in 1 GiB ranges);
  3. verify the copy: equal size and ETag for a single-request copy, else (multipart copies have a different ETag)
     SHA-256 of the archived object against the migrated bundle's SHA-256 from the manifest;
  4. look at the source once more (unchanged since step 1) and delete it from the public bucket.

`restore` copies archived bundles back to the public key (verified the same way). `audit` lists every `data/*.singlet`
key still in the public bucket and how far the archive is. Dry run unless `--apply`; `retire --apply` also needs
`--confirm RETIRE`. A full `retire --all` refuses to start while any study still lacks a manifest (`--allow-partial` to
override, e.g. for a pilot of a few studies named with --gse).

  python scripts/v2_retire.py retire  --gse GSE111113 GSE145370                  # dry run
  python scripts/v2_retire.py retire  --all --shard 0/8 --apply --confirm RETIRE --ledger retire-0.jsonl
  python scripts/v2_retire.py restore --gse GSE111113 --apply
  python scripts/v2_retire.py audit   [--expect-none]

The archive bucket (`singlet-archive`) is created only if it does not exist (needs a token that may create buckets; else
create it with `wrangler r2 bucket create singlet-archive --storage-class InfrequentAccess`). Credentials as for
v2_campaign.py. Archived bundles are meant to be deleted after the grace period in docs/v2-migration.md (D12).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_campaign as vc  # noqa: E402

ARCHIVE_BUCKET = "singlet-archive"
CONFIRM = "RETIRE"
_log_lock = threading.Lock()


class Refused(Exception):
    """A precondition failed; nothing was changed for this study."""


def log(path: Optional[Path], rec: dict) -> None:
    if path is None:
        return
    rec = {"at": vc.iso(), **rec}
    with _log_lock, open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def load_manifest(store, public: str, gse: str) -> Optional[dict]:
    if store.head(public, vc.manifest_key(gse)) is None:
        return None
    try:
        return json.loads(store.get_bytes(public, vc.manifest_key(gse)))
    except ValueError:
        return None


def preconditions(store, a: argparse.Namespace, gse: str, live: vc.Head, pool: cf.Executor) -> dict:
    """The verified manifest of `gse`, or Refused."""
    m = load_manifest(store, a.public, gse)
    if m is None:
        raise Refused("no readable v2 manifest")
    if m.get("format") != "singlet-v2" or m.get("gse") != gse or not m.get("samples"):
        raise Refused("manifest is not a singlet-v2 manifest of this study")
    lg = m.get("legacy") or {}
    if lg.get("etag") != live.etag or lg.get("bytes") != live.size:
        raise Refused(f"legacy object changed since it was migrated (manifest: {lg.get('etag')}/{lg.get('bytes')}, live: {live.etag}/{live.size}); migrate it again")

    def present(s: dict) -> bool:
        h = store.head(a.public, s.get("key", ""))
        return h is not None and h.size == s.get("bytes")

    missing = [s.get("key") for s, ok in zip(m["samples"], pool.map(present, m["samples"])) if not ok]
    if missing:
        raise Refused(f"{len(missing)} sample objects of the manifest are missing or the wrong size, e.g. {missing[0]}")
    summ = vc.last_committed(vc.read_ledger(store, a.public, gse))
    if summ is None or summ.get("status") != "ok" or not (summ.get("verify") or {}).get("ok", True):
        raise Refused("the study's ledger has no committed, verified migration")
    return m


def verify_copy(store, a: argparse.Namespace, src: vc.Head, src_key: str, dst_bucket: str, dst_key: str,
                expected_sha: Optional[str], src_bucket: str) -> str:
    """Size and ETag when the server kept the source ETag, else a SHA-256 read-back against the expected digest."""
    dh = store.head(dst_bucket, dst_key)
    if dh is None:
        raise Refused("the copy is not there")
    if dh.size != src.size:
        raise Refused(f"copy has {dh.size} bytes, source {src.size}")
    if dh.etag == src.etag:
        return "etag"
    got = store.sha256_object(dst_bucket, dst_key, dh.size)
    want = expected_sha or store.sha256_object(src_bucket, src_key, src.size)
    if got != want:
        raise Refused(f"copy SHA-256 {got[:12]} differs from the source {want[:12]}")
    return "sha256"


def archive_exists(store, a: argparse.Namespace) -> bool:
    """Whether the archive bucket exists (asked once per run; a dry run of a first retire finds none)."""
    if not hasattr(a, "_archive_exists"):
        a._archive_exists = store.bucket_exists(a.archive)
    return a._archive_exists


def storage_class(a: argparse.Namespace) -> Optional[str]:
    return None if a.storage_class.lower() in ("none", "") else a.storage_class


def retire_one(store, a: argparse.Namespace, gse: str, ledger: Optional[Path], pool: cf.Executor, guard) -> str:
    key = vc.legacy_key(gse)
    live = store.head(a.public, key)
    arch = store.head(a.archive, key) if archive_exists(store, a) else None
    if live is None:
        return "already-retired" if arch is not None else "missing: no legacy object and no archive copy"
    m = preconditions(store, a, gse, live, pool)
    want_sha = (m.get("legacy") or {}).get("sha256")
    method = None
    if arch is not None and arch.size == live.size:
        try:
            method = verify_copy(store, a, live, key, a.archive, key, want_sha, a.public)
        except Refused:
            method = None  # a partial or foreign object at the archive key: copied over below
    if method is None:
        if not a.apply:
            return f"would-retire: copy {live.size} bytes to {a.archive}/{key}, verify, delete from {a.public}"
        etag, multipart = guard.copy(a.public, key, a.archive, key, size=live.size, storage_class=storage_class(a))
        method = verify_copy(store, a, live, key, a.archive, key, want_sha, a.public)
        log(ledger, {"gse": gse, "stage": "archived", "ok": True, "bytes": live.size, "archive_etag": etag, "multipart": multipart,
                     "verified_by": method})
    elif not a.apply:
        return "would-retire (archive copy already present and verified)"
    again = store.head(a.public, key)
    if again is None or again.etag != live.etag or again.size != live.size:
        raise Refused("the legacy object changed while it was being archived; the archive copy is kept, nothing was deleted")
    guard.delete(a.public, key)
    if store.head(a.public, key) is not None:
        raise Refused("delete did not remove the object")
    rec = {"gse": gse, "stage": "retired", "ok": True, "bytes": live.size, "etag": live.etag, "verified_by": method,
           "archive": f"{a.archive}/{key}"}
    log(ledger, rec)
    vc.write_ledger(guard, a.public, gse, vc.read_ledger(store, a.public, gse) + [{"at": vc.iso(), "gse": gse, "event": "retired", **rec}])
    return f"retired {live.size} bytes (archive verified by {method})"


def restore_one(store, a: argparse.Namespace, gse: str, ledger: Optional[Path], guard) -> str:
    key = vc.legacy_key(gse)
    arch = store.head(a.archive, key) if archive_exists(store, a) else None
    if arch is None:
        return "SKIPPED: no archive copy"
    live = store.head(a.public, key)
    m = load_manifest(store, a.public, gse)
    want_sha = ((m or {}).get("legacy") or {}).get("sha256")
    if live is not None:
        if live.size == arch.size:
            try:
                verify_copy(store, a, arch, key, a.public, key, want_sha, a.archive)
                return "already-present"
            except Refused:
                pass
        if not a.force:
            return "SKIPPED: the public key exists with different content (--force to overwrite)"
    if not a.apply:
        return f"would-restore {arch.size} bytes from {a.archive}/{key}"
    etag, multipart = guard.copy(a.archive, key, a.public, key, size=arch.size, storage_class=None)
    method = verify_copy(store, a, arch, key, a.public, key, want_sha, a.archive)
    rec = {"gse": gse, "stage": "restored", "ok": True, "bytes": arch.size, "etag": etag, "multipart": multipart, "verified_by": method}
    log(ledger, rec)
    vc.write_ledger(guard, a.public, gse, vc.read_ledger(store, a.public, gse) + [{"at": vc.iso(), "gse": gse, "event": "restored", **rec}])
    return f"restored {arch.size} bytes (verified by {method})"


def studies_to_visit(store, a: argparse.Namespace, source: str) -> List[str]:
    """Named studies, or all of them in accession order. `source` is "manifests" (retire: the v2 manifests, which stay when the
    legacy keys go, plus the legacy studies that have none, so --shard I/N keeps meaning the same studies while other shards
    delete) or a bucket whose data/ keys to list."""
    if a.gse:
        gses = list(a.gse)
    elif a.gse_list:
        gses = vc.read_list(a.gse_list)
    elif source == "manifests":
        have = {m.group(1) for k, _ in store.list_keys(a.public, "v2/m/") if (m := vc.RE_MANIFEST.match(k))}
        gses = list(have | {g for g, _ in store.list_studies(a.public)})  # studies without a manifest are listed too (REFUSED)
    else:
        gses = [g for g, _ in store.list_studies(source)]
    gses = vc.accession_order(gses)
    if a.shard:
        i, n = a.shard
        gses = gses[i::n]
    return gses[: a.limit] if a.limit is not None else gses


def completeness_gate(store, a: argparse.Namespace) -> None:
    legacy = {g for g, _ in store.list_studies(a.public)}
    have = {m.group(1) for k, _ in store.list_keys(a.public, "v2/m/") if (m := vc.RE_MANIFEST.match(k))}
    missing = vc.accession_order(legacy - have)
    if missing:
        raise SystemExit(f"refusing to retire: {len(missing)} of {len(legacy)} legacy studies have no v2 manifest yet "
                         f"(first: {', '.join(missing[:10])}). Finish the migration, or name studies with --gse, or pass --allow-partial.")


def run_over(store, a: argparse.Namespace, fn, gses: Sequence[str]) -> int:
    ledger = Path(a.ledger) if a.ledger else None
    bad = 0

    def one(g: str) -> str:
        try:
            return fn(g)
        except Refused as e:
            log(ledger, {"gse": g, "ok": False, "stage": "refused", "error": str(e)[:300]})
            return f"REFUSED: {e}"
        except Exception as e:  # noqa: BLE001  one study must not stop the shard
            log(ledger, {"gse": g, "ok": False, "stage": "error", "error": f"{type(e).__name__}: {e}"[:300]})
            return f"FAILED: {type(e).__name__}: {e}"

    with cf.ThreadPoolExecutor(max(1, a.workers)) as ex:
        for g, msg in zip(gses, ex.map(one, gses)):
            bad += msg.startswith(("REFUSED", "FAILED", "SKIPPED"))
            print(f"{g}\t{msg}", flush=True)
    return 1 if bad else 0


def cmd_retire(store, a: argparse.Namespace) -> int:
    if a.apply and a.confirm != CONFIRM:
        print(f"error: --apply needs --confirm {CONFIRM}", file=sys.stderr)
        return 2
    if not (a.gse or a.gse_list or a.all):
        print("error: give --gse, --gse-list or --all", file=sys.stderr)
        return 2
    if a.all and not a.allow_partial:
        completeness_gate(store, a)
    if not archive_exists(store, a):
        if not a.apply:
            print(f"[retire] archive bucket {a.archive} does not exist; --apply would create it")
        else:
            store.create_bucket(a.archive)
            a._archive_exists = True
            print(f"[retire] created bucket {a.archive}")
    guard = vc.GuardedStore(store, {a.archive: ("data/",), a.public: ("v2/ledger/",)}, allow_delete={a.public: ("data/",)})
    pool = cf.ThreadPoolExecutor(max(1, a.workers))
    ledger = Path(a.ledger) if a.ledger else None
    try:
        return run_over(store, a, lambda g: retire_one(store, a, g, ledger, pool, guard), studies_to_visit(store, a, "manifests"))
    finally:
        pool.shutdown()


def cmd_restore(store, a: argparse.Namespace) -> int:
    if not (a.gse or a.gse_list or a.all):
        print("error: give --gse, --gse-list or --all", file=sys.stderr)
        return 2
    guard = vc.GuardedStore(store, {a.public: ("data/", "v2/ledger/")})
    ledger = Path(a.ledger) if a.ledger else None
    return run_over(store, a, lambda g: restore_one(store, a, g, ledger, guard), studies_to_visit(store, a, a.archive))


def cmd_audit(store, a: argparse.Namespace) -> int:
    remaining = [(k, s) for k, s in store.list_keys(a.public, "data/") if k.endswith(".singlet")]
    legacy = [(vc.RE_LEGACY.match(k).group(1), s) for k, s in remaining if vc.RE_LEGACY.match(k)]
    odd = [k for k, _ in remaining if not vc.RE_LEGACY.match(k)]
    manifests = {m.group(1) for k, _ in store.list_keys(a.public, "v2/m/") if (m := vc.RE_MANIFEST.match(k))}
    archived = {g for g, _ in store.list_studies(a.archive)} if store.bucket_exists(a.archive) else set()
    print("gse\tbytes\thas_v2_manifest\tarchived")
    for g, s in sorted(legacy, key=lambda t: (len(t[0]), t[0])):
        print(f"{g}\t{s}\t{int(g in manifests)}\t{int(g in archived)}")
    for k in odd:
        print(f"{k}\t?\t?\t?\t# not a data/<GSE>/<GSE>.singlet key")
    print(f"[audit] {len(legacy)} legacy bundles ({sum(s for _g, s in legacy) / 1e9:.2f} GB) remain in {a.public}; "
          f"{len(archived)} archived in {a.archive}; {len(manifests)} v2 manifests; {len(odd)} unexpected .singlet keys", flush=True)
    return 1 if (a.expect_none and (legacy or odd)) else 0


def cmd_retire_sql(a: argparse.Namespace) -> int:
    """SQL for D1 after a retirement (or a restore): v2_migration.status verified -> retired, from the retire/restore ledgers.

    A study counts as retired when its last ledger line (over all files, by time) is a successful `retired`; `restored` undoes it.
    `UPDATE ... WHERE status = 'verified'` only ever moves a verified study, so applying a file twice is harmless. Each updated row
    is about 3 D1 row writes (the status index), well inside a day's budget for the whole atlas."""
    last: Dict[str, dict] = {}
    for f in a.ledger:
        for ln in Path(f).read_text("utf-8").splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("stage") in ("retired", "restored") and rec.get("ok") and re.fullmatch(r"GSE\d+", str(rec.get("gse"))):
                if rec["gse"] not in last or rec.get("at", "") >= last[rec["gse"]].get("at", ""):
                    last[rec["gse"]] = rec
    retired = vc.accession_order(g for g, r in last.items() if r["stage"] == "retired")
    restored = vc.accession_order(g for g, r in last.items() if r["stage"] == "restored")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.sql"):
        f.unlink()
    names: List[str] = []
    now = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
    for tag, gses, new, old in (("retired", retired, "retired", "verified"), ("restored", restored, "verified", "retired")):
        for i in range(0, len(gses), 500):
            ids = ", ".join(vc.sql_literal(g) for g in gses[i : i + 500])
            name = f"{tag}_{i // 500:04d}.sql"
            sql = f"UPDATE v2_migration SET status = '{new}', updated_at = {now} WHERE status = '{old}' AND gse_id IN ({ids});"
            (out / name).write_text(sql + "\n", encoding="utf-8", newline="\n")
            names.append(name)
    lines = [f"wrangler d1 execute {a.database} --remote --file={n}" for n in names]
    (out / "APPLY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(retired)} studies to mark retired, {len(restored)} to mark verified again, {len(names)} files in {out} "
          f"(about {3 * (len(retired) + len(restored))} D1 row writes)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["retire", "restore", "audit", "retire-sql"])
    ap.add_argument("--gse", nargs="*")
    ap.add_argument("--gse-list")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", type=vc.parse_shard, metavar="I/N")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--public", default=vc.PUBLIC_BUCKET)
    ap.add_argument("--archive", default=ARCHIVE_BUCKET)
    ap.add_argument("--storage-class", default="STANDARD_IA", help="STANDARD_IA = R2 Infrequent Access; 'none' for the bucket default")
    ap.add_argument("--ledger", help="local JSONL record of what was done")
    ap.add_argument("--ledger-files", nargs="*", default=[], help="retire-sql: the ledgers written by retire/restore")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--apply", action="store_true", help="write; without it only reads and prints")
    ap.add_argument("--confirm", help=f"retire --apply needs --confirm {CONFIRM}")
    ap.add_argument("--allow-partial", action="store_true", help="retire --all even though some studies have no v2 manifest")
    ap.add_argument("--force", action="store_true", help="restore over a different object at the public key")
    ap.add_argument("--expect-none", action="store_true", help="audit exits 1 if any legacy bundle remains")
    ap.add_argument("--local-root", help="use directories under this root instead of R2 (tests)")
    ap.add_argument("--out", help="retire-sql: output directory")
    ap.add_argument("--database", default="singlet-catalog", help="retire-sql: D1 database name for APPLY.txt")
    a = ap.parse_args(argv)
    if a.public == a.archive:
        ap.error("--archive must differ from --public")
    if a.command == "retire-sql":
        if not a.out or not a.ledger_files:
            ap.error("retire-sql needs --out and one or more --ledger files")
        a.ledger = a.ledger_files
        return cmd_retire_sql(a)
    store = vc.LocalStore(Path(a.local_root)) if a.local_root else vc.S3Store()
    return {"retire": cmd_retire, "restore": cmd_restore, "audit": cmd_audit}[a.command](store, a)


if __name__ == "__main__":
    sys.exit(main())
