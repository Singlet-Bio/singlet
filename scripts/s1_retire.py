#!/usr/bin/env python3
"""Retire legacy `.singlet` bundles in three separate steps, so the site never emits a URL that is dead.

  (A) archive    server-side copy `singlet-data/data/<GSE>/<GSE>.singlet` to `singlet-archive/<same key>`; verify (size and ETag, or a
                 SHA-256 read-back when the copy is multipart); append the study to the ARCHIVED LIST (`--list`, JSON lines).
                 Never deletes anything. Refuses unless the study's .s1 manifest exists, names the object that is live now (same
                 ETag and size: the bundle was not rebuilt after it was migrated), every sample object in the manifest is present,
                 and the ledger ends in a committed, verified migration.
  (B) catalog    `s1_campaign.py retire-sql --list archived-*.jsonl --out DIR` writes the SQL that flips `s1_migration.status` from
                 `verified` to `retired` for the archived studies; the orchestrator applies it with wrangler. From then on the site
                 stops offering the legacy file. (Studies that never had a legacy file are posted as `retired` directly.)
  (C) delete     remove the legacy object from `singlet-data`, only for a study that is in the archived list AND whose catalog row
                 has been `retired` for at least --min-retired-age-minutes (5) AND `verified` for at least --min-verified-age-hours
                 (24): cached answers that still name the legacy URL live up to a day. The cluster cannot read D1, so the ages come from
                 a small CSV or JSON the orchestrator exports (`--ages`): gse_id, verified_at, updated_at of the rows with
                 status = 'retired'. A study missing from that file is not retired in the catalog yet and is left alone. Before it
                 deletes, it checks again that the archive copy is there and equal to the list, that the legacy object is still the
                 one that was archived, and that the .s1 manifest and all its sample objects are present.

`restore` copies an archived bundle back to the public key (verified the same way) and appends it to `--list` as restored, so that
`retire-sql` can flip the catalog back. `audit` lists every `data/*.singlet` key still in the public bucket and how far the archive is.
Dry run unless `--apply`; `delete --apply` also needs `--confirm RETIRE`. `archive --all` refuses to start while any study still lacks a
manifest (`--allow-partial` to override).

  python scripts/s1_retire.py archive --all --shard 0/8 --apply --list archived-0.jsonl
  python scripts/s1_campaign.py retire-sql --list archived-*.jsonl --out sql            # (B): applied by the orchestrator
  # D1 -> ages.json (laptop): wrangler d1 execute singlet-catalog --remote --json \\
  #     --command "SELECT gse_id, verified_at, updated_at FROM s1_migration WHERE status = 'retired'" > ages.json
  python scripts/s1_retire.py delete  --list archived.jsonl --ages ages.json                        # dry run
  python scripts/s1_retire.py delete  --list archived.jsonl --ages ages.json --apply --confirm RETIRE
  python scripts/s1_retire.py restore --gse GSE111113 --apply --list restored.jsonl
  python scripts/s1_retire.py audit   [--expect-none]

The archive bucket (`singlet-archive`) is created only if it does not exist (needs a token that may create buckets; else create it
with `wrangler r2 bucket create singlet-archive --storage-class InfrequentAccess`). Credentials as for s1_campaign.py. Archived
bundles are meant to be deleted after the grace period in docs/s1-migration.md (D12).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import io
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_campaign as vc  # noqa: E402

ARCHIVE_BUCKET = "singlet-archive"
CONFIRM = "RETIRE"
_log_lock = threading.Lock()


class Refused(Exception):
    """A precondition failed; nothing was changed for this study."""


class NotYet(Exception):
    """Expected to change with time or after the catalog step: not an error."""


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
        raise Refused("no readable .s1 manifest")
    if m.get("format") not in ("singlet-s1", "singlet-store", "singlet-v2") or m.get("gse") != gse or not m.get("samples"):
        raise Refused("manifest is not a singlet-s1 manifest of this study")
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
    """Whether the archive bucket exists (asked once per run; a dry run of a first archive finds none)."""
    if not hasattr(a, "_archive_exists"):
        a._archive_exists = store.bucket_exists(a.archive)
    return a._archive_exists


def storage_class(a: argparse.Namespace) -> Optional[str]:
    return None if a.storage_class.lower() in ("none", "") else a.storage_class


# ── the archived list ─────────────────────────────────────────────────────────────────────────────────────────────────


def load_list(paths: Sequence[str]) -> Dict[str, dict]:
    return vc.load_event_list(paths)


# ── (A) archive ───────────────────────────────────────────────────────────────────────────────────────────────────────


def archive_one(store, a: argparse.Namespace, gse: str, ledger: Optional[Path], pool: cf.Executor, guard) -> str:
    key = vc.legacy_key(gse)
    live = store.head(a.public, key)
    arch = store.head(a.archive, key) if archive_exists(store, a) else None
    if live is None:
        return "legacy object already gone" + (" (an archive copy exists)" if arch is not None else ", no archive copy: SKIPPED")
    m = preconditions(store, a, gse, live, pool)
    want_sha = (m.get("legacy") or {}).get("sha256")
    method = None
    etag = arch.etag if arch is not None else None
    multipart = bool(arch is not None and arch.etag != live.etag)
    if arch is not None and arch.size == live.size:
        try:
            method = verify_copy(store, a, live, key, a.archive, key, want_sha, a.public)
        except Refused:
            method = None  # a partial or foreign object at the archive key: copied over below
    if method is None:
        if not a.apply:
            return f"would-archive: copy {live.size} bytes to {a.archive}/{key} and verify (nothing is deleted)"
        etag, multipart = guard.copy(a.public, key, a.archive, key, size=live.size, storage_class=storage_class(a))
        method = verify_copy(store, a, live, key, a.archive, key, want_sha, a.public)
    elif not a.apply:
        return "would-list (archive copy already present and verified)"
    again = store.head(a.public, key)
    if again is None or again.etag != live.etag or again.size != live.size:
        raise Refused("the legacy object changed while it was being archived; the study is not listed as archived")
    dh = store.head(a.archive, key)
    rec = {"event": "archived", "gse": gse, "key": key, "bytes": live.size, "etag": live.etag, "archive_bucket": a.archive,
           "archive_etag": dh.etag if dh is not None else etag, "multipart": multipart, "verified_by": method, "sha256": want_sha}
    if a.list:
        log(Path(a.list), rec)
    log(ledger, {"gse": gse, "stage": "archived", "ok": True, **{k: v for k, v in rec.items() if k not in ("event", "gse")}})
    vc.write_ledger(guard, a.public, gse, vc.read_ledger(store, a.public, gse) + [{"at": vc.iso(), "gse": gse, **rec}])
    return f"archived {live.size} bytes (verified by {method}); the legacy object is still in {a.public}"


# ── (C) delete ────────────────────────────────────────────────────────────────────────────────────────────────────────


def parse_ts(v: Any) -> Optional[dt.datetime]:
    """A timestamp as D1 or wrangler writes it (`2026-10-22T15:36:41Z`, `2026-10-22 15:36:41`, with an offset), as UTC; None if absent."""
    s = str(v).strip() if v is not None else ""
    if not s:
        return None
    if re.match(r"\d{4}-\d\d-\d\d \d", s):
        s = s.replace(" ", "T", 1)
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def load_ages(path: str) -> Dict[str, Tuple[Optional[dt.datetime], Optional[dt.datetime]]]:
    """{gse_id: (verified_at, updated_at)} for the catalog rows with status 'retired', from the orchestrator's export: a CSV
    (header gse_id,verified_at,updated_at[,status]), a JSON list of such objects, or `wrangler d1 execute --json` output."""
    text = Path(path).read_text("utf-8-sig")
    rows: List[dict] = []
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    if doc is not None:
        items = doc if isinstance(doc, list) else [doc]
        for it in items:
            if isinstance(it, dict) and "results" in it:  # wrangler: [{"results": [...], "success": true, "meta": {...}}]
                rows.extend(r for r in it["results"] if isinstance(r, dict))
            elif isinstance(it, dict):
                rows.append(it)
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
    out: Dict[str, Tuple[Optional[dt.datetime], Optional[dt.datetime]]] = {}
    for r in rows:
        g = str(r.get("gse_id") or "").strip()
        if not re.fullmatch(r"GSE\d+", g):
            continue
        if r.get("status") not in (None, "", "retired"):
            continue  # an export that carries the status column: only retired rows count
        out[g] = (parse_ts(r.get("verified_at")), parse_ts(r.get("updated_at")))
    return out


def delete_one(store, a: argparse.Namespace, gse: str, entry: dict, ages: Dict[str, tuple], now: dt.datetime, ledger: Optional[Path],
               pool: cf.Executor, guard) -> str:
    if entry.get("event") == "restored":
        raise NotYet("restored after it was archived: it is a live legacy bundle again")
    key = vc.legacy_key(gse)
    # the catalog first: the site must already have stopped offering the file
    if gse not in ages:
        raise NotYet("not 'retired' in the catalog yet (not in the ages file): apply the retire SQL first")
    verified_at, retired_at = ages[gse]
    if verified_at is None or retired_at is None:
        raise Refused("the ages file has no verified_at / updated_at for this study")
    v_age = (now - verified_at).total_seconds() / 3600.0
    r_age = (now - retired_at).total_seconds() / 60.0
    if v_age < a.min_verified_age_hours:
        raise NotYet(f"verified {v_age:.1f} h ago, needs {a.min_verified_age_hours:g} h (from {(verified_at + dt.timedelta(hours=a.min_verified_age_hours)).strftime('%Y-%m-%dT%H:%MZ')})")
    if r_age < a.min_retired_age_minutes:
        raise NotYet(f"retired {r_age:.1f} min ago, needs {a.min_retired_age_minutes:g} min")
    live = store.head(a.public, key)
    arch = store.head(a.archive, key) if archive_exists(store, a) else None
    if live is None:
        return "already-deleted" + ("" if arch is not None else " (WARNING: no archive copy either)")
    if arch is None:
        raise Refused("no archive copy")
    if entry.get("bytes") is not None and arch.size != entry["bytes"]:
        raise Refused(f"the archive copy has {arch.size} bytes, the list says {entry['bytes']}")
    if entry.get("archive_etag") and arch.etag != entry["archive_etag"]:
        raise Refused("the archive copy is not the one in the list (ETag differs)")
    if entry.get("etag") and (live.etag != entry["etag"] or live.size != entry.get("bytes", live.size)):
        raise Refused("the legacy object is not the one that was archived (changed since)")
    if not entry.get("archive_etag"):  # a bare id in the list: verify the copy from scratch
        m0 = load_manifest(store, a.public, gse)
        verify_copy(store, a, live, key, a.archive, key, ((m0 or {}).get("legacy") or {}).get("sha256"), a.public)
    preconditions(store, a, gse, live, pool)  # the manifest still names this object and every .s1 sample object is there
    if not a.apply:
        return f"would-delete {live.size} bytes (verified {v_age:.1f} h, retired {r_age:.1f} min ago; archive copy in {a.archive})"
    again = store.head(a.public, key)
    if again is None or again.etag != live.etag or again.size != live.size:
        raise Refused("the legacy object changed just now; nothing was deleted")
    guard.delete(a.public, key)
    if store.head(a.public, key) is not None:
        raise Refused("delete did not remove the object")
    rec = {"event": "legacy-deleted", "gse": gse, "key": key, "bytes": live.size, "etag": live.etag, "archive": f"{a.archive}/{key}",
           "verified_age_h": round(v_age, 2), "retired_age_min": round(r_age, 2)}
    log(ledger, {"gse": gse, "stage": "deleted", "ok": True, **{k: v for k, v in rec.items() if k not in ("event", "gse")}})
    vc.write_ledger(guard, a.public, gse, vc.read_ledger(store, a.public, gse) + [{"at": vc.iso(), **rec}])
    return f"deleted {live.size} bytes (archive copy kept in {a.archive})"


# ── restore ───────────────────────────────────────────────────────────────────────────────────────────────────────────


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
                if a.apply and a.list:
                    log(Path(a.list), {"event": "restored", "gse": gse, "key": key, "bytes": arch.size, "etag": live.etag})
                return "already-present"
            except Refused:
                pass
        if not a.force:
            return "SKIPPED: the public key exists with different content (--force to overwrite)"
    if not a.apply:
        return f"would-restore {arch.size} bytes from {a.archive}/{key}"
    etag, multipart = guard.copy(a.archive, key, a.public, key, size=arch.size, storage_class=None)
    method = verify_copy(store, a, arch, key, a.public, key, want_sha, a.archive)
    rec = {"event": "restored", "gse": gse, "key": key, "bytes": arch.size, "etag": etag, "multipart": multipart, "verified_by": method}
    if a.list:
        log(Path(a.list), rec)
    log(ledger, {"gse": gse, "stage": "restored", "ok": True, **{k: v for k, v in rec.items() if k not in ("event", "gse")}})
    vc.write_ledger(guard, a.public, gse, vc.read_ledger(store, a.public, gse) + [{"at": vc.iso(), **rec}])
    return f"restored {arch.size} bytes (verified by {method}); flip the catalog back with retire-sql if it was marked retired"


# ── selecting studies, running ────────────────────────────────────────────────────────────────────────────────────────


def studies_to_visit(store, a: argparse.Namespace, source: str) -> List[str]:
    """Named studies, or all of them in accession order. `source` is "manifests" (archive: the .s1 manifests, which stay for good,
    plus the legacy studies that have none, so --shard I/N keeps meaning the same studies) or a bucket whose data/ keys to list."""
    if a.gse:
        gses = list(a.gse)
    elif a.gse_list:
        gses = vc.read_list(a.gse_list)
    elif source == "manifests":
        have = {m.group(1) for k, _ in vc.list_manifest_keys(store, a.public) if (m := vc.RE_MANIFEST.match(k))}
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
    have = {m.group(1) for k, _ in vc.list_manifest_keys(store, a.public) if (m := vc.RE_MANIFEST.match(k))}
    missing = vc.accession_order(legacy - have)
    if missing:
        raise SystemExit(f"refusing to archive: {len(missing)} of {len(legacy)} legacy studies have no .s1 manifest yet "
                         f"(first: {', '.join(missing[:10])}). Finish the migration, or name studies with --gse, or pass --allow-partial.")


def run_over(store, a: argparse.Namespace, fn, gses: Sequence[str]) -> int:
    ledger = Path(a.ledger) if a.ledger else None
    bad = notyet = 0

    def one(g: str) -> str:
        try:
            return fn(g)
        except NotYet as e:
            return f"NOT-YET: {e}"
        except Refused as e:
            log(ledger, {"gse": g, "ok": False, "stage": "refused", "error": str(e)[:300]})
            return f"REFUSED: {e}"
        except Exception as e:  # noqa: BLE001  one study must not stop the shard
            log(ledger, {"gse": g, "ok": False, "stage": "error", "error": f"{type(e).__name__}: {e}"[:300]})
            return f"FAILED: {type(e).__name__}: {e}"

    with cf.ThreadPoolExecutor(max(1, a.workers)) as ex:
        for g, msg in zip(gses, ex.map(one, gses)):
            bad += msg.startswith(("REFUSED", "FAILED", "SKIPPED"))
            notyet += msg.startswith("NOT-YET")
            print(f"{g}\t{msg}", flush=True)
    if notyet:
        print(f"[retire] {notyet} studies not ready yet (see NOT-YET lines); run again later", flush=True)
    return 1 if bad else 0


def cmd_archive(store, a: argparse.Namespace) -> int:
    if not (a.gse or a.gse_list or a.all):
        print("error: give --gse, --gse-list or --all", file=sys.stderr)
        return 2
    if a.apply and not a.list:
        print("error: --apply needs --list FILE (the archived list that the catalog step and the delete step read)", file=sys.stderr)
        return 2
    if a.all and not a.allow_partial:
        completeness_gate(store, a)
    if not archive_exists(store, a):
        if not a.apply:
            print(f"[archive] archive bucket {a.archive} does not exist; --apply would create it")
        else:
            store.create_bucket(a.archive)
            a._archive_exists = True
            print(f"[archive] created bucket {a.archive}")
    # this step can never delete: the guard has no delete permission at all
    guard = vc.GuardedStore(store, {a.archive: ("data/",), a.public: ("s1/ledger/",)})
    pool = cf.ThreadPoolExecutor(max(1, a.workers))
    ledger = Path(a.ledger) if a.ledger else None
    try:
        return run_over(store, a, lambda g: archive_one(store, a, g, ledger, pool, guard), studies_to_visit(store, a, "manifests"))
    finally:
        pool.shutdown()


def cmd_delete(store, a: argparse.Namespace) -> int:
    if not a.list or not a.ages:
        print("error: delete needs --list (the archived list) and --ages (the catalog export)", file=sys.stderr)
        return 2
    if a.apply and a.confirm != CONFIRM:
        print(f"error: --apply needs --confirm {CONFIRM}", file=sys.stderr)
        return 2
    entries = load_list([a.list])
    ages = load_ages(a.ages)
    now = parse_ts(a.now) if a.now else dt.datetime.now(dt.timezone.utc)
    if now is None:
        print("error: --now is not a timestamp", file=sys.stderr)
        return 2
    gses = vc.accession_order(entries)
    if a.shard:
        i, n = a.shard
        gses = gses[i::n]
    print(f"[delete] {len(gses)} studies in the archived list, {len(ages)} retired rows in the ages file, "
          f"min verified age {a.min_verified_age_hours:g} h, min retired age {a.min_retired_age_minutes:g} min, now {now.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    guard = vc.GuardedStore(store, {a.public: ("s1/ledger/",)}, allow_delete={a.public: ("data/",)})
    pool = cf.ThreadPoolExecutor(max(1, a.workers))
    ledger = Path(a.ledger) if a.ledger else None
    try:
        return run_over(store, a, lambda g: delete_one(store, a, g, entries[g], ages, now, ledger, pool, guard), gses)
    finally:
        pool.shutdown()


def cmd_restore(store, a: argparse.Namespace) -> int:
    if not (a.gse or a.gse_list or a.all):
        print("error: give --gse, --gse-list or --all", file=sys.stderr)
        return 2
    guard = vc.GuardedStore(store, {a.public: ("data/", "s1/ledger/")})
    ledger = Path(a.ledger) if a.ledger else None
    return run_over(store, a, lambda g: restore_one(store, a, g, ledger, guard), studies_to_visit(store, a, a.archive))


def cmd_audit(store, a: argparse.Namespace) -> int:
    remaining = [(k, s) for k, s in store.list_keys(a.public, "data/") if k.endswith(".singlet")]
    legacy = [(vc.RE_LEGACY.match(k).group(1), s) for k, s in remaining if vc.RE_LEGACY.match(k)]
    odd = [k for k, _ in remaining if not vc.RE_LEGACY.match(k)]
    manifests = {m.group(1) for k, _ in vc.list_manifest_keys(store, a.public) if (m := vc.RE_MANIFEST.match(k))}
    archived = {g for g, _ in store.list_studies(a.archive)} if store.bucket_exists(a.archive) else set()
    print("gse\tbytes\thas_store_manifest\tarchived")
    for g, s in sorted(legacy, key=lambda t: (len(t[0]), t[0])):
        print(f"{g}\t{s}\t{int(g in manifests)}\t{int(g in archived)}")
    for k in odd:
        print(f"{k}\t?\t?\t?\t# not a data/<GSE>/<GSE>.singlet key")
    print(f"[audit] {len(legacy)} legacy bundles ({sum(s for _g, s in legacy) / 1e9:.2f} GB) remain in {a.public}; "
          f"{len(archived)} archived in {a.archive}; {len(manifests)} .s1 manifests; {len(odd)} unexpected .singlet keys", flush=True)
    return 1 if (a.expect_none and (legacy or odd)) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["archive", "delete", "restore", "audit"])
    ap.add_argument("--gse", nargs="*")
    ap.add_argument("--gse-list")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", type=vc.parse_shard, metavar="I/N")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--public", default=vc.PUBLIC_BUCKET)
    ap.add_argument("--archive", default=ARCHIVE_BUCKET)
    ap.add_argument("--storage-class", default="STANDARD_IA", help="STANDARD_IA = R2 Infrequent Access; 'none' for the bucket default")
    ap.add_argument("--list", help="archive/restore: the JSON-lines list that is appended to; delete: the archived list to read")
    ap.add_argument("--ages", help="delete: CSV/JSON export of the catalog rows with status 'retired' (gse_id, verified_at, updated_at)")
    ap.add_argument("--min-verified-age-hours", type=float, default=24.0)
    ap.add_argument("--min-retired-age-minutes", type=float, default=5.0)
    ap.add_argument("--now", help="delete: the time to measure ages against (default: now; for tests and rehearsals)")
    ap.add_argument("--ledger", help="local JSONL record of what was done")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--apply", action="store_true", help="write; without it only reads and prints")
    ap.add_argument("--confirm", help=f"delete --apply needs --confirm {CONFIRM}")
    ap.add_argument("--allow-partial", action="store_true", help="archive --all even though some studies have no .s1 manifest")
    ap.add_argument("--force", action="store_true", help="restore over a different object at the public key")
    ap.add_argument("--expect-none", action="store_true", help="audit exits 1 if any legacy bundle remains")
    ap.add_argument("--local-root", help="use directories under this root instead of R2 (tests)")
    a = ap.parse_args(argv)
    if a.public == a.archive:
        ap.error("--archive must differ from --public")
    store = vc.LocalStore(Path(a.local_root)) if a.local_root else vc.S3Store()
    return {"archive": cmd_archive, "delete": cmd_delete, "restore": cmd_restore, "audit": cmd_audit}[a.command](store, a)


if __name__ == "__main__":
    sys.exit(main())
