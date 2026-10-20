#!/usr/bin/env python3
"""Take the per-donor germline members out of already-published .singlet bundles, keeping a private original.

Two phases, so nothing public changes until every private copy has been verified:

  copy    For each study: copy the public object to the private bucket (key full/<GSE>/<GSE>.singlet), then read both
          back and compare SHA-256. The public bucket is only read.
  redact  For each study whose copy is recorded as verified: download the original, write the public form
          (singlet._restricted.redact_bundle: restricted members out, ancestry call trimmed, manifest rewritten,
          read back and checked), and overwrite the public object. Skips studies that hold nothing restricted.

Both phases are dry runs unless --apply is given; a dry run only reads (HEAD and the zip central directory) and prints
the counts. Progress is appended to a JSONL ledger (--ledger), which is also the record to cite and what a rollback
reads: `rollback` copies full/<GSE>/<GSE>.singlet back over the public key.

Credentials (S3 API of the Cloudflare account): R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.
Run it on a compute node with scratch space for one bundle at a time (the largest is ~30 GB).

  python scripts/restrict_published_bundles.py copy   --gse-list gses.txt --public singlet-data --private <bucket> --ledger ledger.jsonl [--apply]
  python scripts/restrict_published_bundles.py redact --gse-list gses.txt --public singlet-data --private <bucket> --ledger ledger.jsonl [--apply]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from singlet._restricted import redact_bundle, restricted_summary

CHUNK = 64 * 1024 * 1024


def public_key(gse: str) -> str:
    return f"data/{gse}/{gse}.singlet"


def private_key(gse: str) -> str:
    return f"full/{gse}/{gse}.singlet"


# ── storage ──────────────────────────────────────────────────────────────────


class S3Store:
    """Two R2 buckets over the S3 API."""

    def __init__(self) -> None:
        import boto3
        from boto3.s3.transfer import TransferConfig

        acct = os.environ["R2_ACCOUNT_ID"]
        self.c = boto3.client(
            "s3",
            endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self.cfg = TransferConfig(
            multipart_threshold=CHUNK, multipart_chunksize=CHUNK, max_concurrency=8
        )
        # Server-side copies move no data through this machine, so use big parts: fewer billed operations.
        self.cfg_copy = TransferConfig(
            multipart_threshold=512 * 1024 * 1024,
            multipart_chunksize=512 * 1024 * 1024,
            max_concurrency=8,
        )

    def size(self, bucket: str, key: str) -> Optional[int]:
        try:
            return int(self.c.head_object(Bucket=bucket, Key=key)["ContentLength"])
        except self.c.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def etag(self, bucket: str, key: str) -> str:
        return str(self.c.head_object(Bucket=bucket, Key=key)["ETag"]).strip('"')

    def read_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        return self.c.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")[
            "Body"
        ].read()

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
        self.c.copy(
            {"Bucket": src_bucket, "Key": src_key}, dst_bucket, dst_key, Config=self.cfg_copy
        )  # server side; multipart above 512 MiB

    def download(self, bucket: str, key: str, path: Path) -> None:
        self.c.download_file(bucket, key, str(path), Config=self.cfg)

    def upload(self, path: Path, bucket: str, key: str) -> None:
        self.c.upload_file(str(path), bucket, key, Config=self.cfg)


class LocalStore:
    """Buckets as directories under a root; for tests and dry runs on a copy."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _p(self, bucket: str, key: str) -> Path:
        return self.root / bucket / key

    def size(self, bucket: str, key: str) -> Optional[int]:
        p = self._p(bucket, key)
        return p.stat().st_size if p.exists() else None

    def etag(self, bucket: str, key: str) -> str:
        return hashlib.md5(self._p(bucket, key).read_bytes()).hexdigest()

    def read_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        with open(self._p(bucket, key), "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
        dst = self._p(dst_bucket, dst_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._p(src_bucket, src_key), dst)

    def download(self, bucket: str, key: str, path: Path) -> None:
        shutil.copyfile(self._p(bucket, key), path)

    def upload(self, path: Path, bucket: str, key: str) -> None:
        dst = self._p(bucket, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)


def sha256_object(store, bucket: str, key: str, size: int) -> str:
    h = hashlib.sha256()
    for start in range(0, size, CHUNK):
        h.update(store.read_range(bucket, key, start, min(start + CHUNK, size) - 1))
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


# ── ledger ───────────────────────────────────────────────────────────────────


def read_ledger(path: Path) -> Dict[str, dict]:
    """Latest record per (gse, phase)."""
    out: Dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[f"{r['gse']}:{r['phase']}"] = r
    return out


def log(path: Path, rec: dict) -> None:
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


class RangeFile(io.RawIOBase):
    """A read-only file over ranged GETs, so zipfile can read the central directory without the whole object."""

    def __init__(self, store, bucket: str, key: str, size: int) -> None:
        self.store, self.bucket, self.key, self.size, self.pos = store, bucket, key, size, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if self.pos >= end:
            return b""
        data = self.store.read_range(self.bucket, self.key, self.pos, end - 1)
        self.pos = end
        return data


def central_directory_summary(store, bucket: str, key: str, size: int) -> dict:
    """restricted_summary() from the zip central directory only (ranged reads, not the whole object)."""
    return restricted_summary(RangeFile(store, bucket, key, size))


# ── phases ───────────────────────────────────────────────────────────────────


def phase_copy(store, gse: str, a, ledger: Path) -> str:
    pk, vk = public_key(gse), private_key(gse)
    size = store.size(a.public, pk)
    if size is None:
        return "missing"
    summ = central_directory_summary(store, a.public, pk, size)
    if summ["members"] == 0:
        return "nothing-restricted"
    if not a.apply:
        return f"would-copy {size} bytes ({summ['members']} restricted members, {summ['compressed_bytes']} bytes)"
    if store.size(a.private, vk) == size:
        status = "already-copied"
    else:
        store.copy(a.public, pk, a.private, vk)
        status = "copied"
    h_src = sha256_object(store, a.public, pk, size)
    h_dst = sha256_object(store, a.private, vk, size)
    if h_src != h_dst or store.size(a.private, vk) != size:
        log(ledger, {"gse": gse, "phase": "copy", "ok": False, "bytes": size})
        return "VERIFY-FAILED"
    log(
        ledger,
        {
            "gse": gse,
            "phase": "copy",
            "ok": True,
            "bytes": size,
            "sha256": h_src,
            "restricted_members": summ["members"],
            "restricted_bytes": summ["compressed_bytes"],
        },
    )
    return status


def phase_redact(store, gse: str, a, ledger: Path, led: Dict[str, dict]) -> str:
    pk = public_key(gse)
    size = store.size(a.public, pk)
    if size is None:
        return "missing"
    summ = central_directory_summary(store, a.public, pk, size)
    if summ["members"] == 0:
        return "nothing-restricted"
    done = led.get(f"{gse}:copy")
    if not (done and done.get("ok") and done.get("bytes") == size):
        return "SKIPPED: no verified private copy of this exact object in the ledger"
    if not a.apply:
        return f"would-redact: remove {summ['members']} members, {summ['compressed_bytes']} bytes of {size}"
    etag = store.etag(a.public, pk)
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        src, dst = Path(td) / "src.singlet", Path(td) / "public.singlet"
        store.download(a.public, pk, src)
        if sha256_file(src) != done["sha256"]:
            return "SKIPPED: public object differs from the copy that was verified"
        stats = redact_bundle(src, dst)  # reads the result back and checks every member
        if store.etag(a.public, pk) != etag:
            return "SKIPPED: public object changed while redacting"
        store.upload(dst, a.public, pk)
        new_size = store.size(a.public, pk)
        if new_size != dst.stat().st_size:
            log(ledger, {"gse": gse, "phase": "redact", "ok": False})
            return "UPLOAD-SIZE-MISMATCH"
        log(
            ledger,
            {
                "gse": gse,
                "phase": "redact",
                "ok": True,
                "bytes_before": size,
                "bytes_after": new_size,
                "removed": stats["removed"],
                "removed_bytes": stats["removed_bytes"],
                "sha256_public": sha256_file(dst),
            },
        )
    return f"redacted {size} → {new_size} bytes"


def phase_rollback(store, gse: str, a, ledger: Path, led: Dict[str, dict]) -> str:
    done = led.get(f"{gse}:copy")
    if not (done and done.get("ok")):
        return "SKIPPED: no verified private copy"
    if not a.apply:
        return "would-restore the original over the public key"
    store.copy(a.private, private_key(gse), a.public, public_key(gse))
    log(ledger, {"gse": gse, "phase": "rollback", "ok": True})
    return "restored"


def gses_from(a) -> Iterable[str]:
    if a.gse:
        return a.gse
    return [ln.strip() for ln in Path(a.gse_list).read_text().splitlines() if ln.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phase", choices=["copy", "redact", "rollback"])
    ap.add_argument("--gse", nargs="*")
    ap.add_argument("--gse-list")
    ap.add_argument("--public", default="singlet-data")
    ap.add_argument("--private", required=True, help="private bucket (never the public one)")
    ap.add_argument("--ledger", required=True)
    ap.add_argument(
        "--scratch", default=None, help="directory with room for one bundle (default: system temp)"
    )
    ap.add_argument("--apply", action="store_true", help="write; without it only reads and prints")
    ap.add_argument("--local-root", help="use directories under this root instead of R2 (tests)")
    a = ap.parse_args(argv)
    if not (a.gse or a.gse_list):
        ap.error("give --gse or --gse-list")
    if a.public == a.private:
        ap.error("--private must differ from --public")

    store = LocalStore(Path(a.local_root)) if a.local_root else S3Store()
    ledger = Path(a.ledger)
    led = read_ledger(ledger)
    bad = 0
    for gse in gses_from(a):
        if a.phase == "copy":
            msg = phase_copy(store, gse, a, ledger)
        elif a.phase == "redact":
            msg = phase_redact(store, gse, a, ledger, led)
        else:
            msg = phase_rollback(store, gse, a, ledger, led)
        bad += msg.startswith(("VERIFY", "UPLOAD", "SKIPPED"))
        print(f"{gse}\t{msg}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
