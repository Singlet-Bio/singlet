#!/usr/bin/env python3
"""Split already-published .singlet bundles into a public file and a private restricted sidecar.

Nothing is stored twice: the members that carry per-donor germline information move from the public object to a
sidecar in a private bucket, so total storage stays what it was (it is higher only while a study is in flight).
For each study, in this order:

  1. download the public object; refuse it if any member does not match the checksum in its own manifest;
  2. write the public form and the sidecar locally, then read both back: every member against its checksum, nothing
     restricted in the public file, and public + sidecar equal to the original member for member;
  3. upload the sidecar to the private bucket (restricted/<GSE>/<GSE>.restricted.singlet) and read it back: SHA-256
     of the stored object must equal the local file;
  4. only then overwrite the public object, and read that back the same way (on a mismatch the original, still in
     scratch, is put back).

`restore` rebuilds the complete bundle from the public object plus the sidecar and uploads it over the public key
(same members, same checksums and the original manifest; not the same bytes as the original zip).

Both commands are dry runs unless --apply is given; a dry run only reads (HEAD and the zip central directory) and prints
the counts. Each completed study is appended to a JSONL ledger (--ledger), the record to cite.

Credentials (S3 API of the Cloudflare account): R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, in the environment
of the job. Run it on a compute node with scratch for about three copies of the largest bundle (~30 GB each).

  python scripts/restrict_published_bundles.py migrate --gse-list gses.txt --private singlet-restricted --ledger ledger.jsonl [--apply]
  python scripts/restrict_published_bundles.py migrate --all --shard 0/8 --private singlet-restricted --ledger ledger-0.jsonl --apply
  python scripts/restrict_published_bundles.py restore --gse GSE128639      --private singlet-restricted --ledger ledger.jsonl [--apply]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional


def _load_restricted():
    """singlet/_restricted.py is standard library only; load it by path so a node needs no installed package."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "python" / "singlet" / "_restricted.py"
    spec = importlib.util.spec_from_file_location("singlet_restricted", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_r = _load_restricted()
redact_bundle, restore_bundle, restricted_summary = (
    _r.redact_bundle,
    _r.restore_bundle,
    _r.restricted_summary,
)

CHUNK = 64 * 1024 * 1024
_KEY_RE = re.compile(r"^data/(GSE\d+)/\1\.singlet$")


def public_key(gse: str) -> str:
    return f"data/{gse}/{gse}.singlet"


def sidecar_key(gse: str) -> str:
    return f"restricted/{gse}/{gse}.restricted.singlet"


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

    def download(self, bucket: str, key: str, path: Path) -> None:
        self.c.download_file(bucket, key, str(path), Config=self.cfg)

    def list_studies(self, bucket: str) -> list:
        """[(gse, bytes)] for every data/<GSE>/<GSE>.singlet."""
        out = []
        for page in self.c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="data/"):
            for o in page.get("Contents", []):
                m = _KEY_RE.match(o["Key"])
                if m:
                    out.append((m.group(1), int(o["Size"])))
        return out

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

    def download(self, bucket: str, key: str, path: Path) -> None:
        shutil.copyfile(self._p(bucket, key), path)

    def list_studies(self, bucket: str) -> list:
        base = self.root / bucket / "data"
        return [
            (d.name, (d / f"{d.name}.singlet").stat().st_size)
            for d in sorted(base.glob("GSE*"))
            if (d / f"{d.name}.singlet").exists()
        ]

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


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_migrate(store, gse: str, a, ledger: Path) -> str:
    pk, sk = public_key(gse), sidecar_key(gse)
    size = store.size(a.public, pk)
    if size is None:
        return "missing"
    summ = central_directory_summary(store, a.public, pk, size)
    if summ["members"] == 0:
        return "nothing-restricted"
    if not a.apply:
        return f"would-split: {summ['members']} members, {summ['compressed_bytes']} of {size} bytes move to the sidecar"
    etag = store.etag(a.public, pk)
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        src, dst, side = (
            Path(td) / "src.singlet",
            Path(td) / "public.singlet",
            Path(td) / "side.singlet",
        )
        store.download(a.public, pk, src)
        if src.stat().st_size != size:
            return "SKIPPED: download size differs from the object"
        h_src = sha256_file(src)
        stats = redact_bundle(
            src, dst, sidecar=side
        )  # refuses a corrupt source; reads both outputs back
        h_side, h_pub = sha256_file(side), sha256_file(dst)
        if store.etag(a.public, pk) != etag:
            return "SKIPPED: public object changed while it was being split"

        store.upload(side, a.private, sk)
        side_size = side.stat().st_size
        if (
            store.size(a.private, sk) != side_size
            or sha256_object(store, a.private, sk, side_size) != h_side
        ):
            log(ledger, {"gse": gse, "ok": False, "stage": "sidecar-upload"})
            return "VERIFY-FAILED: sidecar in the private bucket does not match; public object untouched"
        log(
            ledger,
            {
                "gse": gse,
                "stage": "sidecar-verified",
                "ok": True,
                "sidecar_bytes": side_size,
                "sha256_sidecar": h_side,
            },
        )

        pub_size = dst.stat().st_size
        store.upload(dst, a.public, pk)
        if (
            store.size(a.public, pk) != pub_size
            or sha256_object(store, a.public, pk, pub_size) != h_pub
        ):
            store.upload(
                src, a.public, pk
            )  # put the original back; the sidecar is still valid for a retry
            log(ledger, {"gse": gse, "ok": False, "stage": "public-upload"})
            return "UPLOAD-FAILED: public object did not read back; original restored"
        log(
            ledger,
            {
                "gse": gse,
                "stage": "done",
                "ok": True,
                "bytes_before": size,
                "bytes_public": pub_size,
                "bytes_sidecar": side_size,
                "removed": stats["removed"],
                "removed_bytes": stats["removed_bytes"],
                "sha256_original": h_src,
                "sha256_public": h_pub,
                "sha256_sidecar": h_side,
            },
        )
    return f"split {size} -> public {pub_size} + sidecar {side_size} bytes"


def cmd_restore(store, gse: str, a, ledger: Path) -> str:
    pk, sk = public_key(gse), sidecar_key(gse)
    if store.size(a.private, sk) is None:
        return "SKIPPED: no sidecar for this study"
    if not a.apply:
        return "would-rebuild the complete bundle from the public file and the sidecar"
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        pub, side, out = (
            Path(td) / "public.singlet",
            Path(td) / "side.singlet",
            Path(td) / "full.singlet",
        )
        store.download(a.public, pk, pub)
        store.download(a.private, sk, side)
        res = restore_bundle(pub, side, out)
        store.upload(out, a.public, pk)
        if store.size(a.public, pk) != out.stat().st_size:
            return "UPLOAD-FAILED: size mismatch after restore"
        log(ledger, {"gse": gse, "stage": "restored", "ok": True, "bytes": res["bytes_out"]})
    return "restored"


def gses_from(a, store) -> Iterable[str]:
    """The studies to visit: given, listed in a file, or every study in the public bucket (smallest first).

    --shard I/N keeps every N-th study of that order, so N jobs split the work evenly and each run is repeatable.
    """
    if a.gse:
        gses = list(a.gse)
    elif a.gse_list:
        gses = [ln.strip() for ln in Path(a.gse_list).read_text().splitlines() if ln.strip()]
    else:
        gses = [g for g, _ in sorted(store.list_studies(a.public), key=lambda t: (t[1], t[0]))]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        gses = gses[i::n]
    return gses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["migrate", "restore"])
    ap.add_argument("--gse", nargs="*")
    ap.add_argument("--gse-list")
    ap.add_argument("--all", action="store_true", help="every study in the public bucket")
    ap.add_argument("--shard", metavar="I/N", help="take every N-th study, starting at I (0-based)")
    ap.add_argument("--limit", type=int, help="stop after this many studies")
    ap.add_argument("--public", default="singlet-data")
    ap.add_argument("--private", required=True, help="private bucket (never the public one)")
    ap.add_argument("--ledger", required=True)
    ap.add_argument(
        "--scratch", default=None, help="directory with room for ~3 copies of the largest bundle"
    )
    ap.add_argument("--apply", action="store_true", help="write; without it only reads and prints")
    ap.add_argument("--local-root", help="use directories under this root instead of R2 (tests)")
    a = ap.parse_args(argv)
    if not (a.gse or a.gse_list or a.all):
        ap.error("give --gse, --gse-list or --all")
    if a.shard and not re.fullmatch(r"\d+/[1-9]\d*", a.shard):
        ap.error("--shard is I/N, e.g. 0/8")
    if a.public == a.private:
        ap.error("--private must differ from --public")

    store = LocalStore(Path(a.local_root)) if a.local_root else S3Store()
    ledger = Path(a.ledger)
    bad = n = 0
    for gse in gses_from(a, store):
        if a.limit is not None and n >= a.limit:
            break
        msg = (cmd_migrate if a.command == "migrate" else cmd_restore)(store, gse, a, ledger)
        n += not msg.startswith(("nothing-restricted", "missing"))
        bad += msg.startswith(("VERIFY", "UPLOAD", "SKIPPED"))
        print(f"{gse}\t{msg}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
