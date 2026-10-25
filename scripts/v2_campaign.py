#!/usr/bin/env python3
"""Singlet v2 migration campaign: legacy `.singlet` bundles in R2 -> v2 objects in R2, one study at a time.

Per study (GSE), in this order, stopping at the first failure (nothing public changes until the manifest is written):

  1. HEAD the live legacy bundle `data/<GSE>/<GSE>.singlet` (size, ETag) and read its zip central directory with ranged
     GETs: a bundle that still holds restricted members is refused (status blocked_restricted).
  2. Stream it to node-local scratch with ranged GETs pinned to that ETag (If-Match), hash it (sha256).
  3. `python -m singlet.v2.migrate --bundle B --out OUT --gse G` (the transcoder, W2).
  4. Validate the output tree (keys match the data Worker's grammar, every object's name carries its own digest, the
     manifest lists exactly these sample objects with the right sizes), then add `legacy` (key, ETag, bytes, sha256) to
     the manifest.
  5. `python -m singlet.v2.verify --bundle B --v2 OUT` must exit 0.
  6. Upload every packed object (samples, packs, cells) to R2 with the right Content-Type / Cache-Control; an object that
     already exists with the same size and ETag is skipped, so a rerun uploads nothing twice. Single-part uploads (<= part
     size) are checked against the local MD5, multipart uploads (fixed part size) against the composite ETag.
  7. Write `v2/ledger/<GSE>.jsonl`, then the manifest `v2/m/<GSE>.json` LAST (it is what makes the study visible), read
     the manifest back, then append the `committed` event to the ledger.

This script never writes, copies or deletes anything outside `v2/` in the public bucket (the store handed to the study
code refuses it). Retiring legacy bundles is `scripts/v2_retire.py`.

Subcommands
  run          the campaign (modes dry | pilot | full; --shard I/N, --gse, --gse-list, --resume)
  plan         list the legacy studies, assign shards, project wall time
  summarize    aggregate results files (counts, bytes, per-stage throughput, projection to the whole atlas)
  catalog-sql  SQL batch files (wrangler d1 execute --file) for v2_study / v2_sample / v2_migration
  catalog-post POST the same rows to the website's ingest endpoint (X-Ingest-Token)
  refs-index   publish the reference packs and v2/ref/index.json from the package
  retire-sql   step B of the retirement: SQL flipping v2_migration.status verified -> retired for the studies in the archived list

Credentials (S3 API of the Cloudflare account): R2_ACCOUNT_ID (or R2_ENDPOINT), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.
`--local-root DIR` replaces R2 with directories under DIR (tests, dry rehearsals on a copy).

  python scripts/v2_campaign.py run --mode dry   --shard 0/32 --results r.jsonl --scratch /tmp/x
  python scripts/v2_campaign.py run --mode pilot --pilot-n 6   --results r.jsonl --scratch /tmp/x
  python scripts/v2_campaign.py run --mode full  --shard 3/32 --resume --results r.jsonl --scratch /tmp/x
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MIB = 1 << 20
GIB = 1 << 30
PART_SIZE = 8 * MIB  # fixed multipart part size: the composite ETag of an object is then a function of its bytes
PUBLIC_BUCKET = "singlet-data"
REPO = Path(__file__).resolve().parents[1]

# ── key grammar: the data Worker's KEY_PATTERNS (singlet-website workers/data-v2/src/handler.ts) ──────────────────────
RE_SAMPLE = re.compile(r"^v2/s/(GSM\d{1,12})/([0-9a-f]{32})\.s2$")
RE_PACK = re.compile(r"^v2/ref/([0-9a-f]{32})\.pack$")
RE_MANIFEST = re.compile(r"^v2/m/(GSE\d{1,12})\.json$")
RE_CELLS = re.compile(r"^v2/cells/(GSE\d{1,12})\.cells$")
RE_LEGACY = re.compile(r"^data/(GSE\d+)/\1\.singlet$")
REFS_INDEX_KEY = "v2/ref/index.json"

KIND_NAMES = {1: "pz2", 2: "cells", 3: "stored", 4: "derived", 5: "raw"}
BAD_STATUSES = frozenset({"failed", "migrate_failed", "verify_failed", "upload_failed", "legacy_changed", "collect_failed"})
OK_STATUSES = frozenset({"ok", "skipped_done", "dry_ok", "dry_listed"})


def legacy_key(gse: str) -> str:
    return f"data/{gse}/{gse}.singlet"


def manifest_key(gse: str) -> str:
    return f"v2/m/{gse}.json"


def ledger_key(gse: str) -> str:
    return f"v2/ledger/{gse}.jsonl"


def accession_order(gses: Iterable[str]) -> List[str]:
    """GSE ids in accession order (length first, then text): stable as bundles shrink or grow."""
    return sorted(set(gses), key=lambda g: (len(g), g))


def iso(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def write_lf(path: Path, text: str) -> None:
    """UTF-8 with LF line ends on every OS (``Path.write_text`` has no ``newline`` before Python 3.10)."""
    path.write_bytes(text.encode("utf-8"))


def object_headers(key: str) -> Tuple[str, str]:
    """(Content-Type, Cache-Control) for a v2 key."""
    if key == REFS_INDEX_KEY or RE_MANIFEST.match(key):
        return "application/json", "public, max-age=300"
    if RE_SAMPLE.match(key) or RE_PACK.match(key):
        return "application/octet-stream", "public, max-age=31536000, immutable"
    if RE_CELLS.match(key):
        return "application/octet-stream", "public, max-age=300"
    if key.startswith("v2/ledger/"):
        return "application/x-ndjson", "no-store"
    return "application/octet-stream", "no-store"


# ── hashing and ETags ─────────────────────────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class FileDigest:
    size: int
    md5: str
    sha256: str
    etag: str  # what R2/S3 report for this file uploaded with `part_size` parts
    n_parts: int


def composite_etag(part_md5s: Sequence[bytes]) -> str:
    return hashlib.md5(b"".join(part_md5s)).hexdigest() + f"-{len(part_md5s)}"


def hash_file(path: Path, part_size: int = PART_SIZE) -> FileDigest:
    """One pass: whole-file MD5 and SHA-256, and the ETag an upload with fixed `part_size` parts produces.

    A file of at most one part is a single PUT (ETag = MD5); a larger file is multipart (ETag = MD5 of the concatenated part
    MD5s, "-N"). The uploaders in this module use exactly that rule.
    """
    md5_all, sha = hashlib.md5(), hashlib.sha256()
    part_md5s: List[bytes] = []
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(part_size)
            if not block:
                break
            size += len(block)
            md5_all.update(block)
            sha.update(block)
            part_md5s.append(hashlib.md5(block).digest())
    etag = md5_all.hexdigest() if size <= part_size else composite_etag(part_md5s)
    return FileDigest(size, md5_all.hexdigest(), sha.hexdigest(), etag, max(1, len(part_md5s)))


def digest_bytes(data: bytes, part_size: int = PART_SIZE) -> FileDigest:
    md5_all = hashlib.md5(data)
    parts = [hashlib.md5(data[i : i + part_size]).digest() for i in range(0, len(data), part_size)] or [md5_all.digest()]
    etag = md5_all.hexdigest() if len(data) <= part_size else composite_etag(parts)
    return FileDigest(len(data), md5_all.hexdigest(), hashlib.sha256(data).hexdigest(), etag, len(parts))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * MIB), b""):
            h.update(block)
    return h.hexdigest()


# ── stores ────────────────────────────────────────────────────────────────────────────────────────────────────────────


class LegacyChanged(RuntimeError):
    """The legacy object was replaced while it was being read."""


class WriteRefused(RuntimeError):
    """A write outside the prefixes this process is allowed to touch."""


@dataclasses.dataclass
class Head:
    size: int
    etag: str
    meta: Dict[str, str] = dataclasses.field(default_factory=dict)
    content_type: Optional[str] = None
    cache_control: Optional[str] = None


def _err_code(e: BaseException) -> str:
    resp = getattr(e, "response", None) or {}
    return str((resp.get("Error") or {}).get("Code", ""))


def _err_status(e: BaseException) -> int:
    resp = getattr(e, "response", None) or {}
    return int((resp.get("ResponseMetadata") or {}).get("HTTPStatusCode", 0) or 0)


def _not_found(e: BaseException) -> bool:
    return _err_code(e) in ("404", "NoSuchKey", "NotFound", "NoSuchBucket") or _err_status(e) == 404


def _precondition_failed(e: BaseException) -> bool:
    return _err_code(e) in ("412", "PreconditionFailed") or _err_status(e) == 412


class StoreBase:
    """Operations shared by the stores. Subclasses provide head/read_range/get_bytes/download/put_*/list_keys/..."""

    def list_studies(self, bucket: str) -> List[Tuple[str, int]]:
        out = []
        for key, size in self.list_keys(bucket, "data/"):
            m = RE_LEGACY.match(key)
            if m:
                out.append((m.group(1), int(size)))
        return sorted(out, key=lambda t: (len(t[0]), t[0]))

    def iter_chunks(self, bucket: str, key: str, size: int, chunk: int = 64 * MIB) -> Iterator[bytes]:
        for a in range(0, size, chunk):
            yield self.read_range(bucket, key, a, min(a + chunk, size) - 1)

    def sha256_object(self, bucket: str, key: str, size: int) -> str:
        h = hashlib.sha256()
        for block in self.iter_chunks(bucket, key, size):
            h.update(block)
        return h.hexdigest()


class S3Store(StoreBase):
    """Buckets of one R2 account over the S3 API. `client` and `transfer_config` are injectable for tests."""

    def __init__(self, client=None, transfer_config=None, copy_single_max: int = 4 * GIB, copy_part: int = GIB) -> None:
        self.c = client if client is not None else make_s3_client()
        self._tc = transfer_config
        self.copy_single_max = copy_single_max
        self.copy_part = copy_part

    # -- reads
    def head(self, bucket: str, key: str) -> Optional[Head]:
        try:
            r = self.c.head_object(Bucket=bucket, Key=key)
        except Exception as e:  # noqa: BLE001
            if _not_found(e):
                return None
            raise
        return Head(
            int(r["ContentLength"]),
            str(r["ETag"]).strip('"'),
            dict(r.get("Metadata") or {}),
            r.get("ContentType"),
            r.get("CacheControl"),
        )

    def read_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        return self.c.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()

    def get_bytes(self, bucket: str, key: str) -> bytes:
        return self.c.get_object(Bucket=bucket, Key=key)["Body"].read()

    def download(self, bucket: str, key: str, path: Path, *, if_match: Optional[str] = None, size: Optional[int] = None,
                 part: int = 64 * MIB, threads: int = 8) -> None:
        """Ranged GETs, in parallel, each pinned to `if_match` (the ETag seen by HEAD): a replaced object fails the read."""
        if size is None:
            h = self.head(bucket, key)
            if h is None:
                raise FileNotFoundError(f"{bucket}/{key}")
            size = h.size
        with open(path, "wb") as fh:
            fh.truncate(size)
        spans = [(a, min(a + part, size) - 1) for a in range(0, size, part)]

        def fetch(span: Tuple[int, int]) -> None:
            a, b = span
            last: Optional[BaseException] = None
            for attempt in range(4):
                try:
                    kw: Dict[str, Any] = dict(Bucket=bucket, Key=key, Range=f"bytes={a}-{b}")
                    if if_match:
                        kw["IfMatch"] = f'"{if_match}"'
                    body = self.c.get_object(**kw)["Body"]
                    got = 0
                    with open(path, "r+b") as fh:
                        fh.seek(a)
                        for blk in iter(lambda: body.read(MIB), b""):
                            fh.write(blk)
                            got += len(blk)
                    if got != b - a + 1:
                        raise IOError(f"short read {got} of {b - a + 1} at {a}")
                    return
                except Exception as e:  # noqa: BLE001
                    if _precondition_failed(e):
                        raise LegacyChanged(f"{key} changed while it was being read") from e
                    last = e
                    time.sleep(min(30, 2 ** attempt))
            raise IOError(f"{key} bytes {a}-{b}: {last}")

        if not spans:
            return
        with cf.ThreadPoolExecutor(max(1, min(threads, len(spans)))) as ex:
            for f in [ex.submit(fetch, s) for s in spans]:
                f.result()

    def list_keys(self, bucket: str, prefix: str) -> Iterator[Tuple[str, int]]:
        for page in self.c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []) or []:
                yield o["Key"], int(o["Size"])

    # -- writes
    def _transfer_config(self, part_size: int):
        if self._tc is not None:
            return self._tc(part_size) if callable(self._tc) else self._tc
        from boto3.s3.transfer import TransferConfig

        # size > part_size -> multipart with exactly part_size parts; size <= part_size -> one PUT
        return TransferConfig(multipart_threshold=part_size + 1, multipart_chunksize=part_size, max_concurrency=4)

    def put_file(self, bucket: str, key: str, path: Path, *, content_type: str, cache_control: str,
                 metadata: Optional[Dict[str, str]] = None, part_size: int = PART_SIZE) -> str:
        extra: Dict[str, Any] = {"ContentType": content_type, "CacheControl": cache_control}
        if metadata:
            extra["Metadata"] = dict(metadata)
        self.c.upload_file(str(path), bucket, key, ExtraArgs=extra, Config=self._transfer_config(part_size))
        h = self.head(bucket, key)
        if h is None:
            raise IOError(f"{bucket}/{key} not found after upload")
        return h.etag

    def put_bytes(self, bucket: str, key: str, data: bytes, *, content_type: str, cache_control: str,
                  metadata: Optional[Dict[str, str]] = None) -> str:
        kw: Dict[str, Any] = dict(Bucket=bucket, Key=key, Body=data, ContentType=content_type, CacheControl=cache_control)
        if metadata:
            kw["Metadata"] = dict(metadata)
        return str(self.c.put_object(**kw)["ETag"]).strip('"')

    def delete(self, bucket: str, key: str) -> None:
        self.c.delete_object(Bucket=bucket, Key=key)

    def bucket_exists(self, bucket: str) -> bool:
        try:
            self.c.head_bucket(Bucket=bucket)
            return True
        except Exception as e:  # noqa: BLE001
            if _not_found(e) or _err_code(e) in ("NoSuchBucket",):
                return False
            raise

    def create_bucket(self, bucket: str) -> None:
        self.c.create_bucket(Bucket=bucket)

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, *, size: int,
             storage_class: Optional[str] = None) -> Tuple[str, bool]:
        """Server-side copy. Returns (ETag of the new object, multipart?). Over `copy_single_max` it copies ranges."""
        sc = {"StorageClass": storage_class} if storage_class else {}
        src = {"Bucket": src_bucket, "Key": src_key}
        for attempt in (0, 1):
            try:
                if size <= self.copy_single_max:
                    r = self.c.copy_object(Bucket=dst_bucket, Key=dst_key, CopySource=src, MetadataDirective="COPY", **sc)
                    return str((r.get("CopyObjectResult") or {}).get("ETag", "")).strip('"'), False
                return self._copy_multipart(src, dst_bucket, dst_key, size, sc), True
            except Exception as e:  # noqa: BLE001
                if sc and attempt == 0 and _err_code(e) in ("InvalidStorageClass", "NotImplemented", "InvalidArgument", "InvalidRequest"):
                    sc = {}  # the API does not take a storage class here: the bucket default applies
                    continue
                raise
        raise AssertionError("unreachable")

    def _copy_multipart(self, src: Dict[str, str], dst_bucket: str, dst_key: str, size: int, sc: Dict[str, str]) -> str:
        head = self.head(src["Bucket"], src["Key"])
        extra: Dict[str, Any] = dict(sc)
        if head is not None and head.content_type:
            extra["ContentType"] = head.content_type
        if head is not None and head.cache_control:
            extra["CacheControl"] = head.cache_control
        if head is not None and head.meta:
            extra["Metadata"] = head.meta
        up = self.c.create_multipart_upload(Bucket=dst_bucket, Key=dst_key, **extra)
        uid = up["UploadId"]
        spans = [(i + 1, a, min(a + self.copy_part, size) - 1) for i, a in enumerate(range(0, size, self.copy_part))]

        def one(span: Tuple[int, int, int]) -> Dict[str, Any]:
            n, a, b = span
            for attempt in range(4):
                try:
                    r = self.c.upload_part_copy(Bucket=dst_bucket, Key=dst_key, UploadId=uid, PartNumber=n,
                                                CopySource=src, CopySourceRange=f"bytes={a}-{b}")
                    return {"PartNumber": n, "ETag": r["CopyPartResult"]["ETag"]}
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            raise AssertionError("unreachable")

        try:
            with cf.ThreadPoolExecutor(4) as ex:
                parts = list(ex.map(one, spans))
            r = self.c.complete_multipart_upload(Bucket=dst_bucket, Key=dst_key, UploadId=uid, MultipartUpload={"Parts": parts})
        except Exception:
            try:
                self.c.abort_multipart_upload(Bucket=dst_bucket, Key=dst_key, UploadId=uid)
            finally:
                pass
            raise
        return str(r.get("ETag", "")).strip('"')


def make_s3_client():
    """boto3 client for the account's R2 endpoint (credentials from the environment, never printed)."""
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("R2_ENDPOINT") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    kw: Dict[str, Any] = dict(
        signature_version="s3v4",
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=128,
        connect_timeout=30,
        read_timeout=120,
        s3={"addressing_style": "path"},
    )
    for opt in ("request_checksum_calculation", "response_checksum_validation"):
        try:  # botocore >= 1.36 adds CRC32 trailers by default; R2 wants them only when the operation requires them
            Config(**{opt: "when_required"})
            kw[opt] = "when_required"
        except TypeError:
            pass
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(**kw),
    )


class LocalStore(StoreBase):
    """Buckets as directories under `root` (tests, rehearsals). Mirrors the S3 semantics that matter here:
    ETags (MD5, or composite for multipart), metadata, multipart copies. `ops` records every call in order."""

    def __init__(self, root: Path, copy_single_max: int = 4 * GIB, copy_part: int = GIB) -> None:
        self.root = Path(root)
        self.copy_single_max = copy_single_max
        self.copy_part = copy_part
        self.ops: List[Tuple[str, str, str]] = []
        self.fail: Dict[Tuple[str, str], int] = {}  # (op, key) -> number of calls still to fail (fault injection)
        self._lock = threading.Lock()
        # Final replaces and reads take this lock: on Windows a file cannot be replaced while another thread has it open.
        self._io = threading.RLock()

    def _p(self, bucket: str, key: str) -> Path:
        return self.root / bucket / key

    def _mp(self, bucket: str, key: str) -> Path:
        return self.root / ".meta" / bucket / (key + ".json")

    def _log(self, op: str, bucket: str, key: str) -> None:
        with self._lock:
            self.ops.append((op, bucket, key))
            n = self.fail.get((op, key), 0)
            if n:
                self.fail[(op, key)] = n - 1
                raise OSError(f"injected failure: {op} {key}")

    @staticmethod
    def _tmp(dst: Path) -> Path:
        # unique per writer: workers may store the same key (a shared pack) at the same time
        return dst.with_name(f"{dst.name}.{os.getpid()}.{threading.get_ident()}.part")

    def _write_meta(self, bucket: str, key: str, meta: Dict[str, Any]) -> None:
        mp = self._mp(bucket, key)
        mp.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp(mp)
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, mp)  # a concurrent head() never reads a half-written file

    def head(self, bucket: str, key: str) -> Optional[Head]:
        p = self._p(bucket, key)
        with self._io:
            if not p.is_file():
                return None
            mp = self._mp(bucket, key)
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            etag = meta.get("etag") or hashlib.md5(p.read_bytes()).hexdigest()
            return Head(p.stat().st_size, etag, dict(meta.get("metadata") or {}), meta.get("content_type"), meta.get("cache_control"))

    def read_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        self._log("read_range", bucket, key)
        with self._io, open(self._p(bucket, key), "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)

    def get_bytes(self, bucket: str, key: str) -> bytes:
        self._log("get", bucket, key)
        with self._io:
            return self._p(bucket, key).read_bytes()

    def download(self, bucket: str, key: str, path: Path, *, if_match: Optional[str] = None, size: Optional[int] = None,
                 **_: Any) -> None:
        self._log("download", bucket, key)
        h = self.head(bucket, key)
        if h is None:
            raise FileNotFoundError(f"{bucket}/{key}")
        if if_match and h.etag != if_match:
            raise LegacyChanged(f"{key} changed while it was being read")
        with self._io:
            shutil.copyfile(self._p(bucket, key), path)

    def list_keys(self, bucket: str, prefix: str) -> Iterator[Tuple[str, int]]:
        base = self.root / bucket
        if not base.is_dir():
            return
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.endswith(".part"):  # a writer's temporary file is not an object
                key = p.relative_to(base).as_posix()
                if key.startswith(prefix):
                    try:
                        yield key, p.stat().st_size
                    except FileNotFoundError:  # replaced or removed by a concurrent writer
                        continue

    def _store(self, bucket: str, key: str, data_path: Optional[Path], data: Optional[bytes], etag: str, content_type: str,
               cache_control: str, metadata: Optional[Dict[str, str]]) -> None:
        dst = self._p(bucket, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp(dst)
        if data_path is not None:
            shutil.copyfile(data_path, tmp)
        else:
            tmp.write_bytes(data or b"")
        with self._io:
            os.replace(tmp, dst)
            self._write_meta(bucket, key, {"etag": etag, "content_type": content_type, "cache_control": cache_control,
                                           "metadata": dict(metadata or {})})

    def put_file(self, bucket: str, key: str, path: Path, *, content_type: str, cache_control: str,
                 metadata: Optional[Dict[str, str]] = None, part_size: int = PART_SIZE) -> str:
        self._log("put", bucket, key)
        etag = hash_file(path, part_size).etag
        self._store(bucket, key, path, None, etag, content_type, cache_control, metadata)
        return etag

    def put_bytes(self, bucket: str, key: str, data: bytes, *, content_type: str, cache_control: str,
                  metadata: Optional[Dict[str, str]] = None) -> str:
        self._log("put", bucket, key)
        etag = hashlib.md5(data).hexdigest()
        self._store(bucket, key, None, data, etag, content_type, cache_control, metadata)
        return etag

    def delete(self, bucket: str, key: str) -> None:
        self._log("delete", bucket, key)
        with self._io:
            for p in (self._p(bucket, key), self._mp(bucket, key)):
                if p.exists():
                    p.unlink()

    def bucket_exists(self, bucket: str) -> bool:
        return (self.root / bucket).is_dir()

    def create_bucket(self, bucket: str) -> None:
        self._log("create_bucket", bucket, "")
        (self.root / bucket).mkdir(parents=True, exist_ok=True)

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, *, size: int,
             storage_class: Optional[str] = None) -> Tuple[str, bool]:
        self._log("copy", dst_bucket, dst_key)
        h = self.head(src_bucket, src_key)
        if h is None:
            raise FileNotFoundError(f"{src_bucket}/{src_key}")
        with self._io:
            data = self._p(src_bucket, src_key).read_bytes()
        multipart = len(data) > self.copy_single_max
        if multipart:  # like S3: the new object's ETag is the composite over the copied parts
            etag = composite_etag([hashlib.md5(data[i : i + self.copy_part]).digest() for i in range(0, len(data), self.copy_part)])
        else:
            etag = h.etag
        meta = {"etag": etag, "content_type": h.content_type, "cache_control": h.cache_control, "metadata": h.meta,
                "storage_class": storage_class}
        dst = self._p(dst_bucket, dst_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp(dst)
        tmp.write_bytes(data)
        with self._io:
            os.replace(tmp, dst)
            self._write_meta(dst_bucket, dst_key, meta)
        return etag, multipart


class GuardedStore:
    """Wraps a store so that writes, copies and deletes can only touch `allow[bucket]` key prefixes (default: nothing).

    The campaign hands this to the study code with `{public: ("v2/",)}`: legacy objects cannot be modified from there.
    Reads pass straight through."""

    def __init__(self, inner, allow: Dict[str, Tuple[str, ...]], allow_delete: Dict[str, Tuple[str, ...]] | None = None) -> None:
        self._inner = inner
        self._allow = allow
        self._allow_delete = allow_delete or {}

    def _check(self, bucket: str, key: str, table: Dict[str, Tuple[str, ...]], what: str) -> None:
        if not any(key.startswith(p) for p in table.get(bucket, ())):
            raise WriteRefused(f"{what} refused: {bucket}/{key}")

    def put_file(self, bucket, key, *a, **kw):
        self._check(bucket, key, self._allow, "write")
        return self._inner.put_file(bucket, key, *a, **kw)

    def put_bytes(self, bucket, key, *a, **kw):
        self._check(bucket, key, self._allow, "write")
        return self._inner.put_bytes(bucket, key, *a, **kw)

    def copy(self, src_bucket, src_key, dst_bucket, dst_key, **kw):
        self._check(dst_bucket, dst_key, self._allow, "copy")
        return self._inner.copy(src_bucket, src_key, dst_bucket, dst_key, **kw)

    def delete(self, bucket, key):
        self._check(bucket, key, self._allow_delete, "delete")
        return self._inner.delete(bucket, key)

    def create_bucket(self, bucket):
        raise WriteRefused(f"create_bucket refused: {bucket}")

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ── ledger ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def read_ledger(store, bucket: str, gse: str) -> List[dict]:
    if store.head(bucket, ledger_key(gse)) is None:
        return []
    out = []
    for ln in store.get_bytes(bucket, ledger_key(gse)).decode("utf-8", "replace").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def write_ledger(wstore, bucket: str, gse: str, records: Sequence[dict]) -> None:
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records).encode("utf-8")
    ct, cc = object_headers(ledger_key(gse))
    wstore.put_bytes(bucket, ledger_key(gse), body, content_type=ct, cache_control=cc)


def last_committed(records: Sequence[dict]) -> Optional[dict]:
    for r in reversed(records):
        if r.get("event") == "committed" and isinstance(r.get("summary"), dict):
            return r["summary"]
    return None


# ── subprocesses ──────────────────────────────────────────────────────────────────────────────────────────────────────


def run_logged(argv: Sequence[str], *, cwd: Path, env: Dict[str, str], timeout: float, log_path: Path) -> Tuple[int, str, float]:
    """Run a command with stdout+stderr in `log_path`; returns (exit code, last 3000 chars of the log, seconds). A command
    that exceeds `timeout` is killed (its whole process group on POSIX) and reported with exit code 124."""
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kw: Dict[str, Any] = {}
    if os.name == "posix":
        kw["start_new_session"] = True
    with open(log_path, "ab") as lf:
        lf.write(f"$ {' '.join(argv)}\n".encode())
        lf.flush()
        proc = subprocess.Popen(list(argv), cwd=str(cwd), env=env, stdout=lf, stderr=subprocess.STDOUT, **kw)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait()
            rc = 124
            lf.write(f"\n[v2_campaign] killed after {timeout:.0f}s\n".encode())
    tail = log_path.read_bytes()[-3000:].decode("utf-8", "replace")
    return rc, tail, time.time() - t0


class ByteBudget:
    """Limits the scratch bytes reserved by studies in flight (a study alone larger than the budget runs when idle)."""

    def __init__(self, total: int) -> None:
        self.total, self.used, self.cond = total, 0, threading.Condition()

    def acquire(self, n: int) -> None:
        with self.cond:
            while self.used and self.used + n > self.total:
                self.cond.wait(timeout=5)
            self.used += n

    def release(self, n: int) -> None:
        with self.cond:
            self.used -= n
            self.cond.notify_all()


# ── object plan: what a migration produced ────────────────────────────────────────────────────────────────────────────


class StudyFailure(Exception):
    def __init__(self, status: str, stage: str, msg: str) -> None:
        super().__init__(msg)
        self.status, self.stage = status, stage


@dataclasses.dataclass
class Obj:
    key: str
    path: Path
    digest: FileDigest


@dataclasses.dataclass
class Plan:
    root: Path
    manifest: dict
    objects: List[Obj]  # sample objects, packs and cells, never the manifest
    pack_keys: List[str]
    warnings: List[str]


def find_v2_root(out: Path, gse: str) -> Path:
    """The directory whose layout is the R2 key space after `v2/`: `OUT/v2` if the transcoder wrote the full key path, else OUT."""
    for cand in (out / "v2", out):
        if (cand / "m" / f"{gse}.json").is_file():
            return cand
    raise StudyFailure("collect_failed", "collect", f"no m/{gse}.json under {out} (the transcoder wrote no manifest)")


def validate_sample_header(path: Path, gsm: str, size: int) -> None:
    with open(path, "rb") as fh:
        head = fh.read(64)
    if len(head) < 64 or head[:8] != b"SINGLET2":
        raise StudyFailure("collect_failed", "collect", f"{path.name}: not a SINGLET2 object")
    sid = head[40:56].rstrip(b"\0").decode("ascii", "replace")
    file_len = int.from_bytes(head[32:40], "little")
    if sid != gsm:
        raise StudyFailure("collect_failed", "collect", f"{path.name}: header sample id {sid!r} is not {gsm}")
    if file_len != size:
        raise StudyFailure("collect_failed", "collect", f"{path.name}: header length {file_len} != file size {size}")


def collect_objects(root: Path, gse: str, part_size: int, pool: cf.Executor | None = None) -> Plan:
    manifest_path = root / "m" / f"{gse}.json"
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except ValueError as e:
        raise StudyFailure("collect_failed", "collect", f"manifest is not JSON: {e}")
    if manifest.get("format") != "singlet-v2":
        raise StudyFailure("collect_failed", "collect", f"manifest format {manifest.get('format')!r}")
    if manifest.get("gse") != gse:
        raise StudyFailure("collect_failed", "collect", f"manifest is for {manifest.get('gse')!r}, not {gse}")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise StudyFailure("collect_failed", "collect", "manifest lists no samples")

    candidates: List[Tuple[str, Path]] = []
    warnings: List[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        key = "v2/" + p.relative_to(root).as_posix()
        m = RE_MANIFEST.match(key)
        if m:
            if m.group(1) != gse:
                warnings.append(f"ignored manifest of another study: {key}")
            continue
        if key == REFS_INDEX_KEY:
            continue  # the pack index of the whole bucket: `refs-index` publishes it, a single writer
        if key.startswith("v2/ledger/"):
            continue  # the transcoder's per-member ledger: its lines go into the R2 ledger (see publish_bundle)
        if RE_SAMPLE.match(key) or RE_PACK.match(key) or RE_CELLS.match(key):
            candidates.append((key, p))
        else:
            warnings.append(f"ignored file outside the key grammar: {key}")

    def one(item: Tuple[str, Path]) -> Obj:
        key, p = item
        d = hash_file(p, part_size)
        m = RE_SAMPLE.match(key)
        if m:
            if d.sha256[:32] != m.group(2):
                raise StudyFailure("collect_failed", "collect", f"{key}: name is not the digest16 of the object ({d.sha256[:32]})")
            validate_sample_header(p, m.group(1), d.size)
        m = RE_PACK.match(key)
        if m and d.sha256[:32] != m.group(1):
            raise StudyFailure("collect_failed", "collect", f"{key}: name is not the digest16 of the pack ({d.sha256[:32]})")
        return Obj(key, p, d)

    objs = list(pool.map(one, candidates)) if pool is not None else [one(c) for c in candidates]
    by_key = {o.key: o for o in objs}

    referenced = set()
    for s in samples:
        k = s.get("key")
        o = by_key.get(k)
        if o is None:
            raise StudyFailure("collect_failed", "collect", f"manifest sample {s.get('gsm')} points at {k}, which was not produced")
        if s.get("bytes") != o.digest.size:
            raise StudyFailure("collect_failed", "collect", f"{k}: manifest says {s.get('bytes')} bytes, file has {o.digest.size}")
        if s.get("sha256") and s["sha256"] != o.digest.sha256:
            raise StudyFailure("collect_failed", "collect", f"{k}: manifest sha256 differs from the file")
        referenced.add(k)

    pack_keys = []
    for v in (manifest.get("packs") or {}).values():  # an id, a list of ids (several whitelists) or null
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, str) and re.fullmatch(r"[0-9a-f]{32}", x):
                pack_keys.append(f"v2/ref/{x}.pack")
    rm = _restricted_module().is_restricted_member
    for s in samples:  # the v2 object of a public study must not carry per-donor germline members (the endpoint refuses them too)
        bad = [n for n in (s.get("members") or {}) if rm(n)]
        if bad:
            raise StudyFailure("collect_failed", "collect", f"{s.get('gsm')}: restricted members in the v2 object: {', '.join(bad[:3])}")
    unref = [o.key for o in objs if RE_SAMPLE.match(o.key) and o.key not in referenced]
    if unref:
        warnings.append(f"{len(unref)} sample objects are not in the manifest and will not be uploaded")
    keep = [o for o in objs if o.key in referenced or not RE_SAMPLE.match(o.key)]
    return Plan(root, manifest, keep, pack_keys, warnings)


def summarize_manifest(manifest: dict) -> dict:
    """Counts the catalog and the result rows need, from a v2 manifest."""
    kinds: Dict[str, int] = {}
    n_members = 0
    n_called = n_cols = 0
    sample_bytes = 0
    for s in manifest.get("samples", []):
        sample_bytes += int(s.get("bytes") or 0)
        n_called += int(s.get("n_called") or 0)
        n_cols += int(s.get("n_columns") or 0)
        for _name, mem in (s.get("members") or {}).items():
            k = mem.get("kind") if isinstance(mem, dict) else None
            k = KIND_NAMES.get(k, str(k)) if isinstance(k, int) else str(k)
            kinds[k] = kinds.get(k, 0) + 1
            n_members += 1
    return {"n_samples": len(manifest.get("samples", [])), "n_called": n_called, "n_columns": n_cols,
            "sample_bytes": sample_bytes, "members_by_kind": kinds, "n_members": n_members}


def finalize_manifest(manifest: dict, gse: str, legacy: Optional[dict], raw: Optional[bytes] = None) -> bytes:
    """The bytes of the manifest as published. With a legacy bundle, `legacy` gets the facts this campaign measured
    (key, ETag, size, sha256, time, ledger); without one (a new study) the transcoder's bytes are kept as they are."""
    if legacy is None:
        if manifest.get("legacy") is None and raw is not None:
            return raw
        # a new study: the transcoder describes the temporary bundle it was given as a "legacy" object; there is none
        return dumps_manifest(dict(manifest, legacy=None))
    m = dict(manifest)  # key order is kept: the legacy block is updated in place
    lg = dict(m.get("legacy") or {})
    lg.update({"key": legacy["key"], "etag": legacy["etag"], "bytes": legacy["bytes"], "sha256": legacy["sha256"],
               "ledger": ledger_key(gse)})
    lg.setdefault("migrated_at", legacy["migrated_at"])
    m["legacy"] = lg
    return dumps_manifest(m)


def dumps_manifest(m: dict) -> bytes:
    """The serialisation of singlet.v2.manifest.dumps_manifest (indent 1, key order kept, UTF-8, trailing newline)."""
    return (json.dumps(m, indent=1, sort_keys=False, ensure_ascii=False) + "\n").encode("utf-8")


# ── catalog rows (D1 schema/032_v2_catalog.sql: v2_study, v2_sample, v2_migration) ───────────────────────────────────────
#
# The rows are what the website's ingest endpoint (functions/_shared/v2-catalog.ts, parseSample / ingestStatements) would write
# from the same manifest, column for column. The SQL path (`catalog-sql`, applied with `wrangler d1 execute` by the orchestrator)
# is the primary route; `catalog-post` speaks the endpoint's manifest-based contract for when an ingest token exists.

# Mirrors MODALITIES in functions/_shared/modalities.ts and python/singlet/bundle.py (a test compares all three). Order matters:
# `modalities` is stored as the JSON array of registry names in this order, as the endpoint's resolveModalities does.
MODALITY_MEMBERS: List[Tuple[str, Tuple[str, ...]]] = [
    ("exon_counts", ("exon_counts.1pz",)),
    ("intron_counts", ("intron_counts.1pz",)),
    ("cell_calls", ("cell_calls.tsv",)),
    ("cell_qc", ("cell_qc_metrics.tsv",)),
    ("junctions", ("sj_counts.1pz",)),
    ("splice_psi", ("splice_psi.1pz",)),
    ("splice_events", ("splice_events.tsv",)),
    ("mt_heteroplasmy", ("mt_heteroplasmy.1pz",)),
    ("mt_variants", ("mt_variants.tsv", "mt/mt_summary.tsv")),
    ("donor_assignments", ("donor/donor_assignments.tsv", "donor_assignments.tsv")),
    ("ancestry_call", ("ancestry_call.json",)),
    ("sex_call", ("sex_call.json",)),
    ("nonhost_species", ("nonhost/nonhost_em_abundance.tsv", "nonhost_em_abundance.tsv")),
    ("nonhost_summary", ("nonhost/nonhost_summary.json", "nonhost_summary.json")),
    ("vdj_gene_usage", ("vdj_gene_usage.1pz",)),
    ("doublet_scores", ("doublet_scores.tsv",)),
    ("cell_cycle_scores", ("cell_cycle_scores.tsv",)),
    ("ambient_contamination", ("ambient_contamination.tsv",)),
    ("ambient_profile", ("ambient_profile.tsv",)),
    ("summary", ("summary.json",)),
    ("pileup_stats", ("pileup_stats.json",)),
    ("provenance", ("provenance.json",)),
    ("saturation_curve", ("saturation_curve.tsv",)),
    ("star_log", ("star_Log.final.out",)),
]

# Columns in schema order (the endpoint's STUDY_COLUMNS / MIGRATION_COLUMNS and the v2_sample INSERT). `updated_at` is left to the
# table default on insert and set by SQL on update.
STUDY_COLS = ("gse_id", "manifest_key", "rev", "bytes", "n_samples", "created_at", "legacy_bytes", "manifest_bytes", "manifest_sha256",
              "n_cells_called", "reference_build", "packs_json", "pipeline_json")
SAMPLE_COLS = ("gse_id", "gsm_id", "key", "bytes", "sha256", "n_columns", "n_called", "n_members", "median_umi", "median_genes",
               "modalities", "members_json", "cell_stats_json", "rev")
MIGRATION_COLS = ("gse_id", "status", "legacy_key", "legacy_etag", "legacy_bytes", "v2_bytes", "n_samples", "members_by_kind",
                  "verify_ok", "commit_sha", "migrated_at", "verified_at", "ledger_key", "error")
MIGRATION_STATUSES = ("pending", "running", "uploaded", "verified", "failed", "retired")

# What the endpoint refuses, so these rows are never produced either.
MAX_MEMBERS_JSON = 128 * 1024
MAX_STATS_JSON = 16 * 1024
MAX_SMALL_JSON = 8 * 1024
MAX_POST_SAMPLES = 400
MAX_POST_BODY = 8 * MIB
# D1 refuses SQL statements over 100 KB; the values are literals in a file, so a sample row must fit with its text.
MAX_STATEMENT = 95_000

# Row writes D1 bills, indexes included (v2_sample: row + primary key + idx_v2_sample_gsm = 3; the website's notes count the study row
# as 1 and the migration row as 1 or 2; 2 each here, to be on the safe side). A study costs 3 n + 2 + 2.
WRITES_PER_ROW = {"v2_sample": 3, "v2_study": 2, "v2_migration": 2}
DAY_BUDGET = 80_000


class CatalogError(ValueError):
    """The study cannot be described in the catalog as the endpoint would accept it."""


def resolve_modalities(members: Iterable[str]) -> List[str]:
    have = set(members)
    return [name for name, cands in MODALITY_MEMBERS if any(c in have for c in cands)]


def jcompact(obj: Any) -> str:
    """JSON.stringify: no spaces, non-ASCII left as it is."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") else None


def _count(v: Any, what: str) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise CatalogError(f"{what} must be a non-negative integer, got {v!r}")
    return v


def _small_json(v: Any, what: str, limit: int) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, dict):
        raise CatalogError(f"{what} must be an object")
    s = jcompact(v)
    if len(s) > limit:
        raise CatalogError(f"{what} is over {limit} bytes")
    return s


# Fields that can be dropped from a member entry, in order, when a sample row would not fit one SQL statement. Offsets, lengths,
# digests, the chunk index and the codec facts clients use to plan ranged reads are never dropped.
_SHRINK_ORDER = ("names", "columns", "called_source", "params", "needs")


def shrink_members(members: dict, limit: int) -> Tuple[dict, List[str]]:
    dropped: List[str] = []
    cur = {k: dict(v) if isinstance(v, dict) else v for k, v in members.items()}
    for field in _SHRINK_ORDER:
        if len(jcompact(cur)) <= limit:
            break
        hit = False
        for v in cur.values():
            if isinstance(v, dict) and field in v:
                del v[field]
                hit = True
        if hit:
            dropped.append(field)
    return cur, dropped


def sample_row(gse: str, rev: str, s: dict) -> Tuple[dict, List[str]]:
    gsm = s.get("gsm")
    key = s.get("key")
    m = RE_SAMPLE.match(key or "")
    if not isinstance(gsm, str) or not re.fullmatch(r"GSM\d+", gsm) or not m or m.group(1) != gsm:
        raise CatalogError(f"sample {gsm!r}: key must be v2/s/{gsm}/<32 hex>.s2")
    members = s.get("members")
    if not isinstance(members, dict):
        raise CatalogError(f"{gsm}: members must be an object")
    rm = _restricted_module().is_restricted_member
    bad = [n for n in members if rm(n)]
    if bad:
        raise CatalogError(f"{gsm}: restricted members must not be in a public sample object: {', '.join(bad[:5])}")
    for name, mem in members.items():
        if not isinstance(mem, dict) or not all(isinstance(mem.get(k), int) and not isinstance(mem.get(k), bool) and mem.get(k) >= 0
                                                for k in ("off", "len")) or mem.get("kind") not in (1, 2, 3, 4, 5):
            raise CatalogError(f"{gsm}: member {name!r} needs integer kind (1-5), off and len")
    notes: List[str] = []
    mj = jcompact(members)
    if len(mj) > MAX_MEMBERS_JSON:
        raise CatalogError(f"{gsm}: members is over {MAX_MEMBERS_JSON} bytes")
    stats = s.get("stats")
    sha = s.get("sha256")
    row = {
        "gse_id": gse, "gsm_id": gsm, "key": key, "bytes": _count(s.get("bytes"), f"{gsm}: bytes"),
        "sha256": sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None,
        "n_columns": _count(s.get("n_columns"), f"{gsm}: n_columns"), "n_called": _count(s.get("n_called"), f"{gsm}: n_called"),
        "n_members": len(members),
        "median_umi": _num((stats or {}).get("median_umi")) if isinstance(stats, dict) else None,
        "median_genes": _num((stats or {}).get("median_genes")) if isinstance(stats, dict) else None,
        "modalities": jcompact(resolve_modalities(members.keys())),
        "members_json": mj,
        "cell_stats_json": _small_json(stats, f"{gsm}: stats", MAX_STATS_JSON),
        "rev": rev,
    }
    return row, notes


def build_catalog_rows(manifest: Optional[dict], row: dict, members_mode: str = "full") -> Dict[str, Any]:
    """{v2_migration, v2_study, v2_sample[]} for one study, from its manifest (None for a failed attempt) and its result row.

    `members_mode` is kept for the command line: only "full" is exact; "none" stores no members_json (smaller files, a client then
    reads the manifest)."""
    out: Dict[str, Any] = {}
    gse = row["gse"]
    lg = (manifest or {}).get("legacy") or {}
    has_legacy = bool(row.get("legacy_key") or lg)
    status = migration_status(row, has_legacy)
    mig = {
        "gse_id": gse, "status": status,
        "legacy_key": row.get("legacy_key") or lg.get("key"), "legacy_etag": row.get("legacy_etag") or lg.get("etag"),
        "legacy_bytes": row.get("legacy_bytes") if row.get("legacy_bytes") is not None else lg.get("bytes"),
        "v2_bytes": row.get("v2_bytes"), "n_samples": row.get("n_samples"),
        "members_by_kind": jcompact(treatment_counts(row.get("members_by_kind") or {})) if row.get("members_by_kind") else None,
        "verify_ok": (1 if (row.get("verify") or {}).get("ok") else 0) if row.get("verify") else None,
        "commit_sha": (row.get("commit") or None) and str(row["commit"])[:64],
        "migrated_at": lg.get("migrated_at") or row.get("at"),
        "verified_at": row.get("verified_at") if status == "verified" else None,
        "ledger_key": ledger_key(gse),
        "error": ((row.get("error") or "")[:2000] or None) if status == "failed" else None,
    }
    out["v2_migration"] = mig
    if manifest is None:
        return out
    rev = manifest.get("rev")
    if manifest.get("format") != "singlet-v2" or manifest.get("manifest_version") != 1 or not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{16}", rev):
        raise CatalogError(f"{gse}: not a singlet-v2 manifest_version 1 with a 16-hex rev")
    samples = []
    seen = set()
    for s in manifest.get("samples", []):
        r, _notes = sample_row(gse, rev, s)
        if r["gsm_id"] in seen:
            raise CatalogError(f"{gse}: {r['gsm_id']} appears twice")
        seen.add(r["gsm_id"])
        if members_mode == "none":
            r["members_json"] = None
        samples.append(r)
    if not samples:
        raise CatalogError(f"{gse}: the manifest lists no samples")
    out["v2_sample"] = samples
    out["v2_study"] = {
        "gse_id": gse, "manifest_key": manifest_key(gse), "rev": rev, "bytes": sum(r["bytes"] or 0 for r in samples),
        "n_samples": len(samples), "created_at": manifest.get("created_at") if isinstance(manifest.get("created_at"), str) else None,
        "legacy_bytes": _count(lg.get("bytes"), "manifest.legacy.bytes"), "manifest_bytes": row.get("manifest_bytes"),
        "manifest_sha256": row.get("manifest_sha256"), "n_cells_called": sum(r["n_called"] or 0 for r in samples),
        "reference_build": manifest.get("reference_build") if isinstance(manifest.get("reference_build"), str) else None,
        "packs_json": _small_json(manifest.get("packs"), "manifest.packs", MAX_SMALL_JSON),
        "pipeline_json": _small_json(manifest.get("pipeline"), "manifest.pipeline", MAX_SMALL_JSON),
    }
    return out


_TREATMENT = {"pz2": "matrix", "derived": "recipe", "stored": "stored", "raw": "raw", "cells": "cells"}


def treatment_counts(kinds: Dict[str, int]) -> Dict[str, int]:
    """Members per treatment as the catalog names them: matrix / recipe / stored / raw (the cells table is counted as `cells`)."""
    out: Dict[str, int] = {}
    for k, n in kinds.items():
        out[_TREATMENT.get(k, k)] = out.get(_TREATMENT.get(k, k), 0) + int(n)
    return dict(sorted(out.items()))


def migration_status(row: dict, has_legacy: bool = True) -> str:
    """v2_migration.status for a campaign result row: only a fully verified, committed study is `verified`. A study that never had a
    legacy file (published as v2 directly) has nothing to retire later: it is posted as `retired` at once, so the site never offers
    a legacy URL for it."""
    st = row.get("status")
    if st in ("ok", "skipped_done"):
        return "verified" if has_legacy else "retired"
    return "failed"


# ── SQL ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def sql_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v) if v == v and v not in (float("inf"), float("-inf")) else "NULL"
    return "'" + str(v).replace("\x00", "").replace("'", "''") + "'"


NOW = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"


def _insert(table: str, cols: Sequence[str], rec: dict) -> str:
    return f"INSERT INTO {table} ({', '.join(cols)}, updated_at) VALUES ({', '.join(sql_literal(rec.get(c)) for c in cols)}, {NOW})"


def sql_sample(rec: dict, schema: str = "033") -> str:
    """One v2_sample row. 033 (primary key gse_id, rev, gsm_id; what the ingest endpoint writes now): the rows of a rev sit beside the
    served rev and an existing row is left alone. 032 (primary key gse_id, gsm_id): the row is overwritten."""
    if schema == "032":
        sets = ", ".join(f"{c} = excluded.{c}" for c in SAMPLE_COLS if c not in ("gse_id", "gsm_id"))
        return f"{_insert('v2_sample', SAMPLE_COLS, rec)} ON CONFLICT(gse_id, gsm_id) DO UPDATE SET {sets}, updated_at = {NOW};"
    return f"{_insert('v2_sample', SAMPLE_COLS, rec)} ON CONFLICT(gse_id, rev, gsm_id) DO NOTHING;"


def sql_study(rec: dict) -> List[str]:
    """The flip: the v2_study row (readers list samples under its rev), then the sample rows of every other rev."""
    sets = ", ".join(f"{c} = excluded.{c}" for c in STUDY_COLS if c != "gse_id")
    return [
        f"{_insert('v2_study', STUDY_COLS, rec)} ON CONFLICT(gse_id) DO UPDATE SET {sets}, updated_at = {NOW};",
        f"DELETE FROM v2_sample WHERE gse_id = {sql_literal(rec['gse_id'])} AND rev != {sql_literal(rec['rev'])};",
    ]


def sql_migration(rec: dict) -> str:
    """The migration row, the endpoint's upsert (functions/_shared/v2-catalog.ts): COALESCE keeps what an earlier row knew."""
    coalesce = ("legacy_key", "legacy_etag", "legacy_bytes", "v2_bytes", "n_samples", "members_by_kind", "commit_sha", "migrated_at",
                "verified_at", "ledger_key")
    sets = (["status = excluded.status"] + [f"{c} = COALESCE(excluded.{c}, v2_migration.{c})" for c in coalesce[:6]]
            + ["verify_ok = excluded.verify_ok"] + [f"{c} = COALESCE(excluded.{c}, v2_migration.{c})" for c in coalesce[6:]]
            + ["error = excluded.error"])
    return f"{_insert('v2_migration', MIGRATION_COLS, rec)} ON CONFLICT(gse_id) DO UPDATE SET {', '.join(sets)}, updated_at = {NOW};"


def study_statements(rows: Dict[str, Any], schema: str = "033") -> Dict[str, Any]:
    """The SQL of one study: `sample` (any order, idempotent) and `flip`, the statements that must stay together and come after
    every sample row: the study row, the removal of older revs, the `verified` migration row. A failed study has only the migration
    row (it is never sent for a study that has sample rows from an earlier run: see _finish)."""
    out: Dict[str, Any] = {"sample": [], "flip": []}
    samples = rows.get("v2_sample")
    if samples:
        for r in samples:
            st = sql_sample(r, schema)
            if len(st) > MAX_STATEMENT:
                members, _dropped = shrink_members(json.loads(r["members_json"]), MAX_STATEMENT - (len(st) - len(r["members_json"])) - 2000)
                r = dict(r, members_json=jcompact(members))
                st = sql_sample(r, schema)
                if len(st) > MAX_STATEMENT:
                    raise CatalogError(f"{r['gsm_id']}: the sample row is {len(st)} bytes as SQL even without optional member fields (D1 limit 100 KB)")
            out["sample"].append(st)
        out["flip"].extend(sql_study(rows["v2_study"]))
    out["flip"].append(sql_migration(rows["v2_migration"]))
    return out


def est_writes(rows: Dict[str, Any]) -> int:
    n = len(rows.get("v2_sample") or [])
    if not n:
        return WRITES_PER_ROW["v2_migration"]  # a failed study: one migration upsert
    return n * WRITES_PER_ROW["v2_sample"] + WRITES_PER_ROW["v2_study"] + WRITES_PER_ROW["v2_migration"]


def load_rows(sources: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Catalog rows from rows-dir(s) (`<GSE>.json` files); later sources override earlier ones, per study."""
    studies: Dict[str, dict] = {}
    for src in sources:
        files = sorted(Path(src).glob("GSE*.json")) if Path(src).is_dir() else [Path(src)]
        for f in files:
            d = json.loads(f.read_text("utf-8"))
            gse = (d.get("v2_migration") or {}).get("gse_id") or f.stem
            studies[gse] = d
    return studies


def rows_from_r2(store, bucket: str, gses: Sequence[str], members_mode: str = "full", log=print) -> Dict[str, dict]:
    """Rebuild the catalog rows of studies from their manifests and ledgers in R2 (the truth the rows cache)."""
    studies: Dict[str, dict] = {}
    for g in accession_order(gses):
        if store.head(bucket, manifest_key(g)) is None:
            continue
        raw = store.get_bytes(bucket, manifest_key(g))
        m = json.loads(raw)
        prior = last_committed(read_ledger(store, bucket, g))
        if prior is None:
            log(f"{g}: manifest without a committed ledger line: not catalogued")
            continue
        md = digest_bytes(raw)
        prior = dict(prior, gse=g, manifest_bytes=md.size, manifest_sha256=md.sha256, status="ok")
        try:
            studies[g] = build_catalog_rows(m, prior, members_mode)
        except CatalogError as e:
            log(f"{g}: {e}")
    return studies


def plan_days(studies: Dict[str, dict], day_budget: int = DAY_BUDGET) -> List[List[str]]:
    """Whole studies in accession order into days of at most `day_budget` estimated row writes (a larger study gets a day alone)."""
    days: List[List[str]] = []
    cur: List[str] = []
    used = 0
    for g in accession_order(studies):
        w = est_writes(studies[g])
        if cur and used + w > day_budget:
            days.append(cur)
            cur, used = [], 0
        cur.append(g)
        used += w
    if cur:
        days.append(cur)
    return days


def _pack_units(units: List[List[str]], max_rows: int, max_bytes: int) -> List[List[str]]:
    """Files of whole units (a unit is never split across files), at most max_rows statements and max_bytes each."""
    files: List[List[str]] = []
    cur: List[str] = []
    size = 0
    for u in units:
        usize = sum(len(s) + 1 for s in u)
        if cur and (len(cur) + len(u) > max_rows or size + usize > max_bytes):
            files.append(cur)
            cur, size = [], 0
        cur.extend(u)
        size += usize
    if cur:
        files.append(cur)
    return files


def write_sql_files(studies: Dict[str, dict], out: Path, *, day_budget: int, max_rows: int, max_bytes: int, database: str,
                    schema: str = "033", log=print) -> dict:
    """SQL files under out/dayNN/, with APPLY.txt per day and PLAN.txt overall.

    A day holds whole studies and at most `day_budget` estimated D1 row writes. Within a day: first every sample row of the day's
    studies (`1_sample_*.sql`: nobody reads them until a study row names their rev), then `2_flip_*.sql`, where each study's
    statements stay together in one atomic file: the v2_study row, the removal of the other revs' sample rows, and the v2_migration
    row, so a study turns `verified` only together with, and after, its own sample and study rows (the order the ingest endpoint
    uses). Returns the plan summary."""
    out.mkdir(parents=True, exist_ok=True)
    for d in out.glob("day*"):
        if d.is_dir():
            shutil.rmtree(d)
    skipped: Dict[str, str] = {}
    prepared: Dict[str, Dict[str, Any]] = {}
    for g in accession_order(studies):
        try:
            prepared[g] = study_statements(studies[g], schema)
        except CatalogError as e:
            skipped[g] = str(e)
            log(f"{g}: skipped ({e})")
    kept = {g: studies[g] for g in prepared}
    days = plan_days(kept, day_budget)
    summary = {"days": [], "skipped": skipped, "n_studies": len(kept), "est_writes": 0, "n_files": 0, "n_statements": 0, "schema": schema}
    for di, gses in enumerate(days, 1):
        dd = out / f"day{di:02d}"
        dd.mkdir()
        names: List[str] = []
        n_stmt = 0
        phases = (("1_sample", [[s] for g in gses for s in prepared[g]["sample"]]),
                  ("2_flip", [list(prepared[g]["flip"]) for g in gses]))
        for prefix, units in phases:
            for k, buf in enumerate(_pack_units(units, max_rows, max_bytes)):
                name = f"{prefix}_{k:04d}.sql"
                write_lf(dd / name, "\n".join(buf) + "\n")
                names.append(name)
                n_stmt += len(buf)
        writes = sum(est_writes(kept[g]) for g in gses)
        samples = sum(len(kept[g].get("v2_sample") or []) for g in gses)
        write_lf(
            dd / "APPLY.txt",
            f"# day {di}: {len(gses)} studies ({gses[0]} .. {gses[-1]}), {samples} sample rows, about {writes} D1 row writes. "
            f"Apply the files in this order; each file is atomic.\n"
            + "\n".join(f"wrangler d1 execute {database} --remote --file={dd.name}/{n}" for n in names) + "\n")
        write_lf(dd / "STUDIES.txt", "\n".join(gses) + "\n")
        summary["days"].append({"day": di, "studies": len(gses), "first": gses[0], "last": gses[-1], "est_writes": writes,
                                "files": len(names), "statements": n_stmt, "samples": samples})
        summary["est_writes"] += writes
        summary["n_files"] += len(names)
        summary["n_statements"] += n_stmt
    lines = [f"{summary['n_studies']} studies, {summary['n_files']} files, {summary['n_statements']} statements, about {summary['est_writes']} "
             f"estimated D1 row writes in {len(days)} day(s) of at most {day_budget} (v2_sample key: {'gse_id, rev, gsm_id (schema 033)' if schema == '033' else 'gse_id, gsm_id (schema 032)'}):"]
    for d in summary["days"]:
        lines.append(f"  day{d['day']:02d}: {d['studies']:5d} studies  {d['samples']:6d} samples  ~{d['est_writes']:6d} writes  {d['files']:4d} files  ({d['first']}..{d['last']})")
    for g, why in skipped.items():
        lines.append(f"  SKIPPED {g}: {why}")
    write_lf(out / "PLAN.txt", "\n".join(lines) + "\n")
    write_lf(out / "plan.json", json.dumps(summary, indent=1))
    return summary


# ── POST (the website's manifest-based ingest contract) ──────────────────────────────────────────────────────────────


def post_json(url: str, token: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Ingest-Token": token,
                 "User-Agent": "singlet-v2-campaign/1.0"},  # the site answers 403 to the default Python user agent
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def post_payloads(manifest: dict, rows: Dict[str, Any], slice_size: int = MAX_POST_SAMPLES, max_body: int = MAX_POST_BODY - MIB) -> List[dict]:
    """The bodies for POST /api/ingest/v2-study: the manifest without `study_files`, in slices of at most `slice_size` samples that
    share rev, n_samples and bytes (`final: false` until the last), the migration block on the last."""
    base = {k: v for k, v in manifest.items() if k not in ("study_files", "samples")}
    samples = manifest["samples"]
    study = rows["v2_study"]
    mig = rows["v2_migration"]
    migration = {k: v for k, v in mig.items() if k not in ("gse_id", "members_by_kind") and v is not None}
    if mig.get("members_by_kind"):
        migration["members_by_kind"] = json.loads(mig["members_by_kind"])
    if "verify_ok" in migration:
        migration["verify_ok"] = bool(migration["verify_ok"])
    n = slice_size
    while True:
        chunks = [samples[i : i + n] for i in range(0, len(samples), n)]
        bodies = []
        for i, ch in enumerate(chunks):
            last = i == len(chunks) - 1
            man = dict(base, samples=ch, n_samples=len(samples), bytes=study["bytes"], n_cells_called=study["n_cells_called"])
            body: Dict[str, Any] = {"manifest": man, "final": last}
            if last:
                body.update(manifest_bytes=study["manifest_bytes"], manifest_sha256=study["manifest_sha256"], migration=migration)
            bodies.append(body)
        if n == 1 or all(len(json.dumps(b)) <= max_body for b in bodies):
            return bodies
        n = max(1, n // 2)


def post_studies(base_url: str, token: str, items: Dict[str, Tuple[dict, dict]], log=print) -> Dict[str, int]:
    """POST every study: items = {gse: (manifest, rows)}. Retries 429/5xx; a 4xx stops (the body says why)."""
    url = f"{base_url.rstrip('/')}/api/ingest/v2-study"
    counts = {"studies": 0, "posts": 0}
    for g in accession_order(items):
        manifest, rows = items[g]
        for body in post_payloads(manifest, rows):
            for attempt in range(6):
                try:
                    post_json(url, token, body)
                    break
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "replace")[:300]
                    if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                        time.sleep(min(60, 2 ** attempt * 2))
                        continue
                    raise RuntimeError(f"POST {g} -> HTTP {e.code}: {msg}") from None
                except (urllib.error.URLError, socket.timeout):
                    if attempt == 5:
                        raise
                    time.sleep(min(60, 2 ** attempt * 2))
            counts["posts"] += 1
        counts["studies"] += 1
        log(f"posted {g} ({counts['studies']}/{len(items)})")
    return counts


# ── the study pipeline ────────────────────────────────────────────────────────────────────────────────────────────────


class Ctx:
    """Everything one process shares across studies."""

    def __init__(self, a: argparse.Namespace, store) -> None:
        self.a = a
        self.store = store
        self.public: str = a.public
        self.apply: bool = a.mode != "dry"
        self.w = GuardedStore(store, {self.public: ("v2/",)})  # the only writer handed to study code
        self.part_size = int(getattr(a, "part_size_mib", 8) * MIB)
        self.io = cf.ThreadPoolExecutor(max(1, a.io_threads), thread_name_prefix="io")
        self.budget = ByteBudget(int(a.scratch_budget_gb * 1e9))
        self.scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="v2c-"))
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.host = socket.gethostname()
        self.commit = a.commit or os.environ.get("COMMIT", "")[:40] or None
        self.task = a.task or os.environ.get("SLURM_ARRAY_TASK_ID") or "0"
        self.migrate_cmd = _split_cmd(a.migrate_cmd) or [sys.executable, "-m", "singlet.v2.migrate"]
        self.verify_cmd = _split_cmd(a.verify_cmd) or [sys.executable, "-m", "singlet.v2.verify"]
        env = dict(os.environ)
        py = a.pythonpath or str(REPO / "python")
        env["PYTHONPATH"] = py + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # singlet.v2.migrate has no thread option (its native encoder uses 4 threads for a few seconds per sample, everything
        # else is one Python thread): keep BLAS from multiplying that by the number of workers
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[k] = str(a.blas_threads)
        env["SINGLET_PZW_AUTOBUILD"] = "0"  # W3's recipes build a helper executable on Windows only; never race to do it on the cluster
        env["SINGLET_V2_RECIPES_CACHE"] = str(self.scratch / "recipes-cache")
        self.migrate_args = _split_cmd(getattr(a, "migrate_args", None)) or []
        jt = getattr(a, "junction_table", None)
        if jt and "--junction-table" not in self.migrate_args:  # junction rows as table ids (docs/format-v2.md 12)
            self.migrate_args = self.migrate_args + ["--junction-table", jt]
        self.verify_args = _split_cmd(getattr(a, "verify_args", None)) or []
        if getattr(a, "verify_quick", False):
            self.verify_args = self.verify_args + ["--quick"]
        for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "INGEST_TOKEN"):
            env.pop(k, None)  # the transcoder never needs credentials
        self.env = env
        self.results_lock = threading.Lock()

    def close(self) -> None:
        self.io.shutdown(wait=True)


def _split_cmd(s: Optional[str]) -> Optional[List[str]]:
    """A command given as a JSON list (exact, any platform) or as a shell-style string."""
    import shlex

    if not s:
        return None
    if s.lstrip().startswith("["):
        return [str(x) for x in json.loads(s)]
    return shlex.split(s)


class RangeFile(io.RawIOBase):
    """Read-only file over ranged GETs, so zipfile can read the central directory without the whole object."""

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


def restricted_summary_remote(store, bucket: str, key: str, size: int) -> dict:
    """{members, compressed_bytes} of restricted members, from the zip central directory only (singlet/_restricted.py)."""
    return _restricted_module().restricted_summary(RangeFile(store, bucket, key, size))


_RESTRICTED_LOCK = threading.Lock()
_RESTRICTED_MOD = None


def _restricted_module():
    """singlet/_restricted.py (standard library only) loaded by path, once, safely when several study threads ask at once."""
    global _RESTRICTED_MOD
    with _RESTRICTED_LOCK:
        if _RESTRICTED_MOD is None:
            import importlib.util

            spec = importlib.util.spec_from_file_location("singlet_restricted_v2c", REPO / "python" / "singlet" / "_restricted.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            _RESTRICTED_MOD = mod
        return _RESTRICTED_MOD


def ensure_object(ctx: Ctx, key: str, path: Path, d: FileDigest, *, readback: str, bucket: Optional[str] = None) -> dict:
    """Make `bucket/key` hold exactly `path`: skip if it already does, else upload and check the ETag (fallback: a SHA-256
    read-back, which also accepts an ETag the server computes differently from S3's rule)."""
    bucket = bucket or ctx.public
    h = ctx.store.head(bucket, key)
    if h is not None and h.size == d.size and (h.etag == d.etag or h.meta.get("sha256") == d.sha256):
        return {"key": key, "bytes": d.size, "etag": h.etag, "action": "skipped"}
    ct, cc = object_headers(key)
    last: Optional[BaseException] = None
    for attempt in range(3):
        try:
            etag = ctx.w.put_file(bucket, key, path, content_type=ct, cache_control=cc, metadata={"sha256": d.sha256},
                                  part_size=ctx.part_size)
            rec = {"key": key, "bytes": d.size, "etag": etag, "action": "uploaded", "parts": d.n_parts}
            h = ctx.store.head(bucket, key)
            if h is None or h.size != d.size:
                raise IOError(f"{key}: stored size {None if h is None else h.size}, expected {d.size}")
            if h.etag != d.etag or etag != d.etag or readback == "sha256":
                got = ctx.store.sha256_object(bucket, key, d.size)
                if got != d.sha256:
                    raise IOError(f"{key}: read-back sha256 {got[:12]} != local {d.sha256[:12]}")
                if h.etag != d.etag:
                    rec["etag_note"] = f"server ETag {h.etag} != expected {d.etag}; content verified by sha256"
                rec["readback"] = "sha256"
            return rec
        except Exception as e:  # noqa: BLE001
            last = e
            if isinstance(e, WriteRefused):
                raise
            time.sleep(_sleep_for(ctx, attempt))
    raise StudyFailure("upload_failed", "upload", f"{key}: {type(last).__name__}: {last}")


def check_done(ctx: Ctx, gse: str, legacy: Head) -> Optional[dict]:
    """The result row of a study that is already migrated against exactly this legacy object, else None."""
    mh = ctx.store.head(ctx.public, manifest_key(gse))
    if mh is None:
        return None
    try:
        m = json.loads(ctx.store.get_bytes(ctx.public, manifest_key(gse)))
    except ValueError:
        return None
    lg = m.get("legacy") or {}
    if lg.get("etag") != legacy.etag or lg.get("bytes") != legacy.size or not m.get("samples"):
        return None

    def present(s: dict) -> bool:
        h = ctx.store.head(ctx.public, s.get("key", ""))
        return h is not None and h.size == s.get("bytes")

    if not all(ctx.io.map(present, m["samples"])):
        return None
    prior = last_committed(read_ledger(ctx.store, ctx.public, gse))
    if prior is None or prior.get("status") != "ok":
        return None
    return prior


def publish_bundle(ctx: Ctx, gse: str, bundle: Path, work: Path, legacy: Optional[dict], row: dict, events: List[dict]) -> None:
    """Migrate, validate, verify, upload, commit. `legacy` is None for a new study that has no legacy bundle."""
    a = ctx.a
    tm = row["timings"]

    def ev(name: str, **kw: Any) -> None:
        events.append({"at": iso(), "gse": gse, "event": name, **kw})

    out = work / "out"
    out.mkdir(exist_ok=True)
    study_log = Path(a.study_log_dir) / f"{gse}.log" if a.study_log_dir else work / "study.log"

    # 3. transcode
    row["stage"] = "migrate"
    rc, tail, secs = run_logged(ctx.migrate_cmd + ["--bundle", str(bundle), "--out", str(out), "--gse", gse] + ctx.migrate_args,
                                cwd=work, env=ctx.env, timeout=_tool_timeout(ctx, a.migrate_timeout), log_path=study_log)
    tm["migrate"] = round(secs, 2)
    ev("migrated", rc=rc, secs=round(secs, 2))
    if rc != 0:
        row["tail"] = tail[-1500:]
        raise StudyFailure("migrate_failed", "migrate", f"migrate exited {rc}")

    # 4. validate the tree, add the legacy facts to the manifest, write it where verify will look
    row["stage"] = "collect"
    t = time.time()
    root = find_v2_root(out, gse)
    plan = collect_objects(root, gse, ctx.part_size, ctx.io)
    for pk in plan.pack_keys:
        if not any(o.key == pk for o in plan.objects) and ctx.store.head(ctx.public, pk) is None:
            raise StudyFailure("collect_failed", "collect", f"manifest needs pack {pk}, which was not produced and is not in R2")
    manifest_bytes = finalize_manifest(plan.manifest, gse, legacy, (root / "m" / f"{gse}.json").read_bytes())
    (root / "m" / f"{gse}.json").write_bytes(manifest_bytes)
    final_manifest = json.loads(manifest_bytes)
    md = digest_bytes(manifest_bytes, ctx.part_size)
    tm["collect"] = round(time.time() - t, 2)
    sm = summarize_manifest(final_manifest)
    cells_bytes = sum(o.digest.size for o in plan.objects if RE_CELLS.match(o.key))
    sample_objs = [o for o in plan.objects if RE_SAMPLE.match(o.key)]
    row.update(
        n_samples=sm["n_samples"], n_cells_called=sm["n_called"], members_by_kind=sm["members_by_kind"], n_members=sm["n_members"],
        v2_sample_bytes=sum(o.digest.size for o in sample_objs), manifest_bytes=md.size, manifest_sha256=md.sha256,
        v2_bytes=sum(o.digest.size for o in sample_objs) + md.size + cells_bytes,
        v2_pack_bytes=sum(o.digest.size for o in plan.objects if RE_PACK.match(o.key)),
        n_objects=len(plan.objects) + 1, warnings=plan.warnings[:10],
    )
    # the transcoder's own ledger (one line per member: treatment, recipe, why a member was stored) joins ours in R2
    mled = root / "ledger" / f"{gse}.jsonl"
    n_mled = 0
    if mled.is_file():
        for ln in mled.read_text("utf-8").splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if isinstance(rec, dict):
                events.append({"at": iso(), "event": "migrate:" + str(rec.get("type", "line")), **{k: v for k, v in rec.items() if k not in ("at", "event")}})
                n_mled += 1
    row["migrate_ledger_lines"] = n_mled
    ev("collected", n_objects=row["n_objects"], v2_bytes=row["v2_bytes"], members_by_kind=sm["members_by_kind"])

    # 5. verify against the legacy bundle (must exit 0)
    row["stage"] = "verify"
    rc, tail, secs = run_logged(ctx.verify_cmd + ["--bundle", str(bundle), "--v2", str(out)] + ctx.verify_args, cwd=work,
                                env=ctx.env, timeout=_tool_timeout(ctx, a.verify_timeout), log_path=study_log)
    row["verify"] = {"ok": rc == 0, "rc": rc, "secs": round(secs, 2)}
    tm["verify"] = round(secs, 2)
    ev("verified", rc=rc, secs=round(secs, 2))
    if rc != 0:
        row["verify"]["tail"] = tail[-1500:]
        raise StudyFailure("verify_failed", "verify", f"verify exited {rc}")
    row["verified_at"] = iso()

    # dry: report what would be written and stop (HEAD only)
    if not ctx.apply:
        row["stage"] = "dry-head"
        heads = list(ctx.io.map(lambda o: (o, ctx.store.head(ctx.public, o.key)), plan.objects))
        have = [o for o, h in heads if h is not None and h.size == o.digest.size and (h.etag == o.digest.etag or h.meta.get("sha256") == o.digest.sha256)]
        row["uploads"] = {"objects": len(plan.objects), "would_upload": len(plan.objects) - len(have),
                          "would_skip": len(have), "bytes_would_upload": sum(o.digest.size for o in plan.objects if o not in have)}
        row["status"] = "dry_ok"
        return

    # 6. upload everything except the manifest
    row["stage"] = "upload"
    t = time.time()
    results = list(ctx.io.map(lambda o: _safe_ensure(ctx, o), plan.objects))
    failed = [r for r in results if isinstance(r, StudyFailure)]
    if failed:
        raise failed[0]
    tm["upload"] = round(time.time() - t, 2)
    up = [r for r in results if r["action"] == "uploaded"]
    row["uploads"] = {"objects": len(results), "uploaded": len(up), "skipped": len(results) - len(up),
                      "bytes_uploaded": sum(r["bytes"] for r in up)}
    for r in results:
        ev("object", **r)

    # 7. ledger (with every object record), manifest last, read-back, committed
    row["stage"] = "ledger"
    prior = read_ledger(ctx.store, ctx.public, gse)
    write_ledger(ctx.w, ctx.public, gse, prior + events)
    row["stage"] = "manifest"
    t = time.time()
    ct, cc = object_headers(manifest_key(gse))
    ctx.w.put_bytes(ctx.public, manifest_key(gse), manifest_bytes, content_type=ct, cache_control=cc,
                    metadata={"sha256": md.sha256})
    if ctx.store.get_bytes(ctx.public, manifest_key(gse)) != manifest_bytes:
        raise StudyFailure("upload_failed", "manifest", "manifest read back different from what was written")
    tm["manifest"] = round(time.time() - t, 2)
    row["status"] = "ok"
    row["stage"] = "done"
    ev("committed", summary=_public_row(row))
    write_ledger(ctx.w, ctx.public, gse, prior + events)
    row["_manifest"] = final_manifest


def _safe_ensure(ctx: Ctx, o: Obj):
    try:
        return ensure_object(ctx, o.key, o.path, o.digest, readback=ctx.a.readback)
    except StudyFailure as e:
        return e


def _public_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def process_study(ctx: Ctx, gse: str, provider=None) -> dict:
    """One study end to end; never raises. The returned row carries status, stage and error.

    `provider(ctx, gse, work, row, events) -> (bundle path, legacy dict or None)` replaces the legacy download: a new study is
    packed locally from pipeline output (scripts/v2_publish.py) and goes through the same transcode / verify / upload /
    manifest-last steps. Without it the live legacy bundle in R2 is read."""
    a = ctx.a
    t0 = time.time()
    row: Dict[str, Any] = {"gse": gse, "status": "failed", "stage": "init", "at": iso(), "commit": ctx.commit, "host": ctx.host,
                           "task": ctx.task, "mode": a.mode, "timings": {}}
    events: List[dict] = [{"at": iso(), "gse": gse, "event": "start", "mode": a.mode, "commit": ctx.commit, "host": ctx.host}]
    work: Optional[Path] = None
    reserved = 0
    try:
        if ctx.stop.is_set():
            row.update(status="deferred", stage="queue", error="stop requested")
            return row
        if provider is not None:
            work = ctx.scratch / gse
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            row["stage"] = "pack"
            t = time.time()
            bundle, legacy = provider(ctx, gse, work, row, events)
            row["timings"]["pack"] = round(time.time() - t, 2)
            publish_bundle(ctx, gse, bundle, work, legacy, row, events)
            return row
        row["stage"] = "head"
        lk = legacy_key(gse)
        h = ctx.store.head(ctx.public, lk)
        if h is None:
            row["status"] = "missing"
            return row
        row.update(legacy_key=lk, legacy_etag=h.etag, legacy_bytes=h.size)
        if a.deadline_epoch and time.time() + h.size / (a.min_rate_mbs * 1e6) > a.deadline_epoch:
            row.update(status="deferred", stage="queue", error="not enough time left in this job")
            return row
        if a.resume:
            prior = check_done(ctx, gse, h)
            if prior is not None:
                row.update(status="skipped_done", stage="done", _prior=prior, n_samples=prior.get("n_samples"),
                           v2_bytes=prior.get("v2_bytes"), members_by_kind=prior.get("members_by_kind"),
                           legacy_sha256=prior.get("legacy_sha256"))
                return row
        row["stage"] = "restricted-check"
        summ = restricted_summary_remote(ctx.store, ctx.public, lk, h.size)
        row["restricted_members"] = summ["members"]
        if summ["members"] and not a.allow_restricted:
            row.update(status="blocked_restricted",
                       error=f"{summ['members']} restricted members still in the public bundle; run the restrict migration first")
            return row
        if a.list_only:
            row["status"] = "dry_listed"
            return row

        need = int(h.size * 3) + 64 * MIB
        ctx.budget.acquire(need)
        reserved = need
        work = ctx.scratch / gse
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        bundle = work / f"{gse}.singlet"

        row["stage"] = "download"
        t = time.time()
        try:
            ctx.store.download(ctx.public, lk, bundle, if_match=h.etag, size=h.size)
        except LegacyChanged as e:
            raise StudyFailure("legacy_changed", "download", str(e))
        if bundle.stat().st_size != h.size:
            raise StudyFailure("failed", "download", f"downloaded {bundle.stat().st_size} bytes, object is {h.size}")
        row["timings"]["download"] = round(time.time() - t, 2)
        t = time.time()
        legacy_sha = sha256_file(bundle)
        row["legacy_sha256"] = legacy_sha
        row["timings"]["hash"] = round(time.time() - t, 2)
        h2 = ctx.store.head(ctx.public, lk)
        if h2 is None or h2.etag != h.etag or h2.size != h.size:
            raise StudyFailure("legacy_changed", "download", "legacy object changed during the download")
        events.append({"at": iso(), "gse": gse, "event": "downloaded", "bytes": h.size, "etag": h.etag, "sha256": legacy_sha,
                       "secs": row["timings"]["download"]})
        legacy = {"key": lk, "etag": h.etag, "bytes": h.size, "sha256": legacy_sha, "migrated_at": iso()}
        publish_bundle(ctx, gse, bundle, work, legacy, row, events)
    except StudyFailure as e:
        row.update(status=e.status, error=str(e)[:600])
    except Exception as e:  # noqa: BLE001  a bad study must not stop the shard
        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:600], trace=traceback.format_exc()[-1500:])
    finally:
        if reserved:
            ctx.budget.release(reserved)
        if work is not None and not (a.keep_scratch == "always" or (a.keep_scratch == "failed" and row["status"] in BAD_STATUSES)):
            shutil.rmtree(work, ignore_errors=True)
        row["timings"]["total"] = round(time.time() - t0, 2)
        _finish(ctx, row, events)
    return row


def _tool_timeout(ctx: Ctx, default: float) -> float:
    """A transcoder or verifier call may not run past the job's deadline (it would be killed with the job and leave nothing)."""
    dl = getattr(ctx.a, "deadline_epoch", None)
    return default if not dl else max(60.0, min(default, dl - time.time()))


def _sleep_for(ctx: Ctx, attempt: int) -> float:
    return min(30.0, getattr(ctx.a, "retry_sleep", 1.0) * 2 ** attempt)


def _finish(ctx: Ctx, row: dict, events: List[dict]) -> None:
    """Failure events to the ledger (apply mode), the local ledger copy, and the catalog rows file."""
    a = ctx.a
    gse = row["gse"]
    try:
        if row["status"] != "ok" and ctx.apply and row["status"] in BAD_STATUSES | {"blocked_restricted"}:
            events.append({"at": iso(), "gse": gse, "event": "error", "status": row["status"], "stage": row.get("stage"),
                           "error": row.get("error")})
            write_ledger(ctx.w, ctx.public, gse, read_ledger(ctx.store, ctx.public, gse) + events)
    except Exception:  # noqa: BLE001  the ledger is best effort on failure
        pass
    if a.ledger_dir and ctx.apply:
        d = Path(a.ledger_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{gse}.jsonl", "a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e, sort_keys=True) + "\n")
    if a.rows_dir and ctx.apply and row["status"] in ({"ok", "skipped_done"} | BAD_STATUSES | {"blocked_restricted", "missing"}):
        try:
            d = Path(a.rows_dir)
            d.mkdir(parents=True, exist_ok=True)
            if row["status"] == "skipped_done":  # nothing changed: the rows describe the earlier, committed migration
                manifest = json.loads(ctx.store.get_bytes(ctx.public, manifest_key(gse)))
                rows = build_catalog_rows(manifest, dict(row["_prior"], gse=gse), a.members)
            elif row["status"] == "ok":
                rows = build_catalog_rows(row["_manifest"], _public_row(row), a.members)
            else:  # a failed attempt: a `failed` migration row, unless an earlier run left rows for a published study (never downgrade it)
                rows = build_catalog_rows(None, _public_row(row), a.members)
                old = d / f"{gse}.json"
                if old.exists() and "v2_sample" in json.loads(old.read_text("utf-8")):
                    rows = None
            if rows is not None:
                tmp = d / f".{gse}.json.tmp"
                write_lf(tmp, json.dumps(rows, sort_keys=True))
                os.replace(tmp, d / f"{gse}.json")
        except Exception as e:  # noqa: BLE001  (CatalogError: the study is live in R2 but cannot be described to D1 as it is)
            row["catalog_error"] = f"{type(e).__name__}: {e}"[:300]
            row.setdefault("warnings", []).append(f"catalog rows not written: {e}")
    row.pop("_manifest", None)
    row.pop("_prior", None)


# ── selecting studies ─────────────────────────────────────────────────────────────────────────────────────────────────


def read_list(path: str) -> List[str]:
    return [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines() if ln.split("#")[0].strip()]


def parse_shard(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d+)/([1-9]\d*)", s)
    if not m or int(m.group(1)) >= int(m.group(2)):
        raise argparse.ArgumentTypeError("--shard is I/N with 0 <= I < N, e.g. 3/32")
    return int(m.group(1)), int(m.group(2))


def pilot_pick(sized: List[Tuple[str, int]], n: int, max_bytes: float) -> List[str]:
    """n studies spread over the size distribution (excluding the very largest), deterministic."""
    cand = sorted((t for t in sized if t[1] <= max_bytes), key=lambda t: (t[1], t[0]))
    if not cand or n <= 0:
        return []
    if n >= len(cand):
        return [g for g, _ in cand]
    idx = sorted({round(i * (len(cand) - 1) / (n - 1)) if n > 1 else 0 for i in range(n)})
    return [cand[i][0] for i in idx]


def select_studies(a: argparse.Namespace, store) -> List[Tuple[str, Optional[int]]]:
    sized: Optional[List[Tuple[str, int]]] = None
    explicit: List[str] = list(a.gse or [])
    if a.gse_list:
        explicit += read_list(a.gse_list)
    explicit = accession_order(explicit) if explicit else []
    listing: Dict[str, int] = {}
    if not explicit or a.mode == "pilot" or a.listing:  # (an explicit list with a listing gets its sizes: largest first)
        if a.listing:  # a frozen `plan --out` listing: every shard of a campaign slices the same list
            sized = [(ln.split("\t")[0], int(ln.split("\t")[1])) for ln in Path(a.listing).read_text().splitlines()
                     if ln.strip() and not ln.startswith("gse")]
        else:
            sized = store.list_studies(a.public)
        listing = dict(sized)
    if a.mode == "pilot":
        picks = pilot_pick(sized or [], a.pilot_n, a.pilot_max_gb * 1e9)
        gses = accession_order(set(picks) | set(explicit))
    elif explicit:
        gses = explicit
    else:
        gses = accession_order(listing)
    if a.exclude_list:
        ex = set(read_list(a.exclude_list))
        gses = [g for g in gses if g not in ex]
    if a.shard and a.mode != "pilot":
        i, n = a.shard
        gses = gses[i::n]
    if a.limit is not None:
        gses = gses[: a.limit]
    out: List[Tuple[str, Optional[int]]] = [(g, listing.get(g)) for g in gses]
    if a.order == "size-desc" and all(s is not None for _, s in out):
        out.sort(key=lambda t: (-(t[1] or 0), len(t[0]), t[0]))
    return out


def fmt_row(r: dict) -> str:
    lb, vb = r.get("legacy_bytes") or 0, r.get("v2_bytes") or 0
    ratio = f"{vb / lb:.3f}" if lb and vb else "-"
    return "\t".join([r["gse"], r["status"], f"{lb / 1e6:.1f}MB", f"{vb / 1e6:.1f}MB", ratio,
                      f"{(r.get('timings') or {}).get('total', 0):.0f}s", (r.get("error") or "")[:160].replace("\n", " ")])


def make_store(a: argparse.Namespace):
    if getattr(a, "local_root", None):
        return LocalStore(Path(a.local_root))
    return S3Store()


def cmd_run(a: argparse.Namespace) -> int:
    if a.mode == "pilot" and a.readback is None:
        a.readback = "sha256"
    if a.readback is None:
        a.readback = "none"
    store = make_store(a)
    studies = select_studies(a, store)
    ctx = Ctx(a, store)
    if a.deadline_epoch is None and os.environ.get("DEADLINE_EPOCH"):
        a.deadline_epoch = float(os.environ["DEADLINE_EPOCH"])
    old = signal.signal(signal.SIGTERM, lambda *_: ctx.stop.set()) if threading.current_thread() is threading.main_thread() else None
    results_fh = open(a.results, "a", encoding="utf-8") if a.results else None
    total_bytes = sum(s or 0 for _, s in studies)
    print(f"[v2_campaign] mode={a.mode} studies={len(studies)} bytes={total_bytes / 1e9:.1f}GB workers={a.workers} "
          f"io_threads={a.io_threads} blas_threads={a.blas_threads} shard={a.shard} resume={a.resume} "
          f"readback={a.readback} commit={ctx.commit}", flush=True)
    counts: Dict[str, int] = {}
    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(max(1, a.workers), thread_name_prefix="study") as pool:
            futs = [pool.submit(process_study, ctx, g) for g, _ in studies]
            for f in cf.as_completed(futs):
                r = f.result()
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                print(fmt_row(r), flush=True)
                if results_fh:
                    with ctx.results_lock:
                        results_fh.write(json.dumps(_public_row(r), sort_keys=True) + "\n")
                        results_fh.flush()
    finally:
        ctx.close()
        if results_fh:
            results_fh.close()
        if old is not None:
            signal.signal(signal.SIGTERM, old)
    wall = time.time() - t0
    bad = sum(n for s, n in counts.items() if s in BAD_STATUSES)
    print(f"[v2_campaign] done in {wall:.0f}s: " + ", ".join(f"{s}={n}" for s, n in sorted(counts.items())), flush=True)
    return 1 if bad else 0


# ── plan, summarize, catalog commands, refs index ─────────────────────────────────────────────────────────────────────


def project(total_bytes: float, n_studies: int, shards: int, *, rate_dl: float, rate_cpu_core: float, cores: float,
            eff: float = 0.75, startup_min: float = 8.0, max_study_bytes: float = 0.0) -> dict:
    """Per-task and campaign time. A node moves min(download rate, compute rate) MB/s of legacy bytes."""
    node_rate = min(rate_dl, rate_cpu_core * cores * eff)
    shard_bytes = total_bytes / shards
    task_h = shard_bytes / 1e6 / node_rate / 3600 + startup_min / 60
    return {"shards": shards, "studies_per_shard": n_studies / shards, "gb_per_shard": shard_bytes / 1e9, "node_mbs": node_rate,
            "task_hours": task_h, "node_hours_total": task_h * shards,
            "bound": "download" if rate_dl <= rate_cpu_core * cores * eff else "compute",
            "longest_study_hours": (max_study_bytes / 1e6 / rate_cpu_core / 3600) if max_study_bytes else 0.0}


def cmd_plan(a: argparse.Namespace) -> int:
    if a.from_tsv:
        sized = [(ln.split("\t")[0], int(ln.split("\t")[1])) for ln in Path(a.from_tsv).read_text().splitlines()
                 if ln and not ln.startswith("gse")]
    else:
        sized = make_store(a).list_studies(a.public)
    sized.sort(key=lambda t: (len(t[0]), t[0]))
    total = sum(s for _, s in sized)
    big = [t for t in sized if t[1] >= a.big_gb * 1e9]
    rest = [t for t in sized if t[1] < a.big_gb * 1e9]
    rest_total = sum(s for _, s in rest)
    print(f"studies={len(sized)} total={total / 1e12:.3f} TB mean={total / max(1, len(sized)) / 1e6:.1f} MB "
          f"max={max((s for _, s in sized), default=0) / 1e9:.1f} GB")
    big_total = total - rest_total
    print(f"big (>= {a.big_gb} GB, run on their own): {len(big)} studies, {big_total / 1e12:.3f} TB ({100 * big_total / max(1, total):.0f}% of the bytes); "
          f"one migrate process is single-threaded: {a.big_gb} GB takes {a.big_gb * 1e3 / a.rate_cpu / 3600:.1f} h and "
          f"{max((s for _, s in big), default=0) / 1e6 / a.rate_cpu / 3600:.1f} h for the largest, at {a.rate_cpu} MB/s")
    if a.out:
        shard_of = {g: i % a.shards for i, (g, _s) in enumerate(rest)}
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("gse\tbytes\tshard\n")
            for g, s in sized:
                fh.write(f"{g}\t{s}\t{shard_of.get(g, 'big')}\n")
        print(f"wrote {a.out} (shard = position among the non-big studies in accession order mod {a.shards}; big studies: 'big')")
    if a.big_out:
        Path(a.big_out).write_text("".join(f"{g}\t# {s / 1e9:.1f} GB\n" for g, s in big), encoding="utf-8")
    print(f"non-big studies ({len(rest)}, {rest_total / 1e12:.3f} TB), {a.cores:g}-core jobs, {a.rate_cpu} MB/s per core, download {a.rate_dl} MB/s:")
    print("shards  GB/shard  studies/shard  node MB/s  bound     task h  node-h total")
    for n in sorted({16, 32, 64, 96, 128, a.shards}):
        p = project(rest_total, len(rest), n, rate_dl=a.rate_dl, rate_cpu_core=a.rate_cpu, cores=a.cores)
        print(f"{n:6d}  {p['gb_per_shard']:8.1f}  {p['studies_per_shard']:13.0f}  {p['node_mbs']:9.1f}  {p['bound']:8s}  "
              f"{p['task_hours']:6.2f}  {p['node_hours_total']:11.1f}")
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    rows: Dict[str, dict] = {}
    for f in a.results:
        for p in sorted(Path(f).glob("*.jsonl")) if Path(f).is_dir() else [Path(f)]:
            for ln in p.read_text("utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    rows[r["gse"]] = r  # later rows win
    by: Dict[str, int] = {}
    for r in rows.values():
        by[r["status"]] = by.get(r["status"], 0) + 1
    ok = [r for r in rows.values() if r["status"] == "ok"]
    lb = sum(r.get("legacy_bytes") or 0 for r in ok)
    vb = sum(r.get("v2_bytes") or 0 for r in ok)
    print("studies:", len(rows), " ".join(f"{k}={v}" for k, v in sorted(by.items())))
    if ok:
        print(f"ok studies: legacy {lb / 1e9:.2f} GB -> v2 {vb / 1e9:.2f} GB (ratio {vb / lb:.3f}); "
              f"samples {sum(r.get('n_samples') or 0 for r in ok)}")
        kinds: Dict[str, int] = {}
        for r in ok:
            for k, n in (r.get("members_by_kind") or {}).items():
                kinds[k] = kinds.get(k, 0) + n
        print("members by kind:", kinds)
        stage = {k: sum((r.get("timings") or {}).get(k, 0) for r in ok) for k in ("download", "hash", "migrate", "verify", "upload", "total")}
        print("stage seconds (sum over studies):", {k: round(v) for k, v in stage.items()})
        for k in ("download", "migrate", "verify"):
            if stage[k] > 0:
                print(f"  {k}: {lb / 1e6 / stage[k]:.1f} MB/s of legacy bytes per worker")
        up = sum((r.get("uploads") or {}).get("bytes_uploaded", 0) for r in ok)
        if stage["upload"] > 0:
            print(f"  upload: {up / 1e6 / stage['upload']:.1f} MB/s per worker (uploaded bytes)")
        if stage["total"] > 0:
            per_worker = lb / 1e6 / stage["total"]
            total_mb = a.atlas_tb * 1e6
            print(f"  end to end: {per_worker:.1f} MB/s per worker; whole atlas ({a.atlas_tb} TB): "
                  f"{total_mb / per_worker / 3600:.1f} worker-hours = {total_mb / per_worker / 3600 / a.workers:.1f} node-hours at {a.workers} workers/node")
    bad = [r for r in rows.values() if r["status"] in BAD_STATUSES or r["status"] == "blocked_restricted"]
    for r in bad[: a.show]:
        print(f"  {r['gse']}\t{r['status']}\t{r.get('stage')}\t{(r.get('error') or '')[:200]}")
    return 1 if any(r["status"] in BAD_STATUSES for r in rows.values()) else 0


def _catalog_sources(a: argparse.Namespace) -> Dict[str, dict]:
    """Catalog rows from the rows directories of `run`, or rebuilt from the manifests and ledgers in R2 (--from-r2)."""
    if a.from_r2:
        store = make_store(a)
        if a.gse_list:
            gses = read_list(a.gse_list)
        else:  # the manifests that exist: legacy keys disappear when the bundles are retired
            gses = [m.group(1) for k, _ in store.list_keys(a.public, "v2/m/") if (m := RE_MANIFEST.match(k))]
        return rows_from_r2(store, a.public, gses, a.members)
    studies = load_rows([Path(p) for p in a.rows_dir])
    if a.gse_list:
        keep = set(read_list(a.gse_list))
        studies = {g: v for g, v in studies.items() if g in keep}
    return studies


def cmd_catalog_sql(a: argparse.Namespace) -> int:
    studies = _catalog_sources(a)
    if not studies:
        print("no studies: give --rows-dir (written by `run --rows-dir`) or --from-r2", file=sys.stderr)
        return 2
    plan = write_sql_files(studies, Path(a.out), day_budget=a.day_budget, max_rows=a.max_rows, max_bytes=a.max_bytes,
                           database=a.database, schema=a.schema)
    print((Path(a.out) / "PLAN.txt").read_text(encoding="utf-8"), end="")
    return 1 if plan["skipped"] else 0


def cmd_catalog_post(a: argparse.Namespace) -> int:
    token = os.environ.get(a.token_env, "")
    store = make_store(a)
    gses = read_list(a.gse_list) if a.gse_list else [m.group(1) for k, _ in store.list_keys(a.public, "v2/m/") if (m := RE_MANIFEST.match(k))]
    items: Dict[str, Tuple[dict, dict]] = {}
    for g in accession_order(gses):
        if store.head(a.public, manifest_key(g)) is None:
            continue
        raw = store.get_bytes(a.public, manifest_key(g))
        prior = last_committed(read_ledger(store, a.public, g))
        if prior is None:
            print(f"{g}: no committed ledger line, skipped")
            continue
        md = digest_bytes(raw)
        m = json.loads(raw)
        try:
            rows = build_catalog_rows(m, dict(prior, gse=g, manifest_bytes=md.size, manifest_sha256=md.sha256, status="ok"), a.members)
        except CatalogError as e:
            print(f"{g}: {e}")
            continue
        items[g] = (m, rows)
    if a.dry_run:
        n = sum(len(post_payloads(m, r)) for m, r in items.values())
        print(f"would POST {n} bodies for {len(items)} studies to {a.base_url}/api/ingest/v2-study")
        return 0
    if not token:
        print(f"error: ${a.token_env} is empty (the ingest token)", file=sys.stderr)
        return 2
    print("posted", post_studies(a.base_url, token, items))
    return 0


def load_event_list(paths: Sequence[str]) -> Dict[str, dict]:
    """{gse: last entry} from archived / restored lists written by v2_retire.py (JSON lines; a bare GSE id per line counts as archived).
    The last line of a study wins, by its `at` time and then by position; `event` is `archived` (default) or `restored`."""
    out: Dict[str, dict] = {}
    for p in paths:
        for ln in Path(p).read_text("utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            if ln.startswith("{"):
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
            else:
                rec = {"gse": ln}
            g = rec.get("gse")
            if not isinstance(g, str) or not re.fullmatch(r"GSE\d+", g):
                continue
            rec.setdefault("event", "archived")
            if g not in out or str(rec.get("at", "")) >= str(out[g].get("at", "")):
                out[g] = rec
    return out


def cmd_retire_sql(a: argparse.Namespace) -> int:
    """Step (B) of the retirement: SQL that flips v2_migration.status from `verified` to `retired` for the studies in the archived
    list(s) (written by `v2_retire.py archive`, which has verified the archive copy of each), and back from `retired` to `verified`
    for studies the list says were restored. `UPDATE ... WHERE status = 'verified'` moves only a verified row, so applying a file twice
    is harmless; a row is about 3 D1 writes, so the whole atlas (8.2k studies) is one day. The orchestrator applies the files with
    wrangler; the site stops offering the legacy file from the first row on, and only then may `v2_retire.py delete` remove it."""
    last = load_event_list(a.list)
    retired = accession_order(g for g, r in last.items() if r["event"] == "archived")
    restored = accession_order(g for g, r in last.items() if r["event"] == "restored")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.sql"):
        f.unlink()
    names: List[str] = []
    for tag, gses, new, old in (("retired", retired, "retired", "verified"), ("verified", restored, "verified", "retired")):
        for i in range(0, len(gses), a.chunk):
            ids = ", ".join(sql_literal(g) for g in gses[i : i + a.chunk])
            name = f"{tag}_{i // a.chunk:04d}.sql"
            sql = f"UPDATE v2_migration SET status = '{new}', updated_at = {NOW} WHERE status = '{old}' AND gse_id IN ({ids});"
            write_lf(out / name, sql + chr(10))
            names.append(name)
    writes = 3 * (len(retired) + len(restored))  # a status change rewrites the row and the status index entry
    head = (f"# {len(retired)} studies to mark retired, {len(restored)} to mark verified again; about {writes} D1 row writes "
            f"({'within' if writes <= DAY_BUDGET else 'OVER'} one day's {DAY_BUDGET}). Apply in this order.")
    write_lf(out / "APPLY.txt", head + chr(10) + chr(10).join(f"wrangler d1 execute {a.database} --remote --file={n}" for n in names) + chr(10))
    print(head[2:])
    return 0


def cmd_refs_index(a: argparse.Namespace) -> int:
    """Publish the reference packs and `v2/ref/index.json` from the package's packs directory (python/singlet/v2/packs)."""
    d = Path(a.packs_dir)
    idx_path = d / "index.json"
    index = json.loads(idx_path.read_text("utf-8"))
    if index.get("format") != "singlet-v2-packs":
        print(f"{idx_path}: not a singlet-v2-packs index", file=sys.stderr)
        return 2
    store = make_store(a)
    w = GuardedStore(store, {a.public: ("v2/ref/",)})
    bad = 0
    todo: List[Tuple[str, Path]] = []
    for p in index["packs"]:
        f = d / f"{p['id']}.pack"
        if not f.is_file():
            print(f"missing pack file {f}", file=sys.stderr)
            bad += 1
            continue
        dg = hash_file(f)
        if dg.size != p["bytes"] or dg.sha256 != p["sha256"] or dg.sha256[:32] != p["id"]:
            print(f"{f.name}: size/sha256/id do not match index.json", file=sys.stderr)
            bad += 1
            continue
        todo.append((f"v2/ref/{p['id']}.pack", f))
    if bad:
        return 1
    todo.append((REFS_INDEX_KEY, idx_path))
    up = sk = 0
    for key, f in todo:
        dg = hash_file(f)
        h = store.head(a.public, key)
        if h is not None and h.size == dg.size and (h.etag == dg.etag or h.meta.get("sha256") == dg.sha256):
            sk += 1
            continue
        if a.apply:
            ct, cc = object_headers(key)
            w.put_file(a.public, key, f, content_type=ct, cache_control=cc, metadata={"sha256": dg.sha256})
            h = store.head(a.public, key)
            if h is None or h.size != dg.size:
                print(f"{key}: not stored correctly", file=sys.stderr)
                return 1
        up += 1
    print(f"{len(index['packs'])} packs + index.json: {up} {'uploaded' if a.apply else 'to upload (no --apply)'}, {sk} already there")
    return 0


# ── command line ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _cpu_count() -> int:
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n and n.isdigit():
        return int(n)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--public", default=PUBLIC_BUCKET)
        p.add_argument("--local-root", help="use directories under this root instead of R2 (tests, rehearsals)")

    r = sub.add_parser("run", help="migrate studies")
    common(r)
    r.add_argument("--mode", choices=["dry", "pilot", "full"], required=True,
                   help="dry: everything except R2 writes; pilot: --pilot-n studies spread over the size range (+ --gse), "
                        "with read-back; full: the selection with writes")
    r.add_argument("--gse", nargs="*")
    r.add_argument("--gse-list", help="file with one GSE per line (# comments allowed)")
    r.add_argument("--exclude-list")
    r.add_argument("--listing", help="frozen study listing (gse<TAB>bytes, from `plan --out`) instead of listing the bucket: "
                                     "all shards then slice the same list even if new studies appear meanwhile")
    r.add_argument("--shard", type=parse_shard, metavar="I/N", help="every N-th study of accession order, starting at I")
    r.add_argument("--limit", type=int)
    r.add_argument("--pilot-n", type=int, default=6)
    r.add_argument("--pilot-max-gb", type=float, default=8.0)
    r.add_argument("--resume", action="store_true", help="skip studies whose v2 manifest already matches the live legacy object")
    r.add_argument("--order", choices=["size-desc", "accession"], default="size-desc", help="processing order inside the shard")
    r.add_argument("--list-only", action="store_true", help="dry: only HEAD + central directory + restricted check")
    r.add_argument("--allow-restricted", action="store_true", help="migrate bundles that still hold restricted members (NOT for the public bucket)")
    r.add_argument("--workers", type=int, default=0, help="studies in flight (0 = cores - 2: a migrate process is one Python thread)")
    r.add_argument("--io-threads", type=int, default=8)
    r.add_argument("--blas-threads", type=int, default=1, help="OMP/OpenBLAS/MKL threads per transcoder process")
    r.add_argument("--migrate-args", help="extra arguments for singlet.v2.migrate (JSON list or shell-style string), e.g. --samples GSM1,GSM2")
    r.add_argument("--junction-table", default="auto",
                   help="singlet.v2.migrate --junction-table: auto (default) = the newest junction table published for the study's "
                        "reference build in v2/ref/index.json; none = inline junction names")
    r.add_argument("--verify-args", help="extra arguments for singlet.v2.verify")
    r.add_argument("--verify-quick", action="store_true", help="pass --quick to verify (skips its slow digest recomputation; NOT for the campaign)")
    r.add_argument("--migrate-cmd", help="default: <python> -m singlet.v2.migrate")
    r.add_argument("--verify-cmd", help="default: <python> -m singlet.v2.verify")
    r.add_argument("--migrate-timeout", type=float, default=3 * 3600)
    r.add_argument("--verify-timeout", type=float, default=3 * 3600)
    r.add_argument("--pythonpath", help="default: <repo>/python")
    r.add_argument("--scratch", help="node-local directory with room for ~3 copies of the largest bundle per worker")
    r.add_argument("--scratch-budget-gb", type=float, default=400.0)
    r.add_argument("--keep-scratch", choices=["never", "failed", "always"], default="never")
    r.add_argument("--readback", choices=["none", "sha256"], default=None, help="re-read uploaded objects and compare SHA-256 (pilot: on)")
    r.add_argument("--results", help="append one JSON result row per study here")
    r.add_argument("--rows-dir", help="write catalog rows per study (<GSE>.json) here")
    r.add_argument("--ledger-dir", help="also keep a local copy of each ledger here")
    r.add_argument("--study-log-dir", help="migrate/verify output per study")
    r.add_argument("--members", choices=["full", "none"], default="full", help="none: leave members_json empty in the catalog rows")
    r.add_argument("--deadline-epoch", type=float, help="do not start a study that cannot finish before this time (unix seconds)")
    r.add_argument("--min-rate-mbs", type=float, default=0.4, help="assumed worst-case MB/s of legacy bytes per study (download, migrate, verify), for --deadline-epoch")
    r.add_argument("--commit")
    r.add_argument("--task")
    r.add_argument("--part-size-mib", type=float, default=PART_SIZE / MIB, help=argparse.SUPPRESS)
    r.add_argument("--retry-sleep", type=float, default=1.0, help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("plan", help="list studies, shards and a time projection")
    common(p)
    p.add_argument("--from-tsv", help="a listing (gse<TAB>bytes) instead of listing the bucket")
    p.add_argument("--out", help="write gse/bytes/shard here")
    p.add_argument("--shards", type=int, default=64)
    p.add_argument("--rate-dl", type=float, default=150.0, help="MB/s per node from R2")
    p.add_argument("--rate-cpu", type=float, default=1.0, help="MB/s of legacy bytes per core for migrate + verify together (one run of GSE145370 on the laptop: 1.7; W2 on a loaded laptop: 0.5 to 1)")
    p.add_argument("--cores", type=float, default=16.0)
    p.add_argument("--big-gb", type=float, default=5.0, help="studies at least this large (one migrate process is single-threaded: about 3 h at the default rate) go to big.txt")
    p.add_argument("--big-out")
    p.set_defaults(func=cmd_plan)

    s = sub.add_parser("summarize", help="aggregate result files")
    s.add_argument("results", nargs="+", help="results jsonl files or directories of them")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--atlas-tb", type=float, default=2.7)
    s.add_argument("--show", type=int, default=20)
    s.set_defaults(func=cmd_summarize)

    c = sub.add_parser("catalog-sql", help="SQL files for wrangler d1 execute (schema 032), split into days of at most --day-budget row writes")
    common(c)
    c.add_argument("--rows-dir", nargs="*", default=[], help="rows directories written by `run --rows-dir`")
    c.add_argument("--from-r2", action="store_true", help="rebuild the rows from the v2/m manifests and ledgers in R2 instead")
    c.add_argument("--gse-list")
    c.add_argument("--out", required=True)
    c.add_argument("--database", default="singlet-catalog")
    c.add_argument("--day-budget", type=int, default=DAY_BUDGET, help="estimated D1 row writes per day (a sample row is 3)")
    c.add_argument("--max-rows", type=int, default=1000, help="statements per file")
    c.add_argument("--max-bytes", type=int, default=3_000_000, help="bytes per file")
    c.add_argument("--members", choices=["full", "none"], default="full", help="none: leave members_json empty (smaller files)")
    c.add_argument("--schema", choices=["033", "032"], default="033",
                   help="v2_sample key: 033 = (gse_id, rev, gsm_id), what the ingest code writes; 032 = (gse_id, gsm_id), if 033 was not applied")
    c.set_defaults(func=cmd_catalog_sql)

    q = sub.add_parser("catalog-post", help="POST the manifests to /api/ingest/v2-study (needs the ingest token)")
    common(q)
    q.add_argument("--gse-list")
    q.add_argument("--base-url", default="https://singlet.bio")
    q.add_argument("--token-env", default="INGEST_TOKEN")
    q.add_argument("--members", choices=["full", "none"], default="full")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_catalog_post)

    x = sub.add_parser("refs-index", help="publish the reference packs and v2/ref/index.json from the package's packs directory")
    common(x)
    x.add_argument("--packs-dir", default=str(REPO / "python" / "singlet" / "v2" / "packs"))
    x.add_argument("--apply", action="store_true")
    x.set_defaults(func=cmd_refs_index)

    t = sub.add_parser("retire-sql", help="step B of the retirement: SQL flipping v2_migration.status verified -> retired for the archived studies")
    t.add_argument("--list", nargs="+", required=True, help="archived / restored lists written by v2_retire.py")
    t.add_argument("--out", required=True)
    t.add_argument("--database", default="singlet-catalog")
    t.add_argument("--chunk", type=int, default=500, help="studies per statement")
    t.set_defaults(func=cmd_retire_sql)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.workers <= 0:
            a.workers = max(1, _cpu_count() - 2)
        if a.mode in ("dry", "full") and not (a.gse or a.gse_list or a.shard or a.limit is not None or os.environ.get("V2_ALL")):
            ap.error(f"{a.mode} mode needs --shard I/N, --gse, --gse-list or --limit (V2_ALL=1 runs every study in one process)")
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
