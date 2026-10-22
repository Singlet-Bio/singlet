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
  refs-index   rebuild v2/ref/index.json from the packs in v2/ref/

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

    def _write_meta(self, bucket: str, key: str, meta: Dict[str, Any]) -> None:
        mp = self._mp(bucket, key)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta))

    def head(self, bucket: str, key: str) -> Optional[Head]:
        p = self._p(bucket, key)
        if not p.is_file():
            return None
        mp = self._mp(bucket, key)
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        etag = meta.get("etag") or hashlib.md5(p.read_bytes()).hexdigest()
        return Head(p.stat().st_size, etag, dict(meta.get("metadata") or {}), meta.get("content_type"), meta.get("cache_control"))

    def read_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        self._log("read_range", bucket, key)
        with open(self._p(bucket, key), "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)

    def get_bytes(self, bucket: str, key: str) -> bytes:
        self._log("get", bucket, key)
        return self._p(bucket, key).read_bytes()

    def download(self, bucket: str, key: str, path: Path, *, if_match: Optional[str] = None, size: Optional[int] = None,
                 **_: Any) -> None:
        self._log("download", bucket, key)
        h = self.head(bucket, key)
        if h is None:
            raise FileNotFoundError(f"{bucket}/{key}")
        if if_match and h.etag != if_match:
            raise LegacyChanged(f"{key} changed while it was being read")
        shutil.copyfile(self._p(bucket, key), path)

    def list_keys(self, bucket: str, prefix: str) -> Iterator[Tuple[str, int]]:
        base = self.root / bucket
        if not base.is_dir():
            return
        for p in sorted(base.rglob("*")):
            if p.is_file():
                key = p.relative_to(base).as_posix()
                if key.startswith(prefix):
                    yield key, p.stat().st_size

    def _store(self, bucket: str, key: str, data_path: Optional[Path], data: Optional[bytes], etag: str, content_type: str,
               cache_control: str, metadata: Optional[Dict[str, str]]) -> None:
        dst = self._p(bucket, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        if data_path is not None:
            shutil.copyfile(data_path, tmp)
        else:
            tmp.write_bytes(data or b"")
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
        dst.write_bytes(data)
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
            continue  # maintained by `refs-index`, a single writer
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
    for v in (manifest.get("packs") or {}).values():
        if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{32}", v):
            pack_keys.append(f"v2/ref/{v}.pack")
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
        return raw if raw is not None else json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    m = dict(manifest)
    lg = dict(m.get("legacy") or {})
    lg.update({"key": legacy["key"], "etag": legacy["etag"], "bytes": legacy["bytes"], "sha256": legacy["sha256"],
               "migrated_at": legacy["migrated_at"], "ledger": ledger_key(gse)})
    m["legacy"] = lg
    return json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ── catalog rows (D1: v2_study, v2_sample, v2_migration) ──────────────────────────────────────────────────────────────

# ASSUMED until W5 publishes its schema (bench/compression/notes/m_w5.md): names follow docs/format-v2.md section 8 and the
# ingest endpoint's conventions (gse_id / gsm_id, JSON stored as text). Change this table, nothing else, to follow W5.
# kind: text | int | real. `pk` is the ON CONFLICT target. `stamp` is set to the run time when present in the table.
CATALOG_SCHEMA: Dict[str, Dict[str, Any]] = {
    "v2_study": {
        "pk": ["gse_id"],
        "stamp": "updated_at",
        "cols": {
            "gse_id": "text", "manifest_key": "text", "rev": "text", "manifest_bytes": "int", "manifest_sha256": "text",
            "bytes": "int", "n_samples": "int", "n_cells_called": "int", "reference_build": "text",
            "manifest_version": "int", "packs": "text", "legacy_etag": "text", "legacy_bytes": "int", "created_at": "text",
        },
    },
    "v2_sample": {
        "pk": ["gsm_id"],
        "stamp": "updated_at",
        "cols": {
            "gsm_id": "text", "gse_id": "text", "key": "text", "bytes": "int", "sha256": "text", "n_columns": "int",
            "n_called": "int", "n_members": "int", "members": "text", "stats": "text",
        },
    },
    "v2_migration": {
        "pk": ["gse_id"],
        "stamp": "updated_at",
        "cols": {
            "gse_id": "text", "status": "text", "legacy_key": "text", "legacy_etag": "text", "legacy_bytes": "int",
            "legacy_sha256": "text", "v2_bytes": "int", "n_samples": "int", "members_by_kind": "text",
            "verify_ok": "int", "verify_secs": "real", "total_secs": "real", "commit_sha": "text", "migrated_at": "text",
            "ledger_key": "text", "error": "text",
        },
    },
}
MAX_TEXT = 60000  # the ingest endpoint refuses strings over 64 KB


def _compact_members(members: dict, mode: str) -> Optional[str]:
    if mode == "none" or not members:
        return None
    if mode == "full":
        out: Any = members
    else:
        out = {}
        for name, mem in members.items():
            if isinstance(mem, dict):
                out[name] = [mem.get("kind"), mem.get("off"), mem.get("len")]
    s = json.dumps(out, sort_keys=True, separators=(",", ":"))
    return s if len(s) <= MAX_TEXT else None


def build_catalog_rows(manifest: Optional[dict], row: dict, members_mode: str = "compact") -> Dict[str, Any]:
    """{table: row or [rows]} for one study, from the manifest (when there is one) and the result row of the run."""
    out: Dict[str, Any] = {}
    gse = row["gse"]
    mig = {
        "gse_id": gse, "status": row.get("status"), "legacy_key": row.get("legacy_key"), "legacy_etag": row.get("legacy_etag"),
        "legacy_bytes": row.get("legacy_bytes"), "legacy_sha256": row.get("legacy_sha256"), "v2_bytes": row.get("v2_bytes"),
        "n_samples": row.get("n_samples"), "members_by_kind": json.dumps(row.get("members_by_kind") or {}, sort_keys=True),
        "verify_ok": 1 if (row.get("verify") or {}).get("ok") else 0, "verify_secs": (row.get("verify") or {}).get("secs"),
        "total_secs": (row.get("timings") or {}).get("total"), "commit_sha": row.get("commit"), "migrated_at": row.get("at"),
        "ledger_key": ledger_key(gse), "error": (row.get("error") or "")[:2000] or None,
    }
    out["v2_migration"] = mig
    if manifest is None:
        return out
    sm = summarize_manifest(manifest)
    mbytes = row.get("manifest_bytes")
    out["v2_study"] = {
        "gse_id": gse, "manifest_key": manifest_key(gse), "rev": manifest.get("rev"), "manifest_bytes": mbytes,
        "manifest_sha256": row.get("manifest_sha256"), "bytes": row.get("v2_bytes"), "n_samples": sm["n_samples"],
        "n_cells_called": sm["n_called"], "reference_build": manifest.get("reference_build"),
        "manifest_version": manifest.get("manifest_version"), "packs": json.dumps(manifest.get("packs") or {}, sort_keys=True),
        "legacy_etag": (manifest.get("legacy") or {}).get("etag"), "legacy_bytes": (manifest.get("legacy") or {}).get("bytes"),
        "created_at": manifest.get("created_at"),
    }
    samples = []
    for s in manifest.get("samples", []):
        stats = s.get("stats")
        samples.append({
            "gsm_id": s.get("gsm"), "gse_id": gse, "key": s.get("key"), "bytes": s.get("bytes"), "sha256": s.get("sha256"),
            "n_columns": s.get("n_columns"), "n_called": s.get("n_called"), "n_members": len(s.get("members") or {}),
            "members": _compact_members(s.get("members") or {}, members_mode),
            "stats": json.dumps(stats, sort_keys=True, separators=(",", ":")) if stats else None,
        })
    out["v2_sample"] = samples
    return out


def clean_row(table: str, rec: dict) -> dict:
    """Keep only the schema's columns, with the types the ingest endpoint enforces (ints are ints, text is text)."""
    cols = CATALOG_SCHEMA[table]["cols"]
    out = {}
    for c, kind in cols.items():
        v = rec.get(c)
        if v is None:
            out[c] = None
        elif kind == "int":
            out[c] = int(v)
        elif kind == "real":
            out[c] = float(v)
        else:
            out[c] = v if isinstance(v, str) else json.dumps(v, sort_keys=True)
    return out


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


def upsert_sql(table: str, rec: dict, stamp_value: Optional[str], stamp: bool = True) -> str:
    spec = CATALOG_SCHEMA[table]
    cols = list(rec.keys())
    vals = [sql_literal(rec[c]) for c in cols]
    if stamp and spec.get("stamp"):
        cols.append(spec["stamp"])
        vals.append(sql_literal(stamp_value))
    key = spec["pk"]
    upd = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in key)
    return (f'INSERT INTO "{table}" ({", ".join(chr(34) + c + chr(34) for c in cols)}) VALUES ({", ".join(vals)}) '
            f'ON CONFLICT({", ".join(chr(34) + k + chr(34) for k in key)}) DO UPDATE SET {upd};')


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


def catalog_batches(studies: Dict[str, dict], max_rows: int, max_bytes: int, stamp_value: str, stamp: bool = True
                    ) -> List[Tuple[str, str, int]]:
    """[(file name, SQL text, n statements)] in apply order: samples, migration rows, then study rows (the study row is
    what makes a study appear, so it goes last)."""
    order = [("v2_sample", 1), ("v2_migration", 5), ("v2_study", 9)]
    files = []
    for table, rank in order:
        stmts = []
        for gse in accession_order(studies):
            v = studies[gse].get(table)
            if v is None:
                continue
            for rec in (v if isinstance(v, list) else [v]):
                stmts.append(upsert_sql(table, clean_row(table, rec), stamp_value, stamp))
        buf: List[str] = []
        size = 0
        n = 0
        for s in stmts:
            if buf and (len(buf) >= max_rows or size + len(s) > max_bytes):
                files.append((f"{rank}{n:04d}_{table}.sql", "\n".join(buf) + "\n", len(buf)))
                n += 1
                buf, size = [], 0
            buf.append(s)
            size += len(s) + 1
        if buf:
            files.append((f"{rank}{n:04d}_{table}.sql", "\n".join(buf) + "\n", len(buf)))
    return files


def post_rows(base_url: str, token: str, table: str, rows: List[dict], timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/ingest/{table}",
        data=json.dumps({"rows": rows}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Ingest-Token": token,
                 "User-Agent": "singlet-v2-campaign/1.0"},  # the site answers 403 to the default Python user agent
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def post_study_rows(base_url: str, token: str, studies: Dict[str, dict], batch: int = 500, max_body: int = 3 * MIB,
                    log=print) -> Dict[str, int]:
    """POST rows in apply order (samples, migration, study). Retries 429/5xx; a 4xx stops (the body says why)."""
    counts: Dict[str, int] = {}
    for table in ("v2_sample", "v2_migration", "v2_study"):
        rows = []
        for gse in accession_order(studies):
            v = studies[gse].get(table)
            if v is None:
                continue
            rows.extend(clean_row(table, r) for r in (v if isinstance(v, list) else [v]))
        i = 0
        while i < len(rows):
            chunk = rows[i : i + batch]
            while len(json.dumps(chunk)) > max_body and len(chunk) > 1:
                chunk = chunk[: len(chunk) // 2]
            for attempt in range(6):
                try:
                    post_rows(base_url, token, table, chunk)
                    break
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", "replace")[:300]
                    if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                        time.sleep(min(60, 2 ** attempt * 2))
                        continue
                    raise RuntimeError(f"POST {table} -> HTTP {e.code}: {body}") from None
                except (urllib.error.URLError, socket.timeout):
                    if attempt == 5:
                        raise
                    time.sleep(min(60, 2 ** attempt * 2))
            i += len(chunk)
            counts[table] = counts.get(table, 0) + len(chunk)
            log(f"posted {table}: {counts[table]}/{len(rows)}")
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
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "SINGLET_V2_THREADS"):
            env[k] = str(a.migrate_threads)
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
    rc, tail, secs = run_logged(ctx.migrate_cmd + ["--bundle", str(bundle), "--out", str(out), "--gse", gse], cwd=work,
                                env=ctx.env, timeout=a.migrate_timeout, log_path=study_log)
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
    report = out / "migrate_report.json"
    if report.is_file() and report.stat().st_size < 200_000:
        try:
            row["report"] = json.loads(report.read_text("utf-8"))
        except ValueError:
            pass
    ev("collected", n_objects=row["n_objects"], v2_bytes=row["v2_bytes"], members_by_kind=sm["members_by_kind"])

    # 5. verify against the legacy bundle (must exit 0)
    row["stage"] = "verify"
    rc, tail, secs = run_logged(ctx.verify_cmd + ["--bundle", str(bundle), "--v2", str(out)], cwd=work, env=ctx.env,
                                timeout=a.verify_timeout, log_path=study_log)
    row["verify"] = {"ok": rc == 0, "rc": rc, "secs": round(secs, 2)}
    tm["verify"] = round(secs, 2)
    ev("verified", rc=rc, secs=round(secs, 2))
    if rc != 0:
        row["verify"]["tail"] = tail[-1500:]
        raise StudyFailure("verify_failed", "verify", f"verify exited {rc}")

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
            else:  # a failed attempt: record it, keep what an earlier run left for the study and its samples
                rows = build_catalog_rows(None, _public_row(row), a.members)
                old = d / f"{gse}.json"
                if old.exists():
                    rows = {**json.loads(old.read_text("utf-8")), **rows}
            tmp = d / f".{gse}.json.tmp"
            tmp.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
            os.replace(tmp, d / f"{gse}.json")
        except Exception as e:  # noqa: BLE001
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
    if not explicit or a.mode == "pilot":
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
          f"io_threads={a.io_threads} migrate_threads={a.migrate_threads} shard={a.shard} resume={a.resume} "
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
            eff: float = 0.75, startup_min: float = 6.0, max_study_bytes: float = 0.0) -> dict:
    """Per-task and campaign time. A node moves min(download rate, compute rate) MB/s of legacy bytes."""
    node_rate = min(rate_dl, rate_cpu_core * cores * eff)
    shard_bytes = total_bytes / shards
    task_h = shard_bytes / 1e6 / node_rate / 3600 + startup_min / 60
    return {"shards": shards, "studies_per_shard": n_studies / shards, "gb_per_shard": shard_bytes / 1e9, "node_mbs": node_rate,
            "task_hours": task_h, "node_hours_total": task_h * shards,
            "bound": "download" if rate_dl <= rate_cpu_core * cores * eff else "compute",
            "longest_study_hours": (max_study_bytes / 1e6 / (rate_cpu_core * max(1.0, cores / 2)) / 3600) if max_study_bytes else 0.0}


def cmd_plan(a: argparse.Namespace) -> int:
    if a.from_tsv:
        sized = [(ln.split("\t")[0], int(ln.split("\t")[1])) for ln in Path(a.from_tsv).read_text().splitlines()
                 if ln and not ln.startswith("gse")]
    else:
        sized = make_store(a).list_studies(a.public)
    sized.sort(key=lambda t: (len(t[0]), t[0]))
    total = sum(s for _, s in sized)
    big = [t for t in sized if t[1] >= a.big_gb * 1e9]
    print(f"studies={len(sized)} total={total / 1e12:.3f} TB mean={total / max(1, len(sized)) / 1e6:.1f} MB "
          f"max={max((s for _, s in sized), default=0) / 1e9:.1f} GB  >= {a.big_gb} GB: {len(big)}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("gse\tbytes\tshard\n")
            for i, (g, s) in enumerate(sized):
                fh.write(f"{g}\t{s}\t{i % a.shards}\n")
        print(f"wrote {a.out} (shard = position in accession order mod {a.shards})")
    if a.big_out:
        Path(a.big_out).write_text("".join(f"{g}\t# {s / 1e9:.1f} GB\n" for g, s in big), encoding="utf-8")
    print("shards  GB/shard  studies/shard  node MB/s  bound     task h  node-h total")
    for n in sorted({8, 16, 32, 64, 128, a.shards}):
        p = project(total, len(sized), n, rate_dl=a.rate_dl, rate_cpu_core=a.rate_cpu, cores=a.cores,
                    max_study_bytes=max((s for _, s in sized), default=0))
        print(f"{n:6d}  {p['gb_per_shard']:8.1f}  {p['studies_per_shard']:13.0f}  {p['node_mbs']:9.0f}  {p['bound']:8s}  "
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


def cmd_catalog_sql(a: argparse.Namespace) -> int:
    if a.from_r2:
        store = make_store(a)
        studies: Dict[str, dict] = {}
        if a.gse_list:
            gses = read_list(a.gse_list)
        else:  # the manifests that exist: legacy keys disappear when the bundles are retired
            gses = [RE_MANIFEST.match(k).group(1) for k, _ in store.list_keys(a.public, "v2/m/") if RE_MANIFEST.match(k)]
        gses = accession_order(gses)
        for g in gses:
            if store.head(a.public, manifest_key(g)) is None:
                continue
            m = json.loads(store.get_bytes(a.public, manifest_key(g)))
            prior = last_committed(read_ledger(store, a.public, g)) or {"gse": g, "status": "ok"}
            md = digest_bytes(store.get_bytes(a.public, manifest_key(g)))
            prior = dict(prior, manifest_bytes=md.size, manifest_sha256=md.sha256)
            studies[g] = build_catalog_rows(m, prior, a.members)
    else:
        studies = load_rows([Path(p) for p in a.rows_dir])
    if a.gse_list and not a.from_r2:
        keep = set(read_list(a.gse_list))
        studies = {g: v for g, v in studies.items() if g in keep}
    stamp_value = iso()
    files = catalog_batches(studies, a.max_rows, a.max_bytes, stamp_value, stamp=not a.no_stamp)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("[0-9]*_v2_*.sql"):
        f.unlink()
    for name, text, _n in files:
        (out / name).write_text(text, encoding="utf-8")
    lines = [f"wrangler d1 execute {a.database} --remote --file={name}" for name, _t, _n in files]
    (out / "APPLY.txt").write_text(
        "# Run from the laptop, in this order (each file is atomic; the study rows come last).\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    print(f"{len(studies)} studies -> {len(files)} files, {sum(n for *_x, n in files)} statements in {out}")
    return 0


def cmd_catalog_post(a: argparse.Namespace) -> int:
    token = os.environ.get(a.token_env, "")
    studies = load_rows([Path(p) for p in a.rows_dir])
    if a.gse_list:
        keep = set(read_list(a.gse_list))
        studies = {g: v for g, v in studies.items() if g in keep}
    if a.dry_run:
        n = {t: sum(len(v[t]) if isinstance(v.get(t), list) else (1 if v.get(t) else 0) for v in studies.values())
             for t in ("v2_sample", "v2_migration", "v2_study")}
        print(f"would POST to {a.base_url}/api/ingest/<table>: {n}")
        return 0
    if not token:
        print(f"error: ${a.token_env} is empty (the ingest token)", file=sys.stderr)
        return 2
    counts = post_study_rows(a.base_url, token, studies, batch=a.batch)
    print("posted", counts)
    return 0


def cmd_refs_index(a: argparse.Namespace) -> int:
    store = make_store(a)
    packs = []
    for key, size in store.list_keys(a.public, "v2/ref/"):
        m = RE_PACK.match(key)
        if not m:
            continue
        head = store.read_range(a.public, key, 0, min(size, 65536) - 1)
        info: Dict[str, Any] = {"id": m.group(1), "bytes": size}
        if head[:4] == b"SPK2" and len(head) >= 12:
            kind = int.from_bytes(head[6:8], "little")
            hlen = int.from_bytes(head[8:12], "little")
            info["kind"] = {1: "vocab", 2: "whitelist", 3: "composite"}.get(kind, kind)
            try:
                hdr = json.loads(head[12 : 12 + hlen].decode("utf-8"))
                for k in ("build", "member", "n", "protocol", "sha256_names"):
                    if k in hdr:
                        info[k] = hdr[k]
            except ValueError:
                pass
        info["sha256"] = store.sha256_object(a.public, key, size)
        if info["sha256"][:32] != m.group(1):
            print(f"warning: {key} is not named by its digest16 ({info['sha256'][:32]})", file=sys.stderr)
        packs.append(info)
    packs.sort(key=lambda p: p["id"])
    body = json.dumps({"format": "singlet-v2", "updated_at": iso(), "packs": packs}, sort_keys=True, indent=1).encode("utf-8")
    if a.apply:
        ct, cc = object_headers(REFS_INDEX_KEY)
        GuardedStore(store, {a.public: ("v2/ref/",)}).put_bytes(a.public, REFS_INDEX_KEY, body, content_type=ct, cache_control=cc)
    print(f"{len(packs)} packs; index {'written' if a.apply else 'not written (no --apply)'} ({len(body)} bytes)")
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
    r.add_argument("--workers", type=int, default=0, help="studies in flight (0 = cores / migrate-threads)")
    r.add_argument("--io-threads", type=int, default=8)
    r.add_argument("--migrate-threads", type=int, default=2)
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
    r.add_argument("--members", choices=["compact", "full", "none"], default="compact", help="member offsets kept in v2_sample rows")
    r.add_argument("--deadline-epoch", type=float, help="do not start a study that cannot finish before this time (unix seconds)")
    r.add_argument("--min-rate-mbs", type=float, default=15.0, help="assumed worst-case MB/s of legacy bytes, for --deadline-epoch")
    r.add_argument("--commit")
    r.add_argument("--task")
    r.add_argument("--part-size-mib", type=float, default=PART_SIZE / MIB, help=argparse.SUPPRESS)
    r.add_argument("--retry-sleep", type=float, default=1.0, help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("plan", help="list studies, shards and a time projection")
    common(p)
    p.add_argument("--from-tsv", help="a listing (gse<TAB>bytes) instead of listing the bucket")
    p.add_argument("--out", help="write gse/bytes/shard here")
    p.add_argument("--shards", type=int, default=32)
    p.add_argument("--rate-dl", type=float, default=150.0, help="MB/s per node from R2")
    p.add_argument("--rate-cpu", type=float, default=5.0, help="MB/s of legacy bytes per core for migrate + verify")
    p.add_argument("--cores", type=float, default=8.0)
    p.add_argument("--big-gb", type=float, default=8.0)
    p.add_argument("--big-out")
    p.set_defaults(func=cmd_plan)

    s = sub.add_parser("summarize", help="aggregate result files")
    s.add_argument("results", nargs="+", help="results jsonl files or directories of them")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--atlas-tb", type=float, default=2.7)
    s.add_argument("--show", type=int, default=20)
    s.set_defaults(func=cmd_summarize)

    c = sub.add_parser("catalog-sql", help="SQL batch files for wrangler d1 execute")
    common(c)
    c.add_argument("--rows-dir", nargs="*", default=[], help="rows directories written by `run --rows-dir`")
    c.add_argument("--from-r2", action="store_true", help="rebuild rows from v2/m manifests and ledgers in R2 instead")
    c.add_argument("--gse-list")
    c.add_argument("--out", required=True)
    c.add_argument("--database", default="singlet-catalog")
    c.add_argument("--max-rows", type=int, default=1000)
    c.add_argument("--max-bytes", type=int, default=1_500_000)
    c.add_argument("--members", choices=["compact", "full", "none"], default="compact")
    c.add_argument("--no-stamp", action="store_true", help="do not set the updated_at column")
    c.set_defaults(func=cmd_catalog_sql)

    q = sub.add_parser("catalog-post", help="POST rows to the ingest endpoint")
    q.add_argument("--rows-dir", nargs="+", required=True)
    q.add_argument("--gse-list")
    q.add_argument("--base-url", default="https://singlet.bio")
    q.add_argument("--token-env", default="INGEST_TOKEN")
    q.add_argument("--batch", type=int, default=500)
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_catalog_post)

    x = sub.add_parser("refs-index", help="rebuild v2/ref/index.json from the packs")
    common(x)
    x.add_argument("--apply", action="store_true")
    x.set_defaults(func=cmd_refs_index)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.workers <= 0:
            a.workers = max(1, _cpu_count() // max(1, a.migrate_threads))
        if a.mode in ("dry", "full") and not (a.gse or a.gse_list or a.shard or a.limit is not None or os.environ.get("V2_ALL")):
            ap.error(f"{a.mode} mode needs --shard I/N, --gse, --gse-list or --limit (V2_ALL=1 runs every study in one process)")
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
