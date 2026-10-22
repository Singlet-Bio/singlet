# SPDX-License-Identifier: MIT
"""Shared helpers for the v2 campaign tests: script loading, synthetic legacy bundles in a directory-backed R2, a stand-in
for `python -m singlet.v2.migrate|verify`, and an in-memory fake of the boto3 S3 client.

No compiled codec, no network, no credentials. The stand-in transcoder writes real `SINGLET2` headers (the campaign checks
the header, the digest in the key name and the manifest) so the campaign's own validation is exercised, not bypassed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str):
    """Import scripts/<name>.py by path (they are not a package)."""
    key = f"_v2s_{name}"
    if key in sys.modules:
        return sys.modules[key]
    sys.path.insert(0, str(SCRIPTS))  # v2_retire / v2_publish import v2_campaign next to them
    try:
        spec = importlib.util.spec_from_file_location(key, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        if name == "v2_campaign":
            sys.modules["v2_campaign"] = mod  # v2_retire / v2_publish `import v2_campaign`: share this module object
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPTS))
    return mod


# ── a stand-in transcoder/verifier ────────────────────────────────────────────────────────────────────────────────────

FAKE_TOOL = r'''
import hashlib, json, os, struct, sys, zipfile, zlib
from pathlib import Path

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

def count(what, gse):
    f = os.environ.get("FAKE_COUNT_FILE")
    if f:
        with open(f, "a") as fh:
            fh.write(f"{what} {gse}\n")

def s2_object(gsm, body):
    n = 64 + len(body)
    hdr = b"SINGLET2" + struct.pack("<HHIQQQ", 1, 0, 1, 64, 0, n) + gsm.encode().ljust(16, b"\0") + b"\0" * 8
    assert len(hdr) == 64
    return hdr + body

def migrate():
    bundle, out, gse = arg("--bundle"), Path(arg("--out")), arg("--gse")
    count("migrate", gse)
    if gse in os.environ.get("FAKE_MIGRATE_FAIL", "").split(","):
        print("fake migrate: refusing", gse, file=sys.stderr)
        sys.exit(3)
    z = zipfile.ZipFile(bundle)
    gsms = sorted({n.split("/")[1] for n in z.namelist() if n.startswith("samples/")})
    root = out / "v2"
    samples = []
    pad = int(os.environ.get("FAKE_PAD", "0"))
    for gsm in gsms:
        members = sorted(n for n in z.namelist() if n.startswith(f"samples/{gsm}/"))
        body = zlib.compress(b"".join(z.read(m) for m in members)) + os.urandom(0) + b"P" * pad
        obj = s2_object(gsm, body)
        h = hashlib.sha256(obj).hexdigest()
        p = root / "s" / gsm / f"{h[:32]}.s2"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(obj)
        mem = {m.split("/", 2)[2]: {"kind": 3, "off": 64, "len": 10} for m in members}
        mem["exon_counts.1pz"] = {"kind": 1, "off": 128, "len": 99, "digest": "00" * 16}
        mem["cells"] = {"kind": 2, "off": 256, "len": 40}
        mem["splice_psi.1pz"] = {"kind": 4, "off": 0, "len": 0}
        samples.append({"gsm": gsm, "key": f"v2/s/{gsm}/{h[:32]}.s2", "bytes": len(obj), "sha256": h, "n_columns": 10,
                        "n_called": 3, "members": mem, "stats": {"median_umi": 1500, "median_genes": 700}})
    pack = b"SPK2" + struct.pack("<HHI", 1, 1, 2) + b"{}" + (gse.encode() * 3)
    ph = hashlib.sha256(pack).hexdigest()[:32]
    pp = root / "ref" / f"{ph}.pack"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_bytes(pack)
    if os.environ.get("FAKE_BAD_NAME"):
        (root / "s" / gsms[0]).joinpath("0" * 32 + ".s2").write_bytes(b"x")
    man = {"format": "singlet-v2", "manifest_version": 1, "gse": gse, "rev": hashlib.sha256(gse.encode()).hexdigest()[:16],
           "created_at": "2026-10-22T00:00:00Z", "reference_build": "GRCh38-2024-A", "packs": {"exon": ph, "intron": ph, "whitelist": None},
           "study_files": {}, "samples": samples}
    (root / "m").mkdir(parents=True, exist_ok=True)
    (root / "m" / f"{gse}.json").write_text(json.dumps(man))
    (out / "migrate_report.json").write_text(json.dumps({"fallbacks": 1}))

def verify():
    bundle, out = arg("--bundle"), Path(arg("--v2"))
    gse = Path(bundle).stem
    count("verify", gse)
    if gse in os.environ.get("FAKE_VERIFY_FAIL", "").split(","):
        print("fake verify: MISMATCH", file=sys.stderr)
        sys.exit(1)
    man = json.loads((out / "v2" / "m" / f"{gse}.json").read_text())
    assert man["gse"] == gse
    # the campaign verifies the manifest it will publish, so the legacy facts must already be in it
    if os.environ.get("FAKE_EXPECT_LEGACY") and not man.get("legacy", {}).get("etag"):
        print("fake verify: manifest has no legacy block", file=sys.stderr)
        sys.exit(1)

{"migrate": migrate, "verify": verify}[sys.argv[1]]()
'''


def write_fake_tool(dirpath: Path) -> Path:
    p = Path(dirpath) / "fake_v2_tool.py"
    p.write_text(FAKE_TOOL, encoding="utf-8")
    return p


def tool_cmds(tool: Path) -> List[str]:
    """['--migrate-cmd', json, '--verify-cmd', json] for the campaign CLI."""
    return ["--migrate-cmd", json.dumps([sys.executable, str(tool), "migrate"]),
            "--verify-cmd", json.dumps([sys.executable, str(tool), "verify"])]


# ── synthetic legacy bundles ──────────────────────────────────────────────────────────────────────────────────────────


def make_bundle(path: Path, gse: str, gsms: List[str], *, restricted: bool = False, salt: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", json.dumps({"gse_id": gse, "gsm_ids": gsms, "n_gsms": len(gsms)}))
        z.writestr("study_meta.json", json.dumps({"gse_id": gse, "series_title": f"title of {gse} it's quoted"}))
        z.writestr("feature_vocab.json", "{}")
        for g in gsms:
            z.writestr(f"samples/{g}/summary.json", json.dumps({"n_cells_called": 3, "salt": salt}))
            z.writestr(f"samples/{g}/exon_counts.1pz", hashlib.sha256((g + salt).encode()).digest() * 40)
            if restricted:
                z.writestr(f"samples/{g}/donor/snp_ad.1pz", b"germline" * 20)
    return path


def put_legacy(root: Path, gse: str, gsms: List[str], **kw) -> Path:
    return make_bundle(Path(root) / "singlet-data" / "data" / gse / f"{gse}.singlet", gse, gsms, **kw)


def seed(root: Path, n: int = 3, restricted: Optional[str] = None) -> List[str]:
    """n studies GSE000001.. with 2 samples each in <root>/singlet-data."""
    gses = []
    for i in range(1, n + 1):
        gse = f"GSE{i:06d}"
        gsms = [f"GSM{i}0001", f"GSM{i}0002"]
        put_legacy(root, gse, gsms, restricted=(gse == restricted), salt=gse)
        gses.append(gse)
    return gses


# ── an in-memory boto3 S3 client (only what S3Store calls) ────────────────────────────────────────────────────────────


class FakeClientError(Exception):
    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class _Body:
    def __init__(self, data: bytes) -> None:
        self._d, self._p = data, 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._d) - self._p
        out = self._d[self._p : self._p + n]
        self._p += len(out)
        return out


class FakeS3:
    """Buckets in memory. ETags follow S3: MD5 for a single PUT, composite for multipart (upload_file with the given
    chunk size, or complete_multipart_upload over copied parts)."""

    def __init__(self, reject_storage_class: bool = False) -> None:
        self.objs: Dict[tuple, dict] = {}
        self.buckets = set()
        self.calls: List[tuple] = []
        self.uploads: Dict[str, dict] = {}
        self.reject_storage_class = reject_storage_class

    def _put(self, bucket, key, data, etag, **extra):
        self.objs[(bucket, key)] = {"data": data, "etag": etag, **extra}

    def put_raw(self, bucket, key, data: bytes, etag: Optional[str] = None, **kw):
        self.buckets.add(bucket)
        self._put(bucket, key, data, etag or hashlib.md5(data).hexdigest(), **kw)

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        o = self.objs.get((Bucket, Key))
        if o is None:
            raise FakeClientError("404", 404)
        return {"ContentLength": len(o["data"]), "ETag": f'"{o["etag"]}"', "Metadata": o.get("Metadata", {}),
                "ContentType": o.get("ContentType"), "CacheControl": o.get("CacheControl")}

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        self.calls.append(("get", Bucket, Key, Range))
        o = self.objs.get((Bucket, Key))
        if o is None:
            raise FakeClientError("NoSuchKey", 404)
        if IfMatch and IfMatch.strip('"') != o["etag"]:
            raise FakeClientError("PreconditionFailed", 412)
        d = o["data"]
        if Range:
            a, b = Range.split("=")[1].split("-")
            d = d[int(a) : int(b) + 1]
        return {"Body": _Body(d)}

    def put_object(self, Bucket, Key, Body, ContentType=None, CacheControl=None, Metadata=None):
        self.calls.append(("put_object", Bucket, Key))
        etag = hashlib.md5(Body).hexdigest()
        self._put(Bucket, Key, bytes(Body), etag, ContentType=ContentType, CacheControl=CacheControl, Metadata=Metadata or {})
        return {"ETag": f'"{etag}"'}

    def upload_file(self, path, Bucket, Key, ExtraArgs=None, Config=None):
        data = Path(path).read_bytes()
        chunk = getattr(Config, "multipart_chunksize", 8 << 20)
        thr = getattr(Config, "multipart_threshold", 8 << 20)
        self.calls.append(("upload_file", Bucket, Key, "multipart" if len(data) >= thr else "single"))
        if len(data) >= thr:
            parts = [hashlib.md5(data[i : i + chunk]).digest() for i in range(0, len(data), chunk)]
            etag = hashlib.md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"
        else:
            etag = hashlib.md5(data).hexdigest()
        ex = ExtraArgs or {}
        self._put(Bucket, Key, data, etag, ContentType=ex.get("ContentType"), CacheControl=ex.get("CacheControl"),
                  Metadata=ex.get("Metadata", {}))

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete", Bucket, Key))
        self.objs.pop((Bucket, Key), None)

    def head_bucket(self, Bucket):
        if Bucket not in self.buckets:
            raise FakeClientError("404", 404)

    def create_bucket(self, Bucket):
        self.calls.append(("create_bucket", Bucket))
        self.buckets.add(Bucket)

    def copy_object(self, Bucket, Key, CopySource, MetadataDirective=None, StorageClass=None):
        self.calls.append(("copy_object", Bucket, Key, StorageClass))
        if StorageClass and self.reject_storage_class:
            raise FakeClientError("InvalidStorageClass", 400)
        s = self.objs[(CopySource["Bucket"], CopySource["Key"])]
        self._put(Bucket, Key, s["data"], s["etag"], ContentType=s.get("ContentType"), CacheControl=s.get("CacheControl"),
                  Metadata=s.get("Metadata", {}), StorageClass=StorageClass)
        return {"CopyObjectResult": {"ETag": f'"{s["etag"]}"'}}

    def create_multipart_upload(self, Bucket, Key, **extra):
        self.calls.append(("create_mpu", Bucket, Key, extra.get("StorageClass")))
        if extra.get("StorageClass") and self.reject_storage_class:
            raise FakeClientError("InvalidStorageClass", 400)
        uid = f"u{len(self.uploads)}"
        self.uploads[uid] = {"bucket": Bucket, "key": Key, "parts": {}, "extra": extra}
        return {"UploadId": uid}

    def upload_part_copy(self, Bucket, Key, UploadId, PartNumber, CopySource, CopySourceRange):
        self.calls.append(("upload_part_copy", Bucket, Key, PartNumber))
        a, b = CopySourceRange.split("=")[1].split("-")
        d = self.objs[(CopySource["Bucket"], CopySource["Key"])]["data"][int(a) : int(b) + 1]
        etag = hashlib.md5(d).hexdigest()
        self.uploads[UploadId]["parts"][PartNumber] = (d, etag)
        return {"CopyPartResult": {"ETag": f'"{etag}"'}}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        self.calls.append(("complete_mpu", Bucket, Key))
        u = self.uploads.pop(UploadId)
        nums = [p["PartNumber"] for p in MultipartUpload["Parts"]]
        assert nums == sorted(nums)
        data = b"".join(u["parts"][n][0] for n in nums)
        etag = hashlib.md5(b"".join(bytes.fromhex(u["parts"][n][1]) for n in nums)).hexdigest() + f"-{len(nums)}"
        ex = u["extra"]
        self._put(Bucket, Key, data, etag, ContentType=ex.get("ContentType"), CacheControl=ex.get("CacheControl"),
                  Metadata=ex.get("Metadata", {}), StorageClass=ex.get("StorageClass"))
        return {"ETag": f'"{etag}"'}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.calls.append(("abort_mpu", Bucket, Key))
        self.uploads.pop(UploadId, None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class P:
            def paginate(self, Bucket, Prefix=""):
                keys = sorted(k for (b, k) in outer.objs if b == Bucket and k.startswith(Prefix))
                for i in range(0, max(1, len(keys)), 2):  # two keys per page, to exercise paging
                    yield {"Contents": [{"Key": k, "Size": len(outer.objs[(Bucket, k)]["data"])} for k in keys[i : i + 2]]}

        return P()


class TC:
    """Stand-in for boto3.s3.transfer.TransferConfig."""

    def __init__(self, part_size: int) -> None:
        self.multipart_threshold = part_size + 1
        self.multipart_chunksize = part_size
