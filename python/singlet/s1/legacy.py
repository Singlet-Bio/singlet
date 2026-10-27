# SPDX-License-Identifier: MIT
"""Read a legacy ``.singlet`` bundle (ZIP64) from a local path, an HTTPS URL (range requests), bytes, or an unpacked directory.

``LegacyBundle.open(src)``:

* ``src`` a file path or ``https://`` URL: the central directory is parsed (ZIP and ZIP64) from the tail of the archive; members are
  fetched with one range request each and verified against the directory CRC-32. ``data.singlet.bio`` answers 403 to the default
  urllib User-Agent, so every request carries ``User-Agent: singlet/1.0``.
* ``src`` a directory: either an unpacked bundle (``samples/<GSM>/...``) or a *loose sample directory* as used by the study corpora
  (members side by side, ``star_Solo.out__Gene__raw__matrix.mtx`` for ``star_Solo.out/Gene/raw/matrix.mtx``, a ``sample.json`` with
  ``gse``/``gsm``).

Entries are ordered as in the archive. ``entry.crc32`` / ``entry.size`` are the legacy values the ledger records.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Optional

USER_AGENT = "singlet/1.0 (+https://singlet.bio)"
TAIL = 262144

_STUDY_FILES = ("manifest.json", "study_meta.json", "feature_vocab.json")
# member directories of a sample that the study corpora flatten into one directory with "__" for "/"
_FLAT_DIRS = ("star_Solo.out", "donor", "mt", "nonhost")


def _unflatten(name: str) -> str:
    """``star_Solo.out__Gene__raw__matrix.mtx`` -> ``star_Solo.out/Gene/raw/matrix.mtx``, ``donor__donor_assignments.tsv`` ->
    ``donor/donor_assignments.tsv`` (also ``mt__``, ``nonhost__``); any other name is returned unchanged."""
    for d in _FLAT_DIRS:
        if name.startswith(d + "__"):
            return name.replace("__", "/")
    return name


class LegacyError(RuntimeError):
    pass


@dataclass
class LegacyEntry:
    path: str
    size: int  # uncompressed bytes
    csize: int  # compressed bytes in the archive
    method: int  # 0 stored, 8 deflate
    crc32: int
    header_off: int  # local header offset in the archive (-1 for directory sources)
    name_len: int = 0


# ----------------------------------------------------------------------------------------------------------- sources
class _FileSource:
    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        self.etag = None
        self.last_modified = None
        self.n_requests = 0
        self.bytes = 0

    def read(self, off: int, length: int) -> bytes:
        with open(self.path, "rb") as fh:
            fh.seek(off)
            b = fh.read(length)
        self.n_requests += 1
        self.bytes += len(b)
        return b

    def read_tail(self, length: int) -> bytes:
        return self.read(max(0, self.size - length), length)


class _BytesSource:
    def __init__(self, data: bytes):
        self.data = data
        self.size = len(data)
        self.etag = None
        self.last_modified = None
        self.n_requests = 0
        self.bytes = 0

    def read(self, off: int, length: int) -> bytes:
        self.n_requests += 1
        self.bytes += min(length, max(0, self.size - off))
        return self.data[off : off + length]

    def read_tail(self, length: int) -> bytes:
        return self.read(max(0, self.size - length), length)


class _HttpSource:
    def __init__(self, url: str, retries: int = 5, timeout: float = 60.0):
        self.url = url
        self.retries = retries
        self.timeout = timeout
        self.size: Optional[int] = None
        self.etag: Optional[str] = None
        self.last_modified: Optional[str] = None
        self.n_requests = 0
        self.bytes = 0

    def _get(self, rng: str) -> tuple[bytes, dict]:
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self.url, headers={"Range": rng, "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read()
                    hdr = {k.lower(): v for k, v in r.getheaders()}
                    if r.status not in (200, 206):
                        raise LegacyError("HTTP %d for %s" % (r.status, self.url))
                    self.n_requests += 1
                    self.bytes += len(body)
                    return body, hdr
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (403, 404, 416) and attempt >= 1:
                    break
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
            ) as e:
                last = e  # HTTPException covers IncompleteRead: a response cut short by the network is retried
            time.sleep(min(2**attempt, 20))
        raise LegacyError("cannot read %s (%s): %s" % (self.url, rng, last))

    def _learn(self, hdr: dict):
        cr = hdr.get("content-range", "")
        m = re.match(r"bytes\s+\d+-\d+/(\d+)", cr)
        if m:
            self.size = int(m.group(1))
        elif self.size is None and "content-length" in hdr and "content-range" not in hdr:
            self.size = int(hdr["content-length"])
        if hdr.get("etag"):
            self.etag = hdr["etag"].strip('"')
        if hdr.get("last-modified"):
            self.last_modified = hdr["last-modified"]

    def read(self, off: int, length: int) -> bytes:
        body, hdr = self._get("bytes=%d-%d" % (off, off + length - 1))
        self._learn(hdr)
        return body

    def read_tail(self, length: int) -> bytes:
        body, hdr = self._get("bytes=-%d" % length)
        self._learn(hdr)
        return body


# ----------------------------------------------------------------------------------------------------------- zip parsing
_EOCD = b"PK\x05\x06"
_EOCD64 = b"PK\x06\x06"
_LOC64 = b"PK\x06\x07"
_CEN = b"PK\x01\x02"


def parse_central_directory(
    tail: bytes, tail_start: int, total: int, read_range
) -> list[LegacyEntry]:
    """Entries from the archive tail (``tail`` begins at absolute offset ``tail_start``); ``read_range(off, len)`` is used when the
    central directory begins before the tail."""
    i = tail.rfind(_EOCD)
    if i < 0:
        raise LegacyError("no end-of-central-directory record (not a ZIP?)")
    (_s, disk, cdisk, n_disk, n_total, cd_size, cd_off, clen) = struct.unpack_from(
        "<4sHHHHIIH", tail, i
    )
    if n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        j = tail.rfind(_LOC64, 0, i)
        if j < 0:
            raise LegacyError("ZIP64 locator missing")
        (_s2, _d, off64, _nd) = struct.unpack_from("<4sIQI", tail, j)
        rec = (
            tail[off64 - tail_start : off64 - tail_start + 56]
            if off64 >= tail_start
            else read_range(off64, 56)
        )
        (sig, _sz, _vm, _vn, _dn, _dcd, n_disk64, n_total, cd_size, cd_off) = struct.unpack_from(
            "<4sQHHIIQQQQ", rec, 0
        )
        if sig != _EOCD64:
            raise LegacyError("bad ZIP64 end record")
    if cd_off >= tail_start:
        cd = tail[cd_off - tail_start : cd_off - tail_start + cd_size]
    else:
        cd = read_range(cd_off, cd_size)
    if len(cd) != cd_size:
        raise LegacyError("short central directory read")
    out = []
    p = 0
    for _ in range(n_total):
        if cd[p : p + 4] != _CEN:
            raise LegacyError("bad central directory entry signature")
        (
            _sig,
            _vm,
            _vn,
            flags,
            method,
            _t,
            _d,
            crc,
            csize,
            size,
            nl,
            el,
            cl,
            _dn,
            _ia,
            _ea,
            hoff,
        ) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd, p)
        name = cd[p + 46 : p + 46 + nl].decode("utf-8" if flags & 0x800 else "cp437")
        extra = cd[p + 46 + nl : p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack_from("<HH", extra, q)
            if hid == 1:
                z = q + 4
                if size == 0xFFFFFFFF:
                    (size,) = struct.unpack_from("<Q", extra, z)
                    z += 8
                if csize == 0xFFFFFFFF:
                    (csize,) = struct.unpack_from("<Q", extra, z)
                    z += 8
                if hoff == 0xFFFFFFFF:
                    (hoff,) = struct.unpack_from("<Q", extra, z)
                    z += 8
            q += 4 + hsz
        out.append(LegacyEntry(name, size, csize, method, crc, hoff, nl))
        p += 46 + nl + el + cl
    return out


def entry_bytes(src_read, e: LegacyEntry) -> bytes:
    """Fetch, inflate and CRC-check one archive member."""
    want = 30 + e.name_len + 64 + e.csize
    buf = src_read(e.header_off, want)
    if buf[:4] != b"PK\x03\x04":
        raise LegacyError("bad local header for %s" % e.path)
    (nl, el) = struct.unpack_from("<HH", buf, 26)
    start = 30 + nl + el
    if start + e.csize > len(buf):
        buf = buf + src_read(e.header_off + len(buf), start + e.csize - len(buf))
    body = buf[start : start + e.csize]
    if len(body) != e.csize:
        raise LegacyError("short read of %s" % e.path)
    if e.method == 0:
        data = body
    elif e.method == 8:
        data = zlib.decompressobj(-15).decompress(body)
    else:
        raise LegacyError("unsupported ZIP method %d for %s" % (e.method, e.path))
    if len(data) != e.size or (zlib.crc32(data) & 0xFFFFFFFF) != e.crc32:
        raise LegacyError("CRC/size mismatch reading %s" % e.path)
    return data


# ----------------------------------------------------------------------------------------------------------- bundle
class LegacyBundle:
    """A legacy bundle. ``entries``: path -> :class:`LegacyEntry` in archive order."""

    def __init__(self):
        self.entries: dict[str, LegacyEntry] = {}
        self.src = None
        self.kind = "zip"
        self.location = ""
        self.etag: Optional[str] = None
        self.last_modified: Optional[str] = None
        self.size: Optional[int] = None
        self._dir: Optional[str] = None
        self._dirmap: dict[str, str] = {}
        self._cache: dict[str, bytes] = {}
        self.gse_hint: Optional[str] = None
        self._sample_hint: Optional[str] = None

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, src, tail: int = TAIL) -> "LegacyBundle":
        b = cls()
        if isinstance(src, (bytes, bytearray)):
            b.src = _BytesSource(bytes(src))
        elif isinstance(src, str) and re.match(r"https?://", src):
            b.src = _HttpSource(src)
            b.location = src
        elif os.path.isdir(os.fspath(src)):
            b._open_dir(os.fspath(src))
            return b
        else:
            p = os.fspath(src)
            b.src = _FileSource(p)
            b.location = os.path.abspath(p)
        t = b.src.read_tail(tail)
        size = b.src.size if b.src.size is not None else len(t)
        b.size = size
        b.etag, b.last_modified = b.src.etag, b.src.last_modified
        ents = parse_central_directory(t, size - len(t), size, b.src.read)
        b.entries = {e.path: e for e in ents}
        m = re.search(r"(GSE\d+)", os.path.basename(b.location)) if b.location else None
        b.gse_hint = m.group(1) if m else None
        return b

    def _open_dir(self, d: str):
        self.kind = "dir"
        self._dir = d
        self.location = os.path.abspath(d)
        sj = os.path.join(d, "sample.json")
        info = json.load(open(sj)) if os.path.exists(sj) else {}
        names = os.listdir(d)
        ents = []
        if os.path.isdir(os.path.join(d, "samples")):
            for root, _dirs, files in sorted(os.walk(d)):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, d).replace(os.sep, "/")
                    ents.append((rel, full))
        else:
            gsm = info.get("gsm")
            if not gsm:
                m = re.search(r"(GSM\d+)", os.path.basename(os.path.normpath(d)))
                gsm = m.group(1) if m else None
            if not gsm:
                raise LegacyError(
                    "cannot tell which sample directory %s holds (no sample.json, no GSM in its name)"
                    % d
                )
            self._sample_hint = gsm
            for f in sorted(names):
                full = os.path.join(d, f)
                if not os.path.isfile(full) or f == "sample.json":
                    continue
                if f in _STUDY_FILES:
                    rel = f
                else:
                    rel = "samples/%s/%s" % (gsm, _unflatten(f))
                ents.append((rel, full))
        for rel, full in ents:
            data = open(full, "rb").read()
            self._dirmap[rel] = full
            self.entries[rel] = LegacyEntry(
                rel, len(data), len(data), 0, zlib.crc32(data) & 0xFFFFFFFF, -1
            )
        self.gse_hint = info.get("gse")
        m = re.search(r"(GSE\d+)", os.path.basename(os.path.normpath(d)))
        if not self.gse_hint and m:
            self.gse_hint = m.group(1)
        self.size = sum(e.size for e in self.entries.values())

    # ------------------------------------------------------------------ access
    def read(self, path: str) -> bytes:
        e = self.entries[path]
        if self.kind == "dir":
            return open(self._dirmap[path], "rb").read()
        return entry_bytes(self.src.read, e)

    def exists(self, path: str) -> bool:
        return path in self.entries

    def study_paths(self) -> list:
        return [p for p in self.entries if not p.startswith("samples/")]

    def samples(self) -> list:
        seen = []
        for p in self.entries:
            if p.startswith("samples/"):
                g = p.split("/", 2)[1]
                if g not in seen:
                    seen.append(g)
        return seen

    def sample_paths(self, gsm: str) -> list:
        """(relative member name, full path) pairs of one sample, archive order."""
        pre = "samples/%s/" % gsm
        return [
            (p[len(pre) :], p) for p in self.entries if p.startswith(pre) and not p.endswith("/")
        ]

    def stats(self) -> dict:
        return {
            "requests": self.src.n_requests if self.src is not None else 0,
            "bytes": self.src.bytes if self.src is not None else 0,
        }


# ----------------------------------------------------------------------------------------------------------- legacy .1pz
def decode_1pz_bytes(data: bytes):
    """Decode the bytes of a legacy ``.1pz`` member with the compiled reader (``singlet._pz.read_1pz``, path based, so via a
    temporary file) into a :class:`~singlet.s1.matrix.Csc`. Raises on corrupt or unsupported files."""
    import tempfile

    from .matrix import Csc

    try:
        from singlet import _pz
    except ImportError as e:  # pragma: no cover
        raise LegacyError("the compiled legacy reader singlet._pz is not importable") from e
    fd, path = tempfile.mkstemp(suffix=".1pz")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        r = _pz.read_1pz(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return Csc.from_legacy(r)
