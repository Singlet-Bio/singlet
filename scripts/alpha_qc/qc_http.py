# SPDX-License-Identifier: MIT
"""HTTPS helpers for the Research Alpha QC: plain GETs, range GETs with retries, streaming downloads with SHA-256, and a minimal
ZIP64 central-directory reader that fetches exactly one member's byte range at a time (so a read never overlaps a member that
the data guard withholds). Standard library only."""

import hashlib
import io
import os
import struct
import time
import urllib.error
import urllib.request
import zlib

BASE = os.environ.get("QC_DATA_BASE", "https://data.singlet.bio")
UA = "singlet-alpha-qc/1 (read-only audit)"


class HttpStatus(Exception):
    def __init__(self, code, url, body=b""):
        super().__init__(f"HTTP {code} for {url}")
        self.code, self.url, self.body = code, url, body


def _open(url, headers=None, method="GET", timeout=120):
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def request(url, headers=None, method="GET", retries=6, timeout=120):
    """-> (status, headers dict (lower-case keys), body bytes). 4xx other than 429 are returned, not retried."""
    last = None
    for i in range(retries):
        try:
            with _open(url, headers, method, timeout) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            body = e.read() if hasattr(e, "read") else b""
            if e.code in (429, 500, 502, 503, 504) and i + 1 < retries:
                last = e
                time.sleep(min(60, 2 ** i))
                continue
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, body
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            time.sleep(min(60, 2 ** i))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def get(path, **kw):
    return request(f"{BASE}/{path}", **kw)


def get_range(path, start, end_incl, **kw):
    st, h, b = request(f"{BASE}/{path}", headers={"Range": f"bytes={start}-{end_incl}"}, **kw)
    return st, h, b


def download(path, dest, retries=5, chunk=8 << 20):
    """Stream BASE/path to dest; -> (status, bytes written, sha256 hex, headers)."""
    url = f"{BASE}/{path}"
    for i in range(retries):
        try:
            with _open(url, timeout=600) as r:
                h = hashlib.sha256()
                n = 0
                with open(dest, "wb") as fh:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        h.update(b)
                        fh.write(b)
                        n += len(b)
                return r.status, n, h.hexdigest(), {k.lower(): v for k, v in r.headers.items()}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i + 1 < retries:
                time.sleep(min(60, 2 ** i))
                continue
            return e.code, 0, None, {}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if i + 1 == retries:
                raise
            time.sleep(min(60, 2 ** i))


# ---------------------------------------------------------------------------------------------------- ZIP64 over ranges
class RangeZip:
    """Central directory of a remote zip (legacy .singlet bundle) and exact-range member reads."""

    def __init__(self, path, size):
        self.path, self.size = path, int(size)
        self.entries = {}  # name -> dict(off, csize, usize, method, crc, flags)
        self._read_cd()

    def _tail(self, n):
        st, _h, b = get_range(self.path, self.size - n, self.size - 1)
        if st != 206:
            raise HttpStatus(st, self.path, b[:200])
        return b

    def _read_cd(self):
        tail = None
        for n in (66000, 4096, 98):
            n = min(n, self.size)
            try:
                tail = self._tail(n)
                break
            except HttpStatus:
                continue
        if tail is None:
            raise RuntimeError("cannot read the end of central directory")
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError("no EOCD")
        _sig, _d, _dcd, _ne, ntot, cd_size, cd_off, _cl = struct.unpack("<IHHHHIIH", tail[i : i + 22])
        j = tail.rfind(b"PK\x06\x07", 0, i)
        if j >= 0:  # zip64 locator -> zip64 EOCD
            (z64off,) = struct.unpack("<Q", tail[j + 8 : j + 16])
            st, _h, z = get_range(self.path, z64off, z64off + 55)
            if st != 206 or z[:4] != b"PK\x06\x06":
                raise RuntimeError("bad zip64 EOCD")
            ntot, cd_size, cd_off = struct.unpack("<QQQ", z[32:56])
        st, _h, cd = get_range(self.path, cd_off, cd_off + cd_size - 1)
        if st != 206 or len(cd) != cd_size:
            raise RuntimeError(f"central directory read failed: {st}")
        p = 0
        while p + 46 <= len(cd) and cd[p : p + 4] == b"PK\x01\x02":
            (flags, method, _t, _dt, crc, csz, usz, nl, el, cl, _dn, _ia, _ea, off) = struct.unpack(
                "<HHHHIIIHHHHHII", cd[p + 8 : p + 46]
            )
            name = cd[p + 46 : p + 46 + nl].decode("utf-8", "replace")
            ex = cd[p + 46 + nl : p + 46 + nl + el]
            q = 0
            while q + 4 <= len(ex):
                hid, hl = struct.unpack("<HH", ex[q : q + 4])
                d = ex[q + 4 : q + 4 + hl]
                if hid == 1:
                    vals = list(struct.unpack("<%dQ" % (hl // 8), d[: (hl // 8) * 8]))
                    if usz == 0xFFFFFFFF and vals:
                        usz = vals.pop(0)
                    if csz == 0xFFFFFFFF and vals:
                        csz = vals.pop(0)
                    if off == 0xFFFFFFFF and vals:
                        off = vals.pop(0)
                q += 4 + hl
            self.entries[name] = dict(off=off, csize=csz, usize=usz, method=method, crc=crc, flags=flags)
            p += 46 + nl + el + cl
        if len(self.entries) != ntot:
            raise RuntimeError(f"central directory lists {len(self.entries)} of {ntot} entries")

    def data_range(self, name):
        """(start, end_exclusive) of the member's compressed data (reads its local header: 30 + name bytes)."""
        e = self.entries[name]
        st, _h, lh = get_range(self.path, e["off"], e["off"] + 29)
        if st != 206:
            raise HttpStatus(st, self.path)
        if lh[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        nl, el = struct.unpack("<HH", lh[26:30])
        start = e["off"] + 30 + nl + el
        return start, start + e["csize"]

    def read(self, name, dest=None):
        """Member bytes (CRC-checked). With ``dest``, streams to that file and returns its path."""
        e = self.entries[name]
        a, b = self.data_range(name)
        if e["csize"] == 0:
            data = b""
        else:
            st, _h, data = get_range(self.path, a, b - 1, timeout=900)
            if st != 206:
                raise HttpStatus(st, self.path, data[:200])
            if len(data) != e["csize"]:
                raise RuntimeError(f"short read for {name}: {len(data)} of {e['csize']}")
        if e["method"] == 8:
            data = zlib.decompress(data, -15)
        elif e["method"] != 0:
            raise RuntimeError(f"{name}: unsupported zip method {e['method']}")
        if (zlib.crc32(data) & 0xFFFFFFFF) != e["crc"]:
            raise RuntimeError(f"{name}: CRC mismatch")
        if dest:
            with open(dest, "wb") as fh:
                fh.write(data)
            return dest
        return data

    def probe(self, name, n=64):
        """Status of a small range read of the member's data (the guard answers 403 for withheld parts)."""
        e = self.entries[name]
        st, _h, _b = get_range(self.path, e["off"], e["off"] + min(n, 30 + e["csize"]) - 1)
        return st
