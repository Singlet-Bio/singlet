# SPDX-License-Identifier: MIT
"""Shared low-level helpers for the v2 container code: digests, LEB128, zstd, alignment.

Everything here is pure Python + numpy (+ the ``zstandard`` module). No C++ is needed for the container layer.
"""

from __future__ import annotations

import hashlib
import json
import threading
import zlib
from typing import Iterable

import numpy as np

try:  # the module is a hard dependency of the v2 package (pip: zstandard)
    import zstandard as _zstd
except ImportError as e:  # pragma: no cover
    raise ImportError("singlet.v2 needs the 'zstandard' package (pip install zstandard)") from e

ALIGN = 64
ZSTD_LEVEL = 19


# --------------------------------------------------------------------------------------------- digests / crc
def sha256(data) -> bytes:
    return hashlib.sha256(data).digest()


def d16(data) -> bytes:
    """digest16: the first 16 bytes of SHA-256."""
    return hashlib.sha256(data).digest()[:16]


def crc32(data, value: int = 0) -> int:
    return zlib.crc32(data, value) & 0xFFFFFFFF


def hex16(d: bytes) -> str:
    return d.hex()


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def canon_json(obj) -> bytes:
    """Deterministic JSON bytes for meta blobs and manifests (sorted keys, compact)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


# --------------------------------------------------------------------------------------------- zstd
_TLS = (
    threading.local()
)  # zstandard compressor/decompressor objects are not thread safe: one set per thread


def zstd_compress(data, level: int = ZSTD_LEVEL) -> bytes:
    cache = getattr(_TLS, "comp", None)
    if cache is None:
        cache = _TLS.comp = {}
    c = cache.get(level)
    if c is None:
        c = cache[level] = _zstd.ZstdCompressor(
            level=level, write_content_size=True, write_checksum=False
        )
    return c.compress(data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data))


def zstd_decompress(data, max_output_size: int = 0) -> bytes:
    d = getattr(_TLS, "decomp", None)
    if d is None:
        d = _TLS.decomp = _zstd.ZstdDecompressor()
    if max_output_size:
        return d.decompress(data, max_output_size=max_output_size)
    return d.decompress(data)


# --------------------------------------------------------------------------------------------- LEB128 (vectorised)
def leb128_encode(values) -> bytes:
    """Unsigned LEB128 of an integer array (any width up to 64 bits), vectorised."""
    v = np.asarray(values).astype(np.uint64, copy=False)
    n = v.size
    if n == 0:
        return b""
    nb = np.ones(n, dtype=np.int64)
    t = v >> np.uint64(7)
    while t.any():
        nb += t != 0
        t = t >> np.uint64(7)
    ends = np.cumsum(nb)
    starts = ends - nb
    out = np.empty(int(ends[-1]), dtype=np.uint8)
    maxb = int(nb.max())
    for k in range(maxb):
        m = nb > k
        idx = starts[m] + k
        byte = ((v[m] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        cont = nb[m] > (k + 1)
        byte[cont] |= 0x80
        out[idx] = byte
    return out.tobytes()


def leb128_decode(buf, count: int | None = None) -> np.ndarray:
    """Decode unsigned LEB128 (vectorised) into uint64. ``count`` (optional) is verified."""
    a = np.frombuffer(buf, dtype=np.uint8)
    if a.size == 0:
        if count not in (None, 0):
            raise ValueError("LEB128 stream is empty, expected %d values" % count)
        return np.zeros(0, dtype=np.uint64)
    term = a < 0x80
    ends = np.flatnonzero(term)
    if ends.size == 0 or ends[-1] != a.size - 1:
        raise ValueError("truncated LEB128 stream")
    if count is not None and ends.size != count:
        raise ValueError("LEB128 stream holds %d values, expected %d" % (ends.size, count))
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lens = ends - starts + 1
    maxl = int(lens.max())
    if maxl > 10:
        raise ValueError("LEB128 value longer than 10 bytes")
    out = np.zeros(ends.size, dtype=np.uint64)
    for k in range(maxl):
        m = lens > k
        out[m] |= (a[starts[m] + k] & 0x7F).astype(np.uint64) << np.uint64(7 * k)
    return out


# --------------------------------------------------------------------------------------------- name lists
def join_nul(names: Iterable[str]) -> bytes:
    """NUL-terminated concatenation of UTF-8 names (``a\\0b\\0``)."""
    return (
        b"".join(s.encode("utf-8") + b"\0" for s in names)
        if not isinstance(names, list)
        else (("\0".join(names) + "\0").encode("utf-8") if names else b"")
    )


def split_nul(blob: bytes) -> list[str]:
    if not blob:
        return []
    if blob[-1:] != b"\0":
        raise ValueError("name blob is not NUL terminated")
    return blob[:-1].decode("utf-8").split("\0")
