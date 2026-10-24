# SPDX-License-Identifier: MIT
"""Sample object ``.s2`` (docs/format-v2.md section 3): writer and reader.

Layout::

    0   FileHeader (64 B): "SINGLET2" u16 version=1 u16 flags=0 u32 n_members u64 table_off=64 u64 table_len u64 file_len
                           16 B sample id (ASCII, NUL padded) 8 B reserved
    64  MemberTable: n_members x { u16 name_len, name, u8 kind, u8 flags, u64 off, u64 len, u64 out_len, u32 crc32,
                                   16 B digest, u32 meta_len, meta (canonical JSON) }
        member bodies, each starting at a multiple of 64

``kind``: 1 pz2 matrix, 2 cells table, 3 stored blob (zstd), 4 derived (recipe, no body), 5 raw.
``flags``: bit0 legacy member (regenerates a member of the legacy bundle), bit1 required for ``to_anndata``.

The reader works on bytes, a path, a binary file object, or a *range fetcher* ``fetch(offset, length) -> bytes``; with a fetcher
only the header, the table and the ranges that are asked for are read (one request each).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import util

MAGIC = b"SINGLET2"
VERSION = 1
HEADER_SIZE = 64
_HDR = struct.Struct("<8sHHIQQQ16s8s")
assert _HDR.size == HEADER_SIZE

KIND_PZ2 = 1
KIND_CELLS = 2
KIND_STORED = 3
KIND_DERIVED = 4
KIND_RAW = 5
KIND_NAMES = {1: "pz2", 2: "cells", 3: "stored", 4: "derived", 5: "raw"}

FLAG_LEGACY = 1
FLAG_ANNDATA = 2

_ENTRY_FIXED = struct.Struct("<BBQQQI16sI")  # kind flags off len out_len crc32 digest meta_len


class S2Error(ValueError):
    """Malformed or inconsistent ``.s2`` object."""


@dataclass
class MemberInfo:
    name: str
    kind: int
    flags: int
    off: int
    len: int
    out_len: int
    crc32: int
    digest: bytes
    meta: dict = field(default_factory=dict)

    @property
    def legacy(self) -> bool:
        return bool(self.flags & FLAG_LEGACY)


@dataclass
class _Pending:
    name: str
    kind: int
    flags: int
    body: Optional[bytes]
    out_len: int
    crc32: int
    digest: bytes
    meta: dict


class S2Writer:
    """Collects members and lays out the object. ``build()`` returns the complete object bytes."""

    def __init__(self, sample_id: str):
        sid = sample_id.encode("ascii")
        if len(sid) > 16:
            raise S2Error("sample id longer than 16 bytes: %r" % sample_id)
        self.sample_id = sample_id
        self._members: list[_Pending] = []
        self._names: set[str] = set()
        self.infos: list[MemberInfo] = []  # filled by build()

    # ------------------------------------------------------------------ adding members
    def _add(self, p: _Pending) -> None:
        if p.name in self._names:
            raise S2Error("duplicate member name %r" % p.name)
        if len(p.name.encode("utf-8")) > 0xFFFF:
            raise S2Error("member name too long")
        self._names.add(p.name)
        self._members.append(p)

    def add_matrix(
        self, name: str, body: bytes, digest: bytes, meta: Optional[dict] = None, flags: int = 0
    ) -> None:
        self._add(_Pending(name, KIND_PZ2, flags, bytes(body), 0, 0, digest, meta or {}))

    def add_cells(
        self, name: str, body: bytes, digest: bytes, meta: Optional[dict] = None, flags: int = 0
    ) -> None:
        self._add(_Pending(name, KIND_CELLS, flags, bytes(body), 0, 0, digest, meta or {}))

    def add_stored(
        self,
        name: str,
        legacy_bytes: bytes,
        meta: Optional[dict] = None,
        flags: int = FLAG_LEGACY,
        level: int = util.ZSTD_LEVEL,
        crc32: Optional[int] = None,
        digest: Optional[bytes] = None,
    ) -> int:
        """Store ``legacy_bytes`` as zstd (kind 3), or raw (kind 5) when zstd does not shrink it. Returns the kind used."""
        raw = bytes(legacy_bytes)
        z = util.zstd_compress(raw, level) if raw else b""
        m = dict(meta or {})
        if raw and len(z) < len(raw):
            kind, body = KIND_STORED, z
            m["zstd_level"] = level
        else:
            kind, body = KIND_RAW, raw
        self._add(
            _Pending(
                name,
                kind,
                flags,
                body,
                len(raw),
                util.crc32(raw) if crc32 is None else crc32,
                util.d16(raw) if digest is None else digest,
                m,
            )
        )
        return kind

    def add_derived(
        self,
        name: str,
        out_len: int,
        crc32: int,
        digest: bytes,
        meta: dict,
        flags: int = FLAG_LEGACY,
    ) -> None:
        if "recipe" not in meta:
            raise S2Error("derived member needs meta['recipe']")
        self._add(_Pending(name, KIND_DERIVED, flags, None, int(out_len), int(crc32), digest, meta))

    # ------------------------------------------------------------------ layout
    def build(self) -> bytes:
        metas = [util.canon_json(p.meta) for p in self._members]
        names = [p.name.encode("utf-8") for p in self._members]
        table_len = sum(2 + len(nm) + _ENTRY_FIXED.size + len(mb) for nm, mb in zip(names, metas))
        pos = util.align_up(HEADER_SIZE + table_len)
        offs = []
        for p in self._members:
            if p.body is None or len(p.body) == 0:
                offs.append((0, 0))  # derived members and empty bodies occupy no bytes
                continue
            offs.append((pos, len(p.body)))
            pos = util.align_up(pos + len(p.body))
        # the object ends at the end of the last body (no trailing padding)
        end = HEADER_SIZE + table_len
        for o, ln in offs:
            if ln:
                end = max(end, o + ln)
        file_len = end
        tab = bytearray()
        self.infos = []
        for p, nm, mb, (o, ln) in zip(self._members, names, metas, offs):
            tab += struct.pack("<H", len(nm)) + nm
            tab += (
                _ENTRY_FIXED.pack(p.kind, p.flags, o, ln, p.out_len, p.crc32, p.digest, len(mb))
                + mb
            )
            self.infos.append(
                MemberInfo(
                    p.name, p.kind, p.flags, o, ln, p.out_len, p.crc32, p.digest, dict(p.meta)
                )
            )
        assert len(tab) == table_len
        hdr = _HDR.pack(
            MAGIC,
            VERSION,
            0,
            len(self._members),
            HEADER_SIZE,
            table_len,
            file_len,
            self.sample_id.encode("ascii").ljust(16, b"\0"),
            b"\0" * 8,
        )
        out = bytearray(file_len)
        out[:HEADER_SIZE] = hdr
        out[HEADER_SIZE : HEADER_SIZE + table_len] = tab
        for p, (o, ln) in zip(self._members, offs):
            if ln:
                out[o : o + ln] = p.body
        return bytes(out)


# --------------------------------------------------------------------------------------------------------- reading
Fetch = Callable[[int, int], bytes]


class _BytesSource:
    def __init__(self, data):
        self.data = memoryview(data)
        self.size = len(self.data)

    def __call__(self, off: int, length: int) -> bytes:
        return bytes(self.data[off : off + length])


class _FileSource:
    def __init__(self, path):
        self.path = os.fspath(path)
        self.size = os.path.getsize(self.path)

    def __call__(self, off: int, length: int) -> bytes:
        with open(self.path, "rb") as fh:
            fh.seek(off)
            return fh.read(length)


class S2Reader:
    """Read a sample object from bytes, a path, or a ``fetch(offset, length)`` callable.

    With a callable, ``size`` may be given if known; otherwise it is learned from the header (``file_len``). The first request
    reads ``prefetch`` bytes, which normally cover header and table.
    """

    def __init__(self, source, size: Optional[int] = None, prefetch: int = 65536):
        if isinstance(source, (bytes, bytearray, memoryview)):
            self._fetch: Fetch = _BytesSource(source)
            self._whole = self._fetch.data
        elif isinstance(source, (str, os.PathLike)):
            self._fetch = _FileSource(source)
            self._whole = None
        elif callable(source):
            self._fetch = source
            self._whole = None
        else:
            raise TypeError("source must be bytes, a path or a fetch(offset, length) callable")
        self.n_requests = 0
        self.bytes_fetched = 0
        head = self._get(0, prefetch)
        if len(head) < HEADER_SIZE:
            raise S2Error("object shorter than the 64-byte header")
        (magic, ver, flags, n_members, table_off, table_len, file_len, sid, _res) = (
            _HDR.unpack_from(head, 0)
        )
        if magic != MAGIC:
            raise S2Error("bad magic %r" % magic)
        if ver != VERSION:
            raise S2Error("unsupported .s2 version %d" % ver)
        if table_off != HEADER_SIZE:
            raise S2Error("unexpected table offset %d" % table_off)
        self.version, self.flags, self.n_members = ver, flags, n_members
        self.table_off, self.table_len, self.file_len = table_off, table_len, file_len
        self.sample_id = sid.rstrip(b"\0").decode("ascii")
        if size is not None and size != file_len:
            raise S2Error("object is %d bytes, header says %d" % (size, file_len))
        if isinstance(self._fetch, (_BytesSource, _FileSource)) and self._fetch.size != file_len:
            raise S2Error("object is %d bytes, header says %d" % (self._fetch.size, file_len))
        end = table_off + table_len
        if end > file_len:
            raise S2Error("member table runs past the end of the object")
        tab = head[table_off:end] if end <= len(head) else self._get(table_off, table_len)
        if len(tab) != table_len:
            raise S2Error("short read of the member table")
        self.members: dict[str, MemberInfo] = {}
        p = 0
        for _ in range(n_members):
            if p + 2 > len(tab):
                raise S2Error("member table truncated")
            (nl,) = struct.unpack_from("<H", tab, p)
            p += 2
            name = tab[p : p + nl].decode("utf-8")
            p += nl
            if p + _ENTRY_FIXED.size > len(tab):
                raise S2Error("member table truncated")
            kind, fl, off, ln, out_len, crc, dg, ml = _ENTRY_FIXED.unpack_from(tab, p)
            p += _ENTRY_FIXED.size
            meta = json.loads(tab[p : p + ml].decode("utf-8")) if ml else {}
            p += ml
            if kind == KIND_DERIVED:
                if off or ln:
                    raise S2Error("derived member %r has a body" % name)
            else:
                if off % util.ALIGN or off < table_off + table_len and ln:
                    raise S2Error("member %r is misaligned or overlaps the table" % name)
                if off + ln > file_len:
                    raise S2Error("member %r runs past the end of the object" % name)
            if name in self.members:
                raise S2Error("duplicate member %r" % name)
            self.members[name] = MemberInfo(name, kind, fl, off, ln, out_len, crc, dg, meta)
        if p != len(tab):
            raise S2Error("trailing bytes in the member table")
        self._head = head

    # ------------------------------------------------------------------ io
    def _get(self, off: int, length: int) -> bytes:
        self.n_requests += 1
        b = self._fetch(off, length)
        self.bytes_fetched += len(b)
        return b

    def read_range(self, name: str, rel_off: int = 0, length: Optional[int] = None) -> bytes:
        """Bytes ``[rel_off, rel_off+length)`` of a member body (one fetch, or none if already in the first read)."""
        mi = self.members[name]
        if length is None:
            length = mi.len - rel_off
        if rel_off < 0 or rel_off + length > mi.len:
            raise S2Error("range outside member %r" % name)
        a = mi.off + rel_off
        if a + length <= len(self._head):
            return bytes(self._head[a : a + length])
        b = self._get(a, length)
        if len(b) != length:
            raise S2Error("short read of member %r" % name)
        return b

    def read_member(self, name: str) -> bytes:
        return self.read_range(name, 0, self.members[name].len)

    # ------------------------------------------------------------------ convenience
    def names(self) -> list[str]:
        return list(self.members)

    def kind_of(self, name: str) -> int:
        return self.members[name].kind

    def legacy_blob(self, name: str) -> bytes:
        """Regenerated legacy bytes of a stored (kind 3) or raw (kind 5) member, CRC-checked."""
        mi = self.members[name]
        body = self.read_member(name)
        if mi.kind == KIND_STORED:
            raw = util.zstd_decompress(body, max_output_size=mi.out_len or 1)
        elif mi.kind == KIND_RAW:
            raw = body
        else:
            raise S2Error("member %r is kind %d, not a stored blob" % (name, mi.kind))
        if len(raw) != mi.out_len or util.crc32(raw) != mi.crc32:
            raise S2Error("stored member %r fails its length/CRC check" % name)
        return raw

    def object_digest(self) -> bytes:
        """digest16 of the whole object (needs the whole object: fetches it if it is not local)."""
        if self._whole is not None:
            return util.d16(self._whole)
        return util.d16(self._get(0, self.file_len))

    def whole_bytes(self) -> bytes:
        if self._whole is not None:
            return bytes(self._whole)
        return self._get(0, self.file_len)
