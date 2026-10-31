# SPDX-License-Identifier: MIT
"""Cells table (docs/format-s1.md section 5.3, member kind 2).

One row per matrix column of the sample, in matrix column order; columnar::

    "CEL2" u16 version=1 u16 n_cols u32 n_rows
    n_cols x { u16 name_len, name, u8 dtype, u8 enc, u64 off, u64 len }      (off relative to the table start)
    column bodies, each at a multiple of 64

dtype: 1 u8, 2 u16, 3 u32, 4 i32, 5 f32, 6 f64, 7 bool (bitmap), 8 string, 9 barcode.
enc:   0 raw, 1 zstd, 2 delta + zstd (integer columns; wrapping differences).

Column bodies:

* numeric: little-endian values (raw), or the zstd frame of them / of their wrapping first differences.
* bool: ``np.packbits(bitorder="little")`` bytes (raw or zstd).
* string: NUL-terminated UTF-8 strings concatenated (raw or zstd).
* barcode (self describing, ``enc`` 0): ``u8 mode`` then
  mode 0 = zstd(strings), mode 1 = whitelist indices ``8 B pack id prefix, u8 bits, bit-packed (LSB first) indices``,
  mode 2 = 2-bit codes ``u8 k, bit-packed 2k-bit codes``.

Required columns: ``barcode``, ``called``, ``n_umi_exon``, ``n_umi_intron``, ``n_feat_exon``, ``n_feat_intron`` (u32 counts computed from
the matrices at write time); optional columns come from recipes (``cells_columns``).

Canonical digest: SHA-256 over ``b"CEL2CANON1"`` and, for every column in name order, the name (UTF-8, NUL terminated), the dtype byte
and the logical values (little-endian; bool as one byte 0/1; strings and barcodes as NUL-terminated UTF-8). digest16 = first 16 bytes.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import util
from .packs import Whitelist, barcodes_to_codes, codes_to_barcodes

MAGIC = b"CEL2"
VERSION = 1
DT_U8, DT_U16, DT_U32, DT_I32, DT_F32, DT_F64, DT_BOOL, DT_STR, DT_BARCODE = range(1, 10)
_NP = {DT_U8: "<u1", DT_U16: "<u2", DT_U32: "<u4", DT_I32: "<i4", DT_F32: "<f4", DT_F64: "<f8"}
_INT = {DT_U8, DT_U16, DT_U32, DT_I32}
_DT_NAMES = {
    "u8": DT_U8,
    "u16": DT_U16,
    "u32": DT_U32,
    "i32": DT_I32,
    "f32": DT_F32,
    "f64": DT_F64,
    "bool": DT_BOOL,
    "str": DT_STR,
    "string": DT_STR,
    "barcode": DT_BARCODE,
}
REQUIRED = ("barcode", "called", "n_umi_exon", "n_umi_intron", "n_feat_exon", "n_feat_intron")
_FIXED = struct.Struct("<BBQQ")


class CellsError(ValueError):
    pass


def normalize_dtype(dt) -> int:
    """Accept a dtype code, a name (``"u32"``), or a numpy dtype / dtype string."""
    if isinstance(dt, (int, np.integer)) and not isinstance(dt, bool):
        if 1 <= int(dt) <= 9:
            return int(dt)
        raise CellsError("bad cells dtype code %r" % (dt,))
    if isinstance(dt, str) and dt in _DT_NAMES:
        return _DT_NAMES[dt]
    try:
        d = np.dtype(dt)
    except TypeError:
        raise CellsError("bad cells dtype %r" % (dt,))
    if d.kind == "b":
        return DT_BOOL
    if d.kind in "USO":  # str, bytes, object
        return DT_STR
    if d.kind == "u" and d.itemsize <= 4:
        return {1: DT_U8, 2: DT_U16, 4: DT_U32}[d.itemsize]
    if d.kind == "i" and d.itemsize <= 4:
        return DT_I32
    if d.kind == "f":
        return DT_F32 if d.itemsize == 4 else DT_F64
    raise CellsError("unsupported cells dtype %r" % (dt,))


def _as_values(dtype: int, values, n_rows: int):
    if dtype in (DT_STR, DT_BARCODE):
        v = [str(x) for x in values]
    elif dtype == DT_BOOL:
        v = np.asarray(values).astype(bool)
    else:
        v = np.ascontiguousarray(np.asarray(values), dtype=_NP[dtype])
    if len(v) != n_rows:
        raise CellsError("column has %d values, table has %d rows" % (len(v), n_rows))
    return v


def _pack_bits(vals: np.ndarray, bits: int) -> bytes:
    v = np.asarray(vals, dtype=np.uint64)
    if bits == 0 or len(v) == 0:
        return b""
    bm = ((v[:, None] >> np.arange(bits, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bm.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(buf: bytes, n: int, bits: int) -> np.ndarray:
    if n == 0 or bits == 0:
        return np.zeros(n, dtype=np.uint64)
    flat = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")[: n * bits].reshape(
        n, bits
    )
    w = np.uint64(1) << np.arange(bits, dtype=np.uint64)
    return (flat.astype(np.uint64) * w[None, :]).sum(axis=1, dtype=np.uint64)


# DUAL (remove in the cleanup): objects written before the rename carry ``called_v2`` / ``called_v2_fdr``; read them as ``called_ed``.
OLD_COLUMN_NAMES = {"called_ed": "called_v2", "called_ed_fdr": "called_v2_fdr"}


def _alias(cols, name: str) -> str:
    if name not in cols:
        old = OLD_COLUMN_NAMES.get(name)
        if old is not None and old in cols:
            return old
    return name


@dataclass
class _Col:
    dtype: int
    values: Any  # ndarray, or list[str]
    # (Whitelist, indices) for barcode columns coded against a whitelist pack
    whitelist: Optional[tuple] = None


class CellsTable:
    """In-memory cells table. ``column(name)`` records which columns were read (used by the migration to keep only the optional
    columns that a verified recipe really needs)."""

    def __init__(self, n_rows: int):
        self.n_rows = int(n_rows)
        self._cols: dict[str, _Col] = {}
        self.accessed: set[str] = set()
        self.meta: dict = {}

    # ------------------------------------------------------------------ building
    def add(self, name: str, dtype, values, whitelist: Optional[tuple] = None) -> None:
        dt = normalize_dtype(dtype)
        if name in self._cols:
            raise CellsError("duplicate column %r" % name)
        self._cols[name] = _Col(dt, _as_values(dt, values, self.n_rows), whitelist)

    def drop(self, name: str) -> None:
        self._cols.pop(name, None)

    def copy(self, only: Optional[Sequence[str]] = None) -> "CellsTable":
        t = CellsTable(self.n_rows)
        t.meta = dict(self.meta)
        for k, c in self._cols.items():
            if only is None or k in only or k in REQUIRED:
                t._cols[k] = c
        return t

    # ------------------------------------------------------------------ access
    @property
    def names(self) -> list:
        return list(self._cols)

    def has(self, name: str) -> bool:
        return _alias(self._cols, name) in self._cols

    def dtype(self, name: str) -> int:
        return self._cols[_alias(self._cols, name)].dtype

    def column(self, name: str):
        name = _alias(self._cols, name)
        self.accessed.add(name)
        return self._cols[name].values

    __getitem__ = column

    def get(self, name: str, default=None):
        return self.column(name) if _alias(self._cols, name) in self._cols else default

    @property
    def barcodes(self) -> list:
        return self.column("barcode")

    @property
    def called(self) -> np.ndarray:
        return self.column("called")

    # ------------------------------------------------------------------ digest
    def digest(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"CEL2CANON1")
        for name in sorted(self._cols, key=lambda s: s.encode("utf-8")):
            c = self._cols[name]
            h.update(name.encode("utf-8") + b"\0" + bytes([c.dtype]))
            h.update(_logical_bytes(c))
        return h.digest()[:16]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        """Median UMIs / genes over called cells (exon + intron), for the study manifest."""
        out = {}
        if not self.n_rows or "called" not in self._cols:
            return out
        called = self._cols["called"].values
        if not called.any():
            return out
        u = (
            self._cols["n_umi_exon"].values.astype(np.int64)
            + self._cols["n_umi_intron"].values.astype(np.int64)
        )[called]
        g = np.maximum(
            self._cols["n_feat_exon"].values.astype(np.int64),
            self._cols["n_feat_intron"].values.astype(np.int64),
        )[called]
        out["median_umi"] = float(np.median(u))
        out["median_genes"] = float(np.median(g))
        return out

    # ------------------------------------------------------------------ encode
    def encode(self, level: int = util.ZSTD_LEVEL) -> bytes:
        names = list(self._cols)
        if len(names) > 0xFFFF:
            raise CellsError("too many columns")
        enc_list = [(n, self._cols[n].dtype) + _encode_column(self._cols[n], level) for n in names]
        head_len = 12 + sum(2 + len(n.encode("utf-8")) + _FIXED.size for n in names)
        pos = util.align_up(head_len)
        head = bytearray(MAGIC + struct.pack("<HHI", VERSION, len(names), self.n_rows))
        bodies = []
        for name, dt, enc, body in enc_list:
            nb = name.encode("utf-8")
            head += (
                struct.pack("<H", len(nb))
                + nb
                + _FIXED.pack(dt, enc, pos if body else 0, len(body))
            )
            bodies.append((pos, body))
            if body:
                pos = util.align_up(pos + len(body))
        end = max([head_len] + [o + len(b) for o, b in bodies if b])
        out = bytearray(end)
        out[:head_len] = head
        for o, b in bodies:
            if b:
                out[o : o + len(b)] = b
        return bytes(out)

    # ------------------------------------------------------------------ decode
    @classmethod
    def decode(cls, buf, store=None) -> "CellsTable":
        r = CellsReader(lambda off, ln: bytes(buf[off : off + ln]), len(buf), store)
        return r.load_all()


def _logical_bytes(c: _Col) -> bytes:
    if c.dtype in (DT_STR, DT_BARCODE):
        return util.join_nul(c.values)
    if c.dtype == DT_BOOL:
        return np.asarray(c.values, dtype=np.uint8).tobytes()
    return np.ascontiguousarray(c.values, dtype=_NP[c.dtype]).tobytes()


def _encode_column(c: _Col, level: int):
    """-> (enc, body)"""
    dt = c.dtype
    if dt == DT_BARCODE:
        return 0, _encode_barcodes(c, level)
    if dt == DT_STR:
        raw = util.join_nul(c.values)
        z = util.zstd_compress(raw, level) if raw else b""
        return (1, z) if raw and len(z) < len(raw) else (0, raw)
    if dt == DT_BOOL:
        raw = np.packbits(np.asarray(c.values, dtype=bool), bitorder="little").tobytes()
        z = util.zstd_compress(raw, level) if raw else b""
        return (1, z) if raw and len(z) < len(raw) else (0, raw)
    arr = np.ascontiguousarray(c.values, dtype=_NP[dt])
    raw = arr.tobytes()
    if not raw:
        return 0, b""
    cands = [(0, raw), (1, util.zstd_compress(raw, level))]
    if dt in _INT and len(arr) > 1:
        d = np.empty_like(arr)
        d[0] = arr[0]
        with np.errstate(over="ignore"):
            d[1:] = arr[1:] - arr[:-1]
        cands.append((2, util.zstd_compress(d.tobytes(), level)))
    return min(cands, key=lambda t: (len(t[1]), t[0]))


def _encode_barcodes(c: _Col, level: int) -> bytes:
    strs = c.values
    n = len(strs)
    if c.whitelist is not None:
        w, idx = c.whitelist
        bits = max(int(w.n - 1).bit_length(), 1)
        return b"\x01" + w.id[:8] + bytes([bits]) + _pack_bits(idx, bits)
    ks = {len(s) for s in strs}
    if len(ks) == 1 and n:
        k = ks.pop()
        if 1 <= k <= 32:
            codes = barcodes_to_codes(strs, k)
            if codes is not None:
                return b"\x02" + bytes([k]) + _pack_bits(codes, 2 * k)
    raw = util.join_nul(strs)
    return b"\x00" + (util.zstd_compress(raw, level) if raw else b"")


def _decode_barcodes(body: bytes, n: int, store) -> list:
    mode = body[0]
    if mode == 0:
        return util.split_nul(util.zstd_decompress(body[1:])) if n else []
    if mode == 1:
        pid = body[1:9]
        bits = body[9]
        idx = _unpack_bits(body[10:], n, bits).astype(np.int64)
        if store is None:
            from .packs import default_store

            store = default_store()
        w = store.get(pid.hex())
        if not isinstance(w, Whitelist):
            raise CellsError("pack %s is not a whitelist" % pid.hex())
        return w.barcodes(idx)
    if mode == 2:
        k = body[1]
        return codes_to_barcodes(_unpack_bits(body[2:], n, 2 * k), k)
    raise CellsError("unknown barcode mode %d" % mode)


class CellsReader:
    """Lazy reader over a range fetcher ``fetch(rel_off, length)``: the first request reads the header and the column directory,
    each ``column(name)`` then fetches only that column's body."""

    def __init__(
        self, fetch: Callable[[int, int], bytes], total_len: int, store=None, prefetch: int = 4096
    ):
        self._fetch = fetch
        self.total_len = total_len
        self.store = store
        first = fetch(0, min(total_len, prefetch))
        if first[:4] != MAGIC:
            raise CellsError("bad cells magic")
        ver, ncols, nrows = struct.unpack_from("<HHI", first, 4)
        if ver != VERSION:
            raise CellsError("unsupported cells version %d" % ver)
        self.n_rows = nrows
        p = 12
        entries = []
        # directory can exceed the prefetch: extend once
        need = 12 + ncols * (2 + 255 + _FIXED.size)
        if len(first) < min(need, total_len):
            first = fetch(0, min(total_len, need))
        for _ in range(ncols):
            (nl,) = struct.unpack_from("<H", first, p)
            p += 2
            name = first[p : p + nl].decode("utf-8")
            p += nl
            dt, enc, off, ln = _FIXED.unpack_from(first, p)
            p += _FIXED.size
            entries.append((name, dt, enc, off, ln))
        self.entries = {e[0]: e for e in entries}
        self._cache: dict[str, object] = {}
        self._first = first

    @property
    def names(self) -> list:
        return list(self.entries)

    def column(self, name: str):
        name = _alias(self.entries, name)
        if name in self._cache:
            return self._cache[name]
        n, dt, enc, off, ln = self.entries[name]
        body = self._fetch(off, ln) if ln else b""
        if len(body) != ln:
            raise CellsError("short read of column %r" % name)
        v = _decode_column(dt, enc, body, self.n_rows, self.store)
        self._cache[name] = v
        return v

    def load_all(self) -> CellsTable:
        t = CellsTable(self.n_rows)
        for name, (_n, dt, enc, off, ln) in self.entries.items():
            t._cols[name] = _Col(dt, self.column(name))
        return t


def _decode_column(dt: int, enc: int, body: bytes, n: int, store):
    if dt == DT_BARCODE:
        return _decode_barcodes(body, n, store)
    if dt == DT_STR:
        raw = util.zstd_decompress(body) if enc == 1 else body
        return util.split_nul(raw) if n else []
    if dt == DT_BOOL:
        raw = util.zstd_decompress(body) if enc == 1 else body
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:n].astype(bool)
    if dt not in _NP:
        raise CellsError("unknown dtype %d" % dt)
    if enc == 0:
        raw = body
    else:
        raw = util.zstd_decompress(body)
    arr = np.frombuffer(raw, dtype=_NP[dt]).copy() if raw else np.zeros(0, dtype=_NP[dt])
    if enc == 2:
        with np.errstate(over="ignore"):
            arr = np.cumsum(arr, dtype=arr.dtype)
    if len(arr) != n:
        raise CellsError("column length %d != n_rows %d" % (len(arr), n))
    return arr


# ====================================================================================================== required columns
def column_sums(indptr: np.ndarray, data: np.ndarray) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(data, dtype=np.uint64)])
    return cs[indptr[1:]] - cs[indptr[:-1]]


def _u32(a: np.ndarray, what: str) -> np.ndarray:
    if len(a) and int(a.max()) > 0xFFFFFFFF:
        raise CellsError("%s exceeds 32 bits" % what)
    return a.astype(np.uint32)


def base_table(
    barcodes: list, exon, intron, called: Optional[np.ndarray], whitelist: Optional[tuple] = None
) -> CellsTable:
    """Required columns from the (canonical) matrices. ``exon`` defines the rows; ``intron`` may be None or have another column
    list (then it is aligned by barcode, missing = 0). ``called``: bool array (None = all cells called)."""
    n = len(barcodes)
    t = CellsTable(n)
    t.add("barcode", DT_BARCODE, barcodes, whitelist=whitelist)
    t.add("called", DT_BOOL, np.ones(n, bool) if called is None else called)
    for tag, mat in (("exon", exon), ("intron", intron)):
        if mat is None or mat.n == 0:
            umi = np.zeros(n, np.uint64)
            feat = np.zeros(n, np.int64)
        else:
            umi_m = column_sums(mat.indptr, mat.data)
            feat_m = np.diff(mat.indptr)
            if mat.colnames is not None and list(mat.colnames) == list(barcodes):
                umi, feat = umi_m, feat_m
            else:
                pos = {b: i for i, b in enumerate(mat.colnames)}
                src = np.array([pos.get(b, -1) for b in barcodes], dtype=np.int64)
                ok = src >= 0
                umi = np.zeros(n, np.uint64)
                feat = np.zeros(n, np.int64)
                umi[ok] = umi_m[src[ok]]
                feat[ok] = feat_m[src[ok]]
        t.add("n_umi_" + tag, DT_U32, _u32(umi, "n_umi_" + tag))
        t.add("n_feat_" + tag, DT_U32, _u32(feat.astype(np.uint64), "n_feat_" + tag))
    # canonical column order: barcode, called, n_umi_exon, n_umi_intron, n_feat_exon, n_feat_intron
    t._cols = {k: t._cols[k] for k in REQUIRED}
    return t


# ====================================================================================================== cell_calls parsing
_BARCODE_COLUMNS = ("barcode", "cb", "cell_barcode", "CB")
_TRUE_STRINGS = frozenset({"true", "t", "1", "1.0", "yes", "y"})


def parse_cell_calls_column(data: bytes, column: str) -> Optional[list]:
    """Barcodes whose ``column`` is true in a ``cell_calls.tsv`` (``called_knee`` in the called_v2 format: the knee set the
    call starts from). None when the file has no barcode column or no such column."""
    if not data.strip():
        return None
    lines = data.decode("utf-8", "replace").splitlines()
    head = lines[0].split("\t")
    bcol = next((c for c in _BARCODE_COLUMNS if c in head), None)
    if bcol is None or column not in head:
        return None
    bi, ci = head.index(bcol), head.index(column)
    out = []
    for ln in lines[1:]:
        f = ln.split("\t")
        if ln and bi < len(f) and ci < len(f) and f[ci].strip().lower() in _TRUE_STRINGS and f[bi] != "":
            out.append(f[bi])
    return out


def parse_cell_calls_column(data: bytes, column: str) -> Optional[list]:
    """Barcodes whose ``column`` is true in a ``cell_calls.tsv`` (``called_knee`` in the called_v2 format: the knee set the
    call starts from). None when the file has no barcode column or no such column."""
    if not data.strip():
        return None
    lines = data.decode("utf-8", "replace").splitlines()
    head = lines[0].split("	")
    bcol = next((c for c in _BARCODE_COLUMNS if c in head), None)
    if bcol is None or column not in head:
        return None
    bi, ci = head.index(bcol), head.index(column)
    out = []
    for ln in lines[1:]:
        f = ln.split("	")
        if ln and bi < len(f) and ci < len(f) and f[ci].strip().lower() in _TRUE_STRINGS and f[bi] != "":
            out.append(f[bi])
    return out


def parse_cell_calls(data: bytes) -> Optional[list]:
    """Called-cell barcodes of a legacy ``cell_calls.tsv`` in file order (the rule of ``SingletBundle``: any barcode column
    spelling; ``is_cell`` honoured when present, else every listed barcode counts). None if no barcode column."""
    if not data.strip():
        return None
    lines = data.decode("utf-8", "replace").splitlines()
    head = lines[0].split("\t")
    col = next((c for c in _BARCODE_COLUMNS if c in head), None)
    if col is None:
        return None
    bi = head.index(col)
    ci = head.index("is_cell") if "is_cell" in head else -1
    out = []
    for ln in lines[1:]:
        if not ln:
            continue
        f = ln.split("\t")
        if bi >= len(f):
            continue
        if ci >= 0 and (ci >= len(f) or f[ci].strip().lower() not in _TRUE_STRINGS):
            continue
        if f[bi] != "":
            out.append(f[bi])
    return out
