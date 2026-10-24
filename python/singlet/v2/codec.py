# SPDX-License-Identifier: MIT
"""Pluggable matrix codec for ``pz2`` members (docs/format-v2.md section 4 and 5; W1's C++ codec is the reference).

``MatrixCodec.encode(csc, opts) -> bytes`` / ``MatrixCodec.decode(bytes, mode) -> Csc`` is the whole interface the migration
needs. Two implementations:

* :class:`PyCodec2` - pure Python + numpy + zstandard, **codec 2** (generic gaps + zstd). Always available. It writes and reads the
  byte layout of ``include/singlet/pz2/detail/{container,codec2,api}.h`` exactly (256-byte header with CRC-32, sections packed in the order
  chunk_index, tables, rows, chunks, cols, kv, extra, the same names parts and chunk streams), so a member written here is readable by the
  C++ reader and vice versa; only the zstd frames themselves may differ byte for byte between zstd builds.
* :class:`NativeCodec` - adapter for the compiled codec 1 (lane rANS) / 2 (``singlet._pz2``, W4's binding of ``pz2.h``).

``get_codec()`` returns :class:`NativeCodec` when ``singlet._pz2`` imports (``SINGLET_V2_CODEC=py`` forces pure Python), else :class:`PyCodec2`.

Layout recap (little-endian)::

    header 256 B:  0 "PZ2M" | 4 version u16 | 6 codec u16 | 8 m u32 | 12 n u32 | 16 nnz u64 | 24 value_type u8 | 25 row_order u8 |
                   26 classes_per_octave u8 | 27 flags u8 (bit0: F32 coded as integers) | 28 chunk_cols u32 | 32 n_chunks u32 |
                   36 lanes u8 | 37 prob_bits u8 | 38 value_prob_bits u8 | 39 ctx_kind u8 | 40 row_names_id[8] | 48 col_names_id[8] |
                   56 digest[16] | 72 section[7] x {off u64, len u64} (ids: rows 0, cols 1, kv 2, tables 3, chunk_index 4, chunks 5,
                   extra 6; offsets from the member start) | 184 total_len u64 | 192 rows_kind u8 | 193 cols_kind u8 | 252 crc32(0..251)
    names kinds:   0 Inline (STREAM of the NUL-terminated names; rows live in ``extra``, columns in ``cols``), 1 Pack (equal to the
                   pack's list; id in the header), 2 PackSubset (STREAM of LEB128(zigzag(index delta)); id in the header),
                   3 Cells (columns only; id "CELLS\\0\\0\\0"), 4 None (no names; the digest hashes zero names)
    STREAM:        u32 raw_len, u32 zstd_len, zstd frame (raw_len 0: no frame)
    kv section:    u32 count, STREAM of ``key\\0value\\0...`` sorted by key
    chunk index:   n_chunks + 1 records {u64 off into chunks, u32 len, u32 first_col, u64 nnz_before}; the last one is the sentinel
    codec 2 chunk: STREAM(counts: LEB128 per column) STREAM(gaps: LEB128(row - previous_row - 1), previous_row = -1 at a column start)
                   STREAM(values: LEB128, or raw u32 bit patterns for non-integral f32)
"""

from __future__ import annotations

import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import util
from .matrix import VT_F32, VT_U8, VT_U16, VT_U32, Csc, canonical_digest

MAGIC = b"PZ2M"
VERSION = 1
CODEC_LANE_RANS = 1
CODEC_GENERIC = 2
HEADER_SIZE = 256
REC_BYTES = 24
MAX_CHUNK_NNZ = (1 << 28) // 2
CELLS_ID = b"CELLS\0\0\0"
ZERO8 = b"\0" * 8

NK_INLINE, NK_PACK, NK_SUBSET, NK_CELLS, NK_NONE = 0, 1, 2, 3, 4
S_ROWS, S_COLS, S_KV, S_TABLES, S_CHUNK_INDEX, S_CHUNKS, S_EXTRA = range(7)
SEC_NAMES = ("rows", "cols", "kv", "tables", "chunk_index", "chunks", "extra")
_IDX_DT = np.dtype([("off", "<u8"), ("len", "<u4"), ("first", "<u4"), ("nb", "<u8")])


class CodecError(ValueError):
    pass


@dataclass
class EncodeOptions:
    """What the migration tells the codec about names and layout.

    ``row_names`` / ``col_names``: ``"inline"`` (names stored in the member), ``"none"`` (the matrix has no names),
    ``("pack", id8)`` (names equal the pack's list), ``("subset", id8, indices)`` (names are ``pack[indices[i]]``; the index list is
    stored) or, for columns only, ``"cells"``. The Csc always carries the real names (except ``"none"``) so the digest can be computed;
    with a reference they are *not* stored. ``chunk_cols`` is the maximum columns per chunk (0 = W1's default policy);
    ``chunk_nnz_max`` ends a chunk before the column that would take it past that many non-zeros (0 = the library default
    ``DEFAULT_CHUNK_NNZ_MAX``, -1 = no budget: fixed-width chunks, the behaviour before).
    """

    chunk_cols: int = 0
    chunk_nnz_max: int = 0
    row_names: object = "inline"
    col_names: object = "inline"
    level: int = util.ZSTD_LEVEL
    threads: int = 4
    digest: Optional[bytes] = None


DEFAULT_CHUNK_NNZ_MAX = 49152  # pz2.h kDefaultChunkNnzMax


def chunk_budget(chunk_nnz_max: int) -> Optional[int]:
    """The per-chunk non-zero budget for an ``EncodeOptions.chunk_nnz_max`` value (None = no budget)."""
    if chunk_nnz_max < 0:
        return None
    return int(chunk_nnz_max) or DEFAULT_CHUNK_NNZ_MAX


def default_chunk_cols(n: int, nnz: int) -> int:
    """~96k non-zeros per chunk, a power of two in [128, 4096] (the policy of pz2.h ``default_chunk_cols``)."""
    if n == 0 or nnz == 0:
        return 1024
    want = 98304.0 / max(nnz / n, 1.0)
    c = 128
    while c < 4096 and c * 1.414 < want:
        c *= 2
    return c


# ----------------------------------------------------------------------------------------------------------- streams
def put_stream(raw: bytes, level: int) -> bytes:
    if len(raw) > 0xFFFFFFFF:
        raise CodecError("side stream larger than 4 GiB")
    z = util.zstd_compress(raw, level) if raw else b""
    return struct.pack("<II", len(raw), len(z)) + z


def get_stream(buf, p: int, end: int, max_raw: int):
    if end - p < 8:
        raise CodecError("truncated stream header")
    n, zn = struct.unpack_from("<II", buf, p)
    p += 8
    if zn > end - p:
        raise CodecError("truncated stream body")
    if n > max_raw:
        raise CodecError("stream raw length exceeds bound")
    if (n == 0) != (zn == 0):
        raise CodecError("stream length fields inconsistent")
    raw = util.zstd_decompress(bytes(buf[p : p + zn]), max_output_size=n) if n else b""
    if len(raw) != n:
        raise CodecError("zstd frame decodes to the wrong length")
    return raw, p + zn


def _zigzag(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def _unzigzag(u: np.ndarray) -> np.ndarray:
    u = u.astype(np.uint64)
    return (u >> np.uint64(1)).astype(np.int64) ^ -((u & np.uint64(1)).astype(np.int64))


def _names_kind(ref, is_col: bool):
    """-> (kind, id8, indices|None)"""
    if ref == "inline":
        return NK_INLINE, ZERO8, None
    if ref == "none":
        return NK_NONE, ZERO8, None
    if ref == "cells":
        if not is_col:
            raise CodecError("'cells' is only valid for column names")
        return NK_CELLS, CELLS_ID, None
    if isinstance(ref, tuple) and ref[0] == "pack" and len(ref[1]) == 8:
        return NK_PACK, bytes(ref[1]), None
    if isinstance(ref, tuple) and ref[0] == "subset" and len(ref[1]) == 8:
        return NK_SUBSET, bytes(ref[1]), np.asarray(ref[2], dtype=np.int64)
    raise CodecError("bad name reference %r" % (ref[:2] if isinstance(ref, tuple) else ref,))


def _names_part(
    kind: int, names: Optional[list], idx: Optional[np.ndarray], count: int, level: int
) -> bytes:
    if kind == NK_INLINE:
        if names is None or len(names) != count:
            raise CodecError("inline names: expected %d names" % count)
        return put_stream(util.join_nul(names), level)
    if kind == NK_SUBSET:
        if len(idx) != count:
            raise CodecError("pack subset: index list length != count")
        d = np.diff(np.concatenate([[0], idx]))
        return put_stream(util.leb128_encode(_zigzag(d)), level)
    return b""


def _names_decode(kind: int, buf, p: int, end: int, count: int):
    """-> (names list | None, index array | None)"""
    if kind == NK_INLINE:
        raw, _ = get_stream(buf, p, end, 1 << 34)
        names = util.split_nul(raw) if raw else []
        if len(names) != count:
            raise CodecError("inline names: wrong count")
        return names, None
    if kind == NK_SUBSET:
        raw, _ = get_stream(buf, p, end, count * 10 + 16)
        d = _unzigzag(util.leb128_decode(raw, count))
        idx = np.cumsum(d)
        if count and (int(idx.min()) < 0 or int(idx.max()) > 0xFFFFFFFF):
            raise CodecError("pack index out of range")
        return None, idx.astype(np.int64)
    return None, None


# ----------------------------------------------------------------------------------------------------------- header
def _write_header(h: dict) -> bytes:
    b = bytearray(HEADER_SIZE)
    b[0:4] = MAGIC
    struct.pack_into("<HHIIQ", b, 4, VERSION, h["codec"], h["m"], h["n"], h["nnz"])
    struct.pack_into(
        "<BBBBII",
        b,
        24,
        h["value_type"],
        h["row_order"],
        h["classes_per_octave"],
        h["flags"],
        h["chunk_cols"],
        h["n_chunks"],
    )
    struct.pack_into(
        "<BBBB", b, 36, h["lanes"], h["prob_bits"], h["value_prob_bits"], h["ctx_kind"]
    )
    b[40:48] = h["row_names_id"]
    b[48:56] = h["col_names_id"]
    b[56:72] = h["digest"]
    for i, (off, ln) in enumerate(h["sections"]):
        struct.pack_into("<QQ", b, 72 + 16 * i, off, ln)
    struct.pack_into("<Q", b, 184, h["total_len"])
    b[192] = h["rows_kind"]
    b[193] = h["cols_kind"]
    struct.pack_into("<I", b, 252, zlib.crc32(bytes(b[:252])) & 0xFFFFFFFF)
    return bytes(b)


def parse_header(buf) -> dict:
    if len(buf) < HEADER_SIZE:
        raise CodecError("pz2 member shorter than its 256-byte header")
    b = bytes(buf[:HEADER_SIZE])
    if b[:4] != MAGIC:
        raise CodecError("bad pz2 magic %r" % b[:4])
    ver, codec, m, n, nnz = struct.unpack_from("<HHIIQ", b, 4)
    if ver != VERSION:
        raise CodecError("unsupported pz2 version %d" % ver)
    if struct.unpack_from("<I", b, 252)[0] != (zlib.crc32(b[:252]) & 0xFFFFFFFF):
        raise CodecError("header CRC mismatch")
    vt, row_order, cpo, flags, chunk_cols, n_chunks = struct.unpack_from("<BBBBII", b, 24)
    lanes, pb, pbv, ctx = struct.unpack_from("<BBBB", b, 36)
    secs = {}
    for i, nm in enumerate(SEC_NAMES):
        secs[nm] = struct.unpack_from("<QQ", b, 72 + 16 * i)
    (total,) = struct.unpack_from("<Q", b, 184)
    if not 1 <= vt <= 4:
        raise CodecError("bad value_type")
    if b[192] > 4 or b[193] > 4:
        raise CodecError("bad names kind")
    for nm, (off, ln) in secs.items():
        if ln and (off < HEADER_SIZE or off > total or ln > total - off):
            raise CodecError("section %s outside the member" % nm)
    if secs["chunk_index"][1] != (n_chunks + 1) * REC_BYTES:
        raise CodecError("chunk index length does not match n_chunks")
    return dict(
        version=ver,
        codec=codec,
        m=m,
        n=n,
        nnz=nnz,
        value_type=vt,
        row_order=row_order,
        classes_per_octave=cpo,
        flags=flags,
        chunk_cols=chunk_cols,
        n_chunks=n_chunks,
        lanes=lanes,
        prob_bits=pb,
        value_prob_bits=pbv,
        ctx_kind=ctx,
        row_names_id=b[40:48],
        col_names_id=b[48:56],
        digest=b[56:72],
        sections=secs,
        total_len=total,
        rows_kind=b[192],
        cols_kind=b[193],
    )


def prefix_len(h: dict) -> int:
    """Bytes from the member start needed to open it: header + chunk index + tables + rows."""
    e = HEADER_SIZE
    for nm in ("chunk_index", "tables", "rows"):
        off, ln = h["sections"][nm]
        if ln:
            e = max(e, off + ln)
    return e


# ----------------------------------------------------------------------------------------------------------- interface
class MatrixCodec:
    """Interface. ``decode`` returns a :class:`Csc`; names that live outside the member (pack / cells) are left ``None`` with
    ``row_ref`` / ``col_ref`` set, the caller resolves them."""

    codec_id = 0

    def encode(self, csc: Csc, opts: Optional[EncodeOptions] = None) -> bytes:
        raise NotImplementedError

    def decode(self, buf, mode: str = "canonical", cols: Optional[tuple] = None) -> Csc:
        raise NotImplementedError

    def header(self, buf) -> dict:
        return parse_header(buf)


# ----------------------------------------------------------------------------------------------------------- codec 2
def _plan_chunks(
    indptr: np.ndarray, n: int, chunk_cols: int, nnz_budget: Optional[int] = None
) -> list:
    """Chunk starts (+ sentinel ``n``), as pz2.h ``plan_chunks``: at most ``chunk_cols`` columns, and a chunk ends before the column
    that would take it past ``nnz_budget`` non-zeros (at least one column per chunk)."""
    starts = []
    j = 0
    while j < n:
        starts.append(j)
        e = min(n, j + chunk_cols)
        if nnz_budget is not None and int(indptr[e] - indptr[j]) > nnz_budget:
            k = int(
                np.searchsorted(indptr[j + 1 : e + 1], int(indptr[j]) + nnz_budget, side="right")
            )  # columns that fit
            e = j + max(1, k)
        while e - j > 1 and int(indptr[e] - indptr[j]) > MAX_CHUNK_NNZ:
            e = j + max(1, (e - j) // 2)
        j = e
    starts.append(n)
    return starts


def _f32_integral(bits: np.ndarray):
    """(all values are exact non-negative integers < 2^24, their integer values)"""
    f = bits.view(np.float32)
    ok = (f >= 0) & (f < np.float32(16777216.0))
    if not ok.all():
        return False, None
    iv = f.astype(np.uint32)
    if not (iv.astype(np.float32).view(np.uint32) == bits).all():
        return False, None
    return True, iv


def _encode_chunk(indptr, indices, vals, raw_f32: bool, c0: int, c1: int, level: int) -> bytes:
    a, b = int(indptr[c0]), int(indptr[c1])
    counts = np.diff(indptr[c0 : c1 + 1])
    ids = indices[a:b].astype(np.int64)
    gaps = np.empty(b - a, dtype=np.int64)
    if b > a:
        gaps[1:] = ids[1:] - ids[:-1] - 1
        starts = (indptr[c0:c1] - a)[counts > 0]
        gaps[starts] = ids[starts]
    s_counts = put_stream(util.leb128_encode(counts), level)
    s_gaps = put_stream(util.leb128_encode(gaps), level)
    v = vals[a:b]
    s_vals = put_stream(
        np.ascontiguousarray(v, dtype="<u4").tobytes() if raw_f32 else util.leb128_encode(v), level
    )
    return s_counts + s_gaps + s_vals


def _decode_chunk(buf, ncols: int, nnz_chunk: int, raw_f32: bool, m: int):
    end = len(buf)
    raw, p = get_stream(buf, 0, end, ncols * 10 + 8)
    counts = util.leb128_decode(raw, ncols).astype(np.int64)
    if int(counts.sum()) != nnz_chunk:
        raise CodecError("chunk column counts do not match the chunk index")
    raw, p = get_stream(buf, p, end, nnz_chunk * 10 + 8)
    gaps = util.leb128_decode(raw, nnz_chunk).astype(np.int64)
    raw, p = get_stream(buf, p, end, nnz_chunk * 10 + 8)
    if p != end:
        raise CodecError("trailing bytes in a chunk")
    if raw_f32:
        if len(raw) != 4 * nnz_chunk:
            raise CodecError("value stream length")
        vals = np.frombuffer(raw, dtype="<u4").astype(np.uint32)
    else:
        vals64 = util.leb128_decode(raw, nnz_chunk)
        if nnz_chunk and int(vals64.max()) > 0xFFFFFFFF:
            raise CodecError("value overflow")
        vals = vals64.astype(np.uint32)
    if nnz_chunk:
        g1 = gaps + 1
        cs = np.cumsum(g1)
        starts = np.cumsum(counts) - counts
        ne = counts > 0
        base = np.zeros(ncols, dtype=np.int64)
        # id = (sum of (gap+1) since the column start) - 1 ; the first element of a column holds gap = id + 1 - 1 + 1 = id + 1 here
        base[ne] = (cs - g1)[starts[ne]]
        ids = cs - np.repeat(base, counts) - 1
        if int(ids.max()) >= m or int(ids.min()) < 0:
            raise CodecError("row id out of range")
        ids = ids.astype(np.uint32)
    else:
        ids = np.zeros(0, dtype=np.uint32)
    return counts, ids, vals


class PyCodec2(MatrixCodec):
    """Pure-Python implementation of pz2 codec 2 (the byte layout of W1's ``detail/codec2.h``)."""

    codec_id = CODEC_GENERIC

    # ------------------------------------------------------------------ encode
    def encode(self, csc: Csc, opts: Optional[EncodeOptions] = None) -> bytes:
        o = opts or EncodeOptions()
        why = csc.check_canonical()
        if why:
            raise CodecError("matrix is not canonical: " + why)
        m, n, nnz = int(csc.m), int(csc.n), csc.nnz
        vt = csc.value_type
        if vt not in (VT_U8, VT_U16, VT_U32, VT_F32):
            raise CodecError("bad value_type %r" % vt)
        rk, rid, ridx = _names_kind(o.row_names, False)
        ck, cid, cidx = _names_kind(o.col_names, True)
        if rk == NK_NONE and csc.rownames is not None:
            raise CodecError("'none' row names requested but the matrix has names")
        if ck == NK_NONE and csc.colnames is not None:
            raise CodecError("'none' column names requested but the matrix has names")
        digest = o.digest if o.digest is not None else canonical_digest(csc)
        data = np.ascontiguousarray(csc.data, dtype=np.uint32)
        flags, raw_f32 = 0, False
        if vt == VT_F32:
            ok, iv = _f32_integral(data)
            if ok:
                flags, data = 1, iv
            else:
                raw_f32 = True
        indptr = csc.indptr.astype(np.int64, copy=False)
        cc = int(o.chunk_cols) if o.chunk_cols else default_chunk_cols(n, nnz)
        first = _plan_chunks(indptr, n, max(cc, 1), chunk_budget(int(o.chunk_nnz_max)))
        bounds = list(zip(first[:-1], first[1:]))

        def work(bd):
            return _encode_chunk(indptr, csc.indices, data, raw_f32, bd[0], bd[1], o.level)

        if o.threads > 1 and len(bounds) > 1:
            with ThreadPoolExecutor(max_workers=o.threads) as ex:
                blobs = list(ex.map(work, bounds))
        else:
            blobs = [work(b) for b in bounds]
        idx = bytearray()
        off = 0
        for (c0, c1), bl in zip(bounds, blobs):
            idx += struct.pack("<QIIQ", off, len(bl), c0, int(indptr[c0]))
            off += len(bl)
        idx += struct.pack("<QIIQ", off, 0, n, nnz)
        chunks = b"".join(blobs)
        cols = _names_part(ck, csc.colnames, cidx, n, o.level)
        extra = _names_part(rk, csc.rownames, ridx, m, o.level)
        kv = b""
        if csc.kv:
            blob = b"".join(
                k.encode("utf-8") + b"\0" + str(csc.kv[k]).encode("utf-8") + b"\0"
                for k in sorted(csc.kv, key=lambda s: s.encode("utf-8"))
            )
            kv = struct.pack("<I", len(csc.kv)) + put_stream(blob, o.level)
        parts = {
            S_CHUNK_INDEX: bytes(idx),
            S_TABLES: b"",
            S_ROWS: b"",
            S_CHUNKS: chunks,
            S_COLS: cols,
            S_KV: kv,
            S_EXTRA: extra,
        }
        pos = HEADER_SIZE
        secs = [None] * 7
        for s in (S_CHUNK_INDEX, S_TABLES, S_ROWS, S_CHUNKS, S_COLS, S_KV, S_EXTRA):
            secs[s] = (pos, len(parts[s]))
            pos += len(parts[s])
        h = dict(
            codec=CODEC_GENERIC,
            m=m,
            n=n,
            nnz=nnz,
            value_type=vt,
            row_order=0,
            classes_per_octave=4,
            flags=flags,
            chunk_cols=cc,
            n_chunks=len(bounds),
            lanes=32,
            prob_bits=10,
            value_prob_bits=10,
            ctx_kind=0,
            row_names_id=rid if rk in (NK_PACK, NK_SUBSET) else ZERO8,
            col_names_id=cid if ck in (NK_PACK, NK_SUBSET, NK_CELLS) else ZERO8,
            digest=digest,
            sections=secs,
            total_len=pos,
            rows_kind=rk,
            cols_kind=ck,
        )
        out = [_write_header(h)]
        for s in (S_CHUNK_INDEX, S_TABLES, S_ROWS, S_CHUNKS, S_COLS, S_KV, S_EXTRA):
            out.append(parts[s])
        return b"".join(out)

    # ------------------------------------------------------------------ decode
    def decode(self, buf, mode: str = "canonical", cols: Optional[tuple] = None) -> Csc:
        """Whole member (or, with ``cols=(a, b)``, only columns ``a <= j < b``). ``mode`` is accepted for interface parity: codec 2
        stores canonical row order, so ``native`` == ``canonical``."""
        if mode not in ("canonical", "native"):
            raise CodecError("mode %r not supported by codec 2" % mode)
        h = parse_header(buf)
        if h["codec"] != CODEC_GENERIC:
            raise CodecError("PyCodec2 reads codec 2 only (got %d)" % h["codec"])
        return _decode_with(lambda off, ln: bytes(buf[off : off + ln]), h, cols)

    def read_columns(
        self,
        fetch: Callable[[int, int], bytes],
        member_len: int,
        cols: tuple,
        header: Optional[dict] = None,
    ) -> Csc:
        """Random column access over a range fetcher ``fetch(rel_off, length)`` (relative to the member start): one request for the
        header + chunk index (if ``header`` is not given) and one for the run of chunks covering ``cols``."""
        if header is None:
            first = fetch(0, min(member_len, 65536))
            header = parse_header(first)
        return _decode_with(fetch, header, cols)


def _decode_with(fetch, h: dict, cols: Optional[tuple]) -> Csc:
    m, n, nnz, vt = h["m"], h["n"], h["nnz"], h["value_type"]
    a, b = (0, n) if cols is None else (int(cols[0]), int(cols[1]))
    if not (0 <= a <= b <= n):
        raise CodecError("column range out of bounds")
    nch = h["n_chunks"]
    ioff, ilen = h["sections"]["chunk_index"]
    idx = np.frombuffer(fetch(ioff, ilen), dtype=_IDX_DT)
    if len(idx) != nch + 1:
        raise CodecError("chunk index has the wrong length")
    first = idx["first"].astype(np.int64)
    if nch and (
        first[0] != 0
        or (np.diff(first) <= 0).any()
        or int(first[-1]) != n
        or int(idx["nb"][-1]) != nnz
    ):
        raise CodecError("chunk index not monotone / sentinel mismatch")
    coff, clen = h["sections"]["chunks"]
    raw_f32 = vt == VT_F32 and not (h["flags"] & 1)
    if nch == 0 or a == b:
        csc = Csc(
            m,
            b - a,
            np.zeros(b - a + 1, dtype=np.int64),
            np.zeros(0, np.uint32),
            np.zeros(0, np.uint32),
            value_type=vt,
        )
    else:
        c_first = int(np.searchsorted(first, a, side="right")) - 1
        c_last = int(np.searchsorted(first, b - 1, side="right")) - 1
        run_off = int(idx["off"][c_first])
        run_end = int(idx["off"][c_last]) + int(idx["len"][c_last])
        run = fetch(coff + run_off, run_end - run_off)
        cl, il, vl = [], [], []
        for c in range(c_first, c_last + 1):
            s = int(idx["off"][c]) - run_off
            blob = run[s : s + int(idx["len"][c])]
            ct, ids, vals = _decode_chunk(
                blob,
                int(first[c + 1] - first[c]),
                int(idx["nb"][c + 1]) - int(idx["nb"][c]),
                raw_f32,
                m,
            )
            cl.append(ct)
            il.append(ids)
            vl.append(vals)
        counts, ids, vals = np.concatenate(cl), np.concatenate(il), np.concatenate(vl)
        if vt == VT_F32 and (h["flags"] & 1):
            vals = vals.astype(np.float32).view(np.uint32)  # integral floats were coded as integers
        lo = a - int(first[c_first])
        hi = lo + (b - a)
        ipf = np.concatenate([[0], np.cumsum(counts)])
        s0, s1 = int(ipf[lo]), int(ipf[hi])
        csc = Csc(
            m,
            b - a,
            (ipf[lo : hi + 1] - s0).astype(np.int64),
            ids[s0:s1],
            vals[s0:s1],
            value_type=vt,
        )
    # names
    rk, ck = h["rows_kind"], h["cols_kind"]
    if rk == NK_NONE:
        csc.row_ref = {"none": True}
    elif rk == NK_INLINE:
        off, ln = h["sections"]["extra"]
        csc.rownames = _names_decode(rk, fetch(off, ln), 0, ln, m)[0] if ln else []
    elif rk == NK_PACK:
        csc.row_ref = {"pack": h["row_names_id"].hex()}
    elif rk == NK_SUBSET:
        off, ln = h["sections"]["extra"]
        csc.row_ref = {
            "pack": h["row_names_id"].hex(),
            "index": _names_decode(rk, fetch(off, ln), 0, ln, m)[1],
        }
    else:
        raise CodecError("rows cannot reference the cells table")
    if ck == NK_NONE:
        csc.col_ref = {"none": True}
    elif ck == NK_INLINE:
        off, ln = h["sections"]["cols"]
        allc = _names_decode(ck, fetch(off, ln), 0, ln, n)[0] if ln else []
        csc.colnames = allc[a:b]
    elif ck == NK_CELLS:
        csc.col_ref = {"cells": True}
    elif ck == NK_PACK:
        csc.col_ref = {"pack": h["col_names_id"].hex(), "range": (a, b)}
    else:
        off, ln = h["sections"]["cols"]
        idx_all = _names_decode(ck, fetch(off, ln), 0, ln, n)[1]
        csc.col_ref = {"pack": h["col_names_id"].hex(), "index": idx_all[a:b]}
    off, ln = h["sections"]["kv"]
    if ln:
        blob = fetch(off, ln)
        (cnt,) = struct.unpack_from("<I", blob, 0)
        raw, _ = get_stream(blob, 4, len(blob), 1 << 30)
        parts = raw.split(b"\0")
        if parts[-1] != b"" or len(parts) != 2 * cnt + 1:
            raise CodecError("bad key/value section")
        csc.kv = {
            parts[i].decode("utf-8"): parts[i + 1].decode("utf-8") for i in range(0, 2 * cnt, 2)
        }
    csc.meta["digest"] = h["digest"]
    csc.meta["cols"] = (a, b)
    return csc


# ----------------------------------------------------------------------------------------------------------- native
class NativeCodec(MatrixCodec):
    """Adapter for the compiled codec (``singlet._pz2``, pybind11 over ``include/singlet/pz2/pz2.h``): codec 1 (lane rANS, class order,
    ``codec="auto"`` keeps whichever of codec 1 / 2 is smaller) and codec 2. Members are canonical row order on decode.

    The digest is always passed from :func:`singlet.v2.matrix.canonical_digest` (checked equal to the C++ digest by the interop test), so the
    binding never needs the names of a Pack / Cells / None reference."""

    codec_id = CODEC_LANE_RANS
    _VT = {VT_U8: "u8", VT_U16: "u16", VT_U32: "u32", VT_F32: "f32"}

    def __init__(self, codec: str = "auto"):
        try:
            from singlet import _pz2  # type: ignore
        except Exception as e:
            raise ImportError("singlet._pz2 is not available") from e
        self._mod = _pz2
        self.codec = codec

    @staticmethod
    def _spec(ref, names, is_col: bool):
        kind, pid, idx = _names_kind(ref, is_col)
        if kind == NK_INLINE:
            return list(names)
        if kind == NK_NONE:
            return {"kind": "none"}
        if kind == NK_CELLS:
            return {"kind": "cells"}
        if kind == NK_PACK:
            return {"kind": "pack", "pack_id": pid}
        return {
            "kind": "pack_subset",
            "pack_id": pid,
            "pack_index": np.asarray(idx, dtype=np.uint32),
        }

    def encode(self, csc: Csc, opts: Optional[EncodeOptions] = None) -> bytes:
        o = opts or EncodeOptions()
        why = csc.check_canonical()
        if why:
            raise CodecError("matrix is not canonical: " + why)
        vt = csc.value_type
        if vt not in self._VT:
            raise CodecError("bad value_type %r" % vt)
        digest = o.digest if o.digest is not None else canonical_digest(csc)
        data = np.ascontiguousarray(csc.data, dtype=np.uint32)
        if vt == VT_F32:
            data = data.view(np.float32)
        try:
            blob, _stats = self._mod.encode(
                csc.indptr.astype(np.int64, copy=False),
                np.ascontiguousarray(csc.indices, dtype=np.uint32),
                data,
                int(csc.m),
                int(csc.n),
                codec=self.codec,
                chunk_cols=int(o.chunk_cols),
                chunk_nnz_max=int(o.chunk_nnz_max),
                value_type=self._VT[vt],
                row_names=self._spec(o.row_names, csc.rownames, False),
                col_names=self._spec(o.col_names, csc.colnames, True),
                kv=dict(csc.kv) if csc.kv else None,
                digest=bytes(digest),
                compute_digest=False,
                threads=int(o.threads),
                zstd_level=int(o.level),
            )
        except (ValueError, RuntimeError) as e:
            raise CodecError("native encode failed: %s" % e) from e
        return bytes(blob)

    def decode(self, buf, mode: str = "canonical", cols: Optional[tuple] = None) -> Csc:
        if mode not in ("canonical", "native"):
            raise CodecError("mode %r not supported" % mode)
        buf = bytes(buf)
        h = parse_header(buf)
        try:
            d = (
                self._mod.decode(buf, mode="canonical")
                if cols is None
                else self._mod.decode_columns(buf, int(cols[0]), int(cols[1]), mode="canonical")
            )
            mem = self._mod.Member(buf)
            rn, cn, kv = mem.row_names(), mem.col_names(), mem.kv()
        except (ValueError, RuntimeError) as e:
            raise CodecError("native decode failed: %s" % e) from e
        a, b = (0, h["n"]) if cols is None else (int(cols[0]), int(cols[1]))
        data = np.asarray(d["data"])
        data = data.view(np.uint32) if data.dtype == np.float32 else data.astype(np.uint32)
        csc = Csc(
            h["m"],
            b - a,
            np.asarray(d["indptr"]).astype(np.int64),
            np.asarray(d["indices"]).astype(np.uint32),
            data,
            value_type=h["value_type"],
            kv=dict(kv),
        )
        k = rn["kind"]
        if k == "inline":
            csc.rownames = list(rn["names"]) if rn.get("names") is not None else []
        elif k == "none":
            csc.row_ref = {"none": True}
        elif k == "pack":
            csc.row_ref = {"pack": rn["pack_id"]}
        elif k == "pack_subset":
            csc.row_ref = {
                "pack": rn["pack_id"],
                "index": np.asarray(rn["pack_index"], dtype=np.int64),
            }
        k = cn["kind"]
        if k == "inline":
            csc.colnames = (list(cn["names"]) if cn.get("names") is not None else [])[a:b]
        elif k == "none":
            csc.col_ref = {"none": True}
        elif k == "cells":
            csc.col_ref = {"cells": True}
        elif k == "pack":
            csc.col_ref = {"pack": cn["pack_id"], "range": (a, b)}
        elif k == "pack_subset":
            csc.col_ref = {
                "pack": cn["pack_id"],
                "index": np.asarray(cn["pack_index"], dtype=np.int64)[a:b],
            }
        csc.meta["digest"] = h["digest"]
        csc.meta["cols"] = (a, b)
        csc.meta["codec"] = h["codec"]
        return csc


_DEFAULT: Optional[MatrixCodec] = None


def native_available() -> bool:
    try:
        NativeCodec()
        return True
    except Exception:
        return False


def get_codec(name: Optional[str] = None) -> MatrixCodec:
    """``py`` -> :class:`PyCodec2`; ``native`` / ``rans`` / ``generic`` -> :class:`NativeCodec` (``codec="auto"`` / rANS only / codec 2 only);
    default (``auto``): native with automatic codec choice when ``singlet._pz2`` imports, else pure Python. ``SINGLET_V2_CODEC`` overrides."""
    global _DEFAULT
    want = name or os.environ.get("SINGLET_V2_CODEC", "") or "auto"
    if want == "py":
        return PyCodec2()
    if want in ("native", "rans", "generic"):
        return NativeCodec({"native": "auto", "rans": "rans", "generic": "generic"}[want])
    if _DEFAULT is None:
        try:
            _DEFAULT = NativeCodec("auto")
        except Exception:
            _DEFAULT = PyCodec2()
    return _DEFAULT


def decode_any(buf, mode: str = "canonical", cols: Optional[tuple] = None) -> Csc:
    """Decode a pz2 member whatever its codec (codec 2 always works in pure Python; codec 1 needs ``singlet._pz2``)."""
    h = parse_header(buf)
    if h["codec"] == CODEC_GENERIC:
        return PyCodec2().decode(buf, mode, cols)
    return NativeCodec().decode(buf, mode, cols)
