# SPDX-License-Identifier: MIT
"""Reads-per-UMI matrices as compact ``.s1`` members.

The engine writes ``exon_reads.1pz`` / ``intron_reads.1pz`` next to ``exon_counts.1pz`` / ``intron_counts.1pz``: the same rows, the same
barcodes and the same nonzero positions, with the number of reads merged into the UMIs instead of the number of UMIs. Stored as matrices
they are as large as the count matrices (+69-73% per sample). Stored as the flat vector ``reads - UMIs`` aligned to the nonzeros of
the called columns of the count matrix they take about a fifth of that, and the reads matrix is rebuilt exactly from the count matrix:

    reads = counts[:, called] with data += delta

Member names are ``exon_reads_delta.bin`` and ``intron_reads_delta.bin``. Layout (little endian):

    magic ``S1RD``, version u8 (1), itemsize u8 (1, 2 or 4), n_cols u32 (called columns), nnz u64, crc32 u32 of the per-column nonzero
    counts (int64 diff of the indptr of the called columns), then the zstd (level 19) frame of ``nnz`` unsigned values of that width.

:func:`reads_matrix` returns the reads matrix of a sample object; the singlet-py and singlet-r readers need the same helper.
"""

from __future__ import annotations

import struct
import zlib
from typing import Optional

import numpy as np

from . import util
from .matrix import Csc

MAGIC = b"S1RD"
VERSION = 1
_HEAD = struct.Struct("<4sBBIQI")
MEMBERS = {"exon": "exon_reads_delta.bin", "intron": "intron_reads_delta.bin"}
COUNTS = {"exon": "exon_counts.1pz", "intron": "intron_counts.1pz"}
RUN_DIR_READS = {"exon": "exon_reads.1pz", "intron": "intron_reads.1pz"}
LEVEL = 19


class ReadsValuesError(ValueError):
    pass


def _called_columns(csc: Csc, called) -> tuple:
    """(column indices, per-column nonzero counts) of the called columns. ``called`` is a set of barcodes or a boolean array."""
    if isinstance(called, (set, frozenset)):
        sel = np.flatnonzero(np.fromiter((b in called for b in csc.colnames), dtype=bool, count=csc.n))
    else:
        sel = np.flatnonzero(np.asarray(called, dtype=bool))
        if len(sel) and int(sel[-1]) >= csc.n:
            raise ReadsValuesError("called mask is longer than the matrix")
    return sel, np.diff(np.asarray(csc.indptr, np.int64))[sel]


def _gather(csc: Csc, sel: np.ndarray, per_col: np.ndarray) -> np.ndarray:
    """The data of the selected columns in column order (the nonzeros of the called columns)."""
    ip = np.asarray(csc.indptr, np.int64)
    if not len(sel):
        return np.zeros(0, dtype=np.uint32)
    starts = ip[sel]
    total = int(per_col.sum())
    if total == 0:
        return np.zeros(0, dtype=np.uint32)
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(per_col)[:-1]]), per_col) + np.arange(total)
    return np.asarray(csc.data, np.uint32)[offs]


def encode(counts: Csc, reads: Csc, called) -> Optional[bytes]:
    """The member bytes for ``reads`` relative to ``counts`` over the called columns, or None when the two matrices do not line up (same
    rows, barcodes and nonzero positions, reads never below UMIs): the caller then leaves the member out."""
    if (counts.m, counts.n) != (reads.m, reads.n) or list(counts.colnames or []) != list(reads.colnames or []):
        return None
    sel, per_col = _called_columns(counts, called)
    if not np.array_equal(per_col, _called_columns(reads, called)[1]):
        return None
    ip_c, ip_r = np.asarray(counts.indptr, np.int64), np.asarray(reads.indptr, np.int64)
    # positions: row ids of the called columns must agree entry by entry
    ix_c = np.asarray(counts.indices, np.uint32)
    ix_r = np.asarray(reads.indices, np.uint32)
    c_rows = _gather_idx(ip_c, ix_c, sel, per_col)
    r_rows = _gather_idx(ip_r, ix_r, sel, per_col)
    if not np.array_equal(c_rows, r_rows):
        return None
    dc = _gather(counts, sel, per_col).astype(np.int64)
    dr = _gather(reads, sel, per_col).astype(np.int64)
    delta = dr - dc
    if len(delta) and int(delta.min()) < 0:
        return None
    top = int(delta.max()) if len(delta) else 0
    width = 1 if top <= 0xFF else 2 if top <= 0xFFFF else 4
    payload = util.zstd_compress(delta.astype({1: np.uint8, 2: np.uint16, 4: np.uint32}[width]).tobytes(), LEVEL)
    head = _HEAD.pack(MAGIC, VERSION, width, len(sel), len(delta), zlib.crc32(per_col.astype("<i8").tobytes()) & 0xFFFFFFFF)
    return head + payload


def _gather_idx(ip: np.ndarray, ix: np.ndarray, sel: np.ndarray, per_col: np.ndarray) -> np.ndarray:
    total = int(per_col.sum())
    if not len(sel) or total == 0:
        return np.zeros(0, dtype=np.uint32)
    offs = np.repeat(ip[sel] - np.concatenate([[0], np.cumsum(per_col)[:-1]]), per_col) + np.arange(total)
    return ix[offs]


def decode(blob: bytes, counts: Csc, called) -> Csc:
    """The reads matrix (called columns only; same row names, column order and nonzero positions as ``counts``) from a member's bytes."""
    if len(blob) < _HEAD.size:
        raise ReadsValuesError("reads member is truncated")
    magic, version, width, n_cols, nnz, crc = _HEAD.unpack_from(blob)
    if magic != MAGIC or version != VERSION or width not in (1, 2, 4):
        raise ReadsValuesError("not a reads member (magic %r version %d)" % (magic, version))
    sel, per_col = _called_columns(counts, called)
    if len(sel) != n_cols or int(per_col.sum()) != nnz or (zlib.crc32(per_col.astype("<i8").tobytes()) & 0xFFFFFFFF) != crc:
        raise ReadsValuesError("the reads member does not belong to this count matrix (called columns or nonzeros differ)")
    raw = util.zstd_decompress(blob[_HEAD.size :], nnz * width)
    delta = np.frombuffer(raw, dtype={1: np.uint8, 2: np.uint16, 4: np.uint32}[width])
    if len(delta) != nnz:
        raise ReadsValuesError("reads member payload has %d values, expected %d" % (len(delta), nnz))
    ip = np.asarray(counts.indptr, np.int64)
    indices = _gather_idx(ip, np.asarray(counts.indices, np.uint32), sel, per_col)
    data = (_gather(counts, sel, per_col).astype(np.int64) + delta.astype(np.int64)).astype(np.uint32)
    out = Csc(
        m=counts.m,
        n=len(sel),
        indptr=np.concatenate([[0], np.cumsum(per_col)]).astype(np.int64),
        indices=indices.astype(np.uint32),
        data=data,
        rownames=counts.rownames,
        colnames=[counts.colnames[i] for i in sel] if counts.colnames is not None else None,
        value_type=counts.value_type,
    )
    return out


def reads_matrix(sample, kind: str = "exon") -> Csc:
    """The reads-per-UMI matrix (``kind`` ``exon`` or ``intron``) of a sample object (:class:`singlet.s1.sample.S1Sample`), called cells only.

    Raises KeyError when the object holds no reads member (older objects, or the run did not write one)."""
    mem = MEMBERS[kind]
    if mem not in sample.members:
        raise KeyError("sample %s has no %s member" % (sample.sample_id, mem))
    cells = sample.cells()
    if cells is None:
        raise ReadsValuesError("the object has no cells table, so the called columns are unknown")
    counts = sample.matrix(COUNTS[kind])
    return decode(sample.reader.read_member(mem), counts, np.asarray(cells.column("called"), dtype=bool))


def from_run_dir_matrices(counts_1pz: bytes, reads_1pz: bytes, called: set) -> Optional[bytes]:
    """Member bytes from the two run-directory ``.1pz`` files (read with the compiled legacy reader) and the set of called barcodes."""
    from .legacy import decode_1pz_bytes

    counts, reads = decode_1pz_bytes(counts_1pz), decode_1pz_bytes(reads_1pz)
    if counts is None or reads is None:
        return None
    return encode(counts, reads, called)
