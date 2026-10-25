# SPDX-License-Identifier: MIT
"""Chromatin (ATAC) and spatial members of a sample object (docs/format-v2.md section 14).

A pipeline run directory that was processed with the ATAC path or the Visium path holds, next to the RNA outputs::

    fragments.tsv.gz  fragments.tsv.gz.tbi  peaks.bed                       -> atac_fragments.tsv.gz (+ .tbi), atac_peaks.bed
    atac_peak_counts.mtx.gz   + _features.tsv.gz + _barcodes.tsv.gz        -> atac_peak_counts.1pz   (pz2, peaks x columns)
    atac_gene_activity.mtx.gz + _features.tsv.gz + _barcodes.tsv.gz        -> atac_gene_activity.1pz (pz2, genes x columns)
    atac_summary.json                                                       -> atac_summary.json
    spatial_positions.tsv, spatial_summary.json                             -> same names

:func:`add_section14_members` turns those files into member specs; :func:`add_to_writer` puts them in an ``S2Writer``. The members
carry flag bit 0 (legacy) clear: they regenerate no member of the legacy bundle (section 13 pattern, same as ADT).

Column alignment. A matrix of the run has one column per barcode it lists; the sample's columns are the ``cells`` table's rows.
The matrix is re-laid on the cells table: a matrix barcode that is not in the cells table is dropped (counted in the member meta
as ``n_dropped_barcodes``), a cells-table barcode that is not in the matrix gets an empty column (``n_empty_columns``). The matrix
columns are then ``Cells`` names, i.e. they reference the cells table and store no barcodes. ``spatial_positions.tsv`` is rewritten
only when its rows are not already one per cells-table barcode in the cells-table order.
"""

from __future__ import annotations

import gzip
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import util
from .codec import EncodeOptions, PyCodec2, get_codec, parse_header
from .container import S2Writer
from .matrix import VT_U32, Csc

ATAC_FRAGMENTS = "atac_fragments.tsv.gz"
ATAC_FRAGMENTS_TBI = "atac_fragments.tsv.gz.tbi"
ATAC_PEAKS = "atac_peaks.bed"
ATAC_PEAK_COUNTS = "atac_peak_counts.1pz"
ATAC_GENE_ACTIVITY = "atac_gene_activity.1pz"
ATAC_SUMMARY = "atac_summary.json"
SPATIAL_POSITIONS = "spatial_positions.tsv"
SPATIAL_SUMMARY = "spatial_summary.json"

SECTION14_MEMBERS = (
    ATAC_FRAGMENTS,
    ATAC_FRAGMENTS_TBI,
    ATAC_PEAKS,
    ATAC_PEAK_COUNTS,
    ATAC_GENE_ACTIVITY,
    ATAC_SUMMARY,
    SPATIAL_POSITIONS,
    SPATIAL_SUMMARY,
)

# member name -> prefix of the run directory's mtx / features / barcodes files
_MATRICES = (
    (ATAC_PEAK_COUNTS, "atac_peak_counts"),
    (ATAC_GENE_ACTIVITY, "atac_gene_activity"),
)


@dataclass
class MemberSpec:
    """One member to add to a sample object: a pz2 matrix (``body`` + ``digest``) or a stored blob (``data``)."""

    name: str
    kind: str  # "matrix" or "stored"
    body: Optional[bytes] = None  # pz2 bytes (matrix)
    data: Optional[bytes] = None  # member bytes as they are in the run directory (stored)
    digest: Optional[bytes] = None  # canonical digest of the matrix
    as_written: bool = False  # store without recompression (bgzip files)
    flags: int = 0  # bit 0 (legacy) clear
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------------------------ readers
def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _read_lines(path: str) -> List[str]:
    """Lines of a text file, gzip or plain (decided by the magic bytes, not the extension)."""
    raw = _read_bytes(path)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8").splitlines()


def _read_mtx(path: str):
    """Coordinate MatrixMarket (plain or gzip) -> (m, n, row, col, val) 0-based int arrays."""
    raw = _read_bytes(path)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    lines = raw.decode("ascii").splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("%"):
        i += 1
    if i >= len(lines):
        raise ValueError("%s: no MatrixMarket size line" % path)
    m, n, nnz = (int(x) for x in lines[i].split()[:3])
    body = lines[i + 1 : i + 1 + nnz]
    if len(body) != nnz:
        raise ValueError("%s: %d entries declared, %d found" % (path, nnz, len(body)))
    if nnz:
        arr = np.array([ln.split()[:3] for ln in body], dtype=np.float64)
        r = arr[:, 0].astype(np.int64) - 1
        c = arr[:, 1].astype(np.int64) - 1
        v = arr[:, 2]
        if (r < 0).any() or (r >= m).any() or (c < 0).any() or (c >= n).any():
            raise ValueError("%s: an entry is outside the declared %d x %d shape" % (path, m, n))
        if (v < 0).any() or (v != np.floor(v)).any() or (v > 0xFFFFFFFF).any():
            raise ValueError("%s: values must be non-negative integers below 2^32" % path)
    else:
        r = c = np.zeros(0, np.int64)
        v = np.zeros(0, np.float64)
    return m, n, r, c, v.astype(np.uint64)


def _names(path: str) -> List[str]:
    """First tab-separated field of each line (10x features files have id, name, type)."""
    return [ln.split("\t", 1)[0] for ln in _read_lines(path) if ln != ""]


def aligned_matrix(
    mtx: str, features: str, barcodes: str, cells_barcodes: Sequence[str]
) -> "tuple[Csc, dict]":
    """The run's ``mtx`` re-laid on ``cells_barcodes`` (see the module docstring). Returns the canonical :class:`Csc` (rows
    inline names, columns = the cells table) and the alignment counts."""
    import scipy.sparse as sp

    m, n, r, c, v = _read_mtx(mtx)
    rows = _names(features)
    bcs = _names(barcodes)
    if len(rows) != m:
        raise ValueError("%s lists %d features but %s has %d rows" % (features, len(rows), mtx, m))
    if len(bcs) != n:
        raise ValueError("%s lists %d barcodes but %s has %d columns" % (barcodes, len(bcs), mtx, n))
    if len(set(bcs)) != len(bcs):
        raise ValueError("%s: duplicate barcodes" % barcodes)
    cells = list(cells_barcodes)
    pos = {b: j for j, b in enumerate(cells)}
    if len(pos) != len(cells):
        raise ValueError("the cells table has duplicate barcodes")
    to = np.fromiter((pos.get(b, -1) for b in bcs), dtype=np.int64, count=n)
    keep = to[c] >= 0
    r, newc, v = r[keep], to[c][keep], v[keep]
    csc = sp.csc_matrix((v.astype(np.uint64), (r, newc)), shape=(m, len(cells)))
    csc.sum_duplicates()
    csc.eliminate_zeros()
    csc.sort_indices()
    out = Csc(
        m=m,
        n=len(cells),
        indptr=csc.indptr.astype(np.int64),
        indices=csc.indices.astype(np.uint32),
        data=csc.data.astype(np.uint32),
        rownames=rows,
        colnames=cells,
        value_type=VT_U32,
    )
    present = np.zeros(len(cells), bool)
    present[to[to >= 0]] = True
    info = {
        "n_matrix_barcodes": n,
        "n_dropped_barcodes": int((to < 0).sum()),
        "n_empty_columns": int((~present).sum()),
    }
    return out, info


def encode_matrix(name: str, csc: Csc, info: dict, codec=None, chunk_cols: int = 0, level: int = util.ZSTD_LEVEL) -> MemberSpec:
    """pz2 member (inline row names, ``Cells`` column names) of an aligned matrix."""
    codec = codec or get_codec()
    eo = EncodeOptions(chunk_cols=chunk_cols, row_names="inline", col_names="cells", level=level)
    try:
        body = codec.encode(csc, eo)
    except (ImportError, TypeError) as e:  # native module missing or built from older sources (signature mismatch)
        warnings.warn(f"native pz2 encoder unusable ({e}); using the pure-Python codec", RuntimeWarning, stacklevel=2)
        body = PyCodec2().encode(csc, eo)
    meta = {
        "m": csc.m,
        "n": csc.n,
        "nnz": csc.nnz,
        "chunk_cols": parse_header(body)["chunk_cols"],
        "names": {"rows": "inline", "cols": "cells"},
        **info,
    }
    return MemberSpec(name, "matrix", body=body, digest=csc.digest(), meta=meta)


def _json_member(path: str, name: str) -> MemberSpec:
    data = _read_bytes(path)
    json.loads(data.decode("utf-8"))  # must parse; the bytes are stored as written
    return MemberSpec(name, "stored", data=data)


def aligned_positions(path: str, cells_barcodes: Sequence[str]) -> "tuple[bytes, dict]":
    """``spatial_positions.tsv`` with exactly one row per cells-table barcode, in cells-table order. Returns the bytes (the file
    as written when it already satisfies that, else the header plus the rows reordered; extra rows are dropped) and facts."""
    text = _read_bytes(path).decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln != ""]
    if not lines:
        raise ValueError("%s is empty" % path)
    header = lines[0].rstrip("\r").split("\t")
    if "barcode" not in header:
        raise ValueError("%s has no 'barcode' column (columns: %s)" % (path, header))
    bi = header.index("barcode")
    body = [ln.rstrip("\r") for ln in lines[1:]]
    keys = [ln.split("\t")[bi] for ln in body]
    if len(set(keys)) != len(keys):
        raise ValueError("%s: duplicate barcodes" % path)
    cells = list(cells_barcodes)
    if keys == cells:
        return text.encode("utf-8"), {"n_rows": len(cells), "reordered": False, "n_dropped_rows": 0}
    at = {k: i for i, k in enumerate(keys)}
    miss = [b for b in cells if b not in at]
    if miss:
        raise ValueError(
            "%s lacks %d of the %d cells-table barcodes (first: %s)" % (path, len(miss), len(cells), miss[:3])
        )
    out = ["\t".join(header)] + [body[at[b]] for b in cells]
    return ("\n".join(out) + "\n").encode("utf-8"), {
        "n_rows": len(cells),
        "reordered": True,
        "n_dropped_rows": len(keys) - len(cells),
    }


# ------------------------------------------------------------------------------------------------------------- the entry points
def add_section14_members(
    run_dir: str,
    cells_barcodes: Sequence[str],
    *,
    codec=None,
    chunk_cols: int = 0,
    level: int = util.ZSTD_LEVEL,
) -> List[MemberSpec]:
    """Member specs for the section-14 files present in ``run_dir`` (none when the run has none of them).

    ``cells_barcodes`` is the barcode column of the sample's cells table, in column order. A matrix whose ``.mtx.gz`` exists must
    come with its ``_features.tsv.gz`` and ``_barcodes.tsv.gz`` (else ``ValueError``); ``fragments.tsv.gz`` is stored as
    written with its ``.tbi`` when there is one. Order of the result: fragments, peaks, matrices, summaries, spatial."""
    p = lambda f: os.path.join(run_dir, f)  # noqa: E731
    have = lambda f: os.path.isfile(p(f))  # noqa: E731
    out: List[MemberSpec] = []
    if have("fragments.tsv.gz"):
        out.append(MemberSpec(ATAC_FRAGMENTS, "stored", data=_read_bytes(p("fragments.tsv.gz")), as_written=True))
        if have("fragments.tsv.gz.tbi"):
            out.append(
                MemberSpec(ATAC_FRAGMENTS_TBI, "stored", data=_read_bytes(p("fragments.tsv.gz.tbi")), as_written=True)
            )
    if have("peaks.bed"):
        out.append(MemberSpec(ATAC_PEAKS, "stored", data=_read_bytes(p("peaks.bed"))))
    for name, prefix in _MATRICES:
        if not have(prefix + ".mtx.gz"):
            continue
        side = (prefix + "_features.tsv.gz", prefix + "_barcodes.tsv.gz")
        lacking = [f for f in side if not have(f)]
        if lacking:
            raise ValueError("%s: %s present but %s missing" % (run_dir, prefix + ".mtx.gz", lacking))
        csc, info = aligned_matrix(p(prefix + ".mtx.gz"), p(side[0]), p(side[1]), cells_barcodes)
        out.append(encode_matrix(name, csc, info, codec, chunk_cols, level))
    if have(ATAC_SUMMARY):
        out.append(_json_member(p(ATAC_SUMMARY), ATAC_SUMMARY))
    if have(SPATIAL_POSITIONS):
        data, info = aligned_positions(p(SPATIAL_POSITIONS), cells_barcodes)
        out.append(MemberSpec(SPATIAL_POSITIONS, "stored", data=data, meta=info))
    if have(SPATIAL_SUMMARY):
        out.append(_json_member(p(SPATIAL_SUMMARY), SPATIAL_SUMMARY))
    return out


def add_to_writer(w: S2Writer, specs: Sequence[MemberSpec], level: int = util.ZSTD_LEVEL) -> Dict[str, int]:
    """Add the specs to an ``S2Writer`` (before ``build()``). Returns member name -> kind used (1 pz2, 3 zstd, 5 raw)."""
    from . import container as K

    kinds: Dict[str, int] = {}
    for s in specs:
        if s.kind == "matrix":
            w.add_matrix(s.name, s.body, s.digest, s.meta, s.flags)
            kinds[s.name] = K.KIND_PZ2
        else:
            kinds[s.name] = w.add_stored(s.name, s.data, s.meta, s.flags, level, as_written=s.as_written)
    return kinds
