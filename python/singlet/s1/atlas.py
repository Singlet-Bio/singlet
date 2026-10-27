# SPDX-License-Identifier: MIT
"""singlet.s1.atlas -- the atlas-wide cell index and cross-study cell gathering (docs/format-s1-cell-index.md).

::

    from singlet.s1 import atlas
    cells = atlas.find_cells(organism="Homo sapiens", cell_type="CL:0000084", tissue="UBERON:0002048", min_umi=1000)
    n     = atlas.count_cells(organism="Homo sapiens", cell_type="CL:0000084", by=["tissue"])
    adata = atlas.gather(cells.head(5000), modalities=["counts"])      # one streamed request to data.singlet.bio/s1/cidx/gather

Three parts:

* **Format** (spec sections 3-6): SCX1 columnar shards, the directory, sample table, summary and plan blocks, label sidecars.
* **Builder** (:func:`build_index`, ``scripts/s1_cidx_build.py``): manifests + ``cells`` tables + label sidecars -> one index version
  written in the server layout (``s1/cidx/...``); runs on the pilot locally and on Clipper/Anvil for the atlas.
* **Client**: :class:`Index` evaluates queries itself (local directory or ranged reads of the shards) with the same semantics as
  the Worker endpoint ``POST /s1/cidx/query``; :func:`gather` builds an AnnData from the Worker's streamed ``POST /s1/cidx/gather``
  response, or (``via="direct"``) from its own range requests.

A cell is ``(so, col)``: ``so`` = the sample ordinal (u32, assigned once per GSM, never reused), ``col`` = the cell's column in the
sample's ``cells`` table / matrix column order.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import anndata
    import pandas as pd

__all__ = [
    "Index",
    "IndexError_",
    "build_index",
    "find_cells",
    "count_cells",
    "gather",
    "normalize_sidecar",
    "attach_annotations",
    "encode_shard",
    "Shard",
]

CIDX = "s1/cidx"
FORMAT = "singlet-cidx"
FORMAT_VERSION = 1
GROUP_ROWS = 65536  # rows per row group (zone maps, unit of decoding)
SHARD_ROWS = 1 << 21  # rows per shard: ~30 MB, under the data Worker's 32 MiB cache-fill threshold
PLAN_BLOCK = 256  # sample ordinals per plan block
NA8 = 255  # u8 "not available"
SCALE8 = 254  # u8 scores: round(x * 254), 255 = NA


class IndexError_(RuntimeError):
    """Something is wrong with the cell index or with a query against it."""


# ================================================================================================================= SCX1 shards
SCX_MAGIC = b"SCX1"
SCX_VERSION = 1
HEADER = struct.Struct(
    "<4sHHIQII36x"
)  # magic, version, n_cols, n_groups, n_rows, group_rows, header_len (64 B)
COLREC = struct.Struct("<24sB7x")  # name, dtype
GROUPREC = struct.Struct("<QI4x")  # first_row, n_rows
CHUNKREC = struct.Struct("<QIBBHII")  # off, len, enc, bits, 0, min, max
DT_U8, DT_U16, DT_U32 = 1, 2, 3
_NP = {DT_U8: np.uint8, DT_U16: np.uint16, DT_U32: np.uint32}
ENC_CONST, ENC_FOR, ENC_RLE = 0, 1, 2


class ShardError(ValueError):
    pass


def _dtype_code(a: np.ndarray) -> int:
    if a.dtype == np.uint8:
        return DT_U8
    if a.dtype == np.uint16:
        return DT_U16
    if a.dtype == np.uint32:
        return DT_U32
    raise ShardError(f"shard columns are u8/u16/u32, not {a.dtype}")


def _pack_for(v: np.ndarray, bits: int) -> bytes:
    """LSB-first bit packing of ``v`` (u32, all < 2**bits) into little-endian u32 words."""
    n = len(v)
    nwords = -(-n * bits // 32)
    pos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    w = (pos >> np.uint64(5)).astype(np.int64)
    sh = v.astype(np.uint64) << (pos & np.uint64(31))  # at most 63 bits
    acc = np.zeros(nwords + 1, dtype=np.uint64)
    # bit fields never overlap, so OR == ADD; np.bitwise_or.at keeps it exact for 64-bit values
    np.bitwise_or.at(acc, w, sh & np.uint64(0xFFFFFFFF))
    np.bitwise_or.at(acc, w + 1, sh >> np.uint64(32))
    return acc[:nwords].astype("<u4").tobytes()


def _unpack_for(buf, n: int, bits: int) -> np.ndarray:
    if bits == 0:
        return np.zeros(n, dtype=np.uint32)
    words = np.frombuffer(buf, dtype="<u4")
    if len(words) * 32 < n * bits:
        raise ShardError("FOR body too short")
    w64 = np.zeros(len(words) + 1, dtype=np.uint64)
    w64[:-1] = words
    pos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    w = (pos >> np.uint64(5)).astype(np.int64)
    both = w64[w] | (w64[w + 1] << np.uint64(32))
    return ((both >> (pos & np.uint64(31))) & np.uint64((1 << bits) - 1)).astype(np.uint32)


def _encode_chunk(v: np.ndarray):
    """-> (enc, bits, min, max, body), the smallest of CONST / FOR / RLE."""
    v = v.astype(np.uint32, copy=False)
    if len(v) == 0:
        return ENC_CONST, 0, 0, 0, b""
    lo, hi = int(v.min()), int(v.max())
    if lo == hi:
        return ENC_CONST, 0, lo, hi, b""
    bits = int(hi - lo).bit_length()
    for_len = -(-len(v) * bits // 32) * 4
    starts = np.flatnonzero(np.concatenate(([True], v[1:] != v[:-1])))
    if len(starts) * 8 < for_len:
        runs = np.diff(np.append(starts, len(v))).astype(np.uint32)
        body = np.empty(2 * len(starts), dtype="<u4")
        body[0::2] = v[starts]
        body[1::2] = runs
        return ENC_RLE, 0, lo, hi, body.tobytes()
    return ENC_FOR, bits, lo, hi, _pack_for(v - np.uint32(lo), bits)


def encode_shard(cols: Dict[str, np.ndarray], group_rows: int = GROUP_ROWS) -> bytes:
    """Encode equal-length u8/u16/u32 columns as one SCX1 shard (row groups of ``group_rows``)."""
    names = list(cols)
    if not names:
        raise ShardError("a shard needs at least one column")
    n = len(cols[names[0]])
    if any(len(cols[k]) != n for k in names):
        raise ShardError("shard columns differ in length")
    for k in names:
        if len(k.encode()) > 24 or not k.isascii():
            raise ShardError(f"column name {k!r} is not ASCII of at most 24 bytes")
    n_groups = -(-n // group_rows) if n else 0
    header_len = (
        HEADER.size
        + len(names) * COLREC.size
        + n_groups * GROUPREC.size
        + n_groups * len(names) * CHUNKREC.size
    )
    header_len = -(-header_len // 64) * 64
    bodies = bytearray()
    recs = []
    for g in range(n_groups):
        a, b = g * group_rows, min(n, (g + 1) * group_rows)
        for k in names:
            enc, bits, lo, hi, body = _encode_chunk(np.asarray(cols[k][a:b]))
            recs.append(
                CHUNKREC.pack(
                    header_len + len(bodies) if body else 0, len(body), enc, bits, 0, lo, hi
                )
            )
            bodies += body
            bodies += b"\0" * (-len(bodies) % 8)
    out = bytearray(
        HEADER.pack(SCX_MAGIC, SCX_VERSION, len(names), n_groups, n, group_rows, header_len)
    )
    for k in names:
        out += COLREC.pack(k.encode(), _dtype_code(np.asarray(cols[k])))
    for g in range(n_groups):
        out += GROUPREC.pack(g * group_rows, min(n, (g + 1) * group_rows) - g * group_rows)
    for r in recs:
        out += r
    out += b"\0" * (header_len - len(out))
    return bytes(out + bodies)


def scx_header_len(first: bytes) -> int:
    if len(first) < HEADER.size or bytes(first[:4]) != SCX_MAGIC:
        raise ShardError("not an SCX1 shard")
    return HEADER.unpack_from(first, 0)[6]


class Shard:
    """Reader for an SCX1 shard: the whole file in memory, or its header with chunk bodies supplied by ``fetch(off, len)``."""

    def __init__(self, buf, fetch: Optional[Callable[[int, int], Any]] = None):
        mv = memoryview(buf)
        if len(mv) < HEADER.size:
            raise ShardError("shard shorter than its header")
        magic, ver, nc, ng, nr, gr, hl = HEADER.unpack_from(mv, 0)
        if magic != SCX_MAGIC or ver != SCX_VERSION:
            raise ShardError("not an SCX1 shard")
        if len(mv) < hl:
            raise ShardError("shard header truncated")
        self.buf, self.fetch = mv, fetch
        self.n_cols, self.n_groups, self.n_rows, self.group_rows, self.header_len = (
            nc,
            ng,
            nr,
            gr,
            hl,
        )
        p = HEADER.size
        self.names: List[str] = []
        self.dtypes: Dict[str, int] = {}
        for _ in range(nc):
            name, dt = COLREC.unpack_from(mv, p)
            p += COLREC.size
            s = name.rstrip(b"\0").decode()
            self.names.append(s)
            self.dtypes[s] = dt
        self.groups: List[Tuple[int, int]] = []
        for _ in range(ng):
            self.groups.append(GROUPREC.unpack_from(mv, p))
            p += GROUPREC.size
        self.chunks: Dict[Tuple[int, str], tuple] = {}
        for g in range(ng):
            for k in self.names:
                self.chunks[(g, k)] = CHUNKREC.unpack_from(mv, p)
                p += CHUNKREC.size

    def zone(self, g: int, name: str) -> Tuple[int, int]:
        r = self.chunks[(g, name)]
        return r[5], r[6]

    def chunk_range(self, g: int, name: str) -> Tuple[int, int]:
        r = self.chunks[(g, name)]
        return r[0], r[1]

    def column(self, g: int, name: str, body=None) -> np.ndarray:
        off, ln, enc, bits, _z, lo, _hi = self.chunks[(g, name)]
        n = self.groups[g][1]
        dt = _NP[self.dtypes[name]]
        if enc == ENC_CONST:
            return np.full(n, lo, dtype=dt)
        if body is None:
            if off + ln <= len(self.buf):
                body = self.buf[off : off + ln]
            elif self.fetch is not None:
                body = self.fetch(off, ln)
            else:
                raise ShardError("chunk body not in the buffer and no fetch function")
        if enc == ENC_RLE:
            pr = np.frombuffer(body, dtype="<u4").reshape(-1, 2)
            out = np.repeat(pr[:, 0], pr[:, 1])
            if len(out) != n:
                raise ShardError("RLE runs do not add up to the group size")
            return out.astype(dt)
        if enc == ENC_FOR:
            return (_unpack_for(body, n, bits) + np.uint32(lo)).astype(dt)
        raise ShardError(f"unknown encoding {enc}")

    def table(self, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Every row of the given columns (testing / export)."""
        names = list(names or self.names)
        return {
            k: np.concatenate([self.column(g, k) for g in range(self.n_groups)])
            if self.n_groups
            else np.zeros(0, _NP[self.dtypes[k]])
            for k in names
        }


# ================================================================================================================= layout
def d16hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def obj_key(data: bytes, ext: str) -> str:
    """Content-addressed key of an index object: ``s1/cidx/obj/<d16>.<ext>`` (immutable)."""
    return f"{CIDX}/obj/{d16hex(data)}.{ext}"


def version_key(version: str) -> str:
    return f"{CIDX}/v/{version}.json"


CURRENT_KEY = f"{CIDX}/current.json"


def sidecar_key(gsm: str, data: bytes) -> str:
    """``s1/a/<GSM>/<d16>.parquet``: one canonical label sidecar (immutable; referenced from the study manifest)."""
    return f"s1/a/{gsm}/{d16hex(data)}.parquet"


_GSM_RE = re.compile(r"^GSM(\d{1,9})$")
_GSE_RE = re.compile(r"^GSE(\d{1,9})$")


def gsm_num(gsm: str) -> int:
    m = _GSM_RE.match(str(gsm).strip().upper())
    if not m:
        raise IndexError_(f"{gsm!r} is not a GSM accession")
    return int(m.group(1))


def gse_num(gse: str) -> int:
    m = _GSE_RE.match(str(gse).strip().upper())
    if not m:
        raise IndexError_(f"{gse!r} is not a GSE accession")
    return int(m.group(1))


# The shard schema (spec section 4.2). Order is the physical column order; the builder writes exactly these.
SHARD_COLUMNS: Tuple[Tuple[str, int, str], ...] = (
    ("ct", DT_U16, "code:cell_type"),
    ("tissue", DT_U16, "code:tissue"),
    ("sample", DT_U32, "ordinal"),
    ("col", DT_U32, "column"),
    ("umi_exon", DT_U32, "count"),
    ("umi_intron", DT_U32, "count"),
    ("feat_exon", DT_U32, "count"),
    ("feat_intron", DT_U32, "count"),
    ("ct_score", DT_U8, "score"),
    ("sex", DT_U8, "code:sex"),
    ("sex_score", DT_U8, "score"),
    ("anc", DT_U8, "code:ancestry"),
    ("anc_prob", DT_U8, "score"),
)
SORT_KEYS = ("ct", "tissue", "sample", "col")
# query-level field names -> shard columns (``n_umi`` is derived: umi_exon + umi_intron)
RANGE_FIELDS = {
    "n_umi": ("umi_exon", "umi_intron"),
    "n_umi_exon": ("umi_exon",),
    "n_umi_intron": ("umi_intron",),
    "n_genes": ("feat_exon",),
    "n_feat_exon": ("feat_exon",),
    "n_feat_intron": ("feat_intron",),
    "cell_type_score": ("ct_score",),
    "sex_score": ("sex_score",),
    "ancestry_prob": ("anc_prob",),
}
SCORE_FIELDS = {"cell_type_score", "sex_score", "ancestry_prob"}

# chromosomal sex labels of the annotation sidecars -> sex codes (0 = no call)
SEX_DICT = [
    {"code": 0, "id": None, "label": "unknown"},
    {"code": 1, "id": "PATO:0000383", "label": "female"},
    {"code": 2, "id": "PATO:0000384", "label": "male"},
    {"code": 3, "id": None, "label": "ambiguous"},
]
_SEX_CODE = {"female": 1, "male": 2, "ambiguous": 3, "unknown": 0}
ORGANISM_IDS = {
    "Homo sapiens": "NCBITaxon:9606",
    "Mus musculus": "NCBITaxon:10090",
    "Rattus norvegicus": "NCBITaxon:10116",
    "Macaca mulatta": "NCBITaxon:9544",
    "Macaca fascicularis": "NCBITaxon:9541",
    "Danio rerio": "NCBITaxon:7955",
    "Drosophila melanogaster": "NCBITaxon:7227",
    "Sus scrofa": "NCBITaxon:9823",
    "Gallus gallus": "NCBITaxon:9031",
}


def q8(x) -> np.ndarray:
    """Scores in [0, 1] -> u8 ``round(x * 254)``; NaN / missing -> 255."""
    a = np.asarray(x, dtype=np.float64)
    out = np.full(a.shape, NA8, dtype=np.uint8)
    ok = np.isfinite(a)
    out[ok] = np.clip(np.floor(a[ok] * SCALE8 + 0.5), 0, SCALE8).astype(np.uint8)
    return out


def q8_threshold(v: float, upper: bool = False) -> int:
    """The u8 code a threshold compares against: ``floor(v * 254 + 0.5)``, the code the value ``v`` itself is stored as. A range
    keeps cells whose stored code is within the bounds' codes, so no cell whose true score satisfies the range is lost (cells up
    to half a step outside may be kept). ``upper`` is accepted for symmetry; both sides use the same code."""
    return int(np.floor(float(v) * SCALE8 + 0.5))


# ================================================================================================================= request caps and costs
# Per-request caps of the Worker endpoints (spec section 8). Every request is bounded so that its Cloudflare cost stays under
# MAX_COST_USD even when every read misses the cache (docs/format-s1-cell-index-costs.md, section 3).
MAX_CELLS_PER_REQUEST = 10_000
MAX_STUDIES_PER_REQUEST = 10
MAX_COST_USD = 0.01
# Cloudflare list prices, 2026-10-23 (docs/format-s1-cell-index-costs.md section 1)
PRICE = {
    "request": 0.30e-6,  # Worker request
    "cpu_ms": 0.02e-6,  # Worker CPU millisecond
    "class_b": 0.36e-6,  # R2 GetObject
    "do_request": 0.15e-6,  # Durable Object request
    "do_row_write": 1.00e-6,  # Durable Object SQLite row written
}
# CPU model (V8, measured in bench/cidx with margin): decoding index rows, and copying gathered bytes into the stream
CPU_MS_PER_1K_ROWS = 0.25
CPU_MS_PER_MB = 1.0
CPU_MS_FIXED = 5.0


def estimate_cost(
    *,
    reads: int = 0,
    rows: int = 0,
    mb: float = 0.0,
    do_calls: int = 0,
    do_writes: int = 0,
    fills: Optional[int] = None,
) -> dict:
    """Worst-case Cloudflare cost of one request: every read misses the cache and may start one cache fill (another Class B)."""
    fills = reads if fills is None else fills
    cpu = CPU_MS_FIXED + rows / 1000 * CPU_MS_PER_1K_ROWS + mb * CPU_MS_PER_MB
    usd = (
        PRICE["request"]
        + cpu * PRICE["cpu_ms"]
        + (reads + fills) * PRICE["class_b"]
        + do_calls * PRICE["do_request"]
        + do_writes * PRICE["do_row_write"]
    )
    return {
        "reads": int(reads),
        "fills": int(fills),
        "rows": int(rows),
        "mb": round(mb, 3),
        "cpu_ms": round(cpu, 1),
        "usd": usd,
    }


# ================================================================================================================= seeded sampling
_M32 = np.uint64(0xFFFFFFFF)
CHUNK_SALT = 0x5BD1E995


def _mix32(x: np.ndarray) -> np.ndarray:
    """lowbias32 (C. Wellons): a 32-bit integer hash with full avalanche; the Worker computes the same with Math.imul."""
    x = np.asarray(x, dtype=np.uint64) & _M32
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & _M32
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & _M32
    x ^= x >> np.uint64(16)
    return x


def hash32(seed: int, a, b) -> np.ndarray:
    """``mix32(mix32(seed ^ a) ^ b)`` on u32 values: the sampling key of a cell ``(so, col)`` (or of a chunk ``(so, chunk)``)."""
    a = np.asarray(a, dtype=np.uint64) & _M32
    b = np.asarray(b, dtype=np.uint64) & _M32
    return _mix32(_mix32(np.uint64(int(seed) & 0xFFFFFFFF) ^ a) ^ b).astype(np.uint32)


# ================================================================================================================= feature columns
SCALE16 = 65534
NA16 = 65535
MAX_FEATURES = 64
FEATURE_KINDS = ("score", "category", "flag")
_FEATURE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
FEATURES_SCHEMA = (
    "1"  # the singlet_features_schema of feature files (docs CELL-ARCHITECTURE.md section 3)
)


@dataclass
class FeatureSpec:
    """One registered per-cell feature (spec section 4.9), as in the catalog's ``cell_feature`` table.

    ``id`` = ``<feature_set>/<name>`` (or ``name`` without a set) is what queries and exports use; ``column`` is the physical shard
    column (``f1``, ``f2``, ...), assigned by the builder in registry order. Kinds:

    * ``score``: a number in ``[lo, hi]`` (default [0, 1]) as a linear code, u8 ``floor((x - lo)/(hi - lo)*254 + 0.5)`` (255 = NA) or
      u16 (65,534 / 65,535);
    * ``flag``: a boolean stored as a u8 score (true = 254 = 1.0, false = 0, NA = 255);
    * ``category``: a u16 index into ``categories`` (65,535 = NA)."""

    name: str
    feature_set: str = ""
    kind: str = "score"
    dtype: str = "u8"
    lo: float = 0.0
    hi: float = 1.0
    thresholds: Tuple[float, ...] = (0.5, 0.8)
    categories: Tuple[str, ...] = ()
    label: str = ""
    description: str = ""
    method: str = ""
    method_version: str = ""
    source_ref: str = ""
    licence: str = ""
    organisms: Tuple[str, ...] = ()
    ontology_id: str = ""
    comparable: str = "atlas"
    column: str = ""

    @property
    def id(self) -> str:
        return f"{self.feature_set}/{self.name}" if self.feature_set else self.name

    @property
    def version(self) -> str:
        return self.method_version

    @property
    def scale(self) -> int:
        return SCALE8 if self.dtype == "u8" else SCALE16

    @property
    def na(self) -> int:
        return NA8 if self.dtype == "u8" else NA16

    @property
    def np_dtype(self):
        return np.uint8 if self.dtype == "u8" else np.uint16

    def encode(self, x) -> np.ndarray:
        if self.kind == "category":
            a = np.asarray(x, dtype=np.float64)
            out = np.full(a.shape, self.na, dtype=self.np_dtype)
            ok = np.isfinite(a) & (a >= 0) & (a < len(self.categories))
            out[ok] = a[ok].astype(self.np_dtype)
            return out
        a = np.asarray(x, dtype=np.float64)
        out = np.full(a.shape, self.na, dtype=self.np_dtype)
        ok = np.isfinite(a)
        out[ok] = np.floor(
            (np.clip(a[ok], self.lo, self.hi) - self.lo) / (self.hi - self.lo) * self.scale + 0.5
        ).astype(self.np_dtype)
        return out

    def decode(self, codes) -> np.ndarray:
        c = np.asarray(codes)
        if self.kind == "category":
            names = np.array(list(self.categories) + [None], dtype=object)
            return names[np.where(c == self.na, len(self.categories), c).astype(np.int64)]
        out = (self.lo + c.astype(np.float64) / self.scale * (self.hi - self.lo)).astype(np.float32)
        out[c == self.na] = np.nan
        return out

    def bound(self, v: float, upper: bool = False) -> int:
        """Code a range bound compares against (both sides): the code ``v`` itself is stored as, as for scores (``q8_threshold``).
        Out-of-range bounds give codes below 0 or above ``scale``."""
        return int(np.floor((float(v) - self.lo) / (self.hi - self.lo) * self.scale + 0.5))

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "feature_set": self.feature_set,
            "kind": self.kind,
            "dtype": self.dtype,
            "lo": self.lo,
            "hi": self.hi,
            "scale": self.scale,
            "na": self.na,
            "thresholds": list(self.thresholds),
            "categories": list(self.categories),
            "label": self.label,
            "description": self.description,
            "method": self.method,
            "method_version": self.method_version,
            "source_ref": self.source_ref,
            "licence": self.licence,
            "organisms": list(self.organisms),
            "ontology_id": self.ontology_id,
            "comparable": self.comparable,
            "column": self.column,
        }

    def cell_feature_row(self) -> dict:
        """The catalog's ``cell_feature`` row for this feature (storage 'shard')."""
        return {
            "feature_id": self.id,
            "feature_set": self.feature_set,
            "kind": self.kind,
            "label": self.label or self.name,
            "description": self.description,
            "method": self.method,
            "method_version": self.method_version,
            "source_ref": self.source_ref,
            "licence": self.licence,
            "organisms": json.dumps(list(self.organisms)),
            "ontology_id": self.ontology_id,
            "categories": json.dumps(list(self.categories)),
            "thresholds": json.dumps(list(self.thresholds)),
            "comparable": self.comparable,
            "storage": "shard",
        }

    @classmethod
    def from_json(cls, d: dict) -> "FeatureSpec":
        name = str(d.get("name") or "")
        fset = str(d.get("feature_set") or "")
        fid = d.get("feature_id") or d.get("id")
        if fid and not name:
            fset, _, name = str(fid).rpartition("/")
        kind = str(d.get("kind", "score"))
        cats = d.get("categories") or ()
        if isinstance(cats, str):
            cats = json.loads(cats or "[]")
        thr = d.get("thresholds", (0.5, 0.8))
        if isinstance(thr, str):
            thr = json.loads(thr or "[]")
        orgs = d.get("organisms") or ()
        if isinstance(orgs, str):
            orgs = json.loads(orgs or "[]")
        dtype = (
            "u16" if kind == "category" else ("u8" if kind == "flag" else str(d.get("dtype", "u8")))
        )
        return cls(
            name=name,
            feature_set=fset,
            kind=kind,
            dtype=dtype,
            lo=0.0 if kind != "score" else float(d.get("lo", 0.0)),
            hi=1.0 if kind != "score" else float(d.get("hi", 1.0)),
            thresholds=tuple(float(x) for x in thr) if kind == "score" else (),
            categories=tuple(str(x) for x in cats),
            label=str(d.get("label", "")),
            description=str(d.get("description", "")),
            method=str(d.get("method", "")),
            method_version=str(d.get("method_version", d.get("version", ""))),
            source_ref=str(d.get("source_ref", "")),
            licence=str(d.get("licence", "")),
            organisms=tuple(str(x) for x in orgs),
            ontology_id=str(d.get("ontology_id", "")),
            comparable=str(d.get("comparable", "atlas")),
            column=str(d.get("column", "")),
        )


def load_features(src: Union[None, str, Path, Sequence[dict]]) -> List[FeatureSpec]:
    """The feature registry (a JSON file ``{"features": [...]}`` / a list of ``cell_feature``-like dicts), validated, with shard
    columns assigned."""
    if src is None:
        return []
    if isinstance(src, (str, Path)):
        doc = json.loads(Path(src).read_text(encoding="utf-8"))
        items = doc.get("features", doc) if isinstance(doc, dict) else doc
    else:
        items = list(src)
    specs = [FeatureSpec.from_json(d) for d in items]
    if len(specs) > MAX_FEATURES:
        raise IndexError_(f"at most {MAX_FEATURES} features per index version")
    seen = set()
    for i, f in enumerate(specs, start=1):
        if not _FEATURE_NAME.match(f.name) or (
            f.feature_set and not _FEATURE_NAME.match(f.feature_set)
        ):
            raise IndexError_(
                f"feature {f.id!r}: set and name must be [a-z][a-z0-9_]* (<= 48 chars)"
            )
        if f.id in seen or f.name in RANGE_FIELDS or f.id in RANGE_FIELDS:
            raise IndexError_(f"feature {f.id!r} is not unique or clashes with a built-in field")
        if f.kind not in FEATURE_KINDS:
            raise IndexError_(f"feature {f.id}: kind must be one of {FEATURE_KINDS}")
        if f.kind == "category" and not (0 < len(f.categories) < NA16):
            raise IndexError_(f"feature {f.id}: a category feature lists its categories")
        if f.dtype not in ("u8", "u16") or not (
            np.isfinite(f.lo) and np.isfinite(f.hi) and f.hi > f.lo
        ):
            raise IndexError_(f"feature {f.id}: dtype u8/u16 and finite lo < hi are required")
        if len(f.thresholds) > 4:
            raise IndexError_(f"feature {f.id}: at most 4 thresholds")
        if f.comparable not in ("atlas", "sample"):
            raise IndexError_(f"feature {f.id}: comparable is 'atlas' or 'sample'")
        seen.add(f.id)
        f.column = f"f{i}"
    return specs


def _feature_values(t, f: FeatureSpec) -> np.ndarray:
    """One feature column of a feature file as floats (category codes, flags as 0/1, NaN = missing)."""
    c = t.column(f.name)
    if f.kind == "flag":
        vals = c.to_pylist()
        return np.array(
            [np.nan if v is None else (1.0 if v else 0.0) for v in vals], dtype=np.float64
        )
    a = np.asarray(c.to_numpy(zero_copy_only=False), dtype=np.float64)
    return a


def read_feature_sidecar(
    src, specs: Sequence[FeatureSpec], cols: np.ndarray
) -> Tuple[Dict[str, np.ndarray], str]:
    """A feature file of one feature set (``<GSM>.features.<set>.parquet``: ``col`` + one column per feature, named by the feature's
    name) -> ({feature id: codes aligned to ``cols``}, method_version from the file metadata). Features missing from the file, and
    called cells missing from it, are NA; rows for other columns are ignored."""
    t = _read_table(src)
    if "col" not in t.column_names:
        raise IndexError_("feature file has no 'col' column")
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    fs = md.get("feature_set")
    if fs and specs and specs[0].feature_set and fs != specs[0].feature_set:
        raise IndexError_(f"feature file is for set {fs!r}, not {specs[0].feature_set!r}")
    c = np.asarray(t.column("col").to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(np.unique(c)) != len(c):
        raise IndexError_("duplicate 'col' values in a feature file")
    pos = np.searchsorted(cols, c)
    ok = (pos < len(cols)) & (cols[np.minimum(pos, len(cols) - 1)] == c)
    out = {}
    for f in specs:
        v = np.full(len(cols), np.nan)
        if f.name in t.column_names:
            x = _feature_values(t, f)
            if f.kind == "score" and (
                np.nanmin(x, initial=f.lo) < f.lo - 1e-6 or np.nanmax(x, initial=f.hi) > f.hi + 1e-6
            ):
                raise IndexError_(f"feature {f.id}: values outside [{f.lo}, {f.hi}]")
            v[pos[ok]] = x[ok]
        out[f.id] = f.encode(v)
    return out, md.get("method_version", "")


def feature_sidecar_key(gsm: str, feature_set: str, data: bytes) -> str:
    """``s1/cf/<GSM>/<set>.<d16>.parquet``: one canonical feature file (immutable; referenced from the study manifest)."""
    return f"s1/cf/{gsm}/{feature_set}.{d16hex(data)}.parquet"


def normalize_feature_sidecar(
    src,
    *,
    gsm: str,
    specs: Sequence[FeatureSpec],
    n_columns: int,
    called: Optional[np.ndarray] = None,
) -> Tuple[bytes, dict]:
    """A feature job's ``<GSM>.features.<set>.parquet`` -> canonical sidecar bytes + manifest entry ``{key, sha256, bytes, n_rows,
    feature_set, method_version}``: validated (one GSM, unique ``col`` inside the matrix, exactly the registered columns of the set,
    value ranges), called columns only, sorted by ``col``, deterministic."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not specs or len({f.feature_set for f in specs}) != 1:
        raise IndexError_("give the registered features of exactly one feature set")
    fset = specs[0].feature_set
    t = _read_table(src)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    allowed = {"gsm", "col"} | {f.name for f in specs}
    extra = [c for c in t.column_names if c not in allowed]
    if extra:
        raise IndexError_(f"{gsm}: feature file has unregistered columns {extra[:5]}")
    if "gsm" in t.column_names and {x for x in t.column("gsm").to_pylist() if x} - {gsm}:
        raise IndexError_(f"{gsm}: feature file names another sample")
    col = np.asarray(t.column("col").to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(col) and (col.min() < 0 or col.max() >= n_columns):
        raise IndexError_(f"{gsm}: feature column index outside the {n_columns}-column matrix")
    if len(np.unique(col)) != len(col):
        raise IndexError_(f"{gsm}: duplicate 'col' values in the feature file")
    keep = np.ones(len(col), bool) if called is None else np.asarray(called, bool)[col]
    idx = np.flatnonzero(keep)[np.argsort(col[keep], kind="stable")]
    cols = {"col": pa.array(col[idx].astype(np.uint32))}
    for f in specs:
        if f.name not in t.column_names:
            continue
        x = _feature_values(t, f)[idx]
        if f.kind == "score":
            if np.nanmin(x, initial=f.lo) < f.lo - 1e-6 or np.nanmax(x, initial=f.hi) > f.hi + 1e-6:
                raise IndexError_(f"{gsm}: feature {f.id} outside [{f.lo}, {f.hi}]")
            cols[f.name] = pa.array(x.astype(np.float32))
        elif f.kind == "flag":
            cols[f.name] = pa.array([None if np.isnan(v) else bool(v) for v in x], type=pa.bool_())
        else:
            if np.any(np.isfinite(x) & ((x < 0) | (x >= len(f.categories)))):
                raise IndexError_(
                    f"{gsm}: feature {f.id} has a category code outside its {len(f.categories)} categories"
                )
            cols[f.name] = pa.array(np.where(np.isfinite(x), x, NA16).astype(np.uint16))
    meta = {
        "singlet_features_schema": FEATURES_SCHEMA,
        "gsm": gsm,
        "feature_set": fset,
        "method_version": md.get("method_version", specs[0].method_version),
    }
    table = pa.table(cols).replace_schema_metadata(
        {k.encode(): str(v).encode() for k, v in meta.items()}
    )
    bio = io.BytesIO()
    pq.write_table(table, bio, compression="zstd", write_statistics=True)
    out = bio.getvalue()
    entry = {
        "key": feature_sidecar_key(gsm, fset, out),
        "sha256": hashlib.sha256(out).hexdigest(),
        "bytes": len(out),
        "n_rows": int(len(idx)),
        "feature_set": fset,
        "method_version": meta["method_version"],
    }
    return out, entry


def attach_features(manifest: dict, entries: Dict[str, dict]) -> dict:
    """A copy of the study manifest with ``samples[i].features[<set>] = entry`` for each GSM in ``entries`` (one set per call or
    several: ``entries[gsm]`` is one normalize_feature_sidecar entry). Sample objects are untouched."""
    m = json.loads(json.dumps(manifest))
    for s in m.get("samples", []):
        e = entries.get(s["gsm"])
        if e is not None:
            s.setdefault("features", {})[e["feature_set"]] = e
    return m


# ================================================================================================================= ontologies / dictionaries
class Ontology:
    """Terms with labels and parents (is_a, plus part_of for anatomy), from an OBO file or a JSON ``{id: {label, parents}}``."""

    def __init__(self, terms: Optional[Dict[str, dict]] = None):
        self.terms: Dict[str, dict] = terms or {}
        self._anc: Dict[str, Tuple[str, ...]] = {}

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Ontology":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            doc = json.loads(text)
            return cls(
                {
                    k: {"label": v.get("label"), "parents": list(v.get("parents") or [])}
                    for k, v in (doc.get("terms", doc)).items()
                }
            )
        return cls.parse_obo(text)

    @classmethod
    def parse_obo(cls, text: str) -> "Ontology":
        terms: Dict[str, dict] = {}
        cur: Optional[dict] = None
        obsolete = False
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("["):
                if cur is not None and not obsolete:
                    terms[cur["id"]] = cur
                cur = {"id": None, "label": None, "parents": []} if line == "[Term]" else None
                obsolete = False
                continue
            if cur is None or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.split("!")[0].strip()
            if k == "id":
                cur["id"] = v
            elif k == "name":
                cur["label"] = line.split(":", 1)[1].strip()
            elif k == "is_a":
                cur["parents"].append(v.split()[0])
            elif k == "relationship" and v.startswith("part_of "):
                cur["parents"].append(v.split()[1])
            elif k == "is_obsolete" and v == "true":
                obsolete = True
        if cur is not None and not obsolete and cur.get("id"):
            terms[cur["id"]] = cur
        return cls(
            {k: {"label": t["label"], "parents": t["parents"]} for k, t in terms.items() if k}
        )

    def merge(self, other: "Ontology") -> "Ontology":
        t = dict(self.terms)
        t.update(other.terms)
        return Ontology(t)

    def label(self, term: str) -> Optional[str]:
        t = self.terms.get(term)
        return t.get("label") if t else None

    def ancestors(self, term: str) -> Tuple[str, ...]:
        """All ancestors (transitive parents), excluding the term itself, sorted."""
        hit = self._anc.get(term)
        if hit is not None:
            return hit
        seen: set = set()
        stack = list(self.terms.get(term, {}).get("parents", []))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(self.terms.get(p, {}).get("parents", []))
        out = tuple(sorted(seen))
        self._anc[term] = out
        return out

    def primary_path(self, term: str) -> Tuple[str, ...]:
        """Root-to-term path following the first parent: the DFS order key that makes a term's descendants contiguous in a tree."""
        path = [term]
        seen = {term}
        while True:
            ps = self.terms.get(path[-1], {}).get("parents") or []
            ps = [p for p in ps if p not in seen]
            if not ps:
                break
            path.append(ps[0])
            seen.add(ps[0])
        return tuple(reversed(path))


def _merge_anc(anc: List[str], extra: Iterable[str]) -> List[str]:
    """The ontology's ancestors in their order, then any further ones (sorted) the labeller's path names."""
    have = set(anc)
    return anc + sorted(set(extra) - have)


def make_dict(
    terms: Iterable[Optional[str]],
    ontology: Optional[Ontology],
    labels: Optional[Dict[str, str]] = None,
    extra_anc: Optional[Dict[str, Iterable[str]]] = None,
) -> List[dict]:
    """Dictionary of the distinct terms (code 0 = unknown). Codes follow the ontology's primary-parent DFS order (a term and its
    descendants get neighbouring codes), else lexical order. Each entry lists every ancestor so a query term matches its
    descendants."""
    uniq = sorted({t for t in terms if t})
    labels = labels or {}
    if ontology is not None and ontology.terms:

        def key(t):
            return tuple(
                (ontology.label(x) or x).lower() + "\0" + x for x in ontology.primary_path(t)
            )

        uniq.sort(key=key)
    out = [{"code": 0, "id": None, "label": "unknown", "anc": []}]
    for i, t in enumerate(uniq, start=1):
        out.append(
            {
                "code": i,
                "id": t,
                "label": labels.get(t) or (ontology.label(t) if ontology else None) or t,
                "anc": _merge_anc(
                    list(ontology.ancestors(t)) if ontology else [], (extra_anc or {}).get(t, ())
                ),
            }
        )
    if len(out) > 65535:
        raise IndexError_("more than 65,535 dictionary terms")
    return out


def dict_codes(
    entries: Sequence[dict],
    wanted: Sequence[str],
    *,
    descendants: bool = True,
    terms: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Codes of the entries that are a wanted term or (with ``descendants``) have one as an ancestor. A wanted value is an
    ontology id or a label (case-insensitive) of a dictionary entry or of any term in ``terms`` (the directory's id -> label map
    of every ancestor), so ``"T cell"`` also matches CD4/CD8 T cells."""
    want_raw = [str(w).strip() for w in wanted if w is not None and str(w).strip()]
    wl = {w.lower() for w in want_raw}
    ids = set(want_raw)
    for e in entries:
        if e["id"] and (e.get("label") or "").lower() in wl:
            ids.add(e["id"])
    for tid, lab in (terms or {}).items():
        if (lab or "").lower() in wl:
            ids.add(tid)
    out = []
    for e in entries:
        if e["code"] == 0:
            if "unknown" in wl:
                out.append(0)
            continue
        if e["id"] in ids or (descendants and any(a in ids for a in e.get("anc", []))):
            out.append(e["code"])
    return np.array(sorted(out), dtype=np.int64)


def term_labels(dicts: Sequence[Sequence[dict]], ontology: Optional[Ontology]) -> Dict[str, str]:
    """id -> label of every dictionary term and every ancestor they list (so queries can name an ancestor by its label)."""
    out: Dict[str, str] = {}
    for entries in dicts:
        for e in entries:
            if e.get("id"):
                out[e["id"]] = e.get("label") or e["id"]
            for a in e.get("anc", []):
                if a not in out:
                    out[a] = (ontology.label(a) if ontology else None) or a
    return out


# ================================================================================================================= label sidecars
SIDECAR_COLUMNS = (
    ("col", "uint32"),
    ("cell_type_cl", "string"),
    ("cell_type_label", "string"),
    ("cell_type_score", "float32"),
    ("sex_chrom", "string"),
    ("sex_score", "float32"),
    ("ancestry_hancestro", "string"),
    ("ancestry_prob", "float32"),
    ("model_version", "string"),
    # label schema 2 (singlet.celllabels.output): the full CL path and the confidence of each level; absent in schema-1 files
    ("cell_type_path", "string"),
    *((f"cell_type_p{k}", "float32") for k in range(1, 7)),
)
_CL_RE = re.compile(r"^CL:\d{7}$")
_CL_PATH_RE = re.compile(r"^CL:\d{7}(>CL:\d{7}){0,5}$")
_HANC_RE = re.compile(r"^HANCESTRO:\d{4}$")


def _read_table(src) -> "Any":
    import pyarrow.parquet as pq

    if isinstance(src, (bytes, bytearray, memoryview)):
        import pyarrow as pa

        return pq.read_table(pa.BufferReader(bytes(src)))
    return pq.read_table(src)


def normalize_sidecar(
    src,
    *,
    gsm: str,
    n_columns: int,
    called: Optional[np.ndarray] = None,
    cells_digest: Optional[str] = None,
    annot_version: str,
) -> Tuple[bytes, dict]:
    """Interim per-sample label Parquet (the annotation pipeline's ``<GSM>.labels.parquet``) -> canonical sidecar bytes + the
    manifest entry ``{key, sha256, bytes, n_rows, model_version, cells_digest}``.

    Checks: one GSM, ``col`` unique and inside the matrix, labels well formed (CL / HANCESTRO ids, sex in female/male/ambiguous,
    scores in [0, 1]); rows for columns that are not called are dropped (and counted in ``dropped``). Columns are sorted; strings are
    dictionary-encoded; the file metadata records gsm, annot_version and cells_digest. Nothing genotype-like can pass: only these
    columns are written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = _read_table(src)
    names = set(t.column_names)
    if "col" not in names:
        raise IndexError_(f"{gsm}: label file has no 'col' column")
    if "gsm" in names:
        g = set(x for x in t.column("gsm").to_pylist() if x is not None)
        if g and g != {gsm}:
            raise IndexError_(f"{gsm}: label file is for {sorted(g)[:3]}")
    col = np.asarray(t.column("col").to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(col) and (col.min() < 0 or col.max() >= n_columns):
        raise IndexError_(f"{gsm}: label column index outside the {n_columns}-column matrix")
    if len(np.unique(col)) != len(col):
        raise IndexError_(f"{gsm}: duplicate 'col' values in the label file")

    def strcol(name, rx=None, allowed=None):
        if name not in names:
            return [None] * len(col)
        vals = t.column(name).to_pylist()
        for v in vals:
            if v is None:
                continue
            if (rx is not None and not rx.match(v)) or (allowed is not None and v not in allowed):
                raise IndexError_(f"{gsm}: bad {name} value {v!r}")
        return vals

    def fcol(name):
        if name not in names:
            return np.full(len(col), np.nan, dtype=np.float32)
        a = np.asarray(t.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)
        a = np.where(np.isfinite(a), a, np.nan)
        if np.nanmin(a, initial=0.0) < -1e-6 or np.nanmax(a, initial=0.0) > 1 + 1e-6:
            raise IndexError_(f"{gsm}: {name} outside [0, 1]")
        return np.clip(a, 0, 1).astype(np.float32)

    data = {
        "col": col,
        "cell_type_cl": strcol("cell_type_cl", _CL_RE),
        "cell_type_label": strcol("cell_type_label"),
        "cell_type_score": fcol("cell_type_score"),
        "sex_chrom": strcol("sex_chrom", allowed={"female", "male", "ambiguous"}),
        "sex_score": fcol("sex_score"),
        "ancestry_hancestro": strcol("ancestry_hancestro", _HANC_RE),
        "ancestry_prob": fcol("ancestry_prob"),
        "model_version": strcol("model_version"),
        "cell_type_path": strcol("cell_type_path", _CL_PATH_RE),
        **{f"cell_type_p{k}": fcol(f"cell_type_p{k}") for k in range(1, 7)},
    }
    keep = np.ones(len(col), bool)
    if called is not None:
        keep = np.asarray(called, bool)[col]
    order = np.argsort(col[keep], kind="stable")
    idx = np.flatnonzero(keep)[order]
    cols = {}
    for name, typ in SIDECAR_COLUMNS:
        v = data[name]
        if isinstance(v, list):
            cols[name] = pa.array([v[i] for i in idx], type=pa.string()).dictionary_encode()
        else:
            cols[name] = pa.array(
                np.asarray(v)[idx].astype(np.uint32 if typ == "uint32" else np.float32)
            )
    mv = sorted({x for x in data["model_version"] if x})
    meta = {
        "format": "singlet-labels",
        "version": "2",
        "gsm": gsm,
        "annot_version": annot_version,
        "cells_digest": cells_digest or "",
        "model_version": ";".join(mv),
        # cell type levels the API/UI show by default; deeper levels are shipped as previews (docs/cell_annotation.md 3)
        "cell_type_default_levels": "1,2",
        "cell_type_preview_from_level": "3",
    }
    table = pa.table(cols).replace_schema_metadata(
        {k.encode(): str(v).encode() for k, v in meta.items()}
    )
    bio = io.BytesIO()
    pq.write_table(table, bio, compression="zstd", write_statistics=True)
    out = bio.getvalue()
    entry = {
        "key": sidecar_key(gsm, out),
        "sha256": hashlib.sha256(out).hexdigest(),
        "bytes": len(out),
        "n_rows": int(len(idx)),
        "dropped": int(len(col) - len(idx)),
        "model_version": meta["model_version"],
        "cells_digest": cells_digest,
    }
    return out, entry


def attach_annotations(
    manifest: dict, entries: Dict[str, dict], annot_version: str, *, default: bool = True
) -> dict:
    """A copy of the study manifest with ``samples[i].annotations[annot_version] = entry`` for each GSM in ``entries`` (and, with
    ``default``, ``annotations_default = annot_version``). Sample objects are untouched; publishing the manifest is the caller's job."""
    m = json.loads(json.dumps(manifest))
    for s in m.get("samples", []):
        e = entries.get(s["gsm"])
        if e is None:
            continue
        s.setdefault("annotations", {})[annot_version] = {
            k: v for k, v in e.items() if k != "dropped"
        }
    if default:
        m["annotations_default"] = annot_version
    return m


def read_sidecar(src) -> Dict[str, Any]:
    """Canonical (or interim) sidecar -> dict of numpy arrays / lists."""
    t = _read_table(src)
    out: Dict[str, Any] = {}
    for name, typ in SIDECAR_COLUMNS:
        if name not in t.column_names:
            out[name] = None
            continue
        c = t.column(name)
        if typ == "string":
            out[name] = [None if x is None else str(x) for x in c.to_pylist()]
        else:
            out[name] = np.asarray(c.to_numpy(zero_copy_only=False))
    return out


# ================================================================================================================= builder
@dataclass
class _Cat:
    """A per-sample string column: ``cats[codes[i]]`` is row i ("" = no call)."""

    cats: List[str]
    codes: np.ndarray

    def mapped(self, lut: Dict[str, int]) -> np.ndarray:
        return np.array([lut.get(c, 0) if c else 0 for c in self.cats] + [0], dtype=np.int64)[
            self.codes
        ]


@dataclass
class _SampleIn:
    gsm: str
    gse: str
    key: str
    bytes: int
    sha256: Optional[str]
    n_columns: int
    members: dict
    organism: str = ""
    tissue: Optional[str] = None
    tissue_label: Optional[str] = None
    annot: Optional[dict] = None  # manifest annotation entry used
    feat_refs: Dict[str, dict] = field(
        default_factory=dict
    )  # manifest feature sidecars by feature set
    props: Dict[str, Any] = field(
        default_factory=dict
    )  # sample-level properties from --sample-meta
    # filled by _load_sample
    cols: Optional[np.ndarray] = None
    qc: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Optional[Dict[str, Any]] = None
    features: Dict[str, np.ndarray] = field(
        default_factory=dict
    )  # feature id -> codes aligned to cols
    feature_versions: Dict[str, str] = field(
        default_factory=dict
    )  # feature set -> method_version of the file read
    label_version: str = ""
    plan_members: Dict[str, dict] = field(default_factory=dict)
    error: Optional[str] = None
    skip: Optional[str] = None  # left out of the index on purpose (no cells to index), recorded, not an error


def _transport(base: Optional[str]):
    from . import transport as T

    return T.make_transport(base) if base else T.make_transport()


def _get(t, key: str, off: Optional[int] = None, length: Optional[int] = None) -> bytes:
    if off is None:
        return bytes(t.get(key, None, kind="cidx").body)
    if length is None or length <= 0:
        return b""
    return bytes(t.get(key, (off, off + length - 1), kind="cidx").body)


def _matrix_plan(t, sm: _SampleIn, name: str, m: dict, max_col: int) -> Optional[dict]:
    """Plan entry of one column-aligned matrix member: prefix length and the chunk index up to the chunk holding ``max_col``."""
    off, ln = int(m["off"]), int(m["len"])
    if ln < 256:
        return None
    head = _get(t, sm.key, off, 256)
    from singlet import _pz2

    h = _pz2.read_header(head)
    if int(h["n"]) != sm.n_columns:
        return None
    ci_off, ci_len = h["sections"][
        "chunk_index"
    ]  # the header is authoritative; the manifest's chunk_index is a hint
    raw = _get(t, sm.key, off + int(ci_off), int(ci_len))
    rec = np.frombuffer(
        raw, dtype=np.dtype([("off", "<u8"), ("len", "<u4"), ("first", "<u4"), ("nnz", "<u8")])
    )
    nch = len(rec) - 1
    if nch < 0:
        return None
    first = rec["first"].astype(np.int64)
    last = int(np.searchsorted(first, max_col, side="right") - 1) if max_col >= 0 else -1
    k = max(0, min(nch, last + 1))
    sec = h.get("sections") or {}
    chunks_at = off + int(sec["chunks"][0]) if isinstance(sec, dict) and "chunks" in sec else None
    if chunks_at is None:
        raise IndexError_(f"{sm.gsm}: {name}: the pz2 header has no section table")
    return {
        "off": off,
        "len": ln,
        "digest": m.get("digest"),
        "prefix": int(h["prefix_len"]),
        "chunks_at": chunks_at,
        "n_chunks": nch,
        "first_col": first[: k + 1].tolist(),
        "coff": rec["off"][: k + 1].astype(np.int64).tolist(),
        "codec": int(h["codec"]),
        "m": int(h["m"]),
    }


def _load_sample(
    t,
    sm: _SampleIn,
    sidecar_src: Optional[Callable[[_SampleIn], Any]],
    store,
    features: Sequence[FeatureSpec] = (),
    feature_src: Optional[Callable[[_SampleIn, str], Any]] = None,
) -> _SampleIn:
    from .cells import CellsReader

    try:
        cm = sm.members.get("cells")
        if not cm or int(cm.get("kind", 0)) != 2:
            if not any(int(m.get("kind", 0)) == 1 for m in sm.members.values()):
                # a legacy sample without a count matrix (its bundle held only provenance/summary JSON): nothing to index
                sm.skip = "no cells table (the sample object holds no count matrix)"
            else:
                sm.error = "no cells table"
            return sm
        buf = _get(t, sm.key, int(cm["off"]), int(cm["len"]))
        mv = memoryview(buf)
        rd = CellsReader(lambda off, ln: bytes(mv[off : off + ln]), len(mv), store)
        if rd.n_rows != sm.n_columns:
            sm.n_columns = int(rd.n_rows)
        called = np.asarray(rd.column("called"), dtype=bool)
        cols = np.flatnonzero(called).astype(np.uint32)
        sm.cols = cols
        for k in ("n_umi_exon", "n_umi_intron", "n_feat_exon", "n_feat_intron"):
            v = rd.column(k) if k in rd.entries else np.zeros(len(called), np.uint32)
            sm.qc[k] = np.asarray(v, dtype=np.uint32)[cols]
        max_col = int(cols[-1]) if len(cols) else -1
        for name, m in sm.members.items():
            if int(m.get("kind", 0)) == 1 and int(m.get("len", 0)) > 0:
                p = _matrix_plan(t, sm, name, m, max_col)
                if p is not None:
                    sm.plan_members[name] = p
        sm.plan_members["cells"] = {
            "off": int(cm["off"]),
            "len": int(cm["len"]),
            "digest": cm.get("digest"),
        }
        if sidecar_src is not None:
            src = sidecar_src(sm)
            if src is not None:
                lab = read_sidecar(src)
                sm.label_version = next((x for x in (lab.get("model_version") or []) if x), "")
                c = np.asarray(lab["col"], dtype=np.int64)
                pos = np.searchsorted(cols, c)
                ok = (pos < len(cols)) & (cols[np.minimum(pos, len(cols) - 1)] == c)
                # compact per-sample form (an atlas build holds every sample at once): strings as (categories, u16 codes)
                # with "" for no call, scores already as u8
                full: Dict[str, Any] = {}
                for k, v in lab.items():
                    if k in ("col", "model_version") or v is None:
                        continue
                    if isinstance(v, list):
                        arr = np.full(len(cols), "", dtype=object)
                        arr[pos[ok]] = np.array([x or "" for x in v], dtype=object)[ok]
                        cats, inv = np.unique(arr.astype(str), return_inverse=True)
                        full[k] = _Cat(cats.tolist(), inv.astype(np.uint16))
                    else:
                        arr = np.full(len(cols), np.nan, dtype=np.float32)
                        arr[pos[ok]] = np.asarray(v, dtype=np.float32)[ok]
                        full[k] = q8(arr)
                sm.labels = full
        if features:
            sm.features = {f.id: np.full(len(cols), f.na, f.np_dtype) for f in features}
            for fset in dict.fromkeys(f.feature_set for f in features):
                fsrc = feature_src(sm, fset) if feature_src is not None else None
                if fsrc is None:
                    continue
                codes, ver = read_feature_sidecar(
                    fsrc, [f for f in features if f.feature_set == fset], cols.astype(np.int64)
                )
                sm.features.update(codes)
                sm.feature_versions[fset] = ver
    except Exception as e:  # noqa: BLE001 - one bad sample must not stop an atlas build
        sm.error = f"{type(e).__name__}: {e}"
    return sm


def _read_sample_meta(path: Optional[str]) -> Dict[str, dict]:
    """``--sample-meta``: Parquet / CSV / JSON with column gsm and any of organism, tissue (UBERON id), tissue_label."""
    if not path:
        return {}
    p = Path(path)
    if p.suffix.lower() == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        rows = doc if isinstance(doc, list) else [dict(gsm=k, **v) for k, v in doc.items()]
    else:
        import pandas as pd

        df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
        rows = df.to_dict("records")
    out = {}
    for r in rows:
        g = str(r.get("gsm") or r.get("gsm_id") or "").strip().upper()
        if g:
            out[g] = {
                k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()
            }
    return out


def _previous_ordinals(t, base_version: Optional[str]) -> Dict[int, int]:
    """gsm number -> ordinal from an earlier index version (ordinals are never reassigned)."""
    if not base_version:
        return {}
    d = json.loads(_get(t, version_key(base_version)).decode("utf-8"))
    sh = Shard(_get(t, d["samples"]["key"]))
    tb = sh.table(["so", "gsm"])
    return {int(g): int(s) for s, g in zip(tb["so"], tb["gsm"])}


def _qstats(v: np.ndarray, na: Optional[int] = None) -> Tuple[int, int, int, int, int, int]:
    """(n, min, q10, q50, q90, max) of integer values; ``na`` values are left out; quantiles are order statistics
    (``sorted[floor(q * (n - 1))]``), so they are exact stored values and reproducible."""
    x = np.asarray(v)
    if na is not None:
        x = x[x != na]
    n = len(x)
    if n == 0:
        z = 0 if na is None else na
        return 0, z, z, z, z, z
    xs = np.sort(x)
    q = [int(xs[int(np.floor(f * (n - 1)))]) for f in (0.1, 0.5, 0.9)]
    return n, int(xs[0]), q[0], q[1], q[2], int(xs[-1])


SUMMARY_COLUMNS = (
    "gsm_id",
    "gse_id",
    "subject",
    "item",
    "n_cells",
    "n_hi1",
    "n_hi2",
    "q50",
    "q90",
    "mean",
    "version",
)
LABEL_THRESHOLDS = (0.5, 0.8)
DONOR_STATUS = {"healthy": 1, "diseased": 2}
PERTURBATION_BITS = {"treated": 1, "genetic": 2, "control": 4}
SAMPLE_KINDS = {"primary": 1, "cell_line": 2, "stem_cell": 3, "organoid": 4, "xenograft": 5}


def _order_stats(x: np.ndarray) -> Tuple[Any, Any, Any]:
    """(q50, q90, mean) of the values, quantiles as order statistics (``sorted[floor(q (n - 1))]``); empty -> blanks."""
    if not len(x):
        return "", "", ""
    xs = np.sort(x)
    n = len(xs)
    return (
        round(float(xs[int(np.floor(0.5 * (n - 1)))]), 6),
        round(float(xs[int(np.floor(0.9 * (n - 1)))]), 6),
        round(float(xs.mean()), 6),
    )


def _summary_rows(s_: "_SampleIn", fspecs: Sequence[FeatureSpec]) -> List[dict]:
    """``sample_cell_summary`` rows of one sample (catalog table, schema/035 of the website): per cell type / sex / ancestry term
    the cells carry (exact ids, no roll-up) and per feature; thresholds on the label score (0.5, 0.8) or the feature's."""
    rows: List[dict] = []
    lab = s_.labels or {}
    base = {"gsm_id": s_.gsm, "gse_id": s_.gse}
    cut1, cut2 = (q8_threshold(t) for t in LABEL_THRESHOLDS)
    for subject, key, score_key, ids in (
        ("cell_type", "cell_type_cl", "cell_type_score", None),
        ("sex", "sex_chrom", "sex_score", {"female": "PATO:0000383", "male": "PATO:0000384"}),
        ("ancestry", "ancestry_hancestro", "ancestry_prob", None),
    ):
        cat = lab.get(key)
        if cat is None:
            continue
        score = lab.get(score_key)
        for ci, name in enumerate(cat.cats):
            item = (ids or {}).get(name) if ids is not None else name
            if not name or not item:
                continue
            m = cat.codes == ci
            sc = score[m] if score is not None else np.zeros(0, np.uint8)
            sc = sc[sc != NA8]
            q50, q90, mean = _order_stats(sc.astype(np.float64) / SCALE8)
            rows.append(
                {
                    **base,
                    "subject": subject,
                    "item": item,
                    "n_cells": int(m.sum()),
                    "n_hi1": int((sc >= cut1).sum()) if score is not None else "",
                    "n_hi2": int((sc >= cut2).sum()) if score is not None else "",
                    "q50": q50,
                    "q90": q90,
                    "mean": mean,
                    "version": s_.label_version,
                }
            )
    for f in fspecs:
        v = s_.features.get(f.id)
        if v is None:
            continue
        ver = s_.feature_versions.get(f.feature_set) or f.method_version
        have = v[v != f.na]
        if f.kind == "category":
            for ci, cname in enumerate(f.categories):
                k = int((have == ci).sum())
                if k:
                    rows.append(
                        {
                            **base,
                            "subject": "feature",
                            "item": f"{f.id}={cname}",
                            "n_cells": k,
                            "n_hi1": "",
                            "n_hi2": "",
                            "q50": "",
                            "q90": "",
                            "mean": "",
                            "version": ver,
                        }
                    )
            continue
        if not len(have):
            continue
        thr = list(f.thresholds) if f.kind == "score" else [0.5]
        his = [int((have >= f.bound(t)).sum()) for t in thr[:2]] + ["", ""]
        q50, q90, mean = _order_stats(f.decode(have).astype(np.float64))
        rows.append(
            {
                **base,
                "subject": "feature",
                "item": f.id,
                "n_cells": int(len(have)),
                "n_hi1": his[0],
                "n_hi2": his[1],
                "q50": q50,
                "q90": q90,
                "mean": mean,
                "version": ver,
            }
        )
    return rows


@dataclass
class BuildResult:
    version: str
    directory: dict
    objects: Dict[str, bytes]
    stats: dict
    errors: List[dict]
    d1: Dict[str, List[dict]] = field(default_factory=dict)
    skipped: List[dict] = field(default_factory=list)  # samples left out on purpose: {gsm, gse, reason}

    def d1_export(self, out_dir: Union[str, Path]) -> List[str]:
        """Per-sample aggregates for the catalog (tier 1, docs CELL-ARCHITECTURE.md): ``sample_cell_summary.csv`` (the layout of the
        catalog table: gsm_id, gse_id, subject, item, n_cells, n_hi1, n_hi2, q50, q90, mean, version), ``cell_feature.csv`` (the
        feature registry rows) and ``cidx_sample.csv`` (ordinal, QC quantiles and properties per sample). Loading is the catalog's job."""
        import csv

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, rows in self.d1.items():
            p = out / f"{name}.csv"
            cols = list(rows[0].keys()) if rows else []
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            written.append(str(p))
        return written

    def add_extension(
        self, name: str, entry: Any, objects: Optional[Dict[str, bytes]] = None
    ) -> str:
        """Add a top-level directory key (e.g. ``junctions``, built by another module with ``encode_shard``) and its objects, and
        re-derive the version id from the new content. Returns the new version. Readers ignore keys they do not know."""
        if name in self.directory and name not in ("junctions",) and not name.startswith("x_"):
            raise IndexError_(f"directory key {name!r} is taken")
        old = version_key(self.version)
        self.objects.pop(old, None)
        for k, v in (objects or {}).items():
            self.objects[k] = v
        d = dict(self.directory)
        d[name] = entry
        stamp = self.version.rsplit("-", 1)[0]
        created = d.pop("created_at", None)
        d["version"] = None
        digest = hashlib.sha256(
            json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]
        d["version"] = f"{stamp}-{digest}"
        d["created_at"] = created
        self.directory = d
        self.version = d["version"]
        self.objects[version_key(self.version)] = json.dumps(d, separators=(",", ":")).encode()
        for rows in self.d1.values():
            for r in rows:
                if "index_version" in r:
                    r["index_version"] = self.version
        return self.version

    def write(self, root: Union[str, Path], *, set_current: bool = True) -> List[str]:
        """Write every object under ``root`` in the server layout; returns the keys written."""
        root = Path(root)
        keys = []
        for k, v in self.objects.items():
            p = root.joinpath(*k.split("/"))
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists() or p.stat().st_size != len(v):
                p.write_bytes(v)
            keys.append(k)
        if set_current:
            cur = json.dumps(
                {"format": FORMAT, "version": self.version, "directory": version_key(self.version)},
                indent=1,
            ).encode()
            p = root.joinpath(*CURRENT_KEY.split("/"))
            p.write_bytes(cur)
            keys.append(CURRENT_KEY)
        return keys


def _study_gsm_meta(doc: dict) -> dict:
    """``gsm_meta`` of the manifest's ``study_meta.json``, whether the transcoder kept it as inline JSON or stored it (zstd, base64:
    the file is not byte-reproducible with ``json.dumps``, as for most legacy bundles). {} when the study has none or it is unreadable.
    """
    ent = (doc.get("study_files") or {}).get("study_meta.json") or {}
    try:
        if "json" in ent:
            sm = ent["json"]
        elif "zstd_b64" in ent:
            from .manifest import study_file_bytes

            sm = json.loads(study_file_bytes(ent).decode("utf-8"))
        else:
            return {}
        return dict((sm or {}).get("gsm_meta") or {})
    except Exception:  # noqa: BLE001  metadata is optional: the samples are indexed with organism "unknown"
        return {}


def build_index(
    gses: Sequence[str],
    *,
    base: Optional[str] = None,
    sample_meta: Optional[str] = None,
    labels_dir: Optional[str] = None,
    annot_version: Optional[str] = None,
    ontology: Optional[Ontology] = None,
    previous: Optional[str] = None,
    features: Union[None, str, Path, Sequence[dict]] = None,
    features_dir: Optional[str] = None,
    threads: int = 8,
    shard_rows: int = SHARD_ROWS,
    group_rows: int = GROUP_ROWS,
    parquet: bool = False,
    now: Optional[str] = None,
    log: Callable[[str], None] = lambda s: None,
) -> BuildResult:
    """Build one index version from the .s1 manifests of ``gses`` (read through ``base``: a directory with the server layout or a
    URL; default ``SINGLET_DATA_BASE``).

    Labels per sample come from, in this order: ``labels_dir/<GSM>.labels.parquet`` (or ``<GSE>/<GSM>.labels.parquet``, the
    annotation pipeline's output), else the manifest's ``annotations[annot_version or annotations_default]`` sidecar. Sample
    metadata (organism, tissue) from ``sample_meta``, else the manifest's ``study_meta.json``. ``previous`` (an index version)
    keeps its sample ordinals. ``features`` is the feature registry (section 4.9); values come from
    ``features_dir/<GSM>.features.parquet`` (or ``<GSE>/<GSM>.features.parquet``)."""
    from .packs import PackStore

    t = _transport(base)
    store = PackStore()
    meta = _read_sample_meta(sample_meta)
    ont = ontology or Ontology()
    fspecs = load_features(features)
    samples: List[_SampleIn] = []
    for gse in gses:
        try:
            doc = json.loads(_get(t, f"s1/m/{gse}.json").decode("utf-8"))
        except Exception:  # DUAL: studies still published under the old v2/m/ prefix
            doc = json.loads(_get(t, f"v2/m/{gse}.json").decode("utf-8"))
        gm = _study_gsm_meta(doc)
        av = annot_version or doc.get("annotations_default")
        for sd in doc["samples"]:
            g = sd["gsm"]
            mm = meta.get(g, {})
            samples.append(
                _SampleIn(
                    gsm=g,
                    gse=doc["gse"],
                    key=sd["key"],
                    bytes=int(sd["bytes"]),
                    sha256=sd.get("sha256"),
                    n_columns=int(sd.get("n_columns") or 0),
                    members=sd.get("members") or {},
                    organism=str(mm.get("organism") or gm.get(g, {}).get("organism") or ""),
                    tissue=mm.get("tissue") or None,
                    tissue_label=mm.get("tissue_label") or None,
                    annot=(sd.get("annotations") or {}).get(av) if av else None,
                    feat_refs=sd.get("features") or {},
                    props={
                        k: mm.get(k)
                        for k in (
                            "donor_status",
                            "perturbation",
                            "sample_kind",
                            "disease",
                            "disease_label",
                            "development_stage",
                            "development_stage_label",
                        )
                        if mm.get(k)
                    },
                )
            )

    def sidecar_src(sm: _SampleIn):
        if labels_dir:
            for p in (
                Path(labels_dir) / f"{sm.gsm}.labels.parquet",
                Path(labels_dir) / sm.gse / f"{sm.gsm}.labels.parquet",
            ):
                if p.exists():
                    return str(p)
        if sm.annot and sm.annot.get("key"):
            data = _get(t, sm.annot["key"])
            if sm.annot.get("sha256") and hashlib.sha256(data).hexdigest() != sm.annot["sha256"]:
                raise IndexError_(f"{sm.gsm}: label sidecar {sm.annot['key']} fails its checksum")
            return data
        return None

    def feature_src(sm: _SampleIn, fset: str):
        if features_dir:
            names = (
                [f"{sm.gsm}.features.{fset}.parquet"] if fset else [f"{sm.gsm}.features.parquet"]
            )
            for nm in names:
                for fp in (Path(features_dir) / nm, Path(features_dir) / sm.gse / nm):
                    if fp.exists():
                        return str(fp)
        ref = sm.feat_refs.get(fset)
        if ref and ref.get("key"):
            data = _get(t, ref["key"])
            if ref.get("sha256") and hashlib.sha256(data).hexdigest() != ref["sha256"]:
                raise IndexError_(f"{sm.gsm}: feature sidecar {ref['key']} fails its checksum")
            return data
        return None

    t0 = time.time()
    with ThreadPoolExecutor(max(1, threads)) as ex:
        loaded = list(
            ex.map(lambda s: _load_sample(t, s, sidecar_src, store, fspecs, feature_src), samples)
        )
    log(f"read {len(loaded)} samples in {time.time() - t0:.1f}s")
    errors = [{"gsm": s.gsm, "gse": s.gse, "error": s.error} for s in loaded if s.error]
    skipped = [
        {"gsm": s.gsm, "gse": s.gse, "reason": s.skip} for s in loaded if s.skip and not s.error
    ]
    if skipped:
        log(f"{len(skipped)} samples left out of the index (no cells to index): {skipped[0]}")
    ok = [s for s in loaded if not s.error and not s.skip and s.cols is not None]

    # ---- ordinals: keep earlier ones, append the rest in (GSE, GSM) order
    prev = _previous_ordinals(t, previous)
    nxt = (max(prev.values()) + 1) if prev else 1
    ok.sort(key=lambda s: (gse_num(s.gse), gsm_num(s.gsm)))
    so_of: Dict[str, int] = {}
    for s in ok:
        gn = gsm_num(s.gsm)
        if gn in prev:
            so_of[s.gsm] = prev[gn]
        else:
            so_of[s.gsm] = nxt
            nxt += 1

    # ---- dictionaries
    def lab_list(s, k):
        c = None if s.labels is None else s.labels.get(k)
        return [] if c is None else [x for x in c.cats if x]

    ct_labels: Dict[str, str] = {}
    for s in ok:
        a_, b_ = (s.labels or {}).get("cell_type_cl"), (s.labels or {}).get("cell_type_label")
        if a_ is not None and b_ is not None:
            pairs = np.unique(
                np.stack([a_.codes.astype(np.int64), b_.codes.astype(np.int64)], 1), axis=0
            )
            for i, j in ((int(x), int(y)) for x, y in pairs):
                a, b = a_.cats[i], b_.cats[j]
                if a and b and a not in ct_labels:
                    ct_labels[a] = b
    # the labeller's guide path (label schema 2: "CL:a>CL:b>CL:c") names every level of a call, so a query for a compartment or a
    # lineage matches its deeper calls even without an ontology file
    ct_anc: Dict[str, set] = {}
    for s in ok:
        for path in lab_list(s, "cell_type_path"):
            parts = path.split(">")
            ct_anc.setdefault(parts[-1], set()).update(parts[:-1])
    d_ct = make_dict((x for s in ok for x in lab_list(s, "cell_type_cl")), ont, ct_labels, ct_anc)
    d_tissue = make_dict(
        (s.tissue for s in ok),
        ont,
        {s.tissue: s.tissue_label for s in ok if s.tissue and s.tissue_label},
    )
    d_anc = make_dict((x for s in ok for x in lab_list(s, "ancestry_hancestro")), ont)
    d_disease = make_dict(
        (s.props.get("disease") for s in ok),
        ont,
        {
            s.props["disease"]: s.props.get("disease_label")
            for s in ok
            if s.props.get("disease") and s.props.get("disease_label")
        },
    )
    d_stage = make_dict(
        (s.props.get("development_stage") for s in ok),
        ont,
        {
            s.props["development_stage"]: s.props.get("development_stage_label")
            for s in ok
            if s.props.get("development_stage") and s.props.get("development_stage_label")
        },
    )
    orgs = sorted({s.organism for s in ok if s.organism})
    d_org = [{"code": 0, "id": None, "label": "unknown", "anc": []}] + [
        {"code": i, "id": ORGANISM_IDS.get(o, o), "label": o, "anc": []}
        for i, o in enumerate(orgs, start=1)
    ]
    if len(d_anc) > 255:
        raise IndexError_("more than 254 ancestry terms")
    code = {
        "ct": {e["id"]: e["code"] for e in d_ct},
        "tissue": {e["id"]: e["code"] for e in d_tissue},
        "anc": {e["id"]: e["code"] for e in d_anc},
        "org": {e["label"]: e["code"] for e in d_org},
    }

    # ---- rows per organism
    objects: Dict[str, bytes] = {}
    partitions = []
    postings: Dict[int, List[List[int]]] = {so_of[s.gsm]: [] for s in ok}
    summary_rows: List[Tuple[int, ...]] = []
    shard_list: List[str] = []  # global shard index -> key
    n_rows_total = 0
    for org in sorted({s.organism for s in ok}, key=lambda o: code["org"].get(o, 0)):
        part = [s for s in ok if s.organism == org]
        colspec = list(SHARD_COLUMNS) + [
            (f.column, DT_U8 if f.dtype == "u8" else DT_U16, f"feature:{f.id}") for f in fspecs
        ]
        parts: Dict[str, List[np.ndarray]] = {k: [] for k, _, _ in colspec}
        for s in part:
            n = len(s.cols)
            so = so_of[s.gsm]
            lab = s.labels or {}

            def codes(key, dk):
                v = lab.get(key)
                return np.zeros(n, np.int64) if v is None else v.mapped(code[dk])

            sexv = lab.get("sex_chrom")
            parts["ct"].append(codes("cell_type_cl", "ct").astype(np.uint16))
            parts["tissue"].append(
                np.full(n, code["tissue"].get(s.tissue, 0) if s.tissue else 0, np.uint16)
            )
            parts["sample"].append(np.full(n, so, np.uint32))
            parts["col"].append(s.cols.astype(np.uint32))
            parts["umi_exon"].append(s.qc["n_umi_exon"])
            parts["umi_intron"].append(s.qc["n_umi_intron"])
            parts["feat_exon"].append(s.qc["n_feat_exon"])
            parts["feat_intron"].append(s.qc["n_feat_intron"])
            parts["ct_score"].append(
                lab["cell_type_score"]
                if lab.get("cell_type_score") is not None
                else np.full(n, NA8, np.uint8)
            )
            parts["sex"].append(
                sexv.mapped(_SEX_CODE).astype(np.uint8)
                if sexv is not None
                else np.zeros(n, np.uint8)
            )
            parts["sex_score"].append(
                lab["sex_score"] if lab.get("sex_score") is not None else np.full(n, NA8, np.uint8)
            )
            parts["anc"].append(codes("ancestry_hancestro", "anc").astype(np.uint8))
            parts["anc_prob"].append(
                lab["ancestry_prob"]
                if lab.get("ancestry_prob") is not None
                else np.full(n, NA8, np.uint8)
            )
            for f in fspecs:
                parts[f.column].append(s.features.get(f.id, np.full(n, f.na, f.np_dtype)))
        cols = {
            k: (np.concatenate(v) if v else np.zeros(0, _NP[dt])).astype(_NP[dt])
            for (k, dt, _), v in zip(colspec, parts.values())
        }
        order = np.lexsort(tuple(cols[k] for k in reversed(SORT_KEYS)))
        cols = {k: v[order] for k, v in cols.items()}
        n_part = len(order)
        # summary: (sample, ct, sex, anc) counts
        if n_part:
            keyarr = np.stack(
                [
                    cols["sample"].astype(np.int64),
                    cols["ct"].astype(np.int64),
                    cols["sex"].astype(np.int64),
                    cols["anc"].astype(np.int64),
                ],
                1,
            )
            uq, cnt = np.unique(keyarr, axis=0, return_counts=True)
            for (so, ct, sx, an), c in zip(uq.tolist(), cnt.tolist()):
                summary_rows.append((code["org"].get(org, 0), ct, sx, an, so, c))
        shards = []
        for a in range(0, n_part, shard_rows):
            b = min(n_part, a + shard_rows)
            sc = {k: v[a:b] for k, v in cols.items()}
            blob = encode_shard(sc, group_rows)
            key = obj_key(blob, "scx")
            objects[key] = blob
            gi = len(shard_list)
            shard_list.append(key)
            zone = {k: [int(v.min()), int(v.max())] for k, v in sc.items()}
            tis = np.unique(sc["tissue"])
            shards.append(
                {
                    "i": gi,
                    "key": key,
                    "bytes": len(blob),
                    "rows": b - a,
                    "zone": zone,
                    "tissues": tis.tolist() if len(tis) <= 64 else None,
                }
            )
            # postings: runs of one sample inside this shard (rows are sorted by ct, tissue, sample, col)
            smp = sc["sample"].astype(np.int64)
            starts = np.flatnonzero(np.concatenate(([True], smp[1:] != smp[:-1])))
            ends = np.append(starts[1:], len(smp))
            for st, en in zip(starts.tolist(), ends.tolist()):
                postings[int(smp[st])].append([gi, st, en - st])
        n_rows_total += n_part
        partitions.append(
            {"organism": code["org"].get(org, 0), "label": org, "rows": n_part, "shards": shards}
        )
        if parquet and n_part:
            import pyarrow as pa
            import pyarrow.parquet as pq

            bio = io.BytesIO()
            pq.write_table(
                pa.table(cols),
                bio,
                compression="zstd",
                row_group_size=group_rows * 2,
                write_page_index=True,
            )
            pk = obj_key(bio.getvalue(), "parquet")
            objects[pk] = bio.getvalue()
            partitions[-1]["parquet"] = pk

    # ---- sample table (sample-level predicates and the ordinal registry)
    ok.sort(key=lambda s: so_of[s.gsm])
    sample_cols = {
        "so": np.array([so_of[s.gsm] for s in ok], np.uint32),
        "gsm": np.array([gsm_num(s.gsm) for s in ok], np.uint32),
        "gse": np.array([gse_num(s.gse) for s in ok], np.uint32),
        "organism": np.array([code["org"].get(s.organism, 0) for s in ok], np.uint16),
        "tissue": np.array(
            [code["tissue"].get(s.tissue, 0) if s.tissue else 0 for s in ok], np.uint16
        ),
        "n_columns": np.array([s.n_columns for s in ok], np.uint32),
        "n_called": np.array([len(s.cols) for s in ok], np.uint32),
        "labeled": np.array([1 if s.labels else 0 for s in ok], np.uint8),
        # sample-level properties (catalog sample_property / sample_term), filters donor_status, perturbation, sample_kind,
        # disease, development_stage
        "donor": np.array(
            [DONOR_STATUS.get(str(s.props.get("donor_status", "")).lower(), 0) for s in ok],
            np.uint8,
        ),
        "perturb": np.array(
            [
                sum(
                    b_
                    for k_, b_ in PERTURBATION_BITS.items()
                    if k_ in re.split(r"[;,|/ ]+", str(s.props.get("perturbation", "")).lower())
                )
                for s in ok
            ],
            np.uint8,
        ),
        "kind": np.array(
            [SAMPLE_KINDS.get(str(s.props.get("sample_kind", "")).lower(), 0) for s in ok], np.uint8
        ),
        "disease": np.array(
            [{e["id"]: e["code"] for e in d_disease}.get(s.props.get("disease"), 0) for s in ok],
            np.uint16,
        ),
        "dev_stage": np.array(
            [
                {e["id"]: e["code"] for e in d_stage}.get(s.props.get("development_stage"), 0)
                for s in ok
            ],
            np.uint16,
        ),
    }
    # per-sample aggregates (spec 4.5): order statistics of n_umi (exon + intron) and n_genes, and of every feature, plus
    # counts above the registered thresholds; used to prune samples and to rank studies without reading cells
    stat_names = ("min", "q10", "q50", "q90", "max")
    qc_stats = {"umi": [], "genes": []}
    for s_ in ok:
        qc_stats["umi"].append(
            _qstats(s_.qc["n_umi_exon"].astype(np.int64) + s_.qc["n_umi_intron"])[1:]
        )
        qc_stats["genes"].append(_qstats(s_.qc["n_feat_exon"])[1:])
    for k, rows_ in qc_stats.items():
        arr = np.array(rows_, dtype=np.int64).reshape(-1, 5)
        for j, nm in enumerate(stat_names):
            sample_cols[f"{k}_{nm}"] = np.minimum(arr[:, j], 0xFFFFFFFF).astype(np.uint32)
    for f in fspecs:
        st_ = [_qstats(s_.features.get(f.id, np.zeros(0, f.np_dtype)), na=f.na) for s_ in ok]
        arr = np.array(st_, dtype=np.int64).reshape(-1, 6)
        sample_cols[f"{f.column}_n"] = arr[:, 0].astype(np.uint32)
        for j, nm in enumerate(stat_names):
            sample_cols[f"{f.column}_{nm}"] = arr[:, j + 1].astype(f.np_dtype)
        for ti, thr in enumerate(f.thresholds):
            cut = f.bound(thr)
            ge = []
            for s_ in ok:
                v = s_.features.get(f.id, np.zeros(0, f.np_dtype))
                ge.append(int(((v != f.na) & (v >= cut)).sum()))
            sample_cols[f"{f.column}_ge{ti}"] = np.array(ge, np.uint32)
    blob = encode_shard(sample_cols, group_rows)
    samples_key = obj_key(blob, "scx")
    objects[samples_key] = blob

    # ---- summary
    summary_rows.sort()
    sm_arr = np.array(summary_rows, dtype=np.int64).reshape(-1, 6)
    so_tis = dict(zip(sample_cols["so"].tolist(), sample_cols["tissue"].tolist()))
    summary_cols = {
        "organism": sm_arr[:, 0].astype(np.uint16),
        "ct": sm_arr[:, 1].astype(np.uint16),
        "sex": sm_arr[:, 2].astype(np.uint8),
        "anc": sm_arr[:, 3].astype(np.uint8),
        "sample": sm_arr[:, 4].astype(np.uint32),
        "tissue": np.array([so_tis.get(int(x), 0) for x in sm_arr[:, 4]], np.uint16),
        "n": sm_arr[:, 5].astype(np.uint32),
    }
    blob = encode_shard(summary_cols, group_rows)
    summary_key = obj_key(blob, "scx")
    objects[summary_key] = blob

    # ---- plan blocks
    blocks: Dict[int, List[dict]] = {}
    for s in ok:
        so = so_of[s.gsm]
        blocks.setdefault(so // PLAN_BLOCK, []).append(
            {
                "so": so,
                "gsm": s.gsm,
                "gse": s.gse,
                "key": s.key,
                "bytes": s.bytes,
                "n_columns": s.n_columns,
                "n_called": int(len(s.cols)),
                "organism": s.organism,
                "annot": (s.annot or {}).get("key") if not labels_dir else None,
                "members": s.plan_members,
                "runs": postings.get(so, []),
            }
        )
    plan_keys = {}
    for b, lst in sorted(blocks.items()):
        doc = {
            "format": FORMAT + "-plan",
            "version": FORMAT_VERSION,
            "block": b,
            "block_size": PLAN_BLOCK,
            "samples": lst,
        }
        blob = json.dumps(doc, separators=(",", ":")).encode()
        k = obj_key(blob, "json")
        objects[k] = blob
        plan_keys[str(b)] = k

    # ---- directory
    stamp = now or time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    directory = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "version": None,
        "created_at": None,
        "builder": {
            "module": "singlet.s1.atlas",
            "annot_version": annot_version,
            "labels_dir": bool(labels_dir),
        },
        "group_rows": group_rows,
        "shard_rows": shard_rows,
        "columns": [
            {"name": k, "dtype": {DT_U8: "u8", DT_U16: "u16", DT_U32: "u32"}[dt], "kind": kind}
            for k, dt, kind in SHARD_COLUMNS
        ],
        "sort": list(SORT_KEYS),
        "score_scale": SCALE8,
        "na8": NA8,
        "dicts": {
            "organism": d_org,
            "cell_type": d_ct,
            "tissue": d_tissue,
            "sex": SEX_DICT,
            "ancestry": d_anc,
            "disease": d_disease,
            "development_stage": d_stage,
        },
        "properties": {
            "donor_status": DONOR_STATUS,
            "perturbation": PERTURBATION_BITS,
            "sample_kind": SAMPLE_KINDS,
        },
        "terms": term_labels([d_ct, d_tissue, d_anc, d_disease, d_stage], ont),
        "features": [f.to_json() for f in fspecs],
        "sample_stats": {
            "n_umi": "umi",
            "n_genes": "genes",
            **{f.id: f.column for f in fspecs if f.kind != "category"},
        },
        "samples": {
            "key": samples_key,
            "rows": len(ok),
            "max_so": int(max(so_of.values())) if so_of else 0,
        },
        "summary": {"key": summary_key, "rows": int(len(sm_arr))},
        "plan": {"block_size": PLAN_BLOCK, "blocks": plan_keys},
        "shards": shard_list,
        "partitions": partitions,
        "exclude": [],
        "totals": {
            "cells": int(n_rows_total),
            "samples": len(ok),
            "studies": len({s.gse for s in ok}),
            "labeled_samples": int(sample_cols["labeled"].sum()),
        },
    }
    # the version id: build time + a digest of everything the directory says (object keys included), so equal ids mean equal indexes
    del directory["created_at"]
    digest = hashlib.sha256(
        json.dumps(directory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    version = f"{stamp}-{digest}"
    directory["version"] = version
    directory["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dblob = json.dumps(directory, separators=(",", ":")).encode()
    objects[version_key(version)] = dblob
    stats = {
        "samples": len(ok),
        "errors": len(errors),
        "skipped": len(skipped),
        "cells": n_rows_total,
        "shard_bytes": sum(len(objects[k]) for k in shard_list),
        "directory_bytes": len(dblob),
        "plan_bytes": sum(len(objects[k]) for k in plan_keys.values()),
        "seconds": round(time.time() - t0, 2),
        "requests": getattr(t.stats, "requests", None),
        "bytes_read": getattr(t.stats, "bytes", None),
    }
    org_label = {e["code"]: e["label"] for e in d_org}
    tis_id = {e["code"]: e["id"] for e in d_tissue}
    inv = {
        k: {v: k_ for k_, v in m.items()}
        for k, m in (("donor", DONOR_STATUS), ("kind", SAMPLE_KINDS))
    }
    d1_samples = []
    summary_out: List[dict] = []
    for i_, s_ in enumerate(ok):
        r = {
            "gsm_id": s_.gsm,
            "gse_id": s_.gse,
            "so": int(sample_cols["so"][i_]),
            "index_version": version,
            "organism": org_label.get(int(sample_cols["organism"][i_]), ""),
            "tissue": tis_id.get(int(sample_cols["tissue"][i_])) or "",
            "n_called": int(sample_cols["n_called"][i_]),
            "labeled": int(sample_cols["labeled"][i_]),
            "donor_status": inv["donor"].get(int(sample_cols["donor"][i_]), ""),
            "perturbation": ";".join(
                k for k, b_ in PERTURBATION_BITS.items() if int(sample_cols["perturb"][i_]) & b_
            ),
            "sample_kind": inv["kind"].get(int(sample_cols["kind"][i_]), ""),
        }
        for k in ("umi", "genes"):
            for nm in stat_names:
                r[f"{k}_{nm}"] = int(sample_cols[f"{k}_{nm}"][i_])
        d1_samples.append(r)
        summary_out.extend(_summary_rows(s_, fspecs))
    return BuildResult(
        version,
        directory,
        objects,
        stats,
        errors,
        d1={
            "sample_cell_summary": [{k: r_[k] for k in SUMMARY_COLUMNS} for r_ in summary_out],
            "cell_feature": [f.cell_feature_row() for f in fspecs],
            "cidx_sample": d1_samples,
        },
        skipped=skipped,
    )


# ================================================================================================================= query engine
_RANGE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}(/[a-z][a-z0-9_]{0,47})?$")
WHERE_KEYS = (
    "organism",
    "cell_type",
    "tissue",
    "sex",
    "ancestry",
    "samples",
    "studies",
    "donor_status",
    "perturbation",
    "sample_kind",
    "disease",
    "development_stage",
    "ranges",
    "categories",
)
STUDY_RANKS = ("cells", "given", "random")
SAMPLINGS = ("uniform", "chunks")


@dataclass
class Query:
    """A normalised query (spec section 6.1). Lists are OR within a field, fields are ANDed."""

    organism: Optional[str] = None
    cell_type: Optional[List[str]] = None
    tissue: Optional[List[str]] = None
    sex: Optional[List[str]] = None
    ancestry: Optional[List[str]] = None
    samples: Optional[List[str]] = None
    studies: Optional[List[str]] = None
    donor_status: Optional[List[str]] = None
    perturbation: Optional[List[str]] = None
    sample_kind: Optional[List[str]] = None
    disease: Optional[List[str]] = None
    development_stage: Optional[List[str]] = None
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, w: dict) -> "Query":
        def lst(v):
            if v is None:
                return None
            return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]

        bad = [k for k in w if k not in WHERE_KEYS]
        if bad:
            raise IndexError_(f"unknown filter(s) {bad}")
        q = cls(
            organism=w.get("organism"),
            cell_type=lst(w.get("cell_type")),
            tissue=lst(w.get("tissue")),
            sex=lst(w.get("sex")),
            ancestry=lst(w.get("ancestry")),
            samples=lst(w.get("samples")),
            studies=lst(w.get("studies")),
            donor_status=lst(w.get("donor_status")),
            perturbation=lst(w.get("perturbation")),
            sample_kind=lst(w.get("sample_kind")),
            disease=lst(w.get("disease")),
            development_stage=lst(w.get("development_stage")),
        )
        for k, v in (w.get("categories") or {}).items():
            q.categories[str(k)] = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        for k, v in (w.get("ranges") or {}).items():
            if not _RANGE_NAME.match(str(k)):
                raise IndexError_(f"range field {k!r} is not a field name")
            lo, hi = (list(v) + [None, None])[:2] if isinstance(v, (list, tuple)) else (v, None)
            q.ranges[k] = (None if lo is None else float(lo), None if hi is None else float(hi))
        return q


def _bounds(field_name: str, lo: Optional[float], hi: Optional[float]) -> Tuple[int, int]:
    """Integer [lo, hi] bounds on a built-in stored column (scores: u8 codes, NA 255 always excluded)."""
    if field_name in SCORE_FIELDS:
        a = 0 if lo is None else q8_threshold(lo)
        b = SCALE8 if hi is None else min(SCALE8, q8_threshold(hi))
        return a, b
    a = 0 if lo is None else int(np.ceil(lo))
    b = 0xFFFFFFFF * 2 if hi is None else int(np.floor(hi))
    return a, b


class _Plan:
    """What a query needs, resolved against one directory (codes, ranges on columns, sample set, candidate shards)."""

    def __init__(self, idx: "Index", q: Query):
        d = idx.directory
        self.q = q
        dicts = d["dicts"]
        self.terms = d.get("terms") or {}
        self.features = {}  # feature id -> spec (also reachable by its bare name when that is unique)
        specs = [FeatureSpec.from_json(f) for f in d.get("features") or []]
        by_name: Dict[str, List[FeatureSpec]] = {}
        for f in specs:
            self.features[f.id] = f
            by_name.setdefault(f.name, []).append(f)
        self._alias = {
            n: fs[0].id for n, fs in by_name.items() if len(fs) == 1 and n not in self.features
        }
        self.org_codes = None
        if q.organism:
            self.org_codes = set(
                dict_codes(dicts["organism"], [q.organism], descendants=False).tolist()
            )
            for e in dicts["organism"]:
                if e["id"] == q.organism:
                    self.org_codes.add(e["code"])
            if not self.org_codes:
                raise IndexError_(
                    f"organism {q.organism!r} is not in the index ({[e['label'] for e in dicts['organism'][1:]]})"
                )
        self.ct = self._mask(dicts["cell_type"], q.cell_type, 65536)
        self.tissue = self._mask(dicts["tissue"], q.tissue, 65536)
        self.anc = self._mask(dicts["ancestry"], q.ancestry, 256)
        self.sex = None
        if q.sex:
            m = np.zeros(256, bool)
            for s in q.sex:
                c = _SEX_CODE.get(str(s).lower())
                if c is None:
                    raise IndexError_(f"sex must be female, male, ambiguous or unknown, not {s!r}")
                m[c] = True
            self.sex = m
        # ranges as (query name, shard columns, lo code, hi code)
        self.ranges: List[Tuple[str, Tuple[str, ...], int, int]] = []
        for k, (lo, hi) in q.ranges.items():
            if k in RANGE_FIELDS:
                a, b = _bounds(k, lo, hi)
                self.ranges.append((k, RANGE_FIELDS[k], a, b))
            elif self._feature(k) is not None and self._feature(k).kind != "category":  # type: ignore[union-attr]
                f = self._feature(k)
                assert f is not None
                a = 0 if lo is None else max(0, f.bound(lo, upper=False))
                b = f.scale if hi is None else min(f.scale, f.bound(hi, upper=True))
                self.ranges.append((f.id, (f.column,), a, b))
            else:
                raise IndexError_(
                    f"unknown range field {k!r}; use {sorted(RANGE_FIELDS) + sorted(self.features)}"
                )
        # category features: {column: allowed codes}
        self.cats: Dict[str, np.ndarray] = {}
        for k, vals in q.categories.items():
            f = self._feature(k)
            if f is None or f.kind != "category":
                raise IndexError_(f"{k!r} is not a registered category feature")
            unknown = [v for v in vals if v not in f.categories]
            if unknown:
                raise IndexError_(
                    f"{f.id}: unknown categories {unknown[:5]} (use {list(f.categories)[:20]})"
                )
            m = np.zeros(NA16 + 1, bool)
            m[[f.categories.index(v) for v in vals]] = True
            self.cats[f.column] = m
        self.stat_cols = d.get("sample_stats") or {}
        self.st = idx.samples_table()
        self.sample_set: Optional[np.ndarray] = None
        if q.samples or q.studies:
            keep = np.zeros(len(self.st["so"]), bool)
            if q.samples:
                keep |= np.isin(self.st["gsm"], [gsm_num(g) for g in q.samples])
            if q.studies:
                keep |= np.isin(self.st["gse"], [gse_num(g) for g in q.studies])
            self.sample_set = np.sort(self.st["so"][keep].astype(np.int64))
        # sample-level properties (tier-1 facts carried in the sample table) restrict the sample set
        pm = self._property_mask(d)
        if pm is not None:
            allowed = np.sort(self.st["so"][pm].astype(np.int64))
            self.sample_set = (
                allowed if self.sample_set is None else np.intersect1d(self.sample_set, allowed)
            )
        self.exclude = np.array(sorted(d.get("exclude") or []), dtype=np.int64)

    def _feature(self, k: str) -> Optional[FeatureSpec]:
        return self.features.get(k) or self.features.get(self._alias.get(k, ""))

    def _property_mask(self, d: dict) -> Optional[np.ndarray]:
        q, st = self.q, self.st
        props = d.get("properties") or {}
        m = None

        def both(x):
            nonlocal m
            m = x if m is None else (m & x)

        if q.donor_status:
            lut = props.get("donor_status", DONOR_STATUS)
            codes = [
                lut[v.lower()] if v.lower() in lut else (0 if v.lower() == "unknown" else -1)
                for v in q.donor_status
            ]
            if -1 in codes:
                raise IndexError_(f"donor_status is one of {sorted(lut)} or unknown")
            both(np.isin(st["donor"], codes))
        if q.perturbation:
            bits = props.get("perturbation", PERTURBATION_BITS)
            if any(v.lower() not in bits and v.lower() != "unknown" for v in q.perturbation):
                raise IndexError_(f"perturbation is one of {sorted(bits)} or unknown")
            want = sum(bits[v.lower()] for v in q.perturbation if v.lower() in bits)
            x = (st["perturb"].astype(np.int64) & want) != 0
            if any(v.lower() == "unknown" for v in q.perturbation):
                x |= st["perturb"] == 0
            both(x)
        if q.sample_kind:
            lut = props.get("sample_kind", SAMPLE_KINDS)
            codes = [
                lut[v.lower()] if v.lower() in lut else (0 if v.lower() == "unknown" else -1)
                for v in q.sample_kind
            ]
            if -1 in codes:
                raise IndexError_(f"sample_kind is one of {sorted(lut)} or unknown")
            both(np.isin(st["kind"], codes))
        for key, col in (("disease", "disease"), ("development_stage", "dev_stage")):
            wanted = getattr(q, key)
            if wanted:
                entries = d["dicts"].get(key) or [
                    {"code": 0, "id": None, "label": "unknown", "anc": []}
                ]
                both(np.isin(st[col], dict_codes(entries, wanted, terms=self.terms)))
        return m

    def _mask(self, entries, wanted, size) -> Optional[np.ndarray]:
        if not wanted:
            return None
        codes = dict_codes(entries, wanted, terms=self.terms)
        m = np.zeros(size, bool)
        m[codes] = True
        return m

    @staticmethod
    def _overlaps(mask: Optional[np.ndarray], lo: int, hi: int) -> bool:
        return mask is None or bool(mask[lo : hi + 1].any())

    def row_level(self) -> bool:
        return bool(self.ranges or self.cats)

    def _zone_ok(self, z: Callable[[str], Tuple[int, int]]) -> bool:
        if (
            not self._overlaps(self.ct, *z("ct"))
            or not self._overlaps(self.tissue, *z("tissue"))
            or not self._overlaps(self.anc, *z("anc"))
        ):
            return False
        if self.sex is not None and not self.sex[z("sex")[0] : z("sex")[1] + 1].any():
            return False
        for _k, cs, a, b in self.ranges:
            lo = sum(z(c)[0] for c in cs)
            hi = sum(z(c)[1] for c in cs)
            if hi < a or lo > b:
                return False
        for col, m in self.cats.items():
            lo, hi = z(col)
            if not m[lo : hi + 1].any():
                return False
        if self.sample_set is not None:
            lo, hi = z("sample")
            i = np.searchsorted(self.sample_set, lo)
            if i >= len(self.sample_set) or self.sample_set[i] > hi:
                return False
        return True

    def shard_ok(self, sh: dict) -> bool:
        zz = sh["zone"]
        return self._zone_ok(lambda c: tuple(zz.get(c, (0, 0xFFFFFFFF))))  # type: ignore[return-value]

    def group_ok(self, sh: Shard, g: int) -> bool:
        return self._zone_ok(lambda c: sh.zone(g, c))

    def needed_columns(self) -> List[str]:
        need = ["sample"]
        if self.ct is not None:
            need.append("ct")
        if self.tissue is not None:
            need.append("tissue")
        if self.sex is not None:
            need.append("sex")
        if self.anc is not None:
            need.append("anc")
        for _k, cs, _a, _b in self.ranges:
            need.extend(cs)
        need.extend(self.cats)
        return list(dict.fromkeys(need))

    def rows(self, cols: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(cols["sample"])
        m = np.ones(n, bool)
        if self.ct is not None:
            m &= self.ct[cols["ct"]]
        if self.tissue is not None:
            m &= self.tissue[cols["tissue"]]
        if self.sex is not None:
            m &= self.sex[cols["sex"]]
        if self.anc is not None:
            m &= self.anc[cols["anc"]]
        for _k, cs, a, b in self.ranges:
            v = sum(cols[c].astype(np.int64) for c in cs)
            m &= (v >= a) & (v <= b)
        for col, cm in self.cats.items():
            m &= cm[cols[col]]
        s = cols["sample"].astype(np.int64)
        if self.sample_set is not None:
            m &= np.isin(s, self.sample_set)
        if len(self.exclude):
            m &= ~np.isin(s, self.exclude)
        return m

    def samples_ok(self) -> np.ndarray:
        """Mask over the sample table: organism, sample/study set, exclusions, and ranges that the per-sample min/max rule out."""
        st = self.st
        m = np.ones(len(st["so"]), bool)
        if self.org_codes is not None:
            m &= np.isin(st["organism"], list(self.org_codes))
        if self.sample_set is not None:
            m &= np.isin(st["so"].astype(np.int64), self.sample_set)
        if len(self.exclude):
            m &= ~np.isin(st["so"].astype(np.int64), self.exclude)
        for k, _cs, a, b in self.ranges:
            pre = self.stat_cols.get(k)
            if pre and f"{pre}_min" in st:
                lo = st[f"{pre}_min"].astype(np.int64)
                hi = st[f"{pre}_max"].astype(np.int64)
                ok = ~((hi < a) | (lo > b))
                if f"{pre}_n" in st:
                    ok &= st[f"{pre}_n"] > 0
                m &= ok
        return m


def _key_after(
    h: np.ndarray, so: np.ndarray, col: np.ndarray, cur: Tuple[int, int, int]
) -> np.ndarray:
    """Rows whose sampling key (h, so, col) sorts after the cursor."""
    h0, s0, c0 = cur
    h = h.astype(np.int64)
    return (h > h0) | ((h == h0) & ((so > s0) | ((so == s0) & (col > c0))))


class Index:
    """One index version, read from a local directory with the server layout or over HTTP (``data.singlet.bio``).

    Queries run here with the same semantics as the Worker's ``POST /s1/cidx/query`` (spec section 6): the directory prunes
    shards, shard headers prune row groups, only the needed column chunks are read."""

    def __init__(self, directory: dict, transport, *, base: Optional[str] = None):
        self.directory = directory
        self.version = directory["version"]
        self.t = transport
        self.base = base
        self._shards: Dict[str, Shard] = {}
        self._lock = threading.Lock()
        self._samples: Optional[Dict[str, np.ndarray]] = None
        self._summary: Optional[Dict[str, np.ndarray]] = None
        self._plans: Dict[str, dict] = {}
        self.stats = {"requests": 0, "bytes": 0, "groups": 0, "rows_scanned": 0}

    # ---- opening
    @classmethod
    def open(cls, version: str = "current", *, base: Optional[str] = None) -> "Index":
        t = _transport(base)
        if version == "current":
            cur = json.loads(_get(t, CURRENT_KEY).decode("utf-8"))
            version = cur["version"]
        d = json.loads(_get(t, version_key(version)).decode("utf-8"))
        if d.get("format") != FORMAT or int(d.get("format_version", 0)) > FORMAT_VERSION:
            raise IndexError_(
                f"{version}: not a {FORMAT} v{FORMAT_VERSION} directory (upgrade singlet-bio)"
            )
        return cls(d, t, base=base)

    def _read(self, key: str, off: Optional[int] = None, length: Optional[int] = None) -> bytes:
        b = _get(self.t, key, off, length)
        with self._lock:
            self.stats["requests"] += 1
            self.stats["bytes"] += len(b)
        return b

    def shard(self, key: str) -> Shard:
        with self._lock:
            s = self._shards.get(key)
        if s is not None:
            return s
        if self.t.is_local():
            s = Shard(self._read(key))
        else:
            first = self._read(key, 0, 64)
            hl = scx_header_len(first)
            s = Shard(self._read(key, 0, hl), fetch=lambda off, ln, k=key: self._read(k, off, ln))
        with self._lock:
            self._shards[key] = s
        return s

    def samples_table(self) -> Dict[str, np.ndarray]:
        if self._samples is None:
            self._samples = self.shard(self.directory["samples"]["key"]).table()
        return self._samples

    def summary_table(self) -> Dict[str, np.ndarray]:
        if self._summary is None:
            self._summary = self.shard(self.directory["summary"]["key"]).table()
        return self._summary

    @property
    def features(self) -> List[FeatureSpec]:
        return [FeatureSpec.from_json(f) for f in self.directory.get("features") or []]

    def plan_block(self, so: int) -> dict:
        b = str(int(so) // int(self.directory["plan"]["block_size"]))
        key = self.directory["plan"]["blocks"].get(b)
        if key is None:
            raise IndexError_(f"sample ordinal {so} is not in index {self.version}")
        with self._lock:
            hit = self._plans.get(key)
        if hit is None:
            doc = json.loads(self._read(key).decode("utf-8"))
            hit = {int(s["so"]): s for s in doc["samples"]}
            with self._lock:
                self._plans[key] = hit
        return hit

    def sample(self, so: int) -> dict:
        s = self.plan_block(so).get(int(so))
        if s is None:
            raise IndexError_(f"sample ordinal {so} is not in index {self.version}")
        return s

    def gsm_of(self, so: np.ndarray) -> np.ndarray:
        st = self.samples_table()
        pos = np.searchsorted(st["so"], so)
        return np.char.add("GSM", st["gsm"][pos].astype(str))

    def gse_of(self, so: np.ndarray) -> np.ndarray:
        st = self.samples_table()
        pos = np.searchsorted(st["so"], so)
        return st["gse"][pos].astype(np.int64) if len(pos) else np.zeros(0, np.int64)

    def so_of(self, gsms: Sequence[str]) -> np.ndarray:
        st = self.samples_table()
        want = [gsm_num(g) for g in gsms]
        lut = dict(zip(st["gsm"].tolist(), st["so"].tolist()))
        miss = [g for g, w in zip(gsms, want) if w not in lut]
        if miss:
            raise IndexError_(f"{miss[:5]} not in index {self.version}")
        return np.array([lut[w] for w in want], dtype=np.int64)

    # ---- queries
    def _candidate_groups(self, p: _Plan) -> Iterable[Tuple[int, str, Shard, int]]:
        """(shard index, key, shard, group) in directory order."""
        d = self.directory
        use_postings = p.sample_set is not None and len(p.sample_set) <= 2000
        windows: Dict[int, List[Tuple[int, int]]] = {}
        if use_postings:
            assert p.sample_set is not None
            for so in (int(x) for x in p.sample_set):
                for gi, st, n in self.sample(so)["runs"]:
                    windows.setdefault(int(gi), []).append((int(st), int(st) + int(n)))
        for part in d["partitions"]:
            if p.org_codes is not None and part["organism"] not in p.org_codes:
                continue
            for sh in part["shards"]:
                if use_postings and sh["i"] not in windows:
                    continue
                if not p.shard_ok(sh):
                    continue
                s = self.shard(sh["key"])
                for g in range(s.n_groups):
                    g0, gn = s.groups[g]
                    if use_postings and not any(
                        a < g0 + gn and b > g0 for a, b in windows[sh["i"]]
                    ):
                        continue
                    if not p.group_ok(s, g):
                        continue
                    yield sh["i"], sh["key"], s, g

    def _scan(
        self,
        p: _Plan,
        want: Sequence[str],
        *,
        max_rows: Optional[int] = None,
        start: Tuple[int, int] = (0, 0),
    ):
        """Yield (shard index, group, row offsets in group, {col: values}) for matching rows."""
        need = list(dict.fromkeys(list(p.needed_columns()) + list(want)))
        scanned = 0
        for gi, _key, s, g in self._candidate_groups(p):
            if (gi, g) < start:
                continue
            cols = {c: s.column(g, c) for c in need}
            scanned += s.groups[g][1]
            self.stats["groups"] += 1
            self.stats["rows_scanned"] += s.groups[g][1]
            idx = np.flatnonzero(p.rows(cols))
            yield gi, g, idx, {c: cols[c][idx] for c in want}
            if max_rows is not None and scanned >= max_rows:
                return

    def _summary_rows(self, p: _Plan) -> Dict[str, np.ndarray]:
        """Summary rows matching the non-range filters, restricted to samples the per-sample stats do not rule out."""
        sm = self.summary_table()
        m = np.ones(len(sm["n"]), bool)
        if p.org_codes is not None:
            m &= np.isin(sm["organism"], list(p.org_codes))
        if p.ct is not None:
            m &= p.ct[sm["ct"]]
        if p.tissue is not None:
            m &= p.tissue[sm["tissue"]]
        if p.sex is not None:
            m &= p.sex[sm["sex"]]
        if p.anc is not None:
            m &= p.anc[sm["anc"]]
        ok = p.samples_ok()
        m &= np.isin(sm["sample"], p.st["so"][ok])
        return {k: v[m] for k, v in sm.items()}

    def count(self, q: Union[Query, dict], by: Sequence[str] = ()) -> Union[int, "pd.DataFrame"]:
        """Number of cells matching ``q`` (or a DataFrame of counts grouped ``by`` any of organism, cell_type, tissue, sex,
        ancestry, sample, study). Without per-cell ranges this reads only the summary table."""
        q = q if isinstance(q, Query) else Query.from_json(q)
        p = _Plan(self, q)
        by = list(by)
        allowed = {"organism", "cell_type", "tissue", "sex", "ancestry", "sample", "study"}
        bad = [b for b in by if b not in allowed]
        if bad:
            raise IndexError_(f"cannot group by {bad}; use {sorted(allowed)}")
        if not p.row_level():
            cols = self._summary_rows(p)
            weights = cols["n"].astype(np.int64)
        else:
            want = ["sample", "ct", "tissue", "sex", "anc"]
            acc: Dict[str, List[np.ndarray]] = {k: [] for k in want}
            for _gi, _g, _idx, vals in self._scan(p, want):
                for k in want:
                    acc[k].append(vals[k])
            cols = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in acc.items()}
            cols["organism"] = self._org_of(cols["sample"])
            weights = np.ones(len(cols["sample"]), np.int64)
        if not by:
            return int(weights.sum())
        return self._group(cols, weights, by)

    def rank_studies(
        self, q: Union[Query, dict], *, study_rank: str = "cells", seed: int = 0
    ) -> List[dict]:
        """Studies with cells that may match, best first: ``[{gse, eligible_max, samples}]``. ``eligible_max`` counts the cells
        matching every filter except per-cell ranges (an upper bound when ranges are given). ``study_rank``: ``cells`` (most
        eligible cells first, then GSE number), ``given`` (the order of ``q.studies``), ``random`` (seeded)."""
        q = q if isinstance(q, Query) else Query.from_json(q)
        p = _Plan(self, q)
        return self._rank(p, study_rank, seed)

    def _rank(self, p: _Plan, study_rank: str, seed: int) -> List[dict]:
        if study_rank not in STUDY_RANKS:
            raise IndexError_(f"study_rank must be one of {STUDY_RANKS}")
        sm = self._summary_rows(p)
        gse = self.gse_of(sm["sample"].astype(np.int64))
        per: Dict[int, set] = {}
        cells: Dict[int, int] = {}
        for g, so, n in zip(gse.tolist(), sm["sample"].tolist(), sm["n"].tolist()):
            cells[g] = cells.get(g, 0) + int(n)
            per.setdefault(g, set()).add(int(so))
        studies = [
            {"gse": f"GSE{g}", "gse_num": g, "eligible_max": cells[g], "samples": sorted(per[g])}
            for g in cells
            if cells[g] > 0
        ]
        if study_rank == "cells":
            studies.sort(key=lambda r: (-r["eligible_max"], r["gse_num"]))
        elif study_rank == "random":
            hs = (
                hash32(seed, np.array([r["gse_num"] for r in studies], dtype=np.uint64), 0x9E3779B9)
                if studies
                else []
            )
            order = (
                np.lexsort((np.array([r["gse_num"] for r in studies]), np.asarray(hs)))
                if studies
                else []
            )
            studies = [studies[i] for i in order]
        else:
            if not p.q.studies:
                raise IndexError_(
                    "study_rank='given' needs the studies filter, in order of relevance"
                )
            pos = {gse_num(g): i for i, g in enumerate(p.q.studies)}
            studies.sort(key=lambda r: pos.get(r["gse_num"], len(pos)))
        return studies

    def sample_cells(
        self,
        q: Union[Query, dict],
        *,
        n: int = MAX_CELLS_PER_REQUEST,
        seed: int = 0,
        max_studies: int = MAX_STUDIES_PER_REQUEST,
        study_rank: str = "cells",
        sampling: str = "uniform",
        cursor: Optional[Sequence[int]] = None,
        fields: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple["pd.DataFrame", dict]:
        """A seeded sample of at most ``n`` matching cells from at most ``max_studies`` studies (spec section 6.3).

        Studies are ranked by ``study_rank`` and the first ``max_studies`` kept; every matching cell of those studies gets the key
        ``hash32(seed, so, col)`` (``sampling="chunks"``: the key of its chunk, so whole chunks are taken, which reads far fewer
        bytes) and the ``n`` smallest keys after ``cursor`` are returned, sorted by (so, col). Passing the returned cursor gives the
        next, disjoint page of the same uniform sample. Returns (cells, info)."""
        if sampling not in SAMPLINGS:
            raise IndexError_(f"sampling must be one of {SAMPLINGS}")
        if n <= 0 or max_studies <= 0:
            raise IndexError_("n and max_studies must be positive")
        q = q if isinstance(q, Query) else Query.from_json(q)
        p = _Plan(self, q)
        ranked = self._rank(p, study_rank, seed)
        chosen = ranked[:max_studies]
        sel = np.array(sorted({so for r in chosen for so in r["samples"]}), dtype=np.int64)
        e_max = int(sum(r["eligible_max"] for r in chosen))
        info: dict = {
            "version": self.version,
            "n_studies": len(ranked),
            "studies": [
                {"gse": r["gse"], "eligible_max": r["eligible_max"], "selected": i < max_studies}
                for i, r in enumerate(ranked[: max(50, max_studies)])
            ],
            "eligible_max": e_max,
            "seed": int(seed),
            "sampling": sampling,
            "study_rank": study_rank,
        }
        if max_rows is not None and e_max > max_rows:
            info["over_budget"] = {"rows": e_max, "max_rows": max_rows}
        p.sample_set = sel if p.sample_set is None else np.intersect1d(p.sample_set, sel)
        valid = {c for c, _, _ in SHARD_COLUMNS} | {f.column for f in p.features.values()}
        if fields is None:
            fields = [
                "ct",
                "tissue",
                "umi_exon",
                "umi_intron",
                "feat_exon",
                "ct_score",
                "sex",
                "anc",
            ] + [f.column for f in p.features.values()]
        want = list(dict.fromkeys(["sample", "col"] + [f for f in fields if f in valid]))
        acc: Dict[str, List[np.ndarray]] = {k: [] for k in want}
        if len(sel):
            for _gi, _g, _idx, vals in self._scan(p, want):
                for k in want:
                    acc[k].append(vals[k])
        cols = {k: (np.concatenate(v) if v else np.zeros(0, np.uint32)) for k, v in acc.items()}
        so = cols["sample"].astype(np.int64)
        col = cols["col"].astype(np.int64)
        if sampling == "uniform":
            h = hash32(seed, so, col)
        else:
            unit = np.zeros(len(so), np.int64)
            for s_ in np.unique(so):
                fc = self._chunk_cols(int(s_))
                mk = so == s_
                unit[mk] = np.searchsorted(fc, col[mk], side="right") - 1
            h = hash32(seed ^ CHUNK_SALT, so, unit)
        info["eligible"] = int(len(so))
        keep = (
            np.ones(len(so), bool)
            if cursor is None
            else _key_after(h, so, col, tuple(int(x) for x in cursor))
        )  # type: ignore[arg-type]
        idx = np.flatnonzero(keep)
        order = idx[np.lexsort((col[idx], so[idx], h[idx].astype(np.int64)))]
        page = order[:n]
        more = len(order) > n
        info["n"] = int(len(page))
        info["cursor"] = (
            [int(h[page[-1]]), int(so[page[-1]]), int(col[page[-1]])]
            if more and len(page)
            else None
        )
        info["remaining"] = int(len(order) - len(page))
        info["sampled"] = bool(info["eligible"] > len(page))
        out = page[np.lexsort((col[page], so[page]))]
        frame = self._frame({k: v[out] for k, v in cols.items()})
        frame.attrs.update(info)
        return frame, info

    def _chunk_cols(self, so: int) -> np.ndarray:
        """first_col of the chunks of a sample's exon matrix (or its first matrix): the units of ``sampling="chunks"``."""
        mem = self.sample(so)["members"]
        m = mem.get("exon_counts.1pz") or next(
            (v for k, v in sorted(mem.items()) if "first_col" in v), None
        )
        if not m or len(m.get("first_col", [])) < 2:
            return np.array([0, 2**32], dtype=np.int64)
        fc = np.asarray(m["first_col"], dtype=np.int64)
        fc[-1] = 2**32  # the last kept chunk holds every later called column too
        return fc

    def _org_of(self, so: np.ndarray) -> np.ndarray:
        st = self.samples_table()
        pos = np.searchsorted(st["so"], so)
        return st["organism"][pos] if len(so) else np.zeros(0, np.uint16)

    def _group(self, cols, weights, by) -> "pd.DataFrame":
        import pandas as pd

        d = self.directory["dicts"]
        frame = {}
        for b in by:
            if b == "organism":
                frame[b] = cols["organism"].astype(np.int64)
            elif b == "cell_type":
                frame[b] = cols["ct"].astype(np.int64)
            elif b == "tissue":
                frame[b] = cols["tissue"].astype(np.int64)
            elif b == "sex":
                frame[b] = cols["sex"].astype(np.int64)
            elif b == "ancestry":
                frame[b] = cols["anc"].astype(np.int64)
            elif b == "sample":
                frame[b] = cols["sample"].astype(np.int64)
            elif b == "study":
                frame[b] = self.gse_of(cols["sample"].astype(np.int64))
        df = pd.DataFrame(frame)
        df["n_cells"] = weights
        out = (
            df.groupby(by, as_index=False)["n_cells"]
            .sum()
            .sort_values("n_cells", ascending=False, kind="stable")
        )
        names = {
            "organism": d["organism"],
            "cell_type": d["cell_type"],
            "tissue": d["tissue"],
            "sex": SEX_DICT,
            "ancestry": d["ancestry"],
        }
        for b in by:
            if b in names:
                ids = {e["code"]: e["id"] for e in names[b]}
                lab = {e["code"]: e["label"] for e in names[b]}
                out[b + "_label"] = out[b].map(lab)
                out[b] = out[b].map(lambda c, ids=ids, lab=lab: ids.get(c) or lab.get(c))
            elif b == "sample":
                out[b] = self.gsm_of(out[b].to_numpy()) if len(out) else out[b]
            elif b == "study":
                out[b] = "GSE" + out[b].astype(str)
        return out.reset_index(drop=True)

    def find(
        self,
        q: Union[Query, dict],
        *,
        limit: Optional[int] = 100_000,
        fields: Sequence[str] = (
            "ct",
            "tissue",
            "umi_exon",
            "umi_intron",
            "feat_exon",
            "ct_score",
            "sex",
            "anc",
        ),
    ) -> "pd.DataFrame":
        """Every matching cell (no sampling, no caps: for local mirrors and tests) as a DataFrame in index order."""
        q = q if isinstance(q, Query) else Query.from_json(q)
        p = _Plan(self, q)
        valid = {c for c, _, _ in SHARD_COLUMNS} | {f.column for f in p.features.values()}
        want = list(dict.fromkeys(["sample", "col"] + [f for f in fields if f in valid]))
        acc: Dict[str, List[np.ndarray]] = {k: [] for k in want}
        n = 0
        for _gi, _g, _idx, vals in self._scan(p, want):
            for k in want:
                acc[k].append(vals[k])
            n += len(vals["sample"])
            if limit is not None and n >= limit:
                break
        cols = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in acc.items()}
        if limit is not None:
            cols = {k: v[:limit] for k, v in cols.items()}
        return self._frame(cols)

    def _frame(self, cols: Dict[str, np.ndarray]) -> "pd.DataFrame":
        import pandas as pd

        d = self.directory["dicts"]
        feats = {f.column: f for f in self.features}
        so = cols["sample"].astype(np.int64)
        out = {
            "so": so,
            "gsm": self.gsm_of(so) if len(so) else np.zeros(0, str),
            "col": cols["col"].astype(np.int64),
        }
        lut = {
            "ct": ("cell_type", d["cell_type"]),
            "tissue": ("tissue", d["tissue"]),
            "sex": ("sex", SEX_DICT),
            "anc": ("ancestry", d["ancestry"]),
        }
        for k, v in cols.items():
            if k in ("sample", "col"):
                continue
            if k in lut:
                name, entries = lut[k]
                ids = np.array([e["id"] if e["id"] else e["label"] for e in entries], dtype=object)
                out[name] = ids[v.astype(np.int64)] if len(v) else np.zeros(0, object)
            elif k in ("ct_score", "sex_score", "anc_prob"):
                f = v.astype(np.float32) / SCALE8
                f[v == NA8] = np.nan
                out[{"ct_score": "cell_type_score", "anc_prob": "ancestry_prob"}.get(k, k)] = f
            elif k in feats:
                out[feats[k].id] = feats[k].decode(v)
            else:
                out[
                    {
                        "umi_exon": "n_umi_exon",
                        "umi_intron": "n_umi_intron",
                        "feat_exon": "n_feat_exon",
                        "feat_intron": "n_feat_intron",
                    }.get(k, k)
                ] = v.astype(np.int64)
        return pd.DataFrame(out)


# ================================================================================================================= worker client
def _api_base(base: Optional[str]) -> str:
    from . import transport as T

    b = (base or T.data_base()).rstrip("/")
    if not b.startswith(("http://", "https://")):
        raise IndexError_(f"the Worker endpoints need an http(s) base, not {b!r}")
    return b


class OverBudget(IndexError_):
    """The Worker refused a request whose worst-case cost or size exceeds its per-request budget; ``plan`` says how to run it
    elsewhere (client-side, against the immutable index objects)."""

    def __init__(self, message: str, plan: Optional[dict] = None):
        super().__init__(message)
        self.plan = plan or {}


def _post(
    base: Optional[str],
    path: str,
    body: dict,
    *,
    api_key: Optional[str] = None,
    stream: bool = False,
    timeout: float = 300,
):
    url = _api_base(base) + path
    data = json.dumps(body).encode()
    hdr = {"Content-Type": "application/json", "User-Agent": "singlet-bio-atlas"}
    key = api_key or os.environ.get("SINGLET_API_KEY")
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=hdr, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        j: dict = {}
        try:
            j = json.loads(e.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
        msg = j.get("message") or j.get("error")
        if j.get("plan"):
            raise OverBudget(f"{path}: HTTP {e.code}: {msg}", j["plan"]) from e
        raise IndexError_(f"{path}: HTTP {e.code}: {msg or e.reason}") from e
    if stream:
        return resp
    with resp:
        return json.loads(resp.read().decode("utf-8"))


def _query_json(**kw) -> dict:
    w: Dict[str, Any] = {}
    for k in (
        "organism",
        "cell_type",
        "tissue",
        "sex",
        "ancestry",
        "samples",
        "studies",
        "donor_status",
        "perturbation",
        "sample_kind",
        "disease",
        "development_stage",
        "categories",
    ):
        v = kw.get(k)
        if v is not None:
            w[k] = v
    ranges = {k: list(v) for k, v in (kw.get("ranges") or {}).items()}
    for name, fld, side in (
        ("min_umi", "n_umi", 0),
        ("max_umi", "n_umi", 1),
        ("min_genes", "n_genes", 0),
        ("max_genes", "n_genes", 1),
        ("min_cell_type_score", "cell_type_score", 0),
        ("min_ancestry_prob", "ancestry_prob", 0),
    ):
        v = kw.get(name)
        if v is not None:
            cur = list(ranges.get(fld, [None, None]))
            cur[side] = v
            ranges[fld] = cur
    if ranges:
        w["ranges"] = ranges
    return w


def _use_worker(via: str, base: Optional[str]) -> bool:
    from . import transport as T

    b = base or T.data_base()
    if via == "auto":
        return b.startswith(("http://", "https://"))
    return via == "worker"


def find_cells(
    *,
    organism: Optional[str] = None,
    cell_type: Union[None, str, Sequence[str]] = None,
    tissue: Union[None, str, Sequence[str]] = None,
    sex: Union[None, str, Sequence[str]] = None,
    ancestry: Union[None, str, Sequence[str]] = None,
    samples: Optional[Sequence[str]] = None,
    studies: Optional[Sequence[str]] = None,
    min_umi: Optional[int] = None,
    max_umi: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_cell_type_score: Optional[float] = None,
    ranges: Optional[dict] = None,
    donor_status: Union[None, str, Sequence[str]] = None,
    perturbation: Union[None, str, Sequence[str]] = None,
    sample_kind: Union[None, str, Sequence[str]] = None,
    disease: Union[None, str, Sequence[str]] = None,
    development_stage: Union[None, str, Sequence[str]] = None,
    categories: Optional[Dict[str, Sequence[str]]] = None,
    n: int = MAX_CELLS_PER_REQUEST,
    seed: int = 0,
    max_studies: int = MAX_STUDIES_PER_REQUEST,
    study_rank: str = "cells",
    sampling: str = "uniform",
    cursor: Optional[Any] = None,
    version: str = "current",
    base: Optional[str] = None,
    via: str = "auto",
    api_key: Optional[str] = None,
) -> "pd.DataFrame":
    """A seeded sample of at most ``n`` cells matching every filter, from the ``max_studies`` most relevant studies (spec 6.3).

    ``cell_type`` / ``tissue`` / ``ancestry`` take ontology ids (CL, UBERON, HANCESTRO) or labels and match descendants;
    ``ranges`` take any range field or registered feature (``{"hallmark_hypoxia": [0.3, None]}``). The result's ``attrs`` hold
    ``eligible`` (matching cells in the selected studies), ``studies``, ``cursor``: pass ``cursor=df.attrs["cursor"]`` for the
    next, disjoint page of the same sample. Through the Worker every request returns at most 10,000 cells from at most 10
    studies; a larger ``n`` is fetched as several pages (each a request, under the rate limits)."""
    import pandas as pd

    w = _query_json(
        organism=organism,
        cell_type=cell_type,
        tissue=tissue,
        sex=sex,
        ancestry=ancestry,
        samples=samples,
        studies=studies,
        min_umi=min_umi,
        max_umi=max_umi,
        min_genes=min_genes,
        max_genes=max_genes,
        min_cell_type_score=min_cell_type_score,
        ranges=ranges,
        donor_status=donor_status,
        perturbation=perturbation,
        sample_kind=sample_kind,
        disease=disease,
        development_stage=development_stage,
        categories=categories,
    )
    if not _use_worker(via, base):
        df, _info = Index.open(version, base=base).sample_cells(
            w,
            n=n,
            seed=seed,
            max_studies=max_studies,
            study_rank=study_rank,
            sampling=sampling,
            cursor=cursor,
        )
        return df
    frames = []
    got = 0
    info: dict = {}
    while got < n:
        page = min(n - got, MAX_CELLS_PER_REQUEST)
        body = {
            "version": version,
            "where": w,
            "return": "cells",
            "n": page,
            "seed": seed,
            "max_studies": min(max_studies, MAX_STUDIES_PER_REQUEST),
            "study_rank": study_rank,
            "sampling": sampling,
            "cursor": cursor,
        }
        r = _post(base, "/s1/cidx/query", body, api_key=api_key)
        version = r["version"]
        frames.append(pd.DataFrame(r["cells"]))
        got += int(r["n"])
        info = {k: v for k, v in r.items() if k != "cells"}
        cursor = r.get("cursor")
        if not cursor or not r["n"]:
            break
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.attrs.update(info)
    return df


def count_cells(
    *,
    by: Sequence[str] = (),
    version: str = "current",
    base: Optional[str] = None,
    via: str = "auto",
    api_key: Optional[str] = None,
    **filters,
):
    """Number of matching cells, or a DataFrame of counts grouped ``by`` (organism, cell_type, tissue, sex, ancestry, sample, study).
    Same filters as :func:`find_cells`; counts are not sampled."""
    w = _query_json(**filters)
    if not _use_worker(via, base):
        return Index.open(version, base=base).count(w, by=by)
    import pandas as pd

    # a row-scan count stops at the Worker's per-request row budget and returns a cursor: continue and add up
    total = 0
    frames = []
    cursor = None
    while True:
        r = _post(
            base,
            "/s1/cidx/query",
            {"version": version, "where": w, "return": "count", "by": list(by), "cursor": cursor},
            api_key=api_key,
        )
        version = r["version"]  # pin the version the first page used
        total += int(r["count"])
        if by and r.get("groups"):
            frames.append(pd.DataFrame(r["groups"]))
        cursor = r.get("cursor")
        if r.get("complete", True) or not cursor:
            break
    if not by:
        return total
    if not frames:
        return pd.DataFrame(columns=[*by, "n_cells"])
    df = pd.concat(frames, ignore_index=True)
    keys = [c for c in df.columns if c != "n_cells"]
    return (
        df.groupby(keys, as_index=False, dropna=False)["n_cells"]
        .sum()
        .sort_values("n_cells", ascending=False, ignore_index=True)
    )


# ================================================================================================================= gather
GATHER_GAP = 64 << 10  # ranges of one object closer than this share a read
GATHER_READ_MAX = 8 << 20  # longest single read
FRAME = struct.Struct("<IHBBQI")  # sample index, member index, kind, 0, member offset, length
FR_PREFIX, FR_RUN, FR_WHOLE, FR_ERROR, FR_END = 1, 2, 3, 0xFE, 0xFF
GATHER_MAGIC = b"SGA1"

# the modalities= vocabulary of singlet.s1.client (kept in step with it; gathered here: the gene-level ones and raw matrices)
_MOD_MEMBERS = {
    "counts": ("exon_counts.1pz", "intron_counts.1pz"),
    "spliced": ("exon_counts.1pz",),
    "unspliced": ("intron_counts.1pz",),
    "splice_junctions": ("sj_counts.1pz",),
    "mt": ("mt_heteroplasmy.1pz",),
    "vdj": ("vdj_gene_usage.1pz",),
}
_MOD_ALIASES = {
    "x": "counts",
    "exon": "spliced",
    "intron": "unspliced",
    "junctions": "splice_junctions",
    "sj": "splice_junctions",
    "mt_heteroplasmy": "mt",
    "vdj_gene_usage": "vdj",
}
GENE_LEVEL = ("counts", "spliced", "unspliced")


def normalize_modalities(modalities) -> Tuple[str, ...]:
    if modalities is None:
        return ("counts", "spliced", "unspliced")
    items = [modalities] if isinstance(modalities, str) else list(modalities)
    out = []
    for it in items:
        k = _MOD_ALIASES.get(str(it).strip().lower(), str(it).strip().lower())
        if k == "psi":
            raise IndexError_(
                "psi is derived from splice_junctions; gather splice_junctions and compute it (singlet.s1 recipes_psi)"
            )
        if k not in _MOD_MEMBERS:
            raise IndexError_(f"unknown modality {it!r}; gather supports {sorted(_MOD_MEMBERS)}")
        out.append(k)
    return tuple(k for k in _MOD_MEMBERS if k in out)


def members_for(mods: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(m for k in mods for m in _MOD_MEMBERS[k]))


def _runs_for(first_col: Sequence[int], cols: np.ndarray) -> List[Tuple[int, int]]:
    """Chunk-index runs [k0, k1] (inclusive, touching chunks merged) covering ``cols`` (ascending)."""
    fc = np.asarray(first_col, dtype=np.int64)
    k = np.searchsorted(fc, cols, side="right") - 1
    if len(k) and (k.min() < 0 or k.max() >= len(fc) - 1):
        raise IndexError_(
            "a requested column lies outside the indexed chunks (only called cells are indexed)"
        )
    out: List[List[int]] = []
    for c in (int(x) for x in np.unique(k)):
        if out and c <= out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [(a, b) for a, b in out]


def plan_gather(
    samples: Sequence[dict],
    want: Dict[int, np.ndarray],
    members: Sequence[str],
    have_prefix: Iterable[str] = (),
    *,
    gap: int = GATHER_GAP,
    read_max: int = GATHER_READ_MAX,
    whole: Sequence[str] = (),
) -> dict:
    """The same plan the Worker makes (spec 7.2): per sample the member pieces (prefix unless the client has it, chunk runs) and
    the coalesced reads of the sample object. ``samples`` are plan-block entries; ``want[so]`` the ascending columns."""
    have = set(have_prefix)
    out_samples = []
    reads = []
    total = 0
    for i, s in enumerate(samples):
        cols = np.asarray(want[int(s["so"])], dtype=np.int64)
        pieces = []  # (member name, kind, member offset, length, extra)
        mem_out = {}
        for name in list(members) + list(whole):
            m = s["members"].get(name)
            if m is None:
                continue
            if name in whole:
                pieces.append((name, FR_WHOLE, 0, int(m["len"]), None))
                mem_out[name] = {"digest": m.get("digest"), "whole": True, "len": int(m["len"])}
                continue
            runs = []
            for k0, k1 in _runs_for(m["first_col"], cols):
                rel = int(m["chunks_at"]) - int(m["off"]) + int(m["coff"][k0])
                ln = int(m["coff"][k1 + 1]) - int(m["coff"][k0])
                runs.append([int(m["first_col"][k0]), int(m["first_col"][k1 + 1]), rel, ln])
                pieces.append((name, FR_RUN, rel, ln, None))
            send_prefix = m.get("digest") not in have
            if send_prefix:
                pieces.append((name, FR_PREFIX, 0, int(m["prefix"]), None))
            mem_out[name] = {
                "digest": m.get("digest"),
                "prefix": int(m["prefix"]),
                "send_prefix": send_prefix,
                "runs": runs,
                "len": int(m["len"]),
            }
        # coalesce pieces into object reads
        abs_p = sorted(
            (
                (int(s["members"][nm]["off"]) + off, ln, nm, kind, off)
                for nm, kind, off, ln, _ in pieces
                if ln > 0
            )
        )
        cur = None
        for a, ln, nm, kind, off in abs_p:
            if cur and a - cur["end"] <= gap and max(cur["end"], a + ln) - cur["start"] <= read_max:
                cur["end"] = max(cur["end"], a + ln)
                cur["pieces"].append((nm, kind, off, ln, a))
            else:
                cur = {
                    "sample": i,
                    "key": s["key"],
                    "size": int(s["bytes"]),
                    "start": a,
                    "end": a + ln,
                    "pieces": [(nm, kind, off, ln, a)],
                }
                reads.append(cur)
        out_samples.append(
            {
                "i": i,
                "so": int(s["so"]),
                "gsm": s["gsm"],
                "gse": s["gse"],
                "key": s["key"],
                "n_cells": int(len(cols)),
                "members": mem_out,
            }
        )
    total = sum(r["end"] - r["start"] for r in reads)
    return {
        "samples": out_samples,
        "reads": reads,
        "bytes": total,
        "payload": sum(p[3] for r in reads for p in r["pieces"]),
    }


def _cells_by_sample(idx: Index, cells) -> Dict[int, np.ndarray]:
    """Accept a find_cells DataFrame, a list of (gsm or so, cols), or a dict -> {so: ascending unique cols}."""
    out: Dict[int, List[np.ndarray]] = {}
    if hasattr(cells, "columns"):
        df = cells
        so = (
            df["so"].to_numpy(dtype=np.int64)
            if "so" in df.columns
            else idx.so_of(df["gsm"].tolist())
        )
        col = df["col"].to_numpy(dtype=np.int64)
        for s in (int(x) for x in np.unique(so)):
            out[int(s)] = [col[so == s]]
    else:
        items = cells.items() if isinstance(cells, dict) else cells
        for k, cols in items:
            s = int(k) if not isinstance(k, str) else int(idx.so_of([k])[0])
            out.setdefault(s, []).append(np.asarray(cols, dtype=np.int64).ravel())
    return {s: np.unique(np.concatenate(v)) for s, v in out.items()}


class _PrefixCache:
    """On-disk cache of pz2 member prefixes, keyed by the member digest (immutable)."""

    def __init__(self):
        from .client import cache_root

        self.dir = cache_root() / "prefix"

    def path(self, digest: str) -> Path:
        return self.dir / f"{digest}.bin"

    def get(self, digest: Optional[str]) -> Optional[bytes]:
        if not digest:
            return None
        p = self.path(digest)
        try:
            return p.read_bytes()
        except OSError:
            return None

    def put(self, digest: Optional[str], data: bytes) -> None:
        if not digest:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            p = self.path(digest)
            tmp = p.with_suffix(f".part{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, p)
        except OSError:
            pass

    def have(self) -> List[str]:
        try:
            return [p.stem for p in self.dir.glob("*.bin")]
        except OSError:
            return []


def read_gather_stream(fh) -> Tuple[dict, Dict[Tuple[int, str], dict], dict]:
    """Parse an ``SGA1`` stream (spec 7.3) -> (plan header, {(sample i, member): {"prefix": bytes, "runs": {off: bytes}, "whole": bytes}},
    trailer)."""

    def exact(n):
        b = fh.read(n)
        while b is not None and len(b) < n:
            more = fh.read(n - len(b))
            if not more:
                break
            b += more
        if b is None or len(b) != n:
            raise IndexError_("gather stream ended early")
        return b

    if exact(4) != GATHER_MAGIC:
        raise IndexError_("not a gather stream")
    (hl,) = struct.unpack("<I", exact(4))
    head = json.loads(exact(hl).decode("utf-8"))
    names = head["member_names"]
    got: Dict[Tuple[int, str], dict] = {}
    trailer: dict = {}
    while True:
        si, mi, kind, _z, off, ln = FRAME.unpack(exact(FRAME.size))
        if kind == FR_END:
            trailer = json.loads(exact(ln).decode("utf-8")) if ln else {}
            break
        body = exact(ln)
        if kind == FR_ERROR:
            trailer.setdefault("errors", []).append(json.loads(body.decode("utf-8")))
            continue
        ent = got.setdefault((si, names[mi]), {"runs": {}})
        if kind == FR_PREFIX:
            ent["prefix"] = body
        elif kind == FR_RUN:
            ent["runs"][off] = body
        elif kind == FR_WHOLE:
            ent["whole"] = body
        else:
            raise IndexError_(f"unknown gather frame kind {kind}")
    return head, got, trailer


def _fetch_direct(idx: Index, plan: dict, threads: int = 8) -> Dict[Tuple[int, str], dict]:
    """The client-side equivalent of the Worker: one ranged GET per planned read."""
    got: Dict[Tuple[int, str], dict] = {}

    def one(r):
        return r, _get(idx.t, r["key"], r["start"], r["end"] - r["start"])

    with ThreadPoolExecutor(max(1, threads)) as ex:
        for r, buf in ex.map(one, plan["reads"]):
            idx.stats["requests"] += 1
            idx.stats["bytes"] += len(buf)
            for nm, kind, off, ln, a in r["pieces"]:
                body = buf[a - r["start"] : a - r["start"] + ln]
                ent = got.setdefault((r["sample"], nm), {"runs": {}})
                if kind == FR_PREFIX:
                    ent["prefix"] = body
                elif kind == FR_RUN:
                    ent["runs"][off] = body
                else:
                    ent["whole"] = body
    return got


class _GeneSpace:
    """Var axis of one gather: the gene table of the first vocabulary pack seen; other packs translate by gene id."""

    def __init__(self, resolver, genes: Optional[Sequence[str]] = None):
        self.r = resolver
        self.ids = self.names = None
        self.keep = None
        self.want = genes
        self._maps: Dict[str, np.ndarray] = {}

    def init(self, pack_hex: str):
        if self.ids is not None:
            return
        self.ids, self.names = self.r.gene_table(pack_hex)
        if self.want is not None:
            from singlet._loader import _gene_keep_mask

            keep = _gene_keep_mask(
                [str(x) for x in self.ids], [str(x) for x in self.names], self.want
            )
            new = np.cumsum(keep).astype(np.int32) - 1
            self.keep = np.where(keep, new, -1).astype(np.int32)
            self.ids, self.names = self.ids[keep], self.names[keep]

    @property
    def n(self) -> int:
        return len(self.ids)

    def row_map(self, pack_hex: str) -> np.ndarray:
        """stored-agnostic canonical row -> var index (int32, -1 = not kept) for a matrix whose rows are the pack's list."""
        hit = self._maps.get(pack_hex)
        if hit is not None:
            return hit
        self.init(pack_hex)
        rg = np.asarray(self.r.row_gene(pack_hex), dtype=np.int64)
        pids, _ = self.r.gene_table(pack_hex)
        lut = {g: i for i, g in enumerate(self.ids.tolist())}
        to = np.array([lut.get(g, -1) for g in pids.tolist()] + [-1], dtype=np.int32)
        m = np.ascontiguousarray(to[np.where(rg >= 0, rg, len(pids))], dtype=np.int32)
        self._maps[pack_hex] = m
        return m


def _decode_member(
    prefix: bytes,
    runs: Dict[int, bytes],
    cols: np.ndarray,
    gene_map: Optional[np.ndarray],
    n_genes: int,
    threads: int,
    fetch=None,
):
    """-> (indptr, indices, data) for ``cols`` (ascending), genes x cells CSC (== cells x genes CSR); gene_map None = native rows."""
    from singlet import _pz2

    mem = _pz2.Member(prefix, fetch) if fetch is not None else _pz2.Member(prefix)
    for off, buf in runs.items():
        mem.add_range(off, buf)
    fc = mem.chunk_index()[:, 2].astype(np.int64)
    k = np.searchsorted(fc, cols, side="right") - 1
    parts = []
    for c in (int(x) for x in np.unique(k)):
        # decode each touched chunk on its own (runs may skip chunks between them)
        a, b = int(fc[c]), int(fc[c + 1])
        sel = cols[(cols >= a) & (cols < b)]
        rel = (sel - a).astype(np.uint32)
        if gene_map is not None:
            r = mem.decode(
                a, b, mode="gene", threads=threads, cols=rel, gene_map=gene_map, n_genes=n_genes
            )
        else:
            r = mem.decode(a, b, mode="canonical", threads=threads, cols=rel)
        parts.append((r["indptr"], r["indices"], r["data"]))
    if not parts:
        return np.zeros(1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32), mem
    nnz = sum(int(p[0][-1]) for p in parts)
    indptr = np.empty(len(cols) + 1, dtype=np.int64)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32 if parts[0][2].dtype.kind == "f" else np.int32)
    indptr[0] = 0
    c = z = 0
    for ip, ix, dv in parts:
        n = len(ip) - 1
        e = z + int(ip[-1])
        indptr[c + 1 : c + n + 1] = ip[1:] + z
        indices[z:e] = ix
        data[z:e] = dv
        c += n
        z = e
    return indptr, indices, data, mem


def _sum_trailers(ts: List[dict]) -> dict:
    if len(ts) <= 1:
        return ts[0] if ts else {}
    return {
        k: sum(t.get(k, 0) for t in ts)
        for k in ("reads", "bytes_read", "bytes_sent", "cache_hits", "ms")
    } | {"ok": all(t.get("ok", True) for t in ts)}


def gather(
    cells,
    *,
    modalities=None,
    genes: Optional[Sequence[str]] = None,
    version: str = "current",
    base: Optional[str] = None,
    via: str = "auto",
    api_key: Optional[str] = None,
    threads: int = 4,
    max_cells: Optional[int] = None,
    batch_cells: Optional[int] = None,
    batch_studies: int = MAX_STUDIES_PER_REQUEST,
    index: Optional[Index] = None,
    verbose: bool = False,
) -> "anndata.AnnData":
    """Fetch the given cells from any number of samples into one AnnData.

    ``cells``: a :func:`find_cells` DataFrame, ``[(gsm_or_so, [cols...]), ...]`` or ``{gsm_or_so: cols}``. ``modalities``: the
    ``singlet.s1.client`` vocabulary (default counts + spliced + unspliced; also splice_junctions, mt, vdj -> ``obsm``).
    ``via="worker"`` streams everything from ``POST /s1/cidx/gather`` in one request (the Worker reads R2 in parallel inside
    Cloudflare); ``via="direct"`` issues the same coalesced range requests from here. Prefixes of matrices already seen are cached on
    disk and not fetched again. The Worker serves at most 10,000 cells from at most 10 studies per request: more are fetched in
    several requests of at most ``batch_cells`` cells and ``batch_studies`` studies (each a request under the rate limits)."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    from .client import _resolver_for

    mods = normalize_modalities(modalities)
    members = members_for(mods)
    idx = index or Index.open(version, base=base)
    want = _cells_by_sample(idx, cells)
    n_total = sum(len(v) for v in want.values())
    if max_cells is not None and n_total > max_cells:
        raise IndexError_(f"{n_total:,} cells requested, more than max_cells={max_cells:,}")
    sos = sorted(want)
    entries = [idx.sample(s) for s in sos]
    builds = {e["organism"] for e in entries}
    if len(builds) > 1:
        raise IndexError_(
            f"cells from several organisms ({sorted(builds)}): gather one organism at a time"
        )
    pc = _PrefixCache()
    worker = _use_worker(via, base)
    if batch_cells is None:
        batch_cells = MAX_CELLS_PER_REQUEST
    # batches under the per-request caps (cells and studies); samples are taken study by study, and a sample bigger than a batch
    # is split by columns (its pieces stay in column order)
    gse_of = {s: e["gse"] for s, e in zip(sos, entries)}
    sos = sorted(sos, key=lambda s: (gse_num(gse_of[s]), s))
    batches: List[Dict[int, np.ndarray]] = []
    cur: Dict[int, np.ndarray] = {}
    cur_gse: set = set()
    n_cur = 0
    for s in sos:
        cols_all = want[s]
        for a0 in range(0, max(1, len(cols_all)), batch_cells):
            piece = cols_all[a0 : a0 + batch_cells]
            new_study = gse_of[s] not in cur_gse
            if cur and (
                n_cur + len(piece) > batch_cells
                or s in cur
                or (new_study and len(cur_gse) >= batch_studies)
            ):
                batches.append(cur)
                cur, cur_gse, n_cur = {}, set(), 0
            cur[s] = piece
            cur_gse.add(gse_of[s])
            n_cur += len(piece)
    if cur:
        batches.append(cur)

    resolver = _resolver_for(idx.t)
    space = _GeneSpace(resolver, genes)
    layers: Dict[str, List[Any]] = {"exon_counts.1pz": [], "intron_counts.1pz": []}
    raw: Dict[str, List[Any]] = {}
    raw_names: Dict[str, Any] = {}
    obs = []
    order: List[int] = []  # rows of each piece, for empty placeholders
    jobs: List[tuple] = []
    totals: Dict[str, Any] = {
        "planned_bytes": 0,
        "planned_reads": 0,
        "requests": 0,
        "seconds_fetch": 0.0,
        "trailers": [],
    }
    from singlet import _pz2

    for bw in batches:
        b_sos = sorted(bw)
        b_entries = [idx.sample(s) for s in b_sos]
        have = [
            m.get("digest")
            for e in b_entries
            for n, m in e["members"].items()
            if n in members and pc.get(m.get("digest")) is not None
        ]
        t0 = time.time()
        plan = plan_gather(b_entries, bw, members, have)
        trailer: dict = {}
        if worker:
            body = {
                "version": idx.version,
                "cells": [[s, bw[s].tolist()] for s in b_sos],
                "modalities": list(mods),
                "have_prefix": have,
            }
            resp = _post(base, "/s1/cidx/gather", body, api_key=api_key, stream=True)
            with resp:
                head, got, trailer = read_gather_stream(resp)
            if [s["so"] for s in head["samples"]] != b_sos:
                raise IndexError_("the Worker planned other samples than were asked for")
        else:
            got = _fetch_direct(idx, plan, threads=threads)
        totals["seconds_fetch"] += time.time() - t0
        totals["planned_bytes"] += plan["bytes"]
        totals["planned_reads"] += len(plan["reads"])
        totals["requests"] += 1 if worker else len(plan["reads"])
        if trailer:
            totals["trailers"].append(trailer)
        if trailer.get("errors"):
            raise IndexError_(f"gather failed for some samples: {trailer['errors'][:3]}")

        for ps in plan["samples"]:
            i, so = ps["i"], ps["so"]
            cols = bw[so]
            e = b_entries[i]
            order.append(len(cols))
            for nm in members:
                mo = ps["members"].get(nm)
                if mo is None:
                    (layers[nm] if nm in layers else raw.setdefault(nm, [])).append(None)
                    continue
                g = got.get((i, nm), {"runs": {}})
                prefix = g.get("prefix") or pc.get(mo["digest"])
                if prefix is None:
                    raise IndexError_(f"{ps['gsm']}: {nm}: no prefix received")
                if mo.get("send_prefix") and g.get("prefix") is not None:
                    pc.put(mo["digest"], g["prefix"])
                hdr = _pz2.read_header(prefix[:256])
                if mo.get("digest") and hdr["digest"] != str(mo["digest"]).lower():
                    raise IndexError_(
                        f"{ps['gsm']}: {nm}: header digest {hdr['digest']} != plan {mo['digest']}"
                    )
                m_off = int(e["members"][nm]["off"])

                def fetch(off, ln, key=e["key"], base_off=m_off):
                    return _get(idx.t, key, base_off + off, ln)

                slots = layers[nm] if nm in layers else raw.setdefault(nm, [])
                slots.append(None)
                if nm in layers:
                    kind = hdr["row_names_kind"]
                    if kind != "pack":
                        raise IndexError_(
                            f"{ps['gsm']}: {nm}: row names are {kind}, gene aggregation needs a vocabulary pack"
                        )
                    gm = space.row_map(
                        hdr["row_names_id"]
                    )  # built here: the map cache is not thread safe
                    jobs.append(
                        (
                            slots,
                            len(slots) - 1,
                            nm,
                            prefix,
                            g["runs"],
                            cols,
                            gm,
                            space.n,
                            fetch,
                            None,
                        )
                    )
                else:
                    jobs.append(
                        (
                            slots,
                            len(slots) - 1,
                            nm,
                            prefix,
                            g["runs"],
                            cols,
                            None,
                            0,
                            fetch,
                            int(hdr["m"]),
                        )
                    )
            obs.append(pd.DataFrame({"gsm": ps["gsm"], "gse": ps["gse"], "so": so, "col": cols}))

        def run(job):
            _slots, _pos, nm_, prefix_, runs_, cols_, gm_, ng_, fetch_, m_ = job
            ip, ix, dv, mem = _decode_member(prefix_, runs_, cols_, gm_, ng_, 1, fetch_)
            if gm_ is not None:
                return sp.csr_matrix((dv, ix, ip), shape=(len(cols_), ng_)), None
            info = mem.row_names()
            names = info.get("names") if isinstance(info, dict) else None
            return sp.csr_matrix((dv, ix, ip), shape=(len(cols_), m_)), (
                tuple(names) if names is not None else (nm_, m_)
            )

        # decode every (sample, matrix) of the batch in parallel (the decoder releases the GIL)
        with ThreadPoolExecutor(max(1, min(threads, len(jobs) or 1))) as ex:
            results = list(ex.map(run, jobs))
        for job, (mat, key_) in zip(jobs, results):
            slots, pos, nm = job[0], job[1], job[2]
            slots[pos] = mat
            if key_ is not None:
                prev = raw_names.get(nm)
                if prev is not None and prev != key_:
                    raise IndexError_(
                        f"{nm}: samples differ in their feature list; gather them separately"
                    )
                raw_names[nm] = key_
        jobs.clear()
    obs_df = (
        pd.concat(obs, ignore_index=True)
        if obs
        else pd.DataFrame(columns=["gsm", "gse", "so", "col"])
    )
    obs_df.index = (obs_df["gsm"] + "_" + obs_df["col"].astype(str)).astype(object).to_numpy()
    n_cells = len(obs_df)
    nG = space.n if space.ids is not None else 0

    def stack(lst, width):
        mats = [
            m if m is not None else sp.csr_matrix((nrow, width), dtype=np.int32)
            for m, nrow in zip(lst, order)
        ]
        return sp.vstack(mats, format="csr") if mats else sp.csr_matrix((0, width))

    sp_mat = stack(layers["exon_counts.1pz"], nG) if "exon_counts.1pz" in members else None
    un_mat = stack(layers["intron_counts.1pz"], nG) if "intron_counts.1pz" in members else None
    X = None
    if "counts" in mods:
        X = (
            sp_mat + un_mat
            if sp_mat is not None and un_mat is not None
            else (sp_mat if sp_mat is not None else un_mat)
        )
    var = (
        pd.DataFrame(
            {"gene_name": space.names.astype(object)},
            index=pd.Index(space.ids.astype(object), name="gene_id"),
        )
        if nG
        else pd.DataFrame(index=pd.Index([], name="gene_id"))
    )
    adata = ad.AnnData(
        X=X if X is not None else sp.csr_matrix((n_cells, nG), dtype=np.int32), obs=obs_df, var=var
    )
    if "spliced" in mods and sp_mat is not None:
        adata.layers["spliced"] = sp_mat
    if "unspliced" in mods and un_mat is not None:
        adata.layers["unspliced"] = un_mat
    feats = {}
    for k in mods:
        if k in GENE_LEVEL:
            continue
        nm = _MOD_MEMBERS[k][0]
        lst = raw.get(nm)
        if not lst:
            continue
        width = next((m.shape[1] for m in lst if m is not None), 0)
        adata.obsm[k] = stack(lst, width)
        names = raw_names.get(nm)
        feats[k] = list(names) if names and isinstance(names[0], str) else None
    adata.uns["singlet_gather"] = {
        "index_version": idx.version,
        "modalities": list(mods),
        "samples": len(sos),
        "via": "worker" if worker else "direct",
        "batches": len(batches),
        "planned_bytes": totals["planned_bytes"],
        "planned_reads": totals["planned_reads"],
        "requests": totals["requests"],
        "seconds_fetch": round(totals["seconds_fetch"], 3),
        "trailer": _sum_trailers(totals["trailers"]),
    }
    if feats:
        adata.uns["obsm_features"] = {k: v for k, v in feats.items() if v is not None}
    if verbose:
        g_ = adata.uns["singlet_gather"]
        print(
            f"  [singlet.atlas] {n_cells:,} cells from {len(sos)} samples, {g_['planned_bytes']:,} bytes in {g_['planned_reads']} reads "
            f"({g_['requests']} requests, {len(batches)} batches), {g_['seconds_fetch']:.2f}s"
        )
    return adata
