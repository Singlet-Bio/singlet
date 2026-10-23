# SPDX-License-Identifier: MIT
"""singlet.v2.atlas -- the atlas-wide cell index and cross-study cell gathering (docs/format-v2-cell-index.md).

::

    from singlet.v2 import atlas
    cells = atlas.find_cells(organism="Homo sapiens", cell_type="CL:0000084", tissue="UBERON:0002048", min_umi=1000)
    n     = atlas.count_cells(organism="Homo sapiens", cell_type="CL:0000084", by=["tissue"])
    adata = atlas.gather(cells.head(5000), modalities=["counts"])      # one streamed request to data.singlet.bio/v2/cidx/gather

Three parts:

* **Format** (spec sections 3-6): SCX1 columnar shards, the directory, sample table, summary and plan blocks, label sidecars.
* **Builder** (:func:`build_index`, ``scripts/v2_cidx_build.py``): manifests + ``cells`` tables + label sidecars -> one index version
  written in the server layout (``v2/cidx/...``); runs on the pilot locally and on Clipper/Anvil for the atlas.
* **Client**: :class:`Index` evaluates queries itself (local directory or ranged reads of the shards) with the same semantics as
  the Worker endpoint ``POST /v2/cidx/query``; :func:`gather` builds an AnnData from the Worker's streamed ``POST /v2/cidx/gather``
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

CIDX = "v2/cidx"
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
    """Content-addressed key of an index object: ``v2/cidx/obj/<d16>.<ext>`` (immutable)."""
    return f"{CIDX}/obj/{d16hex(data)}.{ext}"


def version_key(version: str) -> str:
    return f"{CIDX}/v/{version}.json"


CURRENT_KEY = f"{CIDX}/current.json"


def sidecar_key(gsm: str, data: bytes) -> str:
    """``v2/a/<GSM>/<d16>.parquet``: one canonical label sidecar (immutable; referenced from the study manifest)."""
    return f"v2/a/{gsm}/{d16hex(data)}.parquet"


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
    out[ok] = np.clip(np.rint(a[ok] * SCALE8), 0, SCALE8).astype(np.uint8)
    return out


def q8_threshold(v: float, upper: bool) -> int:
    """Smallest (lower bound) / largest (upper bound) u8 code that satisfies ``score >= v`` / ``score <= v`` at u8 resolution."""
    c = v * SCALE8
    return int(np.floor(c + 1e-9)) if upper else int(np.ceil(c - 1e-9))


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


def make_dict(
    terms: Iterable[Optional[str]],
    ontology: Optional[Ontology],
    labels: Optional[Dict[str, str]] = None,
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
                "anc": list(ontology.ancestors(t)) if ontology else [],
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
)
_CL_RE = re.compile(r"^CL:\d{7}$")
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
        "version": "1",
        "gsm": gsm,
        "annot_version": annot_version,
        "cells_digest": cells_digest or "",
        "model_version": ";".join(mv),
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
    # filled by _load_sample
    cols: Optional[np.ndarray] = None
    qc: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Optional[Dict[str, Any]] = None
    plan_members: Dict[str, dict] = field(default_factory=dict)
    error: Optional[str] = None


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
    t, sm: _SampleIn, sidecar_src: Optional[Callable[[_SampleIn], Any]], store
) -> _SampleIn:
    from .cells import CellsReader

    try:
        cm = sm.members.get("cells")
        if not cm or int(cm.get("kind", 0)) != 2:
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


@dataclass
class BuildResult:
    version: str
    directory: dict
    objects: Dict[str, bytes]
    stats: dict
    errors: List[dict]

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


def build_index(
    gses: Sequence[str],
    *,
    base: Optional[str] = None,
    sample_meta: Optional[str] = None,
    labels_dir: Optional[str] = None,
    annot_version: Optional[str] = None,
    ontology: Optional[Ontology] = None,
    previous: Optional[str] = None,
    threads: int = 8,
    shard_rows: int = SHARD_ROWS,
    group_rows: int = GROUP_ROWS,
    parquet: bool = False,
    now: Optional[str] = None,
    log: Callable[[str], None] = lambda s: None,
) -> BuildResult:
    """Build one index version from the v2 manifests of ``gses`` (read through ``base``: a directory with the server layout or a
    URL; default ``SINGLET_DATA_BASE``).

    Labels per sample come from, in this order: ``labels_dir/<GSM>.labels.parquet`` (or ``<GSE>/<GSM>.labels.parquet``, the
    annotation pipeline's output), else the manifest's ``annotations[annot_version or annotations_default]`` sidecar. Sample
    metadata (organism, tissue) from ``sample_meta``, else the manifest's ``study_meta.json``. ``previous`` (an index version)
    keeps its sample ordinals."""
    from .packs import PackStore

    t = _transport(base)
    store = PackStore()
    meta = _read_sample_meta(sample_meta)
    ont = ontology or Ontology()
    samples: List[_SampleIn] = []
    for gse in gses:
        doc = json.loads(_get(t, f"v2/m/{gse}.json").decode("utf-8"))
        gm = (
            ((doc.get("study_files") or {}).get("study_meta.json") or {})
            .get("json", {})
            .get("gsm_meta", {})
        )
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

    t0 = time.time()
    with ThreadPoolExecutor(max(1, threads)) as ex:
        loaded = list(ex.map(lambda s: _load_sample(t, s, sidecar_src, store), samples))
    log(f"read {len(loaded)} samples in {time.time() - t0:.1f}s")
    errors = [{"gsm": s.gsm, "gse": s.gse, "error": s.error} for s in loaded if s.error]
    ok = [s for s in loaded if not s.error and s.cols is not None]

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
    d_ct = make_dict((x for s in ok for x in lab_list(s, "cell_type_cl")), ont, ct_labels)
    d_tissue = make_dict(
        (s.tissue for s in ok),
        ont,
        {s.tissue: s.tissue_label for s in ok if s.tissue and s.tissue_label},
    )
    d_anc = make_dict((x for s in ok for x in lab_list(s, "ancestry_hancestro")), ont)
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
        parts: Dict[str, List[np.ndarray]] = {k: [] for k, _, _ in SHARD_COLUMNS}
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
        cols = {
            k: (np.concatenate(v) if v else np.zeros(0, _NP[dt])).astype(_NP[dt])
            for (k, dt, _), v in zip(SHARD_COLUMNS, parts.values())
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
    }
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
            "module": "singlet.v2.atlas",
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
        },
        "terms": term_labels([d_ct, d_tissue, d_anc], ont),
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
        "cells": n_rows_total,
        "shard_bytes": sum(len(objects[k]) for k in shard_list),
        "directory_bytes": len(dblob),
        "plan_bytes": sum(len(objects[k]) for k in plan_keys.values()),
        "seconds": round(time.time() - t0, 2),
        "requests": getattr(t.stats, "requests", None),
        "bytes_read": getattr(t.stats, "bytes", None),
    }
    return BuildResult(version, directory, objects, stats, errors)


# ================================================================================================================= query engine
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
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, w: dict) -> "Query":
        def lst(v):
            if v is None:
                return None
            return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]

        q = cls(
            organism=w.get("organism"),
            cell_type=lst(w.get("cell_type")),
            tissue=lst(w.get("tissue")),
            sex=lst(w.get("sex")),
            ancestry=lst(w.get("ancestry")),
            samples=lst(w.get("samples")),
            studies=lst(w.get("studies")),
        )
        for k, v in (w.get("ranges") or {}).items():
            if k not in RANGE_FIELDS:
                raise IndexError_(f"unknown range field {k!r}; use {sorted(RANGE_FIELDS)}")
            lo, hi = (v + [None, None])[:2] if isinstance(v, list) else (v, None)
            q.ranges[k] = (None if lo is None else float(lo), None if hi is None else float(hi))
        return q

    def to_json(self) -> dict:
        d: Dict[str, Any] = {}
        for k in ("organism", "cell_type", "tissue", "sex", "ancestry", "samples", "studies"):
            v = getattr(self, k)
            if v:
                d[k] = v
        if self.ranges:
            d["ranges"] = {k: [lo, hi] for k, (lo, hi) in sorted(self.ranges.items())}
        return d


def _bounds(field_name: str, lo: Optional[float], hi: Optional[float]) -> Tuple[int, int]:
    """Integer [lo, hi] bounds on the stored column (scores: u8 codes, NA 255 always excluded)."""
    if field_name in SCORE_FIELDS:
        a = 0 if lo is None else q8_threshold(lo, upper=False)
        b = SCALE8 if hi is None else min(SCALE8, q8_threshold(hi, upper=True))
        return a, b
    a = 0 if lo is None else int(np.ceil(lo))
    b = 0xFFFFFFFF * 2 if hi is None else int(np.floor(hi))
    return a, b


class _Plan:
    """What a query needs, resolved against one directory (codes, sample set, candidate shards)."""

    def __init__(self, idx: "Index", q: Query):
        d = idx.directory
        self.q = q
        dicts = d["dicts"]
        self.terms = d.get("terms") or {}
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
        self.ranges = {k: _bounds(k, lo, hi) for k, (lo, hi) in q.ranges.items()}
        # sample set from samples / studies / exclusions
        self.sample_set: Optional[np.ndarray] = None
        if q.samples or q.studies:
            st = idx.samples_table()
            keep = np.zeros(len(st["so"]), bool)
            if q.samples:
                keep |= np.isin(st["gsm"], [gsm_num(g) for g in q.samples])
            if q.studies:
                keep |= np.isin(st["gse"], [gse_num(g) for g in q.studies])
            self.sample_set = np.sort(st["so"][keep].astype(np.int64))
        self.exclude = np.array(sorted(d.get("exclude") or []), dtype=np.int64)

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
        """True when the query needs per-cell columns beyond the summary (ranges or tissue on shards is fine via summary)."""
        return bool(self.ranges)

    def shard_ok(self, sh: dict) -> bool:
        z = sh["zone"]
        if (
            not self._overlaps(self.ct, *z["ct"])
            or not self._overlaps(self.tissue, *z["tissue"])
            or not self._overlaps(self.anc, *z["anc"])
        ):
            return False
        if self.sex is not None and not self.sex[z["sex"][0] : z["sex"][1] + 1].any():
            return False
        for k, (a, b) in self.ranges.items():
            cs = RANGE_FIELDS[k]
            if len(cs) == 1:
                lo, hi = z[cs[0]]
            else:
                lo, hi = sum(z[c][0] for c in cs), sum(z[c][1] for c in cs)
            if hi < a or lo > b:
                return False
        if self.sample_set is not None:
            lo, hi = z["sample"]
            i = np.searchsorted(self.sample_set, lo)
            if i >= len(self.sample_set) or self.sample_set[i] > hi:
                return False
        return True

    def group_ok(self, sh: Shard, g: int) -> bool:
        def z(c):
            return sh.zone(g, c)

        if (
            not self._overlaps(self.ct, *z("ct"))
            or not self._overlaps(self.tissue, *z("tissue"))
            or not self._overlaps(self.anc, *z("anc"))
        ):
            return False
        if self.sex is not None and not self.sex[z("sex")[0] : z("sex")[1] + 1].any():
            return False
        for k, (a, b) in self.ranges.items():
            cs = RANGE_FIELDS[k]
            lo, hi = (
                z(cs[0]) if len(cs) == 1 else (sum(z(c)[0] for c in cs), sum(z(c)[1] for c in cs))
            )
            if hi < a or lo > b:
                return False
        if self.sample_set is not None:
            lo, hi = z("sample")
            i = np.searchsorted(self.sample_set, lo)
            if i >= len(self.sample_set) or self.sample_set[i] > hi:
                return False
        return True

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
        for k in self.ranges:
            need.extend(RANGE_FIELDS[k])
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
        for k, (a, b) in self.ranges.items():
            cs = RANGE_FIELDS[k]
            v = (
                cols[cs[0]].astype(np.int64)
                if len(cs) == 1
                else sum(cols[c].astype(np.int64) for c in cs)
            )
            m &= (v >= a) & (v <= b)
        s = cols["sample"].astype(np.int64)
        if self.sample_set is not None:
            m &= np.isin(s, self.sample_set)
        if len(self.exclude):
            m &= ~np.isin(s, self.exclude)
        return m


class Index:
    """One index version, read from a local directory with the server layout or over HTTP (``data.singlet.bio``).

    Queries run here with the same semantics as the Worker's ``POST /v2/cidx/query`` (spec section 6): the directory prunes
    shards, shard headers prune row groups, only the needed column chunks are read."""

    def __init__(self, directory: dict, transport, *, base: Optional[str] = None):
        self.directory = directory
        self.version = directory["version"]
        self.t = transport
        self.base = base
        self._shards: Dict[str, Shard] = {}
        self._lock = threading.Lock()
        self._samples: Optional[Dict[str, np.ndarray]] = None
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

    def so_of(self, gsms: Sequence[str]) -> np.ndarray:
        st = self.samples_table()
        want = np.array([gsm_num(g) for g in gsms], dtype=np.int64)
        lut = dict(zip(st["gsm"].tolist(), st["so"].tolist()))
        miss = [g for g, w in zip(gsms, want.tolist()) if w not in lut]
        if miss:
            raise IndexError_(f"{miss[:5]} not in index {self.version}")
        return np.array([lut[w] for w in (int(x) for x in want)], dtype=np.int64)

    # ---- queries
    def _candidate_groups(
        self, p: _Plan
    ) -> Iterable[Tuple[int, str, Shard, int, Optional[Tuple[int, int]]]]:
        """(shard index, key, shard, group, row window or None) in directory order."""
        d = self.directory
        use_postings = p.sample_set is not None and len(p.sample_set) <= 2000
        windows: Dict[int, List[Tuple[int, int]]] = {}
        if use_postings:
            for so in p.sample_set.tolist():
                for gi, st, n in self.sample(int(so))["runs"]:
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
                    if use_postings:
                        w = [
                            (max(a, g0), min(b, g0 + gn))
                            for a, b in windows[sh["i"]]
                            if a < g0 + gn and b > g0
                        ]
                        if not w:
                            continue
                    if not p.group_ok(s, g):
                        continue
                    yield sh["i"], sh["key"], s, g, None

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
        for gi, key, s, g, _w in self._candidate_groups(p):
            if (gi, g) < start:
                continue
            cols = {c: s.column(g, c) for c in need}
            scanned += s.groups[g][1]
            self.stats["groups"] += 1
            self.stats["rows_scanned"] += s.groups[g][1]
            m = p.rows(cols)
            idx = np.flatnonzero(m)
            yield gi, g, idx, {c: cols[c][idx] for c in want}
            if max_rows is not None and scanned >= max_rows:
                return

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
            sm = self.shard(self.directory["summary"]["key"]).table()
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
            if p.sample_set is not None:
                m &= np.isin(sm["sample"].astype(np.int64), p.sample_set)
            if len(p.exclude):
                m &= ~np.isin(sm["sample"].astype(np.int64), p.exclude)
            cols = {k: v[m] for k, v in sm.items()}
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

    def _org_of(self, so: np.ndarray) -> np.ndarray:
        st = self.samples_table()
        pos = np.searchsorted(st["so"], so)
        return st["organism"][pos] if len(so) else np.zeros(0, np.uint16)

    def _group(self, cols, weights, by) -> "pd.DataFrame":
        import pandas as pd

        d = self.directory["dicts"]
        st = self.samples_table()
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
                pos = np.searchsorted(st["so"], cols["sample"])
                frame[b] = st["gse"][pos].astype(np.int64) if len(pos) else np.zeros(0, np.int64)
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
        """Matching cells as a DataFrame: so, gsm, col and the requested shard columns (codes decoded to ontology ids)."""

        q = q if isinstance(q, Query) else Query.from_json(q)
        p = _Plan(self, q)
        want = list(
            dict.fromkeys(
                ["sample", "col"] + [f for f in fields if f in {c for c, _, _ in SHARD_COLUMNS}]
            )
        )
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
        try:
            j = json.loads(e.read().decode("utf-8"))
            msg = j.get("message") or j.get("error")
        except Exception:  # noqa: BLE001
            msg = None
        raise IndexError_(f"{path}: HTTP {e.code}: {msg or e.reason}") from e
    if stream:
        return resp
    with resp:
        return json.loads(resp.read().decode("utf-8"))


def _query_json(**kw) -> dict:
    w: Dict[str, Any] = {}
    for k in ("organism", "cell_type", "tissue", "sex", "ancestry", "samples", "studies"):
        v = kw.get(k)
        if v is not None:
            w[k] = v
    ranges = dict(kw.get("ranges") or {})
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
    limit: int = 100_000,
    version: str = "current",
    base: Optional[str] = None,
    via: str = "auto",
    api_key: Optional[str] = None,
) -> "pd.DataFrame":
    """Cells across the atlas that match every given filter: a DataFrame with ``so, gsm, col`` and their labels and QC.

    ``cell_type`` / ``tissue`` / ``ancestry`` take ontology ids (CL, UBERON, HANCESTRO) or labels and match descendants.
    ``via="worker"`` (default for the public host) asks ``POST /v2/cidx/query`` and pages through its cursor; ``via="direct"``
    evaluates the query here with ranged reads of the shards (also what a local directory base uses)."""
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
    )
    if not _use_worker(via, base):
        return Index.open(version, base=base).find(w, limit=limit)
    import pandas as pd

    frames = []
    cursor = None
    got = 0
    while True:
        page = min(limit - got, 100_000)
        r = _post(
            base,
            "/v2/cidx/query",
            {"version": version, "where": w, "return": "cells", "limit": page, "cursor": cursor},
            api_key=api_key,
        )
        frames.append(pd.DataFrame(r["cells"]))
        got += len(r["cells"]["so"])
        cursor = r.get("cursor")
        if not cursor or got >= limit:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    Same filters as :func:`find_cells`."""
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
            "/v2/cidx/query",
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

# the modalities= vocabulary of singlet.v2.client (kept in step with it; gathered here: the gene-level ones and raw matrices)
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
                "psi is derived from splice_junctions; gather splice_junctions and compute it (singlet.v2 recipes_psi)"
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
    index: Optional[Index] = None,
    verbose: bool = False,
) -> "anndata.AnnData":
    """Fetch the given cells from any number of samples into one AnnData.

    ``cells``: a :func:`find_cells` DataFrame, ``[(gsm_or_so, [cols...]), ...]`` or ``{gsm_or_so: cols}``. ``modalities``: the
    ``singlet.v2.client`` vocabulary (default counts + spliced + unspliced; also splice_junctions, mt, vdj -> ``obsm``).
    ``via="worker"`` streams everything from ``POST /v2/cidx/gather`` in one request (the Worker reads R2 in parallel inside
    Cloudflare); ``via="direct"`` issues the same coalesced range requests from here. Prefixes of matrices already seen are cached on
    disk and not fetched again. More cells than ``batch_cells`` (default: the Worker's per-request cap, 20,000 anonymous / 500,000
    with an API key) are fetched in several requests."""
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
        batch_cells = 500_000 if (api_key or os.environ.get("SINGLET_API_KEY")) else 20_000
    # batches under the per-request cap; a sample bigger than a batch is split by columns (its pieces stay in column order)
    batches: List[Dict[int, np.ndarray]] = []
    cur: Dict[int, np.ndarray] = {}
    n_cur = 0
    for s in sos:
        cols_all = want[s]
        for a0 in range(0, max(1, len(cols_all)), batch_cells):
            piece = cols_all[a0 : a0 + batch_cells]
            if cur and (n_cur + len(piece) > batch_cells or s in cur):
                batches.append(cur)
                cur, n_cur = {}, 0
            cur[s] = piece
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
            resp = _post(base, "/v2/cidx/gather", body, api_key=api_key, stream=True)
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
