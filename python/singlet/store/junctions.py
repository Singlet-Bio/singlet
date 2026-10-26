# SPDX-License-Identifier: MIT
"""Shared splice-junction table (docs/format-store.md section 12): one integer id space per reference build for every junction the atlas
uses, so a sample's junction matrix stores row ids instead of name strings.

Coordinates follow the pipeline's junction names ``<chrom>:<start>-<end>``: ``start`` is the 0-based position of the first intron base
(the donor side on the + strand), ``end`` the 0-based position of the first exon base after the intron, so the intron is the half-open
interval ``[start, end)`` and ``name = f"{chrom}:{start}-{end}"`` reproduces the pipeline's name byte for byte.

Objects (all SPK2 packs, ``packs.py``)::

    kind 4  junction segment   a contiguous id range [id_lo, id_lo + n) of one build, rows in genomic order within the segment
    kind 5  junction version   the list of segments that make up one table version (append-only: a later version only adds segments)

A version is what a sample references (``pack_subset`` row names whose pack id is the version's id): row ids below ``LOCAL_BASE`` are
table ids; ids ``LOCAL_BASE + k`` are the sample's own junctions (row k of the sample's local junction list, a member holding a
segment body without ids). Ids are never renumbered: a junction keeps its id in every later version, a junction first seen after a
version was cut stays local in the samples written before and gets an id in a later segment.

Segment body (``coder`` = ``JX1``): ``u32 n``, then streams ``chrom_runs`` (pairs chrom index, count), ``start_delta`` (start minus the
previous row's start, modulo 2^32, across chromosome boundaries too, so one wrapping uint32 cumulative sum restores every start),
``length`` (end - start), ``attr`` (u8: bits 0-1 strand 0 unknown 1 + 2 -, bits 2-4 motif as STAR codes, bit 5 annotated, bit 6 donor on
an annotated exon end, bit 7 acceptor on an annotated exon start), ``gene`` (zigzag of the uint32 difference of gene index + 1 into the
build's exon pack gene table, 0 = none), ``n_tx`` (annotated transcripts using the junction, saturating at 255). Each stream is
``u8 width, u32 n, u32 clen`` + zstd-19 of the values' ``width`` low little-endian bytes stored as planes (all low bytes, then all second
bytes, ...); ``width`` is the fewest bytes that hold the stream's largest value. Decoding is numpy only: zstd, ``frombuffer``, shifts,
one ``cumsum``.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import packs as P
from . import util

KIND_SEGMENT, KIND_VERSION = 4, 5
LOCAL_BASE = 1 << 31
LOCAL_SUFFIX = ".local"  # member "<matrix>.local" (kind 3 or 5) holds the sample's own junctions of that matrix
LEGACY_ORDER_SUFFIX = ".legacy_order"  # member "<matrix>.legacy_order": store row r holds legacy row perm[r] (encode_perm)
CODER = "JX1"
COORDS = "0-based [start, end): start = first intron base, end = first exon base after the intron"

# STAR SJ.out.tab motif codes: 0 non-canonical, 1 GT/AG, 2 CT/AC, 3 GC/AG, 4 CT/GC, 5 AT/AC, 6 GT/AT (odd: + strand, even: - strand)
MOTIFS = {
    ("GT", "AG"): 1,
    ("CT", "AC"): 2,
    ("GC", "AG"): 3,
    ("CT", "GC"): 4,
    ("AT", "AC"): 5,
    ("GT", "AT"): 6,
}
ANNOTATED, DONOR_ANN, ACCEPTOR_ANN = 1 << 5, 1 << 6, 1 << 7


class JunctionError(ValueError):
    pass


# ============================================================================================ columns
@dataclass
class Junctions:
    """Junctions as columns. ``chrom`` indexes ``chroms``; ``attr`` packs strand, motif and annotation bits (module docstring)."""

    chroms: List[str]
    chrom: np.ndarray  # u16
    start: np.ndarray  # u32
    end: np.ndarray  # u32
    attr: np.ndarray = field(default=None)  # type: ignore[assignment]  # u8
    gene: np.ndarray = field(default=None)  # type: ignore[assignment]  # u32, gene index + 1, 0 = none
    n_tx: np.ndarray = field(default=None)  # type: ignore[assignment]  # u8

    def __post_init__(self):
        n = len(self.chrom)
        self.chrom = np.asarray(self.chrom, np.uint16)
        self.start = np.asarray(self.start, np.uint32)
        self.end = np.asarray(self.end, np.uint32)
        self.attr = np.zeros(n, np.uint8) if self.attr is None else np.asarray(self.attr, np.uint8)
        self.gene = (
            np.zeros(n, np.uint32) if self.gene is None else np.asarray(self.gene, np.uint32)
        )
        self.n_tx = np.zeros(n, np.uint8) if self.n_tx is None else np.asarray(self.n_tx, np.uint8)

    def __len__(self) -> int:
        return len(self.chrom)

    @property
    def strand(self) -> np.ndarray:
        return self.attr & 3

    @property
    def motif(self) -> np.ndarray:
        return (self.attr >> 2) & 7

    @property
    def annotated(self) -> np.ndarray:
        return (self.attr & ANNOTATED) != 0

    def take(self, idx) -> "Junctions":
        idx = np.asarray(idx)
        return Junctions(
            self.chroms,
            self.chrom[idx],
            self.start[idx],
            self.end[idx],
            self.attr[idx],
            self.gene[idx],
            self.n_tx[idx],
        )

    def genomic_order(self) -> np.ndarray:
        """Sort order by (chromosome name, start, end): chromosome names in byte order, so the order does not depend on ``chroms``."""
        rank = np.argsort(
            np.argsort(np.array(self.chroms, dtype=object), kind="stable"), kind="stable"
        )
        return np.lexsort((self.end, self.start, rank[self.chrom]))

    def names(self) -> List[str]:
        if not len(self):
            return []
        ch = (
            self.chroms
        )  # a plain comprehension measured faster than numpy / pandas string assembly
        return [
            f"{ch[c]}:{a}-{b}"
            for c, a, b in zip(self.chrom.tolist(), self.start.tolist(), self.end.tolist())
        ]


def parse_names(names: Sequence[str], chroms: Optional[List[str]] = None) -> Junctions:
    """``<chrom>:<start>-<end>`` names -> columns. ``chroms`` (optional) fixes the chromosome list (new names are appended)."""
    chroms = list(chroms or [])
    cidx = {c: i for i, c in enumerate(chroms)}
    n = len(names)
    ch = np.empty(n, np.uint16)
    st = np.empty(n, np.uint32)
    en = np.empty(n, np.uint32)
    for i, nm in enumerate(names):
        c = nm.rfind(":")
        d = nm.find("-", c + 1)
        if c <= 0 or d < 0:
            raise JunctionError(f"not a junction name: {nm!r}")
        a, b = nm[c + 1 : d], nm[d + 1 :]
        if not (a.isdigit() and b.isdigit()) or str(int(a)) != a or str(int(b)) != b:
            raise JunctionError(f"not a canonical junction name: {nm!r}")
        k = cidx.get(nm[:c])
        if k is None:
            k = cidx[nm[:c]] = len(chroms)
            chroms.append(nm[:c])
        ch[i], st[i], en[i] = k, int(a), int(b)
    return Junctions(chroms, ch, st, en)


# ============================================================================================ streams
def _enc_stream(vals: np.ndarray) -> bytes:
    """``width`` low byte planes of uint32 values (width = bytes of the largest value, at least 1) + zstd-19."""
    v = np.ascontiguousarray(np.asarray(vals).astype(np.uint32))
    width = max(1, (int(v.max()).bit_length() + 7) // 8) if v.size else 1
    raw = v.view(np.uint8).reshape(-1, 4)[:, :width].T.tobytes()
    z = util.zstd_compress(raw, 19)
    return struct.pack("<BII", width, len(v), len(z)) + z


def _dec_stream(buf, p: int) -> Tuple[np.ndarray, int]:
    width, n, cl = struct.unpack_from("<BII", buf, p)
    p += 9
    if not 1 <= width <= 4:
        raise JunctionError(f"bad stream width {width}")
    raw = util.zstd_decompress(bytes(buf[p : p + cl]), max_output_size=max(width * n, 1))
    p += cl
    if len(raw) != width * n:
        raise JunctionError("stream length does not match its header")
    planes = np.frombuffer(raw, np.uint8).reshape(width, n)
    out = planes[0].astype(np.uint32)
    for k in range(1, width):
        out |= planes[k].astype(np.uint32) << np.uint32(8 * k)
    return out, p


def _zigzag(d: np.ndarray) -> np.ndarray:
    """int32 differences (as uint32 modulo 2^32) -> zigzag uint32."""
    d = np.asarray(d).astype(np.uint32).view(np.int32)
    return ((d << 1) ^ (d >> 31)).view(np.uint32)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.uint32)
    return (z >> np.uint32(1)) ^ (np.uint32(0) - (z & np.uint32(1)))


def encode_body(j: Junctions) -> bytes:
    """Segment body (rows must already be in genomic order: chromosome runs, start ascending within a run)."""
    n = len(j)
    if n:
        brk = np.flatnonzero(np.diff(j.chrom.astype(np.int64))) + 1
        run_start = np.concatenate([[0], brk])
        run_len = np.diff(np.concatenate([run_start, [n]]))
        if len(set(j.chrom[run_start].tolist())) != len(run_start):
            raise JunctionError("rows are not grouped by chromosome")
        runs = np.stack(
            [j.chrom[run_start].astype(np.uint32), run_len.astype(np.uint32)], 1
        ).ravel()
        within = np.ones(n, bool)
        within[run_start] = False
        st64 = j.start.astype(np.int64)
        if (np.diff(st64)[within[1:]] < 0).any():
            raise JunctionError("rows are not in start order within a chromosome")
        if (j.end <= j.start).any():
            raise JunctionError("a junction must end after it starts")
        sd = np.diff(np.concatenate([[0], st64])) % (
            1 << 32
        )  # wraps at chromosome changes; one uint32 cumsum undoes it
        gd = np.diff(np.concatenate([[0], j.gene.astype(np.int64)])) % (1 << 32)
    else:
        runs = np.zeros(0, np.uint32)
        sd = gd = np.zeros(0, np.int64)
    return b"".join(
        [
            struct.pack("<I", n),
            _enc_stream(runs),
            _enc_stream(sd),
            _enc_stream(j.end.astype(np.int64) - j.start.astype(np.int64)),
            _enc_stream(j.attr),
            _enc_stream(_zigzag(gd)),
            _enc_stream(j.n_tx),
        ]
    )


def decode_body(buf, chroms: List[str]) -> Junctions:
    (n,) = struct.unpack_from("<I", buf, 0)
    p = 4
    runs, p = _dec_stream(buf, p)
    sd, p = _dec_stream(buf, p)
    ln, p = _dec_stream(buf, p)
    attr, p = _dec_stream(buf, p)
    gz, p = _dec_stream(buf, p)
    ntx, p = _dec_stream(buf, p)
    runs = runs.reshape(-1, 2)
    chrom = np.repeat(runs[:, 0].astype(np.uint16), runs[:, 1])
    if len(chrom) != n or len(sd) != n:
        raise JunctionError("streams do not add up to the row count")
    start = np.cumsum(sd, dtype=np.uint32)  # modulo 2^32 by construction
    end = start + ln
    gene = np.cumsum(_unzigzag(gz), dtype=np.uint32)
    return Junctions(chroms, chrom, start, end, attr.astype(np.uint8), gene, ntx.astype(np.uint8))


# ============================================================================================ packs
def build_segment(
    j: Junctions, *, build: str, id_lo: int, tier: str, extra: Optional[dict] = None
) -> bytes:
    """A kind-4 pack for rows ``id_lo .. id_lo + len(j) - 1`` (rows are sorted into genomic order here; the order is the id order)."""
    o = j.genomic_order()
    jj = j.take(o)
    used = sorted(set(jj.chrom.tolist()))
    remap = np.zeros(max(len(j.chroms), 1), np.uint16)
    remap[used] = np.arange(len(used), dtype=np.uint16)
    jj = Junctions(
        [j.chroms[i] for i in used], remap[jj.chrom], jj.start, jj.end, jj.attr, jj.gene, jj.n_tx
    )
    hdr = {
        "build": build,
        "coder": CODER,
        "coords": COORDS,
        "id_lo": int(id_lo),
        "n": len(jj),
        "tier": tier,
        "chroms": jj.chroms,
    }
    hdr.update(extra or {})
    return P.wrap(KIND_SEGMENT, hdr, encode_body(jj))


def build_version(
    segments: Sequence[bytes],
    *,
    build: str,
    version: int,
    previous: Optional[bytes] = None,
    extra: Optional[dict] = None,
) -> bytes:
    """A kind-5 pack listing segments (in id order, contiguous from 0). ``previous`` (an earlier version pack) must be a prefix."""
    ents, nxt = [], 0
    for seg in segments:
        kind, h, _ = P.parse(seg)
        if kind != KIND_SEGMENT or h["build"] != build:
            raise JunctionError("not a junction segment of build %s" % build)
        if h["id_lo"] != nxt:
            raise JunctionError(
                f"segment ids start at {h['id_lo']}, expected {nxt}: ids must be contiguous and never renumbered"
            )
        ents.append(
            {
                "id": P.pack_id(seg).hex(),
                "id_lo": h["id_lo"],
                "n": h["n"],
                "tier": h["tier"],
                "bytes": len(seg),
            }
        )
        nxt += h["n"]
    if previous is not None:
        k, ph, _ = P.parse(previous)
        if k != KIND_VERSION or ph["segments"] != ents[: len(ph["segments"])]:
            raise JunctionError(
                "a new version must keep every segment of the previous one, in order"
            )
    hdr = {"build": build, "version": int(version), "n": nxt, "segments": ents}
    hdr.update(extra or {})
    return P.wrap(KIND_VERSION, hdr, b"")


class Segment:
    def __init__(self, buf):
        kind, hdr, body = P.parse(buf)
        if kind != KIND_SEGMENT or hdr.get("coder") != CODER:
            raise JunctionError("not a junction segment")
        self.id = P.pack_id(buf)
        self.header = hdr
        self.id_lo, self.n = int(hdr["id_lo"]), int(hdr["n"])
        self._body = body
        self._j: Optional[Junctions] = None

    @property
    def j(self) -> Junctions:
        if self._j is None:
            self._j = decode_body(self._body, list(self.header["chroms"]))
        return self._j


class JunctionTable:
    """A table version with the segments it can reach. ``fetch(hexid) -> bytes`` supplies segments that are not given (the R2 tier)."""

    def __init__(self, version_buf, segments: Iterable = (), fetch=None):
        kind, hdr, _ = P.parse(version_buf)
        if kind != KIND_VERSION:
            raise JunctionError("not a junction table version")
        self.id = P.pack_id(version_buf)
        self.header = hdr
        self.build = hdr["build"]
        self.n = int(hdr["n"])
        self._ents = hdr["segments"]
        self._lo = np.array([e["id_lo"] for e in self._ents], np.int64)
        self._seg: Dict[str, Segment] = {}
        for s in segments:
            seg = s if isinstance(s, Segment) else Segment(s)
            self._seg[seg.id.hex()] = seg
        self._fetch = fetch
        self._cat: Optional[Junctions] = None
        self._keys = None

    # ------------------------------------------------------------------ segments
    def segment(self, k: int) -> Segment:
        e = self._ents[k]
        s = self._seg.get(e["id"])
        if s is None:
            if self._fetch is None:
                raise JunctionError(
                    f"segment {e['id']} ({e['tier']}, ids {e['id_lo']}..) is not available"
                )
            s = Segment(self._fetch(e["id"]))
            if s.id.hex() != e["id"]:
                raise JunctionError(f"segment {e['id']} failed its digest")
            self._seg[e["id"]] = s
        return s

    def local_tiers(self) -> List[str]:
        return [e["tier"] for e in self._ents if e["id"] in self._seg]

    # ------------------------------------------------------------------ id -> coordinates (O(1) per id once a segment is decoded)
    def lookup(self, ids) -> Junctions:
        """Columns for table ids (any order, repeats allowed)."""
        ids = np.asarray(ids, np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n):
            raise JunctionError(f"junction id outside the table (0..{self.n - 1})")
        k = np.searchsorted(self._lo, ids, side="right") - 1
        if (
            ids.size and k[0] == k[-1] and (k == k[0]).all()
        ):  # the common case: every id in one segment (the core)
            seg = self.segment(int(k[0]))
            j = seg.j
            r = ids - seg.id_lo
            return Junctions(
                j.chroms, j.chrom[r], j.start[r], j.end[r], j.attr[r], j.gene[r], j.n_tx[r]
            )
        cidx: Dict[str, int] = {}
        out_c = np.empty(len(ids), np.uint16)
        out_s = np.empty(len(ids), np.uint32)
        out_e = np.empty(len(ids), np.uint32)
        out_a = np.empty(len(ids), np.uint8)
        out_g = np.empty(len(ids), np.uint32)
        out_t = np.empty(len(ids), np.uint8)
        for seg_k in range(len(self._ents)):  # few segments: one mask each, no sort of the ids
            m = k == seg_k
            if not m.any():
                continue
            seg = self.segment(seg_k)
            j = seg.j
            r = ids[m] - seg.id_lo
            remap = np.array([cidx.setdefault(c, len(cidx)) for c in j.chroms], np.uint16)
            out_c[m] = remap[j.chrom[r]]
            out_s[m], out_e[m], out_a[m], out_g[m], out_t[m] = (
                j.start[r],
                j.end[r],
                j.attr[r],
                j.gene[r],
                j.n_tx[r],
            )
        return Junctions(list(cidx), out_c, out_s, out_e, out_a, out_g, out_t)

    def names(self, ids) -> List[str]:
        return self.lookup(ids).names()

    # ------------------------------------------------------------------ coordinates -> id
    def _index(self):
        if self._keys is None:
            parts, idl = [], []
            cidx: Dict[str, int] = {}
            for k, e in enumerate(self._ents):
                seg = self.segment(k)
                j = seg.j
                remap = np.array([cidx.setdefault(c, len(cidx)) for c in j.chroms], np.uint16)
                parts.append(Junctions(list(cidx), remap[j.chrom], j.start, j.end))
                idl.append(np.arange(seg.id_lo, seg.id_lo + seg.n, dtype=np.int64))
            chroms = list(cidx)
            cat = Junctions(
                chroms,
                np.concatenate([p.chrom for p in parts]) if parts else np.zeros(0, np.uint16),
                np.concatenate([p.start for p in parts]) if parts else np.zeros(0, np.uint32),
                np.concatenate([p.end for p in parts]) if parts else np.zeros(0, np.uint32),
            )
            ids = np.concatenate(idl) if idl else np.zeros(0, np.int64)
            o = np.lexsort((cat.end, cat.start, cat.chrom))
            self._cat, self._cat_ids, self._cidx = cat.take(o), ids[o], cidx
            self._tk = (
                self._cat.chrom.astype(np.uint64) << np.uint64(32)
            ) | self._cat.start.astype(np.uint64)
            self._keys = True
        return self._cat, self._cat_ids, self._cidx

    def find(self, j: Junctions) -> np.ndarray:
        """Table id of each junction (-1 if the table does not hold it): binary search on (chrom, start), then a vectorised scan of
        the few table rows sharing that start (alternative acceptors) for the end."""
        cat, ids, cidx = self._index()
        if not len(j):
            return np.zeros(0, np.int64)
        remap = np.array([cidx.get(c, -1) for c in j.chroms], np.int64)
        qc = remap[j.chrom]
        tk = self._tk
        qk = (np.maximum(qc, 0).astype(np.uint64) << np.uint64(32)) | j.start.astype(np.uint64)
        lo = np.searchsorted(tk, qk, side="left")
        hi = np.searchsorted(tk, qk, side="right")
        out = np.full(len(j), -1, np.int64)
        cand = np.flatnonzero((hi > lo) & (qc >= 0))  # queries whose (chrom, start) the table has
        off = 0
        while cand.size:  # step through the table rows sharing a start; each pass keeps only the unresolved queries
            r = lo[cand] + off
            hit = cat.end[r] == j.end[cand]
            out[cand[hit]] = ids[r[hit]]
            cand = cand[~hit & (r + 1 < hi[cand])]
            off += 1
        return out


# ============================================================================================ annotation inputs
@dataclass
class Annotation:
    """Annotated introns of a build (from the GTF): key -> (gene index + 1, transcripts), and the exon boundary sets."""

    introns: Dict[Tuple[str, int, int], Tuple[int, int]]
    donors: set  # (chrom, start) of annotated introns' first base
    acceptors: set  # (chrom, end)
    strand: Dict[Tuple[str, int, int], int]


def read_gtf(path: str, gene_index: Optional[Dict[str, int]] = None) -> Annotation:
    """Annotated introns from a GTF: consecutive exons of each transcript. ``gene_index`` maps gene ids (without version) to the
    0-based index in the build's exon pack gene table."""
    tx: Dict[str, list] = {}
    meta: Dict[str, Tuple[str, str, str]] = {}
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            attrs = f[8]
            t = attrs.split('transcript_id "', 1)
            g = attrs.split('gene_id "', 1)
            if len(t) < 2 or len(g) < 2:
                continue
            tid = t[1].split('"', 1)[0]
            gid = g[1].split('"', 1)[0].split(".")[0]
            tx.setdefault(tid, []).append((int(f[3]), int(f[4])))
            meta[tid] = (f[0], f[6], gid)
    introns: Dict[Tuple[str, int, int], Tuple[int, int]] = {}
    strand: Dict[Tuple[str, int, int], int] = {}
    for tid, ex in tx.items():
        chrom, sd, gid = meta[tid]
        ex.sort()
        gi = (gene_index or {}).get(gid, -1) + 1
        for (s1, e1), (s2, _e2) in zip(ex[:-1], ex[1:]):
            key = (
                chrom,
                e1,
                s2 - 1,
            )  # 1-based exon end E -> 0-based first intron base E; next start S -> 0-based S - 1
            g0, n0 = introns.get(key, (gi, 0))
            introns[key] = (g0 or gi, n0 + 1)
            strand[key] = 1 if sd == "+" else 2 if sd == "-" else 0
    return Annotation(
        introns, {(c, s) for c, s, _ in introns}, {(c, e) for c, _, e in introns}, strand
    )


def annotation_from_exon_pack(pack) -> Annotation:
    """A proxy when no GTF is at hand: exon boundaries of the vocabulary pack (merged exons per gene), no transcripts, no strand.
    Only the donor/acceptor bits and the gene can be set from it; ``annotated`` needs the GTF."""
    names = pack.names()
    gid_order: Dict[str, int] = {}
    donors, acceptors = {}, {}
    for nm in names:
        gid, rest = nm.split("_", 1)
        coord = rest.rsplit("_", 1)[1]
        c, se = coord.split(":")
        s, e = se.split("-")
        gi = gid_order.setdefault(gid, len(gid_order)) + 1
        donors.setdefault((c, int(e)), gi)
        acceptors.setdefault((c, int(s)), gi)
    ann = Annotation({}, set(donors), set(acceptors), {})
    ann.gene_by_donor = donors  # type: ignore[attr-defined]
    return ann


def motif_from_genome(j: Junctions, genome: Dict[str, str]) -> np.ndarray:
    """STAR motif code of each junction from the reference sequence (``genome``: chrom -> sequence, upper or lower case)."""
    out = np.zeros(len(j), np.uint8)
    for i, (c, s, e) in enumerate(zip(j.chrom.tolist(), j.start.tolist(), j.end.tolist())):
        seq = genome.get(j.chroms[c])
        if seq is None or e > len(seq) or e - s < 4:
            continue
        out[i] = MOTIFS.get((seq[s : s + 2].upper(), seq[e - 2 : e].upper()), 0)
    return out


def read_fasta(path: str, chroms: Optional[set] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    name, buf = None, []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            if line.startswith(">"):
                if name is not None and (chroms is None or name in chroms):
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None and (chroms is None or name in chroms):
        out[name] = "".join(buf)
    return out


def annotate(
    j: Junctions, ann: Optional[Annotation] = None, genome: Optional[Dict[str, str]] = None
) -> Junctions:
    """Fill ``attr`` (strand, motif, annotation bits), ``gene`` and ``n_tx`` from a GTF annotation and/or the genome."""
    attr = np.zeros(len(j), np.uint8)
    gene = np.zeros(len(j), np.uint32)
    ntx = np.zeros(len(j), np.uint8)
    motif = motif_from_genome(j, genome) if genome else np.zeros(len(j), np.uint8)
    gbd = getattr(ann, "gene_by_donor", None) if ann else None
    for i, (c, s, e) in enumerate(zip(j.chrom.tolist(), j.start.tolist(), j.end.tolist())):
        ch = j.chroms[c]
        a = 0
        st = 0
        if ann is not None:
            hit = ann.introns.get((ch, s, e))
            if hit is not None:
                a |= ANNOTATED
                gene[i] = hit[0]
                ntx[i] = min(hit[1], 255)
                st = ann.strand.get((ch, s, e), 0)
            if (ch, s) in ann.donors:
                a |= DONOR_ANN
            if (ch, e) in ann.acceptors:
                a |= ACCEPTOR_ANN
            if gbd is not None and not gene[i]:
                gene[i] = gbd.get((ch, s), 0)
        m = int(motif[i])
        if not st and m:
            st = 1 if m % 2 else 2
        attr[i] = a | (m << 2) | st
    return Junctions(j.chroms, j.chrom, j.start, j.end, attr, gene, ntx)


# ============================================================================================ samples
@dataclass
class SampleRows:
    """How a sample's junction rows are stored: ``ids`` (pack_subset index, ascending; table ids then LOCAL_BASE + k), the local
    junctions (a segment body without ids) and the permutation back to the legacy row order (``legacy_perm[v2_row] = legacy_row``)."""

    ids: np.ndarray
    local: Junctions
    legacy_perm: np.ndarray


def sample_rows(table: JunctionTable, names: Sequence[str]) -> SampleRows:
    """Map a sample's legacy junction names (legacy row order) to table ids and local junctions; store rows are in id order."""
    j = parse_names(names)
    tid = table.find(j)
    loc = np.flatnonzero(tid < 0)
    lj = j.take(loc)
    lo = lj.genomic_order()
    local_rows = loc[lo]
    ids = np.empty(len(j), np.int64)
    ids[tid >= 0] = tid[tid >= 0]
    ids[local_rows] = LOCAL_BASE + np.arange(len(local_rows), dtype=np.int64)
    perm = np.argsort(ids, kind="stable")  # store row r holds legacy row perm[r]
    return SampleRows(ids[perm].astype(np.uint32), j.take(local_rows), perm.astype(np.uint32))


def encode_local(j: Junctions) -> bytes:
    """The sample's local junction list (member body): u16 chromosome count, NUL-joined names, then a segment body."""
    o = j.genomic_order()
    if not np.array_equal(o, np.arange(len(j))):
        raise JunctionError("local junctions must be in genomic order")
    used = sorted(set(j.chrom.tolist()), key=lambda i: j.chroms[i])
    remap = np.zeros(max(len(j.chroms), 1), np.uint16)
    remap[used] = np.arange(len(used), dtype=np.uint16)
    chroms = [j.chroms[i] for i in used]
    nb = "\0".join(chroms).encode()
    jj = Junctions(chroms, remap[j.chrom], j.start, j.end, j.attr, j.gene, j.n_tx)
    return struct.pack("<HI", len(chroms), len(nb)) + nb + encode_body(jj)


def decode_local(buf) -> Junctions:
    nc, nl = struct.unpack_from("<HI", buf, 0)
    chroms = bytes(buf[6 : 6 + nl]).decode().split("\0") if nc else []
    return decode_body(bytes(buf[6 + nl :]), chroms)


def encode_perm(perm: np.ndarray) -> bytes:
    """legacy_perm as zigzag deltas, byte planes, zstd (the legacy order is close to genomic order, so deltas are small)."""
    return _enc_stream(_zigzag(np.diff(np.concatenate([[0], perm.astype(np.int64)])) % (1 << 32)))


def decode_perm(buf) -> np.ndarray:
    z, _ = _dec_stream(buf, 0)
    return np.cumsum(_unzigzag(z), dtype=np.uint32).astype(np.int64)


def row_names(table: JunctionTable, ids: np.ndarray, local: Optional[Junctions]) -> List[str]:
    """Names of store rows (ids from ``SampleRows.ids``)."""
    ids = np.asarray(ids, np.int64)
    t = ids < LOCAL_BASE
    out: List[str] = [""] * len(ids)
    if t.any():
        for i, nm in zip(np.flatnonzero(t).tolist(), table.names(ids[t])):
            out[i] = nm
    if (~t).any():
        if local is None:
            raise JunctionError(
                "rows refer to local junctions, but no local junction list was given"
            )
        ln = local.names()
        for i, k in zip(np.flatnonzero(~t).tolist(), (ids[~t] - LOCAL_BASE).tolist()):
            out[i] = ln[k]
    return out


# ============================================================================================ legacy order (bit-exact regeneration)
def permute_rows(indptr, indices, data, row_map) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSC with row ``i`` renamed ``row_map[i]``; row ids sorted again inside every column (same indptr)."""
    indptr = np.asarray(indptr, np.int64)
    new = np.asarray(row_map, np.int64)[np.asarray(indices, np.int64)]
    col = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    o = np.lexsort((new, col))
    return indptr, new[o], np.asarray(data)[o]


def to_legacy(indptr, indices, data, ids, table: JunctionTable, local_blob=None, perm_blob=None):
    """A junction matrix stored in table order -> (indptr, indices, data, row names) in the legacy order, when the
    ``.legacy_order`` member is given (else the table order and its names)."""
    local = decode_local(local_blob) if local_blob else None
    names = row_names(table, np.asarray(ids, np.int64), local)
    if perm_blob is None:
        return np.asarray(indptr, np.int64), np.asarray(indices), np.asarray(data), names
    perm = decode_perm(perm_blob)
    if len(perm) != len(names) or not np.array_equal(np.sort(perm), np.arange(len(perm))):
        raise JunctionError("the legacy order is not a permutation of the matrix rows")
    ip, ix, dt = permute_rows(indptr, indices, data, perm)
    legacy_names: List[str] = [""] * len(names)
    for r, k in enumerate(perm.tolist()):
        legacy_names[k] = names[r]
    return ip, ix, dt, legacy_names


def from_legacy(indptr, indices, data, names: Sequence[str], table: JunctionTable):
    """A legacy junction matrix (rows in the pipeline's order) -> (indptr, indices, data) in table order, the row ids
    (pack_subset index), the ``.local`` member body (or None) and the ``.legacy_order`` member body."""
    rows = sample_rows(table, names)
    inv = np.empty(len(rows.legacy_perm), np.int64)
    inv[rows.legacy_perm] = np.arange(len(rows.legacy_perm))
    ip, ix, dt = permute_rows(indptr, indices, data, inv)
    local = encode_local(rows.local) if len(rows.local) else None
    return ip, ix, dt, rows.ids, local, encode_perm(rows.legacy_perm)


# ============================================================================================ published versions
def published_versions(index_entries: Sequence[dict], build: str) -> List[dict]:
    """The ``junction-version`` entries of a pack index (``store/ref/index.json`` ``packs``) for one reference build."""
    return [
        e for e in index_entries if e.get("kind") == "junction-version" and e.get("build") == build
    ]


def published_table(
    build: str, store, index_entries: Optional[Sequence[dict]] = None
) -> Optional[JunctionTable]:
    """The newest published junction table version for ``build`` (highest header ``version``), or None if the index has none.

    ``store`` is a ``packs.PackStore`` that can supply the version and its segments (``download=True`` fetches them from the
    data host, sha256-checked against its index). ``index_entries`` defaults to the remote ``store/ref/index.json``."""
    if index_entries is None:
        index_entries = fetch_index()
    cands = published_versions(index_entries, build)
    if not cands:
        return None
    tables = [store.get(e["id"]) for e in cands]
    tables = [t for t in tables if isinstance(t, JunctionTable)]
    if not tables:
        return None
    top = max(int(t.header.get("version", 0)) for t in tables)
    best = [t for t in tables if int(t.header.get("version", 0)) == top]
    if len(best) > 1:
        raise JunctionError(
            f"{len(best)} junction table versions {top} are published for {build}: the index is ambiguous"
        )
    return best[0]


def fetch_index(remote: Optional[str] = None) -> List[dict]:
    """``packs`` of the remote ``store/ref/index.json`` (the data host's, or ``remote``)."""
    import json
    import urllib.error
    import urllib.request

    urls = [(remote or P.REMOTE) + "index.json"]
    if remote is None:  # DUAL: the old v2/ref/ prefix
        urls.append(P.REMOTE.replace("/store/ref/", "/v2/ref/") + "index.json")
    for i, url in enumerate(urls):
        req = urllib.request.Request(url, headers={"User-Agent": P.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return list(json.loads(r.read().decode("utf-8"))["packs"])
        except urllib.error.HTTPError as e:
            if e.code != 404 or i + 1 == len(urls):
                raise
    raise RuntimeError("unreachable")
