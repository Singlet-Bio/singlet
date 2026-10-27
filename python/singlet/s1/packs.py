# SPDX-License-Identifier: MIT
"""Side-information packs (docs/format-s1.md section 6): row vocabularies and barcode whitelists.

File wrapper (``SPK2``)::

    "SPK2"  u16 version=1  u16 kind  u32 header_len  header JSON (canonical)  body
    kind: 1 row vocabulary, 2 barcode whitelist, 3 composite

Pack id = digest16 (first 16 bytes of SHA-256) of the whole file; matrices store the first 8 bytes. Packs are immutable.

Bodies (the structured coder is the one measured in bench/compression/notes/w4_vocab.md, "SVP0"):

* kind 1, header ``member`` = ``exon``, ``coder`` = ``SVP0``: gene table + per-exon lengths and gaps (zstd-19 streams). Expands to the
  exon row names, the intron row names (the gaps between consecutive exons of a gene) and ``feature_vocab.json``.
* kind 1, ``member`` = ``intron``, ``coder`` = ``SVPD``: body = digest16 of the exon pack it derives from (a few bytes).
* kind 3 (barnyard): ``member`` ``exon`` or ``intron``; body = ``u8 n_parts`` then per part ``16 B component pack digest16, u8 tag_len,
  tag``; the names are the concatenation of the components' names with the species tag inserted before the chromosome.
* kind 2: ``u8 k, u32 n, u8 mode(0)`` then zstd-19 of LEB128 gaps of the sorted 2-bit barcode codes (A0 C1 G2 T3, first symbol most
  significant, so numeric order == lexicographic order).

Reading is vectorised numpy (no per-name Python work until names are asked for): ``VocabPack.names(member)`` is the byte-exact
expansion, ``VocabPack.name_at`` / ``names_at`` are lazy lookups by row index.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import util

MAGIC = b"SPK2"
VERSION = 1
KIND_VOCAB, KIND_WHITELIST, KIND_COMPOSITE = 1, 2, 3
PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs")
REMOTE = "https://data.singlet.bio/s1/ref/"
USER_AGENT = "singlet/1.0 (+https://singlet.bio)"


class PackError(ValueError):
    pass


# =============================================================================================== wrapper
def wrap(kind: int, header: dict, body: bytes) -> bytes:
    h = util.canon_json(header)
    return MAGIC + struct.pack("<HHI", VERSION, kind, len(h)) + h + body


def parse(buf) -> tuple[int, dict, bytes]:
    if len(buf) < 12 or bytes(buf[:4]) != MAGIC:
        raise PackError("not an SPK2 pack")
    ver, kind, hl = struct.unpack_from("<HHI", buf, 4)
    if ver != VERSION:
        raise PackError("unsupported pack version %d" % ver)
    if 12 + hl > len(buf):
        raise PackError("truncated pack header")
    header = json.loads(bytes(buf[12 : 12 + hl]).decode("utf-8"))
    return kind, header, bytes(buf[12 + hl :])


def pack_id(buf) -> bytes:
    """digest16 of the whole pack file."""
    return util.d16(buf)


# =============================================================================================== integer streams
def _leb_enc(vals: np.ndarray) -> bytes:
    return util.leb128_encode(vals)


def _planes_enc(vals: np.ndarray) -> bytes:
    v = vals.astype(np.uint32)
    return b"".join(((v >> (8 * k)) & 255).astype(np.uint8).tobytes() for k in range(4))


def _planes_dec(buf: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(buf, np.uint8).reshape(4, n).astype(np.uint64)
    return a[0] | (a[1] << np.uint64(8)) | (a[2] << np.uint64(16)) | (a[3] << np.uint64(24))


def _pack_stream(vals) -> bytes:
    """u8 mode (0 LEB128, 1 four byte planes), u32 n, u32 clen, zstd-19 frame (the smaller of the two)."""
    vals = np.asarray(vals, np.uint64)
    best = None
    for mode, enc in ((0, _leb_enc), (1, _planes_enc)):
        z = util.zstd_compress(enc(vals), 19)
        if best is None or len(z) < len(best[1]):
            best = (mode, z)
    return struct.pack("<BII", best[0], len(vals), len(best[1])) + best[1]


def _unpack_stream(buf, p: int):
    mode, n, cl = struct.unpack_from("<BII", buf, p)
    p += 9
    raw = util.zstd_decompress(buf[p : p + cl])
    p += cl
    return (util.leb128_decode(raw, n) if mode == 0 else _planes_dec(raw, n)), p


def _pack_blob(b: bytes) -> bytes:
    z = util.zstd_compress(b, 19)
    return struct.pack("<I", len(z)) + z


def _unpack_blob(buf, p: int):
    (cl,) = struct.unpack_from("<I", buf, p)
    p += 4
    return util.zstd_decompress(buf[p : p + cl]), p + cl


# =============================================================================================== SVP0 structure
@dataclass
class VocabStruct:
    build: str
    schema: str
    prefix: str
    width: int
    nums: np.ndarray  # int64 per gene (numeric part of the gene id)
    kind: np.ndarray  # uint8 per gene: 0 empty symbol, 1 symbol == gene id, 2 explicit
    syms: list  # explicit symbols in gene order
    chroms: list
    gch: np.ndarray  # chromosome index per gene
    nex: np.ndarray  # exons per gene
    gstart: np.ndarray  # first exon start per gene
    elen: np.ndarray  # end - start per exon
    egap: np.ndarray  # gap to the next exon of the same gene (introns)


_COORD = re.compile(r"^(.*):(\d+)-(\d+)$")


def _parse_rows(rows, kind, genes, tag=""):
    gids, syms, chroms, starts, ends = [], [], [], [], []
    gmap = {g: n for g, n in genes}
    mid = "intron_" if kind == "intron" else ""
    for i, r in enumerate(rows):
        u = r.find("_")
        if u <= 0:
            raise PackError("row %d: no gene id separator: %r" % (i, r))
        gid = r[:u]
        mt = _COORD.match(r[u + 1 :])
        if not mt:
            raise PackError("row %d: no coordinates: %r" % (i, r))
        pre, a, b = mt.group(1), mt.group(2), mt.group(3)
        if str(int(a)) != a or str(int(b)) != b:
            raise PackError("row %d: non canonical integer: %r" % (i, r))
        if gid not in gmap:
            raise PackError("row %d: gene %s not in the gene table" % (i, gid))
        cands = [gmap[gid]] + ([""] if gmap[gid] == gid else [])
        chrom = None
        sym = None
        for s in cands:
            head = s + "_" + mid + tag
            if pre.startswith(head):
                chrom, sym = pre[len(head) :], s
                break
        if chrom is None or not chrom:
            raise PackError("row %d: prefix does not match the gene table symbol: %r" % (i, r))
        gids.append(gid)
        syms.append(sym)
        chroms.append(chrom)
        starts.append(int(a))
        ends.append(int(b))
    return gids, syms, chroms, starts, ends


def build_struct(exon: list, intron: list, vocab: dict) -> VocabStruct:
    """Names + feature_vocab.json -> structure. Raises PackError if the lists do not follow the grammar (then no pack applies)."""
    genes_tab = [(g["gene_id"], g["gene_name"]) for g in vocab["genes"]]
    gid, sym, chrom, st, en = _parse_rows(exon, "exon", genes_tab)
    grp = []
    seen = set()
    cur = None
    for i in range(len(gid)):
        if cur is None or cur[0] != gid[i]:
            if gid[i] in seen:
                raise PackError("gene %s is not contiguous" % gid[i])
            seen.add(gid[i])
            cur = [gid[i], sym[i], chrom[i], []]
            grp.append(cur)
        elif cur[1] != sym[i] or cur[2] != chrom[i]:
            raise PackError("symbol/chromosome changes inside gene %s" % gid[i])
        cur[3].append((st[i], en[i]))
    if [g[0] for g in grp] != [g for g, _ in genes_tab]:
        raise PackError("gene order differs from feature_vocab.json")
    for g, (_, vn) in zip(grp, genes_tab):
        if (g[1] if g[1] else g[0]) != vn:
            raise PackError("gene_name != symbol/id")
    ids = [g[0] for g in grp]
    pref = re.match(r"^[A-Za-z]+", ids[0]).group(0)
    width = len(ids[0]) - len(pref)
    nums = np.array([int(i[len(pref) :]) for i in ids], np.int64)
    if not all(i == pref + str(n).zfill(width) for i, n in zip(ids, nums)):
        raise PackError("gene ids are not <prefix><zero padded number>")
    if not (np.diff(nums) > 0).all():
        raise PackError("gene ids are not increasing")
    chroms, cidx, gch = [], {}, []
    for g in grp:
        if g[2] not in cidx:
            cidx[g[2]] = len(chroms)
            chroms.append(g[2])
        gch.append(cidx[g[2]])
    kind = np.array([0 if g[1] == "" else (1 if g[1] == g[0] else 2) for g in grp], np.uint8)
    elen, egap = [], []
    for g in grp:
        ee = g[3]
        for i, (a, b) in enumerate(ee):
            if b < a:
                raise PackError("exon end before start")
            elen.append(b - a)
            if i + 1 < len(ee):
                if ee[i + 1][0] < b:
                    raise PackError("overlapping exons")
                egap.append(ee[i + 1][0] - b)
    S = VocabStruct(
        build=vocab["reference_build"],
        schema=vocab.get("schema_version", "1.0"),
        prefix=pref,
        width=width,
        nums=nums,
        kind=kind,
        syms=[g[1] for g, k in zip(grp, kind) if k == 2],
        chroms=chroms,
        gch=np.array(gch, np.uint32),
        nex=np.array([len(g[3]) for g in grp], np.uint32),
        gstart=np.array([g[3][0][0] for g in grp], np.uint64),
        elen=np.array(elen, np.uint64),
        egap=np.array(egap, np.uint64),
    )
    # the introns must be exactly the gaps; otherwise the derived intron list would be wrong
    ex2, it2, vg2 = expand_struct(S)
    if ex2 != exon or it2 != intron:
        raise PackError("names are not reproduced by the structured coder")
    return S


def expand_struct(S: VocabStruct, tag: str = ""):
    """Reference (list based) expansion: exon names, intron names, gene table [(gene_id, gene_name)]."""
    a = _arrays(S)
    return _names_exon(S, a, tag), _names_intron(S, a, tag), _gene_table(S)


def encode_struct(S: VocabStruct) -> bytes:
    b = S.build.encode()
    hdr = b"SVP0" + struct.pack(
        "<HHBBHII", 1, 0, len(S.prefix), S.width, len(S.chroms), len(S.nums), len(S.elen)
    )
    hdr += struct.pack("<B", len(b)) + b + S.prefix.encode() + S.schema.encode().ljust(8, b"\0")
    nums = S.nums.astype(np.uint64)
    d = np.concatenate([nums[:1], np.diff(nums).astype(np.uint64) - np.uint64(1)])
    parts = [
        hdr,
        _pack_stream(d),
        _pack_blob(S.kind.tobytes()),
        _pack_blob(b"\0".join(s.encode() for s in S.syms)),
        _pack_blob(b"\0".join(c.encode() for c in S.chroms)),
        _pack_stream(S.gch),
        _pack_stream(S.nex.astype(np.uint64) - np.uint64(1)),
        _pack_stream(S.gstart),
        _pack_stream(S.elen),
        _pack_stream(S.egap),
    ]
    return b"".join(parts)


def decode_struct(buf) -> VocabStruct:
    buf = bytes(buf)
    if buf[:4] != b"SVP0":
        raise PackError("not an SVP0 body")
    ver, flags, pl, width, nchrom, ngenes, nexons = struct.unpack_from("<HHBBHII", buf, 4)
    p = 4 + struct.calcsize("<HHBBHII")
    bl = buf[p]
    p += 1
    build = buf[p : p + bl].decode()
    p += bl
    prefix = buf[p : p + pl].decode()
    p += pl
    schema = buf[p : p + 8].rstrip(b"\0").decode()
    p += 8
    d, p = _unpack_stream(buf, p)
    nums = np.cumsum(np.concatenate([d[:1], d[1:] + np.uint64(1)]).astype(np.int64))
    kb, p = _unpack_blob(buf, p)
    kind = np.frombuffer(kb, np.uint8)
    sb, p = _unpack_blob(buf, p)
    syms = [x.decode() for x in sb.split(b"\0")] if (kind == 2).any() else []
    cb, p = _unpack_blob(buf, p)
    chroms = [x.decode() for x in cb.split(b"\0")]
    gch, p = _unpack_stream(buf, p)
    nx, p = _unpack_stream(buf, p)
    gstart, p = _unpack_stream(buf, p)
    elen, p = _unpack_stream(buf, p)
    egap, p = _unpack_stream(buf, p)
    if p != len(buf):
        raise PackError("trailing bytes in the SVP0 body")
    if len(nums) != ngenes or len(elen) != nexons or len(chroms) != nchrom:
        raise PackError("SVP0 counts do not match the header")
    return VocabStruct(
        build,
        schema,
        prefix,
        width,
        nums,
        kind,
        syms,
        chroms,
        gch.astype(np.int64),
        (nx + np.uint64(1)).astype(np.int64),
        gstart.astype(np.int64),
        elen.astype(np.int64),
        egap.astype(np.int64),
    )


class _Arr:
    """Derived per-exon / per-intron coordinate arrays (vectorised)."""

    __slots__ = (
        "gidx",
        "ex_start",
        "ex_end",
        "last",
        "in_exon",
        "in_start",
        "in_end",
        "cum_ex",
        "cum_in",
        "gene_ids",
        "heads",
    )


def _gene_table(S: VocabStruct):
    gids = [S.prefix + str(int(n)).zfill(S.width) for n in S.nums]
    out = []
    ns = 0
    kind = S.kind.tolist()
    for g, gid in enumerate(gids):
        if kind[g] == 2:
            out.append((gid, S.syms[ns]))
            ns += 1
        else:
            out.append((gid, gid))
    return out


def _arrays(S: VocabStruct) -> _Arr:
    a = _Arr()
    G = len(S.nums)
    nex = S.nex.astype(np.int64)
    nE = int(nex.sum())
    a.gidx = np.repeat(np.arange(G, dtype=np.int64), nex)
    first = np.cumsum(nex) - nex
    last = np.zeros(nE, dtype=bool)
    last[first + nex - 1] = True
    a.last = last
    gapfull = np.zeros(nE, dtype=np.int64)
    gapfull[~last] = S.egap
    step = S.elen.astype(np.int64) + gapfull
    excl = np.cumsum(step) - step
    a.ex_start = S.gstart.astype(np.int64)[a.gidx] + excl - excl[first][a.gidx]
    a.ex_end = a.ex_start + S.elen.astype(np.int64)
    a.in_exon = np.flatnonzero(~last)
    a.in_start = a.ex_end[a.in_exon]
    a.in_end = a.in_start + S.egap.astype(np.int64)
    a.cum_ex = np.cumsum(nex)
    a.cum_in = np.cumsum(nex - 1)
    a.gene_ids = None
    a.heads = None
    return a


def _heads(S: VocabStruct, a: _Arr):
    if a.heads is not None:
        return a.heads
    tbl = _gene_table(S)
    kind = S.kind.tolist()
    heads = []
    ns = 0
    for g, (gid, _name) in enumerate(tbl):
        if kind[g] == 0:
            sym = ""
        elif kind[g] == 1:
            sym = gid
        else:
            sym = S.syms[ns]
        if kind[g] == 2:
            ns += 1
        heads.append(gid + "_" + sym + "_")
    a.heads = heads
    return heads


def _names_exon(S, a, tag=""):
    heads = _heads(S, a)
    pre = [h + tag + S.chroms[c] + ":" for h, c in zip(heads, S.gch.tolist())]
    return [
        pre[g] + "%d-%d" % (s, e)
        for g, s, e in zip(a.gidx.tolist(), a.ex_start.tolist(), a.ex_end.tolist())
    ]


def _names_intron(S, a, tag=""):
    heads = _heads(S, a)
    pre = [h + "intron_" + tag + S.chroms[c] + ":" for h, c in zip(heads, S.gch.tolist())]
    g = a.gidx[a.in_exon].tolist()
    return [
        pre[gg] + "%d-%d" % (s, e) for gg, s, e in zip(g, a.in_start.tolist(), a.in_end.tolist())
    ]


def vocab_json(build: str, schema: str, genes) -> bytes:
    """``feature_vocab.json`` exactly as the pipeline writes it (``json.dumps(..., indent=2)``, no trailing newline)."""
    return json.dumps(
        {
            "schema_version": schema,
            "reference_build": build,
            "n_genes": len(genes),
            "genes": [{"gene_id": a, "gene_name": b} for a, b in genes],
        },
        indent=2,
    ).encode()


# =============================================================================================== VocabPack
class VocabPack:
    """A row-vocabulary pack (kind 1 or 3) resolved to usable arrays.

    ``member`` is ``"exon"`` or ``"intron"``. ``names()`` -> ``list[str]`` byte-exact; ``blob()`` -> NUL-joined bytes;
    ``name_at(i)`` / ``names_at(idx)`` lazy lookups; ``row_gene(i)`` the gene index of a row; ``vocab_json()`` the regenerated
    ``feature_vocab.json`` (exon packs).
    """

    def __init__(self, buf, store: "PackStore"):
        self.bytes = bytes(buf)
        self.id = pack_id(self.bytes)
        kind, hdr, body = parse(self.bytes)
        if kind not in (KIND_VOCAB, KIND_COMPOSITE):
            raise PackError("not a vocabulary pack")
        self.kind, self.header, self._body = kind, hdr, body
        self.member = hdr["member"]
        self.build = hdr["build"]
        self._store = store
        self._parts = None  # list of (VocabPack, tag) for composites
        self._S = None
        self._A = None
        self._exon_pack = None  # SVPD: the exon pack the introns derive from
        self._names = None
        self._blob = None
        if kind == KIND_VOCAB and hdr.get("coder") == "SVP0":
            if self.member != "exon":
                raise PackError("SVP0 holds the exon member")
        elif kind == KIND_VOCAB and hdr.get("coder") == "SVPD":
            pass
        elif kind == KIND_COMPOSITE:
            pass
        else:
            raise PackError("unknown pack coder %r" % hdr.get("coder"))

    # ------------------------------------------------------------------ lazy structure
    def _struct(self):
        if self._S is None:
            self._S = decode_struct(self._body)
            self._A = _arrays(self._S)
        return self._S, self._A

    def _exon_source(self) -> "VocabPack":
        if self._exon_pack is None:
            if self.kind == KIND_VOCAB and self.header.get("coder") == "SVPD":
                self._exon_pack = self._store.get_id(self._body[:16])
            else:
                self._exon_pack = self
        return self._exon_pack

    def _components(self):
        if self._parts is None:
            n = self._body[0]
            p = 1
            parts = []
            for _ in range(n):
                cid = self._body[p : p + 16]
                tl = self._body[p + 16]
                tag = self._body[p + 17 : p + 17 + tl].decode()
                p += 17 + tl
                parts.append((self._store.get_id(cid), tag))
            self._parts = parts
        return self._parts

    # ------------------------------------------------------------------ expansion
    def n(self) -> int:
        return int(self.header["n"])

    def names(self) -> list:
        if self._names is None:
            self._names = self._expand("", self.member)
        return self._names

    def _expand(self, tag: str, member: str) -> list:
        if self.kind == KIND_COMPOSITE:
            out = []
            for pk, t in self._components():
                out.extend(pk._expand(tag + t, member))
            return out
        src = self._exon_source()
        S, A = src._struct()
        return _names_exon(S, A, tag) if member == "exon" else _names_intron(S, A, tag)

    def blob(self) -> bytes:
        """NUL-terminated concatenation of the names (the legacy on-disk row-name form)."""
        if self._blob is None:
            self._blob = util.join_nul(self.names())
        return self._blob

    def matches(self, names: Sequence[str]) -> bool:
        return len(names) == self.n() and list(names) == self.names()

    # ------------------------------------------------------------------ lazy lookups
    def _layout(self):
        """(list of (pack, tag, start_row)) for lookups across composite parts."""
        if self.kind == KIND_COMPOSITE:
            out, row = [], 0
            for pk, t in self._components():
                out.append((pk, t, row))
                row += pk.n()
            return out
        return [(self, "", 0)]

    def name_at(self, i: int) -> str:
        return self.names_at([i])[0]

    def names_at(self, idx) -> list:
        idx = np.asarray(idx, dtype=np.int64)
        res = [None] * len(idx)
        for pk, tag, start in self._layout():
            sel = np.flatnonzero((idx >= start) & (idx < start + pk.n()))
            if not len(sel):
                continue
            src = pk._exon_source()
            S, A = src._struct()
            local = idx[sel] - start
            if pk.member == "exon":
                g = A.gidx[local]
                st, en = A.ex_start[local], A.ex_end[local]
                inf = ""
            else:
                ex = A.in_exon[local]
                g = A.gidx[ex]
                st, en = A.in_start[local], A.in_end[local]
                inf = "intron_"
            heads = _heads(S, A)
            for k, gg, s, e in zip(sel.tolist(), g.tolist(), st.tolist(), en.tolist()):
                res[k] = "%s%s%s%s:%d-%d" % (heads[gg], inf, tag, S.chroms[int(S.gch[gg])], s, e)
        return res

    def row_gene(self) -> np.ndarray:
        """Gene index (into ``gene_table()``) of every row; composites number genes across their parts."""
        out, goff = [], 0
        for pk, _tag, _start in self._layout():
            src = pk._exon_source()
            S, A = src._struct()
            g = A.gidx if pk.member == "exon" else A.gidx[A.in_exon]
            out.append(g + goff)
            goff += len(S.nums)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def gene_table(self) -> list:
        if self.kind == KIND_COMPOSITE:
            t = []
            for pk, _tag in self._components():
                t.extend(pk.gene_table())
            return t
        src = self._exon_source()
        S, _ = src._struct()
        return _gene_table(S)

    def vocab_json(self) -> bytes:
        """The study-level ``feature_vocab.json`` regenerated byte for byte."""
        if self.kind == KIND_COMPOSITE:
            sch = self._components()[0][0]._exon_source()._struct()[0].schema
            return vocab_json(self.build, sch, self.gene_table())
        src = self._exon_source()
        S, _ = src._struct()
        return vocab_json(S.build, S.schema, _gene_table(S))


# =============================================================================================== whitelists
_B2 = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    _B2[_c] = _i
_ALPHA = np.frombuffer(b"ACGT", dtype=np.uint8)


def barcodes_to_codes(barcodes: Sequence[str], k: int) -> Optional[np.ndarray]:
    """2-bit codes (uint64) of equal-length ACGT strings; None if any barcode is not k-long ACGT."""
    if not len(barcodes):
        return np.zeros(0, np.uint64)
    if any(len(b) != k for b in barcodes):
        return None
    try:
        raw = np.array(barcodes, dtype="S%d" % k)
    except Exception:  # non-ASCII
        return None
    arr = np.frombuffer(raw.tobytes(), dtype=np.uint8).reshape(len(barcodes), k)
    # ``S`` arrays drop trailing NULs: a shorter barcode would be padded with 0 -> invalid symbol 255
    d = _B2[arr]
    if (d == 255).any():
        return None
    codes = np.zeros(len(barcodes), np.uint64)
    for j in range(k):
        codes = (codes << np.uint64(2)) | d[:, j].astype(np.uint64)
    return codes


def codes_to_sarray(codes: np.ndarray, k: int) -> np.ndarray:
    """2-bit codes -> numpy ``S<k>`` array of barcodes (no Python strings)."""
    codes = np.asarray(codes, np.uint64)
    out = np.empty((len(codes), k), np.uint8)
    for j in range(k):
        out[:, k - 1 - j] = _ALPHA[((codes >> np.uint64(2 * j)) & np.uint64(3)).astype(np.int64)]
    return (
        np.frombuffer(out.tobytes(), dtype="S%d" % k)
        if len(codes)
        else np.zeros(0, dtype="S%d" % k)
    )


def codes_to_barcodes(codes: np.ndarray, k: int) -> list:
    return codes_to_sarray(codes, k).astype(str).tolist() if len(codes) else []


class Whitelist:
    """Sorted barcode set (kind 2 pack)."""

    def __init__(self, buf):
        self.bytes = bytes(buf)
        self.id = pack_id(self.bytes)
        kind, hdr, body = parse(self.bytes)
        if kind != KIND_WHITELIST:
            raise PackError("not a whitelist pack")
        self.header = hdr
        k, n, mode = struct.unpack_from("<BIB", body, 0)
        if mode != 0:
            raise PackError("unknown whitelist mode")
        gaps = util.leb128_decode(util.zstd_decompress(body[6:]), n)
        self.k, self.n = k, n
        self.codes = np.cumsum(gaps).astype(np.uint64)
        self.names = list(hdr.get("names", [hdr.get("protocol", "")]))

    def index_of(self, barcodes: Sequence[str]) -> Optional[np.ndarray]:
        """Index of every barcode in the sorted whitelist; None if any is absent or malformed."""
        c = barcodes_to_codes(barcodes, self.k)
        return None if c is None else self.index_of_codes(c)

    def index_of_codes(self, c: np.ndarray) -> Optional[np.ndarray]:
        """Same for barcodes already converted with :func:`barcodes_to_codes` (this whitelist's ``k``)."""
        pos = np.searchsorted(self.codes, c)
        pos_c = np.minimum(pos, self.n - 1) if self.n else pos
        if len(c) and (self.n == 0 or (self.codes[pos_c] != c).any()):
            return None
        return pos.astype(np.int64)

    def barcodes(self, idx) -> list:
        return codes_to_barcodes(self.codes[np.asarray(idx, np.int64)], self.k)

    def all_barcodes(self) -> list:
        return codes_to_barcodes(self.codes, self.k)

    def as_sarray(self) -> np.ndarray:
        """The sorted whitelist as a numpy ``S<k>`` array (what the STARsolo recipes want; avoids 3.7 M Python strings)."""
        return codes_to_sarray(self.codes, self.k)


def build_whitelist(barcodes: Sequence[str], header: dict) -> bytes:
    """Whitelist pack file from a barcode list (duplicates and order are ignored: the pack stores the sorted set)."""
    ks = {len(b) for b in barcodes}
    if len(ks) != 1:
        raise PackError("whitelist barcodes must have one length")
    k = ks.pop()
    codes = barcodes_to_codes(sorted(set(barcodes)), k)
    if codes is None:
        raise PackError("whitelist barcodes must be ACGT")
    codes = np.unique(codes)
    gaps = np.concatenate([codes[:1], np.diff(codes)])
    body = struct.pack("<BIB", k, len(codes), 0) + util.zstd_compress(util.leb128_encode(gaps), 19)
    h = dict(header)
    h.update(
        {
            "k": k,
            "n": int(len(codes)),
            "sha256_names": hashlib.sha256(
                ("\n".join(codes_to_barcodes(codes, k)) + "\n").encode()
            ).hexdigest(),
        }
    )
    return wrap(KIND_WHITELIST, h, body)


# =============================================================================================== building vocabulary packs
def build_exon_pack(S: VocabStruct, header_extra: Optional[dict] = None) -> bytes:
    a = _arrays(S)
    ex = _names_exon(S, a)
    it = _names_intron(S, a)
    h = {
        "build": S.build,
        "member": "exon",
        "coder": "SVP0",
        "n": len(ex),
        "n_genes": len(S.nums),
        "n_intron": len(it),
        "sha256_names": hashlib.sha256(util.join_nul(ex)).hexdigest(),
    }
    h.update(header_extra or {})
    return wrap(KIND_VOCAB, h, encode_struct(S))


def build_intron_pack(exon_pack_bytes: bytes) -> bytes:
    kind, eh, _ = parse(exon_pack_bytes)
    h = {
        "build": eh["build"],
        "member": "intron",
        "coder": "SVPD",
        "n": eh["n_intron"],
        "derived_from": pack_id(exon_pack_bytes).hex(),
    }
    return wrap(KIND_VOCAB, h, pack_id(exon_pack_bytes))


def build_composite(
    build: str, member: str, parts: Sequence[tuple], n: int, sha256_names: str
) -> bytes:
    body = bytes([len(parts)])
    for pid, tag in parts:
        t = tag.encode()
        body += pid + bytes([len(t)]) + t
    return wrap(
        KIND_COMPOSITE,
        {"build": build, "member": member, "coder": "SPKC", "n": n, "sha256_names": sha256_names},
        body,
    )


# =============================================================================================== PackStore
def user_cache_dir() -> str:
    base = util.env("CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "singlet", "s1")
    return os.path.join(base, "ref")


class PackStore:
    """Finds packs by id (any prefix of at least 8 bytes = 16 hex chars) in: the package directory, ``$SINGLET_S1_PACKS``
    (``os.pathsep`` separated), the user cache, extra directories; optionally downloads missing packs (sha256 checked against
    ``index.json`` of the remote)."""

    def __init__(
        self, dirs: Optional[Sequence[str]] = None, download: bool = False, remote: str = REMOTE
    ):
        self.dirs = (
            list(dirs)
            if dirs is not None
            else [PKG_DIR]
            + [d for d in util.env("PACKS").split(os.pathsep) if d]
            + [user_cache_dir()]
        )
        self.download = download
        self.remote = remote
        self._files: dict[str, str] = {}
        self._obj: dict[str, object] = {}
        self._index: list[dict] = []
        self.rescan()

    def rescan(self):
        self._files.clear()
        self._index = []
        for d in self.dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(glob.glob(os.path.join(d, "*.pack"))):
                pid = os.path.basename(f)[:-5]
                if not re.fullmatch(r"[0-9a-f]{32}", pid) or pid in self._files:
                    continue
                try:
                    with open(f, "rb") as fh:
                        head = fh.read(12)
                        ver, kind, hl = struct.unpack_from("<HHI", head, 4)
                        hdr = json.loads(fh.read(hl).decode("utf-8"))
                except Exception:
                    continue
                if head[:4] != MAGIC:
                    continue
                self._files[pid] = f
                ent = {
                    "id": pid,
                    "kind": "whitelist" if kind == KIND_WHITELIST else "vocab",
                    "bytes": os.path.getsize(f),
                    "build": hdr.get("build"),
                    "member": hdr.get("member"),
                    "protocol": hdr.get("protocol"),
                    "n": hdr.get("n"),
                    "names": hdr.get("names"),
                }
                self._index.append({k: v for k, v in ent.items() if v is not None})

    def add_dir(self, d: str):
        if d not in self.dirs:
            self.dirs.append(d)
        self.rescan()

    # ------------------------------------------------------------------ lookup
    def ids(self) -> list:
        return sorted(self._files)

    def _resolve(self, hexid: str) -> str:
        hexid = hexid.lower()
        m = [i for i in self._files if i.startswith(hexid)]
        if len(m) == 1:
            return m[0]
        if len(m) > 1:
            raise PackError("pack id prefix %s is ambiguous" % hexid)
        if self.download:
            return self._fetch_remote(hexid)
        raise PackError("pack %s not found (searched %s)" % (hexid, ", ".join(self.dirs)))

    def _fetch_remote(self, hexid: str) -> str:  # pragma: no cover - needs the published pack set
        remote = self.remote
        req = urllib.request.Request(remote + "index.json", headers={"User-Agent": USER_AGENT})
        try:
            idx = json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as err:  # DUAL: the old v2/ref/ prefix
            if err.code != 404 or "/s1/ref/" not in remote:
                raise
            remote = remote.replace("/s1/ref/", "/v2/ref/")
            req = urllib.request.Request(remote + "index.json", headers={"User-Agent": USER_AGENT})
            idx = json.load(urllib.request.urlopen(req, timeout=30))
        ent = [e for e in idx["packs"] if e["id"].startswith(hexid)]
        if len(ent) != 1:
            raise PackError("pack %s is not in the remote index" % hexid)
        e = ent[0]
        req = urllib.request.Request(remote + e["id"] + ".pack", headers={"User-Agent": USER_AGENT})
        data = urllib.request.urlopen(req, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != e["sha256"] or pack_id(data).hex() != e["id"]:
            raise PackError("downloaded pack %s fails its checksum" % e["id"])
        d = user_cache_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, e["id"] + ".pack")
        with open(path, "wb") as fh:
            fh.write(data)
        self._files[e["id"]] = path
        return e["id"]

    def read(self, hexid: str) -> bytes:
        pid = self._resolve(hexid)
        data = open(self._files[pid], "rb").read()
        if pack_id(data).hex() != pid:
            raise PackError("pack file %s does not match its id" % self._files[pid])
        return data

    def get(self, hexid: str):
        """Pack object (``VocabPack`` or ``Whitelist``) for an id or id prefix (hex)."""
        pid = self._resolve(hexid)
        o = self._obj.get(pid)
        if o is None:
            data = self.read(pid)
            kind = parse(data)[0]
            if kind in (4, 5):  # junction segment / version (singlet.s1.junctions)
                from . import junctions as _J

                o = (
                    _J.JunctionTable(data, fetch=lambda h: self.read(self._resolve(h)))
                    if kind == _J.KIND_VERSION
                    else _J.Segment(data)
                )
            else:
                o = Whitelist(data) if kind == KIND_WHITELIST else VocabPack(data, self)
            self._obj[pid] = o
        return o

    def get_id(self, digest16: bytes):
        return self.get(digest16.hex())

    def entries(self) -> list:
        return list(self._index)

    # ------------------------------------------------------------------ selection
    def vocab_packs(self, member: str, build: Optional[str] = None) -> list:
        out = []
        for e in self._index:
            if (
                e.get("kind") == "vocab"
                and e.get("member") == member
                and (build is None or e.get("build") == build)
            ):
                out.append(self.get(e["id"]))
        return out

    def find_vocab(
        self, member: str, names: Sequence[str], build_hint: Optional[str] = None
    ) -> Optional[VocabPack]:
        """The vocabulary pack whose names equal ``names`` byte for byte (hint tried first), else None."""
        cands = self.vocab_packs(member)
        cands.sort(key=lambda p: 0 if p.build == build_hint else 1)
        for p in cands:
            if p.n() == len(names) and p.matches(names):
                return p
        return None

    def whitelists(self) -> list:
        return [self.get(e["id"]) for e in self._index if e.get("kind") == "whitelist"]

    def whitelist_by_name(self, name: str):
        """The whitelist pack for a protocol list name (``3M-february-2018``, ``737K-cratac-v1`` alias included), or None."""
        for e in self._index:
            if e.get("kind") == "whitelist" and (
                name == e.get("protocol") or name in (e.get("names") or [])
            ):
                return self.get(e["id"])
        return None

    def find_whitelist(self, barcodes: Sequence[str]):
        """(Whitelist, indices) for the smallest whitelist containing all ``barcodes``; None if there is none."""
        codes: dict = {}
        for w in sorted(self.whitelists(), key=lambda w: w.n):
            if w.k not in codes:
                codes[w.k] = barcodes_to_codes(barcodes, w.k)
            if codes[w.k] is None:
                continue
            idx = w.index_of_codes(codes[w.k])
            if idx is not None:
                return w, idx
        return None


_DEFAULT_STORE: Optional[PackStore] = None


def default_store() -> PackStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = PackStore()
    return _DEFAULT_STORE


def expand(pack, member: Optional[str] = None) -> list:
    """Byte-exact name list of a pack (``list[str]``). ``pack`` may be a ``VocabPack``, a ``Whitelist``, pack bytes or a hex id."""
    if isinstance(pack, (bytes, bytearray)):
        kind = parse(pack)[0]
        pack = Whitelist(pack) if kind == KIND_WHITELIST else VocabPack(pack, default_store())
    elif isinstance(pack, str):
        pack = default_store().get(pack)
    if isinstance(pack, Whitelist):
        return pack.all_barcodes()
    return pack.names()


# =============================================================================================== index.json
def write_index(entries: list, path: str) -> None:
    doc = {"format": "singlet-s1-packs", "version": 1, "packs": entries}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")


def index_entry(buf: bytes, kind_name: str, extra: Optional[dict] = None) -> dict:
    e = {
        "id": pack_id(buf).hex(),
        "kind": kind_name,
        "bytes": len(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
    }
    e.update(extra or {})
    return e
