# SPDX-License-Identifier: MIT
"""``features.fbin`` — feature vocabulary (genes / exons / introns / junctions).

Binary layout (little-endian, packed; see §3.1 of the canonical spec):

::

    [Header                                 256 B  ]
    [chrom table  : 256 × char[16]         4096 B  ]
    [genes        : n_genes  × 44 B                ]
    [exons        : n_exons  × 16 B                ]
    [introns      : n_introns× 24 B                ]
    [junctions    : n_jx     × 24 B                ]
    [string pool  : variable                       ]

The string pool stores every gene name / symbol as ``<u32 length><utf-8>``;
``name_offset`` / ``symbol_offset`` fields are byte offsets within the pool
(``0`` is reserved for "absent" — the first 4 bytes of the pool are an empty
string sentinel).

Chrom IDs are ``uint8`` indices into the chrom table (so an fbin supports
up to 256 distinct contigs, which is more than any common assembly).
"""

from __future__ import annotations

import io
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO, Iterable, List, Optional, Sequence, Union

# --------------------------------------------------------------------------
# Format constants
# --------------------------------------------------------------------------

MAGIC = b"SLFEAT01"
VERSION = 1

# Header: magic(8), version(4), build_id(32), gtf_sha256(32),
#         n_genes(4), n_exons(4), n_introns(4), n_junctions(4),
#         n_chroms(4), reserved(4),
#         offset_chroms(8), offset_genes(8), offset_exons(8),
#         offset_introns(8), offset_junctions(8), offset_strings(8)
# Padded to 256 B.
_HEADER_FMT = "<8s I 32s 32s 4I I I 6Q"
_HEADER_RAW = struct.calcsize(_HEADER_FMT)
FEATURES_HEADER_SIZE = 256
assert _HEADER_RAW <= FEATURES_HEADER_SIZE, "header struct exceeds 256 B"

_MAX_CHROMS = 256
_CHROM_NAME_LEN = 16
CHROM_TABLE_SIZE = _MAX_CHROMS * _CHROM_NAME_LEN  # 4096 B

# GeneRec  :  44 B  (4+4+4+8+8+8+8)
_GENE_FMT = "<II BBBB IIIIIIII"
GENE_REC_SIZE = struct.calcsize(_GENE_FMT)
assert GENE_REC_SIZE == 44, GENE_REC_SIZE

# ExonRec  :  16 B  (gene_id u32, chrom u8 + 3 pad, start u32, end u32)
_EXON_FMT = "<I 4s II"
EXON_REC_SIZE = struct.calcsize(_EXON_FMT)
assert EXON_REC_SIZE == 16, EXON_REC_SIZE

# IntronRec : 24 B  (gene_id u32, chrom u8 + 3 pad, start u32, end u32,
#                    flank_exon_lo u32, flank_exon_hi u32)
_INTRON_FMT = "<I 4s II II"
INTRON_REC_SIZE = struct.calcsize(_INTRON_FMT)
assert INTRON_REC_SIZE == 24, INTRON_REC_SIZE

# JunctionRec : 24 B  (gene_id u32, donor_feat u32, acceptor_feat u32,
#                      flags u8, motif u8, reserved u16,
#                      donor_pos u32, acceptor_pos u32)
_JUNCTION_FMT = "<III BBH II"
JUNCTION_REC_SIZE = struct.calcsize(_JUNCTION_FMT)
assert JUNCTION_REC_SIZE == 24, JUNCTION_REC_SIZE


# --------------------------------------------------------------------------
# Public dataclasses (Python-side mirror of the on-disk records)
# --------------------------------------------------------------------------


@dataclass
class GeneRecord:
    """Decoded representation of a single gene."""

    name: str  # ENSG…
    symbol: str  # gene symbol; "" if absent
    chrom: str  # chromosome name (resolved via chrom table)
    strand: int  # 0=+, 1=-
    biotype: int  # numeric biotype code (see _BIOTYPES)
    tx_start: int
    tx_end: int
    exon_lo: int
    exon_hi: int
    intron_lo: int
    intron_hi: int
    junction_lo: int
    junction_hi: int


# Biotype numeric encoding — extend as needed. 0 reserved for "unknown".
BIOTYPE_UNKNOWN = 0
BIOTYPE_PROTEIN_CODING = 1
BIOTYPE_LNCRNA = 2
BIOTYPE_PSEUDOGENE = 3
BIOTYPE_MIRNA = 4
BIOTYPE_RRNA = 5
BIOTYPE_TRNA = 6
BIOTYPE_SNORNA = 7
BIOTYPE_OTHER_NCRNA = 8

_BIOTYPE_MAP = {
    "protein_coding": BIOTYPE_PROTEIN_CODING,
    "lncRNA": BIOTYPE_LNCRNA,
    "lincRNA": BIOTYPE_LNCRNA,
    "pseudogene": BIOTYPE_PSEUDOGENE,
    "miRNA": BIOTYPE_MIRNA,
    "rRNA": BIOTYPE_RRNA,
    "tRNA": BIOTYPE_TRNA,
    "snoRNA": BIOTYPE_SNORNA,
    "snRNA": BIOTYPE_OTHER_NCRNA,
    "misc_RNA": BIOTYPE_OTHER_NCRNA,
}


def biotype_code(name: str) -> int:
    """Encode a GTF ``gene_biotype`` string to its numeric code."""
    return _BIOTYPE_MAP.get(name, BIOTYPE_UNKNOWN)


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


@dataclass
class _GeneIn:
    name: str
    symbol: str
    chrom: str
    strand: int
    biotype: int
    tx_start: int
    tx_end: int
    exons: List[tuple]  # (start, end)
    introns: List[tuple]  # (start, end, flank_exon_lo, flank_exon_hi)
    junctions: List[tuple]  # (donor_pos, acceptor_pos, donor_feat,
    #                          acceptor_feat, flags, motif)


class _StringPool:
    def __init__(self) -> None:
        self._buf = bytearray()
        # Reserve offset 0 = "absent" sentinel (length-0 string).
        self._buf.extend(struct.pack("<I", 0))
        self._cache: dict = {"": 0}

    def add(self, s: str) -> int:
        if s in self._cache:
            return self._cache[s]
        offset = len(self._buf)
        encoded = s.encode("utf-8")
        self._buf.extend(struct.pack("<I", len(encoded)))
        self._buf.extend(encoded)
        self._cache[s] = offset
        return offset

    def bytes(self) -> bytes:
        return bytes(self._buf)


def write_features(
    path: Union[str, Path],
    *,
    build_id: str,
    gtf_sha256: str,
    genes: Sequence[_GeneIn],
) -> None:
    """Serialize a feature bundle to disk in the canonical format.

    Parameters
    ----------
    path
        Output file path.
    build_id
        Build identifier (e.g. ``"GRCh38-2024-A"``); truncated/padded to 32 B.
    gtf_sha256
        Hex-encoded SHA-256 of the source GTF; truncated/padded to 32 B.
    genes
        Sequence of :class:`_GeneIn` records — one per gene.
    """
    path = Path(path)
    pool = _StringPool()

    # Assign chrom IDs in encounter order.
    chrom_to_id: dict = {}
    for g in genes:
        if g.chrom not in chrom_to_id:
            chrom_to_id[g.chrom] = len(chrom_to_id)
        if len(chrom_to_id) > _MAX_CHROMS:
            raise ValueError(f"too many chroms ({len(chrom_to_id)} > {_MAX_CHROMS})")
    chrom_table = bytearray(CHROM_TABLE_SIZE)
    for name, idx in chrom_to_id.items():
        encoded = name.encode("utf-8")[:_CHROM_NAME_LEN]
        chrom_table[idx * _CHROM_NAME_LEN : idx * _CHROM_NAME_LEN + len(encoded)] = (
            encoded
        )

    # Flatten exon/intron/junction tables and record per-gene ranges.
    exon_records: List[bytes] = []
    intron_records: List[bytes] = []
    junction_records: List[bytes] = []
    gene_records: List[bytes] = []

    for gid, g in enumerate(genes):
        chrom_u8 = chrom_to_id[g.chrom]
        chrom_pad = bytes([chrom_u8, 0, 0, 0])
        exon_lo = len(exon_records)
        for (s, e) in g.exons:
            exon_records.append(struct.pack(_EXON_FMT, gid, chrom_pad, s, e))
        exon_hi = len(exon_records)

        intron_lo = len(intron_records)
        for (s, e, fl_lo, fl_hi) in g.introns:
            intron_records.append(
                struct.pack(_INTRON_FMT, gid, chrom_pad, s, e, fl_lo, fl_hi)
            )
        intron_hi = len(intron_records)

        junction_lo = len(junction_records)
        for (dp, ap, df, af, flags, motif) in g.junctions:
            junction_records.append(
                struct.pack(_JUNCTION_FMT, gid, df, af, flags, motif, 0, dp, ap)
            )
        junction_hi = len(junction_records)

        gene_records.append(
            struct.pack(
                _GENE_FMT,
                pool.add(g.name),
                pool.add(g.symbol),
                chrom_u8,
                g.strand & 0xFF,
                g.biotype & 0xFF,
                0,
                g.tx_start,
                g.tx_end,
                exon_lo,
                exon_hi,
                intron_lo,
                intron_hi,
                junction_lo,
                junction_hi,
            )
        )

    n_genes = len(gene_records)
    n_exons = len(exon_records)
    n_introns = len(intron_records)
    n_junctions = len(junction_records)
    n_chroms = len(chrom_to_id)

    offset_chroms = FEATURES_HEADER_SIZE
    offset_genes = offset_chroms + CHROM_TABLE_SIZE
    offset_exons = offset_genes + n_genes * GENE_REC_SIZE
    offset_introns = offset_exons + n_exons * EXON_REC_SIZE
    offset_junctions = offset_introns + n_introns * INTRON_REC_SIZE
    offset_strings = offset_junctions + n_junctions * JUNCTION_REC_SIZE

    header_packed = struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        build_id.encode("utf-8")[:32].ljust(32, b"\0"),
        gtf_sha256.encode("utf-8")[:32].ljust(32, b"\0"),
        n_genes,
        n_exons,
        n_introns,
        n_junctions,
        n_chroms,
        0,
        offset_chroms,
        offset_genes,
        offset_exons,
        offset_introns,
        offset_junctions,
        offset_strings,
    )
    header = header_packed.ljust(FEATURES_HEADER_SIZE, b"\0")

    with open(path, "wb") as f:
        f.write(header)
        f.write(bytes(chrom_table))
        for rec in gene_records:
            f.write(rec)
        for rec in exon_records:
            f.write(rec)
        for rec in intron_records:
            f.write(rec)
        for rec in junction_records:
            f.write(rec)
        f.write(pool.bytes())


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


class FeaturesBundle:
    """Mmap-backed reader for ``features.fbin``.

    Attributes
    ----------
    n_genes, n_exons, n_introns, n_junctions, n_chroms : int
    build_id : str
    gtf_sha256 : str
    chroms : list[str]

    The reader holds an mmap for the lifetime of the instance — call
    :meth:`close` (or use as a context manager) to release it.
    """

    __slots__ = (
        "_path",
        "_file",
        "_mm",
        "magic",
        "version",
        "build_id",
        "gtf_sha256",
        "n_genes",
        "n_exons",
        "n_introns",
        "n_junctions",
        "n_chroms",
        "offset_chroms",
        "offset_genes",
        "offset_exons",
        "offset_introns",
        "offset_junctions",
        "offset_strings",
        "chroms",
    )

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._file = open(self._path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()
        self._parse_chroms()

    def _parse_header(self) -> None:
        raw = bytes(self._mm[:_HEADER_RAW])
        (
            self.magic,
            self.version,
            build_id_raw,
            gtf_sha_raw,
            self.n_genes,
            self.n_exons,
            self.n_introns,
            self.n_junctions,
            self.n_chroms,
            _reserved,
            self.offset_chroms,
            self.offset_genes,
            self.offset_exons,
            self.offset_introns,
            self.offset_junctions,
            self.offset_strings,
        ) = struct.unpack(_HEADER_FMT, raw)
        if self.magic != MAGIC:
            raise ValueError(f"bad magic {self.magic!r}, expected {MAGIC!r}")
        self.build_id = build_id_raw.rstrip(b"\0").decode("utf-8")
        self.gtf_sha256 = gtf_sha_raw.rstrip(b"\0").decode("utf-8")

    def _parse_chroms(self) -> None:
        self.chroms = []
        base = self.offset_chroms
        for i in range(self.n_chroms):
            raw = bytes(self._mm[base + i * _CHROM_NAME_LEN : base + (i + 1) * _CHROM_NAME_LEN])
            self.chroms.append(raw.rstrip(b"\0").decode("utf-8"))

    # ------------------------------------------------------------------ helpers

    def _read_string(self, offset: int) -> str:
        if offset == 0:
            return ""
        base = self.offset_strings + offset
        (length,) = struct.unpack_from("<I", self._mm, base)
        return bytes(self._mm[base + 4 : base + 4 + length]).decode("utf-8")

    # ------------------------------------------------------------------ records

    def gene(self, gene_id: int) -> GeneRecord:
        if not 0 <= gene_id < self.n_genes:
            raise IndexError(gene_id)
        base = self.offset_genes + gene_id * GENE_REC_SIZE
        raw = bytes(self._mm[base : base + GENE_REC_SIZE])
        (
            name_off,
            sym_off,
            chrom_u8,
            strand,
            biotype,
            _reserved,
            tx_start,
            tx_end,
            exon_lo,
            exon_hi,
            intron_lo,
            intron_hi,
            jx_lo,
            jx_hi,
        ) = struct.unpack(_GENE_FMT, raw)
        return GeneRecord(
            name=self._read_string(name_off),
            symbol=self._read_string(sym_off),
            chrom=self.chroms[chrom_u8] if chrom_u8 < self.n_chroms else "",
            strand=strand,
            biotype=biotype,
            tx_start=tx_start,
            tx_end=tx_end,
            exon_lo=exon_lo,
            exon_hi=exon_hi,
            intron_lo=intron_lo,
            intron_hi=intron_hi,
            junction_lo=jx_lo,
            junction_hi=jx_hi,
        )

    def iter_genes(self) -> Iterable[GeneRecord]:
        for i in range(self.n_genes):
            yield self.gene(i)

    def gene_names(self) -> List[str]:
        return [g.name for g in self.iter_genes()]

    def gene_symbols(self) -> List[str]:
        return [g.symbol for g in self.iter_genes()]

    def exon(self, exon_id: int) -> tuple:
        """Returns ``(gene_id, chrom_id, start, end)``."""
        if not 0 <= exon_id < self.n_exons:
            raise IndexError(exon_id)
        base = self.offset_exons + exon_id * EXON_REC_SIZE
        gene_id, chrom_pad, start, end = struct.unpack(
            _EXON_FMT, bytes(self._mm[base : base + EXON_REC_SIZE])
        )
        return gene_id, chrom_pad[0], start, end

    def junction(self, jx_id: int) -> tuple:
        """Returns ``(gene_id, donor_feat, acceptor_feat, flags, motif,
        donor_pos, acceptor_pos)``.
        """
        if not 0 <= jx_id < self.n_junctions:
            raise IndexError(jx_id)
        base = self.offset_junctions + jx_id * JUNCTION_REC_SIZE
        (
            gene_id,
            df,
            af,
            flags,
            motif,
            _reserved,
            dp,
            ap,
        ) = struct.unpack(_JUNCTION_FMT, bytes(self._mm[base : base + JUNCTION_REC_SIZE]))
        return gene_id, df, af, flags, motif, dp, ap

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._file.close()

    def __enter__(self) -> "FeaturesBundle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_features(path: Union[str, Path]) -> FeaturesBundle:
    """Open ``features.fbin`` at ``path`` for read."""
    return FeaturesBundle(path)
