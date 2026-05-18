# SPDX-License-Identifier: MIT
"""``snp_sites.fbin`` — population SNP panel (§3.2 of canonical spec).

Binary layout (little-endian, packed):

::

    [Header                  128 B  ]
    [chrom table : 256×16  4096 B  ]
    [sites       : n_sites × 16 B  ]

Each :class:`SnpSite` record is 16 bytes:

* ``chrom_id  : u8``   index into the chrom table
* ``ref       : char`` reference base (A/C/G/T/N)
* ``alt       : char`` alt base
* ``_pad      : u8``   reserved
* ``pos       : u32``  1-based position
* ``af_pop    : f32``  population minor-allele frequency
* ``rsid      : u32``  numeric dbSNP rs id (0 if absent)
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Union

MAGIC = b"SLSNP01\0"

_HEADER_FMT = "<8s 32s 32s I I 32x"  # magic, build_id, panel_id, n_sites, n_chroms, pad
_HEADER_RAW = struct.calcsize(_HEADER_FMT)
SNP_HEADER_SIZE = 128
assert _HEADER_RAW <= SNP_HEADER_SIZE, _HEADER_RAW

_MAX_CHROMS = 256
_CHROM_NAME_LEN = 16
_CHROM_TABLE_SIZE = _MAX_CHROMS * _CHROM_NAME_LEN

_SNP_FMT = "<B cc B I f I"  # 1+1+1+1+4+4+4 = 16
SNP_REC_SIZE = struct.calcsize(_SNP_FMT)
assert SNP_REC_SIZE == 16, SNP_REC_SIZE


@dataclass
class SnpSite:
    """Decoded representation of a single SNP."""

    chrom: str
    pos: int
    ref: str
    alt: str
    af_pop: float
    rsid: int


def write_snp_panel(
    path: Union[str, Path],
    *,
    build_id: str,
    panel_id: str,
    sites: Sequence[SnpSite],
) -> None:
    """Serialize a SNP panel to disk in the canonical format.

    Sites must already be sorted by ``(chrom, pos)`` — this writer does
    not re-sort, because callers typically iterate a sorted VCF.
    """
    path = Path(path)
    chrom_to_id: dict = {}
    for s in sites:
        if s.chrom not in chrom_to_id:
            chrom_to_id[s.chrom] = len(chrom_to_id)
        if len(chrom_to_id) > _MAX_CHROMS:
            raise ValueError(f"too many chroms ({len(chrom_to_id)} > {_MAX_CHROMS})")

    chrom_table = bytearray(_CHROM_TABLE_SIZE)
    for name, idx in chrom_to_id.items():
        encoded = name.encode("utf-8")[:_CHROM_NAME_LEN]
        chrom_table[idx * _CHROM_NAME_LEN : idx * _CHROM_NAME_LEN + len(encoded)] = (
            encoded
        )

    n_sites = len(sites)
    n_chroms = len(chrom_to_id)
    header_packed = struct.pack(
        _HEADER_FMT,
        MAGIC,
        build_id.encode("utf-8")[:32].ljust(32, b"\0"),
        panel_id.encode("utf-8")[:32].ljust(32, b"\0"),
        n_sites,
        n_chroms,
    )
    header = header_packed.ljust(SNP_HEADER_SIZE, b"\0")

    with open(path, "wb") as f:
        f.write(header)
        f.write(bytes(chrom_table))
        for s in sites:
            ref = (s.ref or "N").encode("ascii")[:1]
            alt = (s.alt or "N").encode("ascii")[:1]
            f.write(
                struct.pack(
                    _SNP_FMT,
                    chrom_to_id[s.chrom],
                    ref,
                    alt,
                    0,
                    s.pos,
                    s.af_pop,
                    s.rsid,
                )
            )


class SnpPanel:
    """Mmap-backed reader for ``snp_sites.fbin``."""

    __slots__ = (
        "_path",
        "_file",
        "_mm",
        "magic",
        "build_id",
        "panel_id",
        "n_sites",
        "n_chroms",
        "chroms",
        "_sites_offset",
    )

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._file = open(self._path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()
        self._parse_chroms()
        self._sites_offset = SNP_HEADER_SIZE + _CHROM_TABLE_SIZE

    def _parse_header(self) -> None:
        raw = bytes(self._mm[:_HEADER_RAW])
        self.magic, build_raw, panel_raw, self.n_sites, self.n_chroms = struct.unpack(
            _HEADER_FMT, raw
        )
        if self.magic != MAGIC:
            raise ValueError(f"bad magic {self.magic!r}, expected {MAGIC!r}")
        self.build_id = build_raw.rstrip(b"\0").decode("utf-8")
        self.panel_id = panel_raw.rstrip(b"\0").decode("utf-8")

    def _parse_chroms(self) -> None:
        self.chroms = []
        base = SNP_HEADER_SIZE
        for i in range(self.n_chroms):
            raw = bytes(
                self._mm[base + i * _CHROM_NAME_LEN : base + (i + 1) * _CHROM_NAME_LEN]
            )
            self.chroms.append(raw.rstrip(b"\0").decode("utf-8"))

    def __len__(self) -> int:
        return self.n_sites

    def site(self, idx: int) -> SnpSite:
        if not 0 <= idx < self.n_sites:
            raise IndexError(idx)
        base = self._sites_offset + idx * SNP_REC_SIZE
        chrom_u8, ref, alt, _pad, pos, af, rsid = struct.unpack(
            _SNP_FMT, bytes(self._mm[base : base + SNP_REC_SIZE])
        )
        return SnpSite(
            chrom=self.chroms[chrom_u8] if chrom_u8 < self.n_chroms else "",
            pos=pos,
            ref=ref.decode("ascii"),
            alt=alt.decode("ascii"),
            af_pop=af,
            rsid=rsid,
        )

    def iter_sites(self) -> Iterable[SnpSite]:
        for i in range(self.n_sites):
            yield self.site(i)

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._file.close()

    def __enter__(self) -> "SnpPanel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_snp_panel(path: Union[str, Path]) -> SnpPanel:
    """Open ``snp_sites.fbin`` at ``path`` for read."""
    return SnpPanel(path)
