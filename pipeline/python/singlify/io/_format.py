"""
Binary format constants and header structs for the singlify .1pz format.

This module encodes the ``TP1Z v1`` specification described in
``singlify/docs/1PZ_FORMAT_SPEC.md``. Changes here must stay in lockstep
with that document.

The two structs exposed here, :class:`PZHeader` and :class:`PZFooter`, are
pure-Python :func:`dataclasses.dataclass` representations. They parse their
respective byte ranges via :mod:`struct` and do not perform any
decompression — higher-level modules in :mod:`singlify.io` handle that.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag

#: Magic number identifying a TP1Z file: little-endian bytes ``b"TP1Z"``.
MAGIC: int = 0x5A315054

#: Size of the :class:`PZHeader` in bytes.
HEADER_SIZE: int = 96

#: Size of the :class:`PZFooter` in bytes.
FOOTER_SIZE: int = 16

#: Current format version understood by this reader.
SUPPORTED_VERSION: int = 1


class VTCode(IntEnum):
    """Value type hint recorded in :attr:`PZHeader.vt_code`.

    The code is assigned by ``pz_writer.h::write_1pz`` based on the
    observed maximum value in the input matrix (not on the caller's C++
    template parameter). It is an *informational* hint about the narrowest
    integer type that would losslessly hold every stored value, not a
    byte-width for unpacking — the on-disk values are varint-encoded inside
    the VOCSC column TLV stream, so the byte width is irrelevant to decoding.

    The authoritative mapping matches ``pz_writer.h`` (NOT the prose
    ``1PZ_FORMAT_SPEC.md`` which documents a different, never-shipped layout):

    ======  ==================  ==========================================
    Code    Hint                 Conditions writer emits this code
    ======  ==================  ==========================================
    1       uint8-compatible     max nonzero value <= 255
    2       uint16-compatible    max nonzero value <= 65535
    3       uint32-compatible    max nonzero value <= 2^32-1
    ======  ==================  ==========================================

    Code values 0, 4, 5, 6 are reserved for future extensions (e.g. float
    or signed integer storage) but are not currently emitted by any version
    of the pipeline.
    """

    UINT8 = 1
    UINT16 = 2
    UINT32 = 3

    @property
    def numpy_dtype(self) -> str:
        """Return the narrowest numpy dtype string that losslessly holds
        every value this hint certifies the matrix contains.

        Useful for downstream readers that want to re-cast the uniform
        uint32 output of the C++ reader back to the hinted narrow type
        before handing it to scipy.sparse.
        """
        return {
            VTCode.UINT8: "|u1",
            VTCode.UINT16: "<u2",
            VTCode.UINT32: "<u4",
        }[self]


class Flags(IntFlag):
    """Header flag byte at offset 7 of :class:`PZHeader`."""

    HAS_PERM = 1 << 0
    GAP16 = 1 << 1
    HAS_METADATA = 1 << 2
    HAS_COLSUMS = 1 << 4


class FeatureFlags(IntFlag):
    """Extended feature flags (32-bit) at offset 44 of :class:`PZHeader`."""

    ZSTD_CHECKSUMS = 1 << 0
    BITPLANE_BITMAP = 1 << 1


# Struct format string for PZHeader. 96 bytes, little-endian, no padding.
#
#   4s  magic (kept as bytes for round-trip; caller validates)
#   H   version
#   B   vt_code
#   B   flags
#   I   m (rows)
#   I   n (cols)
#   Q   nnz
#   B   ptr_width
#   B   codec_level
#   H   _pad0
#   I   num_chunks
#   I   perm_z_sz
#   I   ptr_z_sz
#   I   chunk_cols
#   I   feature_flags
#   Q   metadata_offset
#   I   metadata_z_sz
#   I   colsums_z_sz
#   Q   transpose_offset
#   I   transpose_z_sz
#   I   transpose_chunks
#   16s reserved
_HEADER_STRUCT = struct.Struct("<4sHBBIIQBBHIIIIIQIIQII16s")
assert _HEADER_STRUCT.size == HEADER_SIZE, (
    f"header struct is {_HEADER_STRUCT.size} bytes, expected {HEADER_SIZE}"
)

_FOOTER_STRUCT = struct.Struct("<IIII")
assert _FOOTER_STRUCT.size == FOOTER_SIZE, (
    f"footer struct is {_FOOTER_STRUCT.size} bytes, expected {FOOTER_SIZE}"
)


@dataclass(frozen=True)
class PZHeader:
    """Parsed 96-byte TP1Z header.

    Attributes are named to match the spec field names exactly. Use
    :meth:`from_bytes` to parse a header from raw bytes.
    """

    version: int
    vt_code: VTCode
    flags: Flags
    m: int
    n: int
    nnz: int
    ptr_width: int
    codec_level: int
    num_chunks: int
    perm_z_sz: int
    ptr_z_sz: int
    chunk_cols: int
    feature_flags: FeatureFlags
    metadata_offset: int
    metadata_z_sz: int
    colsums_z_sz: int
    transpose_offset: int
    transpose_z_sz: int
    transpose_chunks: int

    @classmethod
    def from_bytes(cls, buf: bytes) -> "PZHeader":
        """Parse a :class:`PZHeader` from the first :data:`HEADER_SIZE` bytes of ``buf``.

        Raises
        ------
        ValueError
            If the magic number is wrong, the version is unsupported, or
            the ``vt_code`` or ``ptr_width`` fields are out of range.
        """
        if len(buf) < HEADER_SIZE:
            raise ValueError(
                f"header buffer too short: got {len(buf)}, need {HEADER_SIZE}"
            )
        (
            magic_bytes,
            version,
            vt_code_raw,
            flags_raw,
            m,
            n,
            nnz,
            ptr_width,
            codec_level,
            _pad0,
            num_chunks,
            perm_z_sz,
            ptr_z_sz,
            chunk_cols,
            feature_flags_raw,
            metadata_offset,
            metadata_z_sz,
            colsums_z_sz,
            transpose_offset,
            transpose_z_sz,
            transpose_chunks,
            _reserved,
        ) = _HEADER_STRUCT.unpack_from(buf, 0)

        magic_int = int.from_bytes(magic_bytes, "little")
        if magic_int != MAGIC:
            raise ValueError(
                f"bad TP1Z magic: expected 0x{MAGIC:08X}, got 0x{magic_int:08X}"
            )
        if version != SUPPORTED_VERSION:
            raise ValueError(
                f"unsupported TP1Z version: {version} (reader supports {SUPPORTED_VERSION})"
            )
        if ptr_width not in (1, 2, 4):
            raise ValueError(f"ptr_width must be 1, 2, or 4; got {ptr_width}")
        try:
            vt_code = VTCode(vt_code_raw)
        except ValueError as exc:
            raise ValueError(f"unknown vt_code: {vt_code_raw}") from exc

        return cls(
            version=version,
            vt_code=vt_code,
            flags=Flags(flags_raw),
            m=m,
            n=n,
            nnz=nnz,
            ptr_width=ptr_width,
            codec_level=codec_level,
            num_chunks=num_chunks,
            perm_z_sz=perm_z_sz,
            ptr_z_sz=ptr_z_sz,
            chunk_cols=chunk_cols,
            feature_flags=FeatureFlags(feature_flags_raw),
            metadata_offset=metadata_offset,
            metadata_z_sz=metadata_z_sz,
            colsums_z_sz=colsums_z_sz,
            transpose_offset=transpose_offset,
            transpose_z_sz=transpose_z_sz,
            transpose_chunks=transpose_chunks,
        )

    def to_bytes(self) -> bytes:
        """Serialise this header back to its 96-byte on-disk form.

        This is primarily useful for round-trip tests; the singlify pipeline
        writes headers from C++, not from Python.
        """
        return _HEADER_STRUCT.pack(
            MAGIC.to_bytes(4, "little"),
            self.version,
            int(self.vt_code),
            int(self.flags),
            self.m,
            self.n,
            self.nnz,
            self.ptr_width,
            self.codec_level,
            0,  # _pad0
            self.num_chunks,
            self.perm_z_sz,
            self.ptr_z_sz,
            self.chunk_cols,
            int(self.feature_flags),
            self.metadata_offset,
            self.metadata_z_sz,
            self.colsums_z_sz,
            self.transpose_offset,
            self.transpose_z_sz,
            self.transpose_chunks,
            b"\x00" * 16,  # reserved
        )


@dataclass(frozen=True)
class PZFooter:
    """Parsed 16-byte TP1Z footer."""

    file_crc32: int
    num_chunks: int

    @classmethod
    def from_bytes(cls, buf: bytes) -> "PZFooter":
        """Parse a :class:`PZFooter` from the last :data:`FOOTER_SIZE` bytes of ``buf``.

        Raises
        ------
        ValueError
            If the trailing magic does not match :data:`MAGIC`.
        """
        if len(buf) < FOOTER_SIZE:
            raise ValueError(
                f"footer buffer too short: got {len(buf)}, need {FOOTER_SIZE}"
            )
        file_crc32, _reserved, num_chunks, magic = _FOOTER_STRUCT.unpack_from(
            buf, len(buf) - FOOTER_SIZE
        )
        if magic != MAGIC:
            raise ValueError(
                f"bad TP1Z footer magic: expected 0x{MAGIC:08X}, got 0x{magic:08X}"
            )
        return cls(file_crc32=file_crc32, num_chunks=num_chunks)

    def to_bytes(self) -> bytes:
        """Serialise this footer back to its 16-byte on-disk form."""
        return _FOOTER_STRUCT.pack(self.file_crc32, 0, self.num_chunks, MAGIC)
