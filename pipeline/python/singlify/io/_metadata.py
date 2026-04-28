"""
Embedded metadata block decoder for TP1Z files.

The metadata block (when present, indicated by
:attr:`Flags.HAS_METADATA`) is a zstd-compressed stream of
**tag-length-value** records. The on-disk layout after decompression is::

    [tag:1][length:4][data:length]
    [tag:1][length:4][data:length]
    ...
    [META_TAG_END:1]

Tags defined by ``singlet::pz`` in ``pz_writer.h``:

======  ================================  ==========================================
Tag     Meaning                            Payload format
======  ================================  ==========================================
0       ``META_TAG_END``                   no payload (terminator)
1       ``META_TAG_ROWNAMES``              ``\0``-delimited UTF-8 strings
2       ``META_TAG_COLNAMES``              ``\0``-delimited UTF-8 strings
3       ``META_TAG_USER_KV``               ``\0``-delimited ``key\0value\0`` pairs
======  ================================  ==========================================

For singlify pipeline outputs, the ``user_kv`` block carries the GEO context
(``gsm_id``, ``gse_id``, ``organism``, ``protocol``, ``singlify_version``,
``pipeline_date``, etc.) flattened from the original ``meta.json``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import zstandard as zstd

from ._file import PZError, PZFile
from ._format import Flags

META_TAG_END: int = 0
META_TAG_ROWNAMES: int = 1
META_TAG_COLNAMES: int = 2
META_TAG_USER_KV: int = 3


@dataclass(frozen=True)
class PZMetadata:
    """Parsed metadata block from a TP1Z file.

    Attributes
    ----------
    rownames
        Feature identifiers (gene IDs, SNP positions, etc.), in row order.
        Empty list if the file did not carry a rownames TLV entry.
    colnames
        Cell barcode strings, in original (pre-permutation) column order.
        Empty list if absent.
    user_kv
        Key-value pairs from the ``META_TAG_USER_KV`` block. For singlify
        pipeline outputs this contains the GEO context (``gsm_id``,
        ``gse_id``, ``organism``, ``protocol``, ``singlify_version``,
        ``pipeline_date``, ...).
    """

    rownames: List[str] = field(default_factory=list)
    colnames: List[str] = field(default_factory=list)
    user_kv: Dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Shortcut for ``self.user_kv.get(key, default)``."""
        return self.user_kv.get(key, default)


def _parse_null_delimited_strings(payload: bytes) -> List[str]:
    """Parse a run of ``\\0``-terminated UTF-8 strings.

    The writer always terminates every string (including the last), so the
    payload ends with a ``\\0``. If any trailing bytes remain after the
    final ``\\0`` the payload is malformed.
    """
    if not payload:
        return []
    if payload[-1:] != b"\x00":
        raise PZError(
            f"metadata string block does not end with NUL ({payload[-8:]!r})"
        )
    # Split on NUL; the final split is an empty string because of the trailing NUL.
    parts = payload.split(b"\x00")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    try:
        return [p.decode("utf-8") for p in parts]
    except UnicodeDecodeError as exc:
        raise PZError(f"metadata string block is not valid UTF-8: {exc}") from exc


def _parse_kv_block(payload: bytes) -> Dict[str, str]:
    """Parse a run of ``key\\0value\\0`` pairs."""
    parts = _parse_null_delimited_strings(payload)
    if len(parts) % 2 != 0:
        raise PZError(
            f"user_kv block has {len(parts)} strings, expected even count (key\\0value\\0 pairs)"
        )
    return {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}


def _decode_tlv(stream: bytes) -> PZMetadata:
    """Decode a fully-decompressed metadata TLV stream."""
    rownames: List[str] = []
    colnames: List[str] = []
    user_kv: Dict[str, str] = {}

    offset = 0
    end = len(stream)
    while offset < end:
        tag = stream[offset]
        offset += 1
        if tag == META_TAG_END:
            break
        if offset + 4 > end:
            raise PZError(
                f"metadata TLV truncated at tag={tag}, offset={offset-1}, "
                f"stream size={end}"
            )
        (length,) = struct.unpack_from("<I", stream, offset)
        offset += 4
        if offset + length > end:
            raise PZError(
                f"metadata TLV tag={tag} claims length={length} but only "
                f"{end - offset} bytes remain"
            )
        payload = stream[offset : offset + length]
        offset += length

        if tag == META_TAG_ROWNAMES:
            rownames = _parse_null_delimited_strings(payload)
        elif tag == META_TAG_COLNAMES:
            colnames = _parse_null_delimited_strings(payload)
        elif tag == META_TAG_USER_KV:
            user_kv = _parse_kv_block(payload)
        # Unknown tags are silently ignored (forward compatibility), as per
        # the TP1Z v1 spec: "A reader MUST skip unknown flags and unknown
        # metadata fields."
    return PZMetadata(rownames=rownames, colnames=colnames, user_kv=user_kv)


def read_metadata(pz: PZFile) -> Optional[PZMetadata]:
    """Decode and return the embedded metadata block of ``pz``.

    Returns
    -------
    PZMetadata or None
        ``None`` when the file does not carry a metadata block (flag
        ``HAS_METADATA`` is clear or ``metadata_z_sz == 0``).

    Raises
    ------
    PZError
        If the metadata block is declared present but cannot be found,
        decompressed, or parsed as a TLV stream.
    """
    h = pz.header
    if not (h.flags & Flags.HAS_METADATA) or h.metadata_z_sz == 0:
        return None

    if h.metadata_offset <= 0 or h.metadata_offset + h.metadata_z_sz > pz.file_size:
        raise PZError(
            f"{pz.path}: metadata block claims offset={h.metadata_offset}, "
            f"size={h.metadata_z_sz}, but file is only {pz.file_size} bytes"
        )

    fh = pz.handle
    prev_pos = fh.tell()
    try:
        fh.seek(h.metadata_offset)
        compressed = fh.read(h.metadata_z_sz)
    finally:
        fh.seek(prev_pos)

    if len(compressed) != h.metadata_z_sz:
        raise PZError(
            f"{pz.path}: short read on metadata block "
            f"({len(compressed)} of {h.metadata_z_sz} bytes)"
        )

    try:
        raw = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise PZError(f"{pz.path}: metadata zstd decompression failed: {exc}") from exc

    return _decode_tlv(raw)
