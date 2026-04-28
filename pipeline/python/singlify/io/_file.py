"""
High-level :class:`PZFile` wrapper — opens a ``.1pz`` file, parses the
header and footer, and exposes a minimal handle for downstream readers.

The public convenience entry point :func:`open_pz` is designed to work both
as a plain function call and as a context manager::

    from singlify.io import open_pz
    with open_pz("gene_counts.1pz") as pz:
        print(pz.header.m, "×", pz.header.n)
        print(f"nnz={pz.header.nnz}")

No decompression happens until a dedicated reader (e.g. ``_vocsc`` or
``_metadata``) is invoked against the opened handle.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Optional, Union

from ._format import FOOTER_SIZE, HEADER_SIZE, PZFooter, PZHeader


class PZError(Exception):
    """Raised when a ``.1pz`` file fails structural validation."""


@dataclass
class PZFile:
    """An opened ``.1pz`` file with parsed header and footer.

    Instances should be created via :func:`open_pz` rather than directly.
    The underlying file handle is owned by this object; use :meth:`close`
    or the context-manager protocol to release it.
    """

    path: str
    header: PZHeader
    footer: PZFooter
    file_size: int
    _fh: Optional[BinaryIO] = None

    @property
    def handle(self) -> BinaryIO:
        """Return the underlying file handle (raises if already closed)."""
        if self._fh is None:
            raise PZError(f"PZFile is closed: {self.path}")
        return self._fh

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "PZFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PZFile({self.path!r}, {self.header.m}×{self.header.n}, "
            f"nnz={self.header.nnz:,}, vt={self.header.vt_code.name}, "
            f"chunks={self.header.num_chunks})"
        )


def _crc32_stream(fh: BinaryIO, length: int, chunk: int = 1 << 20) -> int:
    """Compute a CRC32 over ``length`` bytes read from ``fh`` at its current position.

    The CRC uses :func:`zlib.crc32`, which implements ISO 3309 / IEEE 802.3
    — the same standard the TP1Z footer records.
    """
    remaining = length
    crc = 0
    while remaining > 0:
        block = fh.read(min(chunk, remaining))
        if not block:
            raise PZError(
                f"unexpected EOF while computing CRC: {remaining} bytes short"
            )
        crc = zlib.crc32(block, crc)
        remaining -= len(block)
    return crc & 0xFFFFFFFF


def open_pz(
    path: Union[str, os.PathLike],
    *,
    verify_crc: bool = False,
) -> PZFile:
    """Open a ``.1pz`` file, parse its header and footer, and return a :class:`PZFile`.

    Parameters
    ----------
    path
        Filesystem path to a ``.1pz`` file.
    verify_crc
        When ``True``, compute a CRC32 over the entire file body (everything
        before the 16-byte footer) and compare it to the CRC recorded in the
        footer. This costs a full read of the file, so it is off by default
        — but it is the strongest integrity check available without
        actually decoding the matrix.

    Returns
    -------
    PZFile
        An opened handle with :attr:`~PZFile.header`, :attr:`~PZFile.footer`
        and a live :attr:`~PZFile.handle` positioned after the header.

    Raises
    ------
    PZError
        If the file is too short, the magic is wrong, the version is
        unsupported, the chunk count disagrees between header and footer,
        or (when ``verify_crc`` is true) the CRC mismatches.
    """
    p = os.fspath(path)
    file_size = os.path.getsize(p)
    if file_size < HEADER_SIZE + FOOTER_SIZE:
        raise PZError(
            f"{p}: file too short ({file_size} bytes) to contain header+footer"
        )

    fh = open(p, "rb")
    try:
        header_bytes = fh.read(HEADER_SIZE)
        try:
            header = PZHeader.from_bytes(header_bytes)
        except ValueError as exc:
            raise PZError(f"{p}: {exc}") from exc

        fh.seek(file_size - FOOTER_SIZE)
        footer_bytes = fh.read(FOOTER_SIZE)
        try:
            footer = PZFooter.from_bytes(footer_bytes)
        except ValueError as exc:
            raise PZError(f"{p}: {exc}") from exc

        if footer.num_chunks != header.num_chunks:
            raise PZError(
                f"{p}: chunk count mismatch — header says {header.num_chunks}, "
                f"footer says {footer.num_chunks}"
            )

        if verify_crc:
            fh.seek(0)
            actual = _crc32_stream(fh, file_size - FOOTER_SIZE)
            if actual != footer.file_crc32:
                raise PZError(
                    f"{p}: CRC32 mismatch — computed 0x{actual:08X}, "
                    f"footer recorded 0x{footer.file_crc32:08X}"
                )

        fh.seek(HEADER_SIZE)
    except Exception:
        fh.close()
        raise

    return PZFile(path=p, header=header, footer=footer, file_size=file_size, _fh=fh)
