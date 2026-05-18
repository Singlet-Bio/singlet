# SPDX-License-Identifier: MIT
"""Canonical .1pz v2 multi-block sparse codec implementation.

On-disk layout (little-endian)::

    [0  ..  8 ]  magic  = b"1PZ02\\0\\0\\0"
    [8  .. 12]  header_len : u32  (length of the JSON header in bytes)
    [12 .. 12+H]  JSON header  (utf-8, no trailing newline)
    [12+H .. EOF]  raw payload region: concatenated zstd frames

The JSON header has this schema::

    {
      "version": 2,
      "n_cells": <int>,
      "cell_barcodes": {
        "offset": <int>, "compressed_len": <int>, "encoding": "zstd",
        "format": "utf8-newline"     # one barcode per line
      },
      "blocks": [
        {
          "name": "exon_body",
          "row_dim": <int>,
          "nnz": <int>,
          "n_data_layers": 1,
          "data_names": ["counts"],
          "data_dtypes": ["<i4"],
          "indices_dtype": "<i4",
          "indptr_dtype": "<i8",
          "streams": {
            "indptr":  {"offset": int, "compressed_len": int, "encoding": "zstd"},
            "indices": {"offset": int, "compressed_len": int, "encoding": "zstd"},
            "data":    [{"offset": int, "compressed_len": int, "encoding": "zstd"}]
          }
        },
        ...
      ]
    }

A two-data-layer block (e.g. SNP AD/DP) has ``n_data_layers = 2`` and
two entries in ``streams.data`` — both share the same ``indptr`` and
``indices``.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import zstandard as zstd

try:
    from scipy.sparse import csc_matrix
except ImportError:  # pragma: no cover — scipy is a hard dep of singlet
    csc_matrix = None  # type: ignore

MAGIC = b"1PZ02\x00\x00\x00"
VERSION = 2

_HEADER_LEN_FMT = "<I"
_HEADER_LEN_SIZE = struct.calcsize(_HEADER_LEN_FMT)

_ZSTD_LEVEL = 3


class PzV2Error(RuntimeError):
    """Raised on .1pz v2 format errors."""


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


@dataclass
class BlockSpec:
    """Specification for one row block.

    Parameters
    ----------
    name
        Block name (``exon_body``, ``intron_body``, ``junctions``,
        ``snp``, ``mt``, ``species`` …).
    matrix
        Primary CSC matrix. Rows = block-local feature axis. Columns
        must match the global ``cell_barcodes`` ordering.
    data2
        Optional second data layer (e.g. ``dp`` companion to ``ad``).
        Must share ``indptr`` and ``indices`` with ``matrix``; only its
        ``data`` array is stored. Pass as a ``np.ndarray`` of length
        ``matrix.nnz`` or as a second CSC matrix from which the data
        array is taken (sparsity pattern must match exactly).
    data_names
        Logical names for the data layer(s). Defaults to
        ``["counts"]`` or ``["ad", "dp"]``.
    """

    name: str
    matrix: "csc_matrix"
    data2: Union["csc_matrix", np.ndarray, None] = None
    data_names: Optional[Sequence[str]] = None


def _to_csc(m) -> "csc_matrix":
    if csc_matrix is None:
        raise PzV2Error("scipy.sparse is required for pz_v2")
    if hasattr(m, "tocsc"):
        return m.tocsc()
    raise PzV2Error(f"unsupported matrix type: {type(m).__name__}")


def _zstd_compress(buf: bytes) -> bytes:
    return zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(buf)


def _zstd_decompress(buf: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(buf)


def _np_dtype_str(arr: np.ndarray) -> str:
    """Numpy dtype as little-endian str (e.g. ``'<i4'``)."""
    dt = np.dtype(arr.dtype)
    if dt.byteorder == ">":
        raise PzV2Error("big-endian arrays are not supported")
    return dt.str


def _validate_data2_pattern(primary: "csc_matrix", data2_in) -> np.ndarray:
    """Return the data-2 1-D array, validating sparsity pattern match."""
    if isinstance(data2_in, np.ndarray):
        if data2_in.shape != (primary.nnz,):
            raise PzV2Error(
                f"data2 length {data2_in.shape} != primary nnz {primary.nnz}"
            )
        return data2_in
    m2 = _to_csc(data2_in)
    if m2.shape != primary.shape:
        raise PzV2Error("data2 shape mismatch")
    if m2.nnz != primary.nnz:
        raise PzV2Error("data2 nnz mismatch — two-layer CSC requires shared pattern")
    if not (
        np.array_equal(m2.indptr, primary.indptr)
        and np.array_equal(m2.indices, primary.indices)
    ):
        raise PzV2Error(
            "data2 sparsity pattern (indptr/indices) must match primary"
        )
    return np.asarray(m2.data)


def write_pz_v2(
    path: Union[str, Path],
    *,
    cell_barcodes: Sequence[str],
    blocks: Sequence[BlockSpec],
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a multi-block .1pz v2 file.

    Parameters
    ----------
    path
        Output path.
    cell_barcodes
        Global cell-barcode axis. Every block's columns must match this
        length and ordering.
    blocks
        One or more :class:`BlockSpec`.
    extra_header
        Optional dict merged into the JSON header (e.g. ``{"row_axis":
        "exon_interval"}``). Reserved keys (``version``, ``n_cells``,
        ``cell_barcodes``, ``blocks``) are silently ignored.
    """
    path = Path(path)
    n_cells = len(cell_barcodes)
    if n_cells <= 0:
        raise PzV2Error("cell_barcodes must be non-empty")

    # Compress the barcode list once.
    bc_bytes = ("\n".join(cell_barcodes)).encode("utf-8")
    bc_compressed = _zstd_compress(bc_bytes)

    # Compress every block's streams.
    block_payloads: List[Dict[str, Any]] = []
    payload_chunks: List[bytes] = []
    current_offset = 0  # offset within the payload region

    def _add_stream(buf: bytes) -> Dict[str, int]:
        nonlocal current_offset
        info = {
            "offset": current_offset,
            "compressed_len": len(buf),
            "encoding": "zstd",
        }
        payload_chunks.append(buf)
        current_offset += len(buf)
        return info

    bc_stream = _add_stream(bc_compressed)
    bc_stream["format"] = "utf8-newline"
    bc_stream["uncompressed_len"] = len(bc_bytes)

    for spec in blocks:
        m = _to_csc(spec.matrix)
        if m.shape[1] != n_cells:
            raise PzV2Error(
                f"block {spec.name!r} has {m.shape[1]} cols but cell axis is {n_cells}"
            )
        indptr = np.ascontiguousarray(m.indptr)
        indices = np.ascontiguousarray(m.indices)
        data = np.ascontiguousarray(m.data)
        data_streams: List[Dict[str, Any]] = []
        data_dtypes: List[str] = [_np_dtype_str(data)]
        names = list(spec.data_names or [])
        if not names:
            names = ["ad", "dp"] if spec.data2 is not None else ["counts"]
        if len(names) != (2 if spec.data2 is not None else 1):
            raise PzV2Error(
                f"data_names length mismatch for block {spec.name!r}: got {names}"
            )

        s_indptr = _add_stream(_zstd_compress(indptr.tobytes()))
        s_indptr["uncompressed_len"] = indptr.nbytes
        s_indices = _add_stream(_zstd_compress(indices.tobytes()))
        s_indices["uncompressed_len"] = indices.nbytes

        d0 = _add_stream(_zstd_compress(data.tobytes()))
        d0["uncompressed_len"] = data.nbytes
        data_streams.append(d0)

        if spec.data2 is not None:
            data2 = _validate_data2_pattern(m, spec.data2)
            data2 = np.ascontiguousarray(data2)
            data_dtypes.append(_np_dtype_str(data2))
            d1 = _add_stream(_zstd_compress(data2.tobytes()))
            d1["uncompressed_len"] = data2.nbytes
            data_streams.append(d1)

        block_payloads.append(
            {
                "name": spec.name,
                "row_dim": int(m.shape[0]),
                "nnz": int(m.nnz),
                "n_data_layers": len(data_streams),
                "data_names": names,
                "data_dtypes": data_dtypes,
                "indices_dtype": _np_dtype_str(indices),
                "indptr_dtype": _np_dtype_str(indptr),
                "streams": {
                    "indptr": s_indptr,
                    "indices": s_indices,
                    "data": data_streams,
                },
            }
        )

    header: Dict[str, Any] = {
        "version": VERSION,
        "n_cells": n_cells,
        "cell_barcodes": bc_stream,
        "blocks": block_payloads,
    }
    if extra_header:
        for k, v in extra_header.items():
            if k not in ("version", "n_cells", "cell_barcodes", "blocks"):
                header[k] = v

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(_HEADER_LEN_FMT, len(header_bytes)))
        f.write(header_bytes)
        for chunk in payload_chunks:
            f.write(chunk)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


@dataclass
class _StreamInfo:
    offset: int
    compressed_len: int
    encoding: str = "zstd"
    uncompressed_len: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_StreamInfo":
        known = {"offset", "compressed_len", "encoding", "uncompressed_len"}
        kwargs = {k: d[k] for k in known if k in d}
        extras = {k: v for k, v in d.items() if k not in known}
        info = cls(**kwargs)
        info.extra = extras
        return info


class Block:
    """Lazy view of one row-block.

    Methods materialize CSC matrices on demand; nothing is decompressed
    until the first call.
    """

    def __init__(
        self,
        reader: "PzReader",
        name: str,
        row_dim: int,
        nnz: int,
        data_names: Sequence[str],
        data_dtypes: Sequence[str],
        indices_dtype: str,
        indptr_dtype: str,
        s_indptr: _StreamInfo,
        s_indices: _StreamInfo,
        s_data: Sequence[_StreamInfo],
    ) -> None:
        self._reader = reader
        self.name = name
        self.row_dim = row_dim
        self.nnz = nnz
        self.data_names = list(data_names)
        self._data_dtypes = list(data_dtypes)
        self._indices_dtype = indices_dtype
        self._indptr_dtype = indptr_dtype
        self._s_indptr = s_indptr
        self._s_indices = s_indices
        self._s_data = list(s_data)
        self._indptr_cache: Optional[np.ndarray] = None
        self._indices_cache: Optional[np.ndarray] = None

    @property
    def n_data_layers(self) -> int:
        return len(self._s_data)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row_dim, self._reader.n_cells)

    def _load(self, info: _StreamInfo, dtype: str) -> np.ndarray:
        raw = self._reader._read_stream(info)
        return np.frombuffer(raw, dtype=np.dtype(dtype))

    def indptr(self) -> np.ndarray:
        if self._indptr_cache is None:
            self._indptr_cache = self._load(self._s_indptr, self._indptr_dtype)
        return self._indptr_cache

    def indices(self) -> np.ndarray:
        if self._indices_cache is None:
            self._indices_cache = self._load(self._s_indices, self._indices_dtype)
        return self._indices_cache

    def data_array(self, layer: Union[int, str] = 0) -> np.ndarray:
        """Raw 1-D data array for the given layer (by index or name)."""
        idx = self._resolve_layer(layer)
        return self._load(self._s_data[idx], self._data_dtypes[idx])

    def _resolve_layer(self, layer: Union[int, str]) -> int:
        if isinstance(layer, int):
            if not 0 <= layer < self.n_data_layers:
                raise IndexError(layer)
            return layer
        try:
            return self.data_names.index(layer)
        except ValueError:
            raise KeyError(
                f"unknown data layer {layer!r}; have {self.data_names}"
            ) from None

    def data(self, layer: Union[int, str] = 0) -> "csc_matrix":
        """CSC matrix view of one data layer."""
        if csc_matrix is None:
            raise PzV2Error("scipy.sparse is required")
        return csc_matrix(
            (self.data_array(layer), self.indices(), self.indptr()),
            shape=self.shape,
        )


class PzReader:
    """Reader for .1pz v2 files."""

    __slots__ = (
        "_path",
        "_file",
        "_payload_offset",
        "_header",
        "cell_barcodes",
        "n_cells",
        "_blocks",
    )

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._file = open(self._path, "rb")
        try:
            self._parse_header()
            self._materialize_barcodes()
            self._build_blocks()
        except Exception:
            self._file.close()
            raise

    # ------------------------------------------------------------------ header

    def _parse_header(self) -> None:
        magic = self._file.read(8)
        if magic != MAGIC:
            raise PzV2Error(f"bad magic {magic!r}, expected {MAGIC!r}")
        (header_len,) = struct.unpack(
            _HEADER_LEN_FMT, self._file.read(_HEADER_LEN_SIZE)
        )
        raw = self._file.read(header_len)
        self._header = json.loads(raw.decode("utf-8"))
        if self._header.get("version") != VERSION:
            raise PzV2Error(
                f"unsupported version: {self._header.get('version')}"
            )
        self.n_cells = int(self._header["n_cells"])
        self._payload_offset = 8 + _HEADER_LEN_SIZE + header_len

    def _materialize_barcodes(self) -> None:
        info = _StreamInfo.from_dict(self._header["cell_barcodes"])
        raw = self._read_stream(info)
        self.cell_barcodes = raw.decode("utf-8").split("\n")
        if len(self.cell_barcodes) != self.n_cells:
            raise PzV2Error(
                f"cell barcode count {len(self.cell_barcodes)} != n_cells {self.n_cells}"
            )

    def _build_blocks(self) -> None:
        self._blocks: Dict[str, Block] = {}
        for spec in self._header["blocks"]:
            streams = spec["streams"]
            s_data = [_StreamInfo.from_dict(d) for d in streams["data"]]
            b = Block(
                reader=self,
                name=spec["name"],
                row_dim=int(spec["row_dim"]),
                nnz=int(spec["nnz"]),
                data_names=spec["data_names"],
                data_dtypes=spec["data_dtypes"],
                indices_dtype=spec["indices_dtype"],
                indptr_dtype=spec["indptr_dtype"],
                s_indptr=_StreamInfo.from_dict(streams["indptr"]),
                s_indices=_StreamInfo.from_dict(streams["indices"]),
                s_data=s_data,
            )
            self._blocks[b.name] = b

    # ------------------------------------------------------------------ streams

    def _read_stream(self, info) -> bytes:
        # Accept either _StreamInfo or raw dict (cell_barcodes uses extras).
        if isinstance(info, dict):
            offset = int(info["offset"])
            length = int(info["compressed_len"])
            encoding = info.get("encoding", "zstd")
        else:
            offset = int(info.offset)
            length = int(info.compressed_len)
            encoding = info.encoding
        self._file.seek(self._payload_offset + offset)
        raw = self._file.read(length)
        if encoding == "zstd":
            return _zstd_decompress(raw)
        if encoding == "raw":
            return raw
        raise PzV2Error(f"unknown encoding {encoding!r}")

    # ------------------------------------------------------------------ public

    @property
    def block_names(self) -> List[str]:
        return list(self._blocks.keys())

    def block(self, name: str) -> Block:
        if name not in self._blocks:
            raise KeyError(
                f"block {name!r} not in file; have {sorted(self._blocks)}"
            )
        return self._blocks[name]

    @property
    def extra_header(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self._header.items()
            if k not in ("version", "n_cells", "cell_barcodes", "blocks")
        }

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "PzReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_pz_v2(path: Union[str, Path]) -> PzReader:
    """Open a .1pz v2 file for read. Returns a lazy :class:`PzReader`."""
    return PzReader(path)
