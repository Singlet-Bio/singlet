# SPDX-License-Identifier: MIT
"""In-memory canonical CSC matrix used across the v2 package, and the canonical digest of docs/format-v2.md section 4.6."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import util

VT_U8, VT_U16, VT_U32, VT_F32 = 1, 2, 3, 4
_VT_MAX = {VT_U8: 0xFF, VT_U16: 0xFFFF, VT_U32: 0xFFFFFFFF}

_DIGEST_BLOCK_NNZ = 4_000_000


@dataclass
class Csc:
    """Canonical CSC: ``indptr`` (n+1, int64), ``indices`` (nnz, uint32, strictly ascending in each column), ``data``
    (nnz, uint32; for ``value_type`` 4 the IEEE-754 float32 bit pattern), names, key/values.

    ``rownames`` / ``colnames`` are ``list[str]`` once resolved. A decoder that does not resolve pack or cells references
    leaves them ``None`` and sets ``row_ref`` / ``col_ref``.
    """

    m: int
    n: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    rownames: Optional[list] = None
    colnames: Optional[list] = None
    kv: dict = field(default_factory=dict)
    value_type: int = VT_U32
    row_ref: Optional[dict] = None  # {"pack": "<hex8>"} when the names live in a pack
    col_ref: Optional[dict] = None  # {"cells": True} or {"pack": "<hex8>"}
    meta: dict = field(default_factory=dict)

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1]) if len(self.indptr) else 0

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_legacy(cls, r: dict) -> "Csc":
        """From the dict returned by ``singlet._pz.read_1pz`` (values are uint32, ``vt_code`` 1/2/3)."""
        vt = int(r["vt_code"]) if int(r["vt_code"]) in (1, 2, 3) else VT_U32
        csc = cls(
            m=int(r["m"]),
            n=int(r["n"]),
            indptr=np.asarray(r["indptr"]).astype(np.int64),
            indices=np.asarray(r["indices"]).astype(np.uint32),
            data=np.asarray(r["data"]).astype(np.uint32),
            rownames=list(r["rownames"]),
            colnames=list(r["colnames"]),
            kv=dict(r.get("user_kv") or {}),
            value_type=vt,
        )
        csc.meta["legacy_vt_code"] = int(r["vt_code"])
        return csc

    def to_scipy(self):
        from scipy.sparse import csc_matrix

        return csc_matrix(
            (self.data, self.indices.astype(np.int64), self.indptr), shape=(self.m, self.n)
        )

    # ------------------------------------------------------------------ checks
    def check_canonical(self) -> Optional[str]:
        """None if canonical (strictly ascending ids per column, ids < m, values fit); else a reason."""
        if len(self.indptr) != self.n + 1:
            return "indptr length"
        if self.n and (self.indptr[0] != 0 or (np.diff(self.indptr) < 0).any()):
            return "indptr not monotone from 0"
        nnz = self.nnz
        if len(self.indices) != nnz or len(self.data) != nnz:
            return "array length != nnz"
        if nnz:
            if int(self.indices.max()) >= self.m:
                return "row id out of range"
            d = np.diff(self.indices.astype(np.int64))
            ok = d > 0
            # entries that start a column are exempt
            starts = self.indptr[1:-1]
            starts = starts[(starts > 0) & (starts < nnz)]
            ok[starts - 1] = True
            if not ok.all():
                return "row ids not strictly ascending within a column"
            if self.value_type in _VT_MAX and int(self.data.max()) > _VT_MAX[self.value_type]:
                return "value exceeds value_type range"
        return None

    def digest(self) -> bytes:
        return canonical_digest(self)


# --------------------------------------------------------------------------------------------------- canonical digest
def _kv_blob(kv: dict) -> bytes:
    out = bytearray()
    for k in sorted(kv, key=lambda s: s.encode("utf-8")):
        out += k.encode("utf-8") + b"\0" + str(kv[k]).encode("utf-8") + b"\0"
    return bytes(out)


def canonical_digest(csc: Csc, rownames=None, colnames=None) -> bytes:
    """digest16 of the canonical matrix (section 4.6). Names default to ``csc.rownames`` / ``csc.colnames``."""
    rn = csc.rownames if rownames is None else rownames
    cn = csc.colnames if colnames is None else colnames
    if (rn is None and not (csc.row_ref or {}).get("none")) or (
        cn is None and not (csc.col_ref or {}).get("none")
    ):
        raise ValueError("canonical digest needs resolved row and column names")
    rn = (
        [] if rn is None else rn
    )  # nameless matrices (row_ref / col_ref {"none": True}) hash an empty name list
    cn = [] if cn is None else cn
    m, n, nnz = int(csc.m), int(csc.n), csc.nnz
    h = hashlib.sha256()
    h.update(b"PZ2CANON1")
    h.update(struct.pack("<IIQB", m, n, nnz, csc.value_type))
    indptr = csc.indptr.astype(np.int64, copy=False)
    ids = np.ascontiguousarray(csc.indices, dtype="<u4")
    vals = np.ascontiguousarray(csc.data, dtype="<u4")
    j0 = 0
    while j0 < n:
        j1 = int(np.searchsorted(indptr, indptr[j0] + _DIGEST_BLOCK_NNZ, side="right")) - 1
        j1 = min(max(j1, j0 + 1), n)
        base = int(indptr[j0])
        end = int(indptr[j1])
        nk = end - base
        ncol = j1 - j0
        cnt = np.diff(indptr[j0 : j1 + 1])
        out = np.empty(ncol + 2 * nk, dtype="<u4")
        col_local = np.arange(ncol, dtype=np.int64)
        # start of column j in the output: j + 2 * (indptr[j] - base)
        start = col_local + 2 * (indptr[j0:j1] - base)
        out[start] = cnt.astype("<u4")
        if nk:
            col_of = np.repeat(col_local, cnt)
            k_local = np.arange(nk, dtype=np.int64)
            ip_of = (indptr[j0:j1] - base)[col_of]
            id_pos = col_of + ip_of + k_local + 1
            val_pos = col_of + (indptr[j0 + 1 : j1 + 1] - base)[col_of] + k_local + 1
            out[id_pos] = ids[base:end]
            out[val_pos] = vals[base:end]
        h.update(out.tobytes())
        j0 = j1
    h.update(struct.pack("<I", len(rn)))
    h.update(util.join_nul(rn))
    h.update(struct.pack("<I", len(cn)))
    h.update(util.join_nul(cn))
    h.update(_kv_blob(csc.kv))
    return h.digest()[:16]
