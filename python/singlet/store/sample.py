# SPDX-License-Identifier: MIT
"""Open a sample object and use it: cells table, matrices with names resolved, regenerated legacy members.

``StoreSample(src)`` where ``src`` is bytes, a path or a ``fetch(offset, length)`` callable (see :class:`container.S2Reader`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import cells as cells_mod
from . import container as C
from .codec import HEADER_SIZE, PyCodec2, decode_any, parse_header
from .matrix import Csc, canonical_digest
from .packs import PackStore, VocabPack, Whitelist, default_store


class StoreError(RuntimeError):
    pass


class StoreSample:
    def __init__(self, source, packs: Optional[PackStore] = None, size: Optional[int] = None):
        self.reader = C.S2Reader(source, size=size)
        self.packs = packs or default_store()
        self.sample_id = self.reader.sample_id
        self._cells: Optional[cells_mod.CellsTable] = None
        self._mat_cache: dict[str, Csc] = {}

    # ------------------------------------------------------------------ listing
    @property
    def members(self) -> dict:
        return self.reader.members

    def legacy_members(self) -> list:
        return [n for n, m in self.reader.members.items() if m.flags & C.FLAG_LEGACY]

    def matrix_members(self) -> list:
        return [n for n, m in self.reader.members.items() if m.kind == C.KIND_PZ2]

    # ------------------------------------------------------------------ cells
    def cells(self) -> Optional[cells_mod.CellsTable]:
        if self._cells is None and "cells" in self.reader.members:
            mi = self.reader.members["cells"]
            rd = cells_mod.CellsReader(
                lambda off, ln: self.reader.read_range("cells", off, ln), mi.len, self.packs
            )
            self._cells = rd.load_all()
        return self._cells

    # ------------------------------------------------------------------ matrices
    def matrix_header(self, name: str) -> dict:
        mi = self.reader.members[name]
        if mi.kind != C.KIND_PZ2:
            raise StoreError("member %r is not a matrix" % name)
        return parse_header(self.reader.read_range(name, 0, min(mi.len, HEADER_SIZE)))

    def matrix(
        self, name: str, cols: Optional[tuple] = None, resolve: bool = True, order: str = "legacy"
    ) -> Csc:
        """Canonical CSC of a pz2 member (all columns, or ``cols=(a, b)`` for columns ``a <= j < b`` using ranged reads only).

        Junction rows stored as table ids (docs/format-store.md 12.4) come back in the legacy row order with the legacy names
        (``order="legacy"``, the logical matrix whose digest the member records), or as stored (``order="stored"``)."""
        mi = self.reader.members[name]
        if mi.kind != C.KIND_PZ2:
            raise StoreError("member %r is not a matrix (kind %d)" % (name, mi.kind))
        key = (name, order) if cols is None else None
        if key and key in self._mat_cache:
            return self._mat_cache[key]
        hdr = parse_header(self.reader.read_range(name, 0, min(mi.len, 65536)))
        if hdr["codec"] == 2:
            csc = PyCodec2().read_columns(
                lambda off, ln: self.reader.read_range(name, off, ln),
                mi.len,
                cols if cols is not None else (0, hdr["n"]),
                hdr,
            )
        else:
            csc = decode_any(self.reader.read_member(name), "canonical", cols)
        if resolve:
            self._resolve(csc, mi, cols, order)
        if key:
            self._mat_cache[key] = csc
        return csc

    def _resolve(
        self, csc: Csc, mi: C.MemberInfo, cols: Optional[tuple], order: str = "legacy"
    ) -> None:
        names = (mi.meta or {}).get("names", {})
        rr = csc.row_ref or {}
        if csc.rownames is None and "pack" in rr:
            ref = names.get("rows")
            pk = self.packs.get(rr["pack"])
            from . import junctions as J

            if isinstance(pk, J.JunctionTable):
                self._resolve_junctions(csc, mi, pk, rr, order)
            elif not isinstance(pk, VocabPack):
                raise StoreError("row names point at a whitelist")
            else:
                member = ref.get("member") if isinstance(ref, dict) else None
                allnames = [g for g, _n in pk.gene_table()] if member == "genes" else pk.names()
                if "index" in rr:
                    csc.rownames = [allnames[i] for i in rr["index"]]
                else:
                    csc.rownames = allnames
            if len(csc.rownames or []) != csc.m:
                raise StoreError(
                    "pack %s has %d names, matrix %r has %d rows"
                    % (rr["pack"], len(csc.rownames or []), mi.name, csc.m)
                )
        elif csc.rownames is None and not rr.get("none"):
            raise StoreError("matrix %r has unresolved row names" % mi.name)
        cr = csc.col_ref or {}
        if csc.colnames is None and not cr.get("none"):
            a, b = csc.meta.get("cols", (0, csc.n))
            if cr.get("cells"):
                cl = self.cells()
                if cl is None:
                    raise StoreError(
                        "matrix %r refers to the cells table, which is missing" % mi.name
                    )
                csc.colnames = cl.barcodes[a:b]
            elif "pack" in cr:
                pk = self.packs.get(cr["pack"])
                if not isinstance(pk, Whitelist):
                    raise StoreError("column names point at a non-whitelist pack")
                csc.colnames = pk.barcodes(cr["index"] if "index" in cr else np.arange(a, b))
            else:
                raise StoreError("matrix %r has unresolved column names" % mi.name)

    def _resolve_junctions(self, csc: Csc, mi: C.MemberInfo, table, rr: dict, order: str) -> None:
        from . import junctions as J

        if "index" not in rr:
            raise StoreError("junction rows of %r carry no ids" % mi.name)
        local = (
            self.reader.legacy_blob(mi.name + J.LOCAL_SUFFIX)
            if mi.name + J.LOCAL_SUFFIX in self.reader.members
            else None
        )
        perm = None
        if order == "legacy" and mi.name + J.LEGACY_ORDER_SUFFIX in self.reader.members:
            perm = self.reader.legacy_blob(mi.name + J.LEGACY_ORDER_SUFFIX)
        ip, ix, dt, rn = J.to_legacy(
            csc.indptr, csc.indices, csc.data, rr["index"], table, local, perm
        )
        csc.indptr, csc.indices, csc.data = (
            ip,
            ix.astype(csc.indices.dtype),
            dt.astype(csc.data.dtype),
        )
        csc.rownames = rn

    def verify_matrix_digest(self, name: str) -> bool:
        csc = self.matrix(name)
        return canonical_digest(csc) == self.reader.members[name].digest == csc.meta["digest"]

    # ------------------------------------------------------------------ legacy bytes
    def stored_bytes(self, name: str) -> bytes:
        return self.reader.legacy_blob(name)


V2Sample = StoreSample  # DUAL: old names, removed in the cleanup
V2Error = StoreError
