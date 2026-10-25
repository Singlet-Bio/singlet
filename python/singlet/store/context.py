# SPDX-License-Identifier: MIT
"""The context object handed to recipes (``plan`` / ``regenerate`` / ``regenerate_matrix`` / ``cells_columns``).

The protocol is W3's :class:`singlet.store.recipes_ctx.Context` (``matrix``, ``member_bytes``, ``cells``, ``vocab_names``, ``genes``,
``whitelist``, ``features_order``, ``vector``, ``meta``, ``has``, ``sample_id``, ``study_id``, ``reference_build``,
``protocol_name``). This module provides the two real implementations:

* :class:`LegacyContext` - migration side: member bytes come from the legacy bundle, matrices from the legacy ``.1pz`` files,
  ``cells()`` is the cells table being built (required columns + candidate columns returned by ``cells_columns``).
* :class:`StoreContext` - read / verify side: everything comes from a finished ``.s1`` object alone (matrices decoded from pz2, stored
  members from their blobs, derived members regenerated recursively through the recipes).

Extensions on top of W3's protocol (see bench/compression/notes/m_w2.md section 0):

* ``ctx.provide_matrix(aux_name, matrix)``, ``ctx.provide_blob(aux_name, data)``, ``ctx.provide_vector(name, ndarray)``: only inside
  ``plan()``; register an auxiliary member that the container stores next to the sample (dropped if the plan does not verify).
  ``ctx.matrix(aux_name)`` / ``ctx.blob(aux_name)`` / ``ctx.vector(name)`` read it back in both phases.
* ``ctx.legacy_bytes`` is an alias of ``member_bytes``; ``ctx.gsm`` / ``ctx.gse`` alias ``sample_id`` / ``study_id``.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from . import container as C
from .legacy import decode_1pz_bytes
from .matrix import Csc
from .packs import PackStore, VocabPack, default_store
from .recipes_ctx import Context as RecipeContext
from .recipes_ctx import Matrix as RMatrix


class ContextError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- conversions
def csc_to_rmatrix(c: Csc) -> RMatrix:
    return RMatrix(
        indptr=c.indptr,
        indices=c.indices,
        data=c.data,
        shape=(c.m, c.n),
        rownames=c.rownames,
        colnames=c.colnames,
        user_kv=dict(c.kv),
    )


def rmatrix_to_csc(r, value_type: int = 3) -> Csc:
    if isinstance(r, Csc):
        return r
    m, n = int(r.shape[0]), int(r.shape[1])
    c = Csc(
        m=m,
        n=n,
        indptr=np.asarray(r.indptr).astype(np.int64),
        indices=np.asarray(r.indices).astype(np.uint32),
        data=np.asarray(r.data).astype(np.uint32),
        rownames=list(r.rownames) if r.rownames is not None else None,
        colnames=list(r.colnames) if r.colnames is not None else None,
        kv=dict(r.user_kv or {}),
        value_type=value_type,
    )
    if c.rownames is None:
        c.row_ref = {"none": True}
    if c.colnames is None:
        c.col_ref = {"none": True}
    return c


def cells_to_dict(t) -> dict:
    """CellsTable -> {name: numpy array} (strings and barcodes as object arrays)."""
    out = {}
    for name in t.names:
        v = t._cols[name].values
        if isinstance(v, list):
            a = np.empty(len(v), dtype=object)
            a[:] = v
            out[name] = a
        else:
            out[name] = v
    return out


# ---------------------------------------------------------------------------------------------- shared base
class _Base(RecipeContext):
    packs: PackStore

    def __init__(self):
        self._vectors: dict[str, np.ndarray] = {}
        self._wl_cache: dict[str, list] = {}

    # ---- aliases
    @property
    def gsm(self):
        return self.sample_id

    @property
    def gse(self):
        return self.study_id

    def legacy_bytes(self, name: str) -> Optional[bytes]:
        return self.member_bytes(name)

    # ---- pack views
    def _vocab_pack(self, kind: str) -> Optional[VocabPack]:
        raise NotImplementedError

    def pack_for(self, kind: str) -> Optional[VocabPack]:
        return self._vocab_pack(kind)

    def vocab_names(self, kind: str) -> Optional[list]:
        pk = self._vocab_pack(kind)
        return pk.names() if pk is not None else None

    def genes(self) -> Optional[list]:
        pk = self._vocab_pack("exon")
        return pk.gene_table() if pk is not None else None

    def whitelist(self, name: str):
        """Sorted barcodes of a whitelist pack as a numpy ``S<k>`` array (cached), or None."""
        if name not in self._wl_cache:
            w = self.packs.whitelist_by_name(name)
            self._wl_cache[name] = w.as_sarray() if w is not None else None
        return self._wl_cache[name]

    def features_order(self):
        """Per-build STARsolo features.tsv permutation (``recipes_data/features_order_<build>.u32.zst``), or None."""
        if not self.reference_build:
            return None
        from .recipes_ctx import load_features_order

        return load_features_order(self.reference_build)

    def vector(self, name: str):
        return self._vectors.get(name)

    def provide_matrix(self, aux_name, matrix):
        raise ContextError("provide_matrix is only valid inside plan() during migration")

    def provide_blob(self, aux_name, data):
        raise ContextError("provide_blob is only valid inside plan() during migration")

    def provide_vector(self, name, arr):
        raise ContextError("provide_vector is only valid inside plan() during migration")


# ============================================================================================ migration side
class LegacyContext(_Base):
    def __init__(
        self,
        bundle,
        gsm: Optional[str],
        packs: Optional[PackStore] = None,
        gse: Optional[str] = None,
        reference_build: Optional[str] = None,
        protocol_name: str = "",
        study_bytes: Optional[dict] = None,
    ):
        super().__init__()
        self.bundle = bundle
        self.sample_id = gsm or ""
        self.study_id = gse or ""
        self.reference_build = reference_build or ""
        self.protocol_name = protocol_name
        self.packs = packs or default_store()
        self._study = study_bytes if study_bytes is not None else {}
        self._paths = dict(bundle.sample_paths(gsm)) if gsm else {}  # relative name -> full path
        self._bytes: dict[str, bytes] = {}
        self._csc: dict[str, Optional[Csc]] = {}
        self._rm: dict[str, Optional[RMatrix]] = {}
        self.cells_table = None  # CellsTable being built (candidate columns included)
        self._cells_dict = None
        self._planning = False
        self.aux_matrices: dict[str, Csc] = {}
        self.aux_blobs: dict[str, bytes] = {}
        self.aux_vectors: dict[str, np.ndarray] = {}
        self._vp: dict[str, Optional[VocabPack]] = {}

    # ---- members
    def has(self, member: str) -> bool:
        return member in self._paths or member in self.aux_matrices or member in self.aux_blobs

    @property
    def members(self) -> list:
        return list(self._paths)

    def member_bytes(self, member: str) -> Optional[bytes]:
        if member in self.aux_blobs:
            return self.aux_blobs[member]
        if member not in self._paths:
            return None
        b = self._bytes.get(member)
        if b is None:
            b = self._bytes[member] = self.bundle.read(self._paths[member])
        return b

    def drop_cache(self, member: str) -> None:
        self._bytes.pop(member, None)

    def csc(self, member: str) -> Optional[Csc]:
        """Decoded legacy matrix as :class:`Csc` (None if absent or undecodable)."""
        if member in self.aux_matrices:
            return self.aux_matrices[member]
        if member in self._csc:
            return self._csc[member]
        b = self.member_bytes(member) if member.endswith(".1pz") else None
        try:
            c = decode_1pz_bytes(b) if b is not None else None
        except Exception:
            c = None
        self._csc[member] = c
        return c

    def matrix(self, member: str) -> Optional[RMatrix]:
        if member not in self._rm:
            c = self.csc(member)
            self._rm[member] = csc_to_rmatrix(c) if c is not None else None
        return self._rm[member]

    def study_file(self, name: str) -> Optional[bytes]:
        if name not in self._study:
            self._study[name] = self.bundle.read(name) if self.bundle.exists(name) else None
        return self._study[name]

    # ---- cells
    def set_cells(self, table) -> None:
        self.cells_table = table
        self._cells_dict = None

    def cells(self) -> Optional[dict]:
        if self.cells_table is None:
            return None
        if self._cells_dict is None:
            self._cells_dict = cells_to_dict(self.cells_table)
        return self._cells_dict

    # ---- aux
    def _check_name(self, name: str) -> None:
        if not self._planning:
            raise ContextError("auxiliary members can only be registered inside plan()")
        if (
            name in self._paths
            or name in self.aux_matrices
            or name in self.aux_blobs
            or name in self.aux_vectors
        ):
            raise ContextError("auxiliary member name %r is taken" % name)

    def provide_matrix(self, aux_name, matrix) -> None:
        self._check_name(aux_name)
        c = rmatrix_to_csc(matrix)
        if c.nnz and c.value_type == 3:  # smallest value type that holds the data
            mx = int(c.data.max())
            c.value_type = 1 if mx <= 0xFF else (2 if mx <= 0xFFFF else 3)
        self.aux_matrices[aux_name] = c
        self._rm.pop(aux_name, None)

    def provide_blob(self, aux_name, data) -> None:
        self._check_name(aux_name)
        self.aux_blobs[aux_name] = bytes(data)

    def provide_vector(self, name, arr) -> None:
        self._check_name("vector:" + name)
        a = np.ascontiguousarray(arr)
        self.aux_vectors[name] = a
        self._vectors[name] = a

    def blob(self, aux_name: str) -> bytes:
        return self.aux_blobs[aux_name]

    # ---- packs
    def _vocab_pack(self, kind: str) -> Optional[VocabPack]:
        if kind not in self._vp:
            m = self.csc("%s_counts.1pz" % kind) if self.sample_id else None
            pk = (
                self.packs.find_vocab(kind, m.rownames, self.reference_build)
                if m is not None and m.rownames
                else None
            )
            if (
                pk is None and not self.sample_id and self.reference_build
            ):  # study level: the pack of the build
                c = self.packs.vocab_packs(kind, self.reference_build)
                pk = c[0] if c else None
            self._vp[kind] = pk
        return self._vp[kind]


# ============================================================================================ read / verify side
class StoreContext(_Base):
    """Context over a finished sample object. ``recipes`` is the adapter (``singlet.store.rcp_adapter.Recipes``) used to regenerate
    derived members; ``study_files`` maps study-level names to bytes."""

    def __init__(
        self,
        sample,
        recipes,
        gse: Optional[str] = None,
        reference_build: Optional[str] = None,
        study_files: Optional[dict] = None,
        protocol_name: str = "",
    ):
        super().__init__()
        self.sample = sample
        self.reader = sample.reader
        self.sample_id = sample.sample_id
        self.study_id = gse or ""
        facts = {}
        for mi in self.reader.members.values():  # sample facts recorded by the migration
            c = (mi.meta or {}).get("ctx")
            if isinstance(c, dict):
                facts = c
                break
        self.reference_build = reference_build or facts.get("reference_build", "")
        self.protocol_name = protocol_name or facts.get("protocol_name", "")
        self.packs = sample.packs
        self.recipes = recipes
        self._study = study_files or {}
        self._out: dict[str, object] = {}  # member -> regenerated bytes or RMatrix
        self._busy: set = set()
        self._cells_dict = None
        self._rm: dict[str, Optional[RMatrix]] = {}
        # aux vectors are stored as blobs named "vector:<name>" (npy-like: dtype string line + raw bytes)
        for n, mi in self.reader.members.items():
            if n.startswith("vector:"):
                self._vectors[n[7:]] = _load_vector(self.reader.legacy_blob(n))

    # ---- members
    @property
    def members(self) -> list:
        return self.sample.legacy_members()

    def has(self, member: str) -> bool:
        return member in self.reader.members

    def study_file(self, name: str) -> Optional[bytes]:
        return self._study.get(name)

    def blob(self, aux_name: str) -> bytes:
        return self.reader.legacy_blob(aux_name)

    def cells(self) -> Optional[dict]:
        if self._cells_dict is None:
            t = self.sample.cells()
            self._cells_dict = cells_to_dict(t) if t is not None else None
        return self._cells_dict

    def csc(self, member: str) -> Optional[Csc]:
        mi = self.reader.members.get(member)
        if mi is None:
            return None
        if mi.kind == C.KIND_PZ2:
            return self.sample.matrix(member)
        if mi.kind == C.KIND_DERIVED and member.endswith(".1pz"):
            out = self._regenerate(member, want_matrix=True)
            return (
                rmatrix_to_csc(out, int((mi.meta or {}).get("legacy_vt_code", 3)))
                if not isinstance(out, (bytes, bytearray))
                else decode_1pz_bytes(out)
            )
        if member.endswith(".1pz") and mi.kind in (C.KIND_STORED, C.KIND_RAW):
            try:
                return decode_1pz_bytes(self.reader.legacy_blob(member))
            except Exception:
                return None
        return None

    def matrix(self, member: str) -> Optional[RMatrix]:
        if member not in self._rm:
            c = self.csc(member)
            self._rm[member] = csc_to_rmatrix(c) if c is not None else None
        return self._rm[member]

    def member_bytes(self, member: str) -> Optional[bytes]:
        mi = self.reader.members.get(member)
        if mi is None:
            return None
        if mi.kind in (C.KIND_STORED, C.KIND_RAW):
            return self.reader.legacy_blob(member)
        if mi.kind == C.KIND_DERIVED:
            out = self._regenerate(member, want_matrix=False)
            return bytes(out) if isinstance(out, (bytes, bytearray)) else None
        return None

    def _regenerate(self, member: str, want_matrix: bool):
        """bytes (preferred for non-matrix members) or an RMatrix (derived ``.1pz`` members) of one derived member."""
        key = (member, want_matrix)
        if key in self._out:
            return self._out[key]
        if key in self._busy:
            raise ContextError("recipe cycle through %r" % member)
        mi = self.reader.members[member]
        self._busy.add(key)
        try:
            meta = mi.meta
            if member.endswith(".1pz"):
                out = (
                    self.recipes.regenerate_member_matrix(
                        meta["recipe"], meta.get("params", {}), self
                    )
                    if want_matrix
                    else self.recipes.regenerate_bytes(meta["recipe"], meta.get("params", {}), self)
                )
            else:
                out = self.recipes.regenerate_bytes(meta["recipe"], meta.get("params", {}), self)
        finally:
            self._busy.discard(key)
        self._out[key] = out
        return out

    def regenerate(self, member: str):
        """Legacy bytes of a derived member (for derived ``.1pz``: the container bytes if a writer reproduces them, else raises)."""
        return self._regenerate(member, want_matrix=False)

    def regenerate_matrix(self, member: str):
        return self._regenerate(member, want_matrix=True)

    # ---- packs
    def _vocab_pack(self, kind: str) -> Optional[VocabPack]:
        mi = self.reader.members.get("%s_counts.1pz" % kind)
        if mi is None:
            return None
        ref = (mi.meta or {}).get("names", {}).get("rows")
        if isinstance(ref, dict):
            pk = self.packs.get(ref["pack"])
            return pk if isinstance(pk, VocabPack) else None
        return None


# ---------------------------------------------------------------------------------------------- vectors
def _dump_vector(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a)
    hdr = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}).encode()
    return len(hdr).to_bytes(4, "little") + hdr + a.tobytes()


def _load_vector(b: bytes) -> np.ndarray:
    n = int.from_bytes(b[:4], "little")
    h = json.loads(b[4 : 4 + n].decode())
    return np.frombuffer(b[4 + n :], dtype=np.dtype(h["dtype"])).reshape(h["shape"]).copy()


# ============================================================================================ study level
class StudyContext(_Base):
    """Context for study-level files (``manifest.json``, ``feature_vocab.json``, ...): no sample, only packs and the other study
    files. ``files`` maps study file names to bytes (legacy bytes in migration, regenerated bytes in verification)."""

    def __init__(
        self, gse: str, reference_build: str, files: dict, packs: Optional[PackStore] = None
    ):
        super().__init__()
        self.study_id = gse
        self.sample_id = ""
        self.reference_build = reference_build or ""
        self.protocol_name = ""
        self.packs = packs or default_store()
        self.files = files

    def has(self, member: str) -> bool:
        return member in self.files

    @property
    def members(self) -> list:
        return list(self.files)

    def member_bytes(self, member: str) -> Optional[bytes]:
        return self.files.get(member)

    def study_file(self, name: str) -> Optional[bytes]:
        return self.files.get(name)

    def matrix(self, member):
        return None

    def cells(self):
        return None

    def _vocab_pack(self, kind: str) -> Optional[VocabPack]:
        if not self.reference_build:
            return None
        c = self.packs.vocab_packs(kind, self.reference_build)
        return c[0] if c else None


V2Context = StoreContext  # DUAL: old name, removed in the cleanup
