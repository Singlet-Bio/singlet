# SPDX-License-Identifier: MIT
"""Context protocol and helpers shared by the store migration recipes (owner: W3).

A *recipe* regenerates one legacy bundle member from information that the store sample object already
carries (decoded matrices, the cells table, packs, small vectors) plus tiny per-sample params.  The
migration accepts a recipe only when its output equals the legacy member exactly (bytes, or for derived
``.1pz`` matrices the decoded canonical CSC); otherwise the member is stored.

Everything here is numpy only.  ``Context`` documents what a recipe may ask for; W2 builds the real one,
``DirContext`` (below) builds one from a directory of legacy members and is used by the golden tests.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional

import numpy as np


@dataclasses.dataclass
class Matrix:
    """Canonical CSC matrix of a legacy ``.1pz`` member: rows = features, columns = cells/barcodes.

    ``indptr`` has n+1 entries, row ``indices`` are strictly ascending inside a column, ``data`` are
    non-negative integers (the legacy reader returns uint32).  Explicit zeros are kept (they are part of
    the matrix).  ``rownames``/``colnames`` are lists of str (None when unknown or not needed).
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple
    rownames: Optional[list] = None
    colnames: Optional[list] = None
    user_kv: Optional[dict] = None

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1]) if len(self.indptr) else 0

    def col_ids(self) -> np.ndarray:
        """Column index of every stored entry (length nnz)."""
        n = self.shape[1]
        return np.repeat(np.arange(n, dtype=np.int64), np.diff(self.indptr.astype(np.int64)))

    def col_sums(self) -> np.ndarray:
        """uint64 sum of the values of each column."""
        n = self.shape[1]
        if self.nnz == 0:
            return np.zeros(n, dtype=np.uint64)
        c = np.add.reduceat(
            self.data.astype(np.uint64), self.indptr[:-1].astype(np.int64).clip(max=self.nnz - 1)
        )
        c[np.diff(self.indptr) == 0] = 0
        return c

    def equals(self, other: "Matrix") -> bool:
        return (
            tuple(self.shape) == tuple(other.shape)
            and np.array_equal(self.indptr.astype(np.int64), other.indptr.astype(np.int64))
            and np.array_equal(self.indices.astype(np.int64), other.indices.astype(np.int64))
            and np.array_equal(self.data.astype(np.int64), other.data.astype(np.int64))
            and (
                self.rownames is None
                or other.rownames is None
                or list(self.rownames) == list(other.rownames)
            )
            and (
                self.colnames is None
                or other.colnames is None
                or list(self.colnames) == list(other.colnames)
            )
        )


class Context:
    """What a recipe may ask the migration for.  Every method may return None ("not available").

    Member names are the legacy names relative to the sample directory, e.g. ``"exon_counts.1pz"``,
    ``"cell_calls.tsv"``, ``"star_Solo.out/Gene/raw/matrix.mtx"``.

    ``plan()`` returns the list of things its recipe will need as ``needs`` strings, so the caller knows what
    must exist (and be restored first) when it calls ``regenerate()``:

        "matrix:<member>"   decoded canonical Matrix of a legacy .1pz member
        "member:<name>"     exact legacy bytes of another member (that member must be restorable first)
        "cells:<column>"    column of the sample cells table (see CELLS_COLUMNS)
        "pack:exon" "pack:intron" "pack:genes" "pack:whitelist" "pack:features_order"
        "vector:<name>"     a small integer vector kept by the container (ambient counts ...)
        "meta:<key>"        sample facts: reference_build, protocol_name, sample_id, study_id
    """

    sample_id: str = ""
    study_id: str = ""
    reference_build: str = ""
    protocol_name: str = ""

    def matrix(self, member: str) -> Optional[Matrix]:  # pragma: no cover - interface
        return None

    def member_bytes(self, member: str) -> Optional[bytes]:  # pragma: no cover - interface
        return None

    def cells(self) -> Optional[dict]:  # pragma: no cover - interface
        """Cells-table columns as numpy arrays keyed by name.  ``barcode`` is an object array of str,
        ``called`` a bool array, the rest numeric (see CELLS_COLUMNS)."""
        return None

    def vocab_names(self, kind: str) -> Optional[list]:  # pragma: no cover - interface
        """Row names of the vocabulary pack of this sample's reference build, kind "exon" or "intron"."""
        return None

    def genes(self) -> Optional[list]:  # pragma: no cover - interface
        """Gene table of the build as [(gene_id, gene_name), ...] in vocabulary order."""
        return None

    def whitelist(self, name: str):  # pragma: no cover - interface
        """Sorted numpy 'S<L>' array of the barcodes of the named whitelist pack (None if unknown)."""
        return None

    def whitelist_names(self) -> list:
        """Names of the whitelist packs ``whitelist()`` can serve, in the order recipes should try them."""
        return [
            "3M-february-2018",
            "737K-august-2016",
            "gex_737K-arc-v1",
            "737K-april-2014",
            "737K-cratac-v1",
        ]

    def features_order(self) -> Optional[np.ndarray]:  # pragma: no cover - interface
        """Per-build permutation: STARsolo features.tsv row i is vocabulary gene order[i]."""
        return None

    def vector(self, name: str) -> Optional[np.ndarray]:  # pragma: no cover - interface
        return None

    def meta(self, key: str):
        return getattr(self, key, None)


# Cells-table columns the recipes rely on (names are the container's column names).
CELLS_COLUMNS = {
    # required by the container (format-store.md section 5.3)
    "barcode": "str (object array)",
    "called": "bool",
    "n_umi_exon": "u32",
    "n_umi_intron": "u32",
    "n_feat_exon": "u32",
    "n_feat_intron": "u32",
}


def read_user_kv(d) -> dict:
    return dict(d) if d else {}


# --------------------------------------------------------------------------------------------------
# Directory-backed context used by the golden tests (and handy for W2's own tests)
# --------------------------------------------------------------------------------------------------
_PZ = None


def load_pz():
    """The compiled legacy reader/writer module ``singlet._pz`` (loaded without importing all of ``singlet``)."""
    global _PZ
    if _PZ is not None:
        return _PZ
    import glob
    import importlib.util
    import sys

    if "singlet._pz" in sys.modules:
        _PZ = sys.modules["singlet._pz"]
        return _PZ
    if "singlet" in sys.modules:
        from singlet import _pz  # type: ignore

        _PZ = _pz
        return _pz
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)  # python/singlet
    cands = sorted(
        glob.glob(os.path.join(pkg, "_pz*.pyd")) + glob.glob(os.path.join(pkg, "_pz*.so"))
    )
    if cands:
        spec = importlib.util.spec_from_file_location("singlet._pz", cands[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["singlet._pz"] = mod
        _PZ = mod
        return mod
    from singlet import _pz  # type: ignore

    _PZ = _pz
    return _pz


def read_1pz_matrix(path: str, with_names: bool = True) -> Matrix:
    r = load_pz().read_1pz(path)
    return Matrix(
        indptr=np.asarray(r["indptr"]),
        indices=np.asarray(r["indices"]),
        data=np.asarray(r["data"]),
        shape=(int(r["m"]), int(r["n"])),
        rownames=r["rownames"] if with_names else None,
        colnames=r["colnames"] if with_names else None,
        user_kv=dict(r["user_kv"]),
    )


def _flat(name: str) -> str:
    return name.replace("/", "__")


class DirSource:
    """Members of one sample as files of a directory (``/`` in a member name may be written ``__``)."""

    def __init__(self, path):
        self.path = str(path)

    def file(self, member):
        for cand in (member, _flat(member)):
            p = os.path.join(self.path, cand)
            if os.path.isfile(p):
                return p
        return None

    def read(self, member):
        p = self.file(member)
        if p is None:
            return None
        with open(p, "rb") as f:
            return f.read()


class ZipSource:
    """Members ``<prefix><member>`` of a legacy ``.singlet`` bundle (a zip)."""

    def __init__(self, zip_path, prefix):
        import zipfile

        self.z = zipfile.ZipFile(zip_path)
        self.prefix = prefix
        self._names = set(self.z.namelist())

    def file(self, member):
        return None

    def read(self, member):
        n = self.prefix + member
        return self.z.read(n) if n in self._names else None


def decode_1pz_bytes(b: bytes, with_names=True) -> Matrix:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".1pz")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(b)
        return read_1pz_matrix(path, with_names)
    finally:
        os.unlink(path)


class DirContext(Context):
    """Context over the legacy members of one sample, from one or more sources.

    ``source`` is a directory path (str/Path), a ``DirSource``/``ZipSource``, or a list of them (first hit wins).
    Vocabulary, genes, whitelists and the cells table are derived from the sample's own legacy members unless
    explicit overrides are passed (the real migration reads them from packs and the cells table instead; both
    yield the same values, the tests check that wherever packs are available).
    """

    def __init__(
        self,
        source,
        sample_id="",
        study_id="",
        reference_build="",
        protocol_name="",
        overrides=None,
        extra_dirs=(),
    ):
        srcs = source if isinstance(source, (list, tuple)) else [source]
        srcs = list(srcs) + list(extra_dirs)
        self.sources = [s if hasattr(s, "read") else DirSource(s) for s in srcs]
        first = self.sources[0]
        self.sample_id = sample_id or os.path.basename(
            getattr(first, "path", getattr(first, "prefix", "")).rstrip("/\\")
        )
        self.study_id = study_id
        self._ov = overrides or {}
        self._mats = {}
        self._cells = None
        summ = self.member_bytes("summary.json")
        sj = json.loads(summ) if summ else {}
        self.reference_build = reference_build or sj.get("reference_build", "")
        self.protocol_name = protocol_name or sj.get("protocol_name", "")

    # -- members ----------------------------------------------------------------------------
    def has(self, member) -> bool:
        return member in self._ov or any(
            s.read(member) is not None if s.file(member) is None else True for s in self.sources
        )

    def member_bytes(self, member):
        if member in self._ov:
            return self._ov[member]
        for s in self.sources:
            b = s.read(member)
            if b is not None:
                return b
        return None

    def matrix(self, member):
        if member in self._mats:
            return self._mats[member]
        m = None
        if not member.endswith(".1pz"):
            return None  # like the migration context: text matrices exist only as auxiliary members (provide_matrix)
        for s in self.sources:
            p = s.file(member)
            if p:
                m = read_1pz_matrix(p)
                break
            b = s.read(member)
            if b is not None:
                m = decode_1pz_bytes(b)
                break
        self._mats[member] = m
        return m

    # -- derived views ----------------------------------------------------------------------
    def cells(self):
        """Cells table as the container will store it: barcodes (exon matrix column order), the called flag and
        per-cell matrix statistics, plus any optional columns added with ``set_cells``/``add_cells``."""
        if self._cells is not None:
            return self._cells
        ex = self.matrix("exon_counts.1pz")
        if ex is None:
            return None
        cols = {"barcode": np.array(ex.colnames, dtype=object)}
        cc = self.member_bytes("cell_calls.tsv")
        if cc is not None:
            from . import recipes_cells

            cols["called"] = recipes_cells.called_flags(cc, ex.colnames)
        cols["n_umi_exon"] = ex.col_sums().astype(np.uint32)
        it = self.matrix("intron_counts.1pz")
        if it is not None and it.shape[1] == ex.shape[1]:
            cols["n_umi_intron"] = it.col_sums().astype(np.uint32)
        self._cells = cols
        return cols

    def set_cells(self, cols):
        self._cells = cols

    def add_cells(self, extra):
        """Add columns returned by ``recipes.cells_columns`` ({name: (dtype, array)}) to the cells table."""
        c = dict(self.cells() or {})
        for k, v in extra.items():
            c[k] = v[1] if isinstance(v, tuple) else v
        self._cells = c

    def vocab_names(self, kind):
        if ("vocab", kind) in self._ov:
            return self._ov[("vocab", kind)]
        m = self.matrix("exon_counts.1pz" if kind == "exon" else "intron_counts.1pz")
        return m.rownames if m is not None else None

    def genes(self):
        if "genes" in self._ov:
            return self._ov["genes"]
        names = self.vocab_names("exon")
        if names is None:
            return None
        from .recipes_qc import split_row_names

        gid, sym, _ = split_row_names(names, False)
        out, last = [], None
        for g, sy in zip(gid, sym):
            if g != last:
                out.append((g, sy if sy else g))
                last = g
        return out

    def whitelist(self, name):
        if ("whitelist", name) in self._ov:
            return self._ov[("whitelist", name)]
        return load_package_whitelist(name)

    def features_order(self):
        if "features_order" in self._ov:
            return self._ov["features_order"]
        return load_features_order(self.reference_build)

    def vector(self, name):
        return self._ov.get(("vector", name))

    # auxiliary members (the migration registers them inside plan())
    def provide_matrix(self, aux_name, matrix):
        self._mats[aux_name] = matrix

    def provide_vector(self, name, arr):
        self._ov[("vector", name)] = np.asarray(arr)

    def provide_blob(self, aux_name, data):
        self._ov[aux_name] = bytes(data)


# --------------------------------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------------------------------
_WL = {}


def load_package_whitelist(name):
    """Sorted unique 'S<L>' array of a whitelist text file shipped in python/singlet/data/whitelists (CRLF tolerated)."""
    if name in _WL:
        return _WL[name]
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(os.path.dirname(here), "data", "whitelists", name + ".txt")
    if not os.path.isfile(p):
        _WL[name] = None
        return None
    with open(p, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    lines = [x for x in data.split(b"\n") if x]
    arr = np.unique(np.array(lines, dtype="S"))
    _WL[name] = arr
    return arr


def resource_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes_data")


def load_features_order(build):
    """Per-build STARsolo features.tsv permutation (see recipes_star.features_order_from_legacy), or None."""
    p = os.path.join(resource_dir(), "features_order_%s.u32.zst" % build)
    if not os.path.isfile(p):
        return None
    import zstandard

    with open(p, "rb") as f:
        raw = zstandard.ZstdDecompressor().decompress(f.read(), max_output_size=1 << 28)
    return np.frombuffer(raw, dtype="<u4").astype(np.int64)


def save_features_order(build, perm):
    import zstandard

    os.makedirs(resource_dir(), exist_ok=True)
    p = os.path.join(resource_dir(), "features_order_%s.u32.zst" % build)
    with open(p, "wb") as f:
        f.write(
            zstandard.ZstdCompressor(level=19).compress(np.asarray(perm, dtype="<u4").tobytes())
        )
    return p
