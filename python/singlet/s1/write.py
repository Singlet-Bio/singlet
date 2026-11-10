# SPDX-License-Identifier: MIT
"""Write and read a count matrix as a ``.s1`` sample object of your own (docs/format-s1.md).

:func:`write_s1` stores one integer count matrix, with its gene ids and cell barcodes, as a sample object holding a single
pz2 member (``counts`` by default) coded by the compiled codec (``singlet._pz2``, the lane-rANS codec, or the generic codec
when that is smaller). :func:`read_s1` reads it back exactly. ``singlet.load(path)`` routes such a file to :func:`read_s1`,
and :class:`singlet.torch.S1Cells` streams minibatches from it, locally or over HTTP ranges.

Orientation follows AnnData: an array or scipy matrix is **cells x genes**. The object stores genes x cells (one column per
cell), which is what lets a reader fetch a subset of cells with range requests. The R package writes the same bytes with
``singlet::write_s1`` (genes x cells, the Bioconductor orientation).

Atlas sample objects (members ``exon_counts.1pz``, ``cells``, ...) are read by :mod:`singlet.s1.client`, not by these
functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from . import container as C
from .codec import EncodeOptions, NativeCodec
from .matrix import VT_U32, Csc, canonical_digest

#: member name used by :func:`write_s1` unless another is given
DEFAULT_MEMBER = "counts"
#: atlas member names; an object with one of these is an atlas sample object, not a :func:`write_s1` object
_ATLAS_MEMBERS = ("exon_counts.1pz", "intron_counts.1pz", "cells")

PathLike = Union[str, os.PathLike]


_PACK_ID_ERROR = "gene_pack must be a vocabulary pack id: 16 to 32 hex characters, or 8 to 16 bytes"


def _pack_hex(gene_pack) -> str:
    """The hex id (or id prefix of at least 8 bytes) of a vocabulary pack given as bytes or hex characters."""
    if isinstance(gene_pack, (bytes, bytearray)):
        pid = bytes(gene_pack)
    else:
        try:
            pid = bytes.fromhex(str(gene_pack))
        except ValueError:
            raise ValueError(_PACK_ID_ERROR) from None
    if not 8 <= len(pid) <= 16:
        raise ValueError(_PACK_ID_ERROR)
    return pid.hex()


def _find_pack(hexid: str):
    """The pack with this id (prefix) from the installed packs, else downloaded into the user cache; ValueError if neither."""
    from .packs import PackError, PackStore, default_store

    # the shared store may already download (migrate turns that on), so a fetch error can come from either get
    store = default_store()
    try:
        try:
            return store.get(hexid)
        except PackError:
            if store.download:
                raise
        pk = PackStore(store.dirs, download=True, remote=store.remote).get(hexid)
    except (PackError, OSError) as e:
        raise ValueError(
            "gene_pack %s: the vocabulary pack is not installed and could not be downloaded (%s); store the gene ids "
            "inline with gene_pack=None" % (hexid, e)
        ) from e
    store.rescan()
    return pk


def pack_rows(gene_pack, genes: Sequence[str]) -> tuple[bytes, dict]:
    """(8-byte id for the pz2 header, ``meta.names.rows``) for gene ids stored as a reference to a vocabulary pack.

    The ids must be the pack's gene table in order (a gene-level matrix: ``member`` ``"genes"``) or its row names in order
    (an exon- or intron-level matrix: ``member`` the pack's member), as ``migrate._row_names_for`` writes them for the atlas.
    Anything else raises ValueError: a reference replaces the ids, so ids the pack does not hold would be lost.
    """
    from .packs import VocabPack

    hexid = _pack_hex(gene_pack)
    pk = _find_pack(hexid)
    if not isinstance(pk, VocabPack):
        raise ValueError("gene_pack %s is not a vocabulary pack" % hexid)
    genes = list(genes)
    table = [g for g, _n in pk.gene_table()]
    if genes == table:
        member = "genes"
    elif len(genes) == pk.n() and genes == pk.names():
        member = pk.member
    else:
        where = ""
        if len(genes) == len(table):
            k = next(i for i, (a, b) in enumerate(zip(genes, table)) if a != b)
            where = " (gene id %d is %r, the pack's is %r)" % (k + 1, genes[k], table[k])
        raise ValueError(
            "gene_pack %s holds %d genes and %d %s rows; the %d gene ids given are neither, in that order%s. Store them "
            "inline with gene_pack=None"
            % (pk.id.hex(), len(table), pk.n(), pk.member, len(genes), where)
        )
    return pk.id[:8], {"pack": pk.id.hex(), "member": member}


def _as_cells_by_genes(x, genes, barcodes, layer):
    """(scipy CSR cells x genes, gene ids, barcodes) from AnnData, a scipy sparse matrix or a dense array."""
    import scipy.sparse as sp

    if hasattr(x, "obs_names") and hasattr(x, "var_names"):  # AnnData
        m = x.X if layer is None else x.layers[layer]
        genes = list(map(str, x.var_names)) if genes is None else list(map(str, genes))
        barcodes = list(map(str, x.obs_names)) if barcodes is None else list(map(str, barcodes))
        if m is None:
            raise ValueError(
                "the AnnData object has no matrix in %s"
                % ("X" if layer is None else "layers[%r]" % layer)
            )
        if hasattr(m, "to_memory"):  # backed / lazy arrays
            m = m.to_memory()
        x = m
    elif layer is not None:
        raise ValueError("layer= applies to AnnData input only")
    if genes is None or barcodes is None:
        raise ValueError("genes= and barcodes= are required unless x is an AnnData object")
    xs = sp.csr_matrix(x)
    n_cells, n_genes = (int(v) for v in xs.shape or (0, 0))
    if len(genes) != n_genes or len(barcodes) != n_cells:
        raise ValueError(
            "x is %d cells x %d genes (cells x genes, as in AnnData) but %d gene ids and %d barcodes were given"
            % (n_cells, n_genes, len(genes), len(barcodes))
        )
    return xs, [str(g) for g in genes], [str(b) for b in barcodes]


def _canonical_csc(xs, genes, barcodes) -> Csc:
    """The canonical genes x cells CSC of a cells x genes CSR: integer counts > 0, row ids strictly ascending per cell."""
    import scipy.sparse as sp

    t = sp.csc_matrix(xs.T, copy=True)  # genes x cells, one column per cell
    t.sum_duplicates()
    t.eliminate_zeros()
    t.sort_indices()
    d = t.data
    if d.size:
        if np.issubdtype(d.dtype, np.floating):
            if not np.all(np.isfinite(d)) or np.any(d != np.floor(d)):
                raise ValueError(
                    "x has non-integer values: write_s1 stores counts, not normalised data"
                )
        if np.any(d < 0):
            raise ValueError("x has negative values: write_s1 stores counts")
        if d.max() > np.iinfo(np.uint32).max:
            raise ValueError("a count exceeds 2**32 - 1")
    m, n = (int(v) for v in t.shape or (0, 0))
    if m >= 2**32 or n >= 2**32:
        raise ValueError("a .s1 matrix has fewer than 2**32 rows and columns")
    return Csc(
        m=int(m),
        n=int(n),
        indptr=t.indptr.astype(np.int64),
        indices=t.indices.astype(np.uint32),
        data=d.astype(np.uint32),
        rownames=genes,
        colnames=barcodes,
        value_type=VT_U32,
    )


def encode_counts(csc: Csc, *, gene_pack=None, threads: int = 4) -> tuple[bytes, bytes, dict]:
    """The pz2 member, canonical digest and member meta of a canonical CSC (gene ids inline, or as a reference to
    ``gene_pack``, checked by :func:`pack_rows`)."""
    meta: dict = {"modality": "counts"}
    row_names: object = "inline"
    if gene_pack is not None:
        pid, rows = pack_rows(gene_pack, csc.rownames)
        row_names = ("pack", pid)
        meta["names"] = {"rows": rows}
    digest = canonical_digest(csc)
    opts = EncodeOptions(
        row_names=row_names,
        col_names="inline",
        threads=int(threads),
        digest=digest,
    )
    return NativeCodec("auto").encode(csc, opts), digest, meta


def write_s1(
    x,
    path: PathLike,
    *,
    genes: Optional[Sequence[str]] = None,
    barcodes: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    sample_id: str = "sample",
    member: str = DEFAULT_MEMBER,
    gene_pack: Union[None, bytes, str] = None,
    threads: int = 4,
) -> Path:
    """Write a count matrix as a ``.s1`` sample object; returns the path.

    Parameters
    ----------
    x
        AnnData (``X``, or ``layers[layer]``), a scipy sparse matrix, or a dense array, **cells x genes**. Values must be
        non-negative integers below 2**32 (integer-valued floats are accepted).
    path
        File to write. It is written to a temporary name next to it and renamed, so a reader never sees a partial file.
    genes, barcodes
        Gene ids and cell barcodes. Taken from ``var_names`` / ``obs_names`` for AnnData; required otherwise.
    sample_id
        ASCII id stored in the object header, at most 16 bytes.
    member
        Name of the matrix member.
    gene_pack
        Store the gene ids as a reference to a vocabulary pack (its id, or an id prefix of at least 16 hex characters)
        instead of inline. The ids must be the pack's gene table in order (a gene-level matrix) or its row names in order
        (exon or intron level); anything else raises ValueError. The pack is taken from the installed packs, or downloaded
        from data.singlet.bio; ValueError if it cannot be. A reader needs the same pack, or the gene ids passed to
        :func:`read_s1`.
    threads
        Encoder threads. The output does not depend on it.
    """
    xs, genes, barcodes = _as_cells_by_genes(x, genes, barcodes, layer)
    csc = _canonical_csc(xs, genes, barcodes)
    body, digest, meta = encode_counts(csc, gene_pack=gene_pack, threads=threads)
    w = C.S2Writer(sample_id)
    w.add_matrix(member, body, digest, meta)
    out = Path(path).expanduser()
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(w.build())
    os.replace(tmp, out)
    return out


def user_member(path: PathLike) -> Optional[str]:
    """The matrix member to read from a sample object that is not an atlas sample (none of the atlas members): ``counts``
    if it has one, else its only pz2 matrix member. None for an atlas sample, a file that is not a sample object, or one
    with several matrices and no ``counts``."""
    try:
        members = C.S2Reader(str(path)).members
    except (OSError, C.S2Error):
        return None
    if any(a in members for a in _ATLAS_MEMBERS):
        return None
    if DEFAULT_MEMBER in members and members[DEFAULT_MEMBER].kind == C.KIND_PZ2:
        return DEFAULT_MEMBER
    mats = [n for n, mi in members.items() if mi.kind == C.KIND_PZ2]
    return mats[0] if len(mats) == 1 else None


def is_user_object(path: PathLike) -> bool:
    """True for a one-matrix sample object such as a :func:`write_s1` file (see :func:`user_member`)."""
    return user_member(path) is not None


def read_s1(
    path: PathLike,
    *,
    member: str = DEFAULT_MEMBER,
    genes: Optional[Sequence[str]] = None,
    dtype=np.float32,
):
    """Read a :func:`write_s1` object as AnnData (cells x genes, counts in ``X`` as a CSR matrix of ``dtype``).

    ``genes`` supplies the gene ids of an object written with ``gene_pack=`` when the pack is not installed. ``dtype``
    float32 holds counts up to 2**24 exactly; pass ``np.uint32`` or ``np.float64`` for deeper data.
    """
    import anndata as ad
    import scipy.sparse as sp

    from .packs import PackError
    from .sample import S1Error, S1Sample

    s = S1Sample(str(path))
    if member not in s.members:
        raise KeyError("%s has no member %r (members: %s)" % (path, member, ", ".join(s.members)))
    csc = s.matrix(member, resolve=False)
    rows = csc.rownames
    if rows is None:
        if genes is not None:
            rows = [str(g) for g in genes]
        else:
            pack = (csc.row_ref or {}).get("pack")
            try:
                s._resolve(csc, s.reader.members[member], None)
                rows = csc.rownames
            except PackError as e:
                raise ValueError(
                    "the gene ids of %s are in vocabulary pack %s, which is not installed; pass genes="
                    % (path, pack)
                ) from e
            except S1Error as e:  # the pack is there but does not describe these rows
                raise ValueError(
                    "%s: the gene ids cannot be taken from vocabulary pack %s: %s; pass genes="
                    % (path, pack, e)
                ) from e
    cols = csc.colnames
    if rows is None or cols is None:
        raise ValueError("%s: member %r has unresolved gene ids or barcodes" % (path, member))
    if len(rows) != csc.m:
        raise ValueError("%d gene ids for %d genes" % (len(rows), csc.m))
    x = sp.csc_matrix(
        (
            np.asarray(csc.data).astype(dtype),
            np.asarray(csc.indices).astype(np.int64),
            np.asarray(csc.indptr).astype(np.int64),
        ),
        shape=(csc.m, csc.n),
    )
    a = ad.AnnData(X=sp.csr_matrix(x.T))
    a.var_names = [str(g) for g in rows]
    a.obs_names = [str(b) for b in cols]
    a.uns["s1"] = {"sample_id": s.sample_id, "member": member}
    return a
