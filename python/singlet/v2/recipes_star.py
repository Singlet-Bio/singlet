# SPDX-License-Identifier: MIT
"""STARsolo ``Solo.out/Gene`` members (owner: W3).

  whitelist-sorted-v1  raw/barcodes.tsv      the sorted protocol whitelist (a pack)
  features-v1          {raw,filtered}/features.tsv   gene_id, gene_name, "Gene Expression" in STAR's gene order
                       (human: sort by contig, start, end; mouse: a per-build permutation resource)
  mtx-filter-v1        filtered/matrix.mtx and filtered/barcodes.tsv from the raw matrix and the called set
  mtx-text-v1          any matrix.mtx as MatrixMarket text from a decoded CSC (header lines in params when not default)

Evidence and counts: bench/compression/notes/w4_mtx.md and notes/m_w3.md.
"""
from __future__ import annotations

import re

import numpy as np

from . import recipes_mtx as rm

RAW_BC = "star_Solo.out/Gene/raw/barcodes.tsv"
FIL_BC = "star_Solo.out/Gene/filtered/barcodes.tsv"
RAW_MTX = "star_Solo.out/Gene/raw/matrix.mtx"
FIL_MTX = "star_Solo.out/Gene/filtered/matrix.mtx"
RAW_FEAT = "star_Solo.out/Gene/raw/features.tsv"
FIL_FEAT = "star_Solo.out/Gene/filtered/features.tsv"


# ---------------------------------------------------------------------------------------------------
# barcode lists as fixed-width byte arrays
# ---------------------------------------------------------------------------------------------------
def lines_to_array(b: bytes):
    """Newline-terminated, equal-length ASCII lines -> numpy 'S<L>' array, or None."""
    if not b:
        return np.zeros(0, dtype="S1")
    if not b.endswith(b"\n"):
        return None
    first = b.find(b"\n")
    L = first
    if L <= 0 or len(b) % (L + 1) != 0:
        return None
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, L + 1)
    if not (a[:, L] == 10).all() or (a[:, :L] == 10).any() or (a[:, :L] == 13).any():
        return None
    return np.ascontiguousarray(a[:, :L]).view("S%d" % L).reshape(-1)


def array_to_lines(arr) -> bytes:
    arr = np.asarray(arr)
    if arr.size == 0:
        return b""
    L = arr.dtype.itemsize
    a = np.frombuffer(arr.tobytes(), dtype=np.uint8).reshape(-1, L)
    out = np.empty((a.shape[0], L + 1), dtype=np.uint8)
    out[:, :L] = a
    out[:, L] = 10
    return out.tobytes()


def _bc_array(strs):
    return np.array([s.encode("ascii") for s in strs], dtype="S")


def wl_array(wl):
    """A whitelist as the context returns it (numpy 'S' array or list of str) -> numpy 'S' array."""
    if wl is None:
        return None
    if isinstance(wl, np.ndarray) and wl.dtype.kind == "S":
        return wl
    return np.array([x.encode("ascii") if isinstance(x, str) else x for x in wl], dtype="S")


# ---------------------------------------------------------------------------------------------------
# whitelist-sorted-v1
# ---------------------------------------------------------------------------------------------------
WHITELIST_NAMES = ("3M-february-2018", "737K-august-2016", "gex_737K-arc-v1", "737K-april-2014", "737K-cratac-v1")
# barcodes per list; used to skip candidates that cannot match (building a 3.7 M entry list costs seconds)
WHITELIST_SIZES = {"3M-february-2018": 3686400, "737K-august-2016": 737280, "737K-cratac-v1": 737280, "737K-april-2014": 737280,
                   "gex_737K-arc-v1": 736320}


def plan_whitelist(member, legacy_bytes, ctx):
    arr = lines_to_array(legacy_bytes)
    if arr is None or arr.size == 0:
        return None
    names = ctx.whitelist_names() if hasattr(ctx, "whitelist_names") else WHITELIST_NAMES
    for nm in names:
        if WHITELIST_SIZES.get(nm, arr.size) != arr.size:
            continue
        wl = ctx.whitelist(nm)
        if wl is None or len(wl) != arr.size:
            continue
        wl = wl_array(wl)
        if wl.dtype.itemsize != arr.dtype.itemsize:
            continue
        if array_to_lines(wl) == legacy_bytes:
            return ("whitelist-sorted-v1", {"wl": nm}, ["pack:whitelist"])
    return None


def regenerate_whitelist(recipe_id, params, ctx) -> bytes:
    wl = wl_array(ctx.whitelist(params["wl"]))
    return array_to_lines(np.sort(wl))


# ---------------------------------------------------------------------------------------------------
# features-v1
# ---------------------------------------------------------------------------------------------------
def _parse_features(b: bytes):
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not s.endswith("\n"):
        return None
    rows = [r.split("\t") for r in s[:-1].split("\n")]
    if any(len(r) != 3 for r in rows):
        return None
    return rows


def gene_spans(ctx):
    """gene_id -> (chrom, start, end) from the exon vocabulary row names (first exon start, last exon end)."""
    from .recipes_qc import split_row_names

    names = ctx.vocab_names("exon")
    gid, _, chrom = split_row_names(names, False)
    out = {}
    for n, g, c in zip(names, gid, chrom):
        m = re.search(r":(\d+)-(\d+)$", n)
        s, e = int(m.group(1)), int(m.group(2))
        v = out.get(g)
        if v is None:
            out[g] = [c, s, e]
        else:
            v[1] = min(v[1], s)
            v[2] = max(v[2], e)
    return out


def _positional_order(ctx, genes, contigs):
    spans = gene_spans(ctx)
    cix = {c: i for i, c in enumerate(contigs)}
    try:
        key = np.array([(cix[spans[g][0]], spans[g][1], spans[g][2]) for g, _ in genes], dtype=np.int64)
    except KeyError:
        return None
    return np.lexsort((np.arange(len(genes)), key[:, 2], key[:, 1], key[:, 0]))


def _features_order(ctx):
    fo = ctx.features_order()
    if fo is None:
        from .recipes_ctx import load_features_order

        fo = load_features_order(ctx.reference_build)
    return fo


def render_features(genes, order, kind="Gene Expression") -> bytes:
    return "".join("%s\t%s\t%s\n" % (genes[i][0], genes[i][1], kind) for i in order).encode("utf-8")


def plan_features(member, legacy_bytes, ctx):
    rows = _parse_features(legacy_bytes)
    genes = ctx.genes()
    if rows is None or genes is None or len(rows) != len(genes):
        return None
    kinds = {r[2] for r in rows}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    gix = {g: i for i, (g, _) in enumerate(genes)}
    if len(gix) != len(genes):
        return None
    try:
        perm = np.array([gix[r[0]] for r in rows], dtype=np.int64)
    except KeyError:
        return None
    if len(set(perm.tolist())) != len(perm):
        return None
    params = {}
    if kind != "Gene Expression":
        params["kind"] = kind
    if np.array_equal(perm, np.arange(len(genes))):
        params["order"] = "vocab"
        needs = ["pack:genes"]
    else:
        params["order"] = None
        needs = ["pack:genes"]
        fo = _features_order(ctx)
        if fo is not None and np.array_equal(np.asarray(fo, dtype=np.int64), perm):
            params["order"] = "perm"
            needs = ["pack:genes", "pack:features_order"]
        else:
            # contig order = first appearance in the legacy file
            try:
                spans = gene_spans(ctx)
            except Exception:
                spans = None
            if spans is not None:
                contigs = []
                seen = set()
                for r in rows:
                    c = spans[r[0]][0] if r[0] in spans else None
                    if c is not None and c not in seen:
                        seen.add(c)
                        contigs.append(c)
                o = _positional_order(ctx, genes, contigs)
                if o is not None and np.array_equal(o, perm):
                    params["order"] = "pos"
                    params["contigs"] = contigs
                    needs = ["pack:genes", "pack:exon"]
        if params["order"] is None:
            return None
    if regenerate_features("features-v1", params, ctx) != legacy_bytes:
        return None
    return ("features-v1", params, needs)


def regenerate_features(recipe_id, params, ctx) -> bytes:
    genes = ctx.genes()
    o = params["order"]
    if o == "vocab":
        order = np.arange(len(genes))
    elif o == "pos":
        order = _positional_order(ctx, genes, params["contigs"])
    elif o == "perm":
        order = np.asarray(_features_order(ctx), dtype=np.int64)
    else:
        raise ValueError(params)
    return render_features(genes, order, params.get("kind", "Gene Expression"))


def features_order_from_legacy(legacy_bytes, genes):
    """Build the per-build permutation resource: features.tsv row i is vocabulary gene perm[i]."""
    rows = _parse_features(legacy_bytes)
    gix = {g: i for i, (g, _) in enumerate(genes)}
    return np.array([gix[r[0]] for r in rows], dtype=np.int64)


# ---------------------------------------------------------------------------------------------------
# mtx-text-v1
# ---------------------------------------------------------------------------------------------------
def decode_mtx_member(legacy_bytes):
    """Decoded canonical matrix of a STARsolo matrix.mtx member (what W2 stores as a pz2 matrix), or None."""
    r = rm.parse_mtx(legacy_bytes)
    return r[0] if r else None


def aux_name(member):
    """Name of the auxiliary decoded matrix of a text matrix member (migration convention '<member>#csc')."""
    return member + "#csc"


def _aux_matrix(ctx, member, legacy_bytes=None, provide=True):
    """The decoded canonical matrix of a matrix.mtx member: from the context if present, else parsed from the legacy
    bytes and (inside plan()) registered with ``ctx.provide_matrix``.  -> Matrix or None."""
    an = aux_name(member)
    m = ctx.matrix(an)
    if m is not None:
        return m
    if legacy_bytes is None:
        legacy_bytes = ctx.member_bytes(member)
    if legacy_bytes is None:
        return None
    r = rm.parse_mtx(legacy_bytes)
    if r is None:
        return None
    if provide and hasattr(ctx, "provide_matrix"):
        ctx.provide_matrix(an, r[0])
        return ctx.matrix(an) or r[0]
    return r[0]


def plan_mtx_text(member, legacy_bytes, ctx):
    r = rm.parse_mtx(legacy_bytes)
    if r is None:
        return None
    prefix = r[1]
    got = _aux_matrix(ctx, member, legacy_bytes)
    if got is None:
        return None
    params = {}
    if prefix != rm.DEFAULT_HEADER:
        params["hdr"] = prefix.decode("ascii")
    if rm.render_mtx(got, prefix) != legacy_bytes:
        return None
    return ("mtx-text-v1", params, ["matrix:" + aux_name(member)])


def regenerate_mtx_text(recipe_id, params, ctx, member=RAW_MTX) -> bytes:
    hdr = params.get("hdr")
    return rm.render_mtx(ctx.matrix(aux_name(member)), hdr.encode("ascii") if hdr else rm.DEFAULT_HEADER)


# ---------------------------------------------------------------------------------------------------
# mtx-filter-v1
# ---------------------------------------------------------------------------------------------------
def _kept_columns(ctx):
    """(raw barcode array, ascending indices of the raw columns that are called cells) or None."""
    rb = ctx.member_bytes(RAW_BC)
    cells = ctx.cells()
    if rb is None or cells is None or "called" not in cells:
        return None
    raw = lines_to_array(rb)
    if raw is None:
        return None
    called_bc = _bc_array([str(b) for b in np.asarray(cells["barcode"])[np.asarray(cells["called"], dtype=bool)]])
    if called_bc.size == 0:
        return raw, np.zeros(0, dtype=np.int64)
    called_bc = called_bc.astype(raw.dtype) if called_bc.dtype.itemsize <= raw.dtype.itemsize else called_bc
    keep = np.flatnonzero(np.isin(raw, called_bc))
    if keep.size != called_bc.size:
        return None  # some called barcode is not in the raw list
    return raw, keep


def render_filtered_barcodes(ctx) -> bytes:
    k = _kept_columns(ctx)
    if k is None:
        raise LookupError("raw barcodes or called set unavailable")
    raw, keep = k
    return array_to_lines(raw[keep])


def plan_filter(member, legacy_bytes, ctx):
    try:
        if member == FIL_BC:
            if render_filtered_barcodes(ctx) != legacy_bytes:
                return None
            return ("mtx-filter-v1", {"what": "barcodes"}, ["member:" + RAW_BC, "cells:barcode", "cells:called"])
        r = rm.parse_mtx(legacy_bytes)
        if r is None:
            return None
        params = {"what": "matrix"}
        if r[1] != rm.DEFAULT_HEADER:
            params["hdr"] = r[1].decode("ascii")
        # the raw matrix is the auxiliary member of mtx-text-v1; make sure it is registered (member order is not ours)
        if _aux_matrix(ctx, RAW_MTX) is None:
            return None
        out = regenerate_filter("mtx-filter-v1", params, ctx)
    except LookupError:
        return None
    if out != legacy_bytes:
        return None
    return ("mtx-filter-v1", params, ["matrix:" + aux_name(RAW_MTX), "member:" + RAW_BC, "cells:barcode", "cells:called"])


def regenerate_filter(recipe_id, params, ctx) -> bytes:
    if params["what"] == "barcodes":
        return render_filtered_barcodes(ctx)
    k = _kept_columns(ctx)
    raw_mat = ctx.matrix(aux_name(RAW_MTX))
    if k is None or raw_mat is None or raw_mat.shape[1] != len(k[0]):
        raise LookupError("raw matrix / barcodes unavailable or inconsistent")
    sub = rm.select_columns(raw_mat, k[1])
    hdr = params.get("hdr")
    return rm.render_mtx(sub, hdr.encode("ascii") if hdr else rm.DEFAULT_HEADER)
