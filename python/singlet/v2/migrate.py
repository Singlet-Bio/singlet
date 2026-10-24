# SPDX-License-Identifier: MIT
"""Legacy ``.singlet`` bundle -> v2 objects.

    python -m singlet.v2.migrate --bundle PATH_OR_URL --out DIR [--gse GSE...] [--samples GSM,GSM] [--recipes MODULE|none]

Output tree under ``DIR`` (the R2 key space)::

    v2/s/<GSM>/<d16hex>.s2     sample objects, named by the digest of their bytes
    v2/m/<GSE>.json            study manifest
    v2/ledger/<GSE>.jsonl      one line per legacy member (and per v2-only member), plus sample / study summaries
    v2/ref/<packid>.pack, v2/ref/index.json    the packs the objects refer to

Verify-or-store (docs/format-v2.md section 1): every component is rebuilt from the finished object alone and compared with the
legacy content; a recipe that does not reproduce a member on that very sample is replaced by storing the member (kind 3, zstd-19).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
from typing import Optional

import numpy as np

from . import cellcall, util
from . import cells as cells_mod
from . import container as C
from . import manifest as M
from . import packs as P
from .codec import CodecError, EncodeOptions, PyCodec2, get_codec, parse_header
from .context import (
    LegacyContext,
    StudyContext,
    V2Context,
    rmatrix_to_csc,
)
from .legacy import LegacyBundle
from .matrix import Csc
from .rcp_adapter import Recipes, load_recipes
from .sample import V2Sample

ANNDATA_MEMBERS = ("exon_counts.1pz", "intron_counts.1pz", "cell_calls.tsv")
NEVER_DERIVED = (
    "exon_counts.1pz",
    "intron_counts.1pz",
)  # always pz2: recipes are not asked about them
KIND_LABEL = {
    C.KIND_PZ2: "pz2",
    C.KIND_CELLS: "cells",
    C.KIND_STORED: "stored",
    C.KIND_DERIVED: "derived",
    C.KIND_RAW: "raw",
}


class Options:
    def __init__(
        self,
        chunk_cols: int = 0,
        level: int = util.ZSTD_LEVEL,
        codec: Optional[str] = None,
        recipes=None,
        created_at: Optional[str] = None,
        threads: int = 4,
        log=None,
        packs: Optional[P.PackStore] = None,
        max_stage2_rounds: int = 4,
        junction_table=None,
        cell_call_v2: Optional[bool] = None,
        chunk_nnz_max: int = 0,
    ):
        self.chunk_cols = chunk_cols
        # 0 = pz2 default budget (codec.DEFAULT_CHUNK_NNZ_MAX), -1 = none
        self.chunk_nnz_max = chunk_nnz_max
        self.level = level
        self.codec = get_codec(codec)
        self.recipes: Recipes = recipes if isinstance(recipes, Recipes) else load_recipes(recipes)
        self.created_at = created_at or _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.threads = threads
        self.log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
        self.packs = packs or P.default_store()
        self.max_stage2_rounds = max_stage2_rounds
        # junction rows stored as table ids (docs/format-v2.md 12.4): a junctions.JunctionTable (its packs reachable through
        # ``packs``), "auto" (the newest table published for the study's reference build in v2/ref/index.json; inline names
        # when there is none), a callable(build) -> table or None, or None (inline names). Resolved per study by migrate_bundle.
        self.junction_spec = junction_table
        self.junction_table = (
            junction_table if not (junction_table == "auto" or callable(junction_table)) else None
        )
        # called_v2 (docs/called-v2.md): True = required (fails without numba), None = when numba is installed, False = never
        self.cell_call_v2 = cell_call_v2


# ================================================================================================ comparisons (migration side)
def csc_diff(a: Csc, b: Csc) -> Optional[str]:
    """None if the two matrices are identical (dims, indptr, indices, data, row/col names, key/values), else the first difference."""
    if (a.m, a.n) != (b.m, b.n):
        return "shape %dx%d vs %dx%d" % (a.m, a.n, b.m, b.n)
    if not np.array_equal(np.asarray(a.indptr, np.int64), np.asarray(b.indptr, np.int64)):
        return "indptr differs"
    if not np.array_equal(np.asarray(a.indices, np.int64), np.asarray(b.indices, np.int64)):
        return "indices differ"
    if not np.array_equal(np.asarray(a.data, np.int64), np.asarray(b.data, np.int64)):
        return "data differs"
    if list(a.rownames or []) != list(b.rownames or []):
        return "row names differ"
    if list(a.colnames or []) != list(b.colnames or []):
        return "column names differ"
    if dict(a.kv or {}) != dict(b.kv or {}):
        return "key/values differ"
    return None


# ================================================================================================ names policy
def _row_names_for(csc: Csc, rel: str, lctx: LegacyContext, opts: Options):
    """-> (codec ref, meta) for the rows of a matrix: vocabulary pack when the names equal it byte for byte, else inline."""
    if csc.rownames is None:
        return "none", "none"
    kind = {"exon_counts.1pz": "exon", "intron_counts.1pz": "intron"}.get(rel)
    if kind and csc.m:
        pk = lctx._vocab_pack(kind)
        if pk is not None and pk.matches(csc.rownames):
            return ("pack", pk.id[:8]), {"pack": pk.id.hex(), "member": kind}
    if (
        kind is None and csc.m >= 1000 and csc.rownames
    ):  # auxiliary gene-level matrices: rows = the gene table in vocabulary order
        pk = lctx._vocab_pack("exon")
        if pk is not None and pk.header.get("n_genes") == csc.m:
            genes = [g for g, _n in pk.gene_table()]
            if genes == csc.rownames:
                return ("pack", pk.id[:8]), {"pack": pk.id.hex(), "member": "genes"}
    return "inline", "inline"


JUNCTION_MEMBERS = ("sj_counts.1pz",)


def _junction_rows(w, rel, m, basis_barcodes, opts: Options, vt_code: int, flags: int):
    """Write a junction matrix with rows as ids into ``opts.junction_table`` (docs/format-v2.md 12.4), plus the ``.local`` and
    ``.legacy_order`` members; the member digest stays the legacy matrix's, so every check against the legacy bundle is unchanged.
    Returns None (and writes nothing) when the codec refuses, so the caller falls back to inline names."""
    from . import junctions as J

    t = opts.junction_table
    c = m.csc
    ip, ix, dt, ids, local, perm = J.from_legacy(c.indptr, c.indices, c.data, c.rownames, t)
    v2 = Csc(
        c.m,
        c.n,
        ip,
        ix.astype(np.uint32),
        dt.astype(c.data.dtype),
        None,
        c.colnames,
        dict(c.kv),
        c.value_type,
    )
    v2.rownames = J.row_names(t, ids, J.decode_local(local) if local else None)
    col_ref, col_meta = _col_names_for(c, basis_barcodes, opts)
    eo = EncodeOptions(
        chunk_cols=opts.chunk_cols,
        chunk_nnz_max=opts.chunk_nnz_max,
        row_names=("subset", t.id[:8], np.asarray(ids, np.uint32)),
        col_names=col_ref,
        level=opts.level,
        threads=opts.threads,
        digest=c.digest(),  # the logical matrix is the legacy one (rows in legacy order, legacy names)
    )
    try:
        body = opts.codec.encode(v2, eo)
    except CodecError:
        return None
    meta = {
        "m": c.m,
        "n": c.n,
        "nnz": c.nnz,
        "legacy_vt_code": vt_code,
        "chunk_cols": parse_header(body)["chunk_cols"],
        "names": {"rows": {"junctions": t.id.hex(), "build": t.build}, "cols": col_meta},
        "legacy_size": m.entry.size,
        "legacy_crc32": m.entry.crc32,
    }
    w.add_matrix(rel, body, c.digest(), meta, flags)
    if local:
        w.add_stored(rel + J.LOCAL_SUFFIX, local, {"of": rel}, 0, opts.level)
    w.add_stored(rel + J.LEGACY_ORDER_SUFFIX, perm, {"of": rel}, 0, opts.level)
    return True


def _col_names_for(csc: Csc, basis_barcodes: Optional[list], opts: Options):
    if csc.colnames is None:
        return "none", "none"
    if csc.n == 0:
        return "inline", "inline"
    if (
        basis_barcodes is not None
        and csc.n == len(basis_barcodes)
        and csc.colnames == basis_barcodes
    ):
        return "cells", "cells"
    if csc.n >= 10000 and csc.colnames:
        hit = opts.packs.find_whitelist(csc.colnames)
        if hit is not None:
            w, idx = hit
            if len(idx) == w.n and np.array_equal(idx, np.arange(w.n)):
                return ("pack", w.id[:8]), {"pack": w.id.hex()}  # the whole list: nothing stored
            return ("subset", w.id[:8], idx), {"pack": w.id.hex()}
    return "inline", "inline"


# ================================================================================================ per sample
class _Member:
    __slots__ = (
        "rel",
        "path",
        "entry",
        "treat",
        "note",
        "csc",
        "plan",
        "bytes_exact",
        "body",
        "meta",
        "verified",
        "verify_mode",
        "contrib",
        "force_py",
    )

    def __init__(self, rel, path, entry):
        self.rel, self.path, self.entry = rel, path, entry
        self.treat = "stored"  # pz2 | derived | stored
        self.note = ""
        self.csc: Optional[Csc] = None
        self.plan = None
        self.bytes_exact = False
        self.body = None
        self.meta = None
        self.verified = False
        self.verify_mode = ""
        self.contrib: list = []
        self.force_py = False


def _parse_json(b: Optional[bytes]) -> dict:
    if not b:
        return {}
    try:
        v = json.loads(b.decode("utf-8"))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def migrate_sample(
    bundle: LegacyBundle, gsm: str, gse: str, reference_build: str, study_bytes: dict, opts: Options
) -> dict:
    t0 = time.time()
    log = opts.log
    packs = opts.packs
    recipes = opts.recipes
    paths = bundle.sample_paths(gsm)
    summary = _parse_json(
        bundle.read("samples/%s/summary.json" % gsm)
        if bundle.exists("samples/%s/summary.json" % gsm)
        else None
    )
    protocol = str(summary.get("protocol_name", ""))
    build = reference_build or summary.get("reference_build", "")
    pipeline = {
        k: summary.get(k) for k in ("git_sha", "singlet_version") if summary.get(k) is not None
    }
    facts = {
        k: v for k, v in (("reference_build", build), ("protocol_name", protocol)) if v
    }  # sample facts recipes may ask for
    lctx = LegacyContext(bundle, gsm, packs, gse, build, protocol, study_bytes)
    mem: dict[str, _Member] = {}
    order = []
    for rel, full in paths:
        mem[rel] = _Member(rel, full, bundle.entries[full])
        order.append(rel)

    # ---- 1. matrices
    for rel in order:
        if not rel.endswith(".1pz"):
            continue
        m = mem[rel]
        c = lctx.csc(rel)
        if c is None:
            m.note = "undecodable .1pz (stored verbatim)"
            continue
        why = c.check_canonical()
        if why:
            m.note = "not canonical CSC (%s); stored verbatim" % why
            continue
        m.csc, m.treat = c, "pz2"

    # ---- 2. cells table (required columns) and the barcode basis
    ex, it = mem.get("exon_counts.1pz"), mem.get("intron_counts.1pz")
    ex_c = ex.csc if ex is not None and ex.treat == "pz2" else None
    it_c = it.csc if it is not None and it.treat == "pz2" else None
    basis = (
        ex_c
        if ex_c is not None and ex_c.n > 0
        else (it_c if it_c is not None and it_c.n > 0 else None)
    )
    base = None
    basis_barcodes = None
    wl_id = None
    if basis is not None:
        basis_barcodes = list(basis.colnames)
        hit = packs.find_whitelist(basis_barcodes)
        wl = None
        if hit is not None:
            wl = hit
            wl_id = hit[0].id.hex()
        ccb = lctx.member_bytes("cell_calls.tsv")
        called_list = cells_mod.parse_cell_calls(ccb) if ccb is not None else None
        if called_list is None:
            called = None
            called_src = "all"
        else:
            cs = set(called_list)
            called = np.fromiter(
                (b in cs for b in basis_barcodes), dtype=bool, count=len(basis_barcodes)
            )
            called_src = "cell_calls.tsv"
        base = cells_mod.base_table(basis_barcodes, ex_c, it_c, called, wl)
        base.meta = {"called_source": called_src, "whitelist": wl_id}
        if called is not None:
            _add_called_v2(base, basis_barcodes, ex_c, it_c, protocol, opts, gsm)

    # ---- 3. candidate cells columns from the recipes
    cand: dict[str, tuple] = {}  # name -> (dtype, values, contributor rel)
    if base is not None:
        full = base.copy()
        lctx.set_cells(full)
        for rel in order:
            m = mem[rel]
            if rel in NEVER_DERIVED:
                continue
            try:
                cols = recipes.cells_columns(rel, lctx.member_bytes(rel), lctx)
            except Exception as e:
                log("  [%s] cells_columns(%s) failed: %s" % (gsm, rel, e))
                continue
            for name, (dt, vals) in (cols or {}).items():
                if name in base.names or name in cand:
                    continue
                try:
                    full.add(name, dt, vals)
                except Exception as e:
                    log("  [%s] cells column %r from %s rejected: %s" % (gsm, name, rel, e))
                    continue
                cand[name] = (dt, vals, rel)
        lctx.set_cells(full)

    # ---- 4. plans (stage 1: against the legacy context)
    for rel in order:
        m = mem[rel]
        if rel in NEVER_DERIVED:
            continue
        data = lctx.member_bytes(rel)
        lctx._planning = True
        try:
            plan = recipes.plan(rel, data, lctx)
        except Exception as e:
            plan = None
            m.note = "plan error: %s" % e
        finally:
            lctx._planning = False
        if plan is None:
            continue
        ok, why, exact = _verify_plan_stage1(recipes, rel, data, plan, lctx, m)
        if ok:
            m.treat, m.plan, m.bytes_exact = "derived", plan, exact
        else:
            m.note = "recipe %s rejected: %s" % (plan[0], why)
        # auxiliary members registered during plan() stay registered (another member's plan may rely on them, e.g. the raw matrix
        # registered by the filtered-matrix plan); only the ones named in the ``needs`` of accepted plans are stored

    # ---- 5/6. assemble + stage 2 (rebuild without a failing recipe until everything verifies)
    rounds = 0
    while True:
        rounds += 1
        obj, infos, cells_final = _assemble(
            gsm, order, mem, base, cand, lctx, opts, basis_barcodes, pipeline, facts
        )
        fails, downgrades = _stage2(
            obj, mem, lctx, opts, gse, build, protocol, study_bytes, packs, basis_barcodes
        )
        if not fails and not downgrades:
            break
        if rounds >= opts.max_stage2_rounds:
            raise RuntimeError(
                "sample %s does not converge after %d stage-2 rounds: %r %r"
                % (gsm, rounds, fails, downgrades)
            )
        for rel in downgrades:
            mem[rel].bytes_exact = False
            mem[
                rel
            ].note = (
                "container bytes not reproduced from the object; canonical matrix derivation kept"
            )
        for rel, why in fails.items():
            m = mem[rel]
            if m.treat == "pz2" and not m.force_py and not isinstance(opts.codec, PyCodec2):
                log(
                    "  [%s] %s: stage-2 check failed (%s); re-encoding with the pure-Python codec"
                    % (gsm, rel, why)
                )
                m.force_py, m.body = True, None
                continue
            m.note = "stage-2 %s: %s" % (m.treat, why)
            if (
                m.treat == "derived" and m.csc is not None
            ):  # a derived matrix falls back to a pz2 member, not to raw bytes
                log(
                    "  [%s] %s: stage-2 check failed (%s); coding the matrix directly"
                    % (gsm, rel, why)
                )
                m.treat, m.plan, m.body = "pz2", None, None
            else:
                log("  [%s] %s: stage-2 check failed (%s); storing the member" % (gsm, rel, why))
                m.treat, m.plan, m.body = "stored", None, None
    for m in mem.values():
        m.verified = True
    key = "v2/s/%s/%s.s2" % (gsm, util.d16(obj).hex())
    n_called = int(cells_final.called.sum()) if cells_final is not None else 0
    n_cols = cells_final.n_rows if cells_final is not None else 0
    members = {}
    for info in infos:
        members[info.name] = _manifest_member(info, obj)
    legacy_bytes = sum(mem[r].entry.size for r in order)
    ledger = _ledger(gse, gsm, key, order, mem, infos, obj)
    stats = cells_final.stats() if cells_final is not None else {}
    res = {
        "gsm": gsm,
        "key": key,
        "obj": obj,
        "entry": M.sample_entry(gsm, key, obj, n_cols, n_called, members, stats),
        "ledger": ledger,
        "legacy_bytes": legacy_bytes,
        "legacy_zip_bytes": sum(mem[r].entry.csize for r in order),
        "v2_bytes": len(obj),
        "seconds": time.time() - t0,
        "rounds": rounds,
        "pipeline": pipeline,
        "kinds": _kind_counts(infos),
        "pack_ids": _pack_ids(infos),
    }
    return res


def _add_called_v2(base, barcodes, exon, intron, protocol, opts: Options, gsm: str) -> None:
    """Optional cells columns ``called_v2`` (bool) and ``called_v2_fdr`` (f32, NaN = not tested) plus the call's parameters in
    ``base.meta["called_v2"]`` (docs/called-v2.md)."""
    if opts.cell_call_v2 is False:
        return
    if opts.cell_call_v2 is None and not cellcall.available():
        opts.log("  [%s] called_v2 skipped: numba is not installed" % gsm)
        return
    t0 = time.time()
    X = cellcall.gene_counts(exon, intron, barcodes)
    cv2, fdr, params = cellcall.call(
        X, base.called, protocol
    )  # params go into the object: nothing run-dependent
    base.add("called_v2", cells_mod.DT_BOOL, cv2)
    base.add("called_v2_fdr", cells_mod.DT_F32, fdr)
    base.meta["called_v2"] = params
    opts.log(
        "  [%s] called_v2: %d (+%d), lower %s, alpha %s, %.1fs"
        % (
            gsm,
            params["n_called_v2"],
            params.get("n_added", 0),
            params.get("lower"),
            params.get("alpha"),
            time.time() - t0,
        )
    )


def _pack_ids(infos) -> set:
    out = set()
    for i in infos:
        meta = i.meta or {}
        for part in (meta.get("names") or {}).values():
            if isinstance(part, dict) and "pack" in part:
                out.add(part["pack"])
            if (
                isinstance(part, dict) and "junctions" in part
            ):  # a junction table version (its segments follow below)
                out.add(part["junctions"])
        if meta.get("whitelist"):
            out.add(meta["whitelist"])
    return out


def _kind_counts(infos) -> dict:
    out = {}
    for i in infos:
        out[KIND_LABEL[i.kind]] = out.get(KIND_LABEL[i.kind], 0) + 1
    return out


def _verify_plan_stage1(
    recipes: Recipes, rel: str, data: bytes, plan, lctx: LegacyContext, m: _Member
):
    """Run the recipe against the legacy context. -> (ok, reason, bytes_exact)."""
    rid, params, needs = plan
    try:
        if rel.endswith(".1pz"):
            leg = lctx.csc(rel)
            out = recipes.regenerate_member_matrix(rid, params, lctx)
            if leg is not None:
                got = rmatrix_to_csc(out, leg.value_type)
                why = csc_diff(leg, got)
                if why:
                    return False, "matrix differs (%s)" % why, False
            exact = False
            try:
                b = recipes.regenerate_bytes(rid, params, lctx)
                exact = bytes(b) == data
            except Exception:
                exact = False
            if leg is None and not exact:
                return False, "undecodable legacy member and no byte-exact regeneration", False
            return True, "", exact
        out = recipes.regenerate_bytes(rid, params, lctx)
        if bytes(out) != data:
            return (
                False,
                "regenerated %d bytes differ from the legacy %d bytes" % (len(out), len(data)),
                False,
            )
        return True, "", True
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e), False


# ------------------------------------------------------------------------------------------------ assembling the object
def _assemble(gsm, order, mem, base, cand, lctx, opts: Options, basis_barcodes, pipeline, facts):
    w = C.S2Writer(gsm)
    cells_final = None
    if base is not None:
        needed = set()
        for rel in order:
            m = mem[rel]
            if m.treat == "derived":
                for n in m.plan[2]:
                    if isinstance(n, str) and n.startswith("cells:"):
                        needed.add(n[6:])
        cells_final = base.copy(only=[k for k in cand if k in needed] + list(cellcall.COLUMNS))
        for k in cand:
            if k in needed and not cells_final.has(k):
                cells_final.add(k, cand[k][0], cand[k][1])
        cells_final.meta = dict(base.meta)
        body = cells_final.encode(opts.level)
        w.add_cells(
            "cells",
            body,
            cells_final.digest(),
            {
                "n_rows": cells_final.n_rows,
                "columns": cells_final.names,
                "called_source": base.meta.get("called_source"),
                "whitelist": base.meta.get("whitelist"),
                **({"called_v2": base.meta["called_v2"]} if "called_v2" in base.meta else {}),
                "ctx": facts,
            },
            flags=C.FLAG_ANNDATA,
        )

    def legacy_vt(c):
        return int(c.meta.get("legacy_vt_code", c.value_type))

    for rel in order:
        m = mem[rel]
        e = m.entry
        flags = C.FLAG_LEGACY | (C.FLAG_ANNDATA if rel in ANNDATA_MEMBERS else 0)
        data = None
        if (
            m.treat == "pz2"
            and rel in JUNCTION_MEMBERS
            and opts.junction_table is not None
            and m.csc.rownames
        ):
            extra = _junction_rows(w, rel, m, basis_barcodes, opts, legacy_vt(m.csc), flags)
            if extra is not None:
                continue
        if m.treat == "pz2":
            if m.body is None:
                row_ref, row_meta = _row_names_for(m.csc, rel, lctx, opts)
                col_ref, col_meta = _col_names_for(m.csc, basis_barcodes, opts)
                eo = EncodeOptions(
                    chunk_cols=opts.chunk_cols,
                    chunk_nnz_max=opts.chunk_nnz_max,
                    row_names=row_ref,
                    col_names=col_ref,
                    level=opts.level,
                    threads=opts.threads,
                )
                try:
                    if m.force_py:
                        raise CodecError("pure-Python codec requested after a failed check")
                    m.body = opts.codec.encode(m.csc, eo)
                except CodecError as ex:
                    try:
                        m.body = PyCodec2().encode(
                            m.csc, eo
                        )  # generic codec in pure Python is the universal fallback
                        m.note = "codec 2 (pure Python): %s" % ex
                    except CodecError as ex2:
                        m.treat, m.note = "stored", "codec refused the matrix: %s" % ex2
                        m.body = None
                if m.body is not None:
                    m.meta = {
                        "m": m.csc.m,
                        "n": m.csc.n,
                        "nnz": m.csc.nnz,
                        "legacy_vt_code": legacy_vt(m.csc),
                        "chunk_cols": parse_header(m.body)["chunk_cols"],
                        "names": {"rows": row_meta, "cols": col_meta},
                        "legacy_size": e.size,
                        "legacy_crc32": e.crc32,
                    }
            if m.treat == "pz2":
                w.add_matrix(rel, m.body, m.csc.digest(), m.meta, flags)
                continue
        if data is None:
            data = lctx.member_bytes(rel)
        if m.treat == "derived":
            rid, params, needs = m.plan
            meta = {"recipe": rid, "params": params, "needs": list(needs)}
            if pipeline:
                meta["pipeline"] = pipeline
            if facts:
                meta["ctx"] = facts
            if rel.endswith(".1pz"):
                leg = lctx.csc(rel)
                meta["bytes_exact"] = bool(m.bytes_exact)
                meta["legacy_vt_code"] = (
                    int(leg.meta.get("legacy_vt_code", leg.value_type)) if leg is not None else 3
                )
                dg = leg.digest() if leg is not None else util.d16(data)
            else:
                dg = util.d16(data)
            exact = (not rel.endswith(".1pz")) or m.bytes_exact
            w.add_derived(
                rel, len(data) if exact else 0, util.crc32(data) if exact else 0, dg, meta, flags
            )
        else:
            meta = {"note": m.note} if m.note else {}
            w.add_stored(rel, data, meta, flags, opts.level)
    # auxiliary members named in the ``needs`` of accepted plans ("matrix:<aux>", "blob:<aux>", "vector:<name>")
    aux_needed = []
    for rel in order:
        m = mem[rel]
        if m.treat == "derived":
            for n in m.plan[2]:
                if isinstance(n, str) and ":" in n:
                    tag, nm = n.split(":", 1)
                    if tag == "matrix" and nm in lctx.aux_matrices and nm not in aux_needed:
                        aux_needed.append(nm)
                    elif tag == "blob" and nm in lctx.aux_blobs and nm not in aux_needed:
                        aux_needed.append(nm)
                    elif tag == "vector" and nm in lctx.aux_vectors and n not in aux_needed:
                        aux_needed.append(n)
    if aux_needed:
        from .context import _dump_vector

        for a in aux_needed:
            if a.startswith("vector:"):
                w.add_stored(a, _dump_vector(lctx.aux_vectors[a[7:]]), {"aux": True}, 0, opts.level)
            elif a in lctx.aux_matrices:
                c = lctx.aux_matrices[a]
                row_ref, row_meta = _row_names_for(c, a, lctx, opts)
                col_ref, col_meta = _col_names_for(c, basis_barcodes, opts)
                eo = EncodeOptions(
                    chunk_cols=opts.chunk_cols,
                    chunk_nnz_max=opts.chunk_nnz_max,
                    row_names=row_ref,
                    col_names=col_ref,
                    level=opts.level,
                    threads=opts.threads,
                )
                try:
                    body = opts.codec.encode(c, eo)
                except CodecError:
                    body = PyCodec2().encode(c, eo)
                w.add_matrix(
                    a,
                    body,
                    c.digest(),
                    {
                        "m": c.m,
                        "n": c.n,
                        "nnz": c.nnz,
                        "legacy_vt_code": c.value_type,
                        "aux": True,
                        "chunk_cols": parse_header(body)["chunk_cols"],
                        "names": {"rows": row_meta, "cols": col_meta},
                    },
                    0,
                )
            elif a in lctx.aux_blobs:
                w.add_stored(a, lctx.aux_blobs[a], {"aux": True}, 0, opts.level)
    obj = w.build()
    return obj, w.infos, cells_final


# ------------------------------------------------------------------------------------------------ stage 2
def _stage2(
    obj: bytes,
    mem: dict,
    lctx: LegacyContext,
    opts: Options,
    gse,
    build,
    protocol,
    study_bytes,
    packs,
    base_barcodes,
) -> tuple:
    """Rebuild everything from the finished object alone and compare with the legacy content.

    -> (fails, downgrades): ``fails`` {rel: reason} members that must be stored; ``downgrades`` [rel] derived ``.1pz`` members whose
    container bytes are not reproduced from the object (the canonical-matrix derivation stays, ``bytes_exact`` is dropped)."""
    fails: dict = {}
    downgrades: list = []
    sample = V2Sample(obj, packs)
    ctx = V2Context(sample, opts.recipes, gse, build, study_bytes, protocol)
    for rel, m in mem.items():
        mi = sample.reader.members[rel]
        if m.treat == "pz2":
            try:
                got = sample.matrix(rel)
                why = csc_diff(m.csc, got)
                if why is None and got.meta.get("digest") != mi.digest:
                    why = "header digest != member table digest"
                if why is None and got.digest() != mi.digest:
                    why = "canonical digest differs after decode"
            except Exception as e:
                why = "%s: %s" % (type(e).__name__, e)
            if why:
                fails[rel] = why
            else:
                m.verify_mode = "canonical"
        elif m.treat == "stored":
            try:
                ok = sample.stored_bytes(rel) == lctx.member_bytes(rel)
            except Exception:
                ok = False
            if not ok:
                raise RuntimeError(
                    "stored member %s does not round-trip" % rel
                )  # only a broken zstd can do this
            m.verify_mode = "bytes"
        else:
            try:
                data = lctx.member_bytes(rel)
                if rel.endswith(".1pz"):
                    leg = lctx.csc(rel)
                    out = ctx.regenerate_matrix(rel)
                    if leg is not None:
                        why = csc_diff(leg, rmatrix_to_csc(out, leg.value_type))
                        if why:
                            fails[rel] = "regenerated matrix differs: " + why
                            continue
                    m.verify_mode = "canonical"
                    if m.bytes_exact:
                        try:
                            same = bytes(ctx.regenerate(rel)) == data
                        except Exception:
                            same = False
                        if same:
                            m.verify_mode = "bytes+canonical"
                        else:
                            downgrades.append(rel)
                else:
                    out = ctx.regenerate(rel)
                    if bytes(out) != data:
                        fails[rel] = "regenerated %d bytes differ from the legacy %d bytes" % (
                            len(out),
                            len(data),
                        )
                    else:
                        m.verify_mode = "bytes"
            except Exception as e:
                fails[rel] = "%s: %s" % (type(e).__name__, e)
    t = sample.cells()
    if t is not None and list(t.barcodes) != list(base_barcodes):
        raise RuntimeError("cells table barcodes differ after decode")
    return fails, downgrades


# ------------------------------------------------------------------------------------------------ records
def _manifest_member(info: C.MemberInfo, obj: bytes) -> dict:
    d = {
        "kind": info.kind,
        "flags": info.flags,
        "off": info.off,
        "len": info.len,
        "digest": info.digest.hex(),
    }
    if info.kind in (C.KIND_STORED, C.KIND_DERIVED, C.KIND_RAW):
        d["out_len"], d["crc32"] = info.out_len, info.crc32
    meta = info.meta or {}
    if info.kind == C.KIND_PZ2:
        from .codec import parse_header

        h = parse_header(obj[info.off : info.off + 256])
        ioff, ilen = h["sections"]["chunk_index"]
        d.update(
            {
                "m": h["m"],
                "n": h["n"],
                "nnz": h["nnz"],
                "codec": h["codec"],
                "chunk_cols": h["chunk_cols"],
                "n_chunks": h["n_chunks"],
                "chunk_index": [info.off + ioff, ilen],
                "names": meta.get("names"),
            }
        )
    elif info.kind == C.KIND_DERIVED:
        d.update(
            {"recipe": meta.get("recipe"), "params": meta.get("params"), "needs": meta.get("needs")}
        )
    elif info.kind == C.KIND_CELLS:
        d.update(
            {
                "columns": meta.get("columns"),
                "n_rows": meta.get("n_rows"),
                "whitelist": meta.get("whitelist"),
                "called_source": meta.get("called_source"),
            }
        )
        if meta.get("called_v2") is not None:
            d["called_v2"] = meta["called_v2"]
    return d


def _ledger(gse, gsm, key, order, mem, infos, obj) -> list:
    by = {i.name: i for i in infos}
    lines = []
    for rel in order:
        m = mem[rel]
        i = by[rel]
        e = m.entry
        treat = {
            C.KIND_PZ2: "matrix",
            C.KIND_DERIVED: "recipe",
            C.KIND_STORED: "stored",
            C.KIND_RAW: "raw",
        }[i.kind]
        lines.append(
            {
                "type": "member",
                "gse": gse,
                "gsm": gsm,
                "member": e.path,
                "legacy_size": e.size,
                "legacy_zip_bytes": e.csize,
                "legacy_method": e.method,
                "legacy_crc32": "%08x" % e.crc32,
                "v2_object": key,
                "v2_name": rel,
                "v2_kind": KIND_LABEL[i.kind],
                "treatment": treat,
                "recipe": (i.meta or {}).get("recipe"),
                "v2_bytes": i.len,
                "verified": bool(m.verified),
                "verify_mode": m.verify_mode,
                "note": m.note,
            }
        )
    for i in infos:
        if i.name not in mem:
            lines.append(
                {
                    "type": "v2member",
                    "gse": gse,
                    "gsm": gsm,
                    "v2_object": key,
                    "v2_name": i.name,
                    "v2_kind": KIND_LABEL[i.kind],
                    "v2_bytes": i.len,
                    "legacy": False,
                }
            )
    return lines


# ================================================================================================ study level
def migrate_study_files(
    bundle: LegacyBundle, gse: str, reference_build: str, opts: Options, only: Optional[set] = None
):
    files = {}
    for p in bundle.study_paths():
        files[p] = bundle.read(p)
    sctx = StudyContext(gse, reference_build, files, opts.packs)
    entries, ledger = {}, []
    for name, data in files.items():
        plan = None
        note = ""
        try:
            plan = opts.recipes.plan(name, data, sctx)
        except Exception as e:
            note = "plan error: %s" % e
        if plan is not None:
            try:
                out = opts.recipes.regenerate_bytes(plan[0], plan[1], sctx)
                if bytes(out) != data:
                    note = "recipe %s rejected: bytes differ" % plan[0]
                    plan = None
            except Exception as e:
                note = "recipe %s rejected: %s" % (plan[0], e)
                plan = None
        ent = M.study_file_entry(data, plan)
        # stage 2: from the manifest entry alone
        sc2 = StudyContext(gse, reference_build, {}, opts.packs)
        back = M.study_file_bytes(
            ent, lambda rid, prm: opts.recipes.regenerate_bytes(rid, prm, sc2)
        )
        assert back == data, "study file %s does not round-trip" % name
        entries[name] = ent
        e = bundle.entries[name]
        treat = "recipe" if "recipe" in ent else ("json" if "json" in ent else "stored")
        ledger.append(
            {
                "type": "study_file",
                "gse": gse,
                "member": name,
                "legacy_size": e.size,
                "legacy_zip_bytes": e.csize,
                "legacy_crc32": "%08x" % e.crc32,
                "treatment": treat,
                "recipe": ent.get("recipe"),
                "verified": True,
                "verify_mode": "bytes",
                "note": note,
            }
        )
    return entries, ledger, files


# ================================================================================================ bundle
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$")


def _safe_id(x: str, what: str) -> str:
    """Accession ids end up in file names and in the 16-byte header field: refuse anything else (archives are untrusted input)."""
    if not isinstance(x, str) or not _SAFE_ID.match(x) or ".." in x:
        raise ValueError("unsafe %s %r in the legacy bundle" % (what, x))
    return x


def _gse_of(bundle: LegacyBundle, files: dict, override: Optional[str]) -> str:
    if override:
        return _safe_id(override, "study id")
    mf = _parse_json(files.get("manifest.json"))
    return _safe_id(mf.get("gse_id") or bundle.gse_hint or "GSE0", "study id")


def migrate_bundle(
    src,
    out_dir: str,
    gse: Optional[str] = None,
    opts: Optional[Options] = None,
    samples: Optional[list] = None,
) -> dict:
    opts = opts or Options()
    log = opts.log
    t0 = time.time()
    bundle = LegacyBundle.open(src)
    study_raw = {p: bundle.read(p) for p in bundle.study_paths()}
    gse = _gse_of(bundle, study_raw, gse)
    mf = _parse_json(study_raw.get("manifest.json"))
    reference_build = mf.get("reference_build") or _parse_json(
        study_raw.get("feature_vocab.json")
    ).get("reference_build", "")
    log(
        "migrate %s: %d members, %d samples, build %s"
        % (gse, len(bundle.entries), len(bundle.samples()), reference_build)
    )
    opts.junction_table = _resolve_junction_table(opts, reference_build)
    root = os.path.join(out_dir, "v2")
    for d in ("s", "m", "ledger", "ref"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    sfiles, ledger, _files = migrate_study_files(bundle, gse, reference_build, opts)
    sample_entries = []
    totals = {"legacy_bytes": 0, "legacy_zip_bytes": 0, "v2_bytes": 0, "seconds": 0.0}
    pipeline: dict = {}
    pack_ids: set = set()
    gsms = [_safe_id(g, "sample id") for g in bundle.samples() if samples is None or g in samples]
    for gsm in gsms:
        r = migrate_sample(bundle, gsm, gse, reference_build, study_raw, opts)
        os.makedirs(os.path.join(root, "s", gsm), exist_ok=True)
        with open(os.path.join(out_dir, r["key"]), "wb") as fh:
            fh.write(r["obj"])
        sample_entries.append(r["entry"])
        ledger.extend(r["ledger"])
        ledger.append(
            {
                "type": "sample",
                "gse": gse,
                "gsm": gsm,
                "v2_object": r["key"],
                "legacy_bytes": r["legacy_bytes"],
                "legacy_zip_bytes": r["legacy_zip_bytes"],
                "v2_bytes": r["v2_bytes"],
                "seconds": round(r["seconds"], 2),
                "kinds": r["kinds"],
                "stage2_rounds": r["rounds"],
            }
        )
        for k in ("legacy_bytes", "legacy_zip_bytes", "v2_bytes"):
            totals[k] += r[k]
        totals["seconds"] += r["seconds"]
        pipeline = pipeline or r["pipeline"]
        pack_ids |= r["pack_ids"]
        log(
            "  %s: legacy %d B (zip %d) -> v2 %d B  [%s]  %.1fs"
            % (
                gsm,
                r["legacy_bytes"],
                r["legacy_zip_bytes"],
                r["v2_bytes"],
                ", ".join("%s %d" % kv for kv in sorted(r["kinds"].items())),
                r["seconds"],
            )
        )
    for ent in sfiles.values():
        for pid in [ent.get("params", {}).get("pack")] if "recipe" in ent else []:
            if pid:
                pack_ids.add(pid)
    legacy = {
        "key": "data/%s/%s.singlet" % (gse, gse),
        "etag": bundle.etag,
        "bytes": bundle.size,
        "migrated_at": opts.created_at,
        "ledger": "v2/ledger/%s.jsonl" % gse,
    }
    return {
        "gse": gse,
        "root": root,
        "samples": sample_entries,
        "study_files": sfiles,
        "ledger": ledger,
        "totals": totals,
        "pipeline": pipeline,
        "reference_build": reference_build,
        "junction_table": opts.junction_table.id.hex() if opts.junction_table is not None else None,
        "legacy": legacy,
        "pack_ids": pack_ids,
        "packs": _manifest_packs(sample_entries, opts.packs),
        "gsms": gsms,
        "bundle": bundle,
        "seconds": time.time() - t0,
    }


def _manifest_packs(sample_entries: list, store: P.PackStore) -> dict:
    """{"exon": id, "intron": id, "genes": id, "whitelist": id | [ids] | None} for the manifest, plus "junctions": the junction
    table version the junction rows use, when any sample stores them as table ids."""
    got = {"exon": set(), "intron": set(), "genes": set(), "whitelist": set()}
    jx = set()
    for e in sample_entries:
        for n, mm in e["members"].items():
            nm = mm.get("names") if isinstance(mm.get("names"), dict) else {}
            r = nm.get("rows")
            if isinstance(r, dict) and r.get("member") in got:
                got[r["member"]].add(r["pack"])
            if isinstance(r, dict) and r.get("junctions"):
                jx.add(r["junctions"])
            if n == "cells" and mm.get("whitelist"):
                got["whitelist"].add(mm["whitelist"])
    out = {}
    for k, v in got.items():
        v = sorted(v)
        out[k] = None if not v else (v[0] if len(v) == 1 else v)
    if jx:
        out["junctions"] = sorted(jx)[0] if len(jx) == 1 else sorted(jx)
    return out


def _resolve_junction_table(opts: "Options", build: str):
    """The junction table for one study (Options.junction_spec); None means inline junction names."""
    spec = opts.junction_spec
    if spec is None:
        return None
    if spec == "auto":
        from . import junctions as J

        opts.packs.download = True  # the version and its segments come from the data host (sha256-checked), then the cache
        t = J.published_table(build, opts.packs)
        opts.log(
            "junction table for %s: %s"
            % (
                build,
                "version %s (%s)" % (t.header.get("version"), t.id.hex())
                if t
                else "none published, inline names",
            )
        )
        return t
    if callable(spec):
        return spec(build)
    if getattr(spec, "build", build) != build:
        raise ValueError("the junction table is for %s, the study is %s" % (spec.build, build))
    return spec


def write_outputs(res: dict, opts: Options) -> dict:
    root = res["root"]
    gse = res["gse"]
    store = opts.packs
    ref_ids = set(res["pack_ids"])
    todo = list(ref_ids)
    while (
        todo
    ):  # close over dependencies: intron packs name their exon pack, composites their parts
        pk = store.get(todo.pop())
        deps = []
        if getattr(pk, "header", {}).get("segments"):  # junction table version -> its segments
            deps = [e["id"] for e in pk.header["segments"]]
        elif isinstance(pk, P.VocabPack):
            if pk.kind == P.KIND_COMPOSITE:
                deps = [c.id.hex() for c, _t in pk._components()]
            elif pk.header.get("coder") == "SVPD":
                deps = [pk._exon_source().id.hex()]
        for d in deps:
            if d not in ref_ids:
                ref_ids.add(d)
                todo.append(d)
    entries = []
    for pid in sorted(ref_ids):
        full = store._resolve(pid)
        data = store.read(full)
        with open(os.path.join(root, "ref", full + ".pack"), "wb") as fh:
            fh.write(data)
        kind, hdr, _ = P.parse(data)
        entries.append(
            P.index_entry(
                data,
                {P.KIND_WHITELIST: "whitelist", 4: "junction-segment", 5: "junction-version"}.get(
                    kind, "vocab"
                ),
                {
                    k: v
                    for k, v in {
                        "build": hdr.get("build"),
                        "member": hdr.get("member"),
                        "protocol": hdr.get("protocol"),
                    }.items()
                    if v
                },
            )
        )
    if entries:
        P.write_index(entries, os.path.join(root, "ref", "index.json"))
    man = M.build_manifest(
        gse,
        res["samples"],
        res["study_files"],
        res["reference_build"],
        res["pipeline"],
        res["packs"],
        res["legacy"],
        opts.created_at,
    )
    mpath = os.path.join(root, "m", gse + ".json")
    with open(mpath, "wb") as fh:
        fh.write(M.dumps_manifest(man))
    lpath = os.path.join(root, "ledger", gse + ".jsonl")
    ledger = list(res["ledger"])
    ledger.append(
        {
            "type": "study",
            "gse": gse,
            "rev": man["rev"],
            "samples": len(res["samples"]),
            "legacy_bytes": res["totals"]["legacy_bytes"],
            "legacy_zip_bytes": res["totals"]["legacy_zip_bytes"],
            "v2_bytes": res["totals"]["v2_bytes"],
            "legacy_file_bytes": res["legacy"]["bytes"],
            "seconds": round(res["seconds"], 2),
            "created_at": opts.created_at,
            "manifest": "v2/m/%s.json" % gse,
        }
    )
    with open(lpath, "w", encoding="utf-8", newline="\n") as fh:
        for ln in ledger:
            fh.write(json.dumps(ln, sort_keys=True, ensure_ascii=False) + "\n")
    return {"manifest": mpath, "ledger": lpath, "rev": man["rev"], "ref_packs": len(entries)}


def _junction_arg(v: str):
    """--junction-table: 'none' / 'auto' / a version pack id (found in the pack store) / a .pack path (its directory is added)."""
    if v in (None, "", "none"):
        return None
    if v == "auto":
        return "auto"

    def load(build):
        st = P.default_store()
        st.download = True
        if os.path.exists(v):
            st.add_dir(os.path.dirname(os.path.abspath(v)))
            t = st.get(os.path.basename(v)[:32])
        else:
            t = st.get(v)
        if getattr(t, "build", None) != build:
            raise SystemExit(
                "--junction-table %s is for build %s, the study is %s"
                % (v, getattr(t, "build", "?"), build)
            )
        return t

    return load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m singlet.v2.migrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--bundle", required=True, help="legacy .singlet path, https URL, or an unpacked directory"
    )
    ap.add_argument("--out", required=True, help="output directory (contains v2/...)")
    ap.add_argument("--gse", help="study id (default: from the bundle)")
    ap.add_argument("--samples", help="comma separated GSM ids (default: all)")
    ap.add_argument(
        "--recipes",
        help="recipes module name or 'none' (default: singlet.v2.recipes if importable)",
    )
    ap.add_argument(
        "--chunk-cols",
        type=int,
        default=0,
        help="maximum columns per chunk (0 = pz2 default policy)",
    )
    ap.add_argument(
        "--chunk-nnz-max",
        type=int,
        default=0,
        help="end a chunk before the column that would take it past this many non-zeros (0 = pz2 default, -1 = none)",
    )
    ap.add_argument("--created-at", help="fixed timestamp for reproducible manifests")
    ap.add_argument(
        "--codec",
        choices=["py", "native", "rans", "generic"],
        default=None,
        help="matrix codec: native = compiled pz2 with automatic codec 1/2 choice (default when singlet._pz2 imports), py = pure-Python codec 2",
    )
    ap.add_argument(
        "--junction-table",
        default="none",
        help="junction rows as ids into a shared junction table (docs/format-v2.md 12): 'auto' = the newest version published "
        "for the study's reference build in v2/ref/index.json (downloaded and sha256-checked; inline names when there is none), "
        "a version pack id or .pack path, or 'none' (inline names, the default)",
    )
    ap.add_argument(
        "--no-cell-call-v2",
        action="store_true",
        help="do not compute called_v2 (default: computed; needs numba, pip install 'singlet-bio[migrate]')",
    )
    a = ap.parse_args(argv)
    opts = Options(
        chunk_cols=a.chunk_cols,
        chunk_nnz_max=a.chunk_nnz_max,
        codec=a.codec,
        recipes=a.recipes,
        created_at=a.created_at,
        cell_call_v2=False if a.no_cell_call_v2 else True,
        junction_table=_junction_arg(a.junction_table),
    )
    res = migrate_bundle(a.bundle, a.out, a.gse, opts, a.samples.split(",") if a.samples else None)
    out = write_outputs(res, opts)
    t = res["totals"]
    print(
        "%s: %d samples, legacy %d B (zip) / %d B (members) -> v2 %d B (%.2fx), rev %s, %.1fs"
        % (
            res["gse"],
            len(res["samples"]),
            t["legacy_zip_bytes"],
            t["legacy_bytes"],
            t["v2_bytes"],
            (t["legacy_zip_bytes"] / t["v2_bytes"]) if t["v2_bytes"] else 0.0,
            out["rev"],
            res["seconds"],
        )
    )
    print("manifest:", out["manifest"])
    print("ledger:  ", out["ledger"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
