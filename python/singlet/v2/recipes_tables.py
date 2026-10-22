# SPDX-License-Identifier: MIT
"""tsvcols-v1: per-cell TSV members regenerated from cells-table columns (owner: W3).

One generic recipe covers every per-cell table the pipeline writes (cell_qc_metrics, doublet_scores, cell_cycle_scores,
ambient_contamination, auto_barcodes, donor_assignments, mt_summary, spatial_coordinates, visium_qc, ...).

Recipe params (JSON, short keys; example for doublet_scores.tsv)::

    {"h": ["barcode","total_umis","doublet_score","is_doublet"],       header line, null = no header
     "r": {"m": "called", "o": "sorted"},                              rows: all | called | col, order sorted | colorder | cc
     "c": [{"n":"total_umis","s":"d","d":"n_umi_exon"},                derived from data the container holds
           {"n":"doublet_score","s":"c","c":"dbl_doublet_score","f":{"k":"float","dt":"f32","fmt":"%.6g"}},
           {"n":"is_doublet","s":"c","c":"dbl_is_doublet","f":{"k":"bool","labels":["False","True"]}}]}

Column sources: ``c`` a cells-table column (typed, ``cells_columns`` returns them), ``d`` an exact derivation
(``DERIVE`` below), ``k`` a constant string.  The first column is always the barcode (derived from the rows).

Rows: ``all`` every matrix column in column order; ``called`` the called cells; ``col`` an explicit row bitmap
(``<stem>_rows``).  Orders: ``sorted`` bytewise by barcode, ``colorder`` ascending matrix column, ``cc`` the row order
of cell_calls.tsv (STAR sorted, EmptyDrops column order, or the explicit ``cc_pos`` of top_n_fallback).
"""
from __future__ import annotations

import json
import zlib

import numpy as np

from . import recipes_cells as rcc
from . import recipes_qc as rqc
from . import recipes_tsv as rt

# -- profiles: column-name prefixes of the cells-table columns and which columns have exact derivations ---------------
PROFILES = {
    "cell_qc_metrics.tsv": {"prefix": "qc_", "header": True,
                            "derive": {"n_umi": ["qc.n_umi"], "n_genes": ["qc.n_genes"], "pct_mt": ["qc.pct_mt"],
                                       "pct_ribo": ["qc.pct_ribo"], "intronic_pct": ["qc.intronic_pct"],
                                       "duplication_rate": ["dup_rate"]},
                            "alias": {"doublet_score": ["dbl_doublet_score"], "cell_cycle_phase": ["cyc_phase"]}},
    "doublet_scores.tsv": {"prefix": "dbl_", "header": True, "derive": {"total_umis": ["n_umi_exon"]}},
    "cell_cycle_scores.tsv": {"prefix": "cyc_", "header": True, "derive": {}},
    "ambient_contamination.tsv": {"prefix": "amb_", "header": True, "derive": {"rho": ["amb.rho"]}},
    "auto_barcodes.tsv": {"prefix": "ab_", "header": False, "derive": {}},
    "donor/donor_assignments.tsv": {"prefix": "don_", "header": True, "derive": {}},
    "mt/mt_summary.tsv": {"prefix": "mts_", "header": True, "derive": {}},
    "spatial_coordinates.tsv": {"prefix": "sp_", "header": True, "derive": {}},
    "visium_qc.tsv": {"prefix": "vq_", "header": True, "derive": {}},
    "nonhost/nonhost_per_cell.tsv": {"prefix": "nh_", "header": True, "derive": {}},
}
# the cells-table column of the owner profile that other profiles may alias
STEM = {m: m.replace("/", "_").rsplit(".", 1)[0] for m in PROFILES}


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------
def dtype_name(a):
    a = np.asarray(a)
    if a.dtype == object:
        return "str"
    return {"uint8": "u8", "uint16": "u16", "uint32": "u32", "int32": "i32", "float32": "f32", "float64": "f64", "bool": "bool"}[a.dtype.name]


def _cc_info(ctx, cells):
    """(params, cols) of cellcalls_analyse for the sample, cached on ctx."""
    c = getattr(ctx, "_w3_cc", None)
    if c is None:
        lb = ctx.member_bytes("cell_calls.tsv")
        c = rcc.cellcalls_analyse(lb, list(cells["barcode"])) if lb is not None else False
        try:
            ctx._w3_cc = c
        except Exception:
            pass
    return c or None


def _row_orders(ctx, cells, rows_idx):
    """Candidate orders for a row subset: name -> ordered row index array."""
    barcode = cells["barcode"]
    out = {"colorder": rows_idx}
    bs = np.array([str(barcode[i]).encode("ascii", "replace") for i in rows_idx], dtype=object)
    out["sorted"] = rows_idx[np.argsort(bs, kind="stable")] if len(rows_idx) else rows_idx
    info = _cc_info(ctx, cells)
    if info and info[0].get("order") == "explicit":
        pos = info[1]["cc_pos"][rows_idx]
        if len(pos) and (pos > 0).all():
            out["cc"] = rows_idx[np.argsort(pos, kind="stable")]
    return out


def _select_rows(params_r, cells, ctx):
    """Row index array for params 'r'."""
    m = params_r["m"]
    n = len(cells["barcode"])
    if m == "all":
        return np.arange(n)
    if m == "called":
        base = np.flatnonzero(np.asarray(cells["called"], dtype=bool))
    elif m == "col":
        base = np.flatnonzero(np.asarray(cells[params_r["c"]], dtype=bool))
    else:
        raise ValueError(params_r)
    o = params_r.get("o", "colorder")
    if o == "colorder":
        return base
    if o == "sorted":
        bs = np.array([str(cells["barcode"][i]).encode("ascii") for i in base], dtype=object)
        return base[np.argsort(bs, kind="stable")] if len(base) else base
    if o == "cc":
        pos = np.asarray(cells["cc_pos"])[base]
        return base[np.argsort(pos, kind="stable")]
    raise ValueError(params_r)


# -- derivations ------------------------------------------------------------------------------------
def _d_barcode(ctx, cells, rows):
    return [str(cells["barcode"][i]) for i in rows]


def _d_n_umi_exon(ctx, cells, rows):
    return [str(int(x)) for x in np.asarray(cells["n_umi_exon"])[rows]]


def _qc_col(name, fmt=None):
    def f(ctx, cells, rows):
        q = rqc.get_qc(ctx)
        if q is None:
            raise LookupError("no matrices")
        a = q[name][rows]
        if fmt is None:
            return [str(int(x)) for x in a]
        return [fmt % x for x in a.astype(np.float64)]

    return f


def _d_dup_rate(ctx, cells, rows):
    """read_stats.h: dup = total > unique ? total - unique : 0 (uint32); rate = float32(dup) / float32(total), 0 if total == 0."""
    q = rqc.get_qc(ctx)
    total = np.asarray(cells["qc_n_reads"]).astype(np.uint32)[rows]
    unique = np.minimum(q["n_umi"][rows], np.uint64(0xFFFFFFFF)).astype(np.uint32)
    dup = np.where(total > unique, total - unique, np.uint32(0)).astype(np.float32)
    tf = total.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(total > 0, dup / np.where(total > 0, tf, np.float32(1)), np.float32(0)).astype(np.float32)
    return ["%.4f" % x for x in rate.astype(np.float64)]


def _d_amb_rho(ctx, cells, rows):
    """ambient_correction.h: rho = clip(S / N_c, 0.001, 0.5), S = ambient total UMI / ambient barcodes (the two integers
    are in ambient_contamination.json), N_c = exon UMI total of the cell (rho_max when N_c == 0); printed with %.6g."""
    jb = ctx.member_bytes("ambient_contamination.json")
    if jb is None:
        raise LookupError("ambient_contamination.json")
    j = json.loads(jb)
    nb, tot = j["ambient_profile_n_barcodes"], j["ambient_profile_total_umis"]
    if not nb or tot < 1:
        raise LookupError("degenerate ambient pool")
    S = float(tot) / float(nb)
    nc = np.asarray(cells["n_umi_exon"]).astype(np.float64)[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(nc > 0, np.maximum(0.001, np.minimum(S / np.where(nc > 0, nc, 1.0), 0.5)), 0.5)
    return ["%.6g" % x for x in rho]


DERIVE = {
    "barcode": (_d_barcode, ["cells:barcode"]),
    "n_umi_exon": (_d_n_umi_exon, ["cells:n_umi_exon"]),
    "qc.n_umi": (_qc_col("n_umi"), ["matrix:exon_counts.1pz", "matrix:intron_counts.1pz"]),
    "qc.n_genes": (_qc_col("n_genes"), ["matrix:exon_counts.1pz", "matrix:intron_counts.1pz"]),
    "qc.pct_mt": (_qc_col("pct_mt", "%.4f"), ["matrix:exon_counts.1pz", "matrix:intron_counts.1pz"]),
    "qc.pct_ribo": (_qc_col("pct_ribo", "%.4f"), ["matrix:exon_counts.1pz", "matrix:intron_counts.1pz"]),
    "qc.intronic_pct": (_qc_col("intronic_pct", "%.4f"), ["matrix:exon_counts.1pz", "matrix:intron_counts.1pz"]),
    "amb.rho": (_d_amb_rho, ["member:ambient_contamination.json", "cells:n_umi_exon"]),
    "dup_rate": (_d_dup_rate, ["cells:qc_n_reads", "matrix:exon_counts.1pz", "matrix:intron_counts.1pz"]),
}


# ---------------------------------------------------------------------------------------------------
# analysis (legacy bytes -> params + typed columns)
# ---------------------------------------------------------------------------------------------------
def _scatter(fit, arr, rows, n):
    k = fit["k"]
    if k == "float":
        full = np.full(n, np.nan, dtype=arr.dtype)
    elif k == "str":
        full = np.full(n, "", dtype=object)
    else:
        full = np.zeros(n, dtype=arr.dtype)
        if k == "int" and fit.get("na") is not None:
            full[:] = arr.dtype.type(fit["na_code"])
    full[rows] = arr
    return full


def _owner_column(ctx, cells, col):
    """(array, fit) of a canonical column owned by another profile (analysed from its legacy member)."""
    for member, prof in PROFILES.items():
        if col.startswith(prof["prefix"]):
            lb = ctx.member_bytes(member)
            if lb is None:
                return None
            res = analyse(member, lb, ctx, cells, _alias_depth=1)
            if res is None or col not in res[1]:
                return None
            return res[1][col][1], res[2].get(col)
    return None


def analyse(member, legacy_bytes, ctx, cells=None, _alias_depth=0):
    """Cached ``_analyse`` (cells_columns and plan both need it; the key includes the member bytes)."""
    cache = getattr(ctx, "_w3_an", None)
    if cache is None:
        cache = {}
        try:
            ctx._w3_an = cache
        except Exception:
            pass
    key = (member, len(legacy_bytes), zlib.crc32(legacy_bytes), _alias_depth)
    if key not in cache:
        cache[key] = _analyse(member, legacy_bytes, ctx, cells, _alias_depth)
    return cache[key]


def _analyse(member, legacy_bytes, ctx, cells=None, _alias_depth=0):
    """-> (params, columns{name: (dtype, array)}, fits{name: fit}) or None.

    ``columns`` are the cells-table columns to store (beyond the required ones)."""
    prof = PROFILES.get(member)
    if prof is None:
        return None
    cells = cells if cells is not None else rcc.cells_view(ctx)
    if "barcode" not in cells:
        return None
    parsed = rt.parse_tsv(legacy_bytes, has_header=prof["header"])
    if parsed is None:
        return None
    header, tcols = parsed
    n = len(cells["barcode"])
    fb = tcols[0]
    ix = {str(b): i for i, b in enumerate(cells["barcode"])}
    if len(ix) != n:
        return None
    try:
        pos = np.array([ix[b] for b in fb], dtype=np.int64)
    except KeyError:
        return None
    if len(set(pos.tolist())) != len(pos):
        return None
    stem = STEM[member]
    columns, fits = {}, {}
    # --- rows
    called = np.asarray(cells.get("called", np.zeros(n, bool)), dtype=bool)
    r = None
    sel_sorted = np.sort(pos)
    if len(pos) == n and np.array_equal(pos, np.arange(n)):
        r = {"m": "all"}
    else:
        called_idx = np.flatnonzero(called)
        if np.array_equal(sel_sorted, called_idx):
            m, base = "called", called_idx
        else:
            m, base = "col", sel_sorted
        orders = _row_orders(ctx, cells, base)
        for o in ("sorted", "colorder", "cc"):
            if o in orders and np.array_equal(orders[o], pos):
                r = {"m": m, "o": o}
                break
        if r is None:
            return None
        if m == "col":
            bm = np.zeros(n, dtype=bool)
            bm[pos] = True
            columns[stem + "_rows"] = ("bool", bm)
            r["c"] = stem + "_rows"
    rows = pos
    # --- columns
    spec = []
    names = header if header is not None else ["barcode"]
    view = dict(cells)
    for k, (dt, a) in columns.items():
        view[k] = a
    for j in range(1, len(tcols)):
        nm = names[j]
        text = tcols[j]
        entry = None
        # constant
        if len(set(text)) == 1 and len(text) > 0:
            entry = {"n": nm, "s": "k", "v": text[0]}
        # exact derivation
        if entry is None:
            for fn in prof["derive"].get(nm, []):
                try:
                    if fn == "dup_rate":
                        # needs n_reads: use the file's own column (stored below) via the view
                        continue
                    if DERIVE[fn][0](ctx, view, rows) == text:
                        entry = {"n": nm, "s": "d", "d": fn}
                        break
                except Exception:
                    pass
        # alias of a canonical column owned by another profile
        if entry is None and _alias_depth == 0:
            for col in prof.get("alias", {}).get(nm, []):
                got = _owner_column(ctx, cells, col)
                if got is None:
                    continue
                arr, ofit = got
                fit_q, arr_q = rt.fit_column(text)
                a = np.asarray(arr)[rows]
                ok = False
                fit_use = None
                if fit_q["k"] == "float" and a.dtype.kind == "f":
                    # the other file may print a rounded view of the same value (%.4f of a %.6g score)
                    fit_use = dict(fit_q, dt=dtype_name(a))
                    ok = rt.render_column(fit_use, a) == text
                elif ofit is not None and ofit["k"] in ("cat", "bool", "int", "str") and fit_q["k"] == ofit["k"]:
                    fit_use = ofit
                    ok = rt.render_column(ofit, a) == text
                if ok:
                    entry = {"n": nm, "s": "c", "c": col, "f": fit_use}
                    break
        # own stored column
        if entry is None:
            fit, arr = rt.fit_column(text)
            cname = prof["prefix"] + nm
            full = arr if r["m"] == "all" else _scatter(fit, arr, rows, n)
            columns[cname] = (dtype_name(full), full)
            fits[cname] = fit
            view[cname] = full
            entry = {"n": nm, "s": "c", "c": cname, "f": fit}
        spec.append(entry)
    # duplication_rate needs qc_n_reads which exists only after the n_reads column was stored: second pass
    for i, e in enumerate(spec):
        nm = e["n"]
        if e["s"] != "d" and "dup_rate" in prof["derive"].get(nm, []) and "qc_n_reads" in view:
            if DERIVE["dup_rate"][0](ctx, view, rows) == tcols[i + 1]:
                # the stored column is no longer needed
                cname = e.get("c")
                if cname in columns and cname != "qc_n_reads":
                    columns.pop(cname)
                    fits.pop(cname, None)
                spec[i] = {"n": nm, "s": "d", "d": "dup_rate"}
    params = {"h": header, "r": r, "c": spec}
    return params, columns, fits


def render(params, ctx, cells) -> bytes:
    rows = _select_rows(params["r"], cells, ctx)
    text = [_d_barcode(ctx, cells, rows)]
    for e in params["c"]:
        s = e["s"]
        if s == "k":
            text.append([e["v"]] * len(rows))
        elif s == "d":
            text.append(DERIVE[e["d"]][0](ctx, cells, rows))
        else:
            text.append(rt.render_column(e["f"], np.asarray(cells[e["c"]])[rows]))
    return rt.render_table(params["h"], text)


def needs_of(params):
    needs = ["cells:barcode"]
    r = params["r"]
    if r["m"] in ("called",):
        needs.append("cells:called")
    if r.get("o") == "cc":
        needs.append("cells:cc_pos")
    if r["m"] == "col":
        needs.append("cells:" + r["c"])
    for e in params["c"]:
        if e["s"] == "c":
            needs.append("cells:" + e["c"])
        elif e["s"] == "d":
            needs.extend(DERIVE[e["d"]][1])
    seen, out = set(), []
    for x in needs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def plan_tsv(member, legacy_bytes, ctx):
    cells = rcc.cells_view(ctx)
    res = analyse(member, legacy_bytes, ctx, cells)
    if res is None:
        return None
    params, columns, _ = res
    view = dict(cells)
    for k, (dt, a) in columns.items():
        view[k] = a
    # shared columns referenced from other profiles
    for e in params["c"]:
        if e["s"] == "c" and e["c"] not in view:
            got = _owner_column(ctx, cells, e["c"])
            if got is None:
                return None
            view[e["c"]] = got[0]
    # cc_pos for order 'cc'
    if params["r"].get("o") == "cc":
        info = _cc_info(ctx, cells)
        if info and "cc_pos" in info[1]:
            view["cc_pos"] = info[1]["cc_pos"]
    try:
        if render(params, ctx, view) != legacy_bytes:
            return None
    except Exception:
        return None
    return ("tsvcols-v1", params, needs_of(params))


def regenerate_tsv(recipe_id, params, ctx) -> bytes:
    return render(params, ctx, rcc.cells_view(ctx))


def cells_columns_tsv(member, legacy_bytes, ctx):
    res = analyse(member, legacy_bytes, ctx)
    if res is None:
        return {}
    cols = dict(res[1])
    out = {k: (v[0], v[1]) for k, v in cols.items()}
    if res[0]["r"].get("o") == "cc":
        info = _cc_info(ctx, rcc.cells_view(ctx))
        if info and "cc_pos" in info[1]:
            out["cc_pos"] = ("u32", info[1]["cc_pos"])
    return out
