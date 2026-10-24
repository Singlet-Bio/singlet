# SPDX-License-Identifier: MIT
"""cellcalls-v1: ``cell_calls.tsv`` from the sample cells table (owner: W3).

``cell_calls.tsv`` (include/singlet/pileup/cell_calling.h:write_cell_calls) is
``barcode, is_cell, log10_umi, empty_drops_pvalue, call_method`` with one row per *tested* barcode in the order of
``result.tested_indices``.  What each pipeline path writes (all verified on the corpora, see notes/m_w3.md):

  star_solo        rows = STARsolo's called cells, sorted bytewise (STAR's filtered/barcodes.tsv); is_cell TRUE;
                   p-value column is the sentinel 0.0000e+00
  top_n_fallback   rows = top-N exon-UMI barcodes in the order of an unstable std::sort (ties!), TRUE, sentinel p
  emptydrops       rows = every barcode EmptyDrops tested, ascending column order, is_cell TRUE/FALSE, BH FDR printed %.4e
  (cr2_fallback, knee_fallback, forced_cells use the same writer; they fall under one of the three row patterns)

``log10_umi`` is ``%.4f`` of log10 of the exon column sum (0.0000 for an empty column), the header and the method
string are constants, so the only independent information is *which* barcodes are rows/called (the cells table's
``called`` bitmap plus, for emptydrops, ``cc_listed``), the p-values (``cc_pvalue``), and for top_n_fallback the
row order inside UMI ties (``cc_pos``).  The recipe asks the cells table for exactly the columns it needs.
"""

from __future__ import annotations

import numpy as np

CC_HEADER = "barcode\tis_cell\tlog10_umi\tempty_drops_pvalue\tcall_method\n"
SENTINEL_P = "0.0000e+00"


# ---------------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------------
def parse_cell_calls(b: bytes):
    """-> (barcodes, is_cell[bool], log10[str], pvalue[str], method[str]) lists, or None if not in the writer's format."""
    try:
        s = b.decode("ascii")
    except UnicodeDecodeError:
        return None
    if not s.startswith(CC_HEADER) or (len(s) > len(CC_HEADER) and not s.endswith("\n")):
        return None
    rows = s[len(CC_HEADER) :].split("\n")
    if rows and rows[-1] == "":
        rows.pop()
    bc, ic, lg, pv, me = [], [], [], [], []
    for r in rows:
        f = r.split("\t")
        if len(f) != 5 or f[1] not in ("TRUE", "FALSE"):
            return None
        bc.append(f[0])
        ic.append(f[1] == "TRUE")
        lg.append(f[2])
        pv.append(f[3])
        me.append(f[4])
    return bc, np.array(ic, dtype=bool), lg, pv, me


def _barcode_index(barcodes):
    return {b: i for i, b in enumerate(barcodes)}


def called_flags(cell_calls_bytes: bytes, colnames) -> np.ndarray:
    """The ``called`` bitmap of the cells table: column i is called when its barcode is a TRUE row of cell_calls.tsv.

    Barcodes of the file that are not matrix columns are ignored here (``plan`` refuses such members)."""
    n = len(colnames)
    out = np.zeros(n, dtype=bool)
    p = parse_cell_calls(cell_calls_bytes)
    if p is None:
        return out
    ix = _barcode_index(colnames)
    for b, c in zip(p[0], p[1]):
        if c and b in ix:
            out[ix[b]] = True
    return out


def _fmt_log10(totals: np.ndarray):
    t = np.asarray(totals, dtype=np.float64)
    v = np.where(t > 0, np.log10(np.where(t > 0, t, 1.0)), 0.0)
    return ["%.4f" % x for x in v]


# ---------------------------------------------------------------------------------------------------
# cells view
# ---------------------------------------------------------------------------------------------------
def cells_view(ctx, extra=None):
    """dict of cells-table columns; barcodes and exon column sums come from the matrix when the table lacks them."""
    cells = dict(ctx.cells() or {})
    if "barcode" not in cells or "n_umi_exon" not in cells:
        ex = ctx.matrix("exon_counts.1pz")
        if ex is not None:
            cells.setdefault("barcode", np.array(ex.colnames, dtype=object))
            cells.setdefault("n_umi_exon", ex.col_sums())
    if extra:
        cells.update(extra)
    return cells


# ---------------------------------------------------------------------------------------------------
# cells columns
# ---------------------------------------------------------------------------------------------------
def cellcalls_analyse(legacy_bytes: bytes, barcodes):
    """Classify a legacy cell_calls.tsv against the matrix column barcodes.

    Returns (params, columns) or None.  ``columns`` are cells-table columns beyond the required ones."""
    p = parse_cell_calls(legacy_bytes)
    if p is None:
        return None
    bc, is_cell, lg, pv, me = p
    ix = _barcode_index(barcodes)
    if len(ix) != len(barcodes):
        return None  # duplicate barcodes in the matrix: row identity would be ambiguous
    try:
        pos = np.array([ix[b] for b in bc], dtype=np.int64)
    except KeyError:
        return None
    if len(set(pos.tolist())) != len(pos):
        return None
    n = len(barcodes)
    methods = sorted(set(me))
    if len(methods) > 1:
        return None
    params = {"method": methods[0] if methods else ""}
    called = np.zeros(n, dtype=bool)
    called[pos[is_cell]] = True
    cols = {"called": called}
    listed = np.zeros(n, dtype=bool)
    listed[pos] = True
    if not np.array_equal(listed, called):
        cols["cc_listed"] = listed
        params["listed"] = "col"
    else:
        params["listed"] = "called"
    # row order
    if len(pos) <= 1 or all(bc[i] <= bc[i + 1] for i in range(len(bc) - 1)):
        params["order"] = "sorted"
    elif bool((np.diff(pos) > 0).all()):
        params["order"] = "colorder"
    else:
        params["order"] = "explicit"
        cp = np.zeros(n, dtype=np.uint32)
        cp[pos] = np.arange(1, len(pos) + 1, dtype=np.uint32)
        cols["cc_pos"] = cp
    # p-values
    if all(x == SENTINEL_P for x in pv):
        params["pvalue"] = "zero"
    else:
        params["pvalue"] = "col"
        arr = np.zeros(n, dtype=np.float32)
        arr[pos] = np.array([float(x) for x in pv], dtype=np.float32)
        cols["cc_pvalue"] = arr
    return params, cols


def cellcalls_needs(params):
    needs = ["cells:barcode", "cells:called", "cells:n_umi_exon"]
    if params.get("listed") == "col":
        needs.append("cells:cc_listed")
    if params.get("order") == "explicit":
        needs.append("cells:cc_pos")
    if params.get("pvalue") == "col":
        needs.append("cells:cc_pvalue")
    return needs


def render_cell_calls(params, cells) -> bytes:
    barcode = cells["barcode"]
    called = np.asarray(cells["called"], dtype=bool)
    listed = np.asarray(cells["cc_listed"], dtype=bool) if params.get("listed") == "col" else called
    rows = np.flatnonzero(listed)
    order = params.get("order", "sorted")
    if order == "sorted":
        rows = (
            rows[
                np.argsort(
                    np.array([str(barcode[i]).encode("ascii") for i in rows], dtype=object),
                    kind="stable",
                )
            ]
            if len(rows)
            else rows
        )
    elif order == "explicit":
        rows = rows[np.argsort(np.asarray(cells["cc_pos"])[rows], kind="stable")]
    # colorder: rows already ascending
    lg = _fmt_log10(np.asarray(cells["n_umi_exon"])[rows])
    if params.get("pvalue") == "col":
        pvs = ["%.4e" % float(x) for x in np.asarray(cells["cc_pvalue"])[rows]]
    else:
        pvs = [SENTINEL_P] * len(rows)
    method = params.get("method", "")
    out = [CC_HEADER]
    isc = called[rows]
    out.extend(
        "%s\t%s\t%s\t%s\t%s\n" % (barcode[i], "TRUE" if c else "FALSE", lgv, pv, method)
        for i, c, lgv, pv in zip(rows.tolist(), isc.tolist(), lg, pvs)
    )
    return "".join(out).encode("ascii")


def plan_cell_calls(member, legacy_bytes, ctx):
    cells = cells_view(ctx)
    if "barcode" not in cells:
        return None
    res = cellcalls_analyse(legacy_bytes, list(cells["barcode"]))
    if res is None:
        return None
    params, cols = res
    view = dict(cells)
    view.update(cols)
    if render_cell_calls(params, view) != legacy_bytes:
        return None
    return ("cellcalls-v1", params, cellcalls_needs(params))


def regenerate_cell_calls(recipe_id, params, ctx) -> bytes:
    return render_cell_calls(params, cells_view(ctx))


def _dt_name(a):
    return {"uint32": "u32", "float32": "f32", "bool": "bool"}[a.dtype.name]


def cells_columns_cell_calls(member, legacy_bytes, ctx):
    cells = cells_view(ctx)
    res = cellcalls_analyse(legacy_bytes, list(cells["barcode"])) if "barcode" in cells else None
    if res is None:
        return {}
    return {k: (_dt_name(v), v) for k, v in res[1].items()}
