"""Value-level comparison of a stream run against prod/campaign's sorted run (gate 3 of order-free-rules).

Usage: python -I prod_classify.py <decoded dir A> <decoded dir B> <run A out> <run B out> <out.json>
Decoded dirs hold <name>.mtx/.rows.tsv/.cols.tsv from pz2mtx. Matrices are aligned by feature and barcode
NAME, so row/column order (R6 barcode order, R7 junction order) drops out; what remains are count changes.
For each matrix: axis sizes and overlap, totals, and on shared rows x shared columns the entries that differ,
the absolute count moved as a fraction of the total, and the per-barcode total correlation; then the same on
the barcodes both runs call as cells.
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p + ".mtx"):
        return None
    m = scipy.io.mmread(p + ".mtx").tocsc().astype(np.int64)
    rows = pd.read_csv(p + ".rows.tsv", header=None, sep="\t", dtype=str, keep_default_na=False)[0].values
    cols = pd.read_csv(p + ".cols.tsv", header=None, sep="\t", dtype=str, keep_default_na=False)[0].values
    return m, rows, cols


def called(out):
    cc = pd.read_csv(os.path.join(out, "cell_calls.tsv"), sep="\t", dtype={"barcode": str})
    col = "called_v2" if "called_v2" in cc else ("is_cell" if "is_cell" in cc else None)
    s = cc[cc[col].astype(str).isin(["1", "True", "true"])]["barcode"] if col else cc["barcode"]
    return set(s.str.replace("-1$", "", regex=True))


def sub(m, rows, cols, r_keep, c_keep):
    ri = pd.Index(rows).get_indexer(r_keep)
    ci = pd.Index(cols).get_indexer(c_keep)
    return m[ri][:, ci] if len(ri) and len(ci) else None


def stats(ma, mb):
    d = (mb - ma)
    d.eliminate_zeros()
    ta, tb = int(ma.sum()), int(mb.sum())
    ca, cb = np.asarray(ma.sum(0)).ravel(), np.asarray(mb.sum(0)).ravel()
    ra, rb = np.asarray(ma.sum(1)).ravel(), np.asarray(mb.sum(1)).ravel()
    pc = float(np.corrcoef(ca, cb)[0, 1]) if len(ca) > 2 and ca.std() > 0 and cb.std() > 0 else None
    pr = float(np.corrcoef(np.log1p(ra), np.log1p(rb))[0, 1]) if len(ra) > 2 and ra.std() > 0 else None
    return {"total_a": ta, "total_b": tb, "net_b_minus_a": tb - ta,
            "entries_differ": int(d.nnz), "nnz_a": int(ma.nnz),
            "abs_moved": int(abs(d).sum()), "abs_moved_frac": (abs(d).sum() / max(ta, 1)),
            "per_barcode_total_r": pc, "per_feature_log_total_r": pr,
            "barcodes_any_diff": int((np.diff(d.indptr) > 0).sum()), "barcodes": int(ma.shape[1])}


def compare(a, b, cells):
    ma, ra, ca = a
    mb, rb, cb = b
    rows = pd.Index(ra).intersection(pd.Index(rb))
    cols = pd.Index(ca).intersection(pd.Index(cb))
    out = {"rows_a": len(ra), "rows_b": len(rb), "rows_shared": len(rows),
           "cols_a": len(ca), "cols_b": len(cb), "cols_shared": len(cols),
           "row_order_equal": bool(len(ra) == len(rb) and (ra == rb).all()),
           "total_a_all": int(ma.sum()), "total_b_all": int(mb.sum())}
    out["shared"] = stats(sub(ma, ra, ca, rows, cols), sub(mb, rb, cb, rows, cols))
    bare = pd.Index(cols).str.replace("-1$", "", regex=True)
    cc = cols[bare.isin(cells)]
    if len(cc):
        out["cells_both"] = stats(sub(ma, ra, ca, rows, cc), sub(mb, rb, cb, rows, cc))
    return out


def main():
    da, db, oa, ob, dest = sys.argv[1:6]
    ca, cb = called(oa), called(ob)
    both = ca & cb
    res = {"cells": {"a": len(ca), "b": len(cb), "both": len(both), "a_only": len(ca - cb), "b_only": len(cb - ca)},
           "matrices": {}}
    names = sorted(f[:-4] for f in os.listdir(da) if f.endswith(".mtx"))
    for n in names:
        a, b = load(da, n), load(db, n)
        if a is None or b is None:
            res["matrices"][n] = {"missing_in": "b" if b is None else "a"}
            continue
        try:
            res["matrices"][n] = compare(a, b, both)
        except Exception as e:  # keep going: one odd matrix must not hide the rest
            res["matrices"][n] = {"error": repr(e)}
        print(n, json.dumps(res["matrices"][n].get("cells_both", res["matrices"][n].get("shared", {})))[:300], flush=True)
    with open(dest, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
