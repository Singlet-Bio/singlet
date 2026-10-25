# SPDX-License-Identifier: MIT
"""Does the pipeline's called_v2 (C++, include/singlet/pileup/called_v2.h) equal python/singlet/v2/cellcall.py?

    python -P bench/called_v2/equiv_cpp.py RUN_DIR [RUN_DIR ...] [--json OUT]

RUN_DIR is a `singlet` output directory written with --output-format mtx (exon_counts / intron_counts .mtx.gz with
_features/_barcodes .tsv.gz) and the per-barcode cell_calls.tsv of the called_v2 caller. For each run the stored exon and
intron matrices go through cellcall.gene_counts and cellcall.call with called = the is_cell column (the knee set), exactly
as the v2 migration does, and the result is compared with the called_v2 / fdr columns the binary wrote.

Acceptance (docs/called-v2.md, C++ port): identical called_v2 sets; otherwise >= 99.9% agreement with every differing
barcode at an FDR within 0.002 of the 0.001 threshold. Exit status 1 if a run fails.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))
if os.environ.get("CELLCALL_REF"):  # another reference file, e.g. bench/called_v2_truth/cellcall_ref.py (VERSION 2.1)
    import importlib.util

    _spec = importlib.util.spec_from_file_location("cellcall", os.environ["CELLCALL_REF"])
    cellcall = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cellcall)
else:
    from singlet.v2 import cellcall  # noqa: E402


class Mat:
    """The attributes of matrix.Csc that cellcall.gene_counts reads."""

    def __init__(self, X: sp.csc_matrix, rownames, colnames):
        X = X.tocsc()
        X.sort_indices()
        self.m, self.n = X.shape
        self.data, self.indices, self.indptr = X.data, X.indices, X.indptr
        self.rownames, self.colnames = list(rownames), list(colnames)


def lines(p):
    with gzip.open(p, "rt") as f:
        return [x.rstrip("\n").split("\t")[0] for x in f]


def load(run, m):
    p = os.path.join(run, m + ".mtx.gz")
    if not os.path.exists(p):
        return None
    X = sp.csc_matrix(scipy.io.mmread(p)).astype(np.int64)
    return Mat(X, lines(os.path.join(run, m + "_features.tsv.gz")), lines(os.path.join(run, m + "_barcodes.tsv.gz")))


def read_calls(run):
    with open(os.path.join(run, "cell_calls.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    col = {h: i for i, h in enumerate(head)}
    need = ("barcode", "is_cell", "called_v2", "fdr")
    if not all(k in col for k in need):
        raise SystemExit("%s: cell_calls.tsv is not the called_v2 format (%s)" % (run, head))
    bc = [r[col["barcode"]] for r in rows]
    is_cell = np.array([r[col["is_cell"]] == "TRUE" for r in rows])
    cv2 = np.array([r[col["called_v2"]] == "1" for r in rows])
    fdr = np.array([np.nan if r[col["fdr"]] == "NA" else float(r[col["fdr"]]) for r in rows])
    return bc, is_cell, cv2, fdr


def stats_cell_call(run):
    p = os.path.join(run, "pileup_stats.json")
    try:
        with open(p) as f:
            return json.load(f).get("provenance", {}).get("cell_call", {})
    except (OSError, ValueError):
        return {}


def compare(run):
    ex, it = load(run, "exon_counts"), load(run, "intron_counts")
    bc, is_cell, cv2_cpp, fdr_cpp = read_calls(run)
    if ex.colnames != bc:
        raise SystemExit("%s: cell_calls.tsv barcodes differ from the exon matrix columns" % run)
    X = cellcall.gene_counts(ex, it, bc)
    cv2_py, fdr_py, params = cellcall.call(X, is_cell)
    n = len(bc)
    differ = np.nonzero(cv2_py != cv2_cpp)[0]
    both = ~np.isnan(fdr_py) & ~np.isnan(fdr_cpp)
    # the binary prints FDR with 6 significant digits
    fdr_close = np.isclose(fdr_py[both], fdr_cpp[both], rtol=2e-6, atol=0)
    near = all(abs(float(fdr_py[i]) - 0.001) <= 0.002 for i in differ if not np.isnan(fdr_py[i]))
    agree = 1 - len(differ) / max(n, 1)
    cpp = stats_cell_call(run).get("called_v2", {})
    res = {
        "run": run,
        "barcodes": n,
        "called": int(is_cell.sum()),
        "called_v2_python": int(cv2_py.sum()),
        "called_v2_cpp": int(cv2_cpp.sum()),
        "identical": bool(len(differ) == 0),
        "n_differ": int(len(differ)),
        "agreement": agree,
        "differ_near_threshold": bool(near),
        "tested_python": int((~np.isnan(fdr_py)).sum()),
        "tested_cpp": int((~np.isnan(fdr_cpp)).sum()),
        "fdr_equal_share": float(fdr_close.mean()) if both.any() else 1.0,
        "fdr_max_abs_diff": float(np.max(np.abs(fdr_py[both] - fdr_cpp[both]))) if both.any() else 0.0,
        "lower": [params.get("lower"), cpp.get("lower")],
        "retain_total": [params.get("retain_total"), cpp.get("retain_total")],
        "alpha": [params.get("alpha"), cpp.get("alpha")],
    }
    res["pass"] = res["identical"] or (agree >= 0.999 and near)
    return res


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        del args[i : i + 2]
    results = [compare(r) for r in args]
    for r in results:
        print(json.dumps(r))
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
