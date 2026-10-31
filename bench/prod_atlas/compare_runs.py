# SPDX-License-Identifier: MIT
"""Are two singlet runs' matrices identical? (the check of bench/atlas_rerun/perf_equiv.py, for two directories)

    python -P compare_runs.py REF_RUN NEW_RUN [--json OUT]

For each member (exon, intron, ambiguous, splice junctions, V(D)J usage) the matrices are aligned on feature name and
barcode; identical = same barcode and feature sets and 0 differing entries. Also prints the strand fields of
pileup_stats.json and the wall time of both runs.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

MEMBERS = ["exon_counts", "intron_counts", "ambiguous_counts", "sj_counts", "vdj_gene_usage"]


def lines(p):
    with gzip.open(p, "rt") as f:
        return [x.rstrip("\n") for x in f]


def load(run, m):
    p = os.path.join(run, m + ".mtx.gz")
    if not os.path.exists(p):
        return None
    x = sp.csr_matrix(scipy.io.mmread(p))
    return x, lines(os.path.join(run, m + "_features.tsv.gz")), lines(os.path.join(run, m + "_barcodes.tsv.gz"))


def compare(a, b):
    (xa, fa, ba), (xb, fb, bb) = a, b
    fs, bs = sorted(set(fa) & set(fb)), sorted(set(ba) & set(bb))
    ia_f = {f: i for i, f in enumerate(fa)}
    ib_f = {f: i for i, f in enumerate(fb)}
    ia_b = {c: i for i, c in enumerate(ba)}
    ib_b = {c: i for i, c in enumerate(bb)}
    A = xa[[ia_f[f] for f in fs]][:, [ia_b[c] for c in bs]]
    B = xb[[ib_f[f] for f in fs]][:, [ib_b[c] for c in bs]]
    d = abs(A - B)
    return {
        "barcodes": [len(ba), len(bb), len(bs)],
        "features": [len(fa), len(fb), len(fs)],
        "total": [float(xa.sum()), float(xb.sum())],
        "entries_differing": int((d > 0).sum()),
        "abs_diff": float(d.sum()),
        "identical": bool(len(ba) == len(bb) == len(bs) and len(fa) == len(fb) == len(fs) and d.sum() == 0),
    }


def stats(run):
    try:
        s = json.load(open(os.path.join(run, "pileup_stats.json")))
    except (OSError, ValueError):
        return {}
    keys = ("reverse_strand", "strand_auto_flipped", "strand_probed", "strand_probe_antisense_frac", "strand_retried",
            "counted_antisense_frac", "cell_caller", "n_called_knee", "n_called_cells")
    return {k: s.get(k) for k in keys if k in s}


def wall(run):
    try:
        t = open(os.path.join(run, "time.txt")).read()
    except OSError:
        return None
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    if not m:
        return None
    parts = [float(p) for p in m.group(1).split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        del args[i : i + 2]
    ref, new = args
    res = {"ref": ref, "new": new, "members": {}, "ref_stats": stats(ref), "new_stats": stats(new),
           "wall_s": [wall(ref), wall(new)]}
    for m in MEMBERS:
        a, b = load(ref, m), load(new, m)
        if a is None or b is None:
            res["members"][m] = "missing in %s" % ("ref" if a is None else "new")
            continue
        res["members"][m] = compare(a, b)
    res["all_identical"] = all(isinstance(v, dict) and v["identical"] for v in res["members"].values())
    print(json.dumps(res))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
