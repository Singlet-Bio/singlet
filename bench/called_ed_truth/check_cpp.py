# SPDX-License-Identifier: MIT
"""Does the C++ called_v2 2.1 (cv2_stored on a run's stored counts) equal the truth-check script's recommended call?

    python -P bench/called_ed_truth/check_cpp.py NAME CPP.tsv TRUTH.barcodes.tsv.gz [--json OUT]

CPP.tsv is cv2_stored's output (barcode, called, called_v2, umi_total, pvalue, fdr). TRUTH.barcodes.tsv.gz is datasets.py's
per-barcode table (every called barcode and every candidate), with the recommended call in the column
"v:bounded knee + tail p, fdr .001" (docs/called-v2-truth.md). Acceptance: identical barcode sets. Exit status 1 otherwise.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

COL = "v:bounded knee + tail p, fdr .001"


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        del args[i : i + 2]
    name, cpp_path, truth_path = args
    cpp = pd.read_csv(cpp_path, sep="\t", dtype={"barcode": str}, na_values=["NA"])
    tr = pd.read_csv(truth_path, sep="\t", dtype={"barcode": str}, usecols=["barcode", "called", "pvalue_tail", COL])
    c_set = set(cpp.loc[cpp["called_v2"] == 1, "barcode"])
    t_set = set(tr.loc[tr[COL].astype(bool), "barcode"])
    only_c, only_t = sorted(c_set - t_set), sorted(t_set - c_set)
    m = cpp.merge(tr, on="barcode", how="inner")
    both = m["pvalue"].notna() & m["pvalue_tail"].notna()
    tail = both & (m["pvalue_tail"] < 1 / 10001) & (m["pvalue_tail"] > 0)
    rel = (m.loc[tail, "pvalue"] / m.loc[tail, "pvalue_tail"] - 1).abs()
    mc = both & ~tail & (m["pvalue_tail"] > 0)
    res = {
        "name": name,
        "called": int(cpp["called"].sum()),
        "called_v2_cpp": len(c_set),
        "called_v2_script": len(t_set),
        "identical": not only_c and not only_t,
        "only_cpp": only_c[:20],
        "only_script": only_t[:20],
        "n_only_cpp": len(only_c),
        "n_only_script": len(only_t),
        "tested_both": int(both.sum()),
        "mc_p_equal_share": float((m.loc[mc, "pvalue"] == m.loc[mc, "pvalue_tail"]).mean()) if mc.any() else 1.0,
        "mc_p_max_abs_diff": float((m.loc[mc, "pvalue"] - m.loc[mc, "pvalue_tail"]).abs().max()) if mc.any() else 0.0,
        "n_tail": int(tail.sum()),
        "tail_p_max_rel_diff": float(rel.max()) if len(rel) else 0.0,
        "tail_p_median_rel_diff": float(np.median(rel)) if len(rel) else 0.0,
    }
    print(json.dumps(res))
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    sys.exit(0 if res["identical"] else 1)


if __name__ == "__main__":
    main()
