"""Barnyard (human-mouse) cross-species truth for harness estimates, using bench/latent_qc/ambient_truth/atlib.py (PR #43).

usage: score_barnyard.py ATLIB_DIR OUT_JSONL LIB NAME=PREFIX [NAME=PREFIX ...]
PREFIX.rho.f64 is in the order of the library's called cells (work/LIB_full/in/cells.idx). One JSON line per estimate: Spearman and
Pearson against truth over pure cells, ratio of mean estimate to mean truth, and estimate / truth by depth tercile.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import atlib  # noqa: E402


def main(out_jsonl, lib, specs):
    ds = atlib.load(lib, "full")
    with open(out_jsonl, "a") as fo:
        for s in specs:
            name, prefix = s.split("=", 1)
            if not os.path.exists(prefix + ".rho.f64"):
                print("missing", prefix, flush=True)
                continue
            rho = np.fromfile(prefix + ".rho.f64", np.float64)
            r = atlib.est_row(ds, name, rho)
            js = prefix + ".json"
            if os.path.exists(js):
                r["seconds"] = json.load(open(js)).get("seconds")
            fo.write(json.dumps(r) + "\n")
            print(f"{lib} {name}: spearman {r['spearman']:.3f} ratio {r['ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3], sys.argv[4:])
