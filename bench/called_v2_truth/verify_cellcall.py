# SPDX-License-Identifier: MIT
"""Check that the fixed ``singlet.v2.cellcall.call`` (2.1) reproduces the study's recommended counts (docs/called-v2-truth.md).

  python -P verify_cellcall.py NAME RUN_DIR DATASET_JSON OUT_JSON

Rebuilds the exon + intron gene matrix of one Singlet output folder (``common.load_run``), runs ``cellcall.call`` from this
checkout with default parameters and compares the count with the ``"bounded knee + tail p, fdr .001"`` variant that
``datasets.py`` recorded in ``DATASET_JSON``; also compares the called sets barcode by barcode with ``common.finish`` on
``common.raw_pvalues`` (the study's own code path).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
os.environ.setdefault("CELLCALL_DIR", os.path.join(ROOT, "python", "singlet", "v2"))
sys.path.insert(0, HERE)

import common as K  # noqa: E402
from singlet.v2 import cellcall  # noqa: E402

VARIANT = "bounded knee + tail p, fdr .001"


def main(name, run_dir, ds_json, out_json):
    t0 = time.time()
    R = K.load_run(run_dir)
    X, called = R["X"], R["called"]
    cv2, fdr, params = cellcall.call(X, called, "10xv3")
    t1 = time.time()
    D = json.load(open(ds_json))
    want = D["variants"][VARIANT]["n"]
    # the study's path: raw p-values for every uncalled barcode above lower, then retain (bounded knee), floor 50, BH
    tot = R["tot"]
    lower = params["lower"]
    _, amb = K.raw_pvalues(X, called, lower)
    ref, _ = K.finish(tot, called, amb["pv_tail"], lower, 0.001, K.knee_bounded(tot, lower), 50)
    out = {
        "name": name,
        "run": run_dir,
        "cellcall_version": cellcall.VERSION,
        "n_called": int(called.sum()),
        "n_called_v2": int(cv2.sum()),
        "recommended_in_study": want,
        "matches_study": int(cv2.sum()) == want,
        "study_path_n": int(ref.sum()),
        "identical_to_study_path": bool(np.array_equal(ref, cv2)),
        "params": params,
        "seconds": round(t1 - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in out.items() if k != "params"}), flush=True)
    print({k: params.get(k) for k in ("lower", "retain_total", "alpha", "n_tested", "n_tail", "n_added")}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
