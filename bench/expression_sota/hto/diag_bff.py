# SPDX-License-Identifier: MIT
"""Diagnostics of the BFF reimplementation against the authors' published BFF calls (Howitt 2023)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_callers as H  # noqa: E402

Z, ART = sys.argv[1], sys.argv[2]
data = H.L.load_howitt(Z)
for ds, tag in (("BAL1", "bal1"), ("CL", "cl")):
    caps, hmap = data[ds]
    P = pd.read_parquet(os.path.join(ART, f"hto_howitt2023_{tag}_published.parquet"))
    print(ds, "published columns", list(P.columns), "rows", len(P))
    print(P.head(3).to_string())
    cname, h, tr = caps[0]
    print("capture", cname, "cells", h.shape[1], "barcode examples", list(h.columns[:2]))
    X = h.to_numpy(float)
    rng = np.random.default_rng(0)
    thr = [H.bff_threshold(X[j], rng) for j in range(X.shape[0])]
    print("bff thresholds", np.round(thr, 1), "median counts per tag", np.median(X, axis=1))
    call, info = H.bff_cluster(X)
    print("Tn,T2", info["Tn"], info["T2"], "S/N/D", (call >= 0).sum(), (call == -1).sum(), (call == -2).sum())
    raw, _ = H.bff_raw(X)
    print("raw S/N/D", (raw >= 0).sum(), (raw == -1).sum(), (raw == -2).sum())
    print("truth", pd.Series(tr.values).value_counts().to_dict())
    for col in P.columns:
        if "BFF" in col:
            print(col, P[col].value_counts().head(12).to_dict())
