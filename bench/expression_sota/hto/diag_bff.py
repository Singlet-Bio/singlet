# SPDX-License-Identifier: MIT
"""Diagnostic: label conventions of the authors' published calls (Howitt 2023)."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_callers as H  # noqa: E402

Z, ART = sys.argv[1], sys.argv[2]
data = H.L.load_howitt(Z)
for ds, short in (("BAL2", "bal2"), ("BAL3", "bal3")):
    caps, hmap = data[ds]
    P = pd.read_parquet(os.path.join(ART, f"hto_howitt2023_{short}_published.parquet"))
    T = pd.read_parquet(os.path.join(ART, f"hto_howitt2023_{short}_truth.parquet"))
    print(ds, "hmap", hmap)
    print("map.json", json.load(open(os.path.join(ART, f"hto_howitt2023_{short}_map.json")))["hashtag_to_donor"])
    print("truth cols", list(T.columns), T.head(2).to_dict("records"))
    for col in ("BFF_cluster", "BFF_raw", "HTODemux"):
        print(col, P[col].value_counts().to_dict())
    allbc = [b for _, h, _ in caps for b in h.columns]
    print("P barcodes", P.barcode.head(3).tolist(), "mine", allbc[:3], "len", len(P), len(allbc), "dups", int(P.barcode.duplicated().sum()), "overlap", len(set(P.barcode) & set(allbc)))
