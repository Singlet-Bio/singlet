# SPDX-License-Identifier: MIT
"""Fidelity of the BFF reimplementation: agreement with the authors' published BFF_raw / BFF_cluster calls (Howitt 2023).

    python -P fidelity.py ZENODO ARTIFACTS OUT.json

Agreement is on classes (singlet / doublet / negative) and, among cells both call singlet, on the tag. No truth label is used,
so choosing the reading of the BFF text that agrees best with the published calls is not selection on the evaluation truth.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_callers as H  # noqa: E402

Z, ART, OUT = sys.argv[1:4]
data = H.L.load_howitt(Z)
TAGNAME = {"BAL1": "bal1", "BAL2": "bal2", "BAL3": "bal3", "CL": "cl"}
VARIANTS = [("raw", None, None)] + [("cluster", a, b) for a in ("edge", "valley") for b in ("edge_M", "edge_own", "valley", "top")]


def pub_classes(col, k):
    c = np.where(col == "Doublet", "D", np.where(col == "Negative", "N", "S"))
    tag = np.array([(int(str(x).split()[-1]) - 1) % k if cl == "S" else -1 for x, cl in zip(col, c)])
    return c, tag


res = {}
for ds, short in TAGNAME.items():
    caps, hmap = data[ds]
    P = pd.read_parquet(os.path.join(ART, f"hto_howitt2023_{short}_published.parquet"))
    allbc = [b for _, h, _ in caps for b in h.columns]
    if len(P) == len(allbc) and (P.barcode.values == np.array(allbc)).all():
        Pa = P.reset_index(drop=True)
    else:
        Pa = P.drop_duplicates("barcode").set_index("barcode").reindex(allbc).reset_index()
    res[ds] = {}
    for kind, tn, t2 in VARIANTS:
        calls = []
        for _, h, _ in caps:
            X = h.to_numpy(float)
            calls.append(H.bff_raw(X)[0] if kind == "raw" else H.bff_cluster(X, tn_rule=tn, t2_rule=t2)[0])
        mine = np.concatenate(calls)
        mc = np.where(mine >= 0, "S", np.where(mine == H.DBL, "D", "N"))
        pcol = Pa["BFF_raw" if kind == "raw" else "BFF_cluster"].astype(object).values
        ok = pd.notna(pcol)
        pc, ptag = pub_classes(pcol[ok], len(caps[0][1]))
        agree = float((mc[ok] == pc).mean())
        both = (mc[ok] == "S") & (pc == "S")
        tagagree = float((mine[ok][both] == ptag[both]).mean()) if both.any() else float("nan")
        name = kind if kind == "raw" else f"cluster/{tn}/{t2}"
        res[ds][name] = {"class_agree": agree, "tag_agree_when_both_singlet": tagagree,
                         "mine_SDN": [int((mc == c).sum()) for c in "SDN"], "pub_SDN": [int((pc == c).sum()) for c in "SDN"]}
        print(ds, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[ds][name].items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
