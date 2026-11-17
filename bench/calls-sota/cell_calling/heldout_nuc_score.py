# SPDX-License-Identifier: MIT
"""Gate amendment A scorer: recall of tag-confirmed nuclei.

    python -P heldout_nuc_score.py TABLE.tsv.gz OUT.json
"""
import json
import sys

import numpy as np
import pandas as pd

t = pd.read_csv(sys.argv[1], sep="\t", dtype={"barcode": str})
T = (t["truth"] == "cell").to_numpy()
tot = t["umi_total"].to_numpy().astype(float)
N = int(T.sum())
calls = {"singlet": t["called_v2"].to_numpy().astype(bool), "emptydrops": t["edcr"].to_numpy().astype(bool)}
auth = t["in_authors_set"].to_numpy().astype(bool)


def topn(a):
    o = np.lexsort((-tot, ~a))
    s = np.zeros(len(a), bool)
    s[o[:N]] = True
    return s


sel = {k: topn(v) for k, v in calls.items()}
res = {"n_tag_confirmed": N, "tools": {}}
for k, a in calls.items():
    res["tools"][k] = {"n_called": int(a.sum()), "recall_own_call": round(float((a & T).sum() / N), 4),
                       "recall_at_N": round(float((sel[k] & T).sum() / N), 4),
                       "share_calls_outside_authors_set_T": round(float((a & ~auth).sum() / max(a.sum(), 1)), 4),
                       "jaccard_with_authors_set_T": round(float((a & auth).sum() / max((a | auth).sum(), 1)), 4)}
rng = np.random.default_rng(17)
idx = rng.integers(0, len(t), size=(300, len(t)))
k = "emptydrops"
d1 = [((calls["singlet"][i] & T[i]).sum() - (calls[k][i] & T[i]).sum()) / max(T[i].sum(), 1) for i in idx]
d2 = [((sel["singlet"][i] & T[i]).sum() - (sel[k][i] & T[i]).sum()) / max(T[i].sum(), 1) for i in idx]
res["tools"][k]["recall_own_singlet_minus"] = [round(res["tools"]["singlet"]["recall_own_call"] - res["tools"][k]["recall_own_call"], 4),
                                                [round(float(x), 4) for x in np.percentile(d1, [2.5, 97.5])]]
res["tools"][k]["recall_at_N_singlet_minus"] = [round(res["tools"]["singlet"]["recall_at_N"] - res["tools"][k]["recall_at_N"], 4),
                                                 [round(float(x), 4) for x in np.percentile(d2, [2.5, 97.5])]]
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res, indent=1))
