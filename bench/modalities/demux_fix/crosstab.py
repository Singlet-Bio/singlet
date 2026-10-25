"""Crosstab of truth (demuxlet donor / doublet) against a Singlet donor_assignments.tsv.
Usage: crosstab.py TRUTH.tsv.gz donor_assignments.tsv [more.tsv ...]"""
import sys
import pandas as pd
from common import load_truth, norm_bc

T = load_truth(sys.argv[1])
for path in sys.argv[2:]:
    d = pd.read_csv(path, sep="\t")
    d["bc"] = [norm_bc(b) for b in d["cell"]]
    d = d.drop_duplicates("bc").set_index("bc")
    com = d.index.intersection(T.index)
    t, p = T.loc[com], d.loc[com]
    truth = t["ind"].where(t["multiplets"] == "singlet", "dbl:" + t["multiplets"])
    print("==", path)
    ct = pd.crosstab(truth, p["donor_id"])
    print(ct.to_string())
    # which comps are populated by truth singlets vs doublets
    print("calls:", p["donor_id"].value_counts().to_dict())
    print("mean prob_doublet by truth class:")
    print(p.groupby(truth)["prob_doublet"].agg(["mean", "median", "count"]).to_string())
