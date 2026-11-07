"""Which donor pairs do falsely called doublets (truth singlets) come from?  fp_pairs.py TRUTH donor_assignments.tsv"""
import sys
import pandas as pd
from common import load_truth, norm_bc

T = load_truth(sys.argv[1])
d = pd.read_csv(sys.argv[2], sep="\t")
d["bc"] = [norm_bc(b) for b in d["cell"]]
d = d.drop_duplicates("bc").set_index("bc")
com = d.index.intersection(T.index)
t, p = T.loc[com], d.loc[com]
# map predicted donor labels to truth donors by majority among truth singlets called a donor
sing = t["multiplets"] == "singlet"
cal = p["donor_id"].str.startswith("donor")
m = pd.crosstab(p.loc[sing & cal, "donor_id"], t.loc[sing & cal, "ind"]).idxmax(axis=1).to_dict()
print("label -> truth donor:", m)
fp = sing & (p["prob_doublet"] > 0.9)
tp = (t["multiplets"] == "doublet") & (p["prob_doublet"] > 0.9)
print("false doublet calls by (truth donor, best pair):")
x = pd.DataFrame({"truth": t.loc[fp, "ind"], "pair": p.loc[fp, "best_doublet"].map(lambda s: ",".join(m.get(q, q) for q in s.split(",")))})
print(x.value_counts().head(15).to_string())
print("true doublet calls, best pair counts (top):")
y = p.loc[tp, "best_doublet"].map(lambda s: ",".join(sorted(m.get(q, q) for q in s.split(","))))
print(y.value_counts().head(15).to_string())
print("llr (best pair - best single) by class:")
print(pd.DataFrame({"cls": t["multiplets"], "llr": p["best_doublet"].str.len() * 0 + 0}).shape)
