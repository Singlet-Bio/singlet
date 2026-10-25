"""Extra paired numbers on the common 333-sample subset (n=30000 available). usage: prior_extra.py RESULTS_DIR"""
import json
import sys

import numpy as np
import pandas as pd

res = sys.argv[1]
R = pd.read_csv(f"{res}/prior_test.csv")
ok = R[(R.method == "emp_alone") & (R.n == 30000)].gsm.unique()
R = R[R.gsm.isin(ok)]
g = R.groupby(["method", "n", "gsm"])[["kl", "tv", "spearman"]].mean().reset_index()
print("common subset", len(ok), "samples")
for m in ["prior_cell", "prior_rw", "prior_mix", "prior_tissue", "prior_glob"]:
    x = g[g.method == m]
    print(m, len(x), round(x.kl.mean(), 3), round(x.tv.mean(), 3), round(x.spearman.mean(), 3))
pm = g[g.method == "prior_mix"].set_index("gsm").kl
for n in [100, 300, 1000, 3000, 10000, 30000]:
    em = g[(g.method == "emp_mix") & (g.n == n)].set_index("gsm").kl
    ea = g[(g.method == "emp_alone") & (g.n == n)].set_index("gsm").kl
    et = g[(g.method == "emp_tissue") & (g.n == n)].set_index("gsm").kl
    print(n, "alone", round(ea.mean(), 3), "tissue", round(et.mean(), 3), "mix", round(em.mean(), 3), "prior_mix(no pool)", round(pm.mean(), 3),
          "frac mix<prior_only", round((em < pm).mean(), 3), "frac alone<prior_only", round((ea < pm).mean(), 3))
h = json.load(open(f"{res}/prior_hparams.json"))
print("mean pi_mix [cell, reweighted, tissue, global, uniform]", np.mean([x["pi_mix"] for x in h], 0).round(3), "lam", pd.Series([x["lam"] for x in h]).value_counts().to_dict())
print("median kappa", {k: float(np.median([x["kappa"][k] for x in h])) for k in h[0]["kappa"] if k.startswith(("mix", "unif"))})
a = R[R.method == "prior_mix"].groupby("tissue").kl.mean()
b = R[R.method == "prior_cell"].groupby("tissue").kl.mean()
c = R[R.method == "prior_tissue"].groupby("tissue").kl.mean()
print(pd.DataFrame({"cell": b, "mix": a, "tissue_only": c}).round(2).to_string())
