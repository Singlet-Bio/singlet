"""Summarise prior_test.csv (per sample x method x n x rep). usage: prior_summarize.py prior_test.csv OUT_DIR
n=-1 denotes 'all of half A' (the whole other half of the pool; its size differs per sample)."""
import sys

import numpy as np
import pandas as pd

R = pd.read_csv(sys.argv[1])
out = sys.argv[2]
NS = {100, 300, 1000, 3000, 10000, 30000}
emp = R.method.str.startswith("emp_")
R.loc[emp & ~R.n.isin(NS), "n"] = -1
R.loc[R.method.str.startswith("raw_"), "kl"] = np.nan  # raw (unsmoothed) predictors have zeros: KL undefined
g = R.groupby(["method", "n", "gsm", "gse"], as_index=False)[["kl", "tv", "spearman"]].mean()
summ = g.groupby(["method", "n"]).agg(n_samples=("tv", "size"), mean_kl_bits=("kl", "mean"), median_kl_bits=("kl", "median"),
                                      mean_tv=("tv", "mean"), median_tv=("tv", "median"), mean_spearman=("spearman", "mean")).reset_index()
summ.to_csv(f"{out}/prior_summary.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 200)
print(summ.to_string())
rng = np.random.default_rng(3)
rows = []
for nn in sorted(g.n.unique()):
    if nn == 0:
        continue
    a = g[(g.method == "emp_alone") & (g.n == nn)].set_index("gsm")
    comps = {"emp_tissue": g[(g.method == "emp_tissue") & (g.n == nn)].set_index("gsm"),
             "emp_mix": g[(g.method == "emp_mix") & (g.n == nn)].set_index("gsm")}
    for nm, b in comps.items():
        common = a.index.intersection(b.index)
        for metric, sign in (("kl", 1), ("tv", 1), ("spearman", -1)):
            d = (b[metric][common] - a[metric][common]) * sign  # negative = prior-shrunk empirical is better
            st = a.gse[common].values
            dv = d.values
            u, inv = np.unique(st, return_inverse=True)
            sums = np.bincount(inv, weights=dv, minlength=len(u))
            cnts = np.bincount(inv, minlength=len(u)).astype(float)
            bs = []
            for _ in range(400):
                pk = rng.integers(0, len(u), len(u))
                bs.append(sums[pk].sum() / cnts[pk].sum())
            rows.append(dict(n=nn, contrast=f"{nm} vs emp_alone", metric=metric, n_samples=len(common), mean_delta_prior_minus_alone=d.mean() * sign,
                             ci_lo=np.percentile(bs, 2.5) * sign if sign > 0 else np.percentile(bs, 97.5) * sign,
                             ci_hi=np.percentile(bs, 97.5) * sign if sign > 0 else np.percentile(bs, 2.5) * sign,
                             frac_samples_prior_better=float((d < 0).mean())))
    for pn in ("prior_mix", "prior_tissue", "prior_cell"):
        pm = g[g.method == pn].set_index("gsm")
        common = a.index.intersection(pm.index)
        d = pm.kl[common] - a.kl[common]
        rows.append(dict(n=nn, contrast=f"{pn} (no data from this pool) vs emp_alone", metric="kl", n_samples=len(common),
                         mean_delta_prior_minus_alone=d.mean(), frac_samples_prior_better=float((d < 0).mean())))
C = pd.DataFrame(rows)
C.to_csv(f"{out}/prior_contrasts.csv", index=False, float_format="%.4g")
print(C[C.metric == "kl"].to_string())
