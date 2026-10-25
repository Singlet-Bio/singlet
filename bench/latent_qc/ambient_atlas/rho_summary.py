"""Collect the production-estimator results (rho/<GSM>_{prod,rule1}.json) into rho_by_sample.csv and group summaries.

usage: rho_summary.py RHO_DIR SAMPLE_SUMMARY_CSV OUT_DIR
"""
import glob
import json
import os
import sys

import pandas as pd

rho_dir, summ_csv, out = sys.argv[1:4]
S = pd.read_csv(summ_csv)
rows = []
for f in glob.glob(f"{rho_dir}/GSM*_*.json"):
    b = os.path.basename(f)[:-5]
    gsm, v = b.split("_")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append({"gsm": gsm, **{f"{v}_{k}": x for k, x in d.items() if k != "method"}})
R = pd.DataFrame(rows).groupby("gsm", as_index=False).first()
D = S.merge(R, on="gsm", how="left")
keep = ["gsm", "gse", "tissue_coarse", "tissue_label", "chem", "nuclei", "intron_frac_called", "n_called", "med_called_umi",
        "pool1_n", "pool1_umi", "pool2_n", "pool2_umi", "amb_mt", "amb_hb", "amb_malat1", "amb_rp", "amb_ig"] + \
       [c for c in D.columns if c.startswith(("prod_", "rule1_"))]
D[keep].to_csv(f"{out}/rho_by_sample.csv", index=False, float_format="%.5g")
print("samples with production result:", int(D.prod_ok.notna().sum()), "ok==1:", int((D.prod_ok == 1).sum()))
print("runtime per sample (s, estimator / incl. load): median", D.prod_seconds.median(), D.prod_wall_seconds.median(),
      "p90", D.prod_seconds.quantile(.9), D.prod_wall_seconds.quantile(.9), "max", D.prod_wall_seconds.max())
for col in ("tissue_coarse", "nuclei", "chem"):
    for v in ("prod", "rule1"):
        g = D[D[f"{v}_ok"] == 1].groupby(col).agg(n=("gsm", "size"), median_of_mean_rho=(f"{v}_mean_rho", "median"),
                                                 q25=(f"{v}_mean_rho", lambda x: x.quantile(.25)), q75=(f"{v}_mean_rho", lambda x: x.quantile(.75)),
                                                 median_umi_weighted=(f"{v}_umi_weighted_rho", "median"),
                                                 median_pool_umi=(f"{v}_pool_umi", "median")).round(3)
        g.to_csv(f"{out}/rho_summary_{col}_{v}.csv")
        print(f"\n== {col}, estimator={v} (production pool rule: {'<=100 UMI, cap 50000' if v == 'prod' else 'pool-rule-1 UMI cap'})")
        print(g.sort_values("median_of_mean_rho", ascending=False).to_string())
print("\nspearman(prod mean rho, rule1 mean rho):", D[["prod_mean_rho", "rule1_mean_rho"]].corr("spearman").iloc[0, 1])
print("spearman(prod mean rho, ambient MT/HB/MALAT1 share):", D[["prod_mean_rho", "amb_mt", "amb_hb", "amb_malat1"]].corr("spearman").iloc[0, 1:].to_dict())
print("overall: mean_rho quantiles", D.prod_mean_rho.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("fraction of samples with rho at upper bound region (median_rho>=0.45):", float((D.prod_median_rho >= 0.45).mean()))
