"""Sensitivity of the ambient profile to the pool rule. usage: pool_sens.py REC_DIR SAMPLES_TSV OUT_CSV
pool1: uncalled, UMI < median(uncalled) and < 10% of called median; pool2: uncalled <= 100 UMIs (production rule);
pool3: uncalled, UMI < 25th pct of uncalled and < 5% of called median. Per sample: Spearman over genes expressed in both and
KL(pool_a || pool_b) with a 0.5-pseudocount floor; and KL(pool_k || own called-cell mean) (how cell-like each pool is)."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aa_load import load_records

S, M, genes, sym, gcl = load_records(sys.argv[1], sys.argv[2])
rows = []


def prob(v, a=0.5):
    v = v.astype(float) + a * (v.sum() / max(len(v), 1)) / max(v.sum() / len(v), 1e-12) * 0 + a
    return v / v.sum()


for i in range(len(S)):
    p = {k: M[k][i].astype(float) for k in ("pool1", "pool2", "pool3", "called")}
    if min(p["pool1"].sum(), p["pool2"].sum(), p["pool3"].sum()) < 2000:
        continue
    q = {k: prob(v) for k, v in p.items()}
    kl = lambda a, b: float((q[a] * np.log2(q[a] / q[b])).sum())
    e = (p["pool1"] > 0) & (p["pool2"] > 0) & (p["pool3"] > 0)
    rows.append(dict(gsm=S.gsm[i], tissue=S.tissue_coarse[i], pool1_umi=p["pool1"].sum(), pool2_umi=p["pool2"].sum(), pool3_umi=p["pool3"].sum(),
                     sp12=spearmanr(p["pool1"][e], p["pool2"][e])[0], sp13=spearmanr(p["pool1"][e], p["pool3"][e])[0],
                     kl_1_2=kl("pool1", "pool2"), kl_1_3=kl("pool1", "pool3"),
                     kl_pool1_cells=kl("pool1", "called"), kl_pool2_cells=kl("pool2", "called"), kl_pool3_cells=kl("pool3", "called"),
                     intron1=S.intron_frac_pool[i]))
D = pd.DataFrame(rows)
D.to_csv(sys.argv[3], index=False, float_format="%.4g")
print(len(D), D.drop(columns=["gsm", "tissue"]).describe().loc[["25%", "50%", "75%"]].round(3).to_string())
