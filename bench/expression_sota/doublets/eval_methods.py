# SPDX-License-Identifier: MIT
"""AUPRC / AUROC per dataset of stored scores vs the experimental labels; paired bootstrap vs a reference method.
  python eval_methods.py OUT.csv REF METHOD [METHOD ...]"""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dbl_common import load, SCORES, DATA
out, ref, *methods = sys.argv[1:]
names = [n for n in sorted(os.listdir(DATA)) if "synth" not in n and "homsim" not in n]
rows = []
for n in names:
    lab = np.load(f"{DATA}/{n}/ds.npz")["label"]
    r = {"dataset": n}
    for m in [ref] + methods:
        p = f"{SCORES}/{n}/{m}.npy"
        if os.path.exists(p):
            s = np.load(p)
            r[m] = average_precision_score(lab, s); r[m + "_auroc"] = roc_auc_score(lab, s)
    rows.append(r)
D = pd.DataFrame(rows).set_index("dataset"); D.to_csv(out)
print(D[[ref] + methods].round(3).to_string())
rng = np.random.default_rng(0)
for m in methods:
    d = (D[m] - D[ref]).dropna()
    bs = [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f"{m} - {ref}: mean {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] wins {(d>0).sum()}/{len(d)} (n={len(d)}); mean AUPRC {D[m].mean():.3f} vs {D[ref].mean():.3f}")
