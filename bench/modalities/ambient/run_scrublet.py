"""Scrublet on the called cells (same matrix as the other tools)."""
import os, sys, json, numpy as np, pandas as pd, scipy.io, scrublet as scr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
meta = pd.read_csv(f"{WORK}/cells.csv")
X = scipy.io.mmread(f"{WORK}/cells.mtx").T.tocsr().astype(np.float32)
res = {}
for er in (0.03, 0.08):
    s = scr.Scrublet(X, expected_doublet_rate=er, random_state=0)
    sc, pred = s.scrub_doublets(verbose=False)
    pd.DataFrame({"barcode": meta.barcode, "score": sc, "call": pred}).to_csv(f"{A}/results/scrublet_er{er}.csv", index=False)
    res[str(er)] = {"threshold": float(s.threshold_), "n_called": int(pred.sum())}
json.dump(res, open(f"{A}/results/scrublet.json", "w"), indent=1); print(res)
