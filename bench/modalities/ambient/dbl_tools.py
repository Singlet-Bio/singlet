"""Scrublet on the gene-level matrix with synthetic doublets, then score all doublet callers (Singlet harness, Scrublet, scDblFinder)."""
import json, os, sys
import numpy as np, pandas as pd, scipy.io, scrublet as scr
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

D = f"{WORK}/dbl"; R = f"{A}/results"
truth = pd.read_csv(f"{D}/truth.csv")
if sys.argv[1:] == ["scrublet"]:
    X = scipy.io.mmread(f"{D}/gene_synth.mtx").T.tocsr().astype(np.float32)
    for er in (0.03, 0.08):
        s = scr.Scrublet(X, expected_doublet_rate=er, random_state=0)
        sc, pred = s.scrub_doublets(verbose=False)
        pd.DataFrame({"col": np.arange(X.shape[0]), "score": sc, "is_doublet": pred.astype(int)}).to_csv(f"{D}/scrublet_er{er}.tsv", sep="\t", index=False)
    sys.exit(0)
calls = {"Singlet detect_doublets (harness)": pd.read_csv(f"{D}/singlet.tsv", sep="\t"),
         "scDblFinder": pd.read_csv(f"{D}/scdblfinder.tsv", sep="\t"),
         "Scrublet (rate 0.03)": pd.read_csv(f"{D}/scrublet_er0.03.tsv", sep="\t"),
         "Scrublet (rate 0.08)": pd.read_csv(f"{D}/scrublet_er0.08.tsv", sep="\t")}
t = truth.truth.values; syn = truth.synthetic.values
rows = []
for k, d in calls.items():
    c = d.is_doublet.values.astype(bool)
    tp = int((c & t).sum()); fp = int((c & ~t).sum()); fn = int((~c & t).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    rows.append(dict(tool=k, n_truth=int(t.sum()), n_called=int(c.sum()), tp=tp, fp=fp, fn=fn, precision=pr, recall=rc,
                     f1=2 * pr * rc / max(pr + rc, 1e-9), auroc=roc_auc_score(t, d.score.values), auprc=average_precision_score(t, d.score.values),
                     fp_among_natural_cells=int((c & ~t & ~syn).sum())))
out = pd.DataFrame(rows); out.to_csv(f"{R}/doublets_synthetic.csv", index=False)
pd.set_option("display.width", 250); print(out.round(4).to_string(index=False))
