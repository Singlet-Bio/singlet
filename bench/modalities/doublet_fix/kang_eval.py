"""Score a dbl_run output on the Kang pool against demuxlet truth (same metrics as genotype_pool/doublets.py).
Usage: kang_eval.py TRUTH PREFIX RUN.tsv OLD_SCORES_CSV_GZ OUT.json [label]
OLD_SCORES_CSV_GZ is results/doublets_k8_scores.csv.gz (old Singlet and Scrublet columns on the labelled cells)."""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
truth_f, pre, run, old, out = sys.argv[1:6]; label = sys.argv[6] if len(sys.argv) > 6 else "singlet_new"
t = pd.read_csv(truth_f, sep="\t", index_col=0); t = t[t["stim"] == "ctrl"].copy(); t["bc"] = [i.split("-")[0] for i in t.index]; t = t.set_index("bc")
bcs = [l.strip().split("-")[0] for l in open(pre + ".barcodes.tsv")]
R = pd.read_csv(run, sep="\t"); R.index = bcs
df = pd.DataFrame({label: R.score.values, label + "_call": R.is_doublet.values.astype(bool)}, index=bcs)
df = df.join(t[["multiplets"]], how="inner"); d = df[df.multiplets.isin(["singlet", "doublet"])].copy()
O = pd.read_csv(old, index_col=0)
O = O[[c for c in ["scrublet", "scrublet_call", "singlet", "singlet_call"] if c in O]].rename(columns={"singlet": "singlet_old", "singlet_call": "singlet_old_call"})
d = d.join(O, how="left")
y = (d.multiplets == "doublet").values; rate = float(y.mean())
def row(s, c=None):
    s = np.asarray(s, float); k = int(round(rate * len(y))); top = np.argsort(-s, kind="stable")[:k]
    r = dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), recall_at_truth_rate=float(y[top].sum() / y.sum()))
    if c is not None:
        c = np.asarray(c, bool); tp = int((c & y).sum())
        r.update(native_calls=int(c.sum()), native_rate=float(c.mean()), native_recall=tp / y.sum(), native_precision=tp / max(c.sum(), 1))
        r["native_f1"] = 2 * r["native_precision"] * r["native_recall"] / max(r["native_precision"] + r["native_recall"], 1e-9)
    return r
res = {"cells_with_truth_label": len(d), "truth_doublets": int(y.sum()), "truth_doublet_rate": rate, label: row(d[label], d[label + "_call"])}
for nm in ["scrublet", "singlet_old"]:
    if nm in d and not d[nm].isna().all():
        res[nm] = row(d[nm].fillna(d[nm].min()), d[nm + "_call"].fillna(False).astype(bool) if nm + "_call" in d else None)
res["tied_zero_fraction_new"] = float((d[label] == 0).mean())
json.dump(res, open(out, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
