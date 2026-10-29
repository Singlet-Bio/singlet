"""Donor demux + genotype-based doublet eval vs Kang truth. Usage: eval_demux.py TRUTH OUTDIR TAG [singlet_dir ref_dir ...]"""
import sys, json, pandas as pd, numpy as np
from sklearn.metrics import adjusted_rand_score
from common import *
truth_f, out, *runs = sys.argv[1:]
T = load_truth(truth_f)
res = {}
def evaluate(name, df, lab, pdbl, bar_col):
    df = df.copy(); df["bc"] = [norm_bc(b) for b in df[bar_col]]; df = df.drop_duplicates("bc").set_index("bc")
    common = df.index.intersection(T.index)
    n_called_cells, n_truth = len(df), len(T)
    t = T.loc[common]; d = df.loc[common]
    sing = t["multiplets"] == "singlet"
    pred = d[lab].astype(str)
    unas = pred.isin(["unassigned", "doublet"])
    # singlets: ARI over all truth singlets (unassigned/doublet as their own labels), accuracy over assigned
    ari = adjusted_rand_score(t.loc[sing, "ind"], pred[sing])
    ari_assigned = adjusted_rand_score(t.loc[sing & ~unas, "ind"], pred[sing & ~unas])
    acc, mp = best_match_acc(t.loc[sing, "ind"], pred[sing])
    # per-donor recall
    inv = {v: k for k, v in mp.items()}
    per = {}
    for dn, g in t[sing].groupby("ind"):
        per[dn] = float((pred[g.index] == inv.get(dn, "")).mean())
    # doublet detection by the demux
    y = (t["multiplets"] == "doublet").values; ok = t["multiplets"].isin(["singlet", "doublet"]).values
    auc, ap = auc_ap(y[ok], d.loc[ok, pdbl].values)
    called_dbl = (pred == "doublet").values
    res[name] = dict(cells_called=n_called_cells, truth_cells=n_truth, cells_in_truth=len(common),
        truth_singlets=int(sing.sum()), ari_all_singlets=ari, ari_assigned_only=ari_assigned,
        acc_assigned=acc, unassigned_rate=float((pred[sing] == "unassigned").mean()),
        singlets_called_doublet=float((pred[sing] == "doublet").mean()),
        per_donor_recall=per, dbl_auroc_genotype=auc, dbl_auprc_genotype=ap,
        dbl_recall_at_call=float(called_dbl[y].mean()) if y.sum() else None,
        dbl_precision_at_call=float(y[called_dbl].mean()) if called_dbl.sum() else None,
        n_donors_found=int(pred[~unas].nunique()))
for r in runs:
    kind, path = r.split("=", 1)
    if kind.startswith("singlet"):
        d = pd.read_csv(path, sep="\t"); evaluate(kind, d, "donor_id", "prob_doublet", "cell")
    elif kind.startswith("vireo"):
        d = pd.read_csv(path, sep="\t"); evaluate(kind, d, "donor_id", "prob_doublet", "cell")
json.dump(res, open(out, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
