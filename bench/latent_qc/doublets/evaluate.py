"""Score every doublet method on every dataset, the cross-study supervised combiner, homotypic blind spots and cost.
Usage: evaluate.py OUT_DIR      (reads $LQC/dbl/data, $LQC/dbl/scores; writes CSV + results.md to OUT_DIR)

Supervised combiner ("atlas-scale real labels"): a gradient-boosted classifier over per-cell scores, each rank-transformed
within its sample, trained on the real experimental labels of every OTHER study and applied to the held-out study
(leave-one-study-out; samples of one study, e.g. the five Kang lanes, are always held out together).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from dbl_common import DATA, SCORES, load

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SYNTH = {"hgmm1k_synth", "kang21_homsim"}
# study of each dataset (leave-one-study-out groups); unknown names are their own study
STUDY = {"pbmc-1A-dm": "kang2018", "pbmc-1B-dm": "kang2018", "pbmc-1C-dm": "kang2018", "pbmc-2ctrl-dm": "kang2018",
         "pbmc-2stim-dm": "kang2018", "kang21": "kang2018", "kang21_homsim": "kang2018", "J293t-dm": "kang2018_293t",
         "pbmc-ch": "stoeckius2018", "cline-ch": "stoeckius2018", "mkidney-ch": "mkidney_hashing",
         "HEK-HMEC-MULTI": "mcginnis2019", "HMEC-orig-MULTI": "mcginnis2019", "HMEC-rep-MULTI": "mcginnis2019",
         "pdx-MULTI": "mcginnis2019", "nuc-MULTI": "mcginnis2019", "hm-6k": "10x_hgmm", "hm-12k": "10x_hgmm",
         "hgmm1k_synth": "10x_hgmm"}
METHODS = ["umi", "scrublet", "prod", "scdblfinder", "nnls", "solo", "atlas_knn", "atlas_clf", "ct_entropy", "ct_maxprob",
           "atlas_nnls"]
FEATS = ["umi", "genes", "scrublet", "prod", "scdblfinder", "nnls", "solo", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls"]
SETS = {"combo_all": FEATS, "combo_no_atlas": ["umi", "genes", "scrublet", "prod", "scdblfinder", "nnls", "solo"],
        "combo_atlas": ["umi", "genes", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls"],
        "combo_prod_atlas": ["umi", "genes", "prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls"]}


def get(name, m, X=None):
    p = f"{SCORES}/{name}/{m}.npy"
    if m == "umi":
        return np.log1p(np.asarray(X.sum(1)).ravel())
    if m == "genes":
        return np.log1p(np.diff(X.indptr))
    if m == "scdblfinder":
        p2 = f"{SCORES}/{name}/scdblfinder.tsv"
        return pd.read_csv(p2, sep="\t").score.values if os.path.exists(p2) else None
    return np.load(p) if os.path.exists(p) else None


def seconds(name, m):
    if m == "scdblfinder":
        p = f"{SCORES}/{name}/scdblfinder.seconds"
        return float(open(p).read()) if os.path.exists(p) else np.nan
    p = f"{SCORES}/{name}/{m}.json"
    return json.load(open(p)).get("seconds", np.nan) if os.path.exists(p) else np.nan


def metrics(y, s):
    s = np.nan_to_num(np.asarray(s, float), nan=np.nanmin(s) if np.isfinite(s).any() else 0)
    k = int(round(y.mean() * len(y)))
    top = np.argsort(-s, kind="stable")[:k]
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), recall_at_rate=y[top].sum() / y.sum())


names = [l.strip() for l in open(f"{os.path.dirname(DATA)}/datasets.txt") if l.strip()]
rows, feats, info = [], {}, []
for n in names:
    d = load(n)
    y = d["label"].astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        continue
    info.append(dict(dataset=n, study=STUDY.get(n, n), cells=len(y), doublets=int(y.sum()), rate=y.mean(), species=d["species"],
                     genes=d["X"].shape[1], median_umi=float(np.median(np.asarray(d["X"].sum(1)).ravel())), synthetic=n in SYNTH))
    F = {}
    for m in sorted(set(METHODS + FEATS)):
        s = get(n, m, d["X"])
        if s is None or len(s) != len(y):
            continue
        F[m] = s
        if m in METHODS:
            rows.append(dict(dataset=n, method=m, **metrics(y, s), seconds=seconds(n, m), cells=len(y)))
    for k in ("x_vireo_geno", "x_prod_geno"):
        if k in d:
            rows.append(dict(dataset=n, method=k[2:], **metrics(y, d[k]), seconds=np.nan, cells=len(y)))
    feats[n] = (F, y, d)

info = pd.DataFrame(info)
info.to_csv(f"{OUT}/datasets.csv", index=False)


def qmat(F, cols):
    return np.column_stack([rankdata(F[c]) / len(F[c]) if c in F else np.full(len(next(iter(F.values()))), 0.5) for c in cols])


# ---- leave-one-study-out supervised combiners (trained on real labels only)
real = [n for n in feats if n not in SYNTH]
for setname, cols in SETS.items():
    for n in feats:
        F, y, _ = feats[n]
        if any(c not in F for c in cols if c not in ("umi", "genes")):
            continue
        tr = [m for m in real if STUDY.get(m, m) != STUDY.get(n, n) and all(c in feats[m][0] for c in cols)]
        if not tr:
            continue
        Xtr = np.vstack([qmat(feats[m][0], cols) for m in tr])
        ytr = np.concatenate([feats[m][1] for m in tr])
        wtr = np.concatenate([np.full(len(feats[m][1]), 1 / len(feats[m][1])) for m in tr])  # each dataset weighs the same
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                             random_state=0).fit(Xtr, ytr, sample_weight=wtr * len(wtr) / len(tr))
        s = clf.predict_proba(qmat(F, cols))[:, 1]
        feats[n][0][setname] = s
        rows.append(dict(dataset=n, method=setname, **metrics(y, s), seconds=np.nan, cells=len(y), train_datasets=len(tr)))
    # unsupervised rank average of the same features (no labels used)
for n in feats:
    F, y, _ = feats[n]
    cols = [c for c in ("scdblfinder", "solo", "atlas_knn", "prod") if c in F]
    if len(cols) == 4:
        rows.append(dict(dataset=n, method="rankavg_sdf_solo_atlas_prod", **metrics(y, qmat(F, cols).mean(1)), seconds=np.nan,
                         cells=len(y)))

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/per_dataset.csv", index=False)
R["synthetic"] = R.dataset.isin(SYNTH)
summ = (R[~R.synthetic].groupby("method")[["auroc", "auprc", "recall_at_rate"]].agg(["mean", "median", "count"]).round(3))
summ.to_csv(f"{OUT}/summary_real.csv")
wide = R.pivot_table(index="dataset", columns="method", values="auroc").round(3)
wide_p = R.pivot_table(index="dataset", columns="method", values="auprc").round(3)
wide.to_csv(f"{OUT}/auroc_wide.csv")
wide_p.to_csv(f"{OUT}/auprc_wide.csv")
# per-dataset wins of each method against production
base = R[R.method == "prod"].set_index("dataset")
win = []
for m in R.method.unique():
    a = R[(R.method == m) & ~R.synthetic].set_index("dataset")
    j = a.join(base, rsuffix="_prod", how="inner")
    if len(j):
        win.append(dict(method=m, datasets=len(j), auroc_delta_mean=(j.auroc - j.auroc_prod).mean(),
                        auprc_delta_mean=(j.auprc - j.auprc_prod).mean(), auprc_better=(j.auprc > j.auprc_prod).sum()))
pd.DataFrame(win).round(4).to_csv(f"{OUT}/vs_prod.csv", index=False)

# ---- cost per 10k cells
cost = R.dropna(subset=["seconds"]).assign(sec_per_10k=lambda x: x.seconds / x.cells * 1e4).groupby("method").sec_per_10k.median()
cost.round(2).to_csv(f"{OUT}/cost_sec_per_10k_cells.csv")

# ---- homotypic blind spot
hom = []
if "kang21_homsim" in feats:
    F, y, d = feats["kang21_homsim"]
    typ = d["x_dbl_type"]
    for m, s in F.items():
        if m == "genes":
            continue
        k = int((typ > 0).sum())
        top = np.zeros(len(s), bool)
        top[np.argsort(-s, kind="stable")[:k]] = True
        hom.append(dict(method=m, recall_heterotypic=top[typ == 2].mean(), recall_homotypic=top[typ == 1].mean(),
                        auroc_het_vs_singlet=roc_auc_score(np.r_[np.zeros((typ == 0).sum()), np.ones((typ == 2).sum())],
                                                           np.r_[s[typ == 0], s[typ == 2]]),
                        auroc_hom_vs_singlet=roc_auc_score(np.r_[np.zeros((typ == 0).sum()), np.ones((typ == 1).sum())],
                                                           np.r_[s[typ == 0], s[typ == 1]])))
pd.DataFrame(hom).round(3).to_csv(f"{OUT}/homotypic_sim.csv", index=False)
# real genotype doublets in Kang lane 2.1, split by whether the atlas network calls them one confident type
hk = []
if "kang21" in feats:
    F, y, d = feats["kang21"]
    am = np.load(f"{SCORES}/kang21/ct_maxprob.npy") if os.path.exists(f"{SCORES}/kang21/ct_maxprob.npy") else None
    if am is not None:
        conf = (1 - am) >= 0.8
        k = int(y.sum())
        allF = dict(F, vireo_geno=d["x_vireo_geno"], prod_geno=d["x_prod_geno"])
        for m, s in allF.items():
            if m == "genes":
                continue
            top = np.zeros(len(s), bool)
            top[np.argsort(-s, kind="stable")[:k]] = True
            hk.append(dict(method=m, doublets_one_confident_type=int((y & conf).sum()), recall_one_type=top[(y == 1) & conf].mean(),
                           doublets_mixed=int((y & ~conf).sum()), recall_mixed=top[(y == 1) & ~conf].mean()))
pd.DataFrame(hk).round(3).to_csv(f"{OUT}/kang_homotypic_proxy.csv", index=False)

def md(df, index=True):
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


with open(f"{OUT}/results.md", "w") as f:
    pd.set_option("display.width", 250)
    f.write("## datasets\n\n" + md(info.round(3), False) + "\n\n")
    f.write("## mean / median over real-label datasets\n\n" + md(summ) + "\n\n")
    f.write("## AUROC\n\n" + md(wide) + "\n\n## AUPRC\n\n" + md(wide_p) + "\n\n")
    f.write("## vs production (real datasets)\n\n" + md(pd.DataFrame(win).round(4), False) + "\n\n")
    f.write("## cost, median seconds per 10k cells\n\n" + md(cost.round(2)) + "\n\n")
    f.write("## homotypic simulation (Kang singlets + 5% homotypic + 5% heterotypic)\n\n" + md(pd.DataFrame(hom).round(3), False))
    f.write("\n\n## Kang 2.1 genotype doublets: one confident atlas type vs mixed\n\n" + md(pd.DataFrame(hk).round(3), False) + "\n")
print(open(f"{OUT}/results.md").read())
