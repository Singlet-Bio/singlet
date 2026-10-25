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
wide = R[~R.synthetic].pivot_table(index="dataset", columns="method", values="auroc").round(3)
wide_p = R[~R.synthetic].pivot_table(index="dataset", columns="method", values="auprc").round(3)
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

# ---- homotypic blind spot, on real labels only: genotype / hashtag doublets of the PBMC pools split by what the atlas network
# sees. "one lineage" = >= 0.9 of the network's mass on one lineage (T, NK, B, monocyte, ...): an expression method sees one cell.
hom_rows, comp = [], []
PBMC = [n for n in feats if n not in SYNTH and (n.startswith("pbmc") or n == "kang21")]
pool = {}
for n in PBMC:
    p = f"{SCORES}/{n}/lineage_prob.npy"
    if not os.path.exists(p):
        continue
    F, y, d = feats[n]
    L = np.load(p)
    srt = np.sort(L, 1)
    cls = np.where(srt[:, -1] >= 0.9, "one_lineage", np.where(srt[:, -2] >= 0.2, "two_lineages", "unclear"))
    sing = (y == 0) & (srt[:, -1] >= 0.9)
    frac = np.bincount(L[sing].argmax(1), minlength=L.shape[1]) / max(sing.sum(), 1)
    comp.append(dict(dataset=n, doublets=int(y.sum()), one_lineage=float(np.mean(cls[y == 1] == "one_lineage")),
                     two_lineages=float(np.mean(cls[y == 1] == "two_lineages")),
                     singlets_one_lineage=float(np.mean(cls[y == 0] == "one_lineage")),
                     expected_same_lineage_pairs=float((frac ** 2).sum())))
    allF = dict(F)
    for k in ("x_vireo_geno", "x_prod_geno"):
        if k in d:
            allF[k[2:]] = d[k]
    for m, sc in allF.items():
        if m == "genes":
            continue
        k = int(y.sum())
        top = np.zeros(len(sc), bool)
        top[np.argsort(-sc, kind="stable")[:k]] = True
        for c in ("one_lineage", "two_lineages"):
            sel = (y == 1) & (cls == c)
            if comp[-1]["singlets_one_lineage"] < 0.25:  # network not confident on this matrix: skip
                continue
            a_ = pool.setdefault((m, n, c), [0, 0, 0])
            a_[0] += int(top[sel].sum()); a_[1] += int(sel.sum()); a_[2] += 1
for (m, n, c), (tp, nn_, nd) in pool.items():
    hom_rows.append(dict(method=m, dataset=n, cls=c, doublets=nn_, recall_at_truth_rate=tp / max(nn_, 1)))
hom = pd.DataFrame(hom_rows)
if len(hom):
    hom = hom.pivot_table(index="method", columns=["dataset", "cls"], values="recall_at_truth_rate").round(3)
    hom.columns = [f"{a}:{b}" for a, b in hom.columns]
    hom = hom.reset_index()
    hom.columns.name = None
hom.to_csv(f"{OUT}/homotypic_real.csv", index=False)
comp = pd.DataFrame(comp).round(3)
comp.to_csv(f"{OUT}/homotypic_composition.csv", index=False)

# ---- per-study means (the Kang lanes are six datasets of one study) and paired tests against production
from scipy.stats import wilcoxon

Rr = R[~R.synthetic].assign(study=lambda x: x.dataset.map(lambda n: STUDY.get(n, n)))
per_study = Rr.groupby(["method", "study"])[["auroc", "auprc"]].mean().groupby("method").mean().round(3)
per_study.columns = ["auroc_mean_over_studies", "auprc_mean_over_studies"]
tests = []
for m in Rr.method.unique():
    j = Rr[Rr.method == m].set_index("dataset")[["auprc", "auroc"]].join(base[["auprc", "auroc"]], rsuffix="_p", how="inner")
    if len(j) >= 5 and m != "prod":
        dd = (j.auprc - j.auprc_p).values
        tests.append(dict(method=m, n=len(j), auprc_delta_median=np.median(dd), better=int((dd > 0).sum()),
                          wilcoxon_p=wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0))
tests = pd.DataFrame(tests).round(4)
per_study.to_csv(f"{OUT}/per_study_means.csv")
tests.to_csv(f"{OUT}/paired_vs_prod.csv", index=False)


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
    NL = chr(10)
    f.write("## per-study means" + NL * 2 + md(per_study) + NL * 2 + "## paired AUPRC vs production" + NL * 2 + md(tests, False) + NL * 2)
    f.write("## real doublets by atlas lineage (PBMC pools)" + NL * 2 + md(comp, False) + NL * 2 + md(hom, False) + NL)
print(open(f"{OUT}/results.md").read())
