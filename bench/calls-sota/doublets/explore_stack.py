"""Development-only exploration of the doublet stacker (D-163): single inputs and stack variants, leave-one-study-out, AUPRC and
paired difference to scDblFinder (study-clustered bootstrap). Real labels of the 17 development datasets only. Usage: explore_stack.py OUT.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
WK = "/mnt/projects/debruinz_project/debruinz/singlet"
STUDY = {"pbmc-1A-dm": "kang", "pbmc-1B-dm": "kang", "pbmc-1C-dm": "kang", "pbmc-2ctrl-dm": "kang", "pbmc-2stim-dm": "kang",
         "kang21": "kang", "J293t-dm": "kang293t", "pbmc-ch": "stoeckius", "cline-ch": "stoeckius", "mkidney-ch": "mkh",
         "HEK-HMEC-MULTI": "mcg", "HMEC-orig-MULTI": "mcg", "HMEC-rep-MULTI": "mcg", "pdx-MULTI": "mcg", "nuc-MULTI": "mcg",
         "hm-6k": "hgmm", "hm-12k": "hgmm"}
NAMES = list(STUDY)
ref = pd.read_csv(f"{LQC}/dbl/eval/auprc_wide.csv", index_col=0)["scdblfinder"]

def load(n):
    import scipy.sparse as sp
    z = np.load(f"{LQC}/dbl/data/{n}/ds.npz")
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    F = {"umi": np.log1p(np.asarray(X.sum(1)).ravel()), "genes": np.log1p(np.diff(X.indptr))}
    for m in ["prod", "scrublet", "atlas_knn", "ct_entropy", "sdfl", "sdflc", "sdfl_k3", "sdfl_k8"]:
        F[m] = np.load(f"{LQC}/dbl/scores/{n}/{m}.npy")
    p = np.load(f"{WK}/doublet_v2/pkgfeat/{n}.npz")
    for m in ("sdfl", "sdflc", "atlas_knn", "atlas_entropy"):
        F[m] = p[m]
    F["scdblfinder"] = pd.read_csv(f"{LQC}/dbl/scores/{n}/scdblfinder.tsv", sep="\t").score.values
    return F, z["label"].astype(int)

D = {n: load(n) for n in NAMES}
R = lambda F, cols: np.column_stack([rankdata(np.nan_to_num(F[c])) / len(F[c]) for c in cols])
BASE = ["umi", "genes", "prod", "sdfl", "sdflc", "atlas_knn", "atlas_entropy"]

def ci(ap):
    d = pd.Series(ap) - ref.reindex(list(ap)); stu = pd.Series(STUDY)
    rng = np.random.default_rng(0)
    g = d.groupby(stu.reindex(d.index)).mean()
    b2 = [g.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(5000)]
    b1 = [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    q = lambda b: [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    return dict(diff=round(float(d.mean()), 4), ci=q(b1), ci_study=q(b2))

def run(name, fitpred):
    ap = {}
    for n in NAMES:
        tr = [m for m in NAMES if STUDY[m] != STUDY[n]]
        ap[n] = float(average_precision_score(D[n][1], fitpred(tr, n)))
    out = dict(mean=round(float(np.mean(list(ap.values()))), 4), **ci(ap), per=ap)
    print(name, {k: v for k, v in out.items() if k != "per"}, flush=True)
    return out

def w_of(tr):
    w = np.concatenate([np.full(len(D[m][1]), 1 / len(D[m][1])) for m in tr]); return w * len(w) / len(tr)

def hgb(cols, **kw):
    P = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0); P.update(kw)
    def f(tr, n):
        c = HistGradientBoostingClassifier(**P).fit(np.vstack([R(D[m][0], cols) for m in tr]), np.concatenate([D[m][1] for m in tr]), sample_weight=w_of(tr))
        return c.predict_proba(R(D[n][0], cols))[:, 1]
    return f

def lr(cols, C=1.0):
    def f(tr, n):
        c = LogisticRegression(C=C, max_iter=500).fit(np.vstack([R(D[m][0], cols) for m in tr]), np.concatenate([D[m][1] for m in tr]), sample_weight=w_of(tr))
        return c.predict_proba(R(D[n][0], cols))[:, 1]
    return f

def ravg(cols):
    return lambda tr, n: R(D[n][0], cols).mean(1)

PORT = "/mnt/projects/debruinz_project/debruinz/singlet/bpcalls/scrub_port"
for n in NAMES:
    D[n][0]["scrub"] = np.load(f"{PORT}/{n}.npy")
res = {}
S7 = BASE + ["scrub"]
for it, lv, l2 in [(100, 6, 5.0), (60, 6, 5.0), (100, 4, 5.0), (100, 6, 20.0), (150, 4, 10.0), (60, 4, 10.0), (200, 6, 20.0)]:
    nm = f"small_{it}_{lv}_{l2}"
    res[nm] = run(nm, hgb(S7, max_iter=it, max_leaf_nodes=lv, l2_regularization=l2))
json.dump(res, open(sys.argv[1], "w"), indent=1)
