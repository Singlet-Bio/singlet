# SPDX-License-Identifier: MIT
"""Leave-one-study-out stacker (same recipe as bench/latent_qc/doublets/evaluate.py) with and without the sdfl feature.
  python stack.py OUT.csv   -> per-dataset AUPRC of combo_prod_atlas (replication) and combo_prod_atlas + sdfl / sdfl+sdflc, paired bootstrap."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dbl_common import SCORES, DATA
STUDY = {"pbmc-1A-dm": "kang2018", "pbmc-1B-dm": "kang2018", "pbmc-1C-dm": "kang2018", "pbmc-2ctrl-dm": "kang2018", "pbmc-2stim-dm": "kang2018",
         "kang21": "kang2018", "J293t-dm": "kang2018_293t", "pbmc-ch": "stoeckius2018", "cline-ch": "stoeckius2018", "mkidney-ch": "mkidney_hashing",
         "HEK-HMEC-MULTI": "mcginnis2019", "HMEC-orig-MULTI": "mcginnis2019", "HMEC-rep-MULTI": "mcginnis2019", "pdx-MULTI": "mcginnis2019",
         "nuc-MULTI": "mcginnis2019", "hm-6k": "10x_hgmm", "hm-12k": "10x_hgmm"}
BASE = ["umi", "genes", "prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls"]
SETS = {"combo_prod_atlas": BASE, "combo_pa_sdfl": BASE + ["sdfl"], "combo_pa_sdfl2": BASE + ["sdfl", "sdflc"], "combo_noatlas_sdfl": ["umi", "genes", "prod", "sdfl"]}
names = [n for n in STUDY]
D = {}
for n in names:
    z = np.load(f"{DATA}/{n}/ds.npz"); X_ip = z["X_indptr"]; F = {"umi": np.log1p(np.asarray(0)), }
    import scipy.sparse as sp
    X = sp.csr_matrix((z["X_data"], z["X_indices"], X_ip), shape=tuple(z["X_shape"]))
    F = {"umi": np.log1p(np.asarray(X.sum(1)).ravel()), "genes": np.log1p(np.diff(X.indptr))}
    for m in ("prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls", "sdfl", "sdflc"):
        p = f"{SCORES}/{n}/{m}.npy"
        if os.path.exists(p): F[m] = np.load(p)
    D[n] = (F, z["label"].astype(int))
q = lambda F, cols: np.column_stack([rankdata(F[c]) / len(F[c]) for c in cols])
rows = []
for sname, cols in SETS.items():
    for n in names:
        tr = [m for m in names if STUDY[m] != STUDY[n] and all(c in D[m][0] for c in cols)]
        Xtr = np.vstack([q(D[m][0], cols) for m in tr]); ytr = np.concatenate([D[m][1] for m in tr])
        w = np.concatenate([np.full(len(D[m][1]), 1 / len(D[m][1])) for m in tr])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(Xtr, ytr, sample_weight=w * len(w) / len(tr))
        s = clf.predict_proba(q(D[n][0], cols))[:, 1]
        rows.append(dict(dataset=n, method=sname, auprc=average_precision_score(D[n][1], s)))
R = pd.DataFrame(rows).pivot(index="dataset", columns="method", values="auprc")
ST = pd.read_csv(f"{os.path.dirname(SCORES)}/eval/auprc_wide.csv", index_col=0)
R["scdblfinder"] = ST.scdblfinder.reindex(R.index)
R.round(4).to_csv(sys.argv[1]); print(R.round(3).to_string()); print(R.mean().round(4).to_dict())
rng = np.random.default_rng(0); stu = pd.Series(STUDY)
def boot(d):
    g = d.groupby(stu.reindex(d.index)).mean()
    return [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [g.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(5000)]
for a, b in (("combo_pa_sdfl", "combo_prod_atlas"), ("combo_pa_sdfl2", "combo_prod_atlas"), ("combo_pa_sdfl", "scdblfinder"), ("combo_pa_sdfl2", "scdblfinder"), ("combo_prod_atlas", "scdblfinder")):
    d = (R[a] - R[b]); b1, b2 = boot(d)
    print(f"{a} - {b}: {d.mean():+.4f} CI [{np.percentile(b1,2.5):+.4f},{np.percentile(b1,97.5):+.4f}] study-clustered [{np.percentile(b2,2.5):+.4f},{np.percentile(b2,97.5):+.4f}] wins {(d>0).sum()}/17; hgmm {d[['hm-6k','hm-12k']].mean():+.3f}")
