"""Do cheap non-R features close the 0.003 between combo_prod_atlas (0.557) and scDblFinder (0.559)?

Reads the cached per-cell scores of the latent-qc doublet study ($LQC/dbl/scores/<dataset>/<method>.npy), all real experimental
labels, and compares leave-one-study-out combiners (same recipe as bench/latent_qc/doublets/evaluate.py) over feature sets:
production score, atlas features, Scrublet (python), and unsupervised rank averages. Usage: cheap_features.py OUT.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from lqc_scores import NAMES, RANKAVG, SETS, STUDY, load, q


def met(y, s):
    return average_precision_score(y, s), roc_auc_score(y, s)


D = {n: load(n) for n in NAMES}
rows = []
for n in NAMES:
    F, y = D[n]
    rows.append((n, "prod", *met(y, F["prod"])))
    rows.append((n, "scdblfinder", *met(y, F["scdblfinder"])))
    for k, cols in RANKAVG.items():
        rows.append((n, k, *met(y, q(F, cols).mean(1))))
    for k, cols in SETS.items():
        tr = [m for m in NAMES if STUDY[m] != STUDY[n]]
        Xtr = np.vstack([q(D[m][0], cols) for m in tr]); ytr = np.concatenate([D[m][1] for m in tr])
        w = np.concatenate([np.full(len(D[m][1]), 1 / len(D[m][1])) for m in tr]) * len(ytr) / len(tr)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                             random_state=0).fit(Xtr, ytr, sample_weight=w)
        rows.append((n, "combo_" + k, *met(y, clf.predict_proba(q(F, cols))[:, 1])))
R = pd.DataFrame(rows, columns=["dataset", "method", "auprc", "auroc"])
R.to_csv(sys.argv[1], index=False)
S = R.groupby("method")[["auprc", "auroc"]].mean().round(4).sort_values("auprc", ascending=False)
print(S.to_string())
kang = R[R.dataset.isin(["kang21", "pbmc-1A-dm", "pbmc-1B-dm", "pbmc-1C-dm", "pbmc-2ctrl-dm", "pbmc-2stim-dm"])]
print(kang.groupby("method").auprc.mean().round(4).sort_values(ascending=False).to_string())
