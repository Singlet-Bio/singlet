"""Train the shipped combiner and produce the leave-one-study-out evidence for the bench/public `doublets` capability.

Inputs: the cached per-cell scores of the latent-qc doublet study (lqc_scores.py), real experimental labels only.
Writes to OUT:
  combiner_prod_atlas.npz   the combiner fitted on all 17 datasets (each dataset weighs the same); packaged with singlet.doublets
  doublets_cells.parquet    one row per cell: dataset, study, label, and the scores of production, scDblFinder, Scrublet and the
                            combiner (leave-one-study-out: the dataset's own study is never in its training set)
  doublets_datasets.tsv     per dataset: study, cells, doublets, truth
  parity.json               numpy evaluator vs scikit-learn on the training set
Usage: train_eval.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from singlet.doublets.combiner import FEATURES, PARAMS, Combiner, qmatrix  # noqa: E402

from lqc_scores import NAMES, STUDY, load  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = {"combo": list(FEATURES), "combo_scrublet": list(FEATURES) + ["scrublet"]}
TRUTH = {
    "kang21": "demuxlet genotype doublets, lane 2.1 on Singlet counts",
    "pbmc-1A-dm": "demuxlet genotype doublets", "pbmc-1B-dm": "demuxlet genotype doublets",
    "pbmc-1C-dm": "demuxlet genotype doublets", "pbmc-2ctrl-dm": "demuxlet genotype doublets",
    "pbmc-2stim-dm": "demuxlet genotype doublets", "J293t-dm": "demuxlet genotype doublets (two cell lines)",
    "pbmc-ch": "cell-hashing HTO doublets", "cline-ch": "cell-hashing HTO doublets", "mkidney-ch": "cell-hashing HTO doublets",
    "HEK-HMEC-MULTI": "MULTI-seq barcode doublets", "HMEC-orig-MULTI": "MULTI-seq barcode doublets",
    "HMEC-rep-MULTI": "MULTI-seq barcode doublets", "pdx-MULTI": "MULTI-seq barcode doublets",
    "nuc-MULTI": "MULTI-seq barcode doublets", "hm-6k": "human + mouse species-mixing doublets",
    "hm-12k": "human + mouse species-mixing doublets",
}


def fit_sk(Q, y, w):
    return HistGradientBoostingClassifier(**PARAMS).fit(Q, y, sample_weight=w)


def weights(ds):
    return np.concatenate([np.full(len(D[m][1]), 1 / len(D[m][1])) for m in ds]) * sum(len(D[m][1]) for m in ds) / len(ds)


D = {n: load(n) for n in NAMES}
cells = []
for n in NAMES:
    F, y = D[n]
    cells.append(pd.DataFrame({"dataset": n, "study": STUDY[n], "i": np.arange(len(y)), "label": y, "prod": F["prod"],
                               "scdblfinder": F["scdblfinder"], "scrublet": F["scrublet"]}))
cells = pd.concat(cells, ignore_index=True)
for name, cols in VARIANTS.items():
    pred = {}
    for n in NAMES:
        tr = [m for m in NAMES if STUDY[m] != STUDY[n]]
        clf = fit_sk(np.vstack([qmatrix(D[m][0], cols) for m in tr]), np.concatenate([D[m][1] for m in tr]), weights(tr))
        pred[n] = clf.predict_proba(qmatrix(D[n][0], cols))[:, 1]
    cells[name] = np.concatenate([pred[n] for n in NAMES])
    print(name, "LOSO done", flush=True)
cells.to_parquet(OUT / "doublets_cells.parquet", index=False)
pd.DataFrame([{"dataset": n, "study": STUDY[n], "cells": len(D[n][1]), "doublets": int(D[n][1].sum()), "truth": TRUTH[n]}
              for n in NAMES]).to_csv(OUT / "doublets_datasets.tsv", sep="\t", index=False)

cols = list(FEATURES)
Q = np.vstack([qmatrix(D[m][0], cols) for m in NAMES])
y = np.concatenate([D[m][1] for m in NAMES])
sk = fit_sk(Q, y, weights(NAMES))
comb = Combiner.from_sklearn(sk, cols, "gbm300-17ds-1")
comb.save(OUT / "combiner_prod_atlas.npz")
back = Combiner.load(OUT / "combiner_prod_atlas.npz")
diff = float(np.abs(back.predict_proba(Q) - sk.predict_proba(Q)[:, 1]).max())
json.dump({"max_abs_diff_numpy_vs_sklearn": diff, "trees": int(len(comb.a["tree_offset"]) - 1), "train_cells": int(len(y)),
           "datasets": NAMES}, open(OUT / "parity.json", "w"), indent=1)
print("parity", diff)
