"""Parity of the batch job's code with the cached study scores, on Kang 2018 lane 2.1 as Singlet counts it (ds.npz of the latent-qc
doublet track, 13,737 cells): runs singlet.doublets (dbl_run, atlas_features, packaged combiner) on the counts and compares each
feature with the cached one. The combiner here was fitted on this lane too, so its AUPRC is a parity number, not a result; the
leave-one-study-out numbers are in bench/doublets/train_eval.py.
Usage: parity_kang.py MODEL.pt CENTROIDS.npz DBL_RUN OUT.json      (PYTHONPATH: bench/doublets and the python tree)
"""
import json
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from singlet.celltype_nn import CellTypePredictor
from singlet.doublets import score as S
from singlet.doublets.atlas import atlas_features
from singlet.doublets.combiner import Combiner, basic_features
from lqc_scores import DATA, SC

import scipy.sparse as sp

model, cen, exe, out = sys.argv[1:5]
z = np.load(f"{DATA}/kang21/ds.npz")
X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"])).astype(np.float32)
y = z["label"].astype(int)
genes = z["genes"].astype(str)
pred = CellTypePredictor.load(model)
t = {}
t0 = time.perf_counter()
F = basic_features(X)
F["prod"] = S.prod_score(X, exe, 8)
t["prod_s"] = time.perf_counter() - t0
F.update(atlas_features(X, genes, "human", pred, np.load(cen)["M"], timings=t))
s = Combiner.load().score(F)
t["total_s"] = time.perf_counter() - t0
r = dict(cells=int(X.shape[0]), timings={k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()},
         spearman_vs_cached={m: round(float(spearmanr(F[m], np.load(f"{SC}/kang21/{m}.npy"))[0]), 4)
                             for m in ("prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls")},
         in_sample_auroc=float(roc_auc_score(y, s)), in_sample_auprc=float(average_precision_score(y, s)))
print(json.dumps(r, indent=1))
json.dump(r, open(out, "w"), indent=1)
