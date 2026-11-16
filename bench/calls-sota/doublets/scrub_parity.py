"""Port check (D-163): doublet_scrub.score against the cached Scrublet package scores of the 17 development datasets, AUPRC and rank
correlation. Usage: scrub_parity.py OUT_DIR   (needs PYTHONPATH=<repo>/python and the LQC data)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from singlet.features import doublet_scrub as S
from singlet.features.doublet import STUDY

LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rep = {}
for n in STUDY:
    z = np.load(f"{LQC}/dbl/data/{n}/ds.npz")
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    y = z["label"].astype(int)
    t = time.perf_counter()
    s = S.score(X)
    w = time.perf_counter() - t
    np.save(out / f"{n}.npy", s)
    ref = np.load(f"{LQC}/dbl/scores/{n}/scrublet.npy")
    rep[n] = dict(n=int(X.shape[0]), auprc_port=float(average_precision_score(y, s)), auprc_pkg=float(average_precision_score(y, ref)),
                  spearman=float(spearmanr(s, ref)[0]), wall_s=round(w, 1))
    print(n, rep[n], flush=True)
json.dump(rep, open(out / "parity.json", "w"), indent=1)
print("mean auprc port", np.mean([v["auprc_port"] for v in rep.values()]), "pkg", np.mean([v["auprc_pkg"] for v in rep.values()]))
