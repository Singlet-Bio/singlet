"""Cached per-cell scores of the latent-qc doublet study ($LQC/dbl, Clipper): 17 real-label datasets, one study each (the six
Kang lanes form one study, as do the five MULTI-seq sets and the two 10x hgmm sets)."""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
SC, DATA = f"{LQC}/dbl/scores", f"{LQC}/dbl/data"
STUDY = {"pbmc-1A-dm": "kang", "pbmc-1B-dm": "kang", "pbmc-1C-dm": "kang", "pbmc-2ctrl-dm": "kang", "pbmc-2stim-dm": "kang",
         "kang21": "kang", "J293t-dm": "kang293t", "pbmc-ch": "stoeckius", "cline-ch": "stoeckius", "mkidney-ch": "mkh",
         "HEK-HMEC-MULTI": "mcg", "HMEC-orig-MULTI": "mcg", "HMEC-rep-MULTI": "mcg", "pdx-MULTI": "mcg", "nuc-MULTI": "mcg",
         "hm-6k": "hgmm", "hm-12k": "hgmm"}
NAMES = list(STUDY)
ATL = ["atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls"]
SETS = {"prod_atlas": ["umi", "genes", "prod"] + ATL,
        "prod_atlas_scrub": ["umi", "genes", "prod", "scrublet"] + ATL,
        "prod_knn_scrub": ["umi", "genes", "prod", "scrublet", "atlas_knn"],
        "prod_scrub": ["umi", "genes", "prod", "scrublet"],
        "prod_knn": ["umi", "genes", "prod", "atlas_knn"],
        "prod_knn_ent": ["umi", "genes", "prod", "atlas_knn", "ct_entropy"],
        "prod_knn_ent_scrub": ["umi", "genes", "prod", "atlas_knn", "ct_entropy", "scrublet"]}
RANKAVG = {"ravg_prod_scrub": ["prod", "scrublet"], "ravg_prod_knn": ["prod", "atlas_knn"],
           "ravg_prod_knn_scrub": ["prod", "atlas_knn", "scrublet"], "ravg_prod_knn_scrub_umi": ["prod", "atlas_knn", "scrublet", "umi"]}


def load(n):
    z = np.load(f"{DATA}/{n}/ds.npz")
    import scipy.sparse as sp
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    F = {"umi": np.log1p(np.asarray(X.sum(1)).ravel()), "genes": np.log1p(np.diff(X.indptr))}
    for m in ATL + ["prod", "scrublet"]:
        F[m] = np.load(f"{SC}/{n}/{m}.npy")
    p = f"{SC}/{n}/scdblfinder.tsv"
    F["scdblfinder"] = pd.read_csv(p, sep="\t").score.values
    return F, z["label"].astype(int)


def q(F, cols):
    return np.column_stack([rankdata(np.nan_to_num(F[c])) / len(F[c]) for c in cols])


