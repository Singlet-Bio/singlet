"""Shared loaders for the ambient / barnyard benchmark (hgmm_1k_v3)."""
import gzip, os
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp

W = "/mnt/projects/debruinz_project/debruinz/singlet"
A = f"{W}/modality-bench/ambient"
SINGLET = f"{A}/singlet"
WORK = f"{A}/work"
CR_H5 = f"{W}/pipeline-benchmark/data/hgmm1k_v3/cellranger_hgmm_1k_v3_filtered_feature_bc_matrix.h5"


def read_mtx_dir(d):
    """10x-style directory -> (genes x barcodes csc, feature ids, feature names, barcodes)."""
    def op(n):
        p = os.path.join(d, n)
        return gzip.open(p + ".gz", "rt") if os.path.exists(p + ".gz") else open(p)
    with op("matrix.mtx") as f:
        X = scipy.io.mmread(f).tocsc()
    feats = pd.read_csv(op("features.tsv"), sep="\t", header=None)
    bcs = pd.read_csv(op("barcodes.tsv"), sep="\t", header=None)[0].astype(str).str.replace(r"-1$", "", regex=True).values
    return X, feats[0].astype(str).values, feats[1].astype(str).values, bcs


def species_of(ids):
    """0 human, 1 mouse (Ensembl prefix; the joint reference keeps ENSG / ENSMUSG ids)."""
    return np.where(pd.Series(ids).str.startswith("ENSMUSG").values, 1, 0)
