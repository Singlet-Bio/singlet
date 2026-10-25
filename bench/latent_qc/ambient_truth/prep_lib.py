"""Download (10x Cell Ranger barnyard matrices) and convert to the dataset layout of atlib.py.  Run via `rt python -I`.
Usage: prep_lib.py LIB     LIB in hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2 | hgmm1k_v3_singlet"""
import gzip
import os
import subprocess
import sys
import tarfile

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlib import DATA, LQC  # noqa: E402

B = "https://cf.10xgenomics.com/samples/cell-exp"
H5 = {
    "hgmm1k_v3": "3.0.0/hgmm_1k_v3/hgmm_1k_v3_{}_feature_bc_matrix.h5",
    "hgmm5k_v3": "3.0.0/hgmm_5k_v3/hgmm_5k_v3_{}_feature_bc_matrix.h5",
    "hgmm10k_v3": "3.0.0/hgmm_10k_v3/hgmm_10k_v3_{}_feature_bc_matrix.h5",
    "hgmm1k_v2": "3.0.0/hgmm_1k_v2/hgmm_1k_v2_{}_feature_bc_matrix.h5",
}
TAR = {"hgmm6k_v2": "2.1.0/hgmm_6k/hgmm_6k_{}_gene_bc_matrices.tar.gz", "hgmm12k_v2": "2.1.0/hgmm_12k/hgmm_12k_{}_gene_bc_matrices.tar.gz"}
DL = f"{LQC}/ambient_truth/dl"


def fetch(url, path):
    if not os.path.exists(path):
        subprocess.check_call(["curl", "-sSfL", "--retry", "3", "-o", path + ".part", url])
        os.rename(path + ".part", path)
    return path


def read_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = m["shape"][:]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(shape))
        ids = m["features/id"][:].astype(str)
        gen = m["features/genome"][:].astype(str) if "genome" in m["features"] else None
        bc = m["barcodes"][:].astype(str)
    return X, ids, gen, bc


def read_tar(path, kind):
    d = path + ".x"
    os.makedirs(d, exist_ok=True)
    if not os.listdir(d):
        with tarfile.open(path) as t:
            t.extractall(d, filter="data")
    mx = [os.path.join(r, "matrix.mtx") for r, _, fs in os.walk(d) if "matrix.mtx" in fs]
    assert len(mx) == 1, mx
    r = os.path.dirname(mx[0])
    X = scipy.io.mmread(mx[0]).tocsc()
    genes = pd.read_csv(os.path.join(r, "genes.tsv"), sep="\t", header=None)
    bc = pd.read_csv(os.path.join(r, "barcodes.tsv"), sep="\t", header=None)[0].astype(str).values
    return X, genes[0].astype(str).values, None, bc


def norm_bc(b):
    return pd.Series(b).str.replace(r"-\d+$", "", regex=True).values


def save(lib, X, ids, mouse, bc_all, cell_bc):
    out = f"{DATA}/{lib}"
    os.makedirs(out, exist_ok=True)
    tot = np.asarray(X.sum(0)).ravel()
    keep = np.where(tot >= 1)[0]
    X = X[:, keep].tocsc()
    X.data = X.data.astype(np.uint32)
    bc = norm_bc(bc_all)[keep]
    pos = pd.Series(np.arange(len(bc)), index=bc)
    cb = pd.Series(norm_bc(cell_bc))
    ci = pos.reindex(cb.values)
    print(lib, "cells in filtered:", len(cb), "found in raw>=1UMI:", int(ci.notna().sum()))
    ci = ci.dropna().astype(int).values
    sp.save_npz(f"{out}/X.npz", X)
    np.save(f"{out}/barcodes.npy", bc.astype(object), allow_pickle=True)
    np.save(f"{out}/gene_ids.npy", np.asarray(ids).astype(object), allow_pickle=True)
    np.save(f"{out}/gene_mouse.npy", np.asarray(mouse, dtype=bool))
    np.save(f"{out}/cell_idx.npy", ci)
    print(lib, "genes", X.shape[0], "mouse genes", int(np.sum(mouse)), "barcodes>=1UMI", X.shape[1], "nnz", X.nnz)


def main(lib):
    os.makedirs(DL, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    if lib == "hgmm1k_v3_singlet":
        # Singlet's own STARsolo counts (previous benchmark inputs); cells = the 880 Singlet called
        W = "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/ambient/work"
        sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modalities/ambient")
        from common import read_mtx_dir
        X, gid, _, rbc = read_mtx_dir(f"{W}/raw")
        meta = pd.read_csv(f"{W}/cells.csv")
        mouse = np.char.startswith(gid.astype(str), "ENSMUSG")
        save(lib, X.tocsc(), gid, mouse, rbc, meta.barcode.values)
        return
    if lib in H5:
        raw = read_h5(fetch(f"{B}/{H5[lib].format('raw')}", f"{DL}/{lib}_raw.h5"))
        fil = read_h5(fetch(f"{B}/{H5[lib].format('filtered')}", f"{DL}/{lib}_filtered.h5"))
    else:
        raw = read_tar(fetch(f"{B}/{TAR[lib].format('raw')}", f"{DL}/{lib}_raw.tgz"), "raw")
        fil = read_tar(fetch(f"{B}/{TAR[lib].format('filtered')}", f"{DL}/{lib}_filtered.tgz"), "filtered")
    X, ids, gen, bc = raw
    mouse = np.array(["ENSMUSG" in s for s in ids])
    if gen is not None:
        print("genome labels", pd.Series(gen).value_counts().to_dict())
        mg = np.array([("mm" in g.lower()) for g in gen])
        assert (mg == mouse).all(), "genome label vs ENSMUSG prefix disagree"
    save(lib, X, ids, mouse, bc, fil[3])


if __name__ == "__main__":
    main(sys.argv[1])
