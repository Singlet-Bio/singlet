# SPDX-License-Identifier: MIT
"""Prepare CellBender inputs (10x mtx directories, gene counts only, barcodes with >= 1 UMI) and expected-cell parameters for the held-out libraries.
python -P cb_prep.py OUT_ROOT  (reads the held-out inputs under bench/calls-sota/cell_calling)"""
import gzip, os, sys
import h5py, numpy as np, scipy.io, scipy.sparse as sp
out = sys.argv[1]
S = "/mnt/projects/debruinz_project/debruinz/singlet/bench/calls-sota/cell_calling"
def write(name, G, bc, genes):
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True)
    tot = np.sort(np.asarray(G.sum(axis=0)).ravel())[::-1]
    exp = int((tot >= np.quantile(tot[:3000], 0.99) / 10.0).sum())
    keep = np.nonzero(np.asarray(G.sum(axis=0)).ravel() >= 1)[0]
    G = G[:, keep].tocsc(); bc = [bc[i] for i in keep]
    with gzip.open(f"{d}/matrix.mtx.gz", "wb") as f: scipy.io.mmwrite(f, G, field="integer")
    with gzip.open(f"{d}/barcodes.tsv.gz", "wt") as f: f.write("\n".join(bc) + "\n")
    with gzip.open(f"{d}/features.tsv.gz", "wt") as f: f.write("\n".join(f"{g}\t{g}\tGene Expression" for g in genes) + "\n")
    open(f"{d}/params.txt", "w").write(f"{exp} {max(3 * exp, 10000)}\n")
    print(name, G.shape, "expected", exp, flush=True)
with h5py.File(f"{S}/heldout_cellplex10k/raw.h5", "r") as f:
    m = f["matrix"]; X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
    ft = np.array([s.decode() for s in m["features"]["feature_type"][:]]); ids = [s.decode() for s in m["features"]["id"][:]]
    bc = [b.decode() for b in m["barcodes"][:]]
    g = np.nonzero(ft == "Gene Expression")[0]
    write("cellplex10k", X[g, :].tocsc(), bc, [ids[i] for i in g])
P = f"{S}/probe_nuc"
M = sp.csc_matrix(scipy.io.mmread(f"{P}/nuc.mtx.gz"))
bc = [l.strip().split("\t")[0] for l in open(f"{P}/POC_nuc_barcodes.tsv")]
genes = [l.strip().split("\t")[0] for l in open(f"{P}/POC_nuc_genes.tsv")]
write("nuclei_poc", M, bc, genes)
