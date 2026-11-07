"""Score a harness run fed by dump_e2e.py on the 879 pure benchmark cells (same truth as hgmm_e2e.py).
Usage: python -I score_e2e.py IN_DIR OUT_DIR   (IN_DIR from dump_e2e.py, OUT_DIR from harness)."""
import json
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

inp, out = sys.argv[1:3]
W = "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/ambient/work"
meta = pd.read_csv(f"{W}/cells.csv").set_index("barcode")
bcs = [b.split("-")[0] for b in open(f"{inp}/cell_barcodes.txt").read().split()]
genes = open(f"{inp}/genes.txt").read().split("\n")[:-1]
mouse = np.array(["ENSMUSG" in g for g in genes])
nc = len(bcs)
rho = np.fromfile(f"{out}/rho.f64", dtype=np.float64)
cl = np.fromfile(f"{out}/cluster.u32", dtype=np.uint32)
ip = np.fromfile(f"{out}/corr.indptr", dtype=np.int64)
ix = np.fromfile(f"{out}/corr.indices", dtype=np.int32)
dv = np.fromfile(f"{out}/corr.data", dtype=np.uint32)
G, B, nnz, _ = map(int, open(f"{inp}/dims.txt").read().split())
import os
crow = int(open(f"{out}/corr.nrows").read()) if os.path.exists(f"{out}/corr.nrows") else G
C = sp.csc_matrix((dv.astype(float), ix, ip), shape=(crow, nc))
if crow == 2 * len(genes):  # split exon / intron features: fold back to genes
    h = len(genes)
    C = (C[:h] + C[h:]).tocsc()
rp = np.fromfile(f"{inp}/raw.indptr", dtype=np.int64)
ri = np.fromfile(f"{inp}/raw.indices", dtype=np.int32)
rd = np.fromfile(f"{inp}/raw.data", dtype=np.uint32)
cidx = np.fromfile(f"{inp}/cells.idx", dtype=np.uint32)
Raw = sp.csc_matrix((rd.astype(float), ri, rp), shape=(G, B))[:, cidx]
if G == 2 * len(genes):
    Raw = (Raw[:len(genes)] + Raw[len(genes):]).tocsc()
pos = {b: i for i, b in enumerate(bcs)}
m = meta[meta.index.isin(pos)].copy()
cols = np.array([pos[b] for b in m.index])
m["rho_new"] = rho[cols]
m["cl"] = cl[cols]
pure = m.species.isin(["human", "mouse"]).values
x, t = m.rho_new.values[pure], m.rho_truth.values[pure]
R = dict(n_called=nc, n_pure=int(pure.sum()), mean=float(x.mean()), truth=float(t.mean()), ratio=float(x.mean() / t.mean()),
         spearman=float(spearmanr(x, t)[0]), pearson=float(pearsonr(x, t)[0]))
hum = cols[(m.species == "human").values]
mou = cols[(m.species == "mouse").values]
before = Raw[mouse][:, hum].sum() + Raw[~mouse][:, mou].sum()
after = C[mouse][:, hum].sum() + C[~mouse][:, mou].sum()
R["cross_species_before"] = float(before)
R["cross_species_left"] = float(after / before)
R["all_cells_rho_mean"] = float(rho.mean())
R["all_cells_rho_median"] = float(np.median(rho))
tot = np.asarray(Raw.sum(0)).ravel()
low = tot < 2000
R["n_cells_lt2000umi"] = int(low.sum())
R["rho_median_lt2000umi"] = float(np.median(rho[low])) if low.any() else None
R["n_clusters"] = int(len(np.unique(cl)))
R["pure_cells_per_cluster"] = {int(k): int(v) for k, v in pd.Series(m.cl.values[pure]).value_counts().items()}
R["all_cells_per_cluster"] = {int(k): int(v) for k, v in pd.Series(cl).value_counts().items()}
print(json.dumps(R))
mm_ = m[pure].copy()
mm_["err"] = mm_.rho_new - mm_.rho_truth
mm_["N"] = np.asarray(Raw.sum(0)).ravel()[cols[pure]]
print(mm_.sort_values("err", ascending=False)[["cl", "species", "N", "rho_new", "rho_truth", "H", "M"]].head(6).round(4).to_string())
