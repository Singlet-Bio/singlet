"""Build the common inputs: cell set, species composition, ambient pool, truth contamination, clusters.
Usage: python -I prep.py   (run inside a Slurm job; reads the Singlet run directory)."""
import json, sys, os
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

os.makedirs(WORK, exist_ok=True)
# Singlet's `--raw-matrix` writes nothing (see docs); the raw barcode x gene matrix is the one from Singlet's own STAR pass
raw, gid, gname, rbc = read_mtx_dir(f"{SINGLET}/star_Solo.out/Gene/raw")
fbc = pd.read_csv(f"{SINGLET}/star_Solo.out/Gene/filtered/barcodes.tsv", header=None)[0].astype(str).str.replace("-1", "", regex=False).values
sfb = pd.read_csv(f"{SINGLET}/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None)[0].astype(str).str.replace("-1", "", regex=False).values
print("Singlet filtered barcodes == STAR filtered barcodes:", set(sfb) == set(fbc), len(fbc))
sp_g = species_of(gid)
tot = np.asarray(raw.sum(0)).ravel()
print(f"raw: {raw.shape} barcodes>=1UMI={int((tot>0).sum())}  filtered cells={len(fbc)}")
# trimmed raw (barcodes with >= 1 UMI) as a 10x directory for CellBender / SoupX
import gzip
keep = np.where(tot > 0)[0]
os.makedirs(f"{WORK}/raw", exist_ok=True)
with gzip.open(f"{WORK}/raw/matrix.mtx.gz", "wb") as fh: scipy.io.mmwrite(fh, raw[:, keep].tocoo(), field="integer")
pd.DataFrame({0: gid, 1: gname, 2: "Gene Expression"}).to_csv(f"{WORK}/raw/features.tsv.gz", sep="	", header=False, index=False)
pd.DataFrame({0: rbc[keep]}).to_csv(f"{WORK}/raw/barcodes.tsv.gz", sep="	", header=False, index=False)
cell_idx = pd.Series(np.arange(len(rbc)), index=rbc).reindex(fbc)
assert cell_idx.notna().all()
cell_idx = cell_idx.astype(int).values
Xc = raw[:, cell_idx]

def species_umis(X):
    H = np.asarray(X[sp_g == 0].sum(0)).ravel(); M = np.asarray(X[sp_g == 1].sum(0)).ravel()
    return H, M
H, M = species_umis(Xc)
N = H + M
frac_m = M / np.maximum(N, 1)

# ambient pool: barcodes with 1..100 UMI (Singlet's rule, top 50k by count); alternatives for sensitivity
def pool_alpha(lo, hi, cap=None):
    idx = np.where((tot >= lo) & (tot <= hi))[0]
    if cap and len(idx) > cap:
        idx = idx[np.argsort(-tot[idx])[:cap]]
    h, m = species_umis(raw[:, idx])
    return float(m.sum() / (h.sum() + m.sum())), len(idx), int(h.sum() + m.sum())
alphas = {k: pool_alpha(*v) for k, v in {"singlet_1_100_top50k": (1, 100, 50000), "10_100": (10, 100), "20_200": (20, 200), "1_20": (1, 20)}.items()}
print("ambient mouse share (alpha_m):", alphas)
alpha_m = alphas["singlet_1_100_top50k"][0]

# labels from own counts
sp_cell = np.where(frac_m >= 0.95, "mouse", np.where(frac_m <= 0.05, "human", "mixed"))
minor = np.minimum(frac_m, 1 - frac_m)

# Cell Ranger (independent alignment) species fractions for the same barcodes
with h5py.File(CR_H5, "r") as f:
    g = f["matrix"]
    bc = np.array([b.decode().replace("-1", "") for b in g["barcodes"][:]])
    fid = np.array([x.decode() for x in g["features/id"][:]])
    X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
crm = np.char.startswith(fid, "ENSMUSG") | np.char.startswith(fid.astype(str), "mm10")
cr_ids = pd.Series(fid)
cr_mouse = (~cr_ids.str.startswith("ENSG")).values if cr_ids.str.startswith("ENS").any() else crm
cr_M = np.asarray(X[cr_mouse].sum(0)).ravel(); cr_N = np.asarray(X.sum(0)).ravel()
cr = pd.DataFrame({"cr_M": cr_M, "cr_N": cr_N}, index=bc)
print("Cell Ranger cells:", len(cr), " of which in Singlet's set:", int(np.isin(fbc, bc).sum()))

meta = pd.DataFrame({"barcode": fbc, "col_in_raw": cell_idx, "N": N, "H": H, "M": M, "frac_mouse": frac_m, "minor_frac": minor,
                     "species": sp_cell})
meta = meta.join(cr, on="barcode")
meta["cr_minor_frac"] = np.minimum(meta.cr_M / meta.cr_N, 1 - meta.cr_M / meta.cr_N)
# truth contamination for pure cells: other-species fraction scaled by the other species' share of the ambient pool
meta["rho_truth"] = np.where(meta.species == "human", meta.frac_mouse / alpha_m,
                     np.where(meta.species == "mouse", (1 - meta.frac_mouse) / (1 - alpha_m), np.nan))
# merge Singlet outputs
amb = pd.read_csv(f"{SINGLET}/ambient_contamination.tsv", sep="\t").rename(columns={"rho": "rho_singlet"})
meta = meta.merge(amb[["barcode", "rho_singlet"]], on="barcode", how="left")
bn = pd.read_csv(f"{SINGLET}/barnyard_classification.tsv", sep="\t").rename(columns={"species_call": "singlet_barnyard"})
meta = meta.merge(bn[["barcode", "singlet_barnyard"]], on="barcode", how="left")
db = pd.read_csv(f"{SINGLET}/doublet_scores.tsv", sep="\t").rename(columns={"doublet_score": "singlet_dbl_score", "is_doublet": "singlet_dbl"})
meta = meta.merge(db[["barcode", "singlet_dbl_score", "singlet_dbl"]], on="barcode", how="left")
print(meta.species.value_counts().to_dict())
print(meta[["minor_frac", "cr_minor_frac"]].describe().round(4))

# clusters (leiden on log-normalised HVG PCA) for SoupX
import scanpy as sc, anndata as ad
a = ad.AnnData(Xc.T.tocsr().astype(np.float32)); a.obs_names = fbc
sc.pp.normalize_total(a, 1e4); sc.pp.log1p(a); sc.pp.highly_variable_genes(a, n_top_genes=2000)
a = a[:, a.var.highly_variable].copy(); sc.pp.pca(a, 30); from sklearn.cluster import KMeans
a.obs["leiden"] = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(a.obsm["X_pca"]).astype(str)  # igraph not installed: k-means stands in for leiden
meta["leiden"] = a.obs["leiden"].values
print(meta.groupby(["species", "leiden"]).size())

meta.to_csv(f"{WORK}/cells.csv", index=False)
pd.DataFrame({"id": gid, "name": gname, "species": np.where(sp_g == 1, "mouse", "human")}).to_csv(f"{WORK}/genes.csv", index=False)
scipy.io.mmwrite(f"{WORK}/cells.mtx", Xc.tocoo())                       # genes x called cells, raw counts
# empty droplets for DecontX background (10..100 UMI, 20k random)
rng = np.random.default_rng(0); e = np.where((tot >= 10) & (tot <= 100))[0]; e = rng.choice(e, min(20000, len(e)), replace=False)
scipy.io.mmwrite(f"{WORK}/empty.mtx", raw[:, e].tocoo())
json.dump({"alpha_m": alpha_m, "alphas": alphas, "n_cells": int(len(fbc)), "n_raw_barcodes": int((tot > 0).sum())}, open(f"{WORK}/prep.json", "w"), indent=1)
