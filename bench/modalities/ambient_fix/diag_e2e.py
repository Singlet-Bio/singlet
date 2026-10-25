"""Why does exon+intron score worse than exon only? Truth variants for the 879 pure cells from the 'sp' dump (rows 0..G-1 exon,
G..2G-1 intron). Usage: python -I diag_e2e.py TAG [TAG...]  (harness outputs under e2e_out/, 'sp' mode runs)."""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

A = "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/ambient"
inp = f"{A}/e2e_in_sp"
meta = pd.read_csv(f"{A}/work/cells.csv").set_index("barcode")
bcs = [b.split("-")[0] for b in open(f"{inp}/cell_barcodes.txt").read().split()]
genes = open(f"{inp}/genes.txt").read().split("\n")[:-1]
h = len(genes)
mouse = np.array(["ENSMUSG" in g for g in genes])
G, B, nnz, nc = map(int, open(f"{inp}/dims.txt").read().split())
Raw = sp.csc_matrix((np.fromfile(f"{inp}/raw.data", dtype=np.uint32).astype(float), np.fromfile(f"{inp}/raw.indices", dtype=np.int32),
                     np.fromfile(f"{inp}/raw.indptr", dtype=np.int64)), shape=(G, B))
cidx = np.fromfile(f"{inp}/cells.idx", dtype=np.uint32)
tot = np.asarray(Raw.sum(0)).ravel()
pool = np.where((tot >= 1) & (tot <= 100))[0]
pool = pool[np.argsort(-tot[pool], kind="stable")[:50000]]
P = np.asarray(Raw[:, pool].sum(1)).ravel()
mm = np.concatenate([mouse, mouse])
alpha_ex = P[:h][mouse].sum() / P[:h].sum()
alpha_ei = (P[:h][mouse].sum() + P[h:][mouse].sum()) / P.sum()
alpha_int = P[h:][mouse].sum() / P[h:].sum()
print(f"pool mouse share: exon {alpha_ex:.4f} intron {alpha_int:.4f} both {alpha_ei:.4f}; pool intron fraction {P[h:].sum() / P.sum():.3f}")
C = Raw[:, cidx]
pos = {b: i for i, b in enumerate(bcs)}
m = meta[meta.index.isin(pos)].copy()
cols = np.array([pos[b] for b in m.index])
pure = m.species.isin(["human", "mouse"]).values
sub = C[:, cols[pure]].tocsr()
sp_ = m.species.values[pure]
Nex = np.asarray(sub[:h].sum(0)).ravel(); Nin = np.asarray(sub[h:].sum(0)).ravel()
Mex = np.asarray(sub[:h][mouse].sum(0)).ravel(); Hex = Nex - Mex
Min = np.asarray(sub[h:][mouse].sum(0)).ravel(); Hin = Nin - Min
hum = sp_ == "human"
cross_ex = np.where(hum, Mex, Hex); cross_in = np.where(hum, Min, Hin)
oth_share_ex = np.where(hum, alpha_ex, 1 - alpha_ex)
oth_share_ei = np.where(hum, alpha_ei, 1 - alpha_ei)
truth_ex = cross_ex / oth_share_ex / np.maximum(Nex, 1)
truth_ei = (cross_ex + cross_in) / oth_share_ei / np.maximum(Nex + Nin, 1)
print(f"cells {pure.sum()}; intron share of cell UMIs: mean {np.mean(Nin / (Nex + Nin)):.3f} sd {np.std(Nin / (Nex + Nin)):.3f}")
print(f"cross intron / cross exon reads: {cross_in.sum() / cross_ex.sum():.3f};  corr(cross_in, cross_ex) per cell: {np.corrcoef(cross_in, cross_ex)[0, 1]:.3f}")
print(f"truth_ex mean {truth_ex.mean():.4f}  truth_ei mean {truth_ei.mean():.4f}  spearman(truth_ex, truth_ei) {spearmanr(truth_ex, truth_ei)[0]:.3f}")
for tag in sys.argv[1:]:
    rho = np.fromfile(f"{A}/e2e_out/{tag}/rho.f64", dtype=np.float64)[cols[pure]]
    print(f"[{tag}] mean {rho.mean():.4f}; vs truth_ex: spearman {spearmanr(rho, truth_ex)[0]:.3f} pearson {pearsonr(rho, truth_ex)[0]:.3f};"
          f" vs truth_ei: spearman {spearmanr(rho, truth_ei)[0]:.3f} pearson {pearsonr(rho, truth_ei)[0]:.3f}; "
          f"rho*Ntot/Nex vs truth_ex: spearman {spearmanr(rho * (Nex + Nin) / np.maximum(Nex, 1), truth_ex)[0]:.3f}")
