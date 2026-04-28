import numpy as np
import scipy.io as sio
import scipy.sparse as sps
import os, sys

sys.path.insert(0, '/mnt/home/debruinz/Singlet-AI/singlepress')
import singlepress as sp

SINGLIFY_DIR = "/dev/shm/n6val_regen_42"
STARSOLO_DIR = "/mnt/projects/debruinz_project/singlify_validation/starsolo/SRR32855204_regen/Solo.out/Gene/filtered"

# Load singlify exon counts (.1pz)
E = sp.read_1pz_int(SINGLIFY_DIR + "/exon_counts.1pz")
ef = list(E.rownames)
eb = list(E.colnames)
print(f"Singlify exon: {E.shape}, nnz={E.nnz}")

# Load STARsolo
S = sio.mmread(os.path.join(STARSOLO_DIR, "matrix.mtx")).tocsc()
sf = [l.strip().split("\t") for l in open(os.path.join(STARSOLO_DIR, "features.tsv"))]
sb = [l.strip() for l in open(os.path.join(STARSOLO_DIR, "barcodes.tsv"))]
sg_idx = {f[0]: i for i, f in enumerate(sf)}
print(f"STARsolo Gene: {S.shape}, nnz={S.nnz}")

# Map exon features to gene IDs
eg = np.array([f.split("_")[0] for f in ef])
unique_genes = sorted(set(eg) & set(sg_idx.keys()))
gene2row = {g: i for i, g in enumerate(unique_genes)}
print(f"Shared genes: {len(unique_genes)}")

# Build sparse aggregation matrix A (n_genes × n_exons) using COO
exon_indices = np.arange(len(ef))
valid = np.array([g in gene2row for g in eg])
row_idx = np.array([gene2row[g] for g in eg[valid]])
col_idx = exon_indices[valid]
A = sps.csr_matrix(
    (np.ones(col_idx.shape[0], dtype=np.float32), (row_idx, col_idx)),
    shape=(len(unique_genes), len(ef))
)
print(f"Aggregation matrix: {A.shape}")

# Shared barcodes
pb = {b: i for i, b in enumerate(eb)}
sb2 = {b: i for i, b in enumerate(sb)}
shared_b = sorted(set(pb) & set(sb2))
pc = np.array([pb[b] for b in shared_b])
sc = np.array([sb2[b] for b in shared_b])
print(f"Shared barcodes: {len(shared_b)}")

# Vectorized gene aggregation: A @ E[:, pc] 
E_sub = E[:, pc]
Sg = A @ E_sub  # (n_genes × n_shared_cells) sparse multiplication
print(f"Gene matrix computed: {Sg.shape}")

# STARsolo submatrix
sr = np.array([sg_idx[g] for g in unique_genes])
Ss = S[sr, :][:, sc]

# Per-gene totals
sg_tot = np.asarray(Sg.sum(1)).ravel()
ss_tot = np.asarray(Ss.sum(1)).ravel()
mask = (sg_tot > 0) | (ss_tot > 0)
rg = np.corrcoef(sg_tot[mask], ss_tot[mask])[0, 1]

# Per-cell totals
sc_sg = np.asarray(Sg.sum(0)).ravel()
sc_ss = np.asarray(Ss.sum(0)).ravel()
rc = np.corrcoef(sc_sg, sc_ss)[0, 1]

tn6 = int(sg_tot.sum())
ts = int(ss_tot.sum())
ratio = tn6 / ts if ts else 0

print(f"\n=== N6 vs REGEN STARSOLO ===")
print(f"Per-gene Pearson r  = {rg:.6f}  {'PASS' if rg>=0.995 else 'FAIL'} (>= 0.995)")
print(f"Per-cell Pearson r  = {rc:.6f}  {'PASS' if rc>=0.99 else 'FAIL'} (>= 0.99)")
print(f"Singlify UMIs       = {tn6:,}")
print(f"STARsolo UMIs       = {ts:,}")
print(f"UMI ratio           = {ratio:.4f}  {'PASS' if abs(ratio-1)<0.20 else 'FAIL'} (within 20% of 1.0)")
print(f"Genes with counts   = {mask.sum()}")

# Top discordant genes
dm = mask & (sg_tot > 0) & (ss_tot > 0)
rp = np.where(dm & (ss_tot > 0), sg_tot / np.maximum(ss_tot, 1), 1.0)
ti = np.argsort(np.abs(rp - 1))[-10:][::-1]
print("\nTop discordant genes (singlify/starsolo ratio):")
for i in ti:
    if dm[i]:
        print(f"  {unique_genes[i]}: n6={int(sg_tot[i])}, star={int(ss_tot[i])}, ratio={rp[i]:.3f}")
