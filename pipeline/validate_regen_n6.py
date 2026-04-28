import numpy as np
import scipy.io as sio
import os, sys

sys.path.insert(0, '/mnt/home/debruinz/Singlet-AI/singlepress')
import singlepress as sp

SINGLIFY_DIR = "/dev/shm/n6val_regen_42"
STARSOLO_DIR = "/mnt/projects/debruinz_project/singlify_validation/starsolo/SRR32855204_regen/Solo.out/Gene/filtered"

# Load singlify exon counts (.1pz) — read_1pz returns the matrix directly
E = sp.read_1pz_int(SINGLIFY_DIR + "/exon_counts.1pz")
ef = list(E.rownames)
eb = list(E.colnames)
print(f"Singlify exon: {E.shape}, nnz={E.nnz}")

# Load STARsolo Gene/filtered
S = sio.mmread(os.path.join(STARSOLO_DIR, "matrix.mtx")).tocsc()
sf = [l.strip().split("\t") for l in open(os.path.join(STARSOLO_DIR, "features.tsv"))]
sb = [l.strip() for l in open(os.path.join(STARSOLO_DIR, "barcodes.tsv"))]
sg_idx = {f[0]: i for i, f in enumerate(sf)}
print(f"STARsolo Gene: {S.shape}, nnz={S.nnz}")

# Aggregate exon -> gene
eg = [f.split("_")[0] for f in ef]
g2e = {}
for i, g in enumerate(eg):
    g2e.setdefault(g, []).append(i)

shared_g = sorted(set(g2e.keys()) & set(sg_idx.keys()))
pb = {b: i for i, b in enumerate(eb)}
sb2 = {b: i for i, b in enumerate(sb)}
shared_b = sorted(set(pb) & set(sb2))
print(f"Shared genes: {len(shared_g)}, shared barcodes: {len(shared_b)}")

pc = np.array([pb[b] for b in shared_b])
sc = np.array([sb2[b] for b in shared_b])
gm = {g: i for i, g in enumerate(shared_g)}
ng = len(shared_g); nb = len(shared_b)

Es = E[:, pc]
Sg = np.zeros((ng, nb), dtype=np.float64)
for g in shared_g:
    Sg[gm[g], :] = Es[g2e[g], :].toarray().sum(0)

sr = np.array([sg_idx[g] for g in shared_g])
Ss = S[sr, :][:, sc].toarray().astype(np.float64)

sg_tot = Sg.sum(1)
ss_tot = Ss.sum(1)
mask = (sg_tot > 0) | (ss_tot > 0)
rg = np.corrcoef(sg_tot[mask], ss_tot[mask])[0, 1]

sc_sg = Sg.sum(0)
sc_ss = Ss.sum(0)
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
print("\nTop discordant genes (ratio singlify/starsolo):")
for i in ti:
    if dm[i]:
        print(f"  {shared_g[i]}: n6={int(sg_tot[i])}, star={int(ss_tot[i])}, ratio={rp[i]:.3f}")
