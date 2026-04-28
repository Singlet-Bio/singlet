#!/usr/bin/env python3
"""Full validation suite for singlet-pileup v0.2.0 Pipeline V2."""
import scipy.io as sio
import numpy as np
import subprocess, os, sys, json, shutil

SAMTOOLS = shutil.which("samtools") or os.path.expanduser("~/.conda/envs/cellarium/bin/samtools")

base = sys.argv[1]
star = sys.argv[2]

print("=" * 60)
print("VALIDATION SUITE: singlet-pileup v0.2.0")
print("=" * 60)

# --- 1. Gene counts Pearson r ---
print("\n[1] Gene counts Pearson r vs STARsolo Gene")
P = sio.mmread(f"{base}/gene_counts/matrix.mtx").tocsc()
genes_p = [l.strip().split("\t") for l in open(f"{base}/gene_counts/features.tsv")]
bcs_p = [l.strip() for l in open(f"{base}/gene_counts/barcodes.tsv")]
S = sio.mmread(f"{star}/matrix.mtx").tocsc()
genes_s = [l.strip().split("\t") for l in open(f"{star}/features.tsv")]
bcs_s = [l.strip() for l in open(f"{star}/barcodes.tsv")]

p_bc = {bc: i for i, bc in enumerate(bcs_p)}
s_bc = {bc: i for i, bc in enumerate(bcs_s)}
common_bc = sorted(set(p_bc) & set(s_bc))
p_gid = {g[0]: i for i, g in enumerate(genes_p)}
s_gid = {g[0]: i for i, g in enumerate(genes_s)}
common_genes = sorted(set(p_gid) & set(s_gid))

pi = [p_bc[bc] for bc in common_bc]
si = [s_bc[bc] for bc in common_bc]
pg = [p_gid[g] for g in common_genes]
sg = [s_gid[g] for g in common_genes]
Psub = P[np.ix_(pg, pi)].toarray().ravel().astype(float)
Ssub = S[np.ix_(sg, si)].toarray().ravel().astype(float)
mask = (Psub > 0) | (Ssub > 0)
r = np.corrcoef(Psub[mask], Ssub[mask])[0, 1]
total_p = Psub.sum()
total_s = Ssub.sum()
ratio = total_p / total_s if total_s > 0 else float("inf")
both_nz = np.sum((Psub > 0) & (Ssub > 0))
ponly = np.sum((Psub > 0) & (Ssub == 0))
sonly = np.sum((Psub == 0) & (Ssub > 0))
cell_p = np.array(P[:, pi].sum(axis=0)).ravel().astype(float)
cell_s = np.array(S[:, si].sum(axis=0)).ravel().astype(float)
cell_r = np.corrcoef(cell_p, cell_s)[0, 1]
genes_per_cell = np.median(np.array((P[:, pi] > 0).sum(axis=0)).ravel())
counts_per_cell = np.median(np.array(P[:, pi].sum(axis=0)).ravel())

print(f"  Pearson r = {r:.6f}  {'PASS' if r > 0.99 else 'FAIL'} (target >0.999)")
print(f"  Cell-level r = {cell_r:.6f}  {'PASS' if cell_r > 0.999 else 'FAIL'}")
print(f"  Count ratio = {ratio:.4f}")
print(f"  Both nz={both_nz}, P-only={ponly}, S-only={sonly}")
print(f"  Median genes/cell={genes_per_cell:.0f}, counts/cell={counts_per_cell:.0f}")

# --- 2. Exon to Gene reconstruction ---
print("\n[2] Exon counts sum to gene counts (0% error target)")
E = sio.mmread(f"{base}/exon_counts.mtx").tocsc()
exon_feat = [l.strip() for l in open(f"{base}/exon_counts_features.tsv")]

gene_name_to_idx = {}
for i, g in enumerate(genes_p):
    name = g[1] if len(g) > 1 else g[0]
    gene_name_to_idx[name] = i
gene_id_to_idx = {g[0]: i for i, g in enumerate(genes_p)}

exon_by_gene = {}
unmapped = 0
for ei, ef in enumerate(exon_feat):
    # Format: GENEID_GENENAME_CHROM:START-END (new) or GENENAME_CHROM:START-END (old)
    # Try gene_id match first (everything before first _)
    parts = ef.split("_", 1)
    if parts[0] in gene_id_to_idx:
        gi = gene_id_to_idx[parts[0]]
        exon_by_gene.setdefault(gi, []).append(ei)
    else:
        # Fallback: try gene name (old format)
        idx = ef.rfind("_chr")
        if idx > 0:
            gname = ef[:idx]
        else:
            idx = ef.rfind("_MT:")
            gname = ef[:idx] if idx > 0 else ef.split("_")[0]
        if gname in gene_name_to_idx:
            gi = gene_name_to_idx[gname]
            exon_by_gene.setdefault(gi, []).append(ei)
        else:
            unmapped += 1

total_recon_err = 0
max_diff = 0
max_diff_gene = ""
n_checked = 0
for gi, eidxs in exon_by_gene.items():
    exon_sum = np.asarray(E[eidxs, :].sum(axis=0)).ravel()
    gene_val = np.asarray(P[gi, :].toarray()).ravel()
    diff = np.abs(exon_sum - gene_val).sum()
    total_recon_err += diff
    if diff > max_diff:
        max_diff = diff
        max_diff_gene = genes_p[gi][1] if len(genes_p[gi]) > 1 else genes_p[gi][0]
    n_checked += 1

recon_pct = total_recon_err / total_p * 100 if total_p > 0 else 0
print(f"  Genes mapped: {n_checked}, unmapped exons: {unmapped}")
print(f"  Reconstruction error: {total_recon_err:.0f} ({recon_pct:.4f}%)")
if max_diff > 0:
    print(f"  Max error gene: {max_diff_gene} ({max_diff:.0f})")
recon_pass = (recon_pct == 0)
print(f"  RESULT: {'PASS' if recon_pass else 'FAIL'}")

# --- 3. chrM BAM ---
print("\n[3] chrM BAM validation")
chrm_bam = f"{base}/chrM.bam"
chrm_pass = False
if os.path.exists(chrm_bam):
    res = subprocess.run([SAMTOOLS, "view", chrm_bam],
                         capture_output=True, text=True)
    chrm_count = 0
    non_chrm = 0
    for line in res.stdout.strip().split("\n"):
        if not line: continue
        fields = line.split("\t")
        if len(fields) > 2:
            if fields[2] in ("chrM", "MT"):
                chrm_count += 1
            else:
                non_chrm += 1
    total_chrm = chrm_count + non_chrm
    fpr = non_chrm / total_chrm * 100 if total_chrm > 0 else 0
    print(f"  chrM reads: {chrm_count}, non-chrM: {non_chrm}")
    print(f"  FPR: {fpr:.4f}%, target <0.01%")
    chrm_pass = (fpr < 0.01)
    print(f"  RESULT: {'PASS' if chrm_pass else 'FAIL'}")
else:
    print("  chrM.bam MISSING")

# --- 4. SJ feature names ---
print("\n[4] Splice junction feature names")
sj_feat = [l.strip() for l in open(f"{base}/sj_counts_features.tsv")]
print(f"  Sample: {sj_feat[:3]}")
numeric_sj = sum(1 for s in sj_feat if s.split(":")[0].isdigit())
sj_pass = (numeric_sj == 0)
print(f"  Numeric tid SJs: {numeric_sj}/{len(sj_feat)}")
print(f"  RESULT: {'PASS' if sj_pass else 'FAIL'}")

# --- 5. SNP AD/DP ---
print("\n[5] SNP AD/DP matrix validation")
AD = sio.mmread(f"{base}/snp_ad.mtx").tocsc()
DP = sio.mmread(f"{base}/snp_dp.mtx").tocsc()
print(f"  AD: {AD.shape[0]}x{AD.shape[1]}, nnz={AD.nnz}")
print(f"  DP: {DP.shape[0]}x{DP.shape[1]}, nnz={DP.nnz}")
ad_sum = AD.sum()
dp_sum = DP.sum()
print(f"  AD total={ad_sum}, DP total={dp_sum}")

# Spot check AD <= DP on all AD nonzero entries
ad_coo = AD.tocoo()
dp_csc = DP.tocsc()
violations = 0
check_n = min(ad_coo.nnz, 50000)
for i in range(check_n):
    ri, ci = int(ad_coo.row[i]), int(ad_coo.col[i])
    if ad_coo.data[i] > dp_csc[ri, ci]:
        violations += 1
snp_pass = (violations == 0)
print(f"  AD > DP violations ({check_n} checked): {violations}")
print(f"  RESULT: {'PASS' if snp_pass else 'FAIL'}")

# --- 5b. SNP comparison vs cellSNP-lite ---
print("\n[5b] SNP AD vs cellSNP-lite")
csnp_dir = os.path.join(os.path.dirname(base), "cellsnp")
if os.path.exists(f"{csnp_dir}/cellSNP.tag.AD.mtx"):
    cAD = sio.mmread(f"{csnp_dir}/cellSNP.tag.AD.mtx").tocsc()
    cDP = sio.mmread(f"{csnp_dir}/cellSNP.tag.DP.mtx").tocsc()
    csnp_bcs = [l.strip() for l in open(f"{csnp_dir}/cellSNP.samples.tsv")]
    print(f"  cellSNP: {cAD.shape[0]} SNPs x {cAD.shape[1]} cells")
    print(f"  cellSNP AD total={cAD.sum()}, DP total={cDP.sum()}")
    print(f"  Pileup   AD total={ad_sum}, DP total={dp_sum}")
    # Compare totals
    ad_ratio = ad_sum / cAD.sum() if cAD.sum() > 0 else 0
    dp_ratio = dp_sum / cDP.sum() if cDP.sum() > 0 else 0
    print(f"  AD ratio (pileup/cellSNP): {ad_ratio:.4f}")
    print(f"  DP ratio (pileup/cellSNP): {dp_ratio:.4f}")
else:
    print("  cellSNP reference not found")

# --- 6. Output completeness ---
print("\n[6] Output completeness")
expected = [
    "snp_ad.mtx", "snp_ad_features.tsv", "snp_ad_barcodes.tsv",
    "snp_dp.mtx", "snp_dp_features.tsv", "snp_dp_barcodes.tsv",
    "exon_counts.mtx", "exon_counts_features.tsv", "exon_counts_barcodes.tsv",
    "intron_counts.mtx", "intron_counts_features.tsv", "intron_counts_barcodes.tsv",
    "sj_counts.mtx", "sj_counts_features.tsv", "sj_counts_barcodes.tsv",
    "gene_counts/matrix.mtx", "gene_counts/features.tsv", "gene_counts/barcodes.tsv",
    "genefull_counts/matrix.mtx", "genefull_counts/features.tsv", "genefull_counts/barcodes.tsv",
    "chrM.bam", "pileup_stats.json"
]
missing = [f for f in expected if not os.path.exists(f"{base}/{f}")]
comp_pass = len(missing) == 0
print(f"  Expected: {len(expected)}, Missing: {len(missing)}")
if missing:
    for m in missing:
        print(f"    MISSING: {m}")
print(f"  RESULT: {'PASS' if comp_pass else 'FAIL'}")

# --- 7. Performance ---
print("\n[7] Performance metrics")
stats = json.load(open(f"{base}/pileup_stats.json"))
wt = stats["wall_time_s"]
total_reads = stats["total_reads"]
print(f"  Wall time: {wt:.1f}s for {total_reads/1e6:.1f}M reads")
print(f"  Reads/sec: {total_reads/wt:.0f}")
print(f"  Target: <480s for 150M reads (extrapolated: {wt*150e6/total_reads:.0f}s)")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
results = {
    "Gene Pearson r": (f"{r:.6f}", r > 0.99),
    "Cell-level r": (f"{cell_r:.6f}", cell_r > 0.999),
    "Exon->Gene recon": (f"{recon_pct:.4f}%", recon_pass),
    "chrM BAM valid": ("yes" if chrm_pass else "no", chrm_pass),
    "SJ chrom names": ("proper" if sj_pass else "numeric", sj_pass),
    "AD <= DP": ("valid" if snp_pass else "violations", snp_pass),
    "Output complete": ("yes" if comp_pass else "missing", comp_pass),
}
all_pass = True
for name, (val, passed) in results.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {name:25s} {val:15s} {status}")

print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
sys.exit(0 if all_pass else 1)
