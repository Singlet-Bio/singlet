#!/usr/bin/env python3
"""Compute gene r for C04 (GRCm39 mouse 10xv3): singlify vs STARsolo gold."""
import numpy as np
import scipy.io
import scipy.stats
import scipy.sparse as sps
import singlepress as sp
from collections import defaultdict

GOLD_DIR = "/mnt/projects/debruinz_project/singlify_validation/starsolo/SRR34789664_matched_v3/Solo.out/Gene/raw"
SING_DIR = "/dev/shm/c04_singlify"

print("Loading gold standard (STARsolo)...")
barcodes_gold = open(f"{GOLD_DIR}/barcodes.tsv").read().splitlines()
gene_ids_gold = [l.split("\t")[0] for l in open(f"{GOLD_DIR}/features.tsv").read().splitlines()]
mtx_gold = scipy.io.mmread(f"{GOLD_DIR}/matrix.mtx").tocsc()
print(f"  Gold: {mtx_gold.shape[0]} genes x {mtx_gold.shape[1]} cells")

print("Loading singlify exon counts...")
sing_mtx = sp.read_1pz_int(f"{SING_DIR}/exon_counts.1pz")
rownames = sing_mtx.rownames  # ENSMUSG..._GeneName_chr:start-end
colnames = sing_mtx.colnames  # barcodes
print(f"  Singlify exon: {sing_mtx.shape[0]} rows x {sing_mtx.shape[1]} cells")

# Extract gene IDs from singlify row names
gene_ids_sing = [r.split("_")[0] for r in rownames]

# Build gene_id -> [row_indices] mapping
gene_to_rows = defaultdict(list)
for i, gid in enumerate(gene_ids_sing):
    gene_to_rows[gid].append(i)

# Find common genes
gene_set_gold = set(gene_ids_gold)
common_genes_idx_gold = []
common_sing_rows = []  # list of arrays of row indices to aggregate
for gi, gid in enumerate(gene_ids_gold):
    if gid in gene_to_rows:
        common_genes_idx_gold.append(gi)
        common_sing_rows.append(gene_to_rows[gid])

print(f"  Common genes: {len(common_genes_idx_gold)} / {len(gene_ids_gold)} gold")

# Barcode alignment (all should match since we used the same list)
bc_idx_sing = {bc: i for i, bc in enumerate(colnames)}
common_bcs = [bc for bc in barcodes_gold if bc in bc_idx_sing]
sing_col_order = [bc_idx_sing[bc] for bc in common_bcs]
gold_col_order = [barcodes_gold.index(bc) for bc in common_bcs]
print(f"  Common barcodes: {len(common_bcs)} / {len(barcodes_gold)}")

# Reorder columns
mtx_gold_sub = mtx_gold[:, gold_col_order]
sing_sub = sing_mtx[:, sing_col_order]

# Build gene-aggregation matrix for singlify (sparse row-sum by gene)
# Use a scipy csr aggregation approach:
# For each common gene, sum its exon rows. Build a gene_agg_matrix (n_common_genes x n_sing_rows)
print("Building gene aggregation matrix...")
n_genes = len(common_genes_idx_gold)
n_sing_rows = sing_mtx.shape[0]

rows_agg = []
cols_agg = []
for gi_out, rows in enumerate(common_sing_rows):
    for r in rows:
        rows_agg.append(gi_out)
        cols_agg.append(r)

agg_data = np.ones(len(rows_agg), dtype=np.float32)
agg_mat = sps.csr_matrix((agg_data, (rows_agg, cols_agg)), shape=(n_genes, n_sing_rows))

print("Aggregating singlify exon counts to gene level...")
# sing_sub is (n_sing_rows x n_cells) csc — aggregate rows
sing_gene_mtx = agg_mat.dot(sing_sub.astype(np.float32))  # (n_genes x n_cells)

# Extract gold genes in common_genes order
gold_gene_mtx = mtx_gold_sub[common_genes_idx_gold, :].toarray().astype(np.float32)  # (n_genes x n_cells)

print(f"  sing_gene_mtx: {sing_gene_mtx.shape}, gold_gene_mtx: {gold_gene_mtx.shape}")

# Gene-level correlation: sum counts per gene across all cells
sing_per_gene = np.asarray(sing_gene_mtx.sum(axis=1)).flatten()
gold_per_gene = gold_gene_mtx.sum(axis=1)

# Filter to expressed genes
mask = (sing_per_gene > 0) | (gold_per_gene > 0)
r_gene, _ = scipy.stats.pearsonr(sing_per_gene[mask], gold_per_gene[mask])
print(f"\n=== RESULTS ===")
print(f"Gene r (sum-across-cells, n={mask.sum()} expressed genes): {r_gene:.6f}")

# Also log-scale
sing_log = np.log1p(sing_per_gene[mask])
gold_log = np.log1p(gold_per_gene[mask])
r_log, _ = scipy.stats.pearsonr(sing_log, gold_log)
print(f"Gene r log1p (n={mask.sum()}): {r_log:.6f}")

# Top-expressed genes check
top_idx = np.argsort(gold_per_gene)[-10:][::-1]
gene_names_gold = [l.split("\t")[1] for l in open(f"{GOLD_DIR}/features.tsv").read().splitlines()]
print(f"\nTop 10 genes by gold UMIs:")
for i in top_idx:
    gname = gene_names_gold[common_genes_idx_gold[i]]
    print(f"  {gname}: gold={gold_per_gene[i]:.0f} sing={sing_per_gene[i]:.0f}")
