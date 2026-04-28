#!/usr/bin/env python3
"""Standalone N14 sex calling validation using pre-existing run outputs."""
import sys
sys.path.insert(0, '/mnt/home/debruinz/Singlet-AI/singlepress')
import singlepress as sp
import numpy as np

# Use existing SRR32855204 exon_counts output
PILEUP_DIR = "/mnt/projects/debruinz_project/singlify_validation/singlify_out/SRR32855204_n6"
GTF = "/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf"

import os
import re

print("=== N14 Sex Calling Validation ===")
print(f"Input: {PILEUP_DIR}")

# Load exon counts
pz = sp.read_1pz_int(PILEUP_DIR + "/exon_counts.1pz")
E = pz
ef = list(pz.rownames)
eb = list(pz.colnames)
print(f"Exon matrix: {E.shape[0]} features x {E.shape[1]} cells, nnz={E.nnz}")

# Parse chromosome from feature name
# Format: "ENSG00000223972_DDX11L1_chr1:11869-12227"
def parse_chrom(feat_name):
    m = re.search(r'_(chr[^:]+):', feat_name)
    if m:
        return m.group(1)
    # Also handle bare chromosomes
    m2 = re.search(r'_([XY\d]+):\d', feat_name)
    if m2:
        return m2.group(1)
    return ""

def parse_gene_name(feat_name):
    parts = feat_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""

# Y marker genes
Y_MARKERS = {"UTY", "DDX3Y", "KDM5D", "EIF1AY", "ZFY", "RPS4Y1", "USP9Y"}

# Identify XIST and chrY exon rows
xist_rows = []
y_rows = {}  # row -> gene_name

for i, feat in enumerate(ef):
    chrom = parse_chrom(feat)
    gene = parse_gene_name(feat)
    if gene == "XIST":
        xist_rows.append(i)
    elif chrom in ("chrY", "Y") and gene in Y_MARKERS:
        y_rows[i] = gene

print(f"XIST exon rows: {len(xist_rows)}")
print(f"chrY marker exon rows: {len(y_rows)}")

# Y genes found
y_genes_found = set(y_rows.values())
print(f"chrY marker genes found in model: {sorted(y_genes_found)}")

# Sum expression across all cells
E_arr = E.toarray()
total_umis = int(E_arr.sum())

xist_umis = int(E_arr[xist_rows, :].sum()) if xist_rows else 0
y_umis = int(sum(E_arr[r, :].sum() for r in y_rows)) if y_rows else 0

# Per-gene Y UMIs
y_gene_umis = {}
for row, gname in y_rows.items():
    y_gene_umis[gname] = y_gene_umis.get(gname, 0) + int(E_arr[row, :].sum())

print(f"\ntotal_umis = {total_umis:,}")
print(f"xist_umis  = {xist_umis:,}")
print(f"y_umis     = {y_umis:,}")
print(f"y_gene_umis: {y_gene_umis}")

if total_umis > 0:
    cpm = 1e6 / total_umis
    xist_cpm = xist_umis * cpm
    y_cpm = y_umis * cpm
    n_y_detected = sum(1 for v in y_gene_umis.values() if v > 0)
    print(f"\nxist_cpm   = {xist_cpm:.4f}")
    print(f"y_cpm      = {y_cpm:.4f}")
    print(f"n_y_genes_detected = {n_y_detected}")

    # Classify
    XIST_THRESH = 1.0
    Y_THRESH    = 2.0
    Y_MIN_GENES = 2
    if xist_cpm > XIST_THRESH and y_cpm < Y_THRESH:
        sex = "female"
        confidence = min(1.0, xist_cpm / (XIST_THRESH * 10.0))
    elif y_cpm > Y_THRESH and n_y_detected >= Y_MIN_GENES:
        sex = "male"
        confidence = min(1.0, y_cpm / (Y_THRESH * 10.0))
    elif xist_cpm > XIST_THRESH and y_cpm > Y_THRESH:
        sex = "unknown"
        confidence = 0.0
    else:
        sex = "unknown"
        partial = max(xist_cpm / XIST_THRESH, y_cpm / Y_THRESH)
        confidence = min(0.5, partial * 0.3)

    print(f"\n=== SEX CALL: {sex} (confidence={confidence:.4f}) ===")
    print(f"SRR32855204 (10x-ARC-GEX, human): {sex}")
