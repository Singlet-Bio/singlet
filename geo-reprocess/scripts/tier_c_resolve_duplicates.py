#!/usr/bin/env python3
"""Resolve and apply Tier C metadata for duplicate-mapped GSMs."""
import json, pandas as pd, os, sys
import anndata as ad

mapping_path = sys.argv[1]
h5ad_path = sys.argv[2]
dataset_dir = sys.argv[3]

mapping = json.load(open(mapping_path))

# Find duplicates
rev = {}
for gsm, m in mapping.items():
    rev.setdefault(m["cx_sample"], []).append(gsm)

dup_samples = {s: gs for s, gs in rev.items() if len(gs) > 1}
print(f"Duplicate samples: {len(dup_samples)}, affecting {sum(len(gs) for gs in dup_samples.values())} GSMs")

# For each duplicate: both GSMs get the same CellxGene metadata
# This is correct because both sorts share the same cell pool
# The barcode matching handles which cells actually match

print("\nLoading h5ad obs...")
adata = ad.read_h5ad(h5ad_path, backed="r")
obs = adata.obs.copy()
obs["bare_bc"] = obs.index.str.split("-").str[0]

META_COLS = [
    "cell_type", "Author_Annotation", "Major_celltypes",
    "donor_id", "tissue", "assay", "disease", "sex",
    "development_stage", "APOE_class", "Brain.Region", "SORT",
    "Braak.stage", "Disease.Group", "Amyloid", "NP.Diagonis",
]
meta_cols = [c for c in META_COLS if c in obs.columns]

# Build per-sample lookup
sample_groups = {}
for s in obs["sample"].unique():
    sg = obs.loc[obs["sample"] == s, ["bare_bc"] + meta_cols].copy()
    sg = sg.drop_duplicates(subset="bare_bc", keep="first")
    sg = sg.set_index("bare_bc")
    sample_groups[s] = sg

success = 0
for cx_sample, gsms in sorted(dup_samples.items()):
    sg = sample_groups.get(cx_sample)
    if sg is None:
        continue
    
    for gsm in gsms:
        cells_path = os.path.join(dataset_dir, gsm, "cells.parquet")
        if not os.path.exists(cells_path):
            continue
        
        our_cells = pd.read_parquet(cells_path, columns=["barcode"])
        our_bcs = set(our_cells["barcode"].astype(str).tolist())
        
        aligned = sg.reindex(sorted(our_bcs))
        n_matched = aligned[meta_cols[0]].notna().sum()
        match_rate = n_matched / len(our_bcs) if our_bcs else 0

        am_path = os.path.join(dataset_dir, gsm, "author_metadata.parquet")
        if os.path.exists(am_path):
            existing = pd.read_parquet(am_path)
            if "barcode" in existing.columns:
                existing = existing.set_index("barcode")
            for col in meta_cols:
                if col not in existing.columns:
                    existing[col] = aligned[col].reindex(existing.index)
            existing.reset_index().to_parquet(am_path, index=False)
        
        # Also check author_metadata SORT to see difference
        am = pd.read_parquet(am_path)
        sort_val = am["sort"].iloc[0] if "sort" in am.columns else "N/A"
        print(f"  {gsm} -> {cx_sample}: {n_matched}/{len(our_bcs)} = {match_rate:.1%}, sort={sort_val}")
        success += 1

print(f"\nProcessed {success} duplicate GSMs")
