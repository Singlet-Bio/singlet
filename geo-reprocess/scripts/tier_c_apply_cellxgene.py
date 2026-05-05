#!/usr/bin/env python3
"""Tier C: Apply CellxGene metadata to per-GSM author_metadata.parquet files.

Usage:
    python tier_c_apply_cellxgene.py <h5ad_path> <mapping_json> <dataset_dir>
"""
import sys
import json
import anndata as ad
import pandas as pd
import numpy as np
import os

def main():
    h5ad_path = sys.argv[1]
    mapping_path = sys.argv[2]
    dataset_dir = sys.argv[3]

    print(f"Loading {h5ad_path} (backed mode)...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    print(f"Cells: {len(obs)}, Columns: {len(obs.columns)}")

    obs["bare_bc"] = obs.index.str.split("-").str[0]

    mapping = json.load(open(mapping_path))

    # Identify duplicate samples
    rev = {}
    for gsm, m in mapping.items():
        rev.setdefault(m["cx_sample"], []).append(gsm)

    # Columns to extract
    META_COLS = [
        "cell_type", "Author_Annotation", "Major_celltypes",
        "donor_id", "tissue", "assay", "disease", "sex",
        "development_stage", "APOE_class", "Brain.Region", "SORT",
        "Braak.stage", "Disease.Group", "Amyloid", "NP.Diagonis",
    ]
    meta_cols = [c for c in META_COLS if c in obs.columns]
    print(f"Extracting columns: {meta_cols}")

    # Build per-sample barcode index for fast lookup
    sample_groups = {}
    for s in obs["sample"].unique():
        mask = obs["sample"] == s
        sg = obs.loc[mask, ["bare_bc"] + meta_cols].copy()
        sg = sg.drop_duplicates(subset="bare_bc", keep="first")
        sg = sg.set_index("bare_bc")
        sample_groups[s] = sg

    success = 0
    failed = 0
    skipped_dup = 0
    results = []

    for gsm, m in sorted(mapping.items()):
        cx_sample = m["cx_sample"]
        gsms_for_sample = rev[cx_sample]

        if len(gsms_for_sample) > 1:
            skipped_dup += 1
            continue

        sg = sample_groups.get(cx_sample)
        if sg is None or len(sg) == 0:
            failed += 1
            continue

        cells_path = os.path.join(dataset_dir, gsm, "cells.parquet")
        if not os.path.exists(cells_path):
            failed += 1
            continue

        our_cells = pd.read_parquet(cells_path, columns=["barcode"])
        our_bcs = our_cells["barcode"].astype(str).tolist()
        our_bcs_set = set(our_bcs)

        # Align CellxGene metadata to our barcodes
        aligned = sg.reindex(sorted(our_bcs_set))
        n_matched = aligned[meta_cols[0]].notna().sum()
        match_rate = n_matched / len(our_bcs_set) if our_bcs_set else 0.0

        if n_matched == 0:
            failed += 1
            continue

        # Merge into existing author_metadata
        am_path = os.path.join(dataset_dir, gsm, "author_metadata.parquet")
        if os.path.exists(am_path):
            existing = pd.read_parquet(am_path)
            if "barcode" in existing.columns:
                existing = existing.set_index("barcode")
            # Add new columns (don't overwrite existing)
            new_cols = []
            for col in meta_cols:
                if col not in existing.columns:
                    existing[col] = aligned[col].reindex(existing.index)
                    new_cols.append(col)
            existing.reset_index().to_parquet(am_path, index=False)
        else:
            aligned.index.name = "barcode"
            aligned.reset_index().to_parquet(am_path, index=False)
            new_cols = meta_cols

        success += 1
        results.append(dict(
            gsm=gsm, cx_sample=cx_sample,
            n_matched=int(n_matched), n_ours=len(our_bcs_set),
            rate=round(match_rate, 4), new_cols=len(new_cols),
        ))

        if success % 25 == 0:
            print(f"  Processed {success}...")

    print(f"\nResults: {success} success, {failed} failed, {skipped_dup} skipped (duplicates)")

    if results:
        rdf = pd.DataFrame(results)
        print(f"Match rates: mean={rdf['rate'].mean():.3f}, median={rdf['rate'].median():.3f}")
        print(f"  >90%: {(rdf['rate'] > 0.9).sum()}")
        print(f"  >50%: {(rdf['rate'] > 0.5).sum()}")
        print(f"  >20%: {(rdf['rate'] > 0.2).sum()}")
        out_csv = os.path.join(os.path.dirname(mapping_path), "GSE263468_tier_c_results.csv")
        rdf.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
