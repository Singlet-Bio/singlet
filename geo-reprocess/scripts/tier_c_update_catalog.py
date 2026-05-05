#!/usr/bin/env python3
"""Update metadata catalog with Tier C results.

Usage:
    python tier_c_update_catalog.py <catalog_parquet> <gse_id> <dataset_dir> [--source SOURCE_LABEL]
"""
import sys
import pandas as pd
import os

def main():
    catalog_path = sys.argv[1]
    gse_id = sys.argv[2]
    dataset_dir = sys.argv[3]
    source_label = "cellxgene"
    if "--source" in sys.argv:
        source_label = sys.argv[sys.argv.index("--source") + 1]

    cat = pd.read_parquet(catalog_path)
    gse_mask = cat["gse_id"] == gse_id
    n_gsms = gse_mask.sum()
    print(f"Updating catalog for {gse_id}: {n_gsms} GSMs")

    META_COLS = [
        "cell_type", "Author_Annotation", "Major_celltypes",
        "donor_id", "tissue", "assay", "disease", "sex",
        "development_stage",
    ]

    updated = 0
    for idx in cat[gse_mask].index:
        gsm_id = cat.loc[idx, "gsm_id"]
        am_path = os.path.join(dataset_dir, gse_id, gsm_id, "author_metadata.parquet")
        if not os.path.exists(am_path):
            continue

        am = pd.read_parquet(am_path)
        # Count Tier C columns that have data
        tier_c_cols = [c for c in META_COLS if c in am.columns]
        if not tier_c_cols:
            continue

        # Compute match rate: fraction of cells with cell_type annotation
        ct_col = "cell_type" if "cell_type" in am.columns else tier_c_cols[0]
        n_total = len(am)
        n_annotated = am[ct_col].notna().sum()
        match_rate = n_annotated / n_total if n_total > 0 else 0.0

        # Update catalog: use stage2a fields for Tier C (since stage2a was 'skipped' before)
        cat.loc[idx, "stage2a_status"] = "done"
        cat.loc[idx, "stage2a_format"] = "h5ad"
        cat.loc[idx, "stage2a_source"] = f"cellxgene:{source_label}"
        cat.loc[idx, "stage2a_n_cols"] = len(tier_c_cols)
        cat.loc[idx, "stage2a_match_rate"] = round(match_rate, 4)
        cat.loc[idx, "updated_at"] = pd.Timestamp.utcnow().isoformat()
        updated += 1

    print(f"Updated {updated}/{n_gsms} GSMs in catalog")

    # Show summary
    gse_cat = cat[gse_mask]
    rates = gse_cat["stage2a_match_rate"].dropna()
    print(f"\nStage2a match_rate for {gse_id}:")
    print(f"  Mean: {rates.mean():.3f}")
    print(f"  Median: {rates.median():.3f}")
    print(f"  >90%: {(rates > 0.9).sum()}")
    print(f"  >50%: {(rates > 0.5).sum()}")
    print(f"  >0%: {(rates > 0).sum()}")

    # Save
    cat.to_parquet(catalog_path, index=False)
    print(f"\nSaved updated catalog to {catalog_path}")


if __name__ == "__main__":
    main()
