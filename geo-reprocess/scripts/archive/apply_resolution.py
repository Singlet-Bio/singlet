#!/usr/bin/env python3
"""
Apply protocol resolution results to the catalog and rebuild batches.

Reads resolution parquet files from dry-run results, updates catalog
protocol_inferred, handles R1/R2 swapped samples, and validates ENA URLs.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CATALOG = "/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet"
OUTDIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")


def load_resolution_results():
    """Load all resolution results from parquet files."""
    results_path = OUTDIR / "protocol_resolution_results.parquet"
    if results_path.exists():
        return pd.read_parquet(results_path)
    return pd.DataFrame()


def apply_resolution(cat, results):
    """Apply protocol resolution to catalog.

    Updates protocol_inferred for resolved samples and handles swapped reads.
    """
    if results.empty:
        print("No resolution results to apply.")
        return cat, {}

    stats = Counter()

    # Map index to resolution
    resolved = results[~results["protocol"].isin(["peek_failed", "no_url", "unknown_se", "ambiguous"])]
    print(f"Total resolved samples: {len(resolved):,}")
    print(f"  Droplet: {(resolved['mode'] == 'droplet').sum():,}")
    print(f"  Smart-seq2: {(resolved['mode'] == 'smartseq').sum():,}")

    for _, row in resolved.iterrows():
        idx = row["catalog_index"]
        protocol = row["protocol"]
        mode = row["mode"]

        if idx not in cat.index:
            stats["not_in_catalog"] += 1
            continue

        old_protocol = cat.at[idx, "protocol_inferred"]

        # Handle swapped reads — swap R1 and R2 URLs
        if "_swapped" in protocol:
            base_protocol = protocol.replace("_swapped", "")
            old_r1 = cat.at[idx, "ena_fastq_r1"]
            old_r2 = cat.at[idx, "ena_fastq_r2"]
            cat.at[idx, "ena_fastq_r1"] = old_r2
            cat.at[idx, "ena_fastq_r2"] = old_r1
            cat.at[idx, "protocol_inferred"] = base_protocol
            cat.at[idx, "notes"] = str(cat.at[idx, "notes"]) + ";reads_swapped"
            stats["swapped"] += 1
        else:
            cat.at[idx, "protocol_inferred"] = protocol
            stats[f"resolved_{mode}"] += 1

    return cat, stats


def validate_ena_urls(cat):
    """Flag fabricated ENA URLs that are known to 404.

    ENA submissions that only have BAM files (no FASTQ) will have URLs
    constructed from SRR accessions but these are fabricated and will 404.
    Mark these samples with empty ENA URLs so the pipeline uses SRA fallback.

    Returns number of samples with cleared URLs.
    """
    non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]
    
    # Samples where peek_failed in resolution results typically have bad URLs
    results_path = OUTDIR / "protocol_resolution_results.parquet"
    if not results_path.exists():
        return 0

    results = pd.read_parquet(results_path)
    failed = results[results["protocol"] == "peek_failed"]

    cleared = 0
    for _, row in failed.iterrows():
        idx = row["catalog_index"]
        if idx in cat.index:
            r1 = cat.at[idx, "ena_fastq_r1"]
            if r1 and isinstance(r1, str) and len(r1) > 10:
                cat.at[idx, "ena_fastq_r1"] = ""
                cat.at[idx, "ena_fastq_r2"] = ""
                cleared += 1

    return cleared


def rebuild_batches(cat):
    """Rebuild non-GEO batch CSVs for processable droplet samples."""
    non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]

    # Droplet protocols that we can process
    droplet_protocols = ["10xv2", "10xv3", "10x_other", "dropseq"]
    processable = non_geo[non_geo["protocol_inferred"].isin(droplet_protocols)]

    # Need SRR accession or valid ENA URL
    has_srr = processable["srr_accessions"].notna() & (processable["srr_accessions"] != "")
    has_ena = processable["ena_fastq_r1"].notna() & (processable["ena_fastq_r1"] != "")
    processable = processable[has_srr | has_ena]

    # Exclude controlled access
    if "controlled_access" in processable.columns:
        processable = processable[processable["controlled_access"] != True]

    print(f"\nProcessable non-GEO droplet samples: {len(processable):,}")
    print(f"  By protocol: {dict(processable['protocol_inferred'].value_counts())}")
    print(f"  By species: {dict(processable['organism'].value_counts().head(10))}")

    # Write batch CSVs (50 samples each)
    batch_dir = OUTDIR / "batches_v11_resolved"
    batch_dir.mkdir(exist_ok=True)

    # Clear existing batches
    for old in batch_dir.glob("*.csv"):
        old.unlink()

    batch_size = 50
    batch_num = 0
    for start in range(0, len(processable), batch_size):
        batch = processable.iloc[start:start + batch_size]
        batch_num += 1
        batch_file = batch_dir / f"batch_{batch_num:03d}.csv"

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "gsm_id": row.get("gsm_id", row.get("experiment_accession", "")),
                "gse_id": row.get("gse_id", row.get("study_accession", "")),
                "organism": row["organism"],
                "protocol": row["protocol_inferred"],
                "ena_r1": row.get("ena_fastq_r1", ""),
                "ena_r2": row.get("ena_fastq_r2", ""),
                "srr": row.get("srr_accessions", ""),
            })

        pd.DataFrame(rows).to_csv(batch_file, index=False)

    print(f"  Wrote {batch_num} batch files to {batch_dir}")
    return batch_num, len(processable)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Don't write catalog")
    parser.add_argument("--skip-url-validation", action="store_true", help="Don't clear bad URLs")
    args = parser.parse_args()

    print("Loading catalog...")
    cat = pd.read_parquet(CATALOG)
    print(f"  {len(cat):,} rows")

    # Load and apply resolution results
    results = load_resolution_results()
    if not results.empty:
        print(f"\nApplying {len(results):,} resolution results...")
        cat, stats = apply_resolution(cat, results)
        print(f"  Stats: {dict(stats)}")
    else:
        print("No resolution results found.")

    # Validate ENA URLs
    if not args.skip_url_validation:
        print("\nValidating ENA URLs...")
        cleared = validate_ena_urls(cat)
        print(f"  Cleared {cleared} fabricated URLs")

    # Summary of non-GEO protocols after update
    non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]
    print(f"\nNon-GEO protocol distribution after update:")
    print(non_geo["protocol_inferred"].value_counts().to_string())

    # Rebuild batches
    n_batches, n_samples = rebuild_batches(cat)

    if args.dry_run:
        print(f"\nDRY RUN — catalog NOT updated")
    else:
        # Backup first
        backup = str(CATALOG).replace(".parquet", "_pre_resolution.parquet")
        import shutil
        shutil.copy2(CATALOG, backup)
        print(f"\nBackup: {backup}")

        cat.to_parquet(CATALOG, index=False)
        print(f"Catalog updated: {len(cat):,} rows")

    print(f"\nDone: {n_samples} processable samples in {n_batches} batches")


if __name__ == "__main__":
    main()
