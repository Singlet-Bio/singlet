#!/usr/bin/env python3
"""
Build processing batch CSVs for non-GEO samples.

Generates batch files for samples with confirmed protocols (10xv2, 10xv3, dropseq)
that have ENA FASTQ URLs or SRR accessions for download.

Also generates a separate list of 10x_suspect samples for protocol resolution
and a Smart-seq2 inventory for future plate-based pipeline development.
"""

import os
from pathlib import Path

import pandas as pd

CATALOG = "/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet"
BATCH_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/batches_v10_non_geo"
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"

BATCH_COLS = [
    "gsm_id", "gse_id", "organism", "protocol_inferred",
    "ena_fastq_r1", "ena_fastq_r2", "srr_accessions", "read_count",
]

BATCH_SIZE = 50


def main():
    cat = pd.read_parquet(CATALOG)
    non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)].copy()
    print(f"Non-GEO samples in catalog: {len(non_geo):,}")

    # Filter to processable protocols
    processable_protocols = ["10xv2", "10xv3", "dropseq"]
    processable = non_geo[non_geo["protocol_inferred"].isin(processable_protocols)].copy()
    print(f"Processable (10x/dropseq): {len(processable):,}")

    # Filter to samples with FASTQ access (ENA URLs or SRR for fasterq-dump)
    has_fastq = (
        (processable["ena_fastq_r1"].fillna("").str.len() > 0) |
        (processable["srr_accessions"].fillna("").str.len() > 0)
    )
    processable = processable[has_fastq].copy()
    print(f"With FASTQ access: {len(processable):,}")

    # Filter to supported species (must have ref genome)
    processable = processable[processable["species_ref_genome"].fillna("").str.len() > 0].copy()
    print(f"With species ref genome: {len(processable):,}")

    # Exclude already-processed (check for existing quant dirs)
    already_processed = []
    for _, row in processable.iterrows():
        manifest = Path(QUANT_DIR) / row["gse_id"] / row["gsm_id"] / "sample_manifest.json"
        if manifest.exists():
            already_processed.append(row.name)

    if already_processed:
        processable = processable.drop(already_processed)
        print(f"After excluding already-processed: {len(processable):,}")

    # Sort by protocol tier for prioritized processing
    proto_priority = {"10xv3": 0, "10xv2": 1, "dropseq": 2}
    processable["_sort"] = processable["protocol_inferred"].map(proto_priority)
    processable = processable.sort_values(["_sort", "gse_id", "gsm_id"]).drop(columns=["_sort"])

    # Write batch CSVs
    outdir = Path(BATCH_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    batch_data = processable[BATCH_COLS].copy()
    n_batches = (len(batch_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(batch_data))
        batch = batch_data.iloc[start:end]
        batch_path = outdir / f"batch_{i:05d}.csv"
        batch.to_csv(batch_path, index=False)

    print(f"\nWrote {n_batches} batch files ({len(batch_data)} samples) to {outdir}")

    # Summary by protocol
    print(f"\nBatch summary:")
    for proto, cnt in batch_data["protocol_inferred"].value_counts().items():
        print(f"  {proto}: {cnt}")

    print(f"\nBatch summary by organism:")
    for org, cnt in batch_data["organism"].value_counts().items():
        print(f"  {org}: {cnt}")

    # Write 10x_suspect list for protocol resolution
    suspect = non_geo[non_geo["protocol_inferred"] == "10x_suspect"].copy()
    if len(suspect) > 0:
        suspect_path = outdir / "10x_suspect_for_resolution.csv"
        suspect[BATCH_COLS + ["library_layout"]].to_csv(suspect_path, index=False)
        print(f"\n10x_suspect samples for protocol resolution: {len(suspect)} -> {suspect_path}")

    # Write Smart-seq2 inventory
    ss2 = non_geo[non_geo["protocol_inferred"] == "smartseq2"].copy()
    if len(ss2) > 0:
        ss2_path = outdir / "smartseq2_inventory.csv"
        ss2_cols = ["gsm_id", "gse_id", "organism", "protocol_inferred",
                    "srr_accessions", "library_layout", "read_count", "notes"]
        ss2[ss2_cols].to_csv(ss2_path, index=False)
        print(f"Smart-seq2 inventory: {len(ss2):,} -> {ss2_path}")

    # Write unknown protocol inventory
    unknown = non_geo[non_geo["protocol_inferred"].isin(["unknown_sc", "unknown"])].copy()
    if len(unknown) > 0:
        unk_path = outdir / "unknown_protocol_inventory.csv"
        unk_cols = ["gsm_id", "gse_id", "organism", "protocol_inferred",
                    "library_layout", "srr_accessions", "read_count", "notes"]
        unknown[unk_cols].to_csv(unk_path, index=False)
        print(f"Unknown protocol inventory: {len(unknown):,} -> {unk_path}")

    # Write batch manifest
    import json
    manifest = {
        "batch_version": "v10_non_geo",
        "total_samples": len(batch_data),
        "n_batches": n_batches,
        "batch_size": BATCH_SIZE,
        "protocols": {k: int(v) for k, v in batch_data["protocol_inferred"].value_counts().items()},
        "organisms": {k: int(v) for k, v in batch_data["organism"].value_counts().items()},
        "pending_resolution": {
            "10x_suspect": int(len(suspect)),
            "smartseq2": int(len(ss2)),
            "unknown_sc_or_unknown": int(len(unknown)),
        },
        "notes": [
            "Tier 1+2 non-GEO samples from HCA, CellxGene, SCEA E-MTAB",
            "Smart-seq2 samples need plate-based pipeline (not simpleaf)",
            "10x_suspect samples need R1 read-length verification",
            "unknown_sc/unknown samples need further protocol investigation"
        ]
    }
    with open(outdir / "batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nBatch manifest written to {outdir / 'batch_manifest.json'}")


if __name__ == "__main__":
    main()
