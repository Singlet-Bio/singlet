#!/usr/bin/env python3
from __future__ import annotations
"""Build per-GSE kraken2.1pz matrices from per-GSM kraken2_cell_taxa.parquet.

Phase 4 of catalog v1.0 MASTER_PLAN: standardized microbiome count matrices
alongside expression data, stored in the same per-GSE directory.

For each GSE directory (post-merge):
  1. Collect all kraken2_cell_taxa.parquet files from GSM subdirs
  2. Union all taxon_ids → feature dimension
  3. Build sparse matrix: taxa (rows) × cells (columns)
  4. Write as kraken2.1pz with same cell ordering as counts.1pz
  5. Write kraken2_features.parquet with taxon metadata

Usage:
  python build_kraken2.py GSE128553
  python build_kraken2.py --task-id 42 --n-tasks 100    # SLURM array
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as ss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "singlepress"))
import singlepress

QUANT_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")


def build_kraken2_matrix(gse_dir: Path) -> dict:
    """Build a kraken2.1pz matrix for a single GSE directory.

    Reads the merged provenance.json to get GSM ordering and cell counts,
    then assembles kraken2 data in the same cell order as counts.1pz.
    """
    report = {"gse_id": gse_dir.name, "status": "unknown"}

    # Read provenance to get GSM ordering
    provenance_path = gse_dir / "provenance.json"
    if not provenance_path.exists():
        report["status"] = "skip"
        report["error"] = "No provenance.json (not merged yet)"
        return report

    with open(provenance_path) as f:
        provenance = json.load(f)

    source_gsms = provenance.get("source_gsms", {})
    if not source_gsms:
        report["status"] = "skip"
        report["error"] = "No source GSMs in provenance"
        return report

    # Collect kraken2 data from each GSM, maintaining cell ordering
    all_taxon_ids = set()
    gsm_data = {}  # gsm_id -> DataFrame of (barcode, taxon_id, umi_count)

    for gsm_id in source_gsms:
        k2_path = gse_dir / gsm_id / "kraken2_cell_taxa.parquet"
        if not k2_path.exists():
            continue
        df = pq.read_table(k2_path).to_pandas()
        all_taxon_ids.update(df["taxon_id"].unique())
        gsm_data[gsm_id] = df

    if not gsm_data:
        report["status"] = "skip"
        report["error"] = "No kraken2 data found"
        return report

    # Sort taxon IDs for consistent feature ordering
    taxon_ids = sorted(all_taxon_ids)
    taxon_to_idx = {tid: i for i, tid in enumerate(taxon_ids)}
    n_taxa = len(taxon_ids)

    # Build sparse matrix: taxa (rows) × cells (columns)
    # Cell ordering must match counts.1pz (GSMs in order, cells within GSM in order)
    row_indices = []
    col_indices = []
    values = []
    col_offset = 0
    gsm_cell_counts = {}

    for gsm_id, gsm_info in source_gsms.items():
        n_cells_gsm = gsm_info["n_cells"]

        if gsm_id in gsm_data:
            df = gsm_data[gsm_id]
            # Map barcodes to column indices within this GSM
            # Barcodes in kraken2 should match cell_metadata order
            barcode_path = gse_dir / gsm_id / "cell_metadata.parquet"
            if barcode_path.exists():
                barcodes_df = pq.read_table(barcode_path, columns=["barcode"]).to_pandas()
                barcode_list = barcodes_df["barcode"].tolist()
                barcode_to_local = {b: i for i, b in enumerate(barcode_list)}
            else:
                # Fallback: unique barcodes in order of appearance
                barcode_list = df["barcode"].unique().tolist()
                barcode_to_local = {b: i for i, b in enumerate(barcode_list)}

            for _, row in df.iterrows():
                bc = row["barcode"]
                local_col = barcode_to_local.get(bc)
                if local_col is not None and local_col < n_cells_gsm:
                    row_indices.append(taxon_to_idx[row["taxon_id"]])
                    col_indices.append(col_offset + local_col)
                    values.append(int(row["umi_count"]))

        gsm_cell_counts[gsm_id] = n_cells_gsm
        col_offset += n_cells_gsm

    total_cells = col_offset

    # Build COO then convert to CSC
    if values:
        mat = ss.coo_matrix(
            (np.array(values, dtype=np.int32),
             (np.array(row_indices, dtype=np.int32),
              np.array(col_indices, dtype=np.int32))),
            shape=(n_taxa, total_cells),
        ).tocsc()
    else:
        mat = ss.csc_matrix((n_taxa, total_cells), dtype=np.int32)

    # Write kraken2.1pz
    taxon_names = [str(tid) for tid in taxon_ids]
    stats = singlepress.write_1pz(
        str(gse_dir / "kraken2.1pz"),
        mat,
        rownames=taxon_names,
        uns={
            "gse_id": gse_dir.name,
            "type": "kraken2_cell_taxa",
            "n_taxa": str(n_taxa),
            "n_cells": str(total_cells),
        },
        num_threads=4,
        level=3,
    )

    # Write kraken2_features.parquet
    features_df = pd.DataFrame({
        "taxon_id": taxon_ids,
        "name": taxon_names,  # NCBI taxonomy name lookup would improve this
    })
    features_df.to_parquet(str(gse_dir / "kraken2_features.parquet"), index=False)

    report["status"] = "ok"
    report["n_taxa"] = n_taxa
    report["n_cells"] = total_cells
    report["nnz"] = int(mat.nnz)
    report["n_gsms_with_kraken2"] = len(gsm_data)
    report["n_gsms_total"] = len(source_gsms)
    return report


def get_merged_gse_dirs() -> list[Path]:
    """Find all GSE directories that have been merged (have provenance.json)."""
    dirs = []
    for entry in sorted(QUANT_DIR.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("GSE"):
            continue
        if (entry / "provenance.json").exists():
            dirs.append(entry)
        # Also check species subdirectories
        for sub in entry.iterdir():
            if sub.is_dir() and not sub.name.startswith("GSM") and (sub / "provenance.json").exists():
                dirs.append(sub)
    return dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gse_ids", nargs="*", help="GSE IDs to process")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--n-tasks", type=int)
    args = parser.parse_args()

    if args.gse_ids:
        gse_dirs = [QUANT_DIR / gid for gid in args.gse_ids]
    elif args.task_id is not None and args.n_tasks is not None:
        all_dirs = get_merged_gse_dirs()
        gse_dirs = [d for j, d in enumerate(all_dirs) if j % args.n_tasks == args.task_id]
        print(f"SLURM task {args.task_id}/{args.n_tasks}: {len(gse_dirs)} GSEs",
              file=sys.stderr)
    else:
        parser.error("Provide GSE IDs or --task-id/--n-tasks")

    reports = []
    for i, gse_dir in enumerate(gse_dirs):
        t0 = time.time()
        try:
            report = build_kraken2_matrix(gse_dir)
        except Exception as e:
            report = {"gse_id": gse_dir.name, "status": "error", "error": str(e)}
        reports.append(report)
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(gse_dirs)}] {report['gse_id']}: {report['status']} "
              f"({elapsed:.1f}s)", file=sys.stderr)

    n_ok = sum(1 for r in reports if r["status"] == "ok")
    n_err = sum(1 for r in reports if r["status"] == "error")
    n_skip = sum(1 for r in reports if r["status"] == "skip")
    print(f"\nOK: {n_ok}, Errors: {n_err}, Skipped: {n_skip}", file=sys.stderr)
    json.dump(reports, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
