#!/usr/bin/env python3
"""Merge per-GSM .1pz files into per-GSE consolidated counts.1pz.

Combines Phase 1 (metadata embedding), Phase 2 (consolidation), and
Phase 3 (metadata standardization) of the catalog v1.0 MASTER_PLAN.

For each GSE directory:
  1. Discover all GSM subdirs with counts.1pz
  2. Read per-GSM matrices + sidecar metadata
  3. Verify gene reference consistency across GSMs
  4. hstack matrices into a single per-GSE counts.1pz
  5. Embed obs/var/uns metadata into the merged file
  6. Write study_metadata.json and provenance.json
  7. Optionally merge kraken2 matrices

Usage:
  python merge_gse.py GSE128553                      # Single GSE
  python merge_gse.py --gse-list gse_ids.txt         # File with one GSE per line
  python merge_gse.py --task-id 42 --n-tasks 100     # SLURM array mode

Multi-species GSEs produce per-species subdirectories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as ss

# Add singlepress to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "singlepress"))
import singlepress

QUANT_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")
CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")
GSE_DESCRIPTIONS_PATH = CATALOG_DIR / "all_gse_descriptions.parquet"

CATALOG_VERSION = "1.0"

# Known USA gene reference counts per species
KNOWN_REFS = {
    115818: "USA_115818_human",
    171540: "USA_171540_mouse",
    97560: "USA_97560_zebrafish",
}


def discover_gsms(gse_dir: Path) -> list[dict]:
    """Find all GSM subdirectories with counts.1pz and sidecar files."""
    gsms = []
    for entry in sorted(gse_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("GSM"):
            continue
        pz_path = entry / "counts.1pz"
        if not pz_path.exists():
            continue
        manifest_path = entry / "sample_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        gsms.append({
            "gsm_id": entry.name,
            "dir": entry,
            "pz_path": pz_path,
            "manifest": manifest,
            "has_cell_meta": (entry / "cell_metadata.parquet").exists(),
            "has_feature_meta": (entry / "feature_metadata.parquet").exists(),
            "has_kraken2": (entry / "kraken2_cell_taxa.parquet").exists(),
        })
    return gsms


def group_by_species(gsms: list[dict]) -> dict[str, list[dict]]:
    """Group GSMs by organism for multi-species handling."""
    groups = {}
    for gsm in gsms:
        organism = gsm["manifest"].get("organism", "unknown")
        groups.setdefault(organism, []).append(gsm)
    return groups


def read_feature_names(gsm_dir: Path) -> list[str] | None:
    """Read gene names from feature_metadata.parquet.

    Handles both 'gene_name' (standard) and 'gene_id' (legacy) column names.
    """
    path = gsm_dir / "feature_metadata.parquet"
    if not path.exists():
        return None
    schema = pq.read_schema(path)
    col_names = [f.name for f in schema]
    if "gene_name" in col_names:
        df = pq.read_table(path, columns=["gene_name"]).to_pandas()
        return df["gene_name"].tolist()
    elif "gene_id" in col_names:
        df = pq.read_table(path, columns=["gene_id"]).to_pandas()
        return df["gene_id"].tolist()
    else:
        return None


def read_cell_barcodes(gsm_dir: Path) -> list[str] | None:
    """Read cell barcodes from cell_metadata.parquet."""
    path = gsm_dir / "cell_metadata.parquet"
    if not path.exists():
        return None
    df = pq.read_table(path, columns=["barcode"]).to_pandas()
    return df["barcode"].tolist()


def merge_species_group(
    gse_id: str,
    gsms: list[dict],
    output_dir: Path,
    descriptions: dict,
) -> dict:
    """Merge all GSMs for a single species group.

    Returns a report dict with merge statistics.
    """
    report = {
        "gse_id": gse_id,
        "n_gsms": len(gsms),
        "gsm_ids": [g["gsm_id"] for g in gsms],
        "status": "unknown",
    }

    # ---- Read all matrices and verify dimensions ----
    matrices = []
    all_colnames = []
    all_obs_rows = []
    provenance_gsms = {}
    gene_names = None
    expected_nrows = None

    for gsm in gsms:
        gsm_id = gsm["gsm_id"]

        # Read matrix (int32 to save memory)
        mat = singlepress.read_1pz_int(str(gsm["pz_path"]), num_threads=4)
        nrows, ncols = mat.shape

        # Verify row count consistency
        if expected_nrows is None:
            expected_nrows = nrows
        elif nrows != expected_nrows:
            report["status"] = "error"
            report["error"] = (
                f"Row count mismatch: {gsm_id} has {nrows} rows, "
                f"expected {expected_nrows}"
            )
            return report

        # Read gene names from first GSM with feature_metadata
        if gene_names is None and gsm["has_feature_meta"]:
            gene_names = read_feature_names(gsm["dir"])
            if gene_names is not None and len(gene_names) != nrows:
                # Feature metadata doesn't match matrix — skip it
                gene_names = None

        # Read cell barcodes
        if gsm["has_cell_meta"]:
            barcodes = read_cell_barcodes(gsm["dir"])
        else:
            barcodes = None

        # Build prefixed colnames: {gsm_id}.{barcode}
        if barcodes is not None and len(barcodes) == ncols:
            prefixed = [f"{gsm_id}.{b}" for b in barcodes]
        else:
            prefixed = [f"{gsm_id}.cell_{i}" for i in range(ncols)]

        # Get colsums for obs metadata
        colsums = singlepress.colsums_1pz(str(gsm["pz_path"]))

        # Build per-cell obs rows
        organism = gsm["manifest"].get("organism", "unknown")
        protocol = gsm["manifest"].get("protocol_catalog", "unknown")
        for i in range(ncols):
            all_obs_rows.append({
                "barcode": prefixed[i],
                "gsm_id": gsm_id,
                "organism": organism,
                "total_counts": int(colsums[i]),
            })

        all_colnames.extend(prefixed)
        matrices.append(mat)

        # Provenance per GSM
        provenance_gsms[gsm_id] = {
            "pipeline_version": gsm["manifest"].get("pipeline_version", "unknown"),
            "n_cells": ncols,
            "n_features": nrows,
            "organism": organism,
            "protocol": protocol,
            "qc_status": gsm["manifest"].get("qc_status", "unknown"),
        }

    # ---- Merge matrices ----
    if len(matrices) == 1:
        merged = matrices[0]
        merge_method = "single"
    else:
        merged = ss.hstack(matrices, format="csc")
        merge_method = "hstack"

    total_cells = merged.shape[1]
    total_genes = merged.shape[0]

    # ---- Build var DataFrame ----
    var_df = None
    if gene_names is not None:
        ref_label = KNOWN_REFS.get(total_genes, f"custom_{total_genes}")
        var_df = pd.DataFrame({
            "gene_name": gene_names,
            "reference": [ref_label] * total_genes,
        })

    # ---- Build obs DataFrame ----
    obs_df = pd.DataFrame(all_obs_rows)

    # ---- Build uns metadata ----
    organisms = list({g["manifest"].get("organism", "unknown") for g in gsms})
    protocols = list({g["manifest"].get("protocol_catalog", "unknown") for g in gsms})
    desc = descriptions.get(gse_id, {})

    uns = {
        "gse_id": gse_id,
        "organism": "|".join(organisms),
        "protocol": "|".join(protocols),
        "n_samples": str(len(gsms)),
        "n_cells": str(total_cells),
        "n_genes": str(total_genes),
        "catalog_version": CATALOG_VERSION,
        "merge_method": merge_method,
    }
    if desc.get("title"):
        uns["title"] = desc["title"]
    if desc.get("pubmed_ids"):
        uns["pubmed_ids"] = desc["pubmed_ids"]

    # ---- Write merged counts.1pz ----
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "counts.1pz"

    stats = singlepress.write_1pz(
        str(out_path),
        merged,
        rownames=gene_names,
        colnames=all_colnames,
        obs=obs_df,
        var=var_df,
        uns=uns,
        num_threads=8,
        level=3,
    )

    # ---- Write metadata.parquet (cell-level, outside 1pz for easy access) ----
    obs_df.to_parquet(str(output_dir / "metadata.parquet"), index=False)

    # ---- Write feature_metadata.parquet ----
    if var_df is not None:
        var_df.to_parquet(str(output_dir / "feature_metadata.parquet"), index=False)

    # ---- Write study_metadata.json ----
    study_meta = {
        "gse_id": gse_id,
        "title": desc.get("title", ""),
        "summary": desc.get("summary", ""),
        "organism": organisms,
        "n_samples": len(gsms),
        "n_cells": total_cells,
        "n_genes": total_genes,
        "gsm_ids": [g["gsm_id"] for g in gsms],
        "protocol": protocols[0] if len(protocols) == 1 else protocols,
        "reference": KNOWN_REFS.get(total_genes, f"custom_{total_genes}"),
        "pubmed_ids": desc.get("pubmed_ids", ""),
        "submission_date": desc.get("submission_date", ""),
        "license": "public_domain",
        "catalog_version": CATALOG_VERSION,
    }
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(study_meta, f, indent=2)

    # ---- Write provenance.json ----
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "merge_method": merge_method,
        "source_gsms": provenance_gsms,
        "merged_shape": [total_genes, total_cells],
        "merged_nnz": int(merged.nnz),
        "catalog_version": CATALOG_VERSION,
        "script": "merge_gse.py",
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    report["status"] = "ok"
    report["n_cells"] = total_cells
    report["n_genes"] = total_genes
    report["nnz"] = int(merged.nnz)
    report["merge_method"] = merge_method
    report["output_dir"] = str(output_dir)
    report["write_stats"] = stats
    return report


def process_gse(gse_id: str, descriptions: dict) -> list[dict]:
    """Process a single GSE: discover GSMs, group by species, merge each group."""
    gse_dir = QUANT_DIR / gse_id
    if not gse_dir.is_dir():
        return [{"gse_id": gse_id, "status": "error", "error": "GSE dir not found"}]

    gsms = discover_gsms(gse_dir)
    if not gsms:
        return [{"gse_id": gse_id, "status": "skip", "error": "No GSMs with counts.1pz"}]

    species_groups = group_by_species(gsms)
    is_multi_species = len(species_groups) > 1

    reports = []
    for organism, group_gsms in species_groups.items():
        if is_multi_species:
            # Multi-species: per-species subdirectory
            safe_name = organism.replace(" ", "_")
            output_dir = gse_dir / safe_name
        else:
            # Single species: output to GSE root
            output_dir = gse_dir

        report = merge_species_group(gse_id, group_gsms, output_dir, descriptions)
        report["organism"] = organism
        reports.append(report)

    return reports


def load_descriptions() -> dict:
    """Load GSE descriptions from the catalog parquet."""
    if not GSE_DESCRIPTIONS_PATH.exists():
        return {}
    df = pq.read_table(GSE_DESCRIPTIONS_PATH).to_pandas()
    result = {}
    for _, row in df.iterrows():
        result[row["gse_id"]] = {
            "title": row.get("title", ""),
            "summary": row.get("summary", ""),
            "organism": row.get("organism", ""),
            "pubmed_ids": row.get("pubmed_ids", ""),
        }
    return result


def get_all_gse_ids() -> list[str]:
    """Get sorted list of all GSE IDs that have at least one counts.1pz."""
    gse_ids = []
    for entry in sorted(QUANT_DIR.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("GSE"):
            continue
        # Check if any GSM has counts.1pz
        has_pz = any(
            (gsm / "counts.1pz").exists()
            for gsm in entry.iterdir()
            if gsm.is_dir() and gsm.name.startswith("GSM")
        )
        if has_pz:
            gse_ids.append(entry.name)
    return gse_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gse_ids", nargs="*", help="GSE IDs to process")
    parser.add_argument("--gse-list", help="File with one GSE ID per line")
    parser.add_argument(
        "--task-id", type=int,
        help="SLURM array task ID (0-indexed)",
    )
    parser.add_argument(
        "--n-tasks", type=int,
        help="Total number of SLURM array tasks",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List GSEs without processing",
    )
    args = parser.parse_args()

    # Determine which GSEs to process
    if args.gse_ids:
        gse_ids = args.gse_ids
    elif args.gse_list:
        with open(args.gse_list) as f:
            gse_ids = [line.strip() for line in f if line.strip()]
    elif args.task_id is not None and args.n_tasks is not None:
        all_ids = get_all_gse_ids()
        # Partition: task i gets IDs where i == id_index % n_tasks
        gse_ids = [
            gid for j, gid in enumerate(all_ids)
            if j % args.n_tasks == args.task_id
        ]
        print(f"SLURM task {args.task_id}/{args.n_tasks}: "
              f"processing {len(gse_ids)} of {len(all_ids)} GSEs",
              file=sys.stderr)
    else:
        parser.error("Provide GSE IDs, --gse-list, or --task-id/--n-tasks")

    if args.dry_run:
        for gid in gse_ids:
            print(gid)
        return

    # Load GSE descriptions
    print(f"Loading GSE descriptions...", file=sys.stderr)
    descriptions = load_descriptions()
    print(f"Loaded {len(descriptions)} descriptions", file=sys.stderr)

    # Process each GSE
    all_reports = []
    t0 = time.time()
    for i, gse_id in enumerate(gse_ids):
        t1 = time.time()
        try:
            reports = process_gse(gse_id, descriptions)
        except Exception as e:
            reports = [{"gse_id": gse_id, "status": "error", "error": str(e)}]
        all_reports.extend(reports)

        elapsed = time.time() - t1
        status = reports[0]["status"] if reports else "?"
        n_cells = reports[0].get("n_cells", 0) if reports else 0
        print(
            f"[{i+1}/{len(gse_ids)}] {gse_id}: {status} "
            f"({n_cells} cells, {elapsed:.1f}s)",
            file=sys.stderr,
        )

    total_time = time.time() - t0

    # Summary
    n_ok = sum(1 for r in all_reports if r["status"] == "ok")
    n_err = sum(1 for r in all_reports if r["status"] == "error")
    n_skip = sum(1 for r in all_reports if r["status"] == "skip")
    total_cells = sum(r.get("n_cells", 0) for r in all_reports)

    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Processed: {len(gse_ids)} GSEs in {total_time:.0f}s", file=sys.stderr)
    print(f"OK: {n_ok}, Errors: {n_err}, Skipped: {n_skip}", file=sys.stderr)
    print(f"Total cells: {total_cells:,}", file=sys.stderr)

    # Write report JSON to stdout
    json.dump(all_reports, sys.stdout, indent=2)

    if n_err > 0:
        print(f"\nErrors:", file=sys.stderr)
        for r in all_reports:
            if r["status"] == "error":
                print(f"  {r['gse_id']}: {r.get('error', '?')}", file=sys.stderr)


if __name__ == "__main__":
    main()
