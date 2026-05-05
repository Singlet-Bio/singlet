#!/usr/bin/env python3
"""Reconcile processing_catalog with on-disk sample manifests.

Scans all quant/{GSE}/{GSM}/sample_manifest.json files and updates
processing_catalog.parquet to reflect actual on-disk state. Also computes
per-phase eligibility columns (eligible_1, eligible_2a, ...) for fast
grab_batch filtering.

Permanent failure categories (NOT eligible for retry):
  - skip_plate_based, skip_vdj (wrong assay type)
  - fail_no_r2 (structural — no R2 fastq available)
  - fail_qc_low_genes, fail_qc_few_cells, fail_qc_other (data quality)
  - fail_low_mapping (wrong organism/reference)
  - fail_simpleaf_permit (too few reads in permit list)

Retriable:
  - fail_download (network), fail_simpleaf_timeout (resource),
  - fail_simpleaf_map (may work with different chemistry),
  - fail_simpleaf_other (unknown), retry_skipped

Usage:
    python reconcile_catalog.py                 # Dry-run: report what would change
    python reconcile_catalog.py --apply         # Write updated catalog
    python reconcile_catalog.py --apply --force # Overwrite even if recent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")
QUANT_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")
CATALOG_PATH = CATALOG_DIR / "processing_catalog.parquet"
BACKUP_PATH = CATALOG_DIR / "processing_catalog_backup.parquet"

# Permanent failures that should NOT be retried
PERMANENT_FAILURES = {
    "skip_plate_based",
    "skip_vdj",
    "fail_no_r2",
    "fail_qc_low_genes",
    "fail_qc_few_cells",
    "fail_qc_other",
    "fail_low_mapping",
    "fail_no_mapping",
    "fail_simpleaf_permit",
}

# Retriable failures
RETRIABLE_FAILURES = {
    "fail_download",
    "fail_simpleaf_timeout",
    "fail_simpleaf_map",
    "fail_simpleaf_other",
    "retry_skipped",
}


def scan_manifests() -> dict[str, dict]:
    """Scan all sample_manifest.json files in quant dir.

    Returns {gsm_id: {status, qc_status, n_cells, ...}}
    """
    manifests = {}
    n_scanned = 0
    n_errors = 0
    t0 = time.time()

    for gse_dir in sorted(QUANT_DIR.iterdir()):
        if not gse_dir.is_dir():
            continue
        for gsm_dir in gse_dir.iterdir():
            if not gsm_dir.is_dir():
                continue
            manifest_path = gsm_dir / "sample_manifest.json"
            if not manifest_path.exists():
                continue
            n_scanned += 1
            try:
                with open(manifest_path) as f:
                    m = json.load(f)
                gsm_id = m.get("gsm_id", gsm_dir.name)
                manifests[gsm_id] = {
                    "status": m.get("status", "unknown"),
                    "qc_status": m.get("qc_status", ""),
                    "n_cells": m.get("n_cells", 0),
                    "n_genes": m.get("n_genes", 0),
                    "mapping_rate": m.get("mapping_rate"),
                    "protocol_detected": m.get("protocol_detected", ""),
                    "chemistry_used": m.get("chemistry_used", ""),
                    "error": m.get("error", ""),
                    "has_1pz": (gsm_dir / "counts.1pz").exists(),
                    "pipeline_version": m.get("pipeline_version", ""),
                }
            except (json.JSONDecodeError, OSError):
                n_errors += 1

    elapsed = time.time() - t0
    print(f"Scanned {n_scanned:,} manifests in {elapsed:.1f}s ({n_errors} errors)")
    return manifests


def classify_manifest(m: dict) -> str:
    """Map manifest status to processing_status value."""
    status = m["status"]
    qc = m["qc_status"]

    if status == "success" and qc in ("qc_pass", "qc_warn"):
        return "done" if qc == "qc_pass" else "done_qc_warn"
    elif status == "success" and qc == "qc_fail":
        # QC failed but quantification succeeded — mark as QC failure
        return "fail_qc_other"
    elif status == "skipped":
        error = m.get("error", "").lower()
        if "plate" in error or "smartseq" in error:
            return "skip_plate_based"
        elif "vdj" in error:
            return "skip_vdj"
        return "skip_reclassified"
    elif status == "failed":
        error = m.get("error", "").lower()
        if "no r2" in error or "single-end" in error:
            return "fail_no_r2"
        elif "zero mapping rate" in error or "no transcriptome content" in error:
            return "fail_no_mapping"
        elif "permit" in error or "permit-list" in error:
            return "fail_simpleaf_permit"
        elif "low genes" in error:
            return "fail_qc_low_genes"
        elif "few cells" in error or "too few cells" in error:
            return "fail_qc_few_cells"
        elif "mapping" in error and ("low" in error or "rate" in error):
            return "fail_low_mapping"
        elif "download" in error or "fasterq" in error or "sra" in error:
            return "fail_download"
        elif "timeout" in error or "timed out" in error:
            return "fail_simpleaf_timeout"
        elif "simpleaf" in error or "alevin" in error or "piscem" in error:
            return "fail_simpleaf_map"
        elif "protocol" in error or "detect" in error:
            return "fail_protocol_detect"
        elif "qc" in error:
            return "fail_qc_other"
        return "fail_simpleaf_other"
    return "unknown"


def compute_all_eligible(pc: pd.DataFrame) -> pd.DataFrame:
    """Compute per-phase eligibility columns.

    Imports phase filters from grab_batch.py so there's a single source of truth.
    Adds columns: eligible_1, eligible_2a, eligible_2b, eligible_2c,
                  eligible_2d, eligible_3, eligible_4a, eligible_4b
    Also retains v12_eligible as alias for eligible_1 (backward compat).
    """
    # Import phase filters from the canonical source
    sys.path.insert(0, str(Path(__file__).parent))
    from grab_batch import PHASE_FILTERS

    for phase_id, filter_fn in PHASE_FILTERS.items():
        col = f"eligible_{phase_id}"
        pc[col] = filter_fn(pc)
        n = pc[col].sum()
        print(f"  {col}: {n:,}")

    # Backward compat alias
    pc["v12_eligible"] = pc["eligible_1"]

    return pc


def reconcile(apply: bool = False, force: bool = False):
    """Run the full reconciliation."""
    print("=" * 60)
    print("  Catalog Reconciliation")
    print("=" * 60)
    print()

    # Load catalog
    print("Loading processing catalog...")
    pc = pd.read_parquet(CATALOG_PATH)
    print(f"  Catalog rows: {len(pc):,}")
    print()

    # Scan manifests
    print("Scanning on-disk manifests...")
    manifests = scan_manifests()
    print(f"  Manifests found: {len(manifests):,}")
    print()

    # Match and reconcile
    print("Building GSM index...")
    t0 = time.time()
    gsm_to_idx = {}
    for i, gsm in enumerate(pc["gsm_id"].values):
        if gsm not in gsm_to_idx:  # keep first occurrence
            gsm_to_idx[gsm] = i
    print(f"  Indexed {len(gsm_to_idx):,} unique GSMs in {time.time() - t0:.1f}s")
    print()

    print("Reconciling statuses...")
    status_updates = Counter()
    changes = []
    n_matched = 0
    n_already_done = 0
    n_unknown = 0
    t0 = time.time()

    for gsm_id, m in manifests.items():
        i = gsm_to_idx.get(gsm_id)
        if i is None:
            continue
        n_matched += 1

        old_status = pc.at[i, "processing_status"]
        new_status = classify_manifest(m)

        # Only update if the new status is "more final" than the old
        if old_status in ("done", "done_qc_warn"):
            n_already_done += 1
            continue  # Already done, don't downgrade

        if new_status == "unknown":
            n_unknown += 1
            continue  # Can't classify

        if old_status != new_status:
            changes.append((gsm_id, old_status, new_status))
            status_updates[f"{old_status} -> {new_status}"] += 1

    elapsed = time.time() - t0
    print(f"  Matched: {n_matched:,} / {len(manifests):,} manifests found in catalog")
    print(f"  Already done: {n_already_done:,}")
    print(f"  Unclassifiable: {n_unknown:,}")
    print(f"  Status changes needed: {len(changes):,}")
    print(f"  Reconciled in {elapsed:.1f}s")
    print()

    if status_updates:
        print("Transition summary:")
        for transition, count in sorted(status_updates.items(), key=lambda x: -x[1])[:20]:
            print(f"  {transition}: {count:,}")
        print()

    # Apply changes
    if apply and changes:
        # Backup first
        if not BACKUP_PATH.exists() or force:
            print(f"Backing up catalog to {BACKUP_PATH}...")
            pc.to_parquet(BACKUP_PATH)

        # Apply status updates
        gsm_to_status = {gsm: new for gsm, _, new in changes}
        mask = pc["gsm_id"].isin(gsm_to_status)
        pc.loc[mask, "processing_status"] = pc.loc[mask, "gsm_id"].map(gsm_to_status)
        print(f"Applied {len(changes):,} status updates")

    # Compute per-phase eligibility columns
    print("\nComputing eligibility columns...")
    pc = compute_all_eligible(pc)
    print()

    if apply:
        print(f"Writing updated catalog to {CATALOG_PATH}...")
        pc.to_parquet(CATALOG_PATH)
        print("Done!")
    else:
        print("DRY RUN — no changes written. Use --apply to write.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write changes to catalog")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite backup even if it exists")
    args = parser.parse_args()
    reconcile(apply=args.apply, force=args.force)


if __name__ == "__main__":
    main()
