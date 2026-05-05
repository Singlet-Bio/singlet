#!/usr/bin/env python3
"""Build v10 filtered batches — excludes completed, QC'd, and undownloadable samples.

Reads the full catalog of batch CSVs, scans manifests for already-processed
samples, and filters out:
  1. Samples with terminal manifest status (qc_pass, qc_warn, qc_fail, skipped, success)
  2. Samples with no download method (no ENA URLs AND no SRR accession)
  3. Plate-based protocols (smartseq2, smart-seq, plate_based)

Writes new batches to batches_v10_filtered/ with only actionable samples.
"""
import argparse
import csv
import glob
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"

# Terminal statuses — these samples are done, don't reprocess
TERMINAL_QC = {"qc_pass", "qc_warn", "qc_fail"}
TERMINAL_STATUS = {"success", "skipped"}

# Plate-based protocols — skip (we only do droplet)
PLATE_PROTOCOLS = {"smartseq2", "smartseq3", "smart-seq", "plate_based", "smart-seq2"}


def load_terminal_gsms(quant_dir: str) -> set:
    """Scan manifests and return set of GSM IDs that are terminally processed."""
    terminal = set()
    n_scanned = 0
    for gse in os.listdir(quant_dir):
        gse_path = os.path.join(quant_dir, gse)
        if not os.path.isdir(gse_path):
            continue
        for gsm in os.listdir(gse_path):
            mf = os.path.join(gse_path, gsm, "sample_manifest.json")
            if not os.path.isfile(mf):
                continue
            n_scanned += 1
            try:
                with open(mf) as f:
                    d = json.load(f)
                qc = d.get("qc_status", "")
                status = d.get("status", "")
                if qc in TERMINAL_QC or status in TERMINAL_STATUS:
                    terminal.add(gsm)
            except Exception:
                pass
    logger.info(f"Scanned {n_scanned:,} manifests, {len(terminal):,} terminal")
    return terminal


def has_download_method(row: dict) -> bool:
    """Check if a sample has at least one download method."""
    ena_r1 = (row.get("ena_fastq_r1") or "").strip()
    srr = (row.get("srr_accessions") or "").strip()
    return bool(ena_r1) or bool(srr)


def is_plate_based(row: dict) -> bool:
    """Check if protocol is plate-based."""
    proto = (row.get("protocol_inferred") or row.get("protocol") or "").strip().lower()
    return proto in PLATE_PROTOCOLS


def main():
    parser = argparse.ArgumentParser(description="Build v10 filtered batches")
    parser.add_argument(
        "--source-dir",
        default="/mnt/projects/debruinz_project/cellarium/pipeline/batches_v6_production",
        help="Source batch directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/projects/debruinz_project/cellarium/pipeline/batches_v10_filtered",
        help="Output batch directory",
    )
    parser.add_argument(
        "--quant-dir",
        default=QUANT_DIR,
        help="Quant directory to scan for manifests",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Samples per batch (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing batches",
    )
    args = parser.parse_args()

    # Phase 1: Load terminal GSMs
    logger.info("Phase 1: Scanning manifests for completed samples...")
    terminal_gsms = load_terminal_gsms(args.quant_dir)

    # Phase 2: Read all source batches and filter
    logger.info("Phase 2: Reading source batches...")
    source_files = sorted(glob.glob(os.path.join(args.source_dir, "batch_*.csv")))
    logger.info(f"Found {len(source_files)} source batch files")

    filter_stats = Counter()
    filtered_rows = []
    fieldnames = None

    for sf in source_files:
        try:
            with open(sf) as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    gsm = row.get("gsm_id", "")
                    filter_stats["total"] += 1

                    if gsm in terminal_gsms:
                        filter_stats["already_processed"] += 1
                        continue

                    if not has_download_method(row):
                        filter_stats["no_download_method"] += 1
                        continue

                    if is_plate_based(row):
                        filter_stats["plate_based"] += 1
                        continue

                    filtered_rows.append(row)
                    filter_stats["actionable"] += 1
        except Exception as e:
            logger.warning(f"Error reading {sf}: {e}")

    logger.info(f"\nFilter results:")
    for k, v in filter_stats.most_common():
        pct = 100 * v / max(filter_stats["total"], 1)
        logger.info(f"  {k:>25s}: {v:>8,} ({pct:.1f}%)")

    if args.dry_run:
        logger.info(f"\nDry run: would write {len(filtered_rows):,} samples "
                     f"in {(len(filtered_rows) + args.batch_size - 1) // args.batch_size} batches")
        return

    # Phase 3: Write filtered batches
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_batches = 0
    for i in range(0, len(filtered_rows), args.batch_size):
        batch = filtered_rows[i:i + args.batch_size]
        batch_file = output_dir / f"batch_{n_batches:05d}.csv"
        with open(batch_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch)
        n_batches += 1

    logger.info(f"\nWrote {n_batches} batches ({len(filtered_rows):,} samples) to {output_dir}")

    # Write manifest
    manifest = {
        "source_dir": args.source_dir,
        "n_batches": n_batches,
        "n_samples": len(filtered_rows),
        "batch_size": args.batch_size,
        "filter_stats": dict(filter_stats),
    }
    with open(output_dir / "batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
