#!/usr/bin/env python3
"""Rebuild production batches for auto-grab processing.

Reads ALL v6 batch CSVs, filters out samples that are already processed
(have a sample_manifest.json with terminal status) or are known plate-based,
then re-sorts and re-batches the remaining samples by protocol confidence
so high-confidence droplet samples are processed first.

Output: batches_v9_autograb/ directory with new batch CSVs.

Protocol priority tiers (processed in this order):
  T1: 10xv3, 10xv2, 10xv3_5prime, 10xv4, 10x_multiome  (known 10x chemistry)
  T2: dropseq, seqwell, dnbelab, parse, bd_rhapsody, indrop, splitseq, surecell, ddseq  (known droplet)
  T3: 10x_suspect, citeseq  (likely droplet but needs detection)
  T4: scirna  (sci-RNA-seq3 — works but complex chemistry)
  T5: unknown_sc, snRNA_unknown, unknown  (needs FASTQ peek; most will be smartseq)

Samples in T5 without ENA URLs (can't be peeked) are placed last.
"""
import csv
import glob
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PIPELINE_BASE = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
V6_BATCH_DIR = PIPELINE_BASE / "batches_v6_production"
OUTPUT_DIR = PIPELINE_BASE / "batches_v9_autograb"
QUANT_DIR = PIPELINE_BASE / "quant"

BATCH_SIZE = 50

# ── Protocol priority tiers ─────────────────────────────────────────────
TIER_MAP = {
    # T1: Known 10x chemistry — highest confidence
    "10xv3": 1, "10xv2": 1, "10xv3_5prime": 1, "10xv4": 1, "10x_multiome": 1,
    # T2: Known droplet with custom geometry
    "dropseq": 2, "seqwell": 2, "dnbelab": 2, "parse": 2,
    "bd_rhapsody": 2, "indrop": 2, "splitseq": 2, "surecell": 2, "ddseq": 2,
    # T3: Likely droplet
    "10x_suspect": 3, "citeseq": 3,
    # T4: sci-RNA-seq3 (works but complex)
    "scirna": 4,
    # T5: Unknown — most expensive (need peek/detection)
    "unknown_sc": 5, "snrna_unknown": 5, "unknown": 5,
}

# Protocols to exclude entirely (plate-based — will never succeed in droplet pipeline)
PLATE_PROTOCOLS = {"smartseq2", "smartseq3", "plate_based", "smart-seq"}

# Terminal manifest statuses that mean "don't retry"
TERMINAL_STATUSES = {"success", "skipped"}


def load_all_v6_samples():
    """Load all samples from v6 batch CSVs."""
    samples = []
    batch_files = sorted(glob.glob(str(V6_BATCH_DIR / "batch_*.csv")))
    # Exclude results files
    batch_files = [f for f in batch_files if "_results" not in f]
    logger.info(f"Loading samples from {len(batch_files)} v6 batch files...")

    seen_gsm = set()
    for bf in batch_files:
        try:
            with open(bf) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    gsm = row.get("gsm_id", "").strip()
                    if gsm and gsm not in seen_gsm:
                        seen_gsm.add(gsm)
                        samples.append(row)
        except Exception as e:
            logger.warning(f"  Skipping {bf}: {e}")

    logger.info(f"Loaded {len(samples)} unique samples")
    return samples


def scan_terminal_manifests():
    """Build a set of (gse_id, gsm_id) pairs that have terminal manifests.

    Scans the quant directory once instead of stat-ing each file individually.
    Returns a set of gsm_ids that should be skipped.
    """
    logger.info("Scanning for existing manifests (this may take a minute on NFS)...")
    terminal_gsms = set()
    n_scanned = 0
    n_failed_retryable = 0

    if not QUANT_DIR.exists():
        logger.info("  No quant directory found — no manifests to check")
        return terminal_gsms

    # Walk quant/<gse>/<gsm>/sample_manifest.json
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
                status = m.get("status", "")
                qc_status = m.get("qc_status", "")
                if status in TERMINAL_STATUSES:
                    terminal_gsms.add(gsm_dir.name)
                elif qc_status in ("qc_pass", "qc_warn", "qc_fail"):
                    terminal_gsms.add(gsm_dir.name)
                else:
                    # "failed" → retryable
                    n_failed_retryable += 1
            except Exception:
                pass

    logger.info(f"  Scanned {n_scanned} manifests: {len(terminal_gsms)} terminal, {n_failed_retryable} retryable (failed)")
    return terminal_gsms


def main():
    samples = load_all_v6_samples()

    # ── Build terminal manifest index ────────────────────────────────────
    terminal_gsms = scan_terminal_manifests()

    # ── Filter ───────────────────────────────────────────────────────────
    logger.info("Filtering samples...")
    filtered = []
    skip_reasons = Counter()

    for row in samples:
        gsm = row.get("gsm_id", "").strip()
        gse = row.get("gse_id", "").strip()
        protocol = row.get("protocol_inferred", "").strip().lower()

        # Skip plate-based protocols
        if protocol in PLATE_PROTOCOLS:
            skip_reasons["plate_based"] += 1
            continue

        # Skip already-processed samples (terminal manifest on disk)
        if gsm in terminal_gsms:
            skip_reasons["already_processed"] += 1
            continue

        filtered.append(row)

    logger.info(f"After filtering: {len(filtered)} samples (removed {len(samples) - len(filtered)})")
    for reason, count in skip_reasons.most_common():
        logger.info(f"  {reason}: {count}")

    # ── Sort by protocol tier, then by GSE for locality ──────────────────
    def sort_key(row):
        protocol = row.get("protocol_inferred", "unknown").strip().lower()
        tier = TIER_MAP.get(protocol, 6)  # Unknown protocols → T6
        gse = row.get("gse_id", "")
        gsm = row.get("gsm_id", "")
        # Within T5 (unknown), samples with ENA URLs go before those without
        has_ena = bool(row.get("ena_fastq_r1", "").strip())
        ena_penalty = 0 if has_ena else 1
        return (tier, ena_penalty, gse, gsm)

    filtered.sort(key=sort_key)

    # ── Protocol distribution after filtering ────────────────────────────
    protocol_counts = Counter()
    tier_counts = Counter()
    for row in filtered:
        proto = row.get("protocol_inferred", "unknown").strip().lower()
        protocol_counts[proto] += 1
        tier_counts[TIER_MAP.get(proto, 6)] += 1

    logger.info("\nProtocol distribution (remaining):")
    for proto, n in protocol_counts.most_common():
        tier = TIER_MAP.get(proto, 6)
        logger.info(f"  T{tier} {proto}: {n}")

    logger.info(f"\nTier summary:")
    for tier in sorted(tier_counts):
        logger.info(f"  Tier {tier}: {tier_counts[tier]} samples")

    # ── Write new batches ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = ["gsm_id", "gse_id", "organism", "protocol_inferred",
                  "ena_fastq_r1", "ena_fastq_r2", "srr_accessions", "read_count"]

    n_batches = (len(filtered) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"\nWriting {n_batches} batches of {BATCH_SIZE} to {OUTPUT_DIR}")

    for i in range(n_batches):
        batch = filtered[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_path = OUTPUT_DIR / f"batch_{i:05d}.csv"
        with open(batch_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(batch)

    # ── Write manifest ───────────────────────────────────────────────────
    manifest = {
        "total_samples": len(filtered),
        "n_batches": n_batches,
        "batch_size": BATCH_SIZE,
        "original_samples": len(samples),
        "filtered_out": len(samples) - len(filtered),
        "skip_reasons": dict(skip_reasons),
        "protocol_counts": dict(protocol_counts),
        "tier_counts": {f"T{k}": v for k, v in sorted(tier_counts.items())},
    }
    with open(OUTPUT_DIR / "batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nDone! {len(filtered)} samples in {n_batches} batches")
    logger.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
