#!/usr/bin/env python3
"""Pre-screen unknown_sc and snRNA_unknown samples via FASTQ peek.

Runs the FASTQ peek heuristic (HTTP Range request, 16KB) on all samples
with protocol_inferred in {unknown_sc, snRNA_unknown, unknown} that have
ENA URLs. Classifies each as smartseq (both reads >50bp) or inconclusive.

For samples classified as smartseq:
  - Writes a skip manifest to quant/{gse}/{gsm}/sample_manifest.json
    so the main pipeline will skip them immediately.

Outputs:
  prescreener_results.csv — full results for every peeked sample
  prescreener_summary.json — counts by classification

This is designed to run as a SLURM job (1-2 CPUs, minimal memory).
Uses ThreadPoolExecutor for concurrent HTTP peek requests.
"""
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add workspace to path for scgeo imports
WORKSPACE = Path("/mnt/home/debruinz/Singlet-AI/geo-reprocess")
sys.path.insert(0, str(WORKSPACE))

from scgeo.pipeline.detect import peek_protocol, peek_fastq_read_length

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PIPELINE_BASE = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
V9_BATCH_DIR = PIPELINE_BASE / "batches_v9_autograb"
V6_BATCH_DIR = PIPELINE_BASE / "batches_v6_production"
QUANT_DIR = PIPELINE_BASE / "quant"
OUTPUT_DIR = PIPELINE_BASE / "prescreener"

PIPELINE_VERSION = "0.2.0"

# Protocols to pre-screen
TARGET_PROTOCOLS = {"unknown_sc", "snrna_unknown", "unknown"}

# Concurrency for HTTP peek requests
MAX_WORKERS = 16
PEEK_TIMEOUT = 15


def load_unknown_samples():
    """Load all unknown-protocol samples from v9 batches (or fall back to v6)."""
    batch_dir = V9_BATCH_DIR if V9_BATCH_DIR.exists() else V6_BATCH_DIR
    logger.info(f"Loading samples from {batch_dir}")

    samples = []
    seen = set()
    batch_files = sorted(batch_dir.glob("batch_*.csv"))
    batch_files = [f for f in batch_files if "_results" not in f.name]

    for bf in batch_files:
        try:
            with open(bf) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    gsm = row.get("gsm_id", "").strip()
                    protocol = row.get("protocol_inferred", "").strip().lower()
                    if protocol in TARGET_PROTOCOLS and gsm not in seen:
                        seen.add(gsm)
                        samples.append(row)
        except Exception:
            pass

    logger.info(f"Found {len(samples)} unknown-protocol samples")
    return samples


def already_screened(gse_id, gsm_id):
    """Check if sample already has a manifest (skip re-screening)."""
    manifest = QUANT_DIR / gse_id / gsm_id / "sample_manifest.json"
    return manifest.exists()


def peek_one_sample(row):
    """Run FASTQ peek on a single sample. Returns (row, classification, detail)."""
    gsm = row.get("gsm_id", "")
    gse = row.get("gse_id", "")
    r1_url = row.get("ena_fastq_r1", "").strip()
    r2_url = row.get("ena_fastq_r2", "").strip()

    if not r1_url:
        return row, "no_url", "no ENA R1 URL available"

    # Run the peek
    try:
        det = peek_protocol(r1_url, r2_url if r2_url else None, timeout=PEEK_TIMEOUT)
    except Exception as e:
        return row, "error", str(e)

    if det is not None and det.mode == "smartseq":
        return row, "smartseq", det.reason
    elif det is None:
        # Peek was inconclusive — get read lengths for logging
        r1_len = peek_fastq_read_length(r1_url, timeout=PEEK_TIMEOUT)
        r2_len = peek_fastq_read_length(r2_url, timeout=PEEK_TIMEOUT) if r2_url else None
        detail = f"inconclusive: R1={r1_len}bp"
        if r2_len is not None:
            detail += f", R2={r2_len}bp"
        return row, "inconclusive", detail
    else:
        return row, "inconclusive", f"peek returned mode={det.mode}"


def write_skip_manifest(gse_id, gsm_id, organism, reason):
    """Write a skip manifest so the main pipeline won't re-download this sample."""
    quant_dir = QUANT_DIR / gse_id / gsm_id
    quant_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = quant_dir / "sample_manifest.json"

    # Don't overwrite existing manifests
    if manifest_path.exists():
        return False

    manifest = {
        "gsm_id": gsm_id,
        "gse_id": gse_id,
        "organism": organism,
        "status": "skipped",
        "protocol_detected": "smartseq2",
        "mode": "smartseq",
        "error": f"Pre-screener FASTQ peek: {reason}",
        "pipeline_version": PIPELINE_VERSION,
        "prescreened": True,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    samples = load_unknown_samples()

    # Filter out already-screened
    to_screen = []
    already_done = 0
    for row in samples:
        gsm = row.get("gsm_id", "")
        gse = row.get("gse_id", "")
        if already_screened(gse, gsm):
            already_done += 1
        else:
            to_screen.append(row)

    logger.info(f"Already screened: {already_done}, remaining: {len(to_screen)}")

    if not to_screen:
        logger.info("Nothing to screen — all done!")
        return

    # ── Run FASTQ peeks concurrently ─────────────────────────────────────
    results = []
    n_smartseq = 0
    n_inconclusive = 0
    n_error = 0
    n_no_url = 0
    n_manifests_written = 0

    logger.info(f"Starting FASTQ peek with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(peek_one_sample, row): row for row in to_screen}

        for i, future in enumerate(as_completed(futures)):
            row, classification, detail = future.result()
            gsm = row.get("gsm_id", "")
            gse = row.get("gse_id", "")
            organism = row.get("organism", "")

            results.append({
                "gsm_id": gsm,
                "gse_id": gse,
                "organism": organism,
                "protocol_inferred": row.get("protocol_inferred", ""),
                "classification": classification,
                "detail": detail,
            })

            if classification == "smartseq":
                n_smartseq += 1
                if write_skip_manifest(gse, gsm, organism, detail):
                    n_manifests_written += 1
            elif classification == "inconclusive":
                n_inconclusive += 1
            elif classification == "error":
                n_error += 1
            elif classification == "no_url":
                n_no_url += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    f"  Progress: {i + 1}/{len(to_screen)} ({rate:.1f}/s) — "
                    f"smartseq={n_smartseq}, inconclusive={n_inconclusive}, "
                    f"error={n_error}, no_url={n_no_url}"
                )

    elapsed = time.time() - t0

    # ── Write results ────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "prescreener_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gsm_id", "gse_id", "organism",
                                                "protocol_inferred", "classification", "detail"])
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "total_screened": len(to_screen),
        "already_done": already_done,
        "smartseq": n_smartseq,
        "inconclusive": n_inconclusive,
        "error": n_error,
        "no_url": n_no_url,
        "manifests_written": n_manifests_written,
        "elapsed_seconds": round(elapsed, 1),
        "rate_per_second": round(len(to_screen) / elapsed, 2) if elapsed > 0 else 0,
    }
    summary_path = OUTPUT_DIR / "prescreener_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pre-screening complete in {elapsed:.0f}s")
    logger.info(f"  Total screened:     {len(to_screen)}")
    logger.info(f"  Smart-seq (skip):   {n_smartseq} ({n_manifests_written} manifests written)")
    logger.info(f"  Inconclusive:       {n_inconclusive} (will be processed by main pipeline)")
    logger.info(f"  Errors:             {n_error}")
    logger.info(f"  No URL:             {n_no_url}")
    logger.info(f"  Results: {results_path}")
    logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
