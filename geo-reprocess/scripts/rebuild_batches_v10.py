#!/usr/bin/env python3
"""Rebuild v10 batches from the cleaned catalog.

Reads the cleaned catalog (post-cleanup), filters out samples that already have
terminal manifests on disk, then sorts by protocol tier and writes batch CSVs.

Output: batches_v10/ directory with fresh batch CSVs.
"""
import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_BASE = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
CATALOG = Path("/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet")
OUTPUT_DIR = PIPELINE_BASE / "batches_v10"
QUANT_DIR = PIPELINE_BASE / "quant"
BATCH_SIZE = 50

# Protocol priority tiers (lower = processed first)
TIER_MAP = {
    # T1: Known 10x chemistry — highest confidence
    "10xv3": 1, "10xv2": 1, "10xv3_5prime": 1, "10xv4": 1, "10x_multiome": 1,
    # T2: Known droplet with custom geometry
    "dropseq": 2, "seqwell": 2, "dnbelab": 2, "parse": 2,
    "bd_rhapsody": 2, "indrop": 2, "splitseq": 2, "surecell": 2, "ddseq": 2,
    # T3: Likely droplet (needs detection)
    "10x_suspect": 3, "citeseq": 3, "microwell": 3,
    # T4: sci-RNA-seq3
    "scirna": 4,
    # T5: Unknown (need FASTQ peek/detection)
    "unknown_sc": 5, "snrna_unknown": 5, "unknown": 5,
    # T6: Multimodal/spatial (process after scRNA-seq completes)
    "10x_atac": 6, "atacseq": 6, "scatac": 6,
    "visium": 6, "10x_visium": 6, "slideseq": 6, "spatial": 6,
    "cite_seq_adt": 6,
}

TERMINAL_STATUSES = {"success", "skipped"}


def scan_terminal_manifests():
    """Build set of gsm_ids that have terminal manifests on disk."""
    logger.info("Scanning for existing manifests...")
    terminal_gsms = set()
    n_scanned = 0
    n_retryable = 0

    if not QUANT_DIR.exists():
        return terminal_gsms

    for gse_dir in QUANT_DIR.iterdir():
        if not gse_dir.is_dir():
            continue
        try:
            for gsm_dir in gse_dir.iterdir():
                if not gsm_dir.is_dir():
                    continue
                mp = gsm_dir / "sample_manifest.json"
                if not mp.exists():
                    continue
                n_scanned += 1
                try:
                    with open(mp) as f:
                        m = json.load(f)
                    status = m.get("status", "")
                    qc = m.get("qc_status", "")
                    if status in TERMINAL_STATUSES or qc in ("qc_pass", "qc_warn", "qc_fail"):
                        terminal_gsms.add(gsm_dir.name)
                    else:
                        n_retryable += 1
                except Exception:
                    pass
        except PermissionError:
            continue

    logger.info(f"  Scanned {n_scanned}: {len(terminal_gsms)} terminal, {n_retryable} retryable")
    return terminal_gsms


def main():
    import pandas as pd

    logger.info(f"Loading catalog from {CATALOG}...")
    cat = pd.read_parquet(CATALOG)
    logger.info(f"  Catalog: {len(cat):,} rows")

    terminal_gsms = scan_terminal_manifests()

    # Filter out already-processed
    before = len(cat)
    cat = cat[~cat['gsm_id'].isin(terminal_gsms)]
    logger.info(f"  After removing {before - len(cat):,} already-processed: {len(cat):,} remaining")

    # Filter out low-confidence and unknown/suspect protocols
    LOW_CONF_PROTOCOLS = {"unknown", "unknown_sc", "10x_suspect", "snrna_unknown"}
    before_conf = len(cat)
    mask_conf = cat['protocol_confidence'].str.lower() != 'low'
    mask_proto = ~cat['protocol_inferred'].str.lower().isin(LOW_CONF_PROTOCOLS)
    cat = cat[mask_conf & mask_proto]
    logger.info(f"  After removing {before_conf - len(cat):,} low-confidence/unknown: {len(cat):,} remaining")

    # Assign tiers
    cat['_tier'] = cat['protocol_inferred'].str.lower().map(TIER_MAP).fillna(6).astype(int)
    
    # Within each tier, sort by gse_id to keep series together (helps with GSE-level caching)
    cat = cat.sort_values(['_tier', 'gse_id', 'gsm_id']).reset_index(drop=True)

    tier_counts = cat.groupby('_tier').size()
    logger.info("Tier distribution:")
    for t, n in tier_counts.items():
        logger.info(f"  T{t}: {n:,}")

    # Write batches
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Column mapping: catalog columns → batch CSV columns
    batch_cols = ['gse_id', 'gsm_id', 'protocol_inferred', 'protocol_confidence',
                  'organism', 'species_ref_genome', 'species_annotation',
                  'srx_accession', 'srr_accessions',
                  'ena_fastq_r1', 'ena_fastq_r2', 'ncbi_sra_s3',
                  'library_layout', 'library_strategy', 'library_source',
                  'instrument_platform', 'instrument_model',
                  'n_gsm_in_series']
    # Only include columns that exist
    batch_cols = [c for c in batch_cols if c in cat.columns]

    n_batches = 0
    for start in range(0, len(cat), BATCH_SIZE):
        batch = cat.iloc[start:start + BATCH_SIZE]
        batch_path = OUTPUT_DIR / f"batch_{n_batches:04d}.csv"
        batch[batch_cols].to_csv(batch_path, index=False)
        n_batches += 1

    logger.info(f"\nWrote {n_batches} batches ({len(cat):,} samples) to {OUTPUT_DIR}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    # Summary
    print(f"\n=== BATCH REBUILD SUMMARY ===")
    print(f"  Catalog rows: {before:,} (after cleanup)")
    print(f"  Already processed: {before - len(cat):,}")
    print(f"  To process: {len(cat):,}")
    print(f"  Batches: {n_batches}")
    for t, n in tier_counts.items():
        tier_names = {1: "10x", 2: "droplet", 3: "suspect", 4: "scirna", 5: "unknown", 6: "untiered"}
        print(f"  T{t} ({tier_names.get(t, '?')}): {n:,}")


if __name__ == "__main__":
    main()
