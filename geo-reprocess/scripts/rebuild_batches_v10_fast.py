#!/usr/bin/env python3
"""Fast v10 batch rebuild — confidence-filtered, no manifest scan.

Reads the cleaned catalog, filters out low-confidence and unknown protocols,
sorts by tier, and writes batch CSVs. Skips the slow manifest scan since the
pipeline already checks for existing manifests at runtime.

Output: batches_v10/ directory with fresh batch CSVs.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_BASE = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
CATALOG = Path("/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet")
OUTPUT_DIR = PIPELINE_BASE / "batches_v10"
BATCH_SIZE = 50

# Protocol priority tiers (lower = processed first)
TIER_MAP = {
    "10xv3": 1, "10xv2": 1, "10xv3_5prime": 1, "10xv4": 1, "10x_multiome": 1,
    "dropseq": 2, "seqwell": 2, "dnbelab": 2, "parse": 2,
    "bd_rhapsody": 2, "indrop": 2, "splitseq": 2, "surecell": 2, "ddseq": 2,
    "10x_suspect": 3, "citeseq": 3, "microwell": 3,
    "scirna": 4,
    "unknown_sc": 5, "snrna_unknown": 5, "unknown": 5,
    "10x_atac": 6, "atacseq": 6, "scatac": 6,
    "visium": 6, "10x_visium": 6, "slideseq": 6, "spatial": 6,
    "cite_seq_adt": 6,
}

# Low-confidence protocols to exclude from initial batches
LOW_CONF_PROTOCOLS = {"unknown", "unknown_sc", "10x_suspect", "snrna_unknown"}

# Plate-based protocols to exclude
PLATE_PROTOCOLS = {"smartseq2", "smartseq3", "smart-seq", "plate_based", "smart-seq2"}


def main():
    import pandas as pd

    logger.info(f"Loading catalog from {CATALOG}...")
    cat = pd.read_parquet(CATALOG)
    total = len(cat)
    logger.info(f"  Catalog: {total:,} rows")

    # Filter plate-based
    before = len(cat)
    cat = cat[~cat['protocol_inferred'].str.lower().isin(PLATE_PROTOCOLS)]
    n_plate = before - len(cat)
    logger.info(f"  Removed {n_plate:,} plate-based → {len(cat):,}")

    # Filter low-confidence protocols
    before = len(cat)
    cat = cat[~cat['protocol_inferred'].str.lower().isin(LOW_CONF_PROTOCOLS)]
    n_lowproto = before - len(cat)
    logger.info(f"  Removed {n_lowproto:,} low-confidence protocols → {len(cat):,}")

    # Filter low protocol_confidence
    before = len(cat)
    cat = cat[cat['protocol_confidence'].str.lower() != 'low']
    n_lowconf = before - len(cat)
    logger.info(f"  Removed {n_lowconf:,} protocol_confidence=low → {len(cat):,}")

    # Assign tiers
    cat['_tier'] = cat['protocol_inferred'].str.lower().map(TIER_MAP).fillna(6).astype(int)
    cat = cat.sort_values(['_tier', 'gse_id', 'gsm_id']).reset_index(drop=True)

    tier_counts = cat.groupby('_tier').size()
    logger.info("Tier distribution:")
    for t, n in tier_counts.items():
        logger.info(f"  T{t}: {n:,}")

    # Write batches
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing batch files
    for old in OUTPUT_DIR.glob("batch_*.csv"):
        old.unlink()

    batch_cols = ['gse_id', 'gsm_id', 'protocol_inferred', 'protocol_confidence',
                  'organism', 'species_ref_genome', 'species_annotation',
                  'srx_accession', 'srr_accessions',
                  'ena_fastq_r1', 'ena_fastq_r2', 'ncbi_sra_s3',
                  'library_layout', 'library_strategy', 'library_source',
                  'instrument_platform', 'instrument_model',
                  'n_gsm_in_series']
    batch_cols = [c for c in batch_cols if c in cat.columns]

    n_batches = 0
    for start in range(0, len(cat), BATCH_SIZE):
        batch = cat.iloc[start:start + BATCH_SIZE]
        batch_path = OUTPUT_DIR / f"batch_{n_batches:04d}.csv"
        batch[batch_cols].to_csv(batch_path, index=False)
        n_batches += 1

    logger.info(f"\nWrote {n_batches} batches ({len(cat):,} samples) to {OUTPUT_DIR}")

    print(f"\n=== BATCH REBUILD SUMMARY ===")
    print(f"  Catalog rows:             {total:,}")
    print(f"  Removed plate-based:      {n_plate:,}")
    print(f"  Removed low-conf protos:  {n_lowproto:,}")
    print(f"  Removed conf=low:         {n_lowconf:,}")
    print(f"  Samples in batches:       {len(cat):,}")
    print(f"  Batches written:          {n_batches}")
    for t, n in tier_counts.items():
        tier_names = {1: "10x", 2: "droplet", 3: "suspect", 4: "scirna", 5: "unknown", 6: "untiered"}
        print(f"  T{t} ({tier_names.get(t, '?')}): {n:,}")


if __name__ == "__main__":
    main()
