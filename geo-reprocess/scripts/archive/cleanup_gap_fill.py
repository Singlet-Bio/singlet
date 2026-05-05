#!/usr/bin/env python3
"""Clean up gap-fill catalog entries: remove plate-based, classify unknowns.

Removes entries that are clearly NOT droplet-based scRNA-seq:
  1. All SINGLE-layout entries (plate-based SmartSeq)
  2. PAIRED entries already classified as smartseq2
  3. No-layout entries from known non-RNA modalities (ATAC, microarray, spatial)
  4. No-layout entries that are bulk/microarray (no SRX, no SC evidence)
  5. No-layout entries from known SmartSeq GSEs (e.g. GSE75330, GSE76381)

For remaining unknowns, re-checks GEO SOFT text to classify protocol.
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG_PATH = Path("/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet")
OUTPUT_PATH = CATALOG_PATH  # Overwrite in place
BATCH_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline/batches_gap_fill")

# ── Known classifications for no-layout GSEs ──
# Based on manual inspection of series titles and metadata
REMOVE_NO_LAYOUT_GSES = {
    "GSE130773": "microarray (not sequencing)",
    "GSE161381": "ATAC-seq (not RNA)",
    "GSE234713": "NanoString CosMx spatial (not standard scRNA-seq)",
    "GSE75330":  "known SmartSeq plate-based (oligodendrocyte lineage)",
    "GSE76381":  "known SmartSeq/STRT plate-based (La Manno midbrain)",
}

# GSEs with no SRX accessions and unknown protocol — can't process, unclear modality
# These need GEO SOFT re-check before deciding
INVESTIGATE_NO_LAYOUT_GSES = [
    "GSE152197",  # 13 no-layout + 2918 PAIRED — check SOFT
    "GSE156702",  # 3 no-layout + 18 SINGLE — mixed series
    "GSE161382",  # 9 no-layout — single nucleus RNA-seq multiomic (no SRX)
    "GSE165837",  # 14 no-layout — cardiac cell type-specific (no SRX)
    "GSE165838",  # 7 no-layout — cardiac cell type-specific (no SRX)
    "GSE172316",  # 6 no-layout — already classified 10xv2, but no SRX
    "GSE196829",  # 1104 no-layout — eQTL study (no SRX)
    "GSE225671",  # 8 no-layout — stromal/immune (8 SRXs)
    "GSE234933",  # 52 no-layout — head-neck squamous (no SRX)
]


def fetch_gse_soft(gse_id: str) -> str:
    """Fetch series-level SOFT text for a GSE."""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}&targ=self&form=text&view=brief"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.ok:
                return resp.text
        except Exception:
            time.sleep(2 ** attempt)
    return ""


def classify_from_soft(soft_text: str) -> str:
    """Classify a GSE from its SOFT text. Returns protocol or 'unknown'."""
    text = soft_text.lower()

    # Check for 10x/droplet
    if any(kw in text for kw in ["10x genomics", "10x chromium", "chromium single cell",
                                   "chromium controller", "chromium next gem"]):
        return "10x_suspect"
    if any(kw in text for kw in ["drop-seq", "dropseq", "drop seq"]):
        return "dropseq"
    if "indrop" in text:
        return "indrop"
    if any(kw in text for kw in ["bd rhapsody", "rhapsody"]):
        return "bd_rhapsody"
    if any(kw in text for kw in ["parse biosciences", "evercode"]):
        return "parse"
    if any(kw in text for kw in ["seq-well", "seqwell"]):
        return "seqwell"
    if any(kw in text for kw in ["sci-rna", "scirna"]):
        return "scirna"
    if "split-seq" in text or "splitseq" in text:
        return "splitseq"
    if "microwell" in text:
        return "microwell"
    if "dnbelab" in text:
        return "dnbelab"

    # Check for plate-based
    if any(kw in text for kw in ["smart-seq", "smartseq", "smart seq"]):
        return "smartseq2"
    if any(kw in text for kw in ["cel-seq", "celseq"]):
        return "celseq"
    if any(kw in text for kw in ["mars-seq", "marsseq"]):
        return "marsseq"
    if "strt-seq" in text or "strtseq" in text:
        return "strtseq"
    if "plate" in text and "single cell" in text:
        return "plate_based"

    # Check for non-RNA
    if any(kw in text for kw in ["atac-seq", "atacseq", "atac seq"]):
        return "scATAC"
    if "microarray" in text:
        return "microarray"
    if any(kw in text for kw in ["nanostring", "cosmx", "visium", "merfish", "spatial"]):
        return "spatial"

    return "unknown"


# Plate-based protocols
PLATE_PROTOCOLS = {"smartseq2", "smartseq3", "plate_based", "smart-seq", "celseq",
                   "celseq2", "marsseq", "strtseq", "quartzseq", "icell8"}
NON_RNA_PROTOCOLS = {"scATAC", "microarray", "spatial", "chipseq", "methylation", "hi_c"}


def main():
    cat = pd.read_parquet(CATALOG_PATH)
    gap = cat[cat["notes"] == "gap_fill_from_repo_crossref"]
    non_gap = cat[cat["notes"] != "gap_fill_from_repo_crossref"]

    logger.info(f"Catalog: {len(cat):,} rows, {cat['gse_id'].nunique():,} GSEs")
    logger.info(f"Gap-fill entries: {len(gap):,} rows, {gap['gse_id'].nunique():,} GSEs")

    removal_reasons = {}

    # ── Step 1: Remove ALL SINGLE-layout gap entries ──
    single_mask = gap["library_layout"] == "SINGLE"
    for _, row in gap[single_mask].iterrows():
        removal_reasons[row["gsm_id"]] = f"SINGLE layout (plate-based)"
    logger.info(f"Step 1: {single_mask.sum():,} SINGLE-layout entries flagged for removal")

    # ── Step 2: Remove PAIRED entries already classified as plate-based ──
    paired_plate_mask = (gap["library_layout"] == "PAIRED") & \
                        gap["protocol_inferred"].str.lower().isin(PLATE_PROTOCOLS)
    for _, row in gap[paired_plate_mask].iterrows():
        removal_reasons[row["gsm_id"]] = f"PAIRED but protocol={row['protocol_inferred']}"
    logger.info(f"Step 2: {paired_plate_mask.sum():,} PAIRED plate-based entries flagged")

    # ── Step 3: Remove known non-RNA no-layout GSEs ──
    for gse, reason in REMOVE_NO_LAYOUT_GSES.items():
        mask = (gap["gse_id"] == gse) & (gap["library_layout"] == "")
        for _, row in gap[mask].iterrows():
            removal_reasons[row["gsm_id"]] = f"No-layout, {reason}"
        if mask.sum() > 0:
            logger.info(f"Step 3: {gse} — {mask.sum()} entries flagged ({reason})")

    # ── Step 4: Re-classify remaining no-layout GSEs via GEO SOFT ──
    remaining_no_layout = gap[(gap["library_layout"] == "") &
                              ~gap["gsm_id"].isin(removal_reasons)]
    remaining_gses = remaining_no_layout["gse_id"].unique()
    logger.info(f"\nStep 4: Re-classifying {len(remaining_gses)} remaining no-layout GSEs via SOFT...")

    soft_classifications = {}
    for gse in remaining_gses:
        soft = fetch_gse_soft(gse)
        proto = classify_from_soft(soft)
        soft_classifications[gse] = proto
        logger.info(f"  {gse}: SOFT → {proto}")
        time.sleep(0.5)

    # Apply classifications
    for gse, proto in soft_classifications.items():
        mask = (gap["gse_id"] == gse) & (gap["library_layout"] == "") & \
               ~gap["gsm_id"].isin(removal_reasons)
        n = mask.sum()
        if proto in PLATE_PROTOCOLS:
            for _, row in gap[mask].iterrows():
                removal_reasons[row["gsm_id"]] = f"No-layout, SOFT classified as {proto}"
            logger.info(f"  {gse}: removing {n} — plate-based ({proto})")
        elif proto in NON_RNA_PROTOCOLS:
            for _, row in gap[mask].iterrows():
                removal_reasons[row["gsm_id"]] = f"No-layout, SOFT classified as {proto}"
            logger.info(f"  {gse}: removing {n} — non-RNA ({proto})")
        elif proto == "unknown":
            # No SRX means can't process → remove
            sub = gap[mask]
            has_srx = (sub["srx_accession"] != "").sum()
            if has_srx == 0:
                for _, row in sub.iterrows():
                    removal_reasons[row["gsm_id"]] = f"No-layout, unknown protocol, no SRX (unprocessable)"
                logger.info(f"  {gse}: removing {n} — unknown protocol, no SRX")
            else:
                logger.info(f"  {gse}: keeping {n} entries ({has_srx} have SRX) — needs FASTQ peek")
        else:
            # Droplet protocol from SOFT → keep and update protocol
            logger.info(f"  {gse}: keeping {n} — classified as {proto}")

    # ── Step 5: Check remaining PAIRED unknowns via SOFT ──
    remaining_paired = gap[(gap["library_layout"] == "PAIRED") &
                           ~gap["gsm_id"].isin(removal_reasons) &
                           gap["protocol_inferred"].isin(["unknown", "unknown_sc"])]
    remaining_paired_gses = remaining_paired["gse_id"].unique()
    logger.info(f"\nStep 5: Re-classifying {len(remaining_paired_gses)} PAIRED unknown GSEs via SOFT...")

    for gse in remaining_paired_gses:
        if gse not in soft_classifications:
            soft = fetch_gse_soft(gse)
            proto = classify_from_soft(soft)
            soft_classifications[gse] = proto
            logger.info(f"  {gse}: SOFT → {proto}")
            time.sleep(0.5)
        else:
            proto = soft_classifications[gse]
            logger.info(f"  {gse}: (cached) → {proto}")

        mask = (gap["gse_id"] == gse) & (gap["library_layout"] == "PAIRED") & \
               ~gap["gsm_id"].isin(removal_reasons) & \
               gap["protocol_inferred"].isin(["unknown", "unknown_sc"])
        n = mask.sum()
        if proto in PLATE_PROTOCOLS:
            for _, row in gap[mask].iterrows():
                removal_reasons[row["gsm_id"]] = f"PAIRED, SOFT classified as {proto}"
            logger.info(f"  {gse}: removing {n} — plate-based ({proto})")
        elif proto in NON_RNA_PROTOCOLS:
            for _, row in gap[mask].iterrows():
                removal_reasons[row["gsm_id"]] = f"PAIRED, SOFT classified as {proto}"
            logger.info(f"  {gse}: removing {n} — non-RNA ({proto})")
        else:
            logger.info(f"  {gse}: keeping {n} — classified as {proto}")

    # ── Apply removals ──
    to_remove = set(removal_reasons.keys())
    gap_keep = gap[~gap["gsm_id"].isin(to_remove)]
    gap_removed = gap[gap["gsm_id"].isin(to_remove)]

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Gap-fill entries before: {len(gap):,}")
    logger.info(f"  Removed:  {len(gap_removed):,}")
    logger.info(f"  Keeping:  {len(gap_keep):,}")
    logger.info(f"  GSEs removed entirely: checking...")

    # Which GSEs are fully removed vs partially kept?
    removed_gses = set(gap_removed["gse_id"].unique())
    kept_gses = set(gap_keep["gse_id"].unique())
    fully_removed = removed_gses - kept_gses
    partially_kept = removed_gses & kept_gses
    logger.info(f"  GSEs fully removed: {len(fully_removed)} — {sorted(fully_removed)}")
    logger.info(f"  GSEs partially kept: {len(partially_kept)} — {sorted(partially_kept)}")
    logger.info(f"  GSEs fully kept: {len(kept_gses - partially_kept)}")

    # Show what's being kept
    logger.info(f"\nKept entries by GSE:")
    for gse, sub in gap_keep.groupby("gse_id"):
        layouts = sub["library_layout"].value_counts().to_dict()
        protocols = sub["protocol_inferred"].value_counts().to_dict()
        logger.info(f"  {gse}: {len(sub)} GSMs, layouts={layouts}, protocols={protocols}")

    # ── Write cleaned catalog ──
    new_cat = pd.concat([non_gap, gap_keep], ignore_index=True)
    new_cat.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nCatalog written: {len(new_cat):,} rows, {new_cat['gse_id'].nunique():,} GSEs")

    # ── Remove batch files for fully removed GSEs ──
    if BATCH_DIR.exists():
        removed_batches = 0
        for batch_file in BATCH_DIR.glob("gap_batch_*.csv"):
            # Check if this batch only has removed GSMs
            try:
                import csv
                with open(batch_file) as f:
                    reader = csv.DictReader(f)
                    gsms_in_batch = [r.get("gsm_id", "") for r in reader]
                all_removed = all(g in to_remove for g in gsms_in_batch if g)
                if all_removed and gsms_in_batch:
                    batch_file.unlink()
                    removed_batches += 1
            except Exception:
                pass
        logger.info(f"Removed {removed_batches} batch files for fully-removed entries")

    # ── Save removal log ──
    log_path = CATALOG_PATH.parent / "gap_fill_cleanup_log.json"
    cleanup_log = {
        "total_removed": len(to_remove),
        "total_kept": len(gap_keep),
        "reasons_summary": {},
        "gses_fully_removed": sorted(fully_removed),
        "gses_partially_kept": sorted(partially_kept),
        "soft_classifications": soft_classifications,
    }
    # Summarize reasons
    from collections import Counter
    reason_counts = Counter(removal_reasons.values())
    cleanup_log["reasons_summary"] = dict(reason_counts.most_common())

    with open(log_path, "w") as f:
        json.dump(cleanup_log, f, indent=2)
    logger.info(f"Cleanup log: {log_path}")


if __name__ == "__main__":
    main()
