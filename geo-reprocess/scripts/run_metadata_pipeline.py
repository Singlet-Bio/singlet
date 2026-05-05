#!/usr/bin/env python3
"""Staged metadata extraction pipeline.

Processes author metadata for all GSMs in the cellarium dataset through
four stages, tracking progress in a metadata_catalog.parquet:

    Stage 0  — Classify supplementary files for every GSE (local, no network)
    Stage 1  — Tier 1: SOFT characteristics → author_metadata.parquet per GSM
    Stage 2a — Tier 2: Cell-level from tabular/h5ad/loom supplementary files
    Stage 2b — Tier 2: Cell-level from RDS/Seurat supplementary files (needs rpy2)
    Stage 3  — Tier 3: NCBI descriptions → update uns in author_metadata.parquet

Usage:
    python run_metadata_pipeline.py --stage 0
    python run_metadata_pipeline.py --stage 1 --batch 0 --total-batches 4
    python run_metadata_pipeline.py --stage 2a --batch 0 --total-batches 4
    python run_metadata_pipeline.py --stage 2b --batch 0 --total-batches 4
    python run_metadata_pipeline.py --stage 3 --batch 0 --total-batches 4

Each run skips GSMs already completed for that stage.
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Add scgeo to path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scgeo.metadata.soft import load_soft_metadata, parse_characteristics, build_gsm_level_obs
from scgeo.metadata.description import fetch_geo_description
from scgeo.metadata.barcodes import align_author_metadata
from scgeo.metadata.download import download_supplementary_file
from scgeo.metadata.extract import (
    extract_metadata_from_h5ad,
    extract_metadata_from_tabular,
    extract_metadata_from_loom,
    extract_metadata_from_tar,
    extract_metadata_from_expression,
    peek_tabular_content,
)
from scgeo.metadata.extract_2c import (
    extract_metadata_from_xlsx,
    extract_metadata_from_h5seurat,
    extract_metadata_from_hdf5,
    extract_metadata_from_h5mu,
)
from scgeo.metadata.api import classify_supplementary_files, _classify_filename, _META_PATTERNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("metadata_pipeline")

# ── Paths ──
PROJECT = Path("/mnt/projects/debruinz_project/cellarium")
DATASET_DIR = PROJECT / "dataset"
CATALOG_DIR = PROJECT / "catalog"
METADATA_CATALOG = CATALOG_DIR / "metadata_catalog.parquet"
WORK_DIR = PROJECT / "pipeline" / "metadata_work"
STAGE3_CATALOG = CATALOG_DIR / "stage3_soft_parsed.parquet"

# Set by main() so checkpoint saves go to the right shard
_CURRENT_BATCH_ID = ""


# ════════════════════════════════════════════════════════════════════════
#  Metadata Catalog Management
# ════════════════════════════════════════════════════════════════════════

def init_metadata_catalog() -> pd.DataFrame:
    """Initialise or load the metadata tracking catalog.

    Creates metadata_catalog.parquet if it doesn't exist, seeded from
    the dataset directory (all GSE/GSM pairs with cells.parquet).

    Tracking columns:
        stage0_status     : "pending" | "done" | "failed"
        stage0_formats    : JSON dict of classified supp file counts
        stage1_status     : "pending" | "done" | "failed"
        stage1_n_cols     : number of Tier 1 columns
        stage2a_status    : "pending" | "skipped" | "done" | "failed"
        stage2a_format    : format used (h5ad/tabular/loom) or None
        stage2a_source    : filename of supp file or None
        stage2a_n_cols    : number of Tier 2 columns extracted
        stage2a_match_rate: barcode match rate (0-1)
        stage2b_status    : "pending" | "skipped" | "done" | "failed"
        stage2b_source    : RDS filename or None
        stage2b_n_cols    : number of Tier 2 columns
        stage2b_match_rate: barcode match rate
        stage3_status     : "pending" | "done" | "failed"
        stage3_has_summary: bool
        error             : last error message
        updated_at        : ISO timestamp of last update
    """
    if METADATA_CATALOG.exists():
        logger.info("Loading existing metadata catalog")
        cat = pd.read_parquet(str(METADATA_CATALOG))
        # Add stage2c columns if missing (catalog schema migration)
        if "stage2c_status" not in cat.columns:
            cat["stage2c_status"] = "pending"
            cat["stage2c_format"] = ""
            cat["stage2c_source"] = ""
            cat["stage2c_n_cols"] = 0
            cat["stage2c_match_rate"] = 0.0
            logger.info("Added stage2c columns to existing catalog")
        return cat

    logger.info("Initialising metadata catalog from dataset directory...")
    rows = []
    for gse_dir in sorted(DATASET_DIR.iterdir()):
        if not gse_dir.name.startswith("GSE"):
            continue
        gse_id = gse_dir.name
        for gsm_dir in sorted(gse_dir.iterdir()):
            if not gsm_dir.name.startswith("GSM"):
                continue
            if not (gsm_dir / "cells.parquet").exists():
                continue
            rows.append({"gse_id": gse_id, "gsm_id": gsm_dir.name})

    df = pd.DataFrame(rows)
    logger.info("Found %d GSMs across %d GSEs", len(df), df["gse_id"].nunique())

    # Add tracking columns
    df["stage0_status"] = "pending"
    df["stage0_formats"] = ""
    df["stage1_status"] = "pending"
    df["stage1_n_cols"] = 0
    df["stage2a_status"] = "pending"
    df["stage2a_format"] = ""
    df["stage2a_source"] = ""
    df["stage2a_n_cols"] = 0
    df["stage2a_match_rate"] = 0.0
    df["stage2b_status"] = "pending"
    df["stage2b_source"] = ""
    df["stage2b_n_cols"] = 0
    df["stage2b_match_rate"] = 0.0
    df["stage2c_status"] = "pending"
    df["stage2c_format"] = ""
    df["stage2c_source"] = ""
    df["stage2c_n_cols"] = 0
    df["stage2c_match_rate"] = 0.0
    df["stage3_status"] = "pending"
    df["stage3_has_summary"] = False
    df["error"] = ""
    df["updated_at"] = ""

    save_catalog(df)
    return df


def save_catalog(df: pd.DataFrame, batch_id: str = ""):
    """Save metadata catalog to parquet (atomic write).
    
    When batch_id is set (or _CURRENT_BATCH_ID is set), writes to a
    shard file.  Use merge_catalog_shards() to combine after all batches.
    """
    bid = batch_id or _CURRENT_BATCH_ID
    if bid:
        path = METADATA_CATALOG.parent / f"metadata_catalog_shard_{bid}.parquet"
    else:
        path = METADATA_CATALOG
    tmp = path.with_suffix(".tmp")
    df.to_parquet(str(tmp), index=False)
    tmp.rename(path)
    logger.info("Saved metadata catalog%s: %d rows", f" (shard {batch_id})" if batch_id else "", len(df))


def merge_catalog_shards():
    """Merge all shard parquets back into the master metadata_catalog.
    
    Merges column-by-column: for each (gse_id, gsm_id), keeps the most
    advanced status for each stage independently.  This is essential when
    different stages run concurrently (e.g. 2a and 2b) and each shard
    only updates its own stage columns.
    """
    shard_dir = METADATA_CATALOG.parent
    shards = sorted(shard_dir.glob("metadata_catalog_shard_*.parquet"))
    if not shards:
        logger.info("No shards to merge")
        return
    
    # Load master
    if METADATA_CATALOG.exists():
        master = pd.read_parquet(str(METADATA_CATALOG))
    else:
        master = pd.DataFrame()
    
    # Load all shards
    shard_frames = []
    for s in shards:
        shard_frames.append(pd.read_parquet(str(s)))
        logger.info("  Loaded shard: %s (%d rows)", s.name, len(shard_frames[-1]))
    
    if master.empty and not shard_frames:
        return
    
    # Start from master (or first shard if no master)
    if not master.empty:
        result = master.set_index(["gse_id", "gsm_id"])
    else:
        result = shard_frames[0].set_index(["gse_id", "gsm_id"])
        shard_frames = shard_frames[1:]
    
    # Stage-specific status columns to merge intelligently
    _STATUS_PRIORITY = {"done": 3, "failed": 2, "skipped": 1, "pending": 0, "": 0}
    _STAGE_GROUPS = {
        "stage0": ["stage0_status", "stage0_formats"],
        "stage1": ["stage1_status", "stage1_n_cols"],
        "stage2a": ["stage2a_status", "stage2a_format", "stage2a_source",
                     "stage2a_n_cols", "stage2a_match_rate"],
        "stage2b": ["stage2b_status", "stage2b_source",
                     "stage2b_n_cols", "stage2b_match_rate"],
        "stage2c": ["stage2c_status", "stage2c_format", "stage2c_source",
                     "stage2c_n_cols", "stage2c_match_rate"],
        "stage3": ["stage3_status", "stage3_has_summary"],
    }
    
    for shard_df in shard_frames:
        shard = shard_df.set_index(["gse_id", "gsm_id"])
        
        for stage_name, cols in _STAGE_GROUPS.items():
            status_col = f"{stage_name}_status"
            # Find rows where this shard has a more advanced status
            common_idx = result.index.intersection(shard.index)
            if common_idx.empty:
                continue
            
            result_status = result.loc[common_idx, status_col].map(
                lambda x: _STATUS_PRIORITY.get(str(x), 0))
            shard_status = shard.loc[common_idx, status_col].map(
                lambda x: _STATUS_PRIORITY.get(str(x), 0))
            
            # Update rows where shard has higher priority status
            better_mask = shard_status > result_status
            better_idx = common_idx[better_mask]
            
            if len(better_idx) > 0:
                for col in cols:
                    if col in shard.columns and col in result.columns:
                        result.loc[better_idx, col] = shard.loc[better_idx, col]
                # Update timestamp for these rows
                result.loc[better_idx, "updated_at"] = shard.loc[better_idx, "updated_at"]
        
        # Also update error column if shard has a non-empty error
        if "error" in shard.columns:
            common_idx = result.index.intersection(shard.index)
            shard_errors = shard.loc[common_idx, "error"]
            has_error = shard_errors.astype(str).str.len() > 0
            error_idx = common_idx[has_error]
            if len(error_idx) > 0:
                result.loc[error_idx, "error"] = shard.loc[error_idx, "error"]
    
    result = result.reset_index().sort_values(["gse_id", "gsm_id"]).reset_index(drop=True)
    
    # Save merged master
    tmp = METADATA_CATALOG.with_suffix(".tmp")
    result.to_parquet(str(tmp), index=False)
    tmp.rename(METADATA_CATALOG)
    
    # Clean up shards
    for s in shards:
        s.unlink()
    
    logger.info("Merged %d shards into master catalog: %d rows", len(shards), len(result))
    return result


def get_batch_slice(df: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Get a slice of the catalog for this batch."""
    n = len(df)
    start = (n * batch) // total_batches
    end = (n * (batch + 1)) // total_batches
    return df.iloc[start:end].copy()


# ════════════════════════════════════════════════════════════════════════
#  Stage 0 — Classify supplementary files
# ════════════════════════════════════════════════════════════════════════

def run_stage0(cat: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Classify supplementary files for each GSE.

    Reads stage3_soft_parsed.parquet and classifies supplementary file
    URLs by format (h5ad, rds, tabular, loom, skip).  No network needed.
    """
    # Work at GSE level — classify once per GSE
    gse_ids = cat[cat["stage0_status"] == "pending"]["gse_id"].unique()
    gse_ids = sorted(gse_ids)
    n = len(gse_ids)
    start = (n * batch) // total_batches
    end = (n * (batch + 1)) // total_batches
    gse_batch = gse_ids[start:end]

    logger.info("Stage 0: classifying %d GSEs (batch %d/%d, %d pending total)",
                len(gse_batch), batch, total_batches, n)

    done = 0
    failed = 0
    for i, gse_id in enumerate(gse_batch):
        if (i + 1) % 100 == 0:
            logger.info("Stage 0 progress: %d/%d GSEs", i + 1, len(gse_batch))
            save_catalog(cat)  # checkpoint

        try:
            classified = classify_supplementary_files(gse_id)
            fmt_counts = {
                fmt: len(entries) for fmt, entries in classified.items()
            }
            fmt_json = json.dumps(fmt_counts)

            # Determine which GSMs have Tier 2 candidates
            gsm_formats = {}  # gsm_id -> best format available
            for fmt in ("h5ad", "tabular", "loom", "tar"):
                for entry in classified.get(fmt, []):
                    gsm = entry["gsm_id"]
                    if gsm not in gsm_formats:
                        gsm_formats[gsm] = fmt

            rds_gsms = set()
            for entry in classified.get("rds", []):
                rds_gsms.add(entry["gsm_id"])

            # Update catalog for all GSMs in this GSE
            mask = cat["gse_id"] == gse_id
            cat.loc[mask, "stage0_status"] = "done"
            cat.loc[mask, "stage0_formats"] = fmt_json
            cat.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()

            # Mark GSMs with no Tier 2 candidates as "skipped" for stage2a/2b
            gse_gsms = cat.loc[mask, "gsm_id"].tolist()
            for gsm_id in gse_gsms:
                gsm_mask = (cat["gse_id"] == gse_id) & (cat["gsm_id"] == gsm_id)
                if gsm_id not in gsm_formats and gse_id not in gsm_formats:
                    cat.loc[gsm_mask, "stage2a_status"] = "skipped"
                if gsm_id not in rds_gsms and gse_id not in rds_gsms:
                    cat.loc[gsm_mask, "stage2b_status"] = "skipped"

            done += 1

        except Exception as e:
            mask = cat["gse_id"] == gse_id
            cat.loc[mask, "stage0_status"] = "failed"
            cat.loc[mask, "error"] = str(e)[:500]
            cat.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
            failed += 1
            logger.warning("Stage 0 failed for %s: %s", gse_id, e)

    save_catalog(cat)
    logger.info("Stage 0 complete: %d done, %d failed", done, failed)
    return cat


# ════════════════════════════════════════════════════════════════════════
#  Stage 1 — Tier 1: SOFT characteristics
# ════════════════════════════════════════════════════════════════════════

def run_stage1(cat: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Extract Tier 1 SOFT characteristics for each GSM.

    Reads stage3_soft_parsed.parquet, builds per-cell obs from sample
    characteristics, and saves to dataset/{GSE}/{GSM}/author_metadata.parquet.
    """
    pending = cat[cat["stage1_status"] == "pending"]
    batch_slice = get_batch_slice(pending, batch, total_batches)
    gse_groups = batch_slice.groupby("gse_id")

    logger.info("Stage 1: processing %d GSMs in %d GSEs (batch %d/%d)",
                len(batch_slice), len(gse_groups), batch, total_batches)

    done = 0
    failed = 0
    # Process by GSE to share SOFT metadata loading
    for gse_i, (gse_id, group) in enumerate(gse_groups):
        if (gse_i + 1) % 50 == 0:
            logger.info("Stage 1 progress: %d/%d GSEs", gse_i + 1, len(gse_groups))
            save_catalog(cat)

        try:
            soft_meta = load_soft_metadata(gse_id)
        except Exception as e:
            mask = cat["gse_id"] == gse_id
            cat.loc[mask & (cat["stage1_status"] == "pending"), "stage1_status"] = "failed"
            cat.loc[mask & (cat["error"] == ""), "error"] = f"SOFT load: {e}"
            cat.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
            failed += len(group)
            continue

        for _, row in group.iterrows():
            gsm_id = row["gsm_id"]
            idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == gsm_id)].index

            try:
                tier1_obs = build_gsm_level_obs(gse_id, gsm_id, soft_meta)
                if tier1_obs.empty:
                    cat.loc[idx, "stage1_status"] = "failed"
                    cat.loc[idx, "error"] = "No cells.parquet"
                    continue

                # Save Tier 1 metadata
                out_path = DATASET_DIR / gse_id / gsm_id / "author_metadata.parquet"
                tier1_obs.to_parquet(str(out_path), index=True)

                cat.loc[idx, "stage1_status"] = "done"
                cat.loc[idx, "stage1_n_cols"] = len(tier1_obs.columns)
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                done += 1

            except Exception as e:
                cat.loc[idx, "stage1_status"] = "failed"
                cat.loc[idx, "error"] = str(e)[:500]
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                failed += 1
                logger.warning("Stage 1 failed for %s/%s: %s", gse_id, gsm_id, e)

    save_catalog(cat)
    logger.info("Stage 1 complete: %d done, %d failed", done, failed)
    return cat


# ════════════════════════════════════════════════════════════════════════
#  Stage 2a — Tier 2: tabular / h5ad / loom
# ════════════════════════════════════════════════════════════════════════

def run_stage2a(cat: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Extract Tier 2 cell-level metadata from Python-readable formats.

    Downloads supplementary files (h5ad, CSV/TSV, loom), extracts all
    metadata columns, aligns to our barcodes, and merges with existing
    author_metadata.parquet.

    V2 architecture: extracts each file ONCE at GSE level, caches the
    resulting DataFrame, then aligns it against every GSM in the series.
    This handles GSE-level metadata files that contain barcodes for
    multiple samples and files attached to other GSMs in the series.
    """
    pending = cat[cat["stage2a_status"] == "pending"]
    batch_slice = get_batch_slice(pending, batch, total_batches)
    gse_groups = batch_slice.groupby("gse_id")

    logger.info("Stage 2a: processing %d GSMs in %d GSEs (batch %d/%d)",
                len(batch_slice), len(gse_groups), batch, total_batches)

    done = 0
    failed = 0
    skipped = 0
    work_dir = WORK_DIR / f"stage2a_batch{batch}"
    work_dir.mkdir(parents=True, exist_ok=True)

    for gse_i, (gse_id, group) in enumerate(gse_groups):
        if (gse_i + 1) % 20 == 0:
            logger.info("Stage 2a progress: %d/%d GSEs (%d done, %d failed)",
                        gse_i + 1, len(gse_groups), done, failed)
            save_catalog(cat)

        try:
            classified = classify_supplementary_files(gse_id)
        except Exception as e:
            for _, row in group.iterrows():
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == row["gsm_id"])].index
                cat.loc[idx, "stage2a_status"] = "failed"
                cat.loc[idx, "error"] = f"classify: {e}"
            failed += len(group)
            continue

        # ── Extract ALL metadata files for this GSE once ──
        gse_work = work_dir / gse_id
        extracted_cache = _extract_gse_metadata_files(
            gse_id, classified, gse_work
        )

        if not extracted_cache:
            # No usable metadata files found for this GSE at all
            for _, row in group.iterrows():
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == row["gsm_id"])].index
                cat.loc[idx, "stage2a_status"] = "skipped"
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                skipped += 1
            _cleanup_dir(gse_work)
            continue

        # ── Try each cached extraction against each GSM ──
        for _, row in group.iterrows():
            gsm_id = row["gsm_id"]
            idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == gsm_id)].index

            try:
                best_aligned = None
                best_stats = {"match_rate": 0.0}
                best_source = ""
                best_fmt = ""

                for source_fn, fmt, author_obs in extracted_cache:
                    aligned, stats = align_author_metadata(author_obs, gse_id, gsm_id)
                    mr = stats.get("match_rate", 0.0)
                    if mr > best_stats["match_rate"]:
                        best_aligned = aligned
                        best_stats = stats
                        best_source = source_fn
                        best_fmt = fmt
                    # If we hit ≥90% match, no need to try more files
                    if mr >= 0.9:
                        break

                if best_aligned is None or best_stats["match_rate"] == 0:
                    cat.loc[idx, "stage2a_status"] = "done"
                    cat.loc[idx, "stage2a_format"] = best_fmt or ""
                    cat.loc[idx, "stage2a_source"] = best_source or ""
                    cat.loc[idx, "stage2a_n_cols"] = 0
                    cat.loc[idx, "stage2a_match_rate"] = 0.0
                    cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                    done += 1
                    continue

                # Merge with existing Tier 1 metadata
                _merge_tier2_into_parquet(gse_id, gsm_id, best_aligned)

                cat.loc[idx, "stage2a_status"] = "done"
                cat.loc[idx, "stage2a_format"] = best_fmt or ""
                cat.loc[idx, "stage2a_source"] = best_source or ""
                cat.loc[idx, "stage2a_n_cols"] = len([c for c in best_aligned.columns if not best_aligned[c].isna().all()])
                cat.loc[idx, "stage2a_match_rate"] = best_stats.get("match_rate", 0.0)
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                done += 1

            except Exception as e:
                cat.loc[idx, "stage2a_status"] = "failed"
                cat.loc[idx, "error"] = str(e)[:500]
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                failed += 1
                logger.warning("Stage 2a failed for %s/%s: %s", gse_id, gsm_id, e)

        # Clean up GSE work dir
        _cleanup_dir(gse_work)

    save_catalog(cat)
    logger.info("Stage 2a complete: %d done, %d skipped, %d failed", done, skipped, failed)
    return cat


def _extract_gse_metadata_files(gse_id, classified, work_dir):
    """Extract ALL metadata-candidate files for a GSE, return cached results.

    Returns list of (source_filename, format, author_obs_DataFrame) for
    each file that successfully extracted cell-level metadata.
    Files are tried in priority order: metadata-like filenames first,
    h5ad > tabular > loom > tar.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Collect ALL candidates across ALL GSMs (not just the target GSM)
    candidates = []
    seen_filenames = set()
    for fmt in ("h5ad", "tabular", "loom", "tar"):
        entries = classified.get(fmt, [])
        for entry in entries:
            fn = entry["url"].rstrip("/").split("/")[-1]
            fn_lower = fn.lower()
            if fn_lower in seen_filenames:
                continue
            seen_filenames.add(fn_lower)
            candidates.append((entry, fmt))

    # Sort: metadata-like filenames first
    candidates.sort(key=lambda x: not x[0].get("is_metadata_like", False))

    extractors = {
        "h5ad": extract_metadata_from_h5ad,
        "tabular": extract_metadata_from_tabular,
        "loom": extract_metadata_from_loom,
    }

    results = []
    expression_salvage_paths = []  # (dest, fn) for expression matrices to salvage later
    for entry, fmt in candidates:
        url = entry["url"]
        fn = url.rstrip("/").split("/")[-1]
        dest = work_dir / fn

        if not dest.exists():
            if not download_supplementary_file(url, dest):
                continue

        # Skip files > 2GB to prevent OOM
        try:
            file_size_mb = dest.stat().st_size / (1024 * 1024)
            if file_size_mb > 2000:
                logger.info("Stage 2a: skipping %s (%.0f MB, too large)", fn, file_size_mb)
                continue
        except OSError:
            pass

        # ── Tar archives: extract metadata candidates from inside ──
        if fmt == "tar":
            try:
                tar_results = extract_metadata_from_tar(dest)
                for member_fn, member_fmt, obs in tar_results:
                    source_label = f"{fn}::{member_fn}"
                    results.append((source_label, member_fmt, obs))
                    if len(results) >= 10:
                        break
                logger.info("Stage 2a: %s tar yielded %d metadata extractions", fn, len(tar_results))
            except Exception as e:
                logger.debug("Stage 2a: tar extraction failed for %s: %s", fn, e)
            if len(results) >= 10:
                break
            continue

        # Content-based check for tabular files: skip expression matrices
        # but remember them for salvage
        if fmt == "tabular":
            try:
                peek = peek_tabular_content(dest)
                if peek["is_expression"]:
                    logger.info("Stage 2a: %s is expression matrix — queued for salvage", fn)
                    expression_salvage_paths.append((dest, fn))
                    continue
            except Exception:
                pass

        extractor = extractors.get(fmt)
        if extractor is None:
            continue

        try:
            obs = extractor(dest)
            if not obs.empty and "barcode" in obs.columns and len(obs) > 0:
                logger.info("Stage 2a: %s extracted %d cells from %s (%s)",
                            gse_id, len(obs), fn, fmt)
                results.append((fn, fmt, obs))
        except Exception as e:
            logger.debug("Stage 2a: %s extract failed: %s", fn, e)
            continue

        # Cap at 10 successful extractions per GSE to avoid excessive memory
        if len(results) >= 10:
            break

    # ── Expression matrix salvage: try to extract annotation columns ──
    if len(results) < 10:
        for dest, fn in expression_salvage_paths:
            try:
                obs = extract_metadata_from_expression(dest)
                if not obs.empty and "barcode" in obs.columns and len(obs) > 0:
                    logger.info("Stage 2a: %s salvaged %d cells from expression matrix %s",
                                gse_id, len(obs), fn)
                    results.append((fn, "expression-salvage", obs))
            except Exception as e:
                logger.debug("Stage 2a: expression salvage failed for %s: %s", fn, e)
            if len(results) >= 10:
                break

    return results


# ════════════════════════════════════════════════════════════════════════
#  Stage 2b — Tier 2: RDS / Seurat
# ════════════════════════════════════════════════════════════════════════

def run_stage2b(cat: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Extract Tier 2 metadata from RDS/Seurat supplementary files.

    Requires rpy2 and R with Seurat/SingleCellExperiment packages.
    """
    try:
        from scgeo.metadata.extract_rds import extract_metadata_from_rds
    except ImportError:
        logger.error("rpy2 not available — cannot run Stage 2b")
        return cat

    pending = cat[cat["stage2b_status"] == "pending"]
    batch_slice = get_batch_slice(pending, batch, total_batches)
    gse_groups = batch_slice.groupby("gse_id")

    logger.info("Stage 2b: processing %d GSMs in %d GSEs (batch %d/%d)",
                len(batch_slice), len(gse_groups), batch, total_batches)

    done = 0
    failed = 0
    skipped = 0
    work_dir = WORK_DIR / f"stage2b_batch{batch}"
    work_dir.mkdir(parents=True, exist_ok=True)

    for gse_i, (gse_id, group) in enumerate(gse_groups):
        if (gse_i + 1) % 10 == 0:
            logger.info("Stage 2b progress: %d/%d GSEs (%d done, %d failed)",
                        gse_i + 1, len(gse_groups), done, failed)
            save_catalog(cat)

        try:
            classified = classify_supplementary_files(gse_id)
        except Exception:
            for _, row in group.iterrows():
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == row["gsm_id"])].index
                cat.loc[idx, "stage2b_status"] = "failed"
            failed += len(group)
            continue

        rds_entries = classified.get("rds", [])
        if not rds_entries:
            for _, row in group.iterrows():
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == row["gsm_id"])].index
                cat.loc[idx, "stage2b_status"] = "skipped"
            skipped += len(group)
            continue

        for _, row in group.iterrows():
            gsm_id = row["gsm_id"]
            idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == gsm_id)].index

            try:
                # Find RDS for this GSM
                gsm_rds = [e for e in rds_entries if e["gsm_id"] in (gsm_id, gse_id)]
                gsm_rds = sorted(gsm_rds, key=lambda e: not e.get("is_metadata_like", False))

                if not gsm_rds:
                    cat.loc[idx, "stage2b_status"] = "skipped"
                    skipped += 1
                    continue

                entry = gsm_rds[0]
                url = entry["url"]
                fn = url.rstrip("/").split("/")[-1]
                dest = work_dir / gse_id / fn
                dest.parent.mkdir(parents=True, exist_ok=True)

                if not dest.exists():
                    if not download_supplementary_file(url, dest):
                        cat.loc[idx, "stage2b_status"] = "failed"
                        cat.loc[idx, "error"] = f"download failed: {fn}"
                        failed += 1
                        continue

                author_obs = extract_metadata_from_rds(dest)

                if author_obs.empty:
                    cat.loc[idx, "stage2b_status"] = "done"
                    cat.loc[idx, "stage2b_n_cols"] = 0
                    done += 1
                    continue

                aligned, stats = align_author_metadata(author_obs, gse_id, gsm_id)
                _merge_tier2_into_parquet(gse_id, gsm_id, aligned)

                cat.loc[idx, "stage2b_status"] = "done"
                cat.loc[idx, "stage2b_source"] = fn
                cat.loc[idx, "stage2b_n_cols"] = len([c for c in aligned.columns if not aligned[c].isna().all()])
                cat.loc[idx, "stage2b_match_rate"] = stats.get("match_rate", 0.0)
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                done += 1

            except Exception as e:
                cat.loc[idx, "stage2b_status"] = "failed"
                cat.loc[idx, "error"] = str(e)[:500]
                failed += 1
                logger.warning("Stage 2b failed for %s/%s: %s", gse_id, gsm_id, e)

        _cleanup_dir(work_dir / gse_id)

    save_catalog(cat)
    logger.info("Stage 2b complete: %d done, %d skipped, %d failed", done, skipped, failed)
    return cat


# ════════════════════════════════════════════════════════════════════════
#  Stage 2c — Tier 2: Additional formats (XLSX, H5Seurat, RData, HDF5)
# ════════════════════════════════════════════════════════════════════════

def run_stage2c(cat: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Extract Tier 2 metadata from additional supplementary formats.

    Handles formats not covered by 2a (tabular/h5ad/loom) or 2b (RDS):
      - XLSX/XLS — Excel files with cell annotations
      - H5Seurat — HDF5-based Seurat objects (meta.data group via h5py)
      - RData/RDA/RObj — R workspace objects (via rpy2)
      - HDF5 — Generic HDF5 with obs-like groups
      - H5MU — MuData format (.obs group)
    """
    pending = cat[cat["stage2c_status"] == "pending"]
    batch_slice = get_batch_slice(pending, batch, total_batches)
    gse_groups = batch_slice.groupby("gse_id")

    logger.info("Stage 2c: processing %d GSMs in %d GSEs (batch %d/%d)",
                len(batch_slice), len(gse_groups), batch, total_batches)

    done = 0
    failed = 0
    skipped = 0
    work_dir = WORK_DIR / f"stage2c_batch{batch}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Format priority: h5seurat > xlsx > rdata > h5mu > hdf5
    _2C_FORMATS = ("h5seurat", "xlsx", "rdata", "h5mu", "hdf5")

    _2C_EXTRACTORS = {
        "xlsx": extract_metadata_from_xlsx,
        "h5seurat": extract_metadata_from_h5seurat,
        "hdf5": extract_metadata_from_hdf5,
        "h5mu": extract_metadata_from_h5mu,
    }

    # RData extractor needs lazy import (rpy2)
    _rdata_extractor = None

    for gse_i, (gse_id, group) in enumerate(gse_groups):
        if (gse_i + 1) % 20 == 0:
            logger.info("Stage 2c progress: %d/%d GSEs (%d done, %d skipped, %d failed)",
                        gse_i + 1, len(gse_groups), done, skipped, failed)
            save_catalog(cat)

        try:
            classified = classify_supplementary_files(gse_id)
        except Exception as e:
            for _, row in group.iterrows():
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == row["gsm_id"])].index
                cat.loc[idx, "stage2c_status"] = "failed"
                cat.loc[idx, "error"] = f"classify: {e}"
            failed += len(group)
            continue

        # Check if this GSE has any 2c-eligible files
        has_2c = any(classified.get(fmt, []) for fmt in _2C_FORMATS)
        if not has_2c:
            for _, row in group.iterrows():
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == row["gsm_id"])].index
                cat.loc[idx, "stage2c_status"] = "skipped"
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
            skipped += len(group)
            continue

        # Per-GSE extraction cache: extract each unique file only once
        _gse_extraction_cache = {}  # {filename: (author_obs, fmt) or None}

        for _, row in group.iterrows():
            gsm_id = row["gsm_id"]
            idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == gsm_id)].index

            try:
                author_obs, source_fn, fmt = _try_extract_2c(
                    gse_id, gsm_id, classified, work_dir / gse_id,
                    _2C_FORMATS, _2C_EXTRACTORS,
                    _extraction_cache=_gse_extraction_cache,
                )

                if author_obs is None or author_obs.empty:
                    cat.loc[idx, "stage2c_status"] = "skipped"
                    cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                    skipped += 1
                    continue

                # Align to our barcodes
                aligned, stats = align_author_metadata(author_obs, gse_id, gsm_id)

                if aligned.empty or stats.get("match_rate", 0) == 0:
                    cat.loc[idx, "stage2c_status"] = "done"
                    cat.loc[idx, "stage2c_format"] = fmt or ""
                    cat.loc[idx, "stage2c_source"] = source_fn or ""
                    cat.loc[idx, "stage2c_n_cols"] = 0
                    cat.loc[idx, "stage2c_match_rate"] = 0.0
                    cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                    done += 1
                    continue

                _merge_tier2_into_parquet(gse_id, gsm_id, aligned)

                cat.loc[idx, "stage2c_status"] = "done"
                cat.loc[idx, "stage2c_format"] = fmt or ""
                cat.loc[idx, "stage2c_source"] = source_fn or ""
                cat.loc[idx, "stage2c_n_cols"] = len([c for c in aligned.columns if not aligned[c].isna().all()])
                cat.loc[idx, "stage2c_match_rate"] = stats.get("match_rate", 0.0)
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                done += 1

            except Exception as e:
                cat.loc[idx, "stage2c_status"] = "failed"
                cat.loc[idx, "error"] = str(e)[:500]
                cat.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
                failed += 1
                logger.warning("Stage 2c failed for %s/%s: %s", gse_id, gsm_id, e)

        _cleanup_dir(work_dir / gse_id)

    save_catalog(cat)
    logger.info("Stage 2c complete: %d done, %d skipped, %d failed", done, skipped, failed)
    return cat


def _try_extract_2c(gse_id, gsm_id, classified, work_dir, formats, extractors,
                    _extraction_cache=None):
    """Try to extract metadata from 2c-eligible supplementary files.

    Returns (author_obs, source_filename, format) or (None, None, None).
    Uses _extraction_cache to avoid re-extracting the same file for each GSM.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if _extraction_cache is None:
        _extraction_cache = {}

    # Lazy-load rdata extractor
    rdata_fn = None
    try:
        from scgeo.metadata.extract_rds import extract_metadata_from_rdata
        rdata_fn = extract_metadata_from_rdata
    except ImportError:
        pass

    candidates = []
    for fmt in formats:
        entries = classified.get(fmt, [])
        entries = sorted(entries, key=lambda e: (
            not e.get("is_metadata_like", False),
            e["gsm_id"] != gsm_id,
        ))
        for entry in entries:
            if entry["gsm_id"] in (gsm_id, gse_id):
                candidates.append((entry, fmt))

    for entry, fmt in candidates:
        url = entry["url"]
        fn = url.rstrip("/").split("/")[-1]

        # Check per-GSE extraction cache
        if fn in _extraction_cache:
            cached = _extraction_cache[fn]
            if cached is not None:
                obs, cached_fmt = cached
                logger.info("Stage 2c: %s/%s using cached extraction from %s (%s, %d cells)",
                            gse_id, gsm_id, fn, cached_fmt, len(obs))
                return obs, fn, cached_fmt
            continue  # previously failed/empty

        dest = work_dir / fn

        if not dest.exists():
            if not download_supplementary_file(url, dest):
                _extraction_cache[fn] = None
                continue

        # Skip files > 2GB
        try:
            if dest.stat().st_size > 2_000_000_000:
                logger.info("Stage 2c: skipping %s (too large)", fn)
                _extraction_cache[fn] = None
                continue
        except OSError:
            pass

        extractor = extractors.get(fmt)
        if fmt == "rdata":
            extractor = rdata_fn

        if extractor is None:
            logger.debug("Stage 2c: no extractor for %s (%s)", fn, fmt)
            _extraction_cache[fn] = None
            continue

        try:
            obs = extractor(dest)
            if not obs.empty and "barcode" in obs.columns and len(obs) > 0:
                logger.info("Stage 2c: %s/%s extracted %d cells from %s (%s)",
                            gse_id, gsm_id, len(obs), fn, fmt)
                _extraction_cache[fn] = (obs, fmt)
                return obs, fn, fmt
            else:
                _extraction_cache[fn] = None
        except Exception as e:
            logger.debug("Stage 2c: %s failed for %s: %s", fn, gsm_id, e)
            _extraction_cache[fn] = None
            continue

    return None, None, None


# ════════════════════════════════════════════════════════════════════════
#  Stage 3 — Tier 3: NCBI descriptions
# ════════════════════════════════════════════════════════════════════════

def run_stage3(cat: pd.DataFrame, batch: int, total_batches: int) -> pd.DataFrame:
    """Fetch NCBI descriptions and save to author_metadata.parquet uns."""
    # Work at GSE level (one API call per GSE)
    gse_ids = cat[cat["stage3_status"] == "pending"]["gse_id"].unique()
    gse_ids = sorted(gse_ids)
    n = len(gse_ids)
    start = (n * batch) // total_batches
    end = (n * (batch + 1)) // total_batches
    gse_batch = gse_ids[start:end]

    logger.info("Stage 3: fetching descriptions for %d GSEs (batch %d/%d)", len(gse_batch), batch, total_batches)

    done = 0
    failed = 0

    for i, gse_id in enumerate(gse_batch):
        if (i + 1) % 50 == 0:
            logger.info("Stage 3 progress: %d/%d GSEs", i + 1, len(gse_batch))
            save_catalog(cat)

        try:
            desc = fetch_geo_description(gse_id)
            has_summary = bool(desc.get("summary", ""))

            # Update all GSMs in this GSE
            mask = cat["gse_id"] == gse_id
            gsm_ids = cat.loc[mask, "gsm_id"].tolist()

            for gsm_id in gsm_ids:
                idx = cat[(cat["gse_id"] == gse_id) & (cat["gsm_id"] == gsm_id)].index
                meta_path = DATASET_DIR / gse_id / gsm_id / "author_metadata.parquet"
                if meta_path.exists():
                    try:
                        _save_uns(meta_path, gse_id, gsm_id, desc)
                    except Exception as e:
                        logger.debug("Failed to update uns for %s/%s: %s", gse_id, gsm_id, e)

            cat.loc[mask, "stage3_status"] = "done"
            cat.loc[mask, "stage3_has_summary"] = has_summary
            cat.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
            done += 1

            # Rate limit: ~3 requests/sec
            time.sleep(0.35)

        except Exception as e:
            mask = cat["gse_id"] == gse_id
            cat.loc[mask, "stage3_status"] = "failed"
            cat.loc[mask, "error"] = str(e)[:500]
            cat.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
            failed += 1
            logger.warning("Stage 3 failed for %s: %s", gse_id, e)

    save_catalog(cat)
    logger.info("Stage 3 complete: %d GSEs done, %d failed", done, failed)
    return cat


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════

def _merge_tier2_into_parquet(gse_id: str, gsm_id: str, tier2_aligned: pd.DataFrame):
    """Merge Tier 2 columns into existing author_metadata.parquet."""
    meta_path = DATASET_DIR / gse_id / gsm_id / "author_metadata.parquet"

    # Coerce mixed-type columns to avoid Arrow schema errors
    for col in tier2_aligned.columns:
        if tier2_aligned[col].dtype == object:
            tier2_aligned[col] = tier2_aligned[col].astype(str)

    if meta_path.exists():
        existing = pd.read_parquet(str(meta_path))
        # Add Tier 2 columns, don't overwrite Tier 1
        for col in tier2_aligned.columns:
            if col not in existing.columns:
                # Align by index (barcodes)
                if set(tier2_aligned.index) & set(existing.index):
                    existing[col] = tier2_aligned[col].reindex(existing.index)
                else:
                    existing[col] = tier2_aligned[col].values[:len(existing)] if len(tier2_aligned) == len(existing) else None
        # Coerce any remaining mixed-type columns
        for col in existing.columns:
            if existing[col].dtype == object:
                existing[col] = existing[col].astype(str)
        existing.to_parquet(str(meta_path), index=True)
    else:
        tier2_aligned.to_parquet(str(meta_path), index=True)


def _save_uns(meta_path: Path, gse_id: str, gsm_id: str, desc: dict):
    """Save Tier 3 description alongside the parquet as a JSON sidecar."""
    uns_path = meta_path.parent / "author_metadata_uns.json"
    uns = {}
    if uns_path.exists():
        with open(uns_path) as f:
            uns = json.load(f)

    uns.update({
        "gse_id": gse_id,
        "gsm_id": gsm_id,
        "title": desc.get("title", ""),
        "summary": desc.get("summary", ""),
        "organism": desc.get("organism", ""),
        "pubmed_ids": desc.get("pubmed_ids", []),
        "ncbi_fetched_at": datetime.now(timezone.utc).isoformat(),
    })

    with open(uns_path, "w") as f:
        json.dump(uns, f, indent=2)


def _cleanup_dir(d: Path):
    """Remove a temporary work directory if it exists."""
    import shutil
    if d.exists():
        try:
            shutil.rmtree(d)
        except Exception:
            pass


def reconcile_catalog(cat: pd.DataFrame) -> pd.DataFrame:
    """Reconcile catalog status against actual files on disk.

    Scans author_metadata.parquet files to determine true stage1/2a/2b
    completion.  Fixes catalog entries that were corrupted by the
    row-level merge when stages ran concurrently.
    """
    updated = 0
    for i, row in cat.iterrows():
        gse_id, gsm_id = row["gse_id"], row["gsm_id"]
        meta_path = DATASET_DIR / gse_id / gsm_id / "author_metadata.parquet"
        uns_path = DATASET_DIR / gse_id / gsm_id / "author_metadata_uns.json"

        if not meta_path.exists():
            continue

        try:
            # Read actual parquet to see what columns exist
            meta = pd.read_parquet(str(meta_path))
            n_cols = len(meta.columns)

            # Stage 1: if file exists with >0 columns, it was processed
            if row["stage1_status"] != "done" and n_cols > 0:
                cat.at[i, "stage1_status"] = "done"
                cat.at[i, "stage1_n_cols"] = n_cols
                cat.at[i, "updated_at"] = datetime.now(timezone.utc).isoformat()
                updated += 1

            # Stage 2a/2b: check if there are columns beyond basic SOFT ones
            # SOFT (Tier 1) columns typically come from characteristics
            # Tier 2 columns are author-specific (cell_type, cluster, etc.)
            # If n_cols > stage1_n_cols, Tier 2 was merged
            stage1_cols = int(row.get("stage1_n_cols", 0) or 0)
            if n_cols > stage1_cols and stage1_cols > 0:
                # Tier 2 columns were added
                if row["stage2a_status"] == "pending":
                    cat.at[i, "stage2a_status"] = "done"
                    cat.at[i, "stage2a_n_cols"] = n_cols - stage1_cols
                    cat.at[i, "updated_at"] = datetime.now(timezone.utc).isoformat()
                    updated += 1

            # Stage 3: check for uns JSON sidecar
            if uns_path.exists() and row["stage3_status"] != "done":
                try:
                    with open(uns_path) as f:
                        uns = json.load(f)
                    cat.at[i, "stage3_status"] = "done"
                    cat.at[i, "stage3_has_summary"] = bool(uns.get("summary", ""))
                    cat.at[i, "updated_at"] = datetime.now(timezone.utc).isoformat()
                    updated += 1
                except Exception:
                    pass

        except Exception as e:
            logger.debug("Reconcile failed for %s/%s: %s", gse_id, gsm_id, e)

        if (i + 1) % 5000 == 0:
            logger.info("Reconcile progress: %d/%d rows, %d updated", i + 1, len(cat), updated)

    logger.info("Reconciled catalog: %d rows updated out of %d total", updated, len(cat))
    return cat


# ════════════════════════════════════════════════════════════════════════
#  Bulk metadata extraction (ALL GSMs, no cells.parquet needed)
# ════════════════════════════════════════════════════════════════════════

BULK_METADATA = CATALOG_DIR / "all_gsm_metadata.parquet"
BULK_GSE_DESC = CATALOG_DIR / "all_gse_descriptions.parquet"


def run_bulk_soft(batch: int, total_batches: int):
    """Extract sample-level SOFT characteristics for ALL GSMs in the catalog.

    Reads stage3_soft_parsed.parquet directly and produces one row per GSM
    with all parsed characteristics as columns.  No cells.parquet required.

    Output: catalog/all_gsm_metadata.parquet (or shards)
    """
    logger.info("Bulk SOFT: loading stage3_soft_parsed.parquet...")
    soft = pd.read_parquet(str(STAGE3_CATALOG), columns=[
        "gse_id", "gsm_id", "sample_characteristics", "sample_source",
        "overall_design", "sample_data_processing",
    ])
    soft = soft[soft["gsm_id"].notna()].copy()
    logger.info("Bulk SOFT: %d rows from SOFT catalog", len(soft))

    # Batch by GSE
    gse_ids = sorted(soft["gse_id"].unique())
    n = len(gse_ids)
    start = (n * batch) // total_batches
    end = (n * (batch + 1)) // total_batches
    batch_gses = set(gse_ids[start:end])
    batch_soft = soft[soft["gse_id"].isin(batch_gses)]
    logger.info("Bulk SOFT batch %d/%d: %d GSEs, %d GSMs",
                batch, total_batches, len(batch_gses), len(batch_soft))

    # Check what's already done
    shard_path = CATALOG_DIR / f"all_gsm_metadata_shard_{batch}.parquet"
    existing_gsms: set = set()
    if shard_path.exists():
        try:
            existing = pd.read_parquet(str(shard_path), columns=["gsm_id"])
            existing_gsms = set(existing["gsm_id"])
            logger.info("  Shard already has %d GSMs, skipping those", len(existing_gsms))
        except Exception:
            pass

    rows = []
    for i, (_, row) in enumerate(batch_soft.iterrows()):
        gsm_id = row["gsm_id"]
        if gsm_id in existing_gsms:
            continue
        gse_id = row["gse_id"]

        chars = parse_characteristics(row["sample_characteristics"])
        r = {
            "gse_id": gse_id,
            "gsm_id": gsm_id,
            "sample_source": row.get("sample_source", "") or "",
            "overall_design": (row.get("overall_design", "") or "")[:2000],
            "data_processing": (row.get("sample_data_processing", "") or "")[:2000],
        }
        for key, val in chars.items():
            col_name = re.sub(r"[^a-z0-9_]", "_", key)
            r[col_name] = val

        rows.append(r)

        if (i + 1) % 100000 == 0:
            logger.info("  Processed %d/%d GSMs", i + 1, len(batch_soft))

    if not rows:
        logger.info("Bulk SOFT: no new rows to add")
        return

    new_df = pd.DataFrame(rows)
    # Coerce all object columns to string for clean parquet
    for col in new_df.columns:
        if new_df[col].dtype == object:
            new_df[col] = new_df[col].fillna("").astype(str)

    # Append to existing shard or create new
    if shard_path.exists() and existing_gsms:
        existing_df = pd.read_parquet(str(shard_path))
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    tmp = shard_path.with_suffix(".tmp")
    combined.to_parquet(str(tmp), index=False)
    tmp.rename(shard_path)
    logger.info("Bulk SOFT: wrote %d GSMs to %s", len(combined), shard_path.name)


def run_bulk_descriptions(batch: int, total_batches: int):
    """Fetch NCBI descriptions for ALL GSEs.

    Output: catalog/all_gse_descriptions.parquet (or shards)
    """
    # Get all GSEs from SOFT catalog
    soft = pd.read_parquet(str(STAGE3_CATALOG), columns=["gse_id"])
    gse_ids = sorted(soft["gse_id"].unique())
    n = len(gse_ids)
    start = (n * batch) // total_batches
    end = (n * (batch + 1)) // total_batches
    batch_gses = gse_ids[start:end]
    logger.info("Bulk descriptions batch %d/%d: %d GSEs", batch, total_batches, len(batch_gses))

    # Check what's already done
    shard_path = CATALOG_DIR / f"all_gse_descriptions_shard_{batch}.parquet"
    done_gses: set = set()
    if shard_path.exists():
        try:
            existing = pd.read_parquet(str(shard_path), columns=["gse_id"])
            done_gses = set(existing["gse_id"])
            logger.info("  Shard already has %d GSEs, skipping those", len(done_gses))
        except Exception:
            pass

    # Also check master file
    if BULK_GSE_DESC.exists():
        try:
            master = pd.read_parquet(str(BULK_GSE_DESC), columns=["gse_id"])
            done_gses |= set(master["gse_id"])
        except Exception:
            pass

    todo = [g for g in batch_gses if g not in done_gses]
    logger.info("  %d GSEs to fetch (%d already done)", len(todo), len(batch_gses) - len(todo))

    rows = []
    for i, gse_id in enumerate(todo):
        try:
            desc = fetch_geo_description(gse_id)
            rows.append({
                "gse_id": gse_id,
                "title": desc.get("title", ""),
                "summary": desc.get("summary", ""),
                "organism": desc.get("organism", ""),
                "pubmed_ids": json.dumps(desc.get("pubmed_ids", [])),
            })
        except Exception as e:
            rows.append({
                "gse_id": gse_id,
                "title": "",
                "summary": "",
                "organism": "",
                "pubmed_ids": "[]",
                "error": str(e)[:200],
            })
            logger.debug("Failed to fetch %s: %s", gse_id, e)

        if (i + 1) % 100 == 0:
            logger.info("  Progress: %d/%d GSEs", i + 1, len(todo))

        # Rate limit: ~0.5 requests/sec per job (conservative for 4 parallel jobs)
        # NCBI allows 3/sec total without API key, so ~0.75/job with 4 jobs
        time.sleep(2.0)

        # Checkpoint every 500 GSEs
        if (i + 1) % 500 == 0 and rows:
            _save_desc_shard(shard_path, rows, done_gses)
            rows = []

    if rows:
        _save_desc_shard(shard_path, rows, done_gses)

    logger.info("Bulk descriptions done for batch %d", batch)


def _save_desc_shard(shard_path, new_rows, done_gses):
    """Append rows to a description shard parquet."""
    new_df = pd.DataFrame(new_rows)
    for col in new_df.columns:
        if new_df[col].dtype == object:
            new_df[col] = new_df[col].fillna("").astype(str)

    if shard_path.exists():
        existing_df = pd.read_parquet(str(shard_path))
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    tmp = shard_path.with_suffix(".tmp")
    combined.to_parquet(str(tmp), index=False)
    tmp.rename(shard_path)
    done_gses.update(set(new_df["gse_id"]))
    logger.info("  Checkpointed %d GSEs to %s (total: %d)", len(new_df), shard_path.name, len(combined))


def merge_bulk_shards(kind: str):
    """Merge bulk metadata or description shards into master file.
    
    For 'soft' kind: uses incremental concatenation to avoid OOM on large
    datasets with heterogeneous columns. Writes a partitioned parquet dataset
    instead of a single file when the data is very large.
    """
    if kind == "soft":
        prefix = "all_gsm_metadata_shard_"
        master = BULK_METADATA
    elif kind == "desc":
        prefix = "all_gse_descriptions_shard_"
        master = BULK_GSE_DESC
    else:
        raise ValueError(f"Unknown kind: {kind}")

    shards = sorted(CATALOG_DIR.glob(f"{prefix}*.parquet"))
    if not shards:
        logger.info("No shards to merge for %s", kind)
        return

    if kind == "desc":
        # Description shards are small — can concat in memory
        frames = []
        for s in shards:
            frames.append(pd.read_parquet(str(s)))
            logger.info("  Loaded shard: %s (%d rows)", s.name, len(frames[-1]))
        if master.exists():
            frames.insert(0, pd.read_parquet(str(master)))
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["gse_id"], keep="last")
        tmp = master.with_suffix(".tmp")
        combined.to_parquet(str(tmp), index=False)
        tmp.rename(master)
        for s in shards:
            s.unlink()
        logger.info("Merged %d shards into %s: %d rows", len(shards), master.name, len(combined))
        return

    # For 'soft': shards have heterogeneous columns, too large for single concat.
    # Strategy: find common core columns + write each shard's rows with only
    # the core columns (gse_id, gsm_id, sample_source, overall_design,
    # data_processing) plus the SOFT characteristics normalized to a JSON column.
    logger.info("Merging %d bulk SOFT shards using streaming approach", len(shards))

    # First pass: discover all column names and row counts
    all_columns = set()
    total_rows = 0
    for s in shards:
        schema = pq.read_schema(str(s))
        all_columns.update(schema.names)
        total_rows += pq.read_metadata(str(s)).num_rows
        logger.info("  Shard %s: %d rows, %d columns",
                     s.name, pq.read_metadata(str(s)).num_rows, len(schema.names))

    logger.info("  Total: %d rows, %d unique columns across all shards", total_rows, len(all_columns))

    # Core columns that every shard has
    core = ["gse_id", "gsm_id", "sample_source", "overall_design", "data_processing"]
    char_cols = sorted(all_columns - set(core))
    all_cols_ordered = core + char_cols
    logger.info("  Core columns: %d, characteristic columns: %d", len(core), len(char_cols))

    # Write incrementally using PyArrow writer — process one shard at a time
    tmp = master.with_suffix(".tmp")
    writer = None
    rows_written = 0

    for si, s in enumerate(shards):
        df = pd.read_parquet(str(s))
        # Add missing columns as empty strings
        for col in all_cols_ordered:
            if col not in df.columns:
                df[col] = ""
        # Reorder and coerce types
        df = df[all_cols_ordered]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("").astype(str)

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(str(tmp), table.schema)

        writer.write_table(table)
        rows_written += len(df)
        logger.info("  Wrote shard %d/%d: %d rows (total: %d)", si + 1, len(shards), len(df), rows_written)
        del df, table  # free memory

    if writer is not None:
        writer.close()
        tmp.rename(master)
        # Clean up shards
        for s in shards:
            s.unlink()
        logger.info("Merged into %s: %d rows, %d columns", master.name, rows_written, len(all_cols_ordered))
    else:
        logger.info("No data written")


def run_retry_stage2(cat: pd.DataFrame, stage: str, batch: int, total_batches: int) -> pd.DataFrame:
    """Retry failed/poor Stage 2a/2b items by resetting them to pending and re-running.

    Resets four categories:
      1. ``failed`` items (errors in previous run)
      2. ``done`` items with 0% match rate AND 0 columns (extracted nothing)
      3. ``skipped`` items whose GSE has h5ad/tabular/loom files (tried
         cross-GSM thanks to GSE-level extraction)
      4. ``skipped`` items whose GSE has tar files or expression matrices
         that may now yield metadata with tar extraction + expression salvage
    """
    status_col = f"stage{stage}_status"
    failed_mask = cat[status_col] == "failed"
    n_failed = failed_mask.sum()

    # Reset done-with-zero-match items that might benefit from improved matching
    zero_match_col = f"stage{stage}_match_rate"
    zero_cols_col = f"stage{stage}_n_cols"
    n_zero = 0
    if zero_match_col in cat.columns and zero_cols_col in cat.columns:
        done_zero_mask = (cat[status_col] == "done") & (cat[zero_match_col] == 0) & (cat[zero_cols_col] == 0)
        n_zero = done_zero_mask.sum()
        cat.loc[done_zero_mask, status_col] = "pending"
        logger.info("Reset %d done-with-zero items for stage%s to pending for retry", n_zero, stage)

    # Reset skipped items whose GSE has metadata files (they were skipped
    # because files were attached to other GSMs, but GSE-level extraction
    # now tries ALL files against ALL GSMs)
    n_skipped_reset = 0
    n_tar_reset = 0
    if stage == "2a":
        skipped_mask = cat[status_col] == "skipped"
        skipped_with_files = skipped_mask.copy()
        skipped_with_tar = skipped_mask.copy()
        for idx in cat[skipped_mask].index:
            try:
                fmt_str = cat.loc[idx, "stage0_formats"]
                d = json.loads(fmt_str) if isinstance(fmt_str, str) and fmt_str else {}
                has_pyfiles = d.get("h5ad", 0) + d.get("tabular", 0) + d.get("loom", 0) > 0
                has_tar = d.get("tar", 0) > 0
                # Also check if skip bucket has tar files (stage0 may have
                # classified them before the tar format was added)
                has_skip_tar = d.get("skip", 0) > 0 and not has_pyfiles and not has_tar
                skipped_with_files.loc[idx] = has_pyfiles
                skipped_with_tar.loc[idx] = has_tar or has_skip_tar
            except Exception:
                skipped_with_files.loc[idx] = False
                skipped_with_tar.loc[idx] = False
        n_skipped_reset = skipped_with_files.sum()
        n_tar_reset = skipped_with_tar.sum()
        cat.loc[skipped_with_files, status_col] = "pending"
        cat.loc[skipped_with_tar, status_col] = "pending"
        logger.info("Reset %d skipped-with-files + %d skipped-with-tar items for stage%s to pending",
                     n_skipped_reset, n_tar_reset, stage)

    cat.loc[failed_mask, status_col] = "pending"
    cat.loc[failed_mask, "error"] = ""
    total_pending = (cat[status_col] == "pending").sum()
    logger.info("Reset %d failed items for stage%s to pending (total pending: %d)",
                n_failed, stage, total_pending)

    if stage == "2a":
        return run_stage2a(cat, batch, total_batches)
    elif stage == "2b":
        return run_stage2b(cat, batch, total_batches)
    elif stage == "2c":
        return run_stage2c(cat, batch, total_batches)
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Staged metadata extraction pipeline")
    parser.add_argument("--stage", required=True,
                        choices=["0", "1", "2a", "2b", "2c", "3",
                                 "merge", "reconcile",
                                 "bulk-soft", "bulk-desc", "bulk-merge-soft", "bulk-merge-desc",
                                 "retry-2a", "retry-2b", "retry-2c"],
                        help="Pipeline stage to run")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch index (0-based)")
    parser.add_argument("--total-batches", type=int, default=1,
                        help="Total number of batches")
    args = parser.parse_args()

    if args.stage == "merge":
        logger.info("Merging catalog shards...")
        merged = merge_catalog_shards()
        if merged is not None:
            for stage in ["stage0", "stage1", "stage2a", "stage2b", "stage2c", "stage3"]:
                col = f"{stage}_status"
                counts = merged[col].value_counts().to_dict()
                logger.info("  %s: %s", stage, counts)
        return

    if args.stage == "reconcile":
        logger.info("Reconciling catalog against actual files on disk...")
        cat = init_metadata_catalog()
        reconciled = reconcile_catalog(cat)
        save_catalog(reconciled)
        for stage in ["stage0", "stage1", "stage2a", "stage2b", "stage2c", "stage3"]:
            col = f"{stage}_status"
            if col in reconciled.columns:
                counts = reconciled[col].value_counts().to_dict()
                logger.info("  %s: %s", stage, counts)
        return

    if args.stage == "bulk-soft":
        run_bulk_soft(args.batch, args.total_batches)
        return

    if args.stage == "bulk-desc":
        run_bulk_descriptions(args.batch, args.total_batches)
        return

    if args.stage.startswith("bulk-merge-"):
        kind = args.stage.replace("bulk-merge-", "")
        merge_bulk_shards(kind)
        return

    batch_id = f"s{args.stage}_b{args.batch}"
    global _CURRENT_BATCH_ID
    _CURRENT_BATCH_ID = batch_id
    logger.info("Starting metadata pipeline stage %s (batch %d/%d, shard=%s)",
                args.stage, args.batch, args.total_batches, batch_id)

    cat = init_metadata_catalog()

    # Print summary
    for stage in ["stage0", "stage1", "stage2a", "stage2b", "stage2c", "stage3"]:
        col = f"{stage}_status"
        if col in cat.columns:
            counts = cat[col].value_counts().to_dict()
            logger.info("  %s: %s", stage, counts)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage.startswith("retry-"):
        stage_suffix = args.stage.replace("retry-", "")
        cat = run_retry_stage2(cat, stage_suffix, args.batch, args.total_batches)
    else:
        runners = {
            "0": run_stage0,
            "1": run_stage1,
            "2a": run_stage2a,
            "2b": run_stage2b,
            "2c": run_stage2c,
            "3": run_stage3,
        }
        cat = runners[args.stage](cat, args.batch, args.total_batches)

    # Final summary
    logger.info("── Final catalog summary (shard %s) ──", batch_id)
    for stage in ["stage0", "stage1", "stage2a", "stage2b", "stage2c", "stage3"]:
        col = f"{stage}_status"
        if col in cat.columns:
            counts = cat[col].value_counts().to_dict()
            logger.info("  %s: %s", stage, counts)


if __name__ == "__main__":
    main()
