#!/usr/bin/env python3
"""Build a comprehensive processing catalog with status for every GEO sample.

Aggregates data from:
  1. Master catalog (geo_single_cell_catalog.parquet) — 1.49M samples
  2. V5 production batch results CSVs (82K samples)
  3. Retry-v3 and retry-v4 batch results CSVs
  4. Migrated dataset manifest.json files (20K+ samples)

Outputs:
  processing_catalog.parquet — one row per gsm_id with:
    - All catalog metadata columns
    - processing_status: authoritative lifecycle status
    - target_assay: intended assay type (scrna, cite, atac, spatial, etc.)
    - production_ready: True if ready for next production batch
    - failure_category: fine-grained failure classification
    - failure_detail: error message from pipeline
    - actionable: whether the failure can be addressed
    - screen_*: pre-compute screening flags
    - migration_status: whether data is in cellarium dataset layout
    - avg_read_length: base_count / read_count for triage

Target Assay Categories
=======================
  scrna             — droplet-based scRNA-seq (10x, Drop-seq, InDrop, etc.)
  cite              — CITE-seq / protein+RNA (RNA component is 10x GEX)
  atac              — scATAC-seq (10x ATAC, sci-ATAC)
  spatial           — spatial transcriptomics (Visium, Slide-seq)
  plate_scrna       — plate-based scRNA (Smart-seq2/3, MARS-seq, CEL-seq, etc.)
  multiome          — 10x Multiome (ATAC + GEX paired)
  vdj               — VDJ/BCR/TCR immune repertoire
  non_rna           — non-RNA assays (ChIP-seq, methylation, Hi-C, etc.)
  ambiguous         — insufficient metadata to classify

Status Taxonomy
===============
Terminal:
  done              — quantified, QC passed, migrated to dataset/
  done_qc_warn      — quantified, migrated, but QC flagged

Skipped (current phase):
  skip_plate_based  — plate-based protocol (droplet-only pipeline)
  skip_likely_plate — heuristic: unknown protocol + bulk SRA tags
  skip_spatial      — spatial data (Visium, Slide-seq) — future target
  skip_atac         — ATAC-seq only — future target
  skip_vdj          — VDJ/BCR/TCR library, not gene expression
  skip_non_rna      — non-RNA protocol (methylation, ChIP-seq, Hi-C, etc.)

Failed (attempted):
  fail_download     — download failed (curl, timeout, missing files)
  fail_no_r2        — no R2 files for droplet protocol (single-end data)
  fail_protocol_detect — protocol detection failed on FASTQ inspection
  fail_simpleaf_permit — alevin-fry permit-list failure (exit 25856)
  fail_simpleaf_map — piscem mapping failure
  fail_simpleaf_timeout — quantification timed out
  fail_simpleaf_other — other simpleaf/alevin-fry error
  fail_low_mapping  — mapping rate below threshold
  fail_qc_low_genes — too few genes per cell
  fail_qc_few_cells — too few cells detected
  fail_qc_other     — other QC failure
  fail_other        — uncategorized pipeline error

Pending:
  pending           — in catalog, not yet attempted
  retry_skipped     — previously skipped by pipeline, worth retrying
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────

PROJECT = Path("/mnt/projects/debruinz_project/cellarium")
CATALOG_PARQUET = PROJECT / "catalog" / "geo_single_cell_catalog.parquet"
PIPELINE_DIR = PROJECT / "pipeline"
DATASET_DIR = PROJECT / "dataset"
OUTPUT_DIR = PROJECT / "catalog"

BATCH_DIRS = [
    PIPELINE_DIR / "batches_v5_production",
    PIPELINE_DIR / "batches_retry_v3",
    PIPELINE_DIR / "batches_retry_v4",
]

# ── Status classification ───────────────────────────────────────────────

# Protocol → target_assay mapping
DROPLET_SCRNA_PROTOCOLS = {
    "10xv3", "10xv2", "10xv3_5prime", "10xv4",
    "dropseq", "indrop", "parse", "bd_rhapsody",
    "seqwell", "dnbelab", "ddseq", "surecell", "splitseq",
    "scirna",  # sci-RNA-seq3 (combinatorial indexing, droplet-like)
    "scopeseq",  # Scope-seq (droplet-like, has chemistry but 0 samples currently)
}
CITE_PROTOCOLS = {"citeseq"}
ATAC_PROTOCOLS = {"10x_atac", "scatac", "atac-seq", "sciatac"}
SPATIAL_PROTOCOLS = {"visium", "slideseq", "slide-seq", "merfish", "seqfish", "stereo-seq"}
MULTIOME_PROTOCOLS = {"10x_multiome", "multiome_atac"}
PLATE_BASED_PROTOCOLS = {
    "smartseq2", "smartseq3", "smart-seq", "smartseq",
    "marsseq", "mars-seq", "mars_seq",
    "icell8", "cel-seq", "celseq", "celseq2",
    "strt-seq", "strt_seq", "strtseq",
    "quartzseq", "quartz-seq",
    "plate_based", "fluidigm_c1", "microwell",
}
NON_RNA_PROTOCOLS = {
    "methylation", "chipseq", "chip-seq", "hi_c", "hi-c", "hic",
    "bulkrna", "bulk_rna", "bulk-rna",
    "dnase-seq", "dnaseseq", "gro-seq", "groseq",
    "cut&tag", "cuttag", "cut&run", "cutrun",
    "ribo-seq", "riboseq", "clip-seq", "clipseq",
    "mirna_seq", "rip_seq", "dnase_seq", "mnase_seq",
}

# VDJ/BCR/TCR detection keywords (case-insensitive, checked against title + summary)
VDJ_KEYWORDS = [
    r"\bvdj\b", r"\bbcr\b", r"\btcr\b",
    r"\bv\(d\)j\b", r"\bimmune repertoire\b",
    r"\bb[- ]?cell receptor\b", r"\bt[- ]?cell receptor\b",
    r"\bibseq\b", r"\bairr\b",
]

# Actionability mapping for failure categories
ACTIONABLE_FAILURES = {
    "fail_download": True,          # retry with different mirror
    "fail_no_r2": False,            # genuinely single-end data
    "fail_protocol_detect": True,   # improve detection rules
    "fail_simpleaf_permit": False,  # bad data / incompatible chemistry
    "fail_simpleaf_map": True,      # might be reference issue
    "fail_simpleaf_timeout": True,  # increase time limit
    "fail_simpleaf_other": True,    # investigate and fix
    "fail_low_mapping": False,      # wrong organism or corrupt data
    "fail_qc_low_genes": False,     # bad quality data
    "fail_qc_few_cells": False,     # bad quality data
    "fail_qc_other": False,         # bad quality data
    "fail_other": True,             # investigate
}


def load_master_catalog() -> pd.DataFrame:
    """Load the master GEO single-cell catalog."""
    logger.info(f"Loading master catalog from {CATALOG_PARQUET}")
    df = pd.read_parquet(CATALOG_PARQUET)
    logger.info(f"  {len(df):,} samples, {len(df.columns)} columns")
    return df


def load_batch_results() -> pd.DataFrame:
    """Load and concatenate all pipeline batch results CSVs."""
    frames = []
    for batch_dir in BATCH_DIRS:
        if not batch_dir.exists():
            continue
        csv_files = sorted(batch_dir.glob("*_results.csv"))
        logger.info(f"  {batch_dir.name}: {len(csv_files)} result files")
        for f in csv_files:
            try:
                df = pd.read_csv(f, dtype=str)
                df["_batch_dir"] = batch_dir.name
                df["_batch_file"] = f.name
                frames.append(df)
            except Exception as e:
                logger.warning(f"  Skipping {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    results = pd.concat(frames, ignore_index=True)
    logger.info(f"  Total batch results: {len(results):,} rows")

    # Deduplicate: keep the latest result per gsm_id.
    # Priority: retry_v4 > retry_v3 > v5_production
    priority = {"batches_retry_v4": 3, "batches_retry_v3": 2, "batches_v5_production": 1}
    results["_priority"] = results["_batch_dir"].map(priority).fillna(0).astype(int)
    results = results.sort_values("_priority", ascending=False).drop_duplicates(
        subset=["gsm_id"], keep="first"
    )
    results = results.drop(columns=["_priority"])
    logger.info(f"  After dedup: {len(results):,} unique GSMs")
    return results


def classify_pipeline_status(row: pd.Series) -> tuple[str, str, str]:
    """Classify a pipeline result row into (status, failure_category, failure_detail).

    Returns:
        (processing_status, failure_category, failure_detail)
    """
    status = str(row.get("status", "")).lower().strip()
    error = str(row.get("error", "")).strip() if pd.notna(row.get("error")) else ""
    qc_status = str(row.get("qc_status", "")).lower().strip() if pd.notna(row.get("qc_status")) else ""

    # Successful processing
    if status == "success":
        if qc_status == "qc_warn":
            return "done_qc_warn", "", ""
        return "done", "", ""

    # Skipped by pipeline
    if status == "skipped":
        error_lower = error.lower()
        # Pipeline found this sample was already processed in a prior run
        if "already processed" in error_lower:
            if "qc_pass" in error_lower:
                return "done", "", ""
            if "qc_warn" in error_lower:
                return "done_qc_warn", "", ""
            if "qc_fail" in error_lower:
                return "fail_qc_other", "fail_qc_other", error
            # "Already processed" but unknown QC — treat as done
            return "done", "", ""
        if "smart" in error_lower or "plate" in error_lower:
            return "skip_plate_based", "skip_plate_based", error
        if "spatial" in error_lower or "visium" in error_lower:
            return "skip_spatial", "skip_spatial", error
        if "atac" in error_lower:
            return "skip_atac", "skip_atac", error
        if "protocol" in error_lower:
            return "retry_skipped", "retry_skipped", error
        return "retry_skipped", "retry_skipped", error

    # QC warnings treated as done with flag
    if status == "qc_warn":
        return "done_qc_warn", "", error

    # Failed processing — classify the failure
    if status == "failed":
        error_lower = error.lower()

        # Download failures
        if "download" in error_lower or "curl" in error_lower or "fasterq" in error_lower:
            return "fail_download", "fail_download", error

        # No R2 files (single-end data)
        if "no r2" in error_lower or "no_r2" in error_lower:
            return "fail_no_r2", "fail_no_r2", error

        # Protocol detection failure
        if "protocol" in error_lower and ("detect" in error_lower or "unknown" in error_lower):
            return "fail_protocol_detect", "fail_protocol_detect", error

        # Simpleaf/alevin-fry failures
        if "simpleaf" in error_lower or "alevin" in error_lower or "piscem" in error_lower:
            if "permit" in error_lower or "25856" in error_lower or "exit code 101" in error_lower:
                return "fail_simpleaf_permit", "fail_simpleaf_permit", error
            if "piscem" in error_lower or "mapping" in error_lower:
                return "fail_simpleaf_map", "fail_simpleaf_map", error
            if "timeout" in error_lower or "timed out" in error_lower:
                return "fail_simpleaf_timeout", "fail_simpleaf_timeout", error
            return "fail_simpleaf_other", "fail_simpleaf_other", error

        # Timeout (generic)
        if "timeout" in error_lower or "timed out" in error_lower:
            return "fail_simpleaf_timeout", "fail_simpleaf_timeout", error

        # Low mapping rate
        if "mapping" in error_lower and ("low" in error_lower or "rate" in error_lower):
            return "fail_low_mapping", "fail_low_mapping", error

        # QC failures
        if "qc" in error_lower or "genes" in error_lower or "cells" in error_lower:
            if "genes" in error_lower:
                return "fail_qc_low_genes", "fail_qc_low_genes", error
            if "cells" in error_lower or "few" in error_lower:
                return "fail_qc_few_cells", "fail_qc_few_cells", error
            return "fail_qc_other", "fail_qc_other", error

        return "fail_other", "fail_other", error

    # Fallback
    return "fail_other", "fail_other", f"unknown_status={status}: {error}"


def classify_catalog_protocol(protocol: str) -> str:
    """Pre-screen a catalog entry by its inferred protocol.

    Returns a skip_* status if the protocol is incompatible with the pipeline,
    or 'pending' if it could be processed.
    """
    if pd.isna(protocol):
        return "pending"
    p = str(protocol).lower().strip()

    if p in PLATE_BASED_PROTOCOLS or any(s in p for s in ("smartseq", "smart-seq", "smart_seq")):
        return "skip_plate_based"
    if p in SPATIAL_PROTOCOLS or any(s in p for s in ("visium", "slideseq", "stereo")):
        return "skip_spatial"
    if p in ATAC_PROTOCOLS or p == "scatac":
        return "skip_atac"
    if p in MULTIOME_PROTOCOLS:
        # Multiome GEX component is processable; ATAC component is not
        return "pending"
    if p in NON_RNA_PROTOCOLS or any(s in p for s in ("methylation", "chipseq", "chip-seq", "hi_c", "hi-c")):
        return "skip_non_rna"
    return "pending"


def classify_unknown_bulk(row: pd.Series) -> str:
    """Heuristic classification for samples with unknown/ambiguous protocol.

    Uses library_source, library_layout, and read_count to distinguish
    likely Smart-seq2 / plate-based from likely droplet.

    Returns a revised processing_status:
      - 'skip_likely_plate' if the sample is almost certainly plate-based
      - 'skip_non_rna' for non-RNA strategies
      - 'pending' otherwise (could be droplet or worth trying)
    """
    protocol = str(row.get("protocol_inferred", "")).lower().strip()
    lib_source = str(row.get("library_source", "")).upper().strip()
    lib_layout = str(row.get("library_layout", "")).upper().strip()
    lib_strategy = str(row.get("library_strategy", "")).upper().strip()

    # Non-RNA strategies should never have been pending
    if lib_strategy in ("HI-C", "CHIP-SEQ", "BISULFITE-SEQ", "MIRNA-SEQ",
                         "RIP-SEQ", "DNASE-HYPERSENSITIVITY", "MNASE-SEQ",
                         "MEDIP-SEQ", "MBD-SEQ", "WGS", "CHIA-PET"):
        return "skip_non_rna"

    # Only apply bulk/smartseq heuristics to unknown-ish protocols
    if protocol not in ("unknown", "snrna_unknown"):
        return "pending"

    # Genomic library source is never droplet single-cell RNA
    if lib_source == "GENOMIC":
        return "skip_non_rna"

    # library_source == "TRANSCRIPTOMIC" (not SINGLE CELL) + SINGLE layout →
    # very strong plate-based signal
    if lib_source == "TRANSCRIPTOMIC" and lib_layout == "SINGLE":
        return "skip_likely_plate"

    # library_source == "TRANSCRIPTOMIC" + PAIRED →
    # check avg read length: droplet ≈60bp (28+90), Smart-seq2 ≈200bp (2×100-150)
    if lib_source == "TRANSCRIPTOMIC" and lib_layout == "PAIRED":
        reads = pd.to_numeric(row.get("read_count"), errors="coerce")
        bases = pd.to_numeric(row.get("base_count"), errors="coerce")
        if pd.notna(reads) and pd.notna(bases) and reads > 0:
            avg_rl = bases / reads
            if avg_rl > 120:
                # Long reads → plate-based (Smart-seq2 paired, or full-length)
                return "skip_likely_plate"
            if avg_rl < 80:
                # Short reads typical of droplet (28+90bp) — keep as pending
                return "pending"
        # Unknown read length but TRANSCRIPTOMIC (not SC) + low reads → likely plate
        if pd.notna(reads) and reads < 5_000_000:
            return "skip_likely_plate"

    return "pending"


def scan_migrated_manifests() -> dict[str, dict]:
    """Scan dataset/ for migrated manifest.json files."""
    logger.info("Scanning migrated manifests in dataset/...")
    manifests = {}
    manifest_files = list(DATASET_DIR.glob("GSE*/GSM*/manifest.json"))
    logger.info(f"  Found {len(manifest_files):,} manifest files")

    for mf in manifest_files:
        try:
            gsm_id = mf.parent.name
            with open(mf) as f:
                data = json.load(f)
            manifests[gsm_id] = {
                "migrated": True,
                "migration_pipeline_version": data.get("pipeline_version", ""),
                "migration_n_cells": data.get("rna", {}).get("n_cells"),
                "migration_n_genes": data.get("rna", {}).get("n_genes"),
                "migration_mapping_rate": data.get("rna", {}).get("mapping_rate"),
                "migration_qc_status": data.get("rna", {}).get("qc_status", ""),
            }
        except Exception:
            continue

    logger.info(f"  Loaded {len(manifests):,} migrated manifests")
    return manifests


def scan_pipeline_quant_output() -> set[str]:
    """Scan pipeline/quant for GSMs that have counts.1pz or counts.spz.

    Returns the set of gsm_ids that have a compressed count matrix,
    regardless of what batch CSVs report.  This is the ground truth
    for "pipeline produced output".
    """
    logger.info("Scanning pipeline/quant for counts.1pz / counts.spz files...")
    quant_dir = PIPELINE_DIR / "quant"
    spz_gsms = set()
    for pz_file in quant_dir.glob("GSE*/GSM*/counts.1pz"):
        spz_gsms.add(pz_file.parent.name)
    for spz_file in quant_dir.glob("GSE*/GSM*/counts.spz"):
        spz_gsms.add(spz_file.parent.name)
    logger.info(f"  Found {len(spz_gsms):,} GSMs with count matrices")
    return spz_gsms


def add_prescreen_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-compute screening flags based on catalog metadata."""
    import re

    # screen_no_fastq: no FASTQ URLs available (check ENA URLs or SRA accessions)
    has_ena = False
    for col in ["ena_fastq_r1", "fastq_ftp"]:
        if col in df.columns:
            has_ena = True
            df["screen_no_fastq"] = df[col].isna() | (df[col] == "")
            break
    if not has_ena:
        df["screen_no_fastq"] = True

    # screen_no_srx: no SRA experiment linked
    if "srx_accession" in df.columns:
        df["screen_no_srx"] = df["srx_accession"].isna() | (df["srx_accession"] == "")
    else:
        df["screen_no_srx"] = True

    # screen_single_end: paired-end expected for 10x but layout says single
    if "library_layout" in df.columns:
        is_10x = df["protocol_inferred"].str.lower().str.contains("10x", na=False)
        is_single = df["library_layout"].str.lower().str.contains("single", na=False)
        df["screen_single_end_10x"] = is_10x & is_single
    else:
        df["screen_single_end_10x"] = False

    # screen_no_r2: no R2 file on ENA (for droplet protocols that need it)
    if "ena_fastq_r2" in df.columns:
        droplet_protocols = df["protocol_inferred"].str.lower().str.contains(
            "10x|dropseq|indrop|celseq|ddseq|rhapsody", na=False, regex=True
        )
        no_r2 = df["ena_fastq_r2"].isna() | (df["ena_fastq_r2"] == "")
        df["screen_no_r2"] = droplet_protocols & no_r2
    else:
        df["screen_no_r2"] = False

    # screen_non_illumina: non-Illumina platform (except DNBSEQ/BGISEQ which work)
    if "instrument_platform" in df.columns:
        platform_upper = df["instrument_platform"].str.upper()
        df["screen_non_illumina"] = (
            ~platform_upper.str.contains("ILLUMINA", na=False)
            & ~platform_upper.str.contains("DNBSEQ", na=False)
            & ~platform_upper.str.contains("BGISEQ", na=False)
            & df["instrument_platform"].notna()
        )
    else:
        df["screen_non_illumina"] = False

    # screen_low_reads: very low read count (< 100K)
    if "read_count" in df.columns:
        reads = pd.to_numeric(df["read_count"], errors="coerce")
        df["screen_low_reads"] = reads.notna() & (reads < 100_000)
    else:
        df["screen_low_reads"] = False

    # screen_unsupported_organism: organisms without reference genomes
    # Organism field may be semicolon-delimited (e.g. "Homo sapiens; Mus musculus")
    # A sample passes if ANY listed species is in our supported set
    # Pull from species config to stay in sync
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scgeo.config.species import ORGANISM_TO_TAXON, SPECIES_REF
        supported_organisms = set(ORGANISM_TO_TAXON.keys())
        # Also add all canonical names from SPECIES_REF (skip=True excluded)
        for tid, info in SPECIES_REF.items():
            if not info.get("skip", False):
                supported_organisms.add(info["name"].lower())
    except ImportError:
        logger.warning("Cannot import species config, using fallback supported set")
        supported_organisms = {
            "homo sapiens", "mus musculus", "rattus norvegicus",
            "danio rerio", "drosophila melanogaster", "caenorhabditis elegans",
            "macaca fascicularis", "macaca mulatta", "callithrix jacchus",
            "pan troglodytes",
            "gallus gallus", "sus scrofa", "sus scrofa domesticus",
            "canis lupus familiaris", "bos taurus", "ovis aries",
            "equus caballus", "felis catus",
            "oryctolagus cuniculus", "mesocricetus auratus",
            "cavia porcellus", "monodelphis domestica",
            "xenopus tropicalis", "xenopus laevis",
            "nothobranchius furzeri", "oryzias latipes",
            "ciona robusta", "ciona intestinalis",
            "schmidtea mediterranea", "nematostella vectensis",
            "apis mellifera", "aplysia californica",
        }
    if "organism" in df.columns:
        def _has_supported_organism(val):
            if pd.isna(val) or val == "":
                return False
            return any(sp.strip() in supported_organisms for sp in str(val).lower().split(";"))
        df["screen_unsupported_organism"] = ~df["organism"].apply(_has_supported_organism)
    else:
        df["screen_unsupported_organism"] = False

    # screen_vdj: VDJ/BCR/TCR immune repertoire library (not gene expression)
    # Only flag samples with AMBIGUOUS protocols (10x_suspect, unknown_sc, snrna_unknown)
    # Known GEX protocols (10xv2, 10xv3, dropseq, etc.) from immune studies are real GEX —
    # series-level keywords like "TCR" or "VDJ" just mean the study also has VDJ libraries
    vdj_pattern = "|".join(VDJ_KEYWORDS)
    ambiguous_protocols = {"10x_suspect", "unknown_sc", "snrna_unknown"}
    text_cols = []
    for col in ["series_title", "summary"]:
        if col in df.columns:
            text_cols.append(df[col].fillna("").astype(str))
    if text_cols:
        combined_text = text_cols[0]
        for tc in text_cols[1:]:
            combined_text = combined_text + " " + tc
        has_vdj_text = combined_text.str.contains(vdj_pattern, case=False, regex=True, na=False)
        is_ambiguous = df["protocol_inferred"].fillna("").str.lower().str.strip().isin(ambiguous_protocols)
        df["screen_vdj"] = has_vdj_text & is_ambiguous
    else:
        df["screen_vdj"] = False

    # screen_spatial_suspect: spatial transcriptomics keywords + strategy=OTHER
    # Catches Visium/SlideSeq samples misclassified as snRNA_unknown
    spatial_pattern = r"\bvisium\b|\bslide.?seq\b|\bstereo.?seq\b|\bmerfish\b|\bseqfish\b|\bspatial transcriptom|\bspatial atlas\b|\bspatial profil"
    if text_cols:
        df["screen_spatial_suspect"] = (
            combined_text.str.contains(spatial_pattern, case=False, regex=True, na=False)
            & (df["library_strategy"].fillna("").str.upper() == "OTHER")
        )
    else:
        df["screen_spatial_suspect"] = False

    return df


def assign_target_assay(df: pd.DataFrame) -> pd.DataFrame:
    """Assign target_assay column based on protocol and status."""
    logger.info("Assigning target assay categories...")

    df["target_assay"] = "ambiguous"  # default

    proto = df["protocol_inferred"].fillna("").str.lower().str.strip()

    # Droplet scRNA-seq
    df.loc[proto.isin(DROPLET_SCRNA_PROTOCOLS), "target_assay"] = "scrna"
    # 10x_suspect: likely 10x but low confidence — still scrna target
    df.loc[proto == "10x_suspect", "target_assay"] = "scrna"
    # unknown_sc: tagged TRANSCRIPTOMIC SINGLE CELL but unknown method
    df.loc[proto == "unknown_sc", "target_assay"] = "scrna"
    # snRNA_unknown: mentioned snRNA in metadata
    df.loc[proto == "snrna_unknown", "target_assay"] = "scrna"

    # CITE-seq
    df.loc[proto.isin(CITE_PROTOCOLS), "target_assay"] = "cite"

    # ATAC
    df.loc[proto.isin(ATAC_PROTOCOLS), "target_assay"] = "atac"

    # Spatial
    df.loc[proto.isin(SPATIAL_PROTOCOLS), "target_assay"] = "spatial"

    # Multiome (GEX component is scrna-processable)
    df.loc[proto.isin(MULTIOME_PROTOCOLS), "target_assay"] = "multiome"

    # Plate-based scRNA
    df.loc[proto.isin(PLATE_BASED_PROTOCOLS), "target_assay"] = "plate_scrna"
    # Also set for fuzzy matches
    smartseq_mask = proto.str.contains("smartseq|smart-seq|smart_seq", regex=True, na=False)
    df.loc[smartseq_mask, "target_assay"] = "plate_scrna"

    # Non-RNA
    df.loc[proto.isin(NON_RNA_PROTOCOLS), "target_assay"] = "non_rna"
    non_rna_mask = proto.str.contains("methylation|chipseq|chip-seq|hi_c|hi-c", regex=True, na=False)
    df.loc[non_rna_mask, "target_assay"] = "non_rna"

    # Override: anything classified as skip_non_rna or skip_likely_plate
    df.loc[df["processing_status"] == "skip_non_rna", "target_assay"] = "non_rna"
    df.loc[df["processing_status"].isin(["skip_plate_based", "skip_likely_plate"]), "target_assay"] = "plate_scrna"
    df.loc[df["processing_status"] == "skip_atac", "target_assay"] = "atac"
    df.loc[df["processing_status"] == "skip_spatial", "target_assay"] = "spatial"
    df.loc[df["processing_status"] == "skip_vdj", "target_assay"] = "vdj"

    # Count
    assay_counts = df["target_assay"].value_counts()
    for assay, count in assay_counts.items():
        logger.info(f"  {assay}: {count:,}")

    return df


def build_processing_catalog(output_path: Path | None = None) -> pd.DataFrame:
    """Build the comprehensive processing catalog."""

    # 1. Load master catalog
    catalog = load_master_catalog()

    # 2. Load all pipeline batch results
    logger.info("Loading batch results...")
    results = load_batch_results()

    # 3. Classify pipeline results
    logger.info("Classifying pipeline results...")
    if len(results) > 0:
        logger.info(f"  Result columns: {list(results.columns)}")
        classified = results.apply(classify_pipeline_status, axis=1)
        results["processing_status"] = classified.apply(lambda x: x[0])
        results["failure_category"] = classified.apply(lambda x: x[1])
        results["failure_detail"] = classified.apply(lambda x: x[2])

        # Extract useful metrics from results into a compact frame
        result_cols = {
            "gsm_id": "gsm_id",
            "processing_status": "processing_status",
            "failure_category": "failure_category",
            "failure_detail": "failure_detail",
            "status": "pipeline_status_raw",
            "error": "pipeline_error",
            "mapping_rate": "pipeline_mapping_rate",
            "total_time_s": "pipeline_total_time_s",
            "protocol": "pipeline_protocol_detected",
            "mode": "pipeline_mode",
            "_batch_dir": "pipeline_batch_source",
            "_batch_file": "pipeline_batch_file",
        }
        # Only include columns that exist
        available_cols = {k: v for k, v in result_cols.items() if k in results.columns}
        results_slim = results[list(available_cols.keys())].rename(columns=available_cols)
        logger.info(f"  Results slim columns: {list(results_slim.columns)}")
    else:
        results_slim = pd.DataFrame(columns=["gsm_id", "processing_status", "failure_category", "failure_detail"])

    # 4. Scan migrated manifests
    migrated = scan_migrated_manifests()
    migration_df = pd.DataFrame.from_dict(migrated, orient="index")
    migration_df.index.name = "gsm_id"
    migration_df = migration_df.reset_index()

    # 5. Merge onto master catalog
    # Drop stale processing_status from catalog if present (we replace it)
    stale_cols = [c for c in catalog.columns if c in results_slim.columns and c != "gsm_id"]
    if stale_cols:
        logger.info(f"  Dropping stale columns from catalog: {stale_cols}")
        catalog = catalog.drop(columns=stale_cols)
    logger.info("Merging pipeline results onto catalog...")
    merged = catalog.merge(results_slim, on="gsm_id", how="left")

    logger.info("Merging migration status...")
    merged = merged.merge(migration_df, on="gsm_id", how="left")
    merged["migrated"] = merged["migrated"].fillna(False).astype(bool)

    # 6. Fill in status for samples not in pipeline results
    logger.info("Assigning status to unprocessed samples...")
    mask_no_status = merged["processing_status"].isna()
    if mask_no_status.any():
        protocol_status = merged.loc[mask_no_status, "protocol_inferred"].apply(
            classify_catalog_protocol
        )
        merged.loc[mask_no_status, "processing_status"] = protocol_status

    # 6b. Apply bulk/Smart-seq heuristic to remaining "pending" unknowns
    logger.info("Applying Smart-seq / bulk heuristic to unknowns...")
    mask_pending = merged["processing_status"] == "pending"
    if mask_pending.any():
        heuristic_status = merged.loc[mask_pending].apply(classify_unknown_bulk, axis=1)
        # Only overwrite samples that the heuristic reclassified
        reclassified = heuristic_status != "pending"
        if reclassified.any():
            merged.loc[heuristic_status[reclassified].index, "processing_status"] = heuristic_status[reclassified]
            logger.info(f"  Reclassified {reclassified.sum():,} samples via heuristic")

    # 6c. Rescue droplet protocols wrongly skipped as plate-based by old pipeline
    # The old pipeline's protocol detection was less accurate; the catalog's
    # protocol_inferred (from metadata analysis) is more reliable for known
    # droplet protocols with high/medium confidence.
    logger.info("Rescuing wrongly-skipped droplet protocols...")
    rescue_protocols = DROPLET_SCRNA_PROTOCOLS | CITE_PROTOCOLS | MULTIOME_PROTOCOLS
    rescue_proto_lower = {p.lower() for p in rescue_protocols}
    rescue_mask = (
        (merged["processing_status"] == "skip_plate_based")
        & merged["protocol_inferred"].fillna("").str.lower().isin(rescue_proto_lower)
        & merged["protocol_confidence"].isin(["high", "medium"])
    )
    if rescue_mask.any():
        merged.loc[rescue_mask, "processing_status"] = "retry_skipped"
        logger.info(f"  Rescued {rescue_mask.sum():,} droplet samples from skip_plate_based")

    # 7. Upgrade "done" status based on migration
    # If a sample was successfully processed AND migrated, it's truly done
    mask_done = merged["processing_status"].isin(["done", "done_qc_warn"])
    mask_migrated = merged["migrated"]
    merged["migration_status"] = "not_migrated"
    merged.loc[mask_migrated, "migration_status"] = "migrated"
    merged.loc[mask_done & ~mask_migrated, "migration_status"] = "pending_migration"

    # 8. Add actionability for failures
    merged["actionable"] = merged["failure_category"].map(ACTIONABLE_FAILURES)

    # 9. Add pre-screening flags
    logger.info("Adding pre-screening flags...")
    merged = add_prescreen_flags(merged)

    # 9b. Apply VDJ screening — mark VDJ libraries among 10x_suspect pending
    logger.info("Screening for VDJ/BCR/TCR libraries...")
    vdj_mask = (
        merged["screen_vdj"]
        & merged["processing_status"].isin(["pending", "retry_skipped"])
        & merged["protocol_inferred"].str.lower().str.contains("10x|suspect", na=False, regex=True)
    )
    if vdj_mask.any():
        merged.loc[vdj_mask, "processing_status"] = "skip_vdj"
        logger.info(f"  Marked {vdj_mask.sum():,} samples as skip_vdj")

    # 9c. Apply spatial screening — catch Visium/spatial mistakenly tagged as snRNA_unknown
    logger.info("Screening for spatial transcriptomics in snRNA_unknown...")
    spatial_mask = (
        merged["screen_spatial_suspect"]
        & merged["processing_status"].isin(["pending", "retry_skipped"])
        & (merged["protocol_inferred"].str.lower() == "snrna_unknown")
    )
    if spatial_mask.any():
        merged.loc[spatial_mask, "processing_status"] = "skip_spatial"
        logger.info(f"  Marked {spatial_mask.sum():,} snRNA_unknown samples as skip_spatial")

    # 9d. Reconcile with on-disk output — ground-truth override
    # Some samples have count matrices from an earlier pipeline run but their batch
    # CSV entry was overwritten by a later "skipped" or "failed" result.
    # If counts.1pz/counts.spz exists, the pipeline DID produce output → mark as done.
    logger.info("Reconciling with on-disk pipeline output...")
    spz_gsms = scan_pipeline_quant_output()
    has_spz = merged["gsm_id"].isin(spz_gsms)
    non_done_with_spz = has_spz & ~merged["processing_status"].isin(["done", "done_qc_warn"])
    if non_done_with_spz.any():
        old_statuses = merged.loc[non_done_with_spz, "processing_status"].value_counts()
        logger.info(f"  Overriding {non_done_with_spz.sum():,} non-done samples that have count matrices:")
        for status, count in old_statuses.items():
            logger.info(f"    {status}: {count:,}")
        merged.loc[non_done_with_spz, "processing_status"] = "done"
        merged.loc[non_done_with_spz, "failure_category"] = ""
        merged.loc[non_done_with_spz, "failure_detail"] = ""

    # 10. Compute composite screening flag
    screen_cols = [c for c in merged.columns if c.startswith("screen_") and c != "screen_any_flag"]
    merged["screen_any_flag"] = merged[screen_cols].any(axis=1)

    # 11. Add avg_read_length for triage
    if "read_count" in merged.columns and "base_count" in merged.columns:
        reads = pd.to_numeric(merged["read_count"], errors="coerce")
        bases = pd.to_numeric(merged["base_count"], errors="coerce")
        merged["avg_read_length"] = (bases / reads).where(reads > 0)

    # 12. Assign target_assay
    merged = assign_target_assay(merged)

    # 13. Compute production_ready: high-confidence droplet scRNA for next batch
    # Initial production focuses on human only; other organisms added later
    def _is_human(val):
        if pd.isna(val) or val == "":
            return False
        return any(sp.strip() == "homo sapiens" for sp in str(val).lower().split(";"))
    is_human = merged["organism"].apply(_is_human) if "organism" in merged.columns else pd.Series(False, index=merged.index)
    merged["production_ready"] = (
        merged["processing_status"].isin(["pending", "retry_skipped"])
        & (merged["target_assay"] == "scrna")
        & ~merged["screen_any_flag"]
        & is_human
    )
    n_ready = merged["production_ready"].sum()
    logger.info(f"Production-ready samples: {n_ready:,}")

    # 14. Sort by GSE/GSM for clean output
    merged = merged.sort_values(["gse_id", "gsm_id"]).reset_index(drop=True)

    # 15. Save
    if output_path is None:
        output_path = OUTPUT_DIR / "processing_catalog.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info(f"Saved processing catalog: {output_path}")
    logger.info(f"  {len(merged):,} rows, {len(merged.columns)} columns")

    # 16. Print summary
    print_summary(merged)

    return merged


def print_summary(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the processing catalog."""
    print("\n" + "=" * 70)
    print("PROCESSING CATALOG SUMMARY")
    print("=" * 70)

    print(f"\nTotal samples: {len(df):,}")
    print(f"Unique GSEs: {df['gse_id'].nunique():,}")

    # ── Target Assay Breakdown ──
    print("\n--- Target Assay Categories ---")
    assay_counts = df["target_assay"].value_counts()
    for assay, count in assay_counts.items():
        pct = 100 * count / len(df)
        print(f"  {assay:20s} {count:>8,}  ({pct:5.1f}%)")

    # ── Processing Status ──
    print("\n--- Processing Status ---")
    status_counts = df["processing_status"].value_counts()
    for status, count in status_counts.items():
        pct = 100 * count / len(df)
        actionable = ACTIONABLE_FAILURES.get(status, None)
        action_str = ""
        if actionable is True:
            action_str = " [ACTIONABLE]"
        elif actionable is False:
            action_str = " [non-actionable]"
        print(f"  {status:30s} {count:>8,}  ({pct:5.1f}%){action_str}")

    # ── Aggregated groups ──
    done_mask = df["processing_status"].str.startswith("done")
    skip_mask = df["processing_status"].str.startswith("skip")
    fail_mask = df["processing_status"].str.startswith("fail")
    pending_mask = df["processing_status"].isin(["pending", "retry_skipped"])

    print(f"\n--- Aggregated ---")
    print(f"  Done:    {done_mask.sum():>8,}  ({100*done_mask.sum()/len(df):5.1f}%)")
    print(f"  Skipped: {skip_mask.sum():>8,}  ({100*skip_mask.sum()/len(df):5.1f}%)")
    print(f"  Failed:  {fail_mask.sum():>8,}  ({100*fail_mask.sum()/len(df):5.1f}%)")
    print(f"  Pending: {pending_mask.sum():>8,}  ({100*pending_mask.sum()/len(df):5.1f}%)")

    # ── Production Ready ──
    prod = df[df["production_ready"]]
    print(f"\n--- Production Ready (scrna, no flags) ---")
    print(f"  Total production-ready:  {len(prod):>8,}")
    if len(prod) > 0:
        print(f"  Unique GSEs:             {prod['gse_id'].nunique():>8,}")
        print(f"\n  By protocol:")
        proto_counts = prod["protocol_inferred"].value_counts()
        for proto, count in proto_counts.items():
            pct = 100 * count / len(prod)
            print(f"    {str(proto):25s} {count:>8,}  ({pct:5.1f}%)")
        print(f"\n  By organism (top 10):")
        org_counts = prod["organism"].value_counts().head(10)
        for org, count in org_counts.items():
            pct = 100 * count / len(prod)
            print(f"    {str(org):35s} {count:>8,}  ({pct:5.1f}%)")
        if "library_source" in prod.columns:
            print(f"\n  By library source:")
            src_counts = prod["library_source"].value_counts()
            for src, count in src_counts.items():
                pct = 100 * count / len(prod)
                print(f"    {str(src):35s} {count:>8,}  ({pct:5.1f}%)")

    # ── Pending but flagged ──
    pending_df = df[pending_mask]
    flagged = pending_df[pending_df["screen_any_flag"]]
    print(f"\n--- Pending but Flagged (not production-ready) ---")
    print(f"  Total flagged pending:   {len(flagged):>8,}")
    if len(flagged) > 0:
        screen_cols = [c for c in df.columns if c.startswith("screen_") and c != "screen_any_flag"]
        for col in sorted(screen_cols):
            n_flagged = flagged[col].sum()
            if n_flagged > 0:
                pct = 100 * n_flagged / len(flagged)
                print(f"    {col:30s} {n_flagged:>8,}  ({pct:5.1f}%)")

    # ── Migration status ──
    print(f"\n--- Migration Status ---")
    mig_counts = df["migration_status"].value_counts()
    for status, count in mig_counts.items():
        print(f"  {status:30s} {count:>8,}")

    # ── On-disk pipeline output ──
    spz_gsms = scan_pipeline_quant_output()
    in_catalog = df["gsm_id"].isin(spz_gsms)
    not_in_catalog = spz_gsms - set(df["gsm_id"])
    print(f"\n--- Pipeline Output (count matrices) ---")
    print(f"  GSMs with counts:           {len(spz_gsms):>8,}")
    print(f"  In catalog:                 {in_catalog.sum():>8,}")
    print(f"  Not in catalog:             {len(not_in_catalog):>8,}")
    done_with_spz = in_catalog & df["processing_status"].str.startswith("done")
    print(f"  Marked done in catalog:     {done_with_spz.sum():>8,}")

    # ── Failure Actionability ──
    actionable_df = df[df["actionable"] == True]
    noaction_df = df[df["actionable"] == False]
    print(f"\n--- Failure Actionability ---")
    print(f"  Actionable failures:     {len(actionable_df):>8,}")
    print(f"  Non-actionable failures: {len(noaction_df):>8,}")

    # ── Future targets (ATAC / Spatial / CITE) ──
    print(f"\n--- Future Processing Targets ---")
    for assay in ["atac", "spatial", "cite", "multiome"]:
        sub = df[df["target_assay"] == assay]
        if len(sub) > 0:
            print(f"  {assay:15s} {len(sub):>8,} samples, {sub['gse_id'].nunique():,} GSEs")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Build comprehensive processing catalog")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output parquet path (default: catalog/processing_catalog.parquet)"
    )
    args = parser.parse_args()

    build_processing_catalog(output_path=args.output)


if __name__ == "__main__":
    main()
