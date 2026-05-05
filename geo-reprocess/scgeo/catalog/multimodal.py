"""
Comprehensive GEO catalog builder — all 10x/droplet assays.

Discovers and catalogs ALL droplet-based single-cell assays:
- Standard scRNA-seq (10x Chromium, Drop-seq, inDrop, etc.)
- Multi-modal assays (Multiome, CITE-seq, Visium, Perturb-seq)

Key design decisions:
- Combines broad scRNA-seq discovery with targeted multi-modal queries
- Preserves per-SRR metadata (library_name, fastq_bytes) instead of aggregating
- Extracts per-sample supplementary file URLs from SOFT
- Classifies each SRR by library_type (rna/atac/adt/guide)
- Classifies each row by assay_class for processing prioritization
- Builds cross-modality linkage tables for multi-omic experiments
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

import pandas as pd

from scgeo.catalog.discovery import esearch_paginated, fetch_series_titles
from scgeo.catalog.metadata import fetch_series_metadata, extract_gsm_samples
from scgeo.catalog.sra import fetch_sra_runinfo
from scgeo.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search queries — multi-modal assays that the droplet pipeline excluded
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-modal search strategy
#
# We split the query into modality-specific sub-queries, then union the UIDs
# in Python. Advantages:
#   1. Avoids NCBI query-length limits (~4 KB)
#   2. Gives per-modality provenance (which sub-query found each GSE)
#   3. Easier to maintain and expand individual categories
# ---------------------------------------------------------------------------

_GEO_SUFFIX = (
    ' AND "expression profiling by high throughput sequencing"[DataSet Type]'
    " AND gse[Entry Type]"
)

# ── Multiome / joint chromatin+transcriptome ──────────────────────────────
MULTIOME_QUERY = (
    # Official 10x product names
    '("10X Multiome"[Description] OR "snMultiome"[Description] OR '
    '"scMultiome"[Description] OR "multi-ome"[Description] OR '
    '"multiome ATAC"[Description] OR '
    '"Chromium Next GEM Multiome"[Description] OR '
    # Generic joint ATAC+RNA terms
    '"ATAC RNA"[Description] OR "joint ATAC RNA"[Description] OR '
    '"GEX ATAC"[Description] OR '
    '("joint profiling"[Description] AND "ATAC"[Description]) OR '
    '("single nucleus"[Description] AND "ATAC"[Description] AND "RNA"[Description]) OR '
    '("snATAC"[Description] AND "snRNA"[Description]) OR '
    '("Chromium"[Description] AND "ARC"[Description]) OR '
    '("10x"[Description] AND "multiome"[Description] AND "GEX"[Description]) OR '
    # Alternative joint chromatin+RNA methods
    '"SHARE-seq"[Description] OR "SNARE-seq"[Description] OR '
    '"SNAREseq"[Description] OR "sci-CAR"[Description] OR '
    '"DOGMA-seq"[Description] OR "ASAP-seq"[Description] OR '
    '"ASAPseq"[Description] OR "TEA-seq"[Description] OR '
    '"Paired-Tag"[Description] OR "CoTECH"[Description] OR '
    '"inCITE-seq"[Description] OR "scNMT-seq"[Description] OR '
    # Broad multi-omic terms (validated — titles confirm joint assays)
    '("multi-omic"[Description] AND "single cell"[Description]) OR '
    '("multiomic"[Description] AND "single cell"[Description]) OR '
    '"single cell multi-omics"[Description] OR '
    '"single-cell multi-omics"[Description] OR '
    '"single cell multiomics"[Description] OR '
    '"scMulti-omics"[Description])'
) + _GEO_SUFFIX

# ── CITE-seq / surface protein quantification ────────────────────────────
CITESEQ_QUERY = (
    '("CITE-seq"[Description] OR "CITEseq"[Description] OR '
    '"CITE seq"[Description] OR "TotalSeq"[Description] OR '
    '"Cellular Indexing of Transcriptomes and Epitopes"[Description] OR '
    '"antibody-derived tag"[Description] OR '
    # Alternative surface protein methods
    '"REAP-seq"[Description] OR "ECCITE-seq"[Description] OR '
    '"AbSeq"[Description] OR '
    # Feature barcoding (10x)
    '"feature barcoding"[Description] OR "Feature Barcode"[Description] OR '
    '("surface protein"[Description] AND "single cell"[Description]) OR '
    '("protein expression"[Description] AND "scRNA"[Description]) OR '
    '"antibody capture"[Description] OR '
    '("ADT"[Description] AND "single cell"[Description]) OR '
    '("cell surface"[Description] AND "scRNA"[Description]) OR '
    # Sample multiplexing / cell hashing (shared barcode-anchored workflow)
    '"cell hashing"[Description] OR '
    '("hashtag"[Description] AND "antibody"[Description]) OR '
    '("sample multiplexing"[Description] AND "HTO"[Description]))'
) + _GEO_SUFFIX

# ── Visium / spatial transcriptomics ──────────────────────────────────────
VISIUM_QUERY = (
    '("Visium"[Description] OR "10X Visium"[Description] OR '
    '"spatial transcriptomics"[Description] OR '
    '"spatial gene expression"[Description] OR '
    '"spatially resolved transcriptomics"[Description] OR '
    '("spatial RNA"[Description] AND "sequencing"[Description]) OR '
    # Alternative sequencing-based spatial methods
    '"Slide-seq"[Description] OR "Stereo-seq"[Description] OR '
    '"DBiT-seq"[Description] OR "Seq-Scope"[Description] OR '
    '"spatial barcoding"[Description])'
) + _GEO_SUFFIX

# ── Perturb-seq / CRISPR screens with single-cell readout ────────────────
PERTURBSEQ_QUERY = (
    '("Perturb-seq"[Description] OR "CROP-seq"[Description] OR '
    '"CROPseq"[Description] OR "Perturbseq"[Description] OR '
    '"pooled CRISPR screen"[Description] OR '
    '"CRISPR screen scRNA"[Description] OR '
    '"genetic perturbation screen"[Description] OR '
    # Expanded CRISPR terms
    '"TAP-seq"[Description] OR "Mosaic-seq"[Description] OR '
    '"scCRISPR"[Description] OR '
    '("CRISPRi"[Description] AND "single cell"[Description]) OR '
    '("CRISPRa"[Description] AND "single cell"[Description]) OR '
    '("CRISPR perturbation"[Description] AND "single cell"[Description]) OR '
    '("sgRNA"[Description] AND "capture"[Description]) OR '
    '("gRNA"[Description] AND "single cell"[Description]) OR '
    '("CRISPR screen"[Description] AND "single-cell RNA"[Description]) OR '
    '("CRISPR interference"[Description] AND "scRNA"[Description]))'
) + _GEO_SUFFIX

# ── Standalone scATAC-seq ─────────────────────────────────────────────────
SCATAC_QUERY = (
    '("scATAC-seq"[Description] OR "snATAC-seq"[Description] OR '
    '"single cell ATAC"[Description] OR "single nucleus ATAC"[Description] OR '
    '"10x Genomics Chromium Single Cell ATAC"[Description] OR '
    '"sciATAC"[Description] OR "sci-ATAC"[Description] OR '
    '"dscATAC"[Description] OR "s3-ATAC"[Description] OR '
    '"scCUT&Tag"[Description] OR "snCUT&Tag"[Description] OR '
    '"single-cell ATAC"[Description] OR '
    '"scATAC seq"[Description] OR "snATAC seq"[Description] OR '
    '("chromatin accessibility"[Description] AND "single cell"[Description]))'
) + _GEO_SUFFIX

# ── Droplet scRNA-seq (broad capture — all 10x-like assays) ──────────────
# This captures standard scRNA-seq that doesn't have multi-modal terms.
# Imported from the production discovery query for full GEO coverage.
from scgeo.catalog.discovery import DEFAULT_SINGLE_CELL_QUERY
SCRNA_QUERY = DEFAULT_SINGLE_CELL_QUERY

# Legacy combined query for backwards compatibility
MULTIMODAL_QUERY = MULTIOME_QUERY

# All sub-queries in execution order
MODALITY_QUERIES = {
    "scrna": SCRNA_QUERY,
    "multiome": MULTIOME_QUERY,
    "citeseq": CITESEQ_QUERY,
    "visium": VISIUM_QUERY,
    "perturbseq": PERTURBSEQ_QUERY,
    "scatac": SCATAC_QUERY,
}

# ---------------------------------------------------------------------------
# Library type classification regexes (applied per SRR)
# ---------------------------------------------------------------------------

_RE_ADT = re.compile(
    r"\badt\b|antibody.derived|totalseq|\bhto\b|hashtag|"
    r"antibody.capture|protein.capture|protein.tag|"
    r"cell.surface.protein|feature.barcod",
    re.IGNORECASE,
)
_RE_GUIDE = re.compile(
    r"\bguide\b|\bsgrna\b|\bcrispr\b|\bcapture\b.*\bguide|"
    r"\bgRNA\b|guide.capture|perturbation.capture",
    re.IGNORECASE,
)
_RE_ATAC = re.compile(
    r"\batac\b|chromatin.access",
    re.IGNORECASE,
)
_RE_GEX = re.compile(
    r"\bgex\b|gene.expression|\brna\b|transcriptom",
    re.IGNORECASE,
)

# Assay class inference from series/sample text
_RE_MULTIOME_CLASS = re.compile(
    r"multiome|multi.ome|10x.{0,50}atac.{0,50}gex|10x.{0,50}gex.{0,50}atac|"
    r"joint.{0,30}atac.{0,30}rna|snMultiome|scMultiome|"
    r"share.?seq|snare.?seq|sci.car|paired.tag|cotech|"
    r"dogma.?seq|tea.?seq|asap.?seq|"
    r"simultaneous.{0,30}atac.{0,30}rna|chromatin.{0,30}transcriptom|"
    r"single.{0,10}nucle.{0,30}atac.{0,30}rna|snatac.{0,10}snrna",
    re.IGNORECASE,
)
_RE_CITESEQ_CLASS = re.compile(
    r"cite.?seq|citeseq|totalseq|antibody.derived.tag|"
    r"cellular.indexing.of.transcriptomes|"
    r"reap.?seq|eccite.?seq|abseq|"
    r"feature.barcod|surface.protein|antibody.capture|"
    r"cell.hashing|sample.multiplex.{0,10}hto",
    re.IGNORECASE,
)
_RE_VISIUM_CLASS = re.compile(
    r"\bvisium\b|spatial.transcriptom|spatial.gene.express|"
    r"spatially.resolved.transcriptom|"
    r"slide.?seq|stereo.?seq|dbit.?seq|seq.scope|"
    r"spatial.barcod",
    re.IGNORECASE,
)
_RE_PERTURBSEQ_CLASS = re.compile(
    r"perturb.?seq|crop.?seq|crispr.screen.{0,30}scrna|"
    r"pooled.crispr|genetic.perturbation.screen|"
    r"tap.?seq|mosaic.?seq|sccrispr|"
    r"crispri.{0,30}single.cell|crispra.{0,30}single.cell|"
    r"crispr.perturbation|sgrna.{0,15}capture|grna.{0,15}single.cell|"
    r"crispr.interference.{0,15}scrna",
    re.IGNORECASE,
)
_RE_SCATAC_CLASS = re.compile(
    r"scatac|snatac|single.cell.atac|single.nucleus.atac|"
    r"sci.?atac|dscatac|s3.?atac|"
    r"sccut.?tag|sncut.?tag|"
    r"chromatin.access.{0,30}single.cell|"
    r"single.cell.{0,30}chromatin.access",
    re.IGNORECASE,
)
_RE_PROBE = re.compile(
    r"\bprobe\b|\bffpe\b|cytassist|fixed.tissue|probe.set",
    re.IGNORECASE,
)
_RE_NUCLEAR = re.compile(
    r"snuc|single.nuc|nuclei|snrna|nuclear|nucleus",
    re.IGNORECASE,
)
_RE_VISIUM_V5 = re.compile(r"\bv5\b|hd\b", re.IGNORECASE)
_RE_VISIUM_V4 = re.compile(r"\bv4\b", re.IGNORECASE)
_RE_TOTALSEQ_A = re.compile(r"totalseq.a\b", re.IGNORECASE)
_RE_TOTALSEQ_B = re.compile(r"totalseq.b\b", re.IGNORECASE)
_RE_TOTALSEQ_C = re.compile(r"totalseq.c\b", re.IGNORECASE)


def infer_library_type(row: pd.Series) -> str:
    """Classify a single SRR as rna / atac / adt / guide / unknown.

    Priority:
      1. library_strategy from ENA (ATAC-Seq → atac)
      2. library_name keywords (ENA)
      3. Sample-level SOFT fields (sample_library_strategy, sample_title, sample_source)
      4. Fallback: "rna" for RNA-Seq, "unknown" otherwise
    """
    lib_strategy = str(row.get("library_strategy", "")).lower().strip()
    lib_name = str(row.get("library_name", "")).lower().strip()

    # Priority 1: ENA library_strategy
    if lib_strategy == "atac-seq":
        return "atac"

    # Priority 2: library_name keywords (ENA)
    if _RE_ADT.search(lib_name):
        return "adt"
    if _RE_GUIDE.search(lib_name):
        return "guide"
    if _RE_ATAC.search(lib_name):
        return "atac"
    if _RE_GEX.search(lib_name):
        return "rna"

    # Priority 3: SOFT fields (sample_library_strategy, sample_title, sample_source)
    soft_strategy = str(row.get("sample_library_strategy", "")).lower().strip()
    if soft_strategy == "atac-seq":
        return "atac"

    # Check sample_title and sample_source for modality hints
    sample_text = " ".join([
        str(row.get("sample_title", "")),
        str(row.get("sample_source", "")),
        str(row.get("gsm_title", "")),
    ]).lower()

    if _RE_ADT.search(sample_text):
        return "adt"
    if _RE_GUIDE.search(sample_text):
        return "guide"
    if _RE_ATAC.search(sample_text):
        return "atac"
    if _RE_GEX.search(sample_text):
        return "rna"

    # Priority 4: fallback
    if lib_strategy == "rna-seq":
        return "rna"
    if soft_strategy == "rna-seq":
        return "rna"

    return "unknown"


def infer_assay_class(text: str) -> str:
    """Classify combined series+sample text into an assay class."""
    if _RE_MULTIOME_CLASS.search(text):
        return "multiome"
    if _RE_CITESEQ_CLASS.search(text):
        return "citeseq"
    if _RE_VISIUM_CLASS.search(text):
        return "visium"
    if _RE_PERTURBSEQ_CLASS.search(text):
        return "perturbseq"
    return "rna_only"


def infer_visium_chemistry(text: str) -> str:
    """Detect Visium chip version from metadata text."""
    if _RE_VISIUM_V5.search(text):
        return "visiumv5"
    if _RE_VISIUM_V4.search(text):
        return "visiumv4"
    return "visiumv1"


def infer_totalseq_panel(text: str) -> str:
    """Detect TotalSeq panel version from metadata text."""
    if _RE_TOTALSEQ_A.search(text):
        return "A"
    if _RE_TOTALSEQ_B.search(text):
        return "B"
    if _RE_TOTALSEQ_C.search(text):
        return "C"
    return "unknown"


def _safe_lower(val) -> str:
    """Safe string lowercase."""
    if pd.isna(val):
        return ""
    return str(val).lower()


def _combine_text(row: pd.Series, cols: list[str]) -> str:
    """Join multiple text columns into one searchable string."""
    parts = []
    for c in cols:
        v = row.get(c)
        if v is not None and not pd.isna(v):
            parts.append(str(v))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Enhanced SOFT parser — adds Sample_supplementary_file
# ---------------------------------------------------------------------------

def process_gse_soft_multimodal(gse_id: str, cache_dir: Path) -> Optional[dict]:
    """Download and parse SOFT with extended per-sample fields.

    Same as process_gse_soft() in soft.py but additionally extracts:
      - Sample_supplementary_file  (per-sample)
      - Sample_title               (per-sample)
      - Sample_description         (per-sample)
      - Series_title / Series_summary (series)
    """
    from scgeo.catalog.soft import download_soft, parse_soft_file, _flatten_field
    from scgeo.catalog.soft import extract_sra_from_relations, extract_srx_from_sample_relations

    path = download_soft(gse_id, cache_dir)
    if path is None:
        return None

    parsed = parse_soft_file(path)
    series = parsed["series"]
    samples = parsed["samples"]

    srp, bioproject = extract_sra_from_relations(
        series.get("Series_relation", "")
    )

    series_record = {
        "gse_id": gse_id,
        "series_title": _flatten_field(series.get("Series_title", "")),
        "summary": _flatten_field(series.get("Series_summary", "")),
        "overall_design": _flatten_field(series.get("Series_overall_design", "")),
        "contact_name": _flatten_field(series.get("Series_contact_name", "")),
        "contact_institute": _flatten_field(series.get("Series_contact_institute", "")),
        "sra_study": srp,
        "bioproject": bioproject,
        "supplementary_files": _flatten_field(series.get("Series_supplementary_file", "")),
        "last_update_date": _flatten_field(series.get("Series_last_update_date", "")),
    }

    sample_records = []
    for gsm_id, sdata in samples.items():
        srx = extract_srx_from_sample_relations(sdata.get("Sample_relation", ""))

        sample_records.append({
            "gse_id": gse_id,
            "gsm_id": gsm_id,
            "srx_accession": srx,
            "sample_title": _flatten_field(sdata.get("Sample_title", "")),
            "sample_organism": _flatten_field(sdata.get("Sample_organism_ch1", "")),
            "sample_source": _flatten_field(sdata.get("Sample_source_name_ch1", "")),
            "sample_characteristics": _flatten_field(
                sdata.get("Sample_characteristics_ch1", "")
            ),
            "sample_molecule": _flatten_field(sdata.get("Sample_molecule_ch1", "")),
            "sample_extract_protocol": _flatten_field(
                sdata.get("Sample_extract_protocol_ch1", "")
            ),
            "sample_library_strategy": _flatten_field(
                sdata.get("Sample_library_strategy", "")
            ),
            "sample_library_source": _flatten_field(
                sdata.get("Sample_library_source", "")
            ),
            "sample_data_processing": _flatten_field(
                sdata.get("Sample_data_processing", "")
            ),
            # NEW: per-sample supplementary files
            "sample_supplementary_files": _flatten_field(
                sdata.get("Sample_supplementary_file", "")
            ),
        })

    return {
        "series": series_record,
        "samples": sample_records,
    }


# ---------------------------------------------------------------------------
# Per-SRR ENA fetch (no aggregation)
# ---------------------------------------------------------------------------

def fetch_sra_per_srr(
    srx_accessions: list[str],
    bioprojects: Optional[list[str]] = None,
    concurrency: int = 30,
) -> pd.DataFrame:
    """Fetch ENA run-level metadata, preserving every SRR row.

    Unlike fetch_sra_runinfo() in sra.py which groups by SRX and drops
    library_name + fastq_bytes, this returns one row per SRR with all fields.
    """
    df = fetch_sra_runinfo(
        srx_accessions=srx_accessions,
        bioprojects=bioprojects,
        concurrency=concurrency,
    )

    if df.empty:
        return df

    # Parse individual FASTQ file URLs and byte sizes
    def _split_fastq_fields(row):
        ftp = str(row.get("fastq_ftp", ""))
        md5 = str(row.get("fastq_md5", ""))
        fbytes = str(row.get("fastq_bytes", ""))

        ftp_parts = [x.strip() for x in ftp.split(";") if x.strip()]
        md5_parts = [x.strip() for x in md5.split(";") if x.strip()]
        byte_parts = [x.strip() for x in fbytes.split(";") if x.strip()]

        r1_url = f"https://{ftp_parts[0]}" if len(ftp_parts) >= 1 and ftp_parts[0] else ""
        r2_url = f"https://{ftp_parts[1]}" if len(ftp_parts) >= 2 and ftp_parts[1] else ""
        r1_md5 = md5_parts[0] if len(md5_parts) >= 1 else ""
        r2_md5 = md5_parts[1] if len(md5_parts) >= 2 else ""

        try:
            r1_bytes = int(byte_parts[0]) if len(byte_parts) >= 1 and byte_parts[0] else 0
        except ValueError:
            r1_bytes = 0
        try:
            r2_bytes = int(byte_parts[1]) if len(byte_parts) >= 2 and byte_parts[1] else 0
        except ValueError:
            r2_bytes = 0

        return pd.Series({
            "fastq_r1_url": r1_url,
            "fastq_r2_url": r2_url,
            "fastq_r1_md5": r1_md5,
            "fastq_r2_md5": r2_md5,
            "fastq_r1_bytes": r1_bytes,
            "fastq_r2_bytes": r2_bytes,
        })

    extras = df.apply(_split_fastq_fields, axis=1)
    df = pd.concat([df, extras], axis=1)

    logger.info(f"Per-SRR catalog: {len(df):,} runs, "
                f"{df['experiment_accession'].nunique():,} experiments")

    return df


# ---------------------------------------------------------------------------
# Linkage table builders
# ---------------------------------------------------------------------------

def build_multiome_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Build GEX↔ATAC linkage table for Multiome GSEs.

    The input df must have library_type and assay_class columns.
    Returns one row per GEX-ATAC pair within each GSE.
    """
    mm = df[df["assay_class"] == "multiome"].copy()
    if mm.empty:
        return pd.DataFrame()

    gex = (
        mm[mm["library_type"] == "rna"]
        .groupby(["gse_id", "gsm_id"])
        .agg(gex_srr_accessions=("run_accession", _safe_join_srr))
        .reset_index()
        .rename(columns={"gsm_id": "gex_gsm"})
    )
    atac = (
        mm[mm["library_type"] == "atac"]
        .groupby(["gse_id", "gsm_id"])
        .agg(atac_srr_accessions=("run_accession", _safe_join_srr))
        .reset_index()
        .rename(columns={"gsm_id": "atac_gsm"})
    )

    # Cross-join within each GSE
    pairs = gex.merge(atac, on="gse_id", how="inner")
    pairs["gex_processed"] = False
    pairs["gex_barcode_whitelist_path"] = None

    logger.info(f"Multiome pairs: {len(pairs):,} GEX↔ATAC pairs "
                f"across {pairs['gse_id'].nunique():,} GSEs")
    return pairs


def build_citeseq_panels(df: pd.DataFrame) -> pd.DataFrame:
    """Build RNA↔ADT linkage table for CITE-seq GSEs."""
    cs = df[df["assay_class"] == "citeseq"].copy()
    if cs.empty:
        return pd.DataFrame()

    # Build combined text for panel inference
    text_cols = [
        "sample_extract_protocol", "sample_characteristics",
        "series_title", "summary",
    ]

    rna = (
        cs[cs["library_type"] == "rna"]
        .groupby(["gse_id", "gsm_id"])
        .agg(rna_srr_accessions=("run_accession", _safe_join_srr))
        .reset_index()
        .rename(columns={"gsm_id": "rna_gsm"})
    )
    adt = (
        cs[cs["library_type"] == "adt"]
        .groupby(["gse_id", "gsm_id"])
        .agg(adt_srr_accessions=("run_accession", _safe_join_srr))
        .reset_index()
        .rename(columns={"gsm_id": "adt_gsm"})
    )

    panels = rna.merge(adt, on="gse_id", how="inner")

    # Infer TotalSeq panel per GSE from series text
    gse_text = (
        cs.groupby("gse_id")
        .apply(lambda g: _combine_text(g.iloc[0], text_cols), include_groups=False)
        .rename("combined_text")
    )
    panels = panels.merge(gse_text, on="gse_id", how="left")
    panels["totalseq_panel"] = panels["combined_text"].fillna("").apply(infer_totalseq_panel)
    panels = panels.drop(columns=["combined_text"])
    panels["feature_csv_url"] = None

    logger.info(f"CITE-seq panels: {len(panels):,} RNA↔ADT pairs "
                f"across {panels['gse_id'].nunique():,} GSEs")
    return panels


def build_visium_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """Build Visium spatial metadata table."""
    vis = df[df["assay_class"] == "visium"].copy()
    if vis.empty:
        return pd.DataFrame()

    text_cols = [
        "series_title", "summary", "sample_extract_protocol",
        "sample_data_processing",
    ]

    # Collapse to per-GSM (multiple SRRs per GSM possible)
    gsm_vis = (
        vis.groupby(["gse_id", "gsm_id"])
        .agg(
            srr_accessions=("run_accession", _safe_join_srr),
            sample_supplementary_files=("sample_supplementary_files", "first"),
        )
        .reset_index()
    )

    # Merge back series text for chemistry detection
    series_text = (
        vis.groupby("gse_id")
        .apply(lambda g: _combine_text(g.iloc[0], text_cols), include_groups=False)
        .rename("combined_text")
    )
    gsm_vis = gsm_vis.merge(series_text, on="gse_id", how="left")

    combined = gsm_vis["combined_text"].fillna("")
    gsm_vis["chemistry_version"] = combined.apply(infer_visium_chemistry)
    gsm_vis["is_probe_based"] = combined.apply(lambda t: bool(_RE_PROBE.search(t)))

    # Parse supplementary file URLs for spatial data
    gsm_vis["tissue_positions_url"] = gsm_vis["sample_supplementary_files"].apply(
        lambda s: _find_supp_url(s, r"tissue.positions")
    )
    gsm_vis["scalefactors_url"] = gsm_vis["sample_supplementary_files"].apply(
        lambda s: _find_supp_url(s, r"scalefactors")
    )
    gsm_vis["he_image_url"] = gsm_vis["sample_supplementary_files"].apply(
        lambda s: _find_supp_url(s, r"hires.*image|tissue.*hires")
    )

    gsm_vis = gsm_vis.drop(columns=["combined_text"])

    logger.info(f"Visium spatial: {len(gsm_vis):,} GSMs "
                f"({gsm_vis['is_probe_based'].sum()} probe-based excluded) "
                f"across {gsm_vis['gse_id'].nunique():,} GSEs")
    return gsm_vis


def build_perturbseq_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Build Perturb-seq candidate table."""
    ps = df[df["assay_class"] == "perturbseq"].copy()
    if ps.empty:
        return pd.DataFrame()

    rna_rows = ps[ps["library_type"] == "rna"]
    guide_rows = ps[ps["library_type"] == "guide"]

    # Per-GSM: RNA SRRs
    gsm_rna = (
        rna_rows.groupby(["gse_id", "gsm_id"])
        .agg(rna_srr_accessions=("run_accession", _safe_join_srr))
        .reset_index()
    )

    # Per-GSM: guide SRRs (may be empty for many GSMs)
    gsm_guide = (
        guide_rows.groupby(["gse_id", "gsm_id"])
        .agg(guide_srr_accessions=("run_accession", _safe_join_srr))
        .reset_index()
    )

    result = gsm_rna.merge(gsm_guide, on=["gse_id", "gsm_id"], how="left")
    result["rna_processed"] = False
    result["sgrna_csv_url"] = None

    logger.info(f"Perturb-seq candidates: {len(result):,} GSMs "
                f"({(result['guide_srr_accessions'].notna()).sum()} with guide SRRs) "
                f"across {result['gse_id'].nunique():,} GSEs")
    return result


def _safe_join_srr(series: pd.Series) -> str:
    """Join SRR accessions, filtering NaN values."""
    return ";".join(sorted(str(x) for x in series if pd.notna(x)))


def _find_supp_url(supp_str: str, pattern: str) -> Optional[str]:
    """Find a URL in semicolon-delimited supplementary file list matching pattern."""
    if not supp_str or pd.isna(supp_str):
        return None
    for part in str(supp_str).split(";;"):
        part = part.strip()
        if re.search(pattern, part, re.IGNORECASE):
            return part
    return None


# ---------------------------------------------------------------------------
# Main catalog builder
# ---------------------------------------------------------------------------

def build_multimodal_catalog(
    output_dir: Optional[Path] = None,
    config=None,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build the Stage 7 multi-modal catalog.

    Stages:
      1. NCBI ESearch with MULTIMODAL_QUERY
      2. ESummary title fetch (NO true-negative filter)
      3. SOFT download + parse (with Sample_supplementary_file)
      4. ENA filereport per-SRR (no aggregation)
      5. library_type + assay_class inference
      6. Build linkage tables
      7. Write parquet outputs

    Args:
        output_dir: Where to write parquet files (default: catalog_dir)
        config: Configuration object
        dry_run: If True, stop after discovery (stage 1) and print counts

    Returns:
        Dict of DataFrames: {
            "catalog": full per-SRR catalog,
            "multiome_pairs": GEX↔ATAC linkage,
            "citeseq_panels": RNA↔ADT linkage,
            "visium_spatial": Visium metadata,
            "perturbseq_candidates": Perturb-seq metadata,
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    if config is None:
        config = get_config()

    if output_dir is None:
        output_dir = config.paths.catalog_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Discovery (per-modality sub-queries, unioned) ────────────
    logger.info("=" * 60)
    logger.info("Stage 7.1: Discovering multi-modal series")

    all_uids = set()
    uid_provenance = {}  # uid → set of modality labels
    modality_counts = {}

    for modality, query in MODALITY_QUERIES.items():
        logger.info(f"  Sub-query: {modality}")
        uids_m, _, _ = esearch_paginated(query=query)
        modality_counts[modality] = len(uids_m)
        for uid in uids_m:
            uid_provenance.setdefault(uid, set()).add(modality)
        all_uids.update(uids_m)
        logger.info(f"    {modality}: {len(uids_m):,} series")

    uids = sorted(all_uids)
    logger.info(f"  Union: {len(uids):,} unique series")
    logger.info(f"  Per-modality: {modality_counts}")

    if dry_run:
        logger.info("DRY RUN — stopping after discovery")
        return {
            "discovery_count": len(uids),
            "modality_counts": modality_counts,
        }

    # ── Stage 2: ESummary metadata ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 7.2: Fetching series metadata")
    series_metadata = fetch_series_metadata(uids, batch_size=100)
    samples = extract_gsm_samples(series_metadata)
    df = pd.DataFrame(samples)
    logger.info(f"  Extracted {len(df):,} samples from {len(series_metadata):,} series")

    # Add discovery provenance per GSE
    # Add discovery provenance per GSE
    uid_to_gse = {}
    for rec in series_metadata:
        uid = str(rec.get("gds_uid", ""))
        gse = rec.get("gse_id", "")
        if uid and gse:
            uid_to_gse[uid] = gse

    gse_provenance = {}
    for uid, modalities in uid_provenance.items():
        gse = uid_to_gse.get(uid, "")
        if gse:
            gse_provenance.setdefault(gse, set()).update(modalities)

    df["discovery_modalities"] = df["gse_id"].map(
        lambda g: ";".join(sorted(gse_provenance.get(g, set())))
    )

    # ── Stage 3: SOFT files (with Sample_supplementary_file) ────────────
    logger.info("=" * 60)
    logger.info("Stage 7.3: Downloading + parsing SOFT files")
    soft_cache = config.paths.catalog_dir / "soft_cache"
    gse_ids = df["gse_id"].unique().tolist()

    all_series = []
    all_samples = []
    failed = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_gse_soft_multimodal, gse, soft_cache): gse
            for gse in gse_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="SOFT download+parse"):
            gse = futures[future]
            try:
                result = future.result()
                if result is None:
                    failed.append(gse)
                    continue
                all_series.append(result["series"])
                all_samples.extend(result["samples"])
            except Exception as e:
                logger.warning(f"Error processing {gse}: {e}")
                failed.append(gse)

    logger.info(f"  Parsed: {len(all_series):,} GSEs, failed: {len(failed):,}")

    df_soft_series = pd.DataFrame(all_series)
    df_soft_samples = pd.DataFrame(all_samples)
    df_soft = df_soft_samples.merge(df_soft_series, on="gse_id", how="left")

    # Merge SOFT into ESummary samples
    df = df.merge(df_soft, on=["gse_id", "gsm_id"], how="left", suffixes=("", "_soft"))

    # ── Stage 4: ENA per-SRR (no aggregation) ──────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 7.4: Querying ENA for per-SRR metadata")
    srx_list = df["srx_accession"].dropna().unique().tolist()
    bioprojects = None
    if "bioproject" in df.columns:
        bioprojects = df["bioproject"].dropna().unique().tolist()

    df_sra = fetch_sra_per_srr(
        srx_accessions=srx_list,
        bioprojects=bioprojects,
        concurrency=30,
    )

    if not df_sra.empty:
        # Merge SRA onto samples — per-SRR granularity
        # First merge to get one row per (gsm_id, run_accession)
        df_sra = df_sra.rename(columns={"experiment_accession": "srx_accession"})
        df = df.merge(df_sra, on="srx_accession", how="left", suffixes=("", "_sra"))
        logger.info(f"  Merged SRA: {len(df):,} rows (per-SRR)")
    else:
        logger.warning("  No SRA data retrieved")

    # ── Stage 5: Library type + assay class inference ──────────────────
    logger.info("=" * 60)
    logger.info("Stage 7.5: Classifying library types")

    # ---- Vectorized library_type inference ----
    # Build text columns once, fillna to avoid NaN issues
    lib_strategy = df["library_strategy"].fillna("").str.lower().str.strip() if "library_strategy" in df.columns else pd.Series("", index=df.index)
    lib_name = df["library_name"].fillna("").str.lower().str.strip() if "library_name" in df.columns else pd.Series("", index=df.index)
    soft_strategy = df["sample_library_strategy"].fillna("").str.lower().str.strip() if "sample_library_strategy" in df.columns else pd.Series("", index=df.index)

    # Build sample_text for SOFT fallback
    _st = df["sample_title"].fillna("") if "sample_title" in df.columns else pd.Series("", index=df.index)
    _ss = df["sample_source"].fillna("") if "sample_source" in df.columns else pd.Series("", index=df.index)
    _gt = df["gsm_title"].fillna("") if "gsm_title" in df.columns else pd.Series("", index=df.index)
    sample_text = (_st + " " + _ss + " " + _gt).str.lower()

    # Vectorized regex matching
    lib_name_adt = lib_name.str.contains(_RE_ADT.pattern, case=False, na=False)
    lib_name_guide = lib_name.str.contains(_RE_GUIDE.pattern, case=False, na=False)
    lib_name_atac = lib_name.str.contains(_RE_ATAC.pattern, case=False, na=False)
    lib_name_gex = lib_name.str.contains(_RE_GEX.pattern, case=False, na=False)

    sample_adt = sample_text.str.contains(_RE_ADT.pattern, case=False, na=False)
    sample_guide = sample_text.str.contains(_RE_GUIDE.pattern, case=False, na=False)
    sample_atac = sample_text.str.contains(_RE_ATAC.pattern, case=False, na=False)
    sample_gex = sample_text.str.contains(_RE_GEX.pattern, case=False, na=False)

    # Apply priority chain: P1 → P2 → P3 → P4
    lt = pd.Series("unknown", index=df.index)
    # P4: fallbacks (applied first, overridden by higher priorities)
    lt = lt.where(~(soft_strategy == "rna-seq"), "rna")
    lt = lt.where(~(lib_strategy == "rna-seq"), "rna")
    # P3: SOFT fields
    lt = lt.where(~sample_gex, "rna")
    lt = lt.where(~sample_atac, "atac")
    lt = lt.where(~sample_guide, "guide")
    lt = lt.where(~sample_adt, "adt")
    lt = lt.where(~(soft_strategy == "atac-seq"), "atac")
    # P2: ENA library_name
    lt = lt.where(~lib_name_gex, "rna")
    lt = lt.where(~lib_name_atac, "atac")
    lt = lt.where(~lib_name_guide, "guide")
    lt = lt.where(~lib_name_adt, "adt")
    # P1: ENA library_strategy
    lt = lt.where(~(lib_strategy == "atac-seq"), "atac")

    df["library_type"] = lt

    # ---- Vectorized assay_class inference (keyword-based for speed) ----
    # NOTE: Replaced .{0,50} regex with keyword co-occurrence for 100x speedup
    text_cols = [
        "series_title", "summary", "overall_design",
        "sample_extract_protocol", "sample_characteristics",
        "sample_data_processing", "sample_title",
    ]
    parts = []
    for c in text_cols:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
        else:
            parts.append(pd.Series("", index=df.index))
    combined_text = parts[0].str.cat(parts[1:], sep=" ").str.lower()
    logger.info("    Built combined_text for assay classification")

    # Fast literal substring check (regex=False, case=True since text is pre-lowered)
    def _has(kw):
        return combined_text.str.contains(kw, case=True, na=False, regex=False)

    # Simple regex (only for .? optional-char patterns, no .{0,N})
    def _has_re(pat):
        return combined_text.str.contains(pat, case=True, na=False, regex=True)

    # Multiome: simple keywords OR keyword co-occurrence
    is_multiome = (
        _has("multiome") | _has_re(r"multi.ome") |
        _has_re(r"share.?seq") | _has_re(r"snare.?seq") |
        _has("sci-car") | _has("sci car") |
        _has("paired-tag") | _has("paired tag") |
        _has("cotech") |
        _has_re(r"dogma.?seq") | _has_re(r"tea.?seq") | _has_re(r"asap.?seq") |
        # Keyword co-occurrence (replaces expensive .{0,50} patterns)
        (_has("10x") & _has("atac") & _has("gex")) |
        (_has("joint") & _has("atac") & _has("rna")) |
        (_has("simultaneous") & _has("atac") & _has("rna")) |
        (_has("chromatin") & _has("transcriptom")) |
        (_has("snatac") & _has("snrna"))
    )
    logger.info("    Multiome classification done")

    # CITE-seq
    is_citeseq = (
        _has_re(r"cite.?seq") | _has("totalseq") |
        _has("antibody derived tag") |
        _has("cellular indexing of transcriptomes") |
        _has_re(r"reap.?seq") | _has_re(r"eccite.?seq") | _has("abseq") |
        _has("feature barcod") | _has("surface protein") |
        _has("antibody capture") | _has("cell hashing") |
        (_has("multiplex") & _has("hto"))
    )
    logger.info("    CITE-seq classification done")

    # Visium
    is_visium = (
        _has("visium") | _has("spatial transcriptom") |
        _has("spatial gene express") | _has("spatially resolved transcriptom") |
        _has_re(r"slide.?seq") | _has_re(r"stereo.?seq") | _has_re(r"dbit.?seq") |
        _has("seq-scope") | _has("seqscope") | _has("spatial barcod")
    )
    logger.info("    Visium classification done")

    # Perturb-seq
    is_perturbseq = (
        _has_re(r"perturb.?seq") | _has_re(r"crop.?seq") |
        _has("pooled crispr") | _has("genetic perturbation screen") |
        _has_re(r"tap.?seq") | _has_re(r"mosaic.?seq") | _has("sccrispr") |
        _has("crispr perturbation") |
        # Keyword co-occurrence (replaces expensive .{0,30} patterns)
        (_has("crispr") & _has("screen") & _has("scrna")) |
        (_has("crispri") & _has("single cell")) |
        (_has("crispra") & _has("single cell")) |
        (_has("sgrna") & _has("capture")) |
        (_has("grna") & _has("single cell")) |
        (_has("crispr interference") & _has("scrna"))
    )
    logger.info("    Perturb-seq classification done")

    # Standalone scATAC (only if NOT already classified as multiome)
    is_scatac = (
        _has_re(r"scatac|snatac") |
        (_has("single cell") & _has("atac")) |
        (_has("single nucleus") & _has("atac")) |
        _has_re(r"sci.?atac") | _has("dscatac") | _has_re(r"s3.?atac") |
        _has_re(r"sccut.?tag") | _has_re(r"sncut.?tag") |
        ((_has("chromatin") | _has("accessibility")) & _has("single cell"))
    )
    logger.info("    scATAC classification done")

    ac = pd.Series("rna_only", index=df.index)
    ac = ac.where(~is_scatac, "scatac")
    ac = ac.where(~is_perturbseq, "perturbseq")
    ac = ac.where(~is_visium, "visium")
    ac = ac.where(~is_citeseq, "citeseq")
    ac = ac.where(~is_multiome, "multiome")
    df["assay_class"] = ac

    # Nuclear detection (simple alternation — fast)
    df["is_nuclear"] = (
        _has("snuc") | _has("nuclei") | _has("snrna") |
        _has("nuclear") | _has("nucleus") |
        _has("single nuc") | _has("single-nuc")
    )
    # Force Multiome to nuclear
    df.loc[df["assay_class"] == "multiome", "is_nuclear"] = True

    # ---- Protocol inference (for reprocessing pipeline) ----
    # Detect 10x/droplet protocol from combined_text + instrument metadata
    # Priority: explicit keywords > instrument model > library_strategy
    _PLATE_PROTOCOLS = {"smartseq2", "smartseq3", "plate_based", "smartseq"}
    _HIGH_BATCH_RISK = {"icell8", "strtseq"}
    _MEDIUM_BATCH_RISK = {"celseq", "celseq2", "marsseq", "scirna", "quartzseq",
                          "surecell", "scopeseq"}
    _LOW_BATCH_RISK_DROPLET = {"10xv2", "10xv3", "10xv4", "10xv3_5prime",
                               "10x_multiome", "10x_suspect", "dropseq", "indrop",
                               "dnbelab", "bd_rhapsody", "ddseq", "microwell",
                               "parse", "seqwell", "splitseq"}

    proto = pd.Series("unknown", index=df.index)
    # 10x Chromium versions
    proto = proto.where(~_has("chromium"), "10x_suspect")
    proto = proto.where(~(_has("10x") & ~_has("atac") & ~_has("visium")), "10x_suspect")
    proto = proto.where(~_has("10x genomic"), "10x_suspect")
    proto = proto.where(~_has("10x chromium"), "10x_suspect")
    proto = proto.where(~_has_re(r"chromium.*v2"), "10xv2")
    proto = proto.where(~_has_re(r"10x.*v2"), "10xv2")
    proto = proto.where(~_has_re(r"chromium.*v3"), "10xv3")
    proto = proto.where(~_has_re(r"10x.*v3\b"), "10xv3")
    proto = proto.where(~_has("3' gene expression"), "10xv3")
    proto = proto.where(~_has_re(r"chromium.*v4"), "10xv4")
    proto = proto.where(~_has_re(r"10x.*v4"), "10xv4")
    proto = proto.where(~_has("5' gene expression"), "10xv3_5prime")
    proto = proto.where(~_has_re(r"chromium.*5'"), "10xv3_5prime")
    proto = proto.where(~(_has("multiome") & _has("10x")), "10x_multiome")
    # Other droplet
    proto = proto.where(~_has_re(r"drop.?seq"), "dropseq")
    proto = proto.where(~_has("indrop"), "indrop")
    proto = proto.where(~_has("dnbelab"), "dnbelab")
    proto = proto.where(~_has("bd rhapsody"), "bd_rhapsody")
    proto = proto.where(~_has("rhapsody"), "bd_rhapsody")
    proto = proto.where(~_has("ddseq"), "ddseq")
    proto = proto.where(~_has("microwell"), "microwell")
    proto = proto.where(~_has("parse bioscience"), "parse")
    proto = proto.where(~_has("evercode"), "parse")
    proto = proto.where(~_has_re(r"seq.?well"), "seqwell")
    proto = proto.where(~_has("split-seq"), "splitseq")
    proto = proto.where(~_has("splitseq"), "splitseq")
    proto = proto.where(~_has("sci-rna"), "scirna")
    proto = proto.where(~_has("surecell"), "surecell")
    # Plate-based (to flag for exclusion)
    proto = proto.where(~_has("smart-seq"), "smartseq2")
    proto = proto.where(~_has("smartseq"), "smartseq2")
    proto = proto.where(~_has("plate-based"), "plate_based")
    proto = proto.where(~_has("plate based"), "plate_based")
    # Well-based with UMI
    proto = proto.where(~_has("cel-seq"), "celseq")
    proto = proto.where(~_has("celseq"), "celseq")
    proto = proto.where(~_has("cel-seq2"), "celseq2")
    proto = proto.where(~_has("celseq2"), "celseq2")
    proto = proto.where(~_has("mars-seq"), "marsseq")
    proto = proto.where(~_has("marsseq"), "marsseq")
    proto = proto.where(~_has("icell8"), "icell8")
    proto = proto.where(~_has("strt-seq"), "strtseq")
    proto = proto.where(~_has("strtseq"), "strtseq")
    df["protocol_inferred"] = proto
    logger.info("    Protocol inference done")

    # Droplet flag: True if suitable for standard droplet reprocessing
    df["is_droplet"] = df["protocol_inferred"].isin(
        _LOW_BATCH_RISK_DROPLET | _MEDIUM_BATCH_RISK
    )
    # Exclude plate-based even if they slipped through
    df.loc[df["protocol_inferred"].isin(_PLATE_PROTOCOLS), "is_droplet"] = False

    # Batch risk tier for downstream model training decisions
    batch_risk = pd.Series("unknown", index=df.index)
    batch_risk = batch_risk.where(
        ~df["protocol_inferred"].isin(_LOW_BATCH_RISK_DROPLET), "low"
    )
    batch_risk = batch_risk.where(
        ~df["protocol_inferred"].isin(_MEDIUM_BATCH_RISK), "medium"
    )
    batch_risk = batch_risk.where(
        ~df["protocol_inferred"].isin(_HIGH_BATCH_RISK), "high"
    )
    batch_risk = batch_risk.where(
        ~df["protocol_inferred"].isin(_PLATE_PROTOCOLS), "exclude"
    )
    df["batch_risk"] = batch_risk

    logger.info(f"  Library types: {df['library_type'].value_counts().to_dict()}")
    logger.info(f"  Assay classes: {df['assay_class'].value_counts().to_dict()}")
    logger.info(f"  Protocols: {df['protocol_inferred'].value_counts().to_dict()}")
    logger.info(f"  Droplet: {df['is_droplet'].sum():,} / {len(df):,} ({100*df['is_droplet'].mean():.1f}%)")
    logger.info(f"  Batch risk: {df['batch_risk'].value_counts().to_dict()}")

    # ── Stage 6: Build linkage tables ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 7.6: Building linkage tables")

    multiome_pairs = build_multiome_pairs(df)
    citeseq_panels = build_citeseq_panels(df)
    visium_spatial = build_visium_spatial(df)
    perturbseq_candidates = build_perturbseq_candidates(df)

    # ── Stage 7: Write outputs ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 7.7: Writing parquet outputs")

    catalog_path = output_dir / "stage7_multimodal_catalog.parquet"
    df.to_parquet(catalog_path, index=False, compression="zstd")
    logger.info(f"  Catalog: {catalog_path} ({len(df):,} rows)")

    pairs_path = output_dir / "stage7_multiome_pairs.parquet"
    multiome_pairs.to_parquet(pairs_path, index=False, compression="zstd")
    logger.info(f"  Multiome pairs: {pairs_path} ({len(multiome_pairs):,} rows)")

    panels_path = output_dir / "stage7_citeseq_panels.parquet"
    citeseq_panels.to_parquet(panels_path, index=False, compression="zstd")
    logger.info(f"  CITE-seq panels: {panels_path} ({len(citeseq_panels):,} rows)")

    vis_path = output_dir / "stage7_visium_spatial.parquet"
    visium_spatial.to_parquet(vis_path, index=False, compression="zstd")
    logger.info(f"  Visium spatial: {vis_path} ({len(visium_spatial):,} rows)")

    ps_path = output_dir / "stage7_perturbseq_candidates.parquet"
    perturbseq_candidates.to_parquet(ps_path, index=False, compression="zstd")
    logger.info(f"  Perturb-seq: {ps_path} ({len(perturbseq_candidates):,} rows)")

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 7 catalog complete:")
    logger.info(f"  Total SRR rows: {len(df):,}")
    logger.info(f"  Unique GSEs: {df['gse_id'].nunique():,}")
    logger.info(f"  Unique GSMs: {df['gsm_id'].nunique():,}")
    logger.info(f"  Multiome pairs: {len(multiome_pairs):,}")
    logger.info(f"  CITE-seq panels: {len(citeseq_panels):,}")
    logger.info(f"  Visium GSMs: {len(visium_spatial):,}")
    logger.info(f"  Perturb-seq GSMs: {len(perturbseq_candidates):,}")

    return {
        "catalog": df,
        "multiome_pairs": multiome_pairs,
        "citeseq_panels": citeseq_panels,
        "visium_spatial": visium_spatial,
        "perturbseq_candidates": perturbseq_candidates,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build Stage 7 multi-modal GEO catalog"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for parquet files (default: catalog_dir from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run discovery (stage 1) and print counts, then exit",
    )
    args = parser.parse_args()

    import signal

    def _sig_handler(signum, frame):
        logging.getLogger(__name__).error(f"Received signal {signum}")
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGHUP, _sig_handler)

    try:
        result = build_multimodal_catalog(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except BaseException as exc:
        logging.getLogger(__name__).exception(f"BUILD FAILED ({type(exc).__name__})")
        sys.exit(1)

    if args.dry_run:
        print(f"Would discover {result['discovery_count']} series (union)")
        for mod, cnt in result.get("modality_counts", {}).items():
            print(f"  {mod}: {cnt:,}")
        sys.exit(0)

    print("\nStage 7 catalog written. Summary:")
    print(f"  Catalog rows: {len(result['catalog']):,}")
    print(f"  Multiome pairs: {len(result['multiome_pairs']):,}")
    print(f"  CITE-seq panels: {len(result['citeseq_panels']):,}")
    print(f"  Visium GSMs: {len(result['visium_spatial']):,}")
    print(f"  Perturb-seq GSMs: {len(result['perturbseq_candidates']):,}")
