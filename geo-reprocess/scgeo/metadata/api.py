"""High-level API for author metadata extraction.

Orchestrates the three-tier metadata pipeline:

1. **Tier 1** — Sample-level SOFT characteristics (100 % coverage)
2. **Tier 2** — Cell-level annotations from supplementary files
3. **Tier 3** — Natural language descriptions from NCBI E-utilities

The primary entry point is :func:`build_metadata`, which produces a
:class:`MetadataResult` per GSM containing an aligned
:class:`~anndata.AnnData`-ready ``.obs`` DataFrame and ``uns`` dict.
"""

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scgeo.config import get_config
from scgeo.metadata.soft import load_soft_metadata, parse_characteristics, build_gsm_level_obs
from scgeo.metadata.description import fetch_geo_description
from scgeo.metadata.barcodes import align_author_metadata
from scgeo.metadata.download import download_supplementary_file
from scgeo.metadata.extract import (
    extract_metadata_from_h5ad,
    extract_metadata_from_tabular,
    extract_metadata_from_loom,
)
from scgeo.metadata.extract_rds import extract_metadata_from_rds

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────── formats ──

# Map supplementary-file extensions to extraction format names
_EXT_FORMAT = {
    ".h5ad": "h5ad",
    ".rds": "rds",
    ".rds.gz": "rds",
    ".csv": "tabular",
    ".csv.gz": "tabular",
    ".tsv": "tabular",
    ".tsv.gz": "tabular",
    ".txt": "tabular",
    ".txt.gz": "tabular",
    ".tab": "tabular",
    ".tab.gz": "tabular",
    ".loom": "loom",
    # Stage 2c formats
    ".xlsx": "xlsx",
    ".xlsx.gz": "xlsx",
    ".xls": "xlsx",
    ".xls.gz": "xlsx",
    ".h5seurat": "h5seurat",
    ".h5mu": "h5mu",
    ".hdf5": "hdf5",
    ".rdata.gz": "rdata",
    ".rdata": "rdata",
    ".rda.gz": "rdata",
    ".rda": "rdata",
    ".robj.gz": "rdata",
    ".robj": "rdata",
    ".tar": "tar",
    ".tar.gz": "tar",
}

# Exclude files that are count matrices, not metadata
_SKIP_PATTERNS = re.compile(
    r"(matrix\.mtx|genes\.tsv|barcodes\.tsv|features\.tsv"
    r"|\.bam$|\.bai$|\.bed$|\.bw$|\.bigwig$|\.gtf$|\.gff$"
    r"|\.fasta$|\.fa$|\.fastq|\.sam$|\.cram$)",
    re.IGNORECASE,
)

# Patterns that suggest a file contains cell-level metadata
_META_PATTERNS = re.compile(
    r"(meta|annot|obs|cell.?type|cluster|label|barcode|pheno|coldata"
    r"|cell.?info|sample.?info|cell.?assign|ident|classif)",
    re.IGNORECASE,
)


# ────────────────────────────────────────────────────────── dataclass ──

@dataclass
class MetadataResult:
    """Result of metadata extraction for a single GSM.

    Attributes:
        gse_id: GEO series accession.
        gsm_id: GEO sample accession.
        obs: Cell-level DataFrame indexed by our barcodes.  Contains
            columns from all three tiers.
        uns: Experiment-level metadata dict (title, summary, organism,
            SOFT design, processing notes).
        tier1_columns: Columns originating from SOFT characteristics.
        tier2_columns: Columns originating from supplementary files.
        tier2_source: Filename of the supplementary file used, or ``None``.
        tier2_format: Format of the supplementary file (``h5ad``,
            ``rds``, ``tabular``, ``loom``), or ``None``.
        match_stats: Barcode matching statistics dict.
        status: ``"success"``, ``"partial"`` (Tier 1+3 only), or
            ``"failed"``.
        error: Error message if status is ``"failed"``.
    """
    gse_id: str
    gsm_id: str
    obs: pd.DataFrame = field(default_factory=pd.DataFrame)
    uns: Dict = field(default_factory=dict)
    tier1_columns: List[str] = field(default_factory=list)
    tier2_columns: List[str] = field(default_factory=list)
    tier2_source: Optional[str] = None
    tier2_format: Optional[str] = None
    match_stats: Dict = field(default_factory=dict)
    status: str = "pending"
    error: str = ""


# ───────────────────────────────────────── supplementary file routing ──

def classify_supplementary_files(
    gse_id: str,
    soft_meta: Optional[Dict] = None,
) -> Dict[str, List[Dict]]:
    """Classify supplementary files for a GSE by extraction format.

    Parses the ``supplementary_files`` column from the SOFT catalog
    and groups files into categories: ``h5ad``, ``rds``, ``tabular``,
    ``loom``, and ``skip``.

    Files whose names match metadata-like patterns are prioritized.

    Args:
        gse_id: GEO series accession.
        soft_meta: Pre-loaded SOFT metadata dict (from
            :func:`~scgeo.metadata.soft.load_soft_metadata`).
            Loaded automatically if ``None``.

    Returns:
        ``{format: [{url, filename, gsm_id, is_metadata_like}, ...]}``
    """
    if soft_meta is None:
        soft_meta = load_soft_metadata(gse_id)

    result = {"h5ad": [], "rds": [], "tabular": [], "loom": [], "tar": [],
              "xlsx": [], "h5seurat": [], "h5mu": [], "hdf5": [], "rdata": [],
              "skip": []}

    supp_files = soft_meta.get("supplementary_files", {})
    for gsm_id, raw_str in supp_files.items():
        if not raw_str:
            continue
        urls = [u.strip() for u in str(raw_str).split(";;") if u.strip()]
        for url in urls:
            fn = url.rstrip("/").split("/")[-1].lower()
            if _SKIP_PATTERNS.search(fn):
                result["skip"].append({"url": url, "filename": fn, "gsm_id": gsm_id})
                continue

            fmt = _classify_filename(fn)
            if fmt:
                entry = {
                    "url": url,
                    "filename": fn,
                    "gsm_id": gsm_id,
                    "is_metadata_like": bool(_META_PATTERNS.search(fn)),
                }
                result[fmt].append(entry)
            else:
                result["skip"].append({"url": url, "filename": fn, "gsm_id": gsm_id})

    return result


def _classify_filename(fn: str) -> Optional[str]:
    """Classify a single filename by extension."""
    fn_lower = fn.lower()
    # Check compound extensions first (.rds.gz, .csv.gz, etc.)
    for ext in sorted(_EXT_FORMAT, key=len, reverse=True):
        if fn_lower.endswith(ext):
            return _EXT_FORMAT[ext]
    return None


# ───────────────────────────────────────────────── extraction routing ──

_EXTRACTORS = {
    "h5ad": extract_metadata_from_h5ad,
    "rds": extract_metadata_from_rds,
    "tabular": extract_metadata_from_tabular,
    "loom": extract_metadata_from_loom,
}


def _extract_supplementary(
    url: str,
    fmt: str,
    work_dir: Path,
) -> pd.DataFrame:
    """Download and extract metadata from a supplementary file.

    Args:
        url: GEO URL to the supplementary file.
        fmt: Format key (``h5ad``, ``rds``, ``tabular``, ``loom``).
        work_dir: Temporary working directory for downloads.

    Returns:
        DataFrame with ``barcode`` column, or empty DataFrame on failure.
    """
    fn = url.rstrip("/").split("/")[-1]
    dest = work_dir / fn

    if not download_supplementary_file(url, dest):
        return pd.DataFrame()

    extractor = _EXTRACTORS.get(fmt)
    if extractor is None:
        logger.warning("No extractor for format %s (%s)", fmt, fn)
        return pd.DataFrame()

    return extractor(dest)


# ────────────────────────────────────────────────── primary entry pts ──

def build_metadata(
    gse_id: str,
    gsm_id: str,
    work_dir: Optional[Path] = None,
    include_tier2: bool = True,
    include_tier3: bool = True,
) -> MetadataResult:
    """Build complete metadata for a single GSM.

    Executes the three-tier metadata pipeline and returns a
    :class:`MetadataResult` with an ``.obs`` DataFrame aligned to
    the sample's cell barcodes.

    Args:
        gse_id: GEO series accession.
        gsm_id: GEO sample accession.
        work_dir: Directory for temporary downloads.  A temporary
            directory is created if ``None``.
        include_tier2: Whether to attempt cell-level extraction from
            supplementary files.
        include_tier3: Whether to fetch NCBI descriptions.

    Returns:
        :class:`MetadataResult` with fields populated from all
        available tiers.

    Example:
        >>> from scgeo.metadata import build_metadata
        >>> result = build_metadata("GSE281311", "GSM8618551")
        >>> result.obs.head()
    """
    res = MetadataResult(gse_id=gse_id, gsm_id=gsm_id)
    cleanup_work_dir = False

    try:
        # ── Tier 1: SOFT characteristics ──
        soft_meta = load_soft_metadata(gse_id)
        if not soft_meta:
            res.status = "failed"
            res.error = f"No SOFT metadata found for {gse_id}"
            return res

        tier1_obs = build_gsm_level_obs(gse_id, gsm_id, soft_meta)
        res.tier1_columns = list(tier1_obs.columns) if not tier1_obs.empty else []

        # ── Tier 2: Cell-level supplementary metadata ──
        tier2_obs = pd.DataFrame()
        if include_tier2:
            if work_dir is None:
                work_dir = Path(tempfile.mkdtemp(prefix="scgeo_meta_"))
                cleanup_work_dir = True

            tier2_obs, t2_source, t2_fmt = _try_tier2(gse_id, gsm_id, soft_meta, work_dir)
            if not tier2_obs.empty:
                aligned, stats = align_author_metadata(tier2_obs, gse_id, gsm_id)
                tier2_obs = aligned
                res.match_stats = stats
                res.tier2_source = t2_source
                res.tier2_format = t2_fmt
                res.tier2_columns = [c for c in aligned.columns if not aligned[c].isna().all()]

        # ── Tier 3: NCBI description ──
        description = {}
        if include_tier3:
            description = fetch_geo_description(gse_id)

        # ── Assemble obs DataFrame ──
        obs = _merge_tiers(tier1_obs, tier2_obs, gse_id, gsm_id)
        res.obs = obs

        # ── Assemble uns dict ──
        res.uns = {
            "title": description.get("title", ""),
            "summary": description.get("summary", ""),
            "organism": description.get("organism", ""),
            "pubmed_ids": description.get("pubmed_ids", []),
            "overall_design": soft_meta.get("overall_design", ""),
            "data_processing": soft_meta.get("data_processing", ""),
            "gse_id": gse_id,
            "gsm_id": gsm_id,
        }

        res.status = "success" if not tier2_obs.empty else "partial"
        return res

    except Exception as e:
        logger.error("Metadata build failed for %s/%s: %s", gse_id, gsm_id, e)
        res.status = "failed"
        res.error = str(e)
        return res

    finally:
        if cleanup_work_dir and work_dir and work_dir.exists():
            import shutil
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass


def _try_tier2(
    gse_id: str,
    gsm_id: str,
    soft_meta: Dict,
    work_dir: Path,
) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Attempt Tier 2 extraction for a single GSM.

    Prioritises GSE-level metadata files (shared across all GSMs in the
    series) and files whose names suggest they contain metadata.

    Returns:
        Tuple of (author_obs, source_filename, format_key).
    """
    classified = classify_supplementary_files(gse_id, soft_meta)

    # Build candidate list, prioritising metadata-like filenames
    candidates = []
    for fmt in ("h5ad", "tabular", "rds", "loom"):
        entries = classified.get(fmt, [])
        # Sort: metadata-like first, then GSE-level, then GSM-level
        entries = sorted(entries, key=lambda e: (
            not e.get("is_metadata_like", False),
            e["gsm_id"] == gsm_id,  # GSE-level files first
        ))
        for entry in entries:
            # Only include GSE-level or matching GSM files
            if entry["gsm_id"] == gsm_id or entry["gsm_id"] == gse_id:
                candidates.append((entry, fmt))

    if not candidates:
        return pd.DataFrame(), None, None

    # Try candidates until one works
    for entry, fmt in candidates:
        url = entry["url"]
        fn = entry["filename"]
        logger.info("Tier 2: trying %s (%s) for %s/%s", fn, fmt, gse_id, gsm_id)

        author_obs = _extract_supplementary(url, fmt, work_dir)
        if not author_obs.empty and "barcode" in author_obs.columns:
            logger.info(
                "Tier 2: extracted %d cells × %d columns from %s",
                len(author_obs), len(author_obs.columns) - 1, fn,
            )
            return author_obs, fn, fmt

    return pd.DataFrame(), None, None


def _merge_tiers(
    tier1_obs: pd.DataFrame,
    tier2_obs: pd.DataFrame,
    gse_id: str,
    gsm_id: str,
) -> pd.DataFrame:
    """Merge Tier 1 and Tier 2 obs DataFrames.

    Tier 2 (cell-level) takes precedence where available.
    Tier 1 (sample-level) fills out remaining cells and adds any
    sample-level columns not covered by Tier 2.

    If neither tier produced results, reads our cell barcodes and
    returns an empty-column DataFrame indexed by them.
    """
    # If Tier 2 is available, use it as the base (already aligned to our barcodes)
    if not tier2_obs.empty:
        obs = tier2_obs.copy()
        # Add Tier 1 columns that aren't already in Tier 2
        if not tier1_obs.empty:
            for col in tier1_obs.columns:
                if col not in obs.columns:
                    # Broadcast sample-level value to all cells
                    obs[col] = tier1_obs[col].iloc[0] if len(tier1_obs) > 0 else None
        return obs

    # Tier 2 not available — try to build from Tier 1 + our barcodes
    if not tier1_obs.empty:
        return tier1_obs

    # Nothing available — read barcodes and return bare index
    try:
        import pyarrow.parquet as pq
        config = get_config()
        cells_path = config.paths.project_base / "dataset" / gse_id / gsm_id / "cells.parquet"
        if cells_path.exists():
            cells = pq.read_table(str(cells_path), columns=["barcode"]).to_pandas()
            return pd.DataFrame(index=cells["barcode"].tolist())
    except Exception:
        pass

    return pd.DataFrame()


# ────────────────────────────────────────────── batch / GSE-level API ──

def build_metadata_gse(
    gse_id: str,
    work_dir: Optional[Path] = None,
    include_tier2: bool = True,
    include_tier3: bool = True,
) -> Dict[str, MetadataResult]:
    """Build metadata for all GSMs in a GSE.

    Args:
        gse_id: GEO series accession.
        work_dir: Working directory for downloads.
        include_tier2: Whether to attempt Tier 2 extraction.
        include_tier3: Whether to fetch Tier 3 descriptions.

    Returns:
        ``{gsm_id: MetadataResult}`` for each GSM in the series.
    """
    soft_meta = load_soft_metadata(gse_id)
    if not soft_meta:
        return {}

    gsm_ids = list(soft_meta.get("samples", {}).keys())
    results = {}

    for gsm_id in gsm_ids:
        results[gsm_id] = build_metadata(
            gse_id, gsm_id,
            work_dir=work_dir,
            include_tier2=include_tier2,
            include_tier3=include_tier3,
        )

    n_success = sum(1 for r in results.values() if r.status == "success")
    n_partial = sum(1 for r in results.values() if r.status == "partial")
    logger.info(
        "%s: %d GSMs — %d success, %d partial, %d failed",
        gse_id, len(results), n_success, n_partial,
        len(results) - n_success - n_partial,
    )
    return results


def build_metadata_batch(
    gse_ids: List[str],
    work_dir: Optional[Path] = None,
    include_tier2: bool = True,
    include_tier3: bool = True,
) -> Dict[str, Dict[str, MetadataResult]]:
    """Build metadata for multiple GSEs.

    Args:
        gse_ids: List of GEO series accessions.
        work_dir: Working directory for downloads.
        include_tier2: Whether to attempt Tier 2 extraction.
        include_tier3: Whether to fetch Tier 3 descriptions.

    Returns:
        ``{gse_id: {gsm_id: MetadataResult}}`` nested dict.
    """
    all_results = {}
    for gse_id in gse_ids:
        logger.info("Processing GSE %s (%d/%d)", gse_id, len(all_results) + 1, len(gse_ids))
        all_results[gse_id] = build_metadata_gse(
            gse_id,
            work_dir=work_dir,
            include_tier2=include_tier2,
            include_tier3=include_tier3,
        )
    return all_results
