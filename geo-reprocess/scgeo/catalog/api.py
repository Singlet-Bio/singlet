"""
High-level catalog building API.

Provides functions to build complete GEO single-cell catalogs by orchestrating
discovery, metadata extraction, and enrichment stages.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

from scgeo.catalog.discovery import discover_single_cell_series, esearch_paginated
from scgeo.catalog.metadata import fetch_series_metadata, extract_gsm_samples
from scgeo.catalog.soft import fetch_soft_metadata
from scgeo.catalog.sra import fetch_sra_runinfo
from scgeo.catalog.inference import infer_protocols
from scgeo.catalog.encode import build_encode_catalog
from scgeo.config import get_config

logger = logging.getLogger(__name__)


def build_catalog(
    query: Optional[str] = None,
    output_file: Optional[Path] = None,
    include_metadata: bool = True,
    include_soft: bool = True,
    include_sra: bool = True,
    include_protocols: bool = True,
    config = None
) -> pd.DataFrame:
    """Build a complete GEO single-cell catalog.
    
    Pipeline stages:
    1. Discovery: Find single-cell series via ESearch
    2. Metadata: Fetch series and sample metadata via ESummary
    3. SOFT: Download and parse SOFT files for detailed metadata
    4. SRA: Query ENA for SRA run information and FASTQ URLs
    5. Protocol: Infer single-cell protocols from metadata
    
    Args:
        query: Custom NCBI search query (default: single-cell query)
        output_file: Output CSV/parquet file path (optional)
        include_metadata: Fetch basic metadata (default: True)
        include_soft: Include SOFT file parsing (default: True)
        include_sra: Include SRA RunInfo queries (default: True)
        include_protocols: Include protocol inference (default: True)
        config: Configuration object
        
    Returns:
        DataFrame with catalog data
    """
    if config is None:
        config = get_config()
    
    logger.info("Building GEO single-cell catalog...")
    
    # Stage 1: Discovery
    logger.info("=" * 60)
    logger.info("Stage 1: Discovering single-cell series")
    if query:
        discovery_result = esearch_paginated(
            db="gds",
            term=query,
            batch_size=500
        )
    else:
        discovery_result = discover_single_cell_series()
    
    uids = discovery_result["uids"]
    logger.info(f"✓ Found {len(uids):,} series")
    
    if not include_metadata:
        # Return minimal catalog with just UIDs
        df = pd.DataFrame({
            "gds_uid": uids,
            "discovery_date": pd.Timestamp.now()
        })
        
        if output_file:
            save_catalog(df, output_file)
        
        return df
    
    # Stage 2: Metadata
    logger.info("=" * 60)
    logger.info("Stage 2: Fetching series metadata")
    series_metadata = fetch_series_metadata(uids, batch_size=100)
    
    # Extract individual samples
    samples = extract_gsm_samples(series_metadata)
    df = pd.DataFrame(samples)
    logger.info(f"✓ Extracted {len(df):,} samples from {len(series_metadata):,} series")
    
    # Stage 3: SOFT files
    if include_soft:
        logger.info("=" * 60)
        logger.info("Stage 3: Parsing SOFT files")
        gse_ids = df["gse_id"].unique().tolist()
        
        soft_cache_dir = config.paths.catalog_dir / "soft_cache"
        df_soft = fetch_soft_metadata(
            gse_ids,
            cache_dir=soft_cache_dir,
            max_workers=16
        )
        
        # Merge with existing data
        df = df.merge(
            df_soft,
            on=["gse_id", "gsm_id"],
            how="left",
            suffixes=("", "_soft")
        )
        logger.info(f"✓ Merged SOFT metadata: {df.shape}")
    
    # Stage 4: SRA RunInfo
    if include_sra and "srx_accession" in df.columns:
        logger.info("=" * 60)
        logger.info("Stage 4: Querying SRA RunInfo")
        
        # Get unique SRX and BioProjects
        srx_list = df["srx_accession"].dropna().unique().tolist()
        bioproject_list = None
        if "bioproject" in df.columns:
            bioproject_list = df["bioproject"].dropna().unique().tolist()
        
        df_sra = fetch_sra_runinfo(
            srx_accessions=srx_list,
            bioprojects=bioproject_list,
            concurrency=30
        )
        
        if not df_sra.empty:
            # Aggregate runs per SRX
            df_sra_agg = df_sra.groupby("experiment_accession").agg({
                "run_accession": lambda x: ";".join(sorted(x.unique())),
                "library_strategy": "first",
                "library_selection": "first",
                "library_source": "first",
                "library_layout": "first",
                "instrument_platform": "first",
                "instrument_model": "first",
                "read_count": "sum",
                "base_count": "sum",
                "tax_id": "first",
                "scientific_name": "first",
                "fastq_ftp": "first",
                "fastq_md5": "first",
            }).reset_index()
            
            df_sra_agg = df_sra_agg.rename(columns={"experiment_accession": "srx_accession"})
            
            # Merge with catalog
            df = df.merge(
                df_sra_agg,
                on="srx_accession",
                how="left",
                suffixes=("", "_sra")
            )
            logger.info(f"✓ Merged SRA metadata: {df.shape}")
        else:
            logger.warning("No SRA data retrieved")
    
    # Stage 5: Protocol inference
    if include_protocols:
        logger.info("=" * 60)
        logger.info("Stage 5: Inferring protocols")
        df = infer_protocols(df)
        logger.info(f"✓ Protocol inference complete")
    
    logger.info("=" * 60)
    logger.info(f"✓ Catalog complete: {len(df):,} samples from {df['gse_id'].nunique():,} series")
    
    # Save if output specified
    if output_file:
        save_catalog(df, output_file)
    
    return df


def save_catalog(df: pd.DataFrame, output_file: Path):
    """Save catalog to file (CSV, parquet, or JSON).
    
    Args:
        df: Catalog DataFrame
        output_file: Output file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    ext = output_file.suffix.lower()
    
    if ext == ".parquet":
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved catalog to {output_file} (parquet)")
    elif ext == ".csv":
        df.to_csv(output_file, index=False)
        logger.info(f"Saved catalog to {output_file} (CSV)")
    elif ext == ".json":
        df.to_json(output_file, orient="records", indent=2)
        logger.info(f"Saved catalog to {output_file} (JSON)")
    else:
        # Default to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved catalog to {output_file} (CSV)")


def load_catalog(input_file: Path) -> pd.DataFrame:
    """Load catalog from file.
    
    Args:
        input_file: Input file path (.csv, .parquet, or .json)
        
    Returns:
        Catalog DataFrame
    """
    input_file = Path(input_file)
    ext = input_file.suffix.lower()
    
    if ext == ".parquet":
        df = pd.read_parquet(input_file)
    elif ext == ".csv":
        df = pd.read_csv(input_file)
    elif ext == ".json":
        df = pd.read_json(input_file, orient="records")
    else:
        # Try CSV by default
        df = pd.read_csv(input_file)
    
    logger.info(f"Loaded catalog: {len(df):,} records from {input_file}")
    return df


def filter_catalog(
    df: pd.DataFrame,
    organisms: Optional[List[str]] = None,
    min_samples: Optional[int] = None,
    max_samples: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> pd.DataFrame:
    """Filter catalog by various criteria.
    
    Args:
        df: Catalog DataFrame
        organisms: List of organisms to include (e.g., ["Homo sapiens", "Mus musculus"])
        min_samples: Minimum number of samples per series
        max_samples: Maximum number of samples per series
        date_from: Include only series submitted after this date (YYYY-MM-DD)
        date_to: Include only series submitted before this date (YYYY-MM-DD)
        
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if organisms:
        filtered = filtered[filtered["organism"].isin(organisms)]
        logger.info(f"Filtered by organism: {len(filtered):,} samples remain")
    
    if min_samples is not None or max_samples is not None:
        # Count samples per GSE
        gse_counts = filtered.groupby("gse_id").size()
        
        if min_samples is not None:
            valid_gses = gse_counts[gse_counts >= min_samples].index
            filtered = filtered[filtered["gse_id"].isin(valid_gses)]
            logger.info(f"Filtered by min_samples={min_samples}: {len(filtered):,} samples remain")
        
        if max_samples is not None:
            valid_gses = gse_counts[gse_counts <= max_samples].index
            filtered = filtered[filtered["gse_id"].isin(valid_gses)]
            logger.info(f"Filtered by max_samples={max_samples}: {len(filtered):,} samples remain")
    
    if date_from:
        filtered["submission_date"] = pd.to_datetime(filtered["submission_date"])
        filtered = filtered[filtered["submission_date"] >= pd.to_datetime(date_from)]
        logger.info(f"Filtered by date_from={date_from}: {len(filtered):,} samples remain")
    
    if date_to:
        filtered["submission_date"] = pd.to_datetime(filtered["submission_date"])
        filtered = filtered[filtered["submission_date"] <= pd.to_datetime(date_to)]
        logger.info(f"Filtered by date_to={date_to}: {len(filtered):,} samples remain")
    
    return filtered


def get_catalog_stats(df: pd.DataFrame) -> Dict:
    """Get summary statistics for a catalog.
    
    Args:
        df: Catalog DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_samples": len(df),
        "total_series": df["gse_id"].nunique() if "gse_id" in df.columns else 0,
        "organisms": df["organism"].value_counts().to_dict() if "organism" in df.columns else {},
        "date_range": {
            "earliest": str(df["submission_date"].min()) if "submission_date" in df.columns else None,
            "latest": str(df["submission_date"].max()) if "submission_date" in df.columns else None,
        },
        "samples_per_series": {
            "mean": float(df.groupby("gse_id").size().mean()) if "gse_id" in df.columns else 0,
            "median": float(df.groupby("gse_id").size().median()) if "gse_id" in df.columns else 0,
            "min": int(df.groupby("gse_id").size().min()) if "gse_id" in df.columns else 0,
            "max": int(df.groupby("gse_id").size().max()) if "gse_id" in df.columns else 0,
        }
    }
    
    return stats


def build_unified_catalog(
    include_geo: bool = True,
    include_encode: bool = True,
    output_file: Optional[Path] = None,
    config=None,
) -> pd.DataFrame:
    """Build a unified catalog across all open-access sources.

    Merges GEO and ENCODE into a single DataFrame with a ``source`` column
    to indicate provenance.  Both catalogs share a common schema where
    possible; ENCODE-only columns are preserved.

    Args:
        include_geo: Include GEO single-cell catalog (default: True).
        include_encode: Include ENCODE single-cell catalog (default: True).
        output_file: Output path (.csv / .parquet).
        config: Configuration object.

    Returns:
        Combined DataFrame.
    """
    frames = []

    if include_geo:
        logger.info("Building GEO catalog...")
        df_geo = build_catalog(config=config)
        df_geo["source"] = "GEO"
        frames.append(df_geo)

    if include_encode:
        logger.info("Building ENCODE catalog...")
        df_encode = build_encode_catalog()
        frames.append(df_encode)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(
        f"Unified catalog: {len(df):,} rows from "
        f"{df['source'].value_counts().to_dict()}"
    )

    if output_file:
        save_catalog(df, output_file)

    return df
