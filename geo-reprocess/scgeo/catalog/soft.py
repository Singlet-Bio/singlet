"""
SOFT file download and parsing for GEO series.

Handles downloading SOFT files from GEO FTP and extracting series-level
and sample-level metadata including SRA accessions and detailed characteristics.
"""
import gzip
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from scgeo.utils.ncbi_client import get_client

logger = logging.getLogger(__name__)


def geo_soft_url(gse_id: str) -> str:
    """Construct FTP URL for a GSE family SOFT file."""
    prefix = gse_id[:-3] + "nnn"
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{prefix}/{gse_id}/soft/{gse_id}_family.soft.gz"
    )


def download_soft(
    gse_id: str,
    cache_dir: Path,
    max_retries: int = 6,
    base_backoff: float = 10.0,
) -> Optional[Path]:
    """Download SOFT file if not cached. Returns path or None on failure.

    Retries on 403/429/5xx with exponential backoff up to ~5 minutes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{gse_id}_family.soft.gz"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = geo_soft_url(gse_id)
    client = get_client()

    for attempt in range(max_retries):
        try:
            resp = client.session.get(url, timeout=60)

            if resp.status_code in (403, 429):
                wait = min(base_backoff * (2 ** attempt), 300)
                logger.warning(
                    f"{gse_id}: HTTP {resp.status_code}, backoff {wait:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = min(base_backoff * (2 ** attempt), 300)
                logger.warning(
                    f"{gse_id}: HTTP {resp.status_code}, backoff {wait:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            logger.debug(f"Downloaded SOFT for {gse_id}")
            return out_path

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            wait = min(base_backoff * (2 ** attempt), 300)
            logger.warning(
                f"{gse_id}: {type(e).__name__}, backoff {wait:.0f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"Failed to download SOFT for {gse_id}: {e}")
            return None

    logger.error(f"Exhausted retries for {gse_id}")
    return None


def parse_soft_file(path: Path) -> dict:
    """
    Parse a SOFT family file and extract key metadata.
    Returns dict with 'series' and 'samples' sections.
    """
    result = {
        "series": {},
        "samples": {},  # GSM_id → {field: value}
    }
    
    try:
        with gzip.open(path, "rt", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"Failed to read {path.name}: {e}")
        return result
    
    current_entity = None  # "SERIES" or "SAMPLE"
    current_id = None
    current_data = {}
    
    for line in lines:
        line = line.rstrip("\n")
        
        # Entity boundaries
        if line.startswith("^SERIES"):
            # Save previous entity
            if current_entity == "SAMPLE" and current_id:
                result["samples"][current_id] = current_data
            current_entity = "SERIES"
            current_id = line.split("=")[-1].strip() if "=" in line else ""
            current_data = {}
            continue
        elif line.startswith("^SAMPLE"):
            if current_entity == "SERIES":
                result["series"] = current_data
            elif current_entity == "SAMPLE" and current_id:
                result["samples"][current_id] = current_data
            current_entity = "SAMPLE"
            current_id = line.split("=")[-1].strip() if "=" in line else ""
            current_data = {}
            continue
        elif line.startswith("^PLATFORM") or line.startswith("!dataset_table_begin"):
            # Save last entity before platform section
            if current_entity == "SERIES":
                result["series"] = current_data
            elif current_entity == "SAMPLE" and current_id:
                result["samples"][current_id] = current_data
            current_entity = None
            continue
        
        # Parse field = value lines
        if line.startswith("!") and "=" in line:
            key, _, val = line.partition("=")
            key = key.strip().lstrip("!")
            val = val.strip()
            
            # Accumulate multi-valued fields
            if key in current_data:
                existing = current_data[key]
                if isinstance(existing, list):
                    existing.append(val)
                else:
                    current_data[key] = [existing, val]
            else:
                current_data[key] = val
    
    # Save last entity
    if current_entity == "SERIES":
        result["series"] = current_data
    elif current_entity == "SAMPLE" and current_id:
        result["samples"][current_id] = current_data
    
    return result


def _flatten_field(val) -> str:
    """Convert field value (str or list) to semicolon-separated string."""
    if isinstance(val, list):
        return " ;; ".join(str(v) for v in val)
    return str(val) if val else ""


def _classify_license(series: dict) -> str:
    """Classify redistribution license from SOFT series metadata.

    Returns one of: public_domain, cc_by, cc_by_sa, cc_by_nc,
    cc_by_nc_sa, cc0, restricted, unknown.
    """
    dua = _flatten_field(series.get("Series_data_use_agreement", "")).lower()
    lic = _flatten_field(series.get("Series_license", "")).lower()
    combined = f"{dua} {lic}"

    if not combined.strip():
        # NCBI/INSDC default: unrestricted public data
        return "public_domain"

    if "cc0" in combined or "public domain" in combined:
        return "cc0"
    if "cc-by-nc-sa" in combined or "cc by-nc-sa" in combined:
        return "cc_by_nc_sa"
    if "cc-by-nc" in combined or "cc by-nc" in combined or "noncommercial" in combined:
        return "cc_by_nc"
    if "cc-by-sa" in combined or "cc by-sa" in combined:
        return "cc_by_sa"
    if "cc-by" in combined or "cc by" in combined or "creative commons attribution" in combined:
        return "cc_by"
    if any(kw in combined for kw in ("restricted", "controlled", "dbgap", "no redistribution")):
        return "restricted"

    return "unknown"


def extract_sra_from_relations(relations) -> tuple[str, str]:
    """Extract SRP and BioProject from Series/Sample relations."""
    if not relations:
        return "", ""
    
    if isinstance(relations, str):
        relations = [relations]
    
    srp = ""
    bioproject = ""
    for rel in relations:
        if "SRA:" in rel or "sra?term=" in rel:
            match = re.search(r"(SRP\d+|ERP\d+|DRP\d+)", rel)
            if match:
                srp = match.group(1)
        elif "BioProject:" in rel or "bioproject/" in rel:
            match = re.search(r"(PRJNA\d+|PRJEB\d+|PRJDB\d+)", rel)
            if match:
                bioproject = match.group(1)
    return srp, bioproject


def extract_srx_from_sample_relations(relations) -> str:
    """Extract SRX accession from a sample's relations."""
    if not relations:
        return ""
    if isinstance(relations, str):
        relations = [relations]
    for rel in relations:
        match = re.search(r"(SRX\d+|ERX\d+|DRX\d+)", rel)
        if match:
            return match.group(1)
    return ""


def process_gse_soft(gse_id: str, cache_dir: Path) -> Optional[dict]:
    """Download, parse, and extract metadata for one GSE."""
    path = download_soft(gse_id, cache_dir)
    if path is None:
        return None
    
    parsed = parse_soft_file(path)
    series = parsed["series"]
    samples = parsed["samples"]
    
    # Extract series-level fields
    srp, bioproject = extract_sra_from_relations(
        series.get("Series_relation", "")
    )
    
    series_record = {
        "gse_id": gse_id,
        "overall_design": _flatten_field(series.get("Series_overall_design", "")),
        "contact_name": _flatten_field(series.get("Series_contact_name", "")),
        "contact_institute": _flatten_field(series.get("Series_contact_institute", "")),
        "sra_study": srp,
        "bioproject": bioproject,
        "supplementary_files": _flatten_field(series.get("Series_supplementary_file", "")),
        "last_update_date": _flatten_field(series.get("Series_last_update_date", "")),
        "data_use_agreement": _flatten_field(series.get("Series_data_use_agreement", "")),
        "license": _classify_license(series),
    }
    
    # Extract per-sample fields
    sample_records = []
    for gsm_id, sdata in samples.items():
        srx = extract_srx_from_sample_relations(sdata.get("Sample_relation", ""))
        
        # Characteristics
        chars = sdata.get("Sample_characteristics_ch1", "")
        chars_str = _flatten_field(chars)
        
        sample_records.append({
            "gse_id": gse_id,
            "gsm_id": gsm_id,
            "srx_accession": srx,
            "sample_organism": _flatten_field(sdata.get("Sample_organism_ch1", "")),
            "sample_source": _flatten_field(sdata.get("Sample_source_name_ch1", "")),
            "sample_characteristics": chars_str,
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
        })
    
    return {
        "series": series_record,
        "samples": sample_records,
    }


def fetch_soft_metadata(
    gse_ids: list[str],
    cache_dir: Path,
    max_workers: int = 16,
) -> pd.DataFrame:
    """
    Download and parse SOFT files for multiple GSE series.
    
    Args:
        gse_ids: List of GSE accessions
        cache_dir: Directory for caching downloaded SOFT files
        max_workers: Number of concurrent downloads
        
    Returns:
        DataFrame with combined series and sample metadata
    """
    logger.info(f"Fetching SOFT metadata for {len(gse_ids):,} GSE series")
    
    all_series = []
    all_samples = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_gse_soft, gse, cache_dir): gse 
            for gse in gse_ids
        }
        
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="SOFT download+parse",
        ):
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
    
    logger.info(f"Successfully processed: {len(all_series):,} GSEs")
    logger.info(f"Failed: {len(failed):,} GSEs")
    
    # Build DataFrames
    df_series = pd.DataFrame(all_series)
    df_samples = pd.DataFrame(all_samples)
    
    logger.info(f"Series metadata: {df_series.shape}")
    logger.info(f"Sample metadata: {df_samples.shape}")
    logger.info(f"  Unique GSMs: {df_samples['gsm_id'].nunique():,}")
    logger.info(f"  GSMs with SRX: {(df_samples['srx_accession'] != '').sum():,}")
    
    # Merge series + samples
    df_merged = df_samples.merge(df_series, on="gse_id", how="left")
    
    return df_merged
