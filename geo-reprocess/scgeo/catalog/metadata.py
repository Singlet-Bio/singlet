"""
Metadata extraction module for GEO catalog (Stage 2).

Retrieves structured metadata for GEO series using NCBI ESummary API.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

import pandas as pd

from scgeo.utils.ncbi_client import get_client

logger = logging.getLogger(__name__)


def parse_docsum(uid: str, doc: Dict) -> Optional[Dict]:
    """Parse an ESummary DocSum for a GDS GSE entry.
    
    Args:
        uid: GDS UID
        doc: ESummary document dictionary
        
    Returns:
        Parsed metadata dictionary or None if not a GSE
    """
    accession = doc.get("accession", "")
    
    # Only process GSE entries
    if not accession.startswith("GSE"):
        return None
    
    # Extract sample metadata
    samples = doc.get("samples", [])
    gsm_ids = []
    gsm_titles = []
    for s in samples:
        acc = s.get("accession", "")
        if acc.startswith("GSM"):
            gsm_ids.append(acc)
            gsm_titles.append(s.get("title", ""))
    
    # Extract PubMed IDs
    pubmed_ids = doc.get("pubmedids", [])
    if isinstance(pubmed_ids, list):
        pubmed_str = ";".join(str(p) for p in pubmed_ids)
    else:
        pubmed_str = str(pubmed_ids)
    
    return {
        "gds_uid": uid,
        "gse_id": accession,
        "title": doc.get("title", ""),
        "summary": doc.get("summary", ""),
        "organism": doc.get("taxon", ""),
        "entrytype": doc.get("entrytype", ""),
        "gpl": doc.get("gpl", ""),
        "gse_type": doc.get("gdstype", ""),
        "n_samples": doc.get("n_samples", len(gsm_ids)),
        "submission_date": doc.get("pdat", ""),
        "suppfile": doc.get("suppfile", ""),
        "ftplink": doc.get("ftplink", ""),
        "pubmed_ids": pubmed_str,
        "gsm_ids": ";".join(gsm_ids),
        "gsm_titles": ";".join(gsm_titles),
    }


def fetch_series_metadata(
    uids: List[str],
    batch_size: int = 100,
    show_progress: bool = True
) -> List[Dict]:
    """Fetch ESummary metadata for GDS UIDs.
    
    Args:
        uids: List of GDS UIDs
        batch_size: Number of UIDs per batch (default: 100)
        show_progress: Show progress bar
        
    Returns:
        List of parsed metadata dictionaries
    """
    client = get_client()
    all_records = []
    
    iterator = range(0, len(uids), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Fetching series metadata")
    
    for i in iterator:
        batch = uids[i:i + batch_size]
        uid_str = ",".join(batch)
        
        try:
            data = client.get_json("esummary.fcgi", {
                "db": "gds",
                "id": uid_str,
            })
            
            result = data.get("result", {})
            for uid in batch:
                doc = result.get(uid)
                if doc is None:
                    logger.warning(f"No DocSum for UID {uid}")
                    continue
                    
                parsed = parse_docsum(uid, doc)
                if parsed:
                    all_records.append(parsed)
        
        except Exception as e:
            logger.error(f"Failed to fetch batch at index {i}: {e}")
            continue
    
    logger.info(f"Retrieved metadata for {len(all_records):,} series")
    return all_records


def extract_gsm_samples(metadata_records: List[Dict]) -> List[Dict]:
    """Extract individual GSM samples from series metadata.
    
    Args:
        metadata_records: List of series metadata dictionaries
        
    Returns:
        List of GSM sample dictionaries with GSE context
    """
    samples = []
    
    for record in metadata_records:
        gse_id = record["gse_id"]
        gsm_ids = record.get("gsm_ids", "").split(";")
        gsm_titles = record.get("gsm_titles", "").split(";")
        
        # Pad titles if missing
        while len(gsm_titles) < len(gsm_ids):
            gsm_titles.append("")
        
        for gsm_id, title in zip(gsm_ids, gsm_titles):
            if not gsm_id.strip():
                continue
            
            samples.append({
                "gsm_id": gsm_id.strip(),
                "gse_id": gse_id,
                "gsm_title": title.strip(),
                "gse_title": record["title"],
                "organism": record["organism"],
                "submission_date": record["submission_date"],
            })
    
    logger.info(f"Extracted {len(samples):,} GSM samples from {len(metadata_records):,} series")
    return samples


# ============================================================================
# Optimized Parallel Metadata Extraction for HPC
# ============================================================================

def fetch_series_metadata_chunk(
    gse_ids: List[str],
    batch_size: int = 100,
    output_file: Optional[str] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """Fetch metadata for a chunk of GSE series (HPC-optimized).
    
    This function is designed for use in SLURM array jobs where each task
    processes a chunk of series. It saves results to a file so they can be
    combined later.
    
    Args:
        gse_ids: List of GSE accessions (e.g., ['GSE12345', 'GSE67890'])
        batch_size: UIDs per API call (default: 100, NCBI recommended)
        output_file: Optional parquet file to save results
        show_progress: Show progress bar
        
    Returns:
        DataFrame with sample-level metadata (expanded from series)
        
    Example (in SLURM array job):
        >>> from scgeo.catalog.metadata import fetch_series_metadata_chunk
        >>> import os
        >>> task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        >>> gse_chunk = all_gse_ids[task_id*500:(task_id+1)*500]
        >>> df = fetch_series_metadata_chunk(
        ...     gse_chunk,
        ...     output_file=f'metadata_chunk_{task_id}.parquet'
        ... )
    """
    # Convert GSE IDs to UIDs via ESearch
    client = get_client()
    uids = []
    
    # Batch ESearch queries (100 GSEs at a time)
    for i in range(0, len(gse_ids), 100):
        batch = gse_ids[i:i+100]
        query = " OR ".join([f"{gse}[Accession]" for gse in batch])
        
        try:
            data = client.get_json("esearch.fcgi", {
                "db": "gds",
                "term": query,
            })
            batch_uids = data["esearchresult"].get("idlist", [])
            uids.extend(batch_uids)
        except Exception as e:
            logger.warning(f"ESearch failed for batch: {e}")
            continue
    
    logger.info(f"Found {len(uids)} UIDs for {len(gse_ids)} GSE IDs")
    
    # Fetch metadata
    metadata_records = fetch_series_metadata(
        uids,
        batch_size=batch_size,
        show_progress=show_progress
    )
    
    # Extract samples
    samples = extract_gsm_samples(metadata_records)
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Save if requested
    if output_file:
        from pathlib import Path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info(f"Saved {len(df)} samples to {output_path}")
    
    return df


def combine_metadata_chunks(
    chunk_dir: str,
    output_file: str,
    pattern: str = "metadata_chunk_*.parquet"
) -> pd.DataFrame:
    """Combine metadata chunks from parallel HPC job.
    
    Args:
        chunk_dir: Directory containing chunk files
        output_file: Output path for combined parquet
        pattern: Glob pattern for chunk files
        
    Returns:
        Combined DataFrame
        
    Example:
        >>> from scgeo.catalog.metadata import combine_metadata_chunks
        >>> df = combine_metadata_chunks(
        ...     'data/stage3_chunks',
        ...     'data/stage3_metadata_full.parquet'
        ... )
    """
    from pathlib import Path
    
    chunk_path = Path(chunk_dir)
    chunk_files = sorted(chunk_path.glob(pattern))
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_dir} matching {pattern}")
    
    logger.info(f"Combining {len(chunk_files)} chunk files...")
    
    chunks = []
    for chunk_file in chunk_files:
        try:
            df = pd.read_parquet(chunk_file)
            chunks.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {chunk_file}: {e}")
            continue
    
    combined = pd.concat(chunks, ignore_index=True)
    
    # Save combined file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)
    
    logger.info(f"Combined {len(combined):,} samples from {len(chunks)} chunks")
    logger.info(f"Saved to {output_path}")
    
    return combined
