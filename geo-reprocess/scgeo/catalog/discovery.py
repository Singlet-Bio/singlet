"""GEO catalog discovery module.

Functions for discovering single-cell RNA-seq datasets on GEO via NCBI E-utilities.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scgeo.utils import get_client

logger = logging.getLogger(__name__)

# Permissive droplet-based single-cell RNA-seq query
# Captures all droplet platforms - Phase 3 inference filters plate-based methods
# No species filtering - all organisms included
DEFAULT_SINGLE_CELL_QUERY = (
    # 10x Genomics (no version required - inference will determine version)
    '("10x genomics"[Description] OR "10x chromium"[Description] OR '
    '"10X Genomics"[Description] OR "10X Chromium"[Description] OR '
    '"chromium single cell"[Description] OR "chromium controller"[Description] OR '
    
    # Drop-seq and variants
    '"Drop-seq"[Description] OR "DropSeq"[Description] OR "drop seq"[Description] OR '
    
    # Generic droplet terms (broad but specific to methodology)
    '"droplet-based"[Description] OR "droplet based"[Description] OR '
    '"droplet sequencing"[Description] OR "microfluidic droplet"[Description] OR '
    '"droplet microfluidic"[Description] OR '
    
    # inDrop
    '"inDrop"[Description] OR "inDrops"[Description] OR "indexing droplet"[Description] OR '
    
    # Seq-Well
    '"Seq-Well"[Description] OR "SeqWell"[Description] OR "seq well"[Description] OR '
    
    # dscRNA-seq
    '"dscRNA-seq"[Description] OR "dscRNA seq"[Description] OR '
    
    # BD Rhapsody
    '"BD Rhapsody"[Description] OR "rhapsody single cell"[Description] OR '
    
    # Parse Biosciences
    '"Parse Biosciences"[Description] OR "Evercode"[Description] OR '
    
    # Combinatorial indexing (sci/SPLiT) - droplet-like
    '"sci-RNA-seq"[Description] OR "sciRNA-seq"[Description] OR "sci RNA seq"[Description] OR '
    '"SPLiT-seq"[Description] OR "split-seq"[Description] OR "split seq"[Description] OR '
    '"combinatorial indexing"[Description] OR '
    
    # ddSEQ / BioRad
    '"ddSEQ"[Description] OR "dd-seq"[Description] OR "Illumina Bio-Rad"[Description] OR '
    
    # Microwell-based
    '"Microwell-seq"[Description] OR "microwell"[Description] OR '
    
    # DNBelab (MGI/BGI)
    '"DNBelab"[Description] OR "DNB-seq"[Description] OR '
    
    # SureCell (Illumina)
    '"SureCell"[Description] OR "sure cell"[Description]) AND '
    
    # Restrict to RNA-seq expression profiling
    '"expression profiling by high throughput sequencing"[DataSet Type] AND '
    
    # Only GSE series
    "gse[Entry Type]"
)


def esearch_paginated(
    query: str = DEFAULT_SINGLE_CELL_QUERY,
    db: str = "gds",
    batch_size: int = 10000,
    api_key: Optional[str] = None,
) -> Tuple[List[str], str, str]:
    """Run ESearch with pagination, returning all UIDs.
    
    Args:
        query: NCBI search query (defaults to single-cell query)
        db: NCBI database (default: "gds" for GEO DataSets)
        batch_size: Records per page
        api_key: NCBI API key (optional, uses config default)
        
    Returns:
        Tuple of (uid_list, webenv, query_key)
        
    Example:
        >>> from scgeo.catalog import esearch_paginated
        >>> uids, webenv, qkey = esearch_paginated()
        >>> print(f"Found {len(uids)} GSE series")
    """
    client = get_client()  # Uses API key from config if not specified
    
    logger.info(f"ESearch db={db}: {query[:80]}...")
    
    # First call: get total count and first batch
    data = client.get_json("esearch.fcgi", {
        "db": db,
        "term": query,
        "retmax": batch_size,
        "retstart": 0,
        "usehistory": "y",
    })
    
    result = data["esearchresult"]
    total = int(result["count"])
    uids = result.get("idlist", [])
    webenv = result.get("webenv", "")
    query_key = result.get("querykey", "")
    
    logger.info(f"  Total hits: {total:,}")
    logger.info(f"  First batch: {len(uids):,} UIDs")
    
    # Paginate if needed
    while len(uids) < total:
        retstart = len(uids)
        logger.info(
            f"  Fetching UIDs {retstart:,}–{min(retstart + batch_size, total):,}..."
        )
        
        data = client.get_json("esearch.fcgi", {
            "db": db,
            "term": query,
            "retmax": batch_size,
            "retstart": retstart,
            "WebEnv": webenv,
            "query_key": query_key,
        })
        
        batch = data["esearchresult"].get("idlist", [])
        if not batch:
            logger.warning(f"  Empty batch at retstart={retstart}, stopping")
            break
        
        uids.extend(batch)
    
    logger.info(f"  Total UIDs retrieved: {len(uids):,}")
    return uids, webenv, query_key


def discover_single_cell_series(
    query: str = DEFAULT_SINGLE_CELL_QUERY,
    output_file: Optional[str] = None,
) -> Dict:
    """Discover all single-cell RNA-seq GSE series on GEO.
    
    Args:
        query: NCBI search query (defaults to comprehensive single-cell query)
        output_file: Optional JSON file to save results
        
    Returns:
        Dict with keys: query, db, count, uids, webenv, query_key, timestamp
        
    Example:
        >>> from scgeo.catalog import discover_single_cell_series
        >>> result = discover_single_cell_series()
        >>> print(f"Found {result['count']} GSE series")
        >>> print(f"First 5 UIDs: {result['uids'][:5]}")
    """
    uids, webenv, query_key = esearch_paginated(query=query)
    
    output = {
        "query": query,
        "db": "gds",
        "count": len(uids),
        "uids": uids,
        "webenv": webenv,
        "query_key": query_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if output_file:
        import json
        from pathlib import Path
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(uids):,} UIDs to {output_path}")
    
    return output


# ============================================================================
# Early True Negative Filtering
# ============================================================================

# True negative keywords (case-insensitive regex patterns)
# NOTE: Do NOT filter out modalities we want to process:
#   - Visium (spatial), scATAC, multiome, CITE-seq, ADT — these are valued
#   - Only filter technologies we truly cannot use
TRUE_NEGATIVE_KEYWORDS = [
    # Spatial technologies we don't process (non-Visium)
    r'\bmerfish\b', r'\bseqfish\b', r'\bslide.?seq\b',
    r'\bstereo.?seq\b', r'\bst.?array\b',
    # Epigenomics (non-ATAC)
    r'\bchip.?seq\b', r'\bmethyl', r'\bbisulfite\b',
    r'\bcut.?run\b', r'\bcut.?tag\b', r'\bdnase.?seq\b',
    # Plate-based methods with strong signals
    r'\bsmart.?seq', r'\bcel.?seq\b', r'\bmars.?seq\b', r'\bicell8\b',
    r'\bquartz.?seq\b',
    # Bulk RNA-seq indicators
    r'\bbulk\s+rna', r'\bpooled\s+rna', r'\bwhole\s+tissue\b',
    # Other technologies we don't process
    r'\bpatch.?seq\b',
]

# Compile patterns once
_TRUE_NEGATIVE_PATTERN = re.compile('|'.join(TRUE_NEGATIVE_KEYWORDS), re.IGNORECASE)


def fetch_series_titles(
    uids: List[str],
    batch_size: int = 500,
    show_progress: bool = True
) -> pd.DataFrame:
    """Fetch series titles and summaries using lightweight ESummary calls.
    
    This is faster than full metadata extraction since we only need minimal
    fields to filter out true negatives early.
    
    Args:
        uids: List of GDS UIDs
        batch_size: UIDs per API call (default: 500, higher than full metadata)
        show_progress: Show progress bar
        
    Returns:
        DataFrame with columns: gds_uid, gse_id, title, summary
        
    Example:
        >>> from scgeo.catalog.discovery import fetch_series_titles
        >>> titles = fetch_series_titles(['200022601', '200022602'])
        >>> print(titles[['gse_id', 'title']])
    """
    from tqdm import tqdm
    
    client = get_client()
    records = []
    
    iterator = range(0, len(uids), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Fetching titles", total=len(uids)//batch_size + 1)
    
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
                doc = result.get(uid, {})
                accession = doc.get("accession", "")
                
                # Only process GSE entries
                if not accession.startswith("GSE"):
                    continue
                
                records.append({
                    "gds_uid": uid,
                    "gse_id": accession,
                    "title": doc.get("title", ""),
                    "summary": doc.get("summary", ""),
                })
        except Exception as e:
            logger.warning(f"Error fetching batch at index {i}: {e}")
            continue
    
    df = pd.DataFrame(records)
    logger.info(f"Fetched titles for {len(df):,} GSE series")
    return df


def filter_true_negatives_early(
    series_df: pd.DataFrame,
    text_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filter out obvious true negatives based on title/summary keywords.
    
    This early filter removes non-droplet protocols BEFORE expensive metadata
    operations (SOFT downloads, SRA queries). It uses regex pattern matching
    on series titles and summaries.
    
    Args:
        series_df: DataFrame with at least one text column (title/summary)
        text_columns: Columns to search (default: ['title', 'summary'])
        
    Returns:
        Filtered DataFrame with true negatives removed
        
    Example:
        >>> from scgeo.catalog.discovery import filter_true_negatives_early
        >>> filtered = filter_true_negatives_early(series_df)
        >>> print(f"Kept {len(filtered)} / {len(series_df)} series")
    """
    if text_columns is None:
        text_columns = ['title', 'summary']
    
    # Combine all text columns
    series_df = series_df.copy()
    series_df['_combined_text'] = ''
    for col in text_columns:
        if col in series_df.columns:
            series_df['_combined_text'] += ' ' + series_df[col].fillna('')
    
    # Apply regex filter (pattern already compiled with re.IGNORECASE)
    series_df['_is_true_negative'] = series_df['_combined_text'].str.contains(
        _TRUE_NEGATIVE_PATTERN,
        na=False
    )
    
    # Count removals
    n_removed = series_df['_is_true_negative'].sum()
    n_kept = len(series_df) - n_removed
    pct_removed = 100 * n_removed / len(series_df)
    
    logger.info(f"Early filter results:")
    logger.info(f"  Input: {len(series_df):,} series")
    logger.info(f"  Removed: {n_removed:,} true negatives ({pct_removed:.1f}%)")
    logger.info(f"  Kept: {n_kept:,} series for further processing")
    
    # Return filtered (non-true-negative) series
    filtered = series_df[~series_df['_is_true_negative']].copy()
    filtered = filtered.drop(columns=['_combined_text', '_is_true_negative'])
    
    return filtered


def discover_and_filter(
    query: str = DEFAULT_SINGLE_CELL_QUERY,
    apply_early_filter: bool = True,
    output_dir: str = "data"
) -> Dict:
    """Complete discovery workflow with optional early filtering.
    
    This combines discovery + early filtering into one function. It saves
    intermediate results to the filesystem.
    
    Args:
        query: NCBI search query
        apply_early_filter: Whether to filter true negatives early
        output_dir: Directory for output files
        
    Returns:
        Dict with keys: discovery, filtered (if apply_early_filter=True)
        
    Example:
        >>> from scgeo.catalog.discovery import discover_and_filter
        >>> result = discover_and_filter(apply_early_filter=True)
        >>> print(f"Filtered: {len(result['filtered'])} series")
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Discovery
    logger.info("=" * 80)
    logger.info("STAGE 1: Discovery")
    logger.info("=" * 80)
    
    discovery_file = output_path / "stage1_discovery.json"
    discovery = discover_single_cell_series(
        query=query,
        output_file=str(discovery_file)
    )
    
    result = {"discovery": discovery}
    
    # Stage 2: Early filtering (optional)
    if apply_early_filter:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Early True Negative Filter")
        logger.info("=" * 80)
        
        # Fetch titles
        series_df = fetch_series_titles(discovery['uids'])
        
        # Apply filter
        filtered_df = filter_true_negatives_early(series_df)
        
        # Save filtered uidsseries
        filtered_file = output_path / "stage2_filtered_series.csv"
        filtered_df.to_csv(filtered_file, index=False)
        logger.info(f"Saved filtered series to {filtered_file}")
        
        result["filtered"] = filtered_df
    
    return result
