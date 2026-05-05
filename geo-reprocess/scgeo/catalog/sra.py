"""
SRA run-level metadata via ENA API.

Retrieves SRR run accessions, library metadata, and FASTQ download URLs
from ENA (European Nucleotide Archive) which mirrors all NCBI SRA data
with better API performance.
"""
import asyncio
import csv
import io
import logging
import sys
import time
from typing import Optional

import aiohttp
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ENA API configuration
ENA_FILEREPORT_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = ",".join([
    "run_accession",
    "experiment_accession",
    "study_accession",
    "secondary_study_accession",
    "sample_accession",
    "secondary_sample_accession",
    "library_name",
    "library_strategy",
    "library_source",
    "library_selection",
    "library_layout",
    "instrument_platform",
    "instrument_model",
    "read_count",
    "base_count",
    "fastq_ftp",
    "fastq_md5",
    "fastq_bytes",
    "submitted_ftp",
    "tax_id",
    "scientific_name",
])


async def fetch_ena_filereport(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    accession: str,
    result_type: str = "read_run",
    max_retries: int = 3,
) -> tuple[str, Optional[pd.DataFrame]]:
    """
    Query ENA filereport for a single accession.
    
    Args:
        session: aiohttp session
        semaphore: Concurrency control
        accession: Study (SRP/PRJNA) or experiment (SRX) accession
        result_type: "read_run" for run-level data
        max_retries: Number of retry attempts
        
    Returns:
        (accession, DataFrame) or (accession, None) on failure
    """
    params = {
        "accession": accession,
        "result": result_type,
        "fields": ENA_FIELDS,
        "format": "tsv",
        "limit": "0",  # no limit
    }
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(
                    ENA_FILEREPORT_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 204:  # No content
                        return accession, None
                    
                    resp.raise_for_status()
                    text = await resp.text()
                    
                    if not text or len(text) < 10:
                        return accession, None
                    
                    # Parse TSV with stdlib csv to avoid pandas C parser
                    # segfaults on certain ENA responses
                    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
                    rows = list(reader)
                    if not rows:
                        return accession, None
                    
                    df = pd.DataFrame(rows)
                    return accession, df
                    
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.warning(f"Timeout for {accession} after {max_retries} retries")
                return accession, None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.warning(f"Failed to fetch {accession}: {e}")
                return accession, None
    
    return accession, None


async def fetch_ena_batch(
    accessions: list[str],
    concurrency: int = 30,
    result_type: str = "read_run",
) -> dict[str, pd.DataFrame]:
    """
    Fetch ENA filereports for multiple accessions in parallel.
    
    Args:
        accessions: List of accessions to query
        concurrency: Maximum concurrent requests
        result_type: "read_run" for run-level data
        
    Returns:
        Dict mapping accession → DataFrame (only successful queries)
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_ena_filereport(session, semaphore, acc, result_type)
            for acc in accessions
        ]
        
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Querying ENA",
        ):
            accession, df = await coro
            if df is not None:
                results[accession] = df
    
    return results


def fetch_sra_runinfo(
    srx_accessions: list[str],
    bioprojects: Optional[list[str]] = None,
    concurrency: int = 30,
) -> pd.DataFrame:
    """
    Fetch SRA run information from ENA API.
    
    Strategy:
    1. Query by BioProject (PRJNA) when available (gets all runs in one query)
    2. Query by SRX for remaining experiments
    
    Args:
        srx_accessions: List of SRX experiment accessions
        bioprojects: Optional list of BioProject accessions to query first
        concurrency: Maximum concurrent requests
        
    Returns:
        DataFrame with run-level metadata
    """
    logger.info(f"Fetching SRA RunInfo for {len(srx_accessions):,} experiments")
    
    all_runs = []
    covered_srx = set()
    
    # Phase 1: Query by BioProject (more efficient) — in batches to limit memory
    if bioprojects:
        unique_bioprojects = list(set(bp for bp in bioprojects if bp))
        logger.info(f"Phase 1: Querying {len(unique_bioprojects):,} BioProjects")
        
        batch_size = 200
        for i in range(0, len(unique_bioprojects), batch_size):
            batch = unique_bioprojects[i : i + batch_size]
            logger.info(
                f"  BP batch {i // batch_size + 1}/"
                f"{(len(unique_bioprojects) + batch_size - 1) // batch_size}: "
                f"{len(batch)} BioProjects"
            )
            
            bioproject_results = asyncio.run(
                fetch_ena_batch(batch, concurrency=concurrency)
            )
            
            for bp, df in bioproject_results.items():
                all_runs.append(df)
                if "experiment_accession" in df.columns:
                    covered_srx.update(df["experiment_accession"].unique())
        
        logger.info(f"  Covered {len(covered_srx):,} experiments via BioProject")
    
    # Phase 2: Query remaining SRX individually — in batches
    remaining_srx = [srx for srx in srx_accessions if srx and srx not in covered_srx]
    
    if remaining_srx:
        logger.info(f"Phase 2: Querying {len(remaining_srx):,} individual SRX")
        
        batch_size = 500
        for i in range(0, len(remaining_srx), batch_size):
            batch = remaining_srx[i : i + batch_size]
            logger.info(
                f"  SRX batch {i // batch_size + 1}/"
                f"{(len(remaining_srx) + batch_size - 1) // batch_size}: "
                f"{len(batch)} experiments"
            )
            
            srx_results = asyncio.run(
                fetch_ena_batch(batch, concurrency=min(concurrency, 20))
            )
            
            for srx, df in srx_results.items():
                all_runs.append(df)
            
            # Brief pause between SRX batches to avoid ENA rate limits
            time.sleep(1)
            sys.stdout.flush()
            sys.stderr.flush()
    
    # Combine all results
    if not all_runs:
        logger.warning("No SRA RunInfo retrieved")
        return pd.DataFrame()
    
    df_combined = pd.concat(all_runs, ignore_index=True)
    
    # Deduplicate runs
    df_combined = df_combined.drop_duplicates(subset=["run_accession"])
    
    logger.info(f"Retrieved {len(df_combined):,} unique runs")
    logger.info(f"  Runs with FASTQ URLs: {df_combined['fastq_ftp'].notna().sum():,}")
    
    return df_combined
