"""
ENA (European Nucleotide Archive) utilities for FASTQ URL validation.

This module provides functions to validate FASTQ URLs via the ENA API,
with parallel processing support for HPC workflows.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ENA API endpoints
ENA_FILEREPORT_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"


async def fetch_ena_filereport(
    session: aiohttp.ClientSession,
    srr_accession: str,
    timeout: int = 10
) -> Optional[Dict]:
    """Fetch ENA filereport for a single SRR accession.
    
    Args:
        session: aiohttp client session
        srr_accession: SRR accession (e.g., 'SRR12345678')
        timeout: Request timeout in seconds
        
    Returns:
        Dict with fastq_ftp, fastq_md5, read_count, etc., or None if failed
    """
    params = {
        "accession": srr_accession,
        "result": "read_run",
        "fields": "run_accession,fastq_ftp,fastq_md5,read_count,base_count",
    }
    
    try:
        async with session.get(
            ENA_FILEREPORT_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                return None
            
            text = await response.text()
            lines = text.strip().split('\n')
            
            if len(lines) < 2:  # No data
                return None
            
            # Parse TSV response (first line is header)
            header = lines[0].split('\t')
            values = lines[1].split('\t')
            
            return dict(zip(header, values))
    
    except Exception as e:
        logger.debug(f"ENA fetch failed for {srr_accession}: {e}")
        return None


async def validate_fastq_url(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = 10
) -> bool:
    """Validate FASTQ URL accessibility via HTTP HEAD request.
    
    Args:
        session: aiohttp client session
        url: FASTQ URL (http://ftp.sra.ebi.ac.uk/...)
        timeout: Request timeout in seconds
        
    Returns:
        True if URL is accessible (status 200), False otherwise
    """
    try:
        # Use HEAD request (faster than GET, only checks headers)
        async with session.head(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True
        ) as response:
            return response.status == 200
    
    except Exception as e:
        logger.debug(f"URL validation failed for {url}: {e}")
        return False


async def validate_sample_urls(
    sample_row: Dict,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    timeout: int = 10
) -> Optional[Dict]:
    """Validate ENA URLs for a single sample.
    
    Args:
        sample_row: Dict with 'gsm_id' and 'srr_id' keys
        session: aiohttp client session
        semaphore: Concurrency limiter
        timeout: Request timeout
        
    Returns:
        sample_row dict with additional 'ena_accessible' field, or None if failed
    """
    async with semaphore:
        # Fetch ENA filereport
        ena_data = await fetch_ena_filereport(
            session,
            sample_row.get('srr_id', ''),
            timeout=timeout
        )
        
        if not ena_data:
            return None
        
        # Parse FASTQ URLs
        fastq_ftp = ena_data.get('fastq_ftp', '')
        if not fastq_ftp:
            return None
        
        urls = fastq_ftp.split(';')
        
        # Validate first URL (if both R1/R2 exist, checking R1 is usually enough)
        first_url = f"http://{urls[0]}" if urls else ""
        
        if not first_url:
            return None
        
        is_accessible = await validate_fastq_url(session, first_url, timeout)
        
        if not is_accessible:
            return None
        
        # Add ENA data to sample row
        result = sample_row.copy()
        result.update({
            'ena_accessible': True,
            'fastq_ftp': fastq_ftp,
            'fastq_md5': ena_data.get('fastq_md5', ''),
            'read_count': ena_data.get('read_count', ''),
            'base_count': ena_data.get('base_count', ''),
        })
        
        return result


async def validate_chunk_async(
    samples: List[Dict],
    max_concurrent: int = 50,
    timeout: int = 10
) -> List[Dict]:
    """Validate ENA URLs for a chunk of samples (async).
    
    Args:
        samples: List of sample dicts with 'srr_id' field
        max_concurrent: Max concurrent requests
        timeout: Request timeout per sample
        
    Returns:
        List of validated sample dicts (only accessible samples)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            validate_sample_urls(sample, session, semaphore, timeout)
            for sample in samples
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None and exceptions
    validated = [r for r in results if r is not None and isinstance(r, dict)]
    
    logger.info(f"Validated {len(validated)}/{len(samples)} samples")
    
    return validated


def validate_ena_urls_chunk(
    samples_df: pd.DataFrame,
    max_concurrent: int = 50,
    timeout: int = 10,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """Validate ENA URLs for a chunk of samples (HPC-optimized).
    
    This function is designed for SLURM array jobs where each task validates
    a chunk of samples in parallel.
    
    Args:
        samples_df: DataFrame with 'gsm_id' and 'srr_id' columns
        max_concurrent: Max concurrent HTTP requests (default: 50)
        timeout: Request timeout in seconds (default: 10)
        output_file: Optional parquet file to save results
        
    Returns:
        DataFrame with only validated samples (accessible ENA URLs)
        
    Example (in SLURM array job):
        >>> from scgeo.io.ena import validate_ena_urls_chunk
        >>> import os
        >>> task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        >>> samples = pd.read_csv('stage5_qc_passed.csv')
        >>> chunk = samples.iloc[task_id*500:(task_id+1)*500]
        >>> validated = validate_ena_urls_chunk(
        ...     chunk,
        ...     output_file=f'validated_chunk_{task_id}.parquet'
        ... )
    """
    # Convert DataFrame to list of dicts
    sample_dicts = samples_df.to_dict('records')
    
    # Run async validation
    validated_dicts = asyncio.run(
        validate_chunk_async(sample_dicts, max_concurrent, timeout)
    )
    
    # Convert back to DataFrame
    validated_df = pd.DataFrame(validated_dicts)
    
    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validated_df.to_parquet(output_path)
        logger.info(f"Saved {len(validated_df)} validated samples to {output_path}")
    
    logger.info(
        f"Validation complete: {len(validated_df)}/{len(samples_df)} "
        f"samples have accessible ENA URLs ({100*len(validated_df)/len(samples_df):.1f}%)"
    )
    
    return validated_df


def combine_validated_chunks(
    chunk_dir: str,
    output_file: str,
    pattern: str = "validated_chunk_*.parquet"
) -> pd.DataFrame:
    """Combine validated chunks from parallel HPC job.
    
    Args:
        chunk_dir: Directory containing chunk files
        output_file: Output path for combined parquet
        pattern: Glob pattern for chunk files
        
    Returns:
        Combined DataFrame with all validated samples
        
    Example:
        >>> from scgeo.io.ena import combine_validated_chunks
        >>> df = combine_validated_chunks(
        ...     'data/stage6_chunks',
        ...     'data/stage6_validated_samples.csv'
        ... )
    """
    chunk_path = Path(chunk_dir)
    chunk_files = sorted(chunk_path.glob(pattern))
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_dir} matching {pattern}")
    
    logger.info(f"Combining {len(chunk_files)} validated chunk files...")
    
    chunks = []
    for chunk_file in chunk_files:
        try:
            df = pd.read_parquet(chunk_file)
            chunks.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {chunk_file}: {e}")
            continue
    
    combined = pd.concat(chunks, ignore_index=True)
    
    # Remove duplicates (in case of overlaps)
    combined = combined.drop_duplicates(subset=['gsm_id'], keep='first')
    
    # Save combined file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.endswith('.csv'):
        combined.to_csv(output_path, index=False)
    else:
        combined.to_parquet(output_path)
    
    logger.info(f"Combined {len(combined):,} unique validated samples from {len(chunks)} chunks")
    logger.info(f"Saved to {output_path}")
    
    return combined
