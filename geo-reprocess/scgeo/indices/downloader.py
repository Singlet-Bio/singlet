"""Reference genome and annotation downloader.

Downloads genome FASTA and GTF annotation files from configured URLs.
"""
import gzip
import logging
import shutil
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from scgeo.config import get_config

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, decompress: bool = False) -> Path:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Output file path
        decompress: If True, decompress gzipped file after download
        
    Returns:
        Path to downloaded (and optionally decompressed) file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {url}")
    
    # Stream download with progress bar
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        
        total_size = int(resp.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.name,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    logger.info(f"Downloaded to {output_path}")
    
    # Decompress if needed
    if decompress and output_path.suffix == '.gz':
        decompressed_path = output_path.with_suffix('')
        logger.info(f"Decompressing {output_path.name}")
        
        with gzip.open(output_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file
        output_path.unlink()
        logger.info(f"Decompressed to {decompressed_path}")
        
        return decompressed_path
    
    return output_path


def download_references(
    organism: str,
    output_dir: Optional[Path] = None,
    config=None,
) -> dict:
    """
    Download genome FASTA and GTF annotation for an organism.
    
    Args:
        organism: Organism name (e.g., "human", "mouse")
        output_dir: Output directory (default: from config)
        config: Configuration object
        
    Returns:
        Dict with paths: {"genome": Path, "gtf": Path}
    """
    if config is None:
        config = get_config()
    
    # Get species mapping
    species_map = {
        "human": "Homo_sapiens",
        "mouse": "Mus_musculus",
        "rat": "Rattus_norvegicus",
        "zebrafish": "Danio_rerio",
        "drosophila": "Drosophila_melanogaster",
        "celegans": "Caenorhabditis_elegans",
    }
    
    species_name = species_map.get(organism.lower(), organism)
    
    # Find species config by name match
    species_config = None
    for taxid, spec in config.species_ref.items():
        if spec["name"] == species_name:
            species_config = spec
            break
    
    if species_config is None:
        raise ValueError(
            f"Organism '{organism}' not found in configuration. "
            f"Available: {', '.join(species_map.keys())}"
        )
    
    logger.info(f"Downloading references for {species_name}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = config.paths.index_dir / species_name / "references"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download genome FASTA
    genome_url = species_config["genome"]
    genome_path = output_dir / f"{species_name}.genome.fa.gz"
    genome_final = output_dir / f"{species_name}.genome.fa"
    
    if genome_final.exists():
        logger.info(f"Genome already exists: {genome_final}")
    else:
        download_file(genome_url, genome_path, decompress=True)
    
    # Download GTF annotation
    gtf_url = species_config["gtf"]
    gtf_path = output_dir / f"{species_name}.gtf.gz"
    gtf_final = output_dir / f"{species_name}.gtf"
    
    if gtf_final.exists():
        logger.info(f"GTF already exists: {gtf_final}")
    else:
        download_file(gtf_url, gtf_path, decompress=True)
    
    logger.info(f"✓ References downloaded to {output_dir}")
    
    return {
        "genome": genome_final,
        "gtf": gtf_final,
    }


def check_references_exist(organism: str, config=None) -> dict:
    """
    Check if references exist for an organism.
    
    Args:
        organism: Organism name
        config: Configuration object
        
    Returns:
        Dict: {"genome": bool, "gtf": bool}
    """
    if config is None:
        config = get_config()
    
    species_map = {
        "human": "Homo_sapiens",
        "mouse": "Mus_musculus",
        "rat": "Rattus_norvegicus",
        "zebrafish": "Danio_rerio",
        "drosophila": "Drosophila_melanogaster",
        "celegans": "Caenorhabditis_elegans",
    }
    
    species_name = species_map.get(organism.lower(), organism)
    ref_dir = config.paths.index_dir / species_name / "references"
    
    return {
        "genome": (ref_dir / f"{species_name}.genome.fa").exists(),
        "gtf": (ref_dir / f"{species_name}.gtf").exists(),
    }
