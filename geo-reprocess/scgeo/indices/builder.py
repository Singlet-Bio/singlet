"""Index builder for single-cell quantification.

Wraps simpleaf index building for creating splici reference indices
for transcript quantification.
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

from scgeo.config import get_config

logger = logging.getLogger(__name__)


def build_splici_index(
    organism: str,
    index_type: str = "piscem",
    genome_fasta: Optional[Path] = None,
    gtf_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    threads: Optional[int] = None,
    config = None,
) -> Path:
    """
    Build a splici reference index for an organism.
    
    Args:
        organism: Organism name (e.g., "human", "mouse")
        index_type: Index type ("piscem")
        genome_fasta: Path to genome FASTA (if None, expects it to exist)
        gtf_file: Path to GTF annotation (if None, expects it to exist)
        output_dir: Output directory (default: from config)
        threads: Number of threads (default: from config)
        config: Configuration object
        
    Returns:
        Path to built index directory
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
    
    logger.info(f"Building {index_type} splici index for {species_name}")
    
    # Determine paths
    if output_dir is None:
        output_dir = config.paths.index_dir / species_name / index_type
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected reference paths
    ref_dir = config.paths.index_dir / species_name / "references"
    if genome_fasta is None:
        genome_fasta = ref_dir / f"{species_name}.genome.fa"
    if gtf_file is None:
        gtf_file = ref_dir / f"{species_name}.gtf"
    
    genome_fasta = Path(genome_fasta)
    gtf_file = Path(gtf_file)
    
    # Check if references exist
    if not genome_fasta.exists():
        raise FileNotFoundError(
            f"Genome FASTA not found: {genome_fasta}\n"
            f"Run download_references('{organism}') first."
        )
    if not gtf_file.exists():
        raise FileNotFoundError(
            f"GTF annotation not found: {gtf_file}\n"
            f"Run download_references('{organism}') first."
        )
    
    # Build simpleaf index command
    if threads is None:
        threads = config.resources.simpleaf_threads
    
    index_dir = output_dir / "index"
    
    cmd = [
        "simpleaf",
        "index",
        "--output", str(output_dir),
        "--fasta", str(genome_fasta),
        "--gtf", str(gtf_file),
        "--rlen", "91",  # Standard for 10x v2/v3
        "--threads", str(threads),
        "--use-piscem",
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Index building complete")
        logger.debug(result.stdout)
        
        return index_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"simpleaf index failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Failed to build index for {organism}") from e


def check_index_exists(organism: str, index_type: str = "piscem", config=None) -> bool:
    """
    Check if an index exists for an organism.
    
    Args:
        organism: Organism name
        index_type: Index type ("piscem")
        config: Configuration object
        
    Returns:
        True if index exists
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
    index_dir = config.paths.index_dir / species_name / index_type / "index"
    
    return index_dir.exists() and (index_dir / "info.json").exists()


def list_available_indices(config=None) -> dict:
    """
    List all available indices.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict mapping organism → list of index types
    """
    if config is None:
        config = get_config()
    
    indices = {}
    index_root = config.paths.index_dir
    
    if not index_root.exists():
        return indices
    
    for species_dir in index_root.iterdir():
        if not species_dir.is_dir():
            continue
        
        species = species_dir.name
        available = []
        
        for index_type in ["piscem"]:
            index_dir = species_dir / index_type / "index"
            if index_dir.exists():
                available.append(index_type)
        
        if available:
            indices[species] = available
    
    return indices
