"""High-level API for index building.

Provides convenience functions for downloading references and building indices.
"""
import logging
from pathlib import Path
from typing import Optional

from scgeo.indices.builder import (
    build_splici_index,
    check_index_exists,
    list_available_indices,
)
from scgeo.indices.downloader import (
    download_references,
    check_references_exist,
)
from scgeo.config import get_config

logger = logging.getLogger(__name__)


def build_index(
    organism: str,
    index_type: str = "piscem",
    force_rebuild: bool = False,
    config=None,
) -> Path:
    """
    Build an index for an organism (downloads references if needed).
    
    This is a convenience function that:
    1. Checks if references exist, downloads if not
    2. Checks if index exists, builds if not (or force_rebuild=True)
    3. Returns path to index
    
    Args:
        organism: Organism name (e.g., "human", "mouse")
        index_type: Index type ("piscem")
        force_rebuild: Rebuild even if index exists
        config: Configuration object
        
    Returns:
        Path to index directory
    """
    if config is None:
        config = get_config()
    
    logger.info(f"Setting up {index_type} index for {organism}")
    
    # Check if index already exists
    if not force_rebuild and check_index_exists(organism, index_type, config):
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
        logger.info(f"✓ Index already exists: {index_dir}")
        return index_dir
    
    # Check if references exist
    refs_status = check_references_exist(organism, config)
    if not (refs_status["genome"] and refs_status["gtf"]):
        logger.info("Downloading references...")
        download_references(organism, config=config)
    else:
        logger.info("✓ References already downloaded")
    
    # Build index
    logger.info(f"Building {index_type} index...")
    index_dir = build_splici_index(
        organism=organism,
        index_type=index_type,
        config=config,
    )
    
    logger.info(f"✓ Index ready: {index_dir}")
    return index_dir


def build_indices(
    organisms: list[str],
    index_type: str = "piscem",
    force_rebuild: bool = False,
    config=None,
) -> dict[str, Path]:
    """
    Build indices for multiple organisms.
    
    Args:
        organisms: List of organism names
        index_type: Index type ("piscem")
        force_rebuild: Rebuild even if indices exist
        config: Configuration object
        
    Returns:
        Dict mapping organism → index path
    """
    indices = {}
    
    for organism in organisms:
        try:
            index_path = build_index(
                organism=organism,
                index_type=index_type,
                force_rebuild=force_rebuild,
                config=config,
            )
            indices[organism] = index_path
        except Exception as e:
            logger.error(f"Failed to build index for {organism}: {e}")
            indices[organism] = None
    
    return indices


def get_index_path(organism: str, index_type: str = "piscem", config=None) -> Optional[Path]:
    """
    Get the path to an index if it exists.
    
    Args:
        organism: Organism name
        index_type: Index type ("piscem")
        config: Configuration object
        
    Returns:
        Path to index or None if not found
    """
    if config is None:
        config = get_config()
    
    if not check_index_exists(organism, index_type, config):
        return None
    
    species_map = {
        "human": "Homo_sapiens",
        "mouse": "Mus_musculus",
        "rat": "Rattus_norvegicus",
        "zebrafish": "Danio_rerio",
        "drosophila": "Drosophila_melanogaster",
        "celegans": "Caenorhabditis_elegans",
    }
    
    species_name = species_map.get(organism.lower(), organism)
    return config.paths.index_dir / species_name / index_type / "index"
