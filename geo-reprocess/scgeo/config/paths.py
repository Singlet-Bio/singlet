"""Path resolution for sc-geo package.

Provides environment-aware path configuration with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional


def get_base_dir(env_var: str = "SCGEO_BASE", default: Optional[str] = None) -> Path:
    """Get base directory from environment or default.
    
    Args:
        env_var: Environment variable name
        default: Default path if env var not set
        
    Returns:
        Resolved Path object
    """
    if default is None:
        default = "/mnt/projects/debruinz_project/cellarium"
    
    path_str = os.environ.get(env_var, default)
    return Path(path_str).expanduser().resolve()


class Paths:
    """Path configuration for sc-geo.
    
    All paths can be overridden by environment variables:
    - SCGEO_BASE: Base directory (default: /mnt/projects/debruinz_project/cellarium)
    - SCGEO_WORKSPACE: Workspace directory (default: /mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess)
    - Or set individual paths via SCGEO_CATALOG_DIR, SCGEO_INDEX_DIR, etc.
    
    Attributes:
        project_base: Base directory for all data
        workspace: Workspace directory for scripts
        catalog_dir: Catalog outputs
        pipeline_dir: Pipeline outputs
        index_dir: Reference indices
        reference_dir: Reference genomes/annotations
        log_dir: Log files
        batch_dir: Batch files
        results_dir: Result CSVs
        download_dir: Temporary FASTQ downloads
        quant_dir: Quantification outputs
        failure_dir: Failure JSONs
        af_home: simpleaf home directory
    """
    
    def __init__(
        self,
        project_base: Optional[Path] = None,
        workspace: Optional[Path] = None,
    ):
        # Base directories
        self.project_base = project_base or get_base_dir("SCGEO_BASE")
        self.workspace = workspace or get_base_dir(
            "SCGEO_WORKSPACE",
            default="/mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess"
        )
        
        # Catalog
        self.catalog_dir = Path(os.environ.get(
            "SCGEO_CATALOG_DIR",
            self.project_base / "catalog"
        ))
        
        # Pipeline
        self.pipeline_dir = Path(os.environ.get(
            "SCGEO_PIPELINE_DIR",
            self.project_base / "pipeline"
        ))
        
        # Indices and references
        self.index_dir = Path(os.environ.get(
            "SCGEO_INDEX_DIR",
            self.project_base / "index"
        ))
        self.reference_dir = Path(os.environ.get(
            "SCGEO_REFERENCE_DIR",
            self.project_base / "reference"
        ))
        
        # Logs
        self.log_dir = Path(os.environ.get(
            "SCGEO_LOG_DIR",
            self.project_base / "logs"
        ))
        
        # Pipeline subdirectories
        self.batch_dir = self.pipeline_dir / "batches"
        self.results_dir = self.pipeline_dir / "results"
        self.download_dir = self.pipeline_dir / "downloads"
        self.quant_dir = self.pipeline_dir / "quant"
        self.failure_dir = self.pipeline_dir / "failures"
        
        # simpleaf home
        self.af_home = Path(os.environ.get(
            "ALEVIN_FRY_HOME",
            self.project_base / "af_home"
        ))
    
    def get_index_path(self, species_name: str, assembly: str) -> Path:
        """Get index directory path for a species.
        
        Args:
            species_name: Scientific name (e.g., "Homo sapiens")
            assembly: Assembly name (e.g., "GRCh38")
            
        Returns:
            Path to index directory
        """
        # Special cases for human/mouse
        if species_name == "Homo sapiens" and assembly == "GRCh38":
            return self.index_dir / "human_splici"
        elif species_name == "Mus musculus" and assembly == "GRCm39":
            return self.index_dir / "mouse_splici"
        else:
            # Standard naming: species_name_splici
            safe_name = species_name.lower().replace(" ", "_")
            return self.index_dir / f"{safe_name}_splici"
    
    def ensure_directories(self):
        """Create all necessary directories."""
        dirs = [
            self.catalog_dir,
            self.pipeline_dir,
            self.index_dir,
            self.reference_dir,
            self.log_dir,
            self.batch_dir,
            self.results_dir,
            self.download_dir,
            self.quant_dir,
            self.failure_dir,
            self.af_home,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
