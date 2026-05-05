"""Default configuration for sc-geo package.

Provides a dataclass-based configuration system with sensible defaults
and user-overridable parameters.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


def _sra_dir(subdir: str) -> Path:
    """Return SRA temp/staging path on scratch (if available) or /tmp."""
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / subdir
    return Path("/tmp") / subdir

from .paths import Paths
from .species import SPECIES_REF, ORGANISM_TO_TAXON
from .protocols import (
    PROTOCOL_CHEMISTRY,
    NON_RNA_PROTOCOLS,
    ACCEPTED_STRATEGIES,
    BUILTIN_CHEMISTRIES,
)


@dataclass
class DownloadConfig:
    """Download configuration."""
    segments: int = 4              # Parallel curl segments per file
    retries: int = 5               # Retry attempts per file
    retry_delay: int = 10          # Seconds between retries
    timeout: int = 7200            # Max seconds per file (2 hours for large files)
    verify_md5: bool = True        # Verify MD5 checksums
    max_file_bytes: int = 0        # 0 = no limit
    max_backoff: int = 300         # Max exponential backoff (5 minutes)
    

@dataclass
class SRAConfig:
    """SRA toolkit configuration."""
    fallback_enabled: bool = True   # Enable SRA fallback for missing R2 recovery
    fasterq_dump: str = "fasterq-dump"
    prefetch: str = "prefetch"
    max_size: str = "100G"
    timeout: int = 7200            # 2 hours per SRR
    temp_dir: Path = field(default_factory=lambda: _sra_dir("sra_temp"))
    local_staging: Path = field(default_factory=lambda: _sra_dir("sra_staging"))
    compress_output: bool = True   # Compress FASTQs with pigz after fasterq-dump
    

@dataclass
class CompressionConfig:
    """Compression configuration."""
    pigz_level: int = 1            # Fastest compression
    pigz_threads: int = 8
    fasterq_threads: int = 2       # Match 1-CPU DL job; 8 was oversubscribing


@dataclass
class ResourceConfig:
    """Compute resource configuration."""
    simpleaf_threads: int = 36
    simpleaf_timeout: int = 7200   # 2 hours (large samples with 200M+ reads need >1h)
    max_memory_gb: int = 96        # Max memory for quantification
    gse_timeout: int = 388800      # 4.5 days (SLURM walltime is safety net)
    

@dataclass
class BatchConfig:
    """Batch creation configuration."""
    target_samples: int = 200      # Target samples per batch
    max_gse_per_batch: int = 500
    prefetch_ahead: int = 2        # Prefetch N GSEs while processing


@dataclass
class QCConfig:
    """Quality control thresholds."""
    # Droplet QC
    min_mapping_rate: float = 0.10
    min_cells: int = 10
    min_genes_per_cell: int = 50
    min_unspliced_frac: float = 0.01
    min_cell_counts: int = 100
    
    # Relaxed threshold for protocols with inherently lower gene detection
    # (Drop-seq family: seqwell, dropseq, microwell, etc.)
    low_sensitivity_min_genes_per_cell: int = 20
    
    # Smart-seq QC
    ss_min_genes: int = 500
    ss_min_counts: int = 50000
    
    # Adaptive QC (MAD-based)
    use_adaptive: bool = True
    mad_n_mads: float = 3.0
    mad_min_batch: int = 20
    

@dataclass
class DetectionConfig:
    """Protocol detection configuration."""
    droplet_r1_max: int = 50       # R1 > 50bp → full-length
    r1_10xv2: int = 26
    r1_10xv3: int = 28
    r1_10xv4: int = 28
    r1_dropseq: int = 20
    min_fastq_bytes: int = 1000
    min_reads_for_mapping: int = 500
    min_r2_len_droplet: int = 40
    rlen_long_threshold: int = 120
    

@dataclass
class Kraken2Config:
    """Kraken2 non-host classification configuration."""
    enabled: bool = True
    db: Optional[Path] = None      # Set to PROJECT_BASE/kraken2_db/pluspf by default
    threads: int = 8
    confidence: float = 0.2
    timeout: int = 1800            # 30 min
    min_nonhost_umis: int = 5
    memory_mapping: bool = True    # Use mmap; False = load DB into RAM (needs ~104 GB)


@dataclass
class CleanupConfig:
    """Cleanup configuration."""
    after_qc: bool = True          # Delete FASTQs after QC
    intermediates: bool = True     # Delete af_map/ and af_quant/


@dataclass
class NCBIConfig:
    """NCBI API configuration."""
    api_key: Optional[str] = None  # Set from NCBI_API_KEY env var
    email: str = "debruinz@gvsu.edu"
    tool: str = "sc-geo"
    requests_per_second: int = 10  # With API key


@dataclass
class Config:
    """Main configuration for sc-geo package.
    
    All parameters have sensible defaults but can be overridden by users.
    
    Example:
        >>> from scgeo.config import get_default_config
        >>> config = get_default_config()
        >>> config.qc.min_mapping_rate = 0.2
        >>> config.download.segments = 16
    """
    
    # Path configuration
    paths: Paths = field(default_factory=Paths)
    
    # Component configurations
    download: DownloadConfig = field(default_factory=DownloadConfig)
    sra: SRAConfig = field(default_factory=SRAConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    kraken2: Kraken2Config = field(default_factory=Kraken2Config)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    ncbi: NCBIConfig = field(default_factory=NCBIConfig)
    
    # Species and protocol references (read-only)
    species_ref: dict = field(default_factory=lambda: SPECIES_REF, repr=False)
    organism_to_taxon: dict = field(default_factory=lambda: ORGANISM_TO_TAXON, repr=False)
    protocol_chemistry: dict = field(default_factory=lambda: PROTOCOL_CHEMISTRY, repr=False)
    non_rna_protocols: set = field(default_factory=lambda: NON_RNA_PROTOCOLS, repr=False)
    accepted_strategies: set = field(default_factory=lambda: ACCEPTED_STRATEGIES, repr=False)
    builtin_chemistries: set = field(default_factory=lambda: BUILTIN_CHEMISTRIES, repr=False)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set kraken2 DB path if not specified
        if self.kraken2.db is None:
            self.kraken2.db = self.paths.project_base / "kraken2_db" / "pluspf"
        
        # Set NCBI API key from environment if not specified
        if self.ncbi.api_key is None:
            import os
            self.ncbi.api_key = os.environ.get("NCBI_API_KEY")
            if self.ncbi.api_key is None:
                self.ncbi.api_key = "f23708a84cd8d9ce0cf1f47eeedc3f8fbe09"  # Default
        
        # Adjust NCBI rate limit based on API key
        if self.ncbi.api_key:
            self.ncbi.requests_per_second = 10
        else:
            self.ncbi.requests_per_second = 3
    
    def get_index_path(self, taxon_id: int) -> Optional[Path]:
        """Get index path for a species by taxon ID.
        
        Args:
            taxon_id: NCBI taxonomy ID
            
        Returns:
            Path to splici index directory, or None if species not supported
        """
        info = self.species_ref.get(taxon_id)
        if info is None:
            return None
        return self.paths.get_index_path(info["name"], info["assembly"])
    
    def get_taxon_id(self, organism_name: str) -> Optional[int]:
        """Get taxon ID from organism name (case-insensitive).
        
        Handles semicolon-separated multi-species strings by trying each
        component (e.g. 'blank sample; Homo sapiens; Mus musculus').
        
        Args:
            organism_name: Scientific or common name
            
        Returns:
            Taxon ID or None if not found
        """
        result = self.organism_to_taxon.get(organism_name.lower().strip())
        if result is not None:
            return result
        if ';' in organism_name:
            for part in organism_name.split(';'):
                part = part.strip().lower()
                if part and part not in ('blank sample', 'blank', 'unknown', 'other', 'na', 'n/a'):
                    result = self.organism_to_taxon.get(part)
                    if result is not None:
                        return result
        return None


def get_default_config() -> Config:
    """Get default configuration with all settings.
    
    Returns:
        Config object with default values
    """
    return Config()


# Status codes
STATUS_DOWNLOAD_OK = "downloaded"
STATUS_DOWNLOAD_FAIL = "download_failed"
STATUS_DOWNLOAD_SKIP = "download_skip"
STATUS_DETECT_OK = "protocol_detected"
STATUS_DETECT_FAIL = "unknown_protocol"
STATUS_PROCESS_OK = "processed"
STATUS_PROCESS_FAIL = "process_failed"
STATUS_QC_PASS = "qc_pass"
STATUS_QC_WARN = "qc_warn"
STATUS_QC_FAIL = "qc_fail"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"

# Failure categories
FAIL_NO_URL = "no_fastq_url"
FAIL_WGET_ERROR = "wget_error"
FAIL_SRA_ERROR = "sra_fallback_error"
FAIL_EMPTY_FILE = "empty_or_corrupt_fastq"
FAIL_NO_INDEX = "no_splici_index"
FAIL_NO_T2G = "no_t2g_file"
FAIL_SIMPLEAF_CRASH = "simpleaf_crash"
FAIL_SIMPLEAF_TIMEOUT = "simpleaf_timeout"
FAIL_ZERO_CELLS = "zero_cells"
FAIL_LOW_MAPPING = "low_mapping_rate"
FAIL_LOW_GENES = "low_genes_per_cell"
FAIL_PROTOCOL_UNKNOWN = "cannot_determine_protocol"
FAIL_R1R2_MISMATCH = " r1_r2_count_mismatch"
FAIL_INCOMPATIBLE_READS = "incompatible_read_lengths"
FAIL_TINY_FASTQ = "fastq_too_small"
FAIL_MD5_MISMATCH = "md5_mismatch"
FAIL_SIZE_MISMATCH = "file_size_mismatch"
FAIL_PISCEM_SIGABRT = "piscem_sigabrt"
FAIL_SRA_DEFERRED = "sra_deferred"
FAIL_FASTERQ_NO_OUTPUT = "fasterq_no_output"
FAIL_SINGLE_END_DROPLET = "single_end_droplet"
