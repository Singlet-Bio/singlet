"""
sc-geo: HPC-native toolkit for single-cell RNA-seq catalog building and FASTQ reprocessing.

Build comprehensive GEO catalogs, download FASTQs from ENA/SRA, quantify with
simpleaf/alevin-fry, and submit batch reprocessing jobs via SLURM.
"""

__version__ = "1.1.0"
__author__ = "Zach DeBruine"
__email__ = "debruinz@gvsu.edu"
__license__ = "MIT"

# High-level API imports
from scgeo.config.api import (
    get_config,
    set_config,
    save_config,
    load_config,
    get_default_config,
)

from scgeo.catalog.discovery import (
    discover_single_cell_series,
    esearch_paginated,
)

from scgeo.catalog.api import (
    build_catalog,
    filter_catalog,
    get_catalog_stats,
)

from scgeo.utils.ncbi_client import get_client

# Pipeline API
from scgeo.pipeline.api import (
    process_sample,
    process_gse,
    process_samples,
)

# Indices API
from scgeo.indices.api import (
    build_index,
    build_indices,
    get_index_path,
)

# SLURM API
from scgeo.slurm.api import (
    submit_batch,
    monitor_job,
    cancel_job,
    list_jobs,
)

# Metadata API
from scgeo.metadata.api import (
    build_metadata,
    build_metadata_gse,
    build_metadata_batch,
    classify_supplementary_files,
    MetadataResult,
)

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Config API
    "get_config",
    "set_config", 
    "save_config",
    "load_config",
    "get_default_config",
    # Catalog API
    "discover_single_cell_series",
    "esearch_paginated",
    "build_catalog",
    "filter_catalog",
    "get_catalog_stats",
    # Utils
    "get_client",
    # Pipeline API
    "process_sample",
    "process_gse",
    "process_samples",
    # Indices API
    "build_index",
    "build_indices",
    "get_index_path",
    # SLURM API
    "submit_batch",
    "monitor_job",
    "cancel_job",
    "list_jobs",
    # Metadata API
    "build_metadata",
    "build_metadata_gse",
    "build_metadata_batch",
    "classify_supplementary_files",
    "MetadataResult",
]

