"""sc-geo catalog module: GEO dataset discovery and catalog building.

This module provides functions for discovering and cataloging single-cell
RNA-seq datasets from NCBI GEO with SRA information and protocol inference.
"""

# High-level API
from scgeo.catalog.api import (
    build_catalog,
    save_catalog,
    load_catalog,
    filter_catalog,
    get_catalog_stats,
)

# Discovery functions
from scgeo.catalog.discovery import (
    discover_single_cell_series,
    esearch_paginated,
    DEFAULT_SINGLE_CELL_QUERY,
)

# Metadata extraction
from scgeo.catalog.metadata import (
    fetch_series_metadata,
    extract_gsm_samples,
)

# SOFT file parsing
from scgeo.catalog.soft import (
    fetch_soft_metadata,
    parse_soft_file,
    download_soft,
)

# SRA queries
from scgeo.catalog.sra import (
    fetch_sra_runinfo,
    fetch_ena_batch,
)

# Protocol inference
from scgeo.catalog.inference import (
    infer_protocol,
    infer_protocols,
)

# ENCODE discovery
from scgeo.catalog.encode import (
    discover_encode_experiments,
    parse_encode_experiments,
    build_encode_catalog,
)

__all__ = [
    # High-level API
    "build_catalog",
    "save_catalog",
    "load_catalog",
    "filter_catalog",
    "get_catalog_stats",
    # Discovery
    "discover_single_cell_series",
    "esearch_paginated",
    "DEFAULT_SINGLE_CELL_QUERY",
    # Metadata
    "fetch_series_metadata",
    "extract_gsm_samples",
    # SOFT parsing
    "fetch_soft_metadata",
    "parse_soft_file",
    "download_soft",
    # SRA queries
    "fetch_sra_runinfo",
    "fetch_ena_batch",
    # Protocol inference
    "infer_protocol",
    "infer_protocols",
    # ENCODE
    "discover_encode_experiments",
    "parse_encode_experiments",
    "build_encode_catalog",
]
