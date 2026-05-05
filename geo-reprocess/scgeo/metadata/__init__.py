"""
Author metadata extraction for processed single-cell samples.

Extracts metadata at three tiers:
  - Tier 1: GSM-level characteristics from GEO SOFT files (100% coverage)
  - Tier 2: Cell-level annotations from supplementary files (h5ad, RDS, CSV/TSV)
  - Tier 3: Natural language descriptions from NCBI E-utilities (100% coverage)
"""

from scgeo.metadata.api import (
    build_metadata,
    build_metadata_gse,
    build_metadata_batch,
    classify_supplementary_files,
    MetadataResult,
)
from scgeo.metadata.soft import (
    load_soft_metadata,
    parse_characteristics,
    build_gsm_level_obs,
)
from scgeo.metadata.description import (
    fetch_geo_description,
)
from scgeo.metadata.barcodes import (
    normalize_barcode,
    match_barcodes,
    align_author_metadata,
)
from scgeo.metadata.extract import (
    extract_metadata_from_h5ad,
    extract_metadata_from_tabular,
    extract_metadata_from_loom,
)
from scgeo.metadata.extract_rds import (
    extract_metadata_from_rds,
)
from scgeo.metadata.download import (
    download_supplementary_file,
)
