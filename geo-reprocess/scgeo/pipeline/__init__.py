"""
sc-geo pipeline module: Download, quantify, and QC single-cell samples.

This module provides functions for processing single-cell RNA-seq samples:
- Download FASTQ files from ENA/SRA
- Protocol detection from FASTQ inspection
- Quantification with simpleaf
- Quality control and metrics computation
- Output packaging (MTX format)
"""

# High-level API exports
from scgeo.pipeline.api import (
    process_sample,
    process_gse,
    process_samples,
    SampleResult,
)

# Low-level module exports
from scgeo.pipeline.download import (
    download_sample,
    download_from_ena,
    download_from_sra,
    DownloadResult,
)

from scgeo.pipeline.detect import (
    detect_protocol_from_files,
    infer_protocol,
    get_chemistry_string,
    ProtocolDetection,
)

from scgeo.pipeline.quantify import (
    quantify_simpleaf,
    QuantResult,
)

from scgeo.pipeline.qc import (
    compute_qc_simpleaf,
    check_qc_thresholds,
    QCMetrics,
)

from scgeo.pipeline.kraken2 import (
    classify_nonhost,
    Kraken2Result,
)

from scgeo.pipeline.merge import (
    merge_gse,
    merge_species_group,
    discover_gsms,
    group_by_species,
    MergeReport,
)

__all__ = [
    # High-level API
    "process_sample",
    "process_gse",
    "process_samples",
    "SampleResult",
    # Download
    "download_sample",
    "download_from_ena",
    "download_from_sra",
    "DownloadResult",
    # Detection
    "detect_protocol_from_files",
    "infer_protocol",
    "get_chemistry_string",
    "ProtocolDetection",
    # Quantification
    "quantify_simpleaf",
    "QuantResult",
    # QC
    "compute_qc_simpleaf",
    "check_qc_thresholds",
    "QCMetrics",
    # Kraken2
    "classify_nonhost",
    "Kraken2Result",
    # Merge
    "merge_gse",
    "merge_species_group",
    "discover_gsms",
    "group_by_species",
    "MergeReport",
]

