"""Preprocessing tools for building and processing single-cell datasets.

This subpackage wraps the scgeo production pipeline for use within the
singlet Python package. These tools are used internally to process raw
FASTQ data into compressed .1pz files — they are not documented on the
website but are available for power users and infrastructure operators.

Typical workflow::

    from singlet.preprocessing import download_fastq, detect_protocol, quantify, qc

    result = download_fastq("GSM1234567", ena_r1_url="...", output_dir="/tmp")
    protocol = detect_protocol(result.r1_paths[0], result.r2_paths[0])
    quant = quantify(result.r1_paths, result.r2_paths, protocol, "human", "/tmp/quant")
    metrics = qc("/tmp/quant")
"""

from singlet.preprocessing._detect import ProtocolDetection, detect_protocol
from singlet.preprocessing._download import DownloadResult, download_fastq
from singlet.preprocessing._export import export_to_1pz, export_to_spz
from singlet.preprocessing._qc import QCMetrics, run_qc
from singlet.preprocessing._quantify import QuantResult, quantify
from singlet.preprocessing._species import (
    get_species_info,
    get_taxon_id,
    list_supported_species,
)

__all__ = [
    "download_fastq",
    "DownloadResult",
    "detect_protocol",
    "ProtocolDetection",
    "quantify",
    "QuantResult",
    "run_qc",
    "QCMetrics",
    "list_supported_species",
    "get_species_info",
    "get_taxon_id",
    "export_to_1pz",
    "export_to_spz",
]
