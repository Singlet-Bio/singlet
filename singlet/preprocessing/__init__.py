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

from singlet.preprocessing._download import download_fastq, DownloadResult
from singlet.preprocessing._detect import detect_protocol, ProtocolDetection
from singlet.preprocessing._quantify import quantify, QuantResult
from singlet.preprocessing._qc import run_qc, QCMetrics
from singlet.preprocessing._species import (
    list_supported_species,
    get_species_info,
    get_taxon_id,
)
from singlet.preprocessing._export import export_to_1pz, export_to_spz

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
