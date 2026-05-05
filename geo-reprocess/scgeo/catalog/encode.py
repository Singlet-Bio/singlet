"""
ENCODE portal discovery module.

Discovers single-cell experiments from the ENCODE Data Portal
(encodeproject.org) which are fully open access with no login required.
Fetches experiment metadata and FASTQ download URLs via the ENCODE REST API.

ENCODE data is NOT deposited in GEO — these are entirely non-overlapping
samples that supplement the GEO catalog.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENCODE_SEARCH_URL = "https://www.encodeproject.org/search/"
ENCODE_BASE_URL = "https://www.encodeproject.org"

# Single-cell assay titles in ENCODE
SC_ASSAY_TITLES = [
    "scRNA-seq",
    "snRNA-seq",
    "snATAC-seq",
    "scATAC-seq",
    "long read scRNA-seq",
    "snMethyl-seq",
]

# Map ENCODE assay titles to our protocol taxonomy
_ASSAY_TO_PROTOCOL = {
    "scRNA-seq": "scRNA",
    "snRNA-seq": "snRNA",
    "snATAC-seq": "snATAC",
    "scATAC-seq": "scATAC",
    "long read scRNA-seq": "scRNA_longread",
    "snMethyl-seq": "snMethyl",
}


def _encode_get(url: str, params: dict = None, timeout: int = 120) -> dict:
    """GET request to ENCODE API with JSON accept header."""
    headers = {"Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def discover_encode_experiments(
    assay_titles: Optional[List[str]] = None,
    status: str = "released",
) -> List[dict]:
    """Discover all single-cell experiments from ENCODE.

    Args:
        assay_titles: Which assay types to query (default: all SC assays).
        status: Experiment status filter (default: released).

    Returns:
        List of experiment JSON objects from ENCODE API.
    """
    if assay_titles is None:
        assay_titles = SC_ASSAY_TITLES

    all_experiments = []
    seen_ids = set()

    for assay in assay_titles:
        logger.info(f"Querying ENCODE for assay_title={assay!r}")
        params = {
            "type": "Experiment",
            "assay_title": assay,
            "status": status,
            "format": "json",
            "limit": "all",
            "field": [
                "@id", "accession", "assay_title", "assay_term_name",
                "biosample_ontology.term_name",
                "biosample_ontology.organ_slims",
                "replicates.library.biosample.organism.scientific_name",
                "replicates.library.construction_platform.term_name",
                "target.label",
                "date_released",
                "lab.title",
                "files.@id",
                "files.file_type",
                "files.output_type",
                "files.href",
                "files.replicate.biological_replicate_number",
                "files.read_count",
                "files.file_size",
                "files.status",
                "files.paired_end",
                "files.paired_with",
                "files.run_type",
                "files.platform.term_name",
            ],
        }

        data = _encode_get(ENCODE_SEARCH_URL, params=params)
        experiments = data.get("@graph", [])
        for exp in experiments:
            acc = exp.get("accession", "")
            if acc and acc not in seen_ids:
                seen_ids.add(acc)
                all_experiments.append(exp)

        logger.info(f"  {assay}: {len(experiments)} experiments")
        time.sleep(0.5)

    logger.info(f"Total unique ENCODE SC experiments: {len(all_experiments)}")
    return all_experiments


def _extract_fastq_files(exp: dict) -> List[dict]:
    """Extract FASTQ file metadata from an experiment object."""
    fastqs = []
    for f in exp.get("files", []):
        if f.get("file_type") != "fastq":
            continue
        if f.get("status") != "released":
            continue
        href = f.get("href", "")
        fastqs.append({
            "file_id": f.get("@id", ""),
            "href": f"{ENCODE_BASE_URL}{href}" if href else "",
            "read_count": f.get("read_count"),
            "file_size": f.get("file_size"),
            "paired_end": f.get("paired_end"),
            "paired_with": f.get("paired_with"),
            "run_type": f.get("run_type", ""),
            "replicate_num": (
                f.get("replicate", {}).get("biological_replicate_number")
                if isinstance(f.get("replicate"), dict) else None
            ),
            "platform": (
                f.get("platform", {}).get("term_name", "")
                if isinstance(f.get("platform"), dict) else ""
            ),
        })
    return fastqs


def _extract_organism(exp: dict) -> str:
    """Extract organism from nested replicates."""
    for rep in exp.get("replicates", []):
        lib = rep.get("library", {})
        biosample = lib.get("biosample", {})
        org = biosample.get("organism", {})
        name = org.get("scientific_name", "")
        if name:
            return name
    return ""


def _extract_platform(exp: dict) -> str:
    """Extract construction platform from nested replicates."""
    for rep in exp.get("replicates", []):
        lib = rep.get("library", {})
        plat = lib.get("construction_platform", {})
        name = plat.get("term_name", "")
        if name:
            return name
    return ""


def parse_encode_experiments(experiments: List[dict]) -> pd.DataFrame:
    """Parse ENCODE experiment JSON into a sample-level DataFrame.

    Each row corresponds to one FASTQ file (analogous to one SRR run in GEO).
    Experiments without released FASTQ files are included as a single row
    with empty file fields.

    Returns:
        DataFrame with columns matching the GEO catalog schema where
        possible, plus ENCODE-specific columns.
    """
    rows = []
    for exp in experiments:
        acc = exp.get("accession", "")
        assay = exp.get("assay_title", "")
        protocol = _ASSAY_TO_PROTOCOL.get(assay, assay)

        biosample = exp.get("biosample_ontology", {})
        if isinstance(biosample, list):
            biosample = biosample[0] if biosample else {}
        tissue = biosample.get("term_name", "")
        organs = biosample.get("organ_slims", [])

        organism = _extract_organism(exp)
        platform = _extract_platform(exp)
        lab = exp.get("lab", {}).get("title", "") if isinstance(exp.get("lab"), dict) else ""
        date_released = exp.get("date_released", "")

        fastqs = _extract_fastq_files(exp)

        if not fastqs:
            rows.append({
                "source": "ENCODE",
                "experiment_accession": acc,
                "assay_title": assay,
                "protocol_inferred": protocol,
                "organism": organism,
                "tissue": tissue,
                "organ_slims": ";".join(organs) if organs else "",
                "construction_platform": platform,
                "lab": lab,
                "date_released": date_released,
                "fastq_url": "",
                "read_count": None,
                "file_size": None,
                "paired_end": "",
                "run_type": "",
                "replicate_num": None,
                "n_fastq_files": 0,
            })
        else:
            for fq in fastqs:
                rows.append({
                    "source": "ENCODE",
                    "experiment_accession": acc,
                    "assay_title": assay,
                    "protocol_inferred": protocol,
                    "organism": organism,
                    "tissue": tissue,
                    "organ_slims": ";".join(organs) if organs else "",
                    "construction_platform": platform,
                    "lab": lab,
                    "date_released": date_released,
                    "fastq_url": fq["href"],
                    "read_count": fq["read_count"],
                    "file_size": fq["file_size"],
                    "paired_end": fq["paired_end"],
                    "run_type": fq["run_type"],
                    "replicate_num": fq["replicate_num"],
                    "n_fastq_files": len(fastqs),
                })

    df = pd.DataFrame(rows)
    logger.info(
        f"Parsed {len(df)} FASTQ rows from "
        f"{df['experiment_accession'].nunique()} experiments"
    )
    return df


def build_encode_catalog(
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Build a complete ENCODE single-cell catalog.

    Pipeline:
    1. Discover experiments via ENCODE REST API
    2. Parse into sample-level DataFrame with FASTQ URLs
    3. Optionally save to disk

    Returns:
        DataFrame with one row per FASTQ file.
    """
    logger.info("=" * 60)
    logger.info("Building ENCODE single-cell catalog")

    experiments = discover_encode_experiments()
    df = parse_encode_experiments(experiments)

    if output_file:
        path = str(output_file)
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info(f"Saved ENCODE catalog to {path}")

    return df
