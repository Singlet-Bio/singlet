"""Protocol detection for single-cell RNA-seq samples."""

from __future__ import annotations

import gzip
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Chemistry strings for simpleaf
CHEMISTRY_MAP = {
    "10xv3": "10xv3",
    "10xv2": "10xv2",
    "10xv3_5p": "10xv3",
    "10xv2_5p": "10xv2",
    "dropseq": "dropseq",
}


@dataclass
class ProtocolDetection:
    """Result of single-cell protocol detection."""

    protocol: str  # "10xv3", "10xv2", "smartseq2", "dropseq", etc.
    mode: str  # "droplet" or "smartseq"
    confidence: str  # "high", "medium", "low"
    reason: str
    r1_len: int = 0
    r2_len: int = 0
    chemistry: Optional[str] = None


def _detect_read_length(fastq_path: Path, num_reads: int = 100) -> int:
    """Detect median read length from first N reads."""
    import statistics

    lengths = []
    opener = gzip.open if str(fastq_path).endswith(".gz") else open

    with opener(fastq_path, "rt") as f:
        for i, line in enumerate(f):
            if i % 4 == 1:  # Sequence line
                lengths.append(len(line.strip()))
            if len(lengths) >= num_reads:
                break

    return int(statistics.median(lengths)) if lengths else 0


def _load_barcode_whitelist() -> Set[str]:
    """Load 10x Chromium barcode whitelist."""
    alevin_home = os.environ.get("ALEVIN_FRY_HOME", "")
    barcodes = set()

    for name in ["3M-february-2018.txt", "737K-august-2016.txt"]:
        path = Path(alevin_home) / name
        if path.exists():
            with open(path) as f:
                barcodes.update(line.strip() for line in f if line.strip())

    return barcodes


def _check_barcode_fraction(
    fastq_path: Path, whitelist: Set[str], n_reads: int = 200, bc_len: int = 16
) -> float:
    """Check fraction of reads whose first bc_len bases match the whitelist."""
    if not whitelist:
        return 0.0

    matches = 0
    total = 0
    opener = gzip.open if str(fastq_path).endswith(".gz") else open

    with opener(fastq_path, "rt") as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                bc = line.strip()[:bc_len]
                if bc in whitelist:
                    matches += 1
                total += 1
            if total >= n_reads:
                break

    return matches / total if total > 0 else 0.0


def _infer_protocol(
    r1_len: int,
    r2_len: int,
    catalog_hint: Optional[str] = None,
) -> ProtocolDetection:
    """Infer protocol from read lengths and optional catalog hint."""
    # Stage 1: Catalog hint takes priority
    if catalog_hint:
        hint = catalog_hint.lower().strip()
        for proto, mode in [
            ("10xv3", "droplet"), ("10xv2", "droplet"),
            ("dropseq", "droplet"), ("smartseq", "smartseq"),
        ]:
            if proto in hint:
                return ProtocolDetection(
                    protocol=proto,
                    mode=mode,
                    confidence="high",
                    reason=f"Catalog hint: {catalog_hint}",
                    r1_len=r1_len,
                    r2_len=r2_len,
                    chemistry=CHEMISTRY_MAP.get(proto),
                )

    # Stage 2: Read-length heuristics
    if 24 <= r1_len <= 30 and r2_len >= 50:
        # Classic 10x: short R1 (barcode+UMI), long R2 (cDNA)
        proto = "10xv3" if r1_len >= 28 else "10xv2"
        return ProtocolDetection(
            protocol=proto, mode="droplet", confidence="high",
            reason=f"R1={r1_len}bp (barcode), R2={r2_len}bp (cDNA)",
            r1_len=r1_len, r2_len=r2_len,
            chemistry=CHEMISTRY_MAP.get(proto),
        )

    if r1_len >= 50 and 24 <= r2_len <= 30:
        # Swapped orientation
        proto = "10xv3" if r2_len >= 28 else "10xv2"
        return ProtocolDetection(
            protocol=proto, mode="droplet", confidence="medium",
            reason=f"Swapped: R1={r1_len}bp (cDNA), R2={r2_len}bp (barcode)",
            r1_len=r1_len, r2_len=r2_len,
            chemistry=CHEMISTRY_MAP.get(proto),
        )

    if r1_len >= 50 and r2_len >= 50:
        return ProtocolDetection(
            protocol="ambiguous", mode="unknown", confidence="low",
            reason=f"Both reads long: R1={r1_len}bp, R2={r2_len}bp",
            r1_len=r1_len, r2_len=r2_len,
        )

    return ProtocolDetection(
        protocol="unknown", mode="unknown", confidence="low",
        reason=f"Unrecognized: R1={r1_len}bp, R2={r2_len}bp",
        r1_len=r1_len, r2_len=r2_len,
    )


def detect_protocol(
    r1_path: str | Path,
    r2_path: Optional[str | Path] = None,
    *,
    catalog_hint: Optional[str] = None,
    num_reads: int = 100,
) -> ProtocolDetection:
    """Detect single-cell protocol from FASTQ files.

    Parameters
    ----------
    r1_path : path
        Path to R1 FASTQ file.
    r2_path : path, optional
        Path to R2 FASTQ file.
    catalog_hint : str, optional
        Protocol hint from GEO metadata (e.g. "10x Chromium 3' v3").
    num_reads : int
        Number of reads to sample for length detection.

    Returns
    -------
    ProtocolDetection
        Detected protocol with confidence.
    """
    r1_path = Path(r1_path)
    r1_len = _detect_read_length(r1_path, num_reads)
    r2_len = _detect_read_length(Path(r2_path), num_reads) if r2_path else 0

    result = _infer_protocol(r1_len, r2_len, catalog_hint)

    # Stage 3: Barcode whitelist fallback for ambiguous cases
    if result.confidence == "low" and result.protocol == "ambiguous":
        whitelist = _load_barcode_whitelist()
        if whitelist:
            r1_frac = _check_barcode_fraction(r1_path, whitelist)
            r2_frac = (
                _check_barcode_fraction(Path(r2_path), whitelist) if r2_path else 0.0
            )

            if r1_frac >= 0.3:
                return ProtocolDetection(
                    protocol="10xv3", mode="droplet", confidence="medium",
                    reason=f"Barcode match: R1={r1_frac:.0%}",
                    r1_len=r1_len, r2_len=r2_len,
                    chemistry="10xv3",
                )
            elif r2_frac >= 0.3:
                return ProtocolDetection(
                    protocol="10xv3", mode="droplet", confidence="medium",
                    reason=f"Barcode match R2 (swapped): R2={r2_frac:.0%}",
                    r1_len=r1_len, r2_len=r2_len,
                    chemistry="10xv3",
                )

    return result


def get_chemistry_string(protocol: str) -> Optional[str]:
    """Map protocol name to simpleaf chemistry string."""
    return CHEMISTRY_MAP.get(protocol.lower())
