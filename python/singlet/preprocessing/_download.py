"""Download FASTQ files from ENA/SRA with automatic fallback."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a FASTQ download operation."""

    success: bool = False
    r1_paths: List[Path] = field(default_factory=list)
    r2_paths: List[Path] = field(default_factory=list)
    method: str = ""
    time_s: float = 0.0
    error: str = ""


def _download_parallel_segments(
    url: str,
    dest: Path,
    segments: int = 8,
    timeout: int = 3600,
    retries: int = 3,
) -> Tuple[bool, Optional[str]]:
    """Download file using parallel HTTP byte-range segments."""
    for attempt in range(retries):
        try:
            # HEAD to get Content-Length
            head = subprocess.run(
                ["curl", "-sI", url],
                capture_output=True, text=True, timeout=30,
            )
            content_length = 0
            for line in head.stdout.splitlines():
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":")[1].strip())
                    break

            if content_length == 0:
                # Fall back to single download
                subprocess.run(
                    ["curl", "-sL", "-o", str(dest), url],
                    timeout=timeout, check=True,
                )
                return (True, None)

            # Split into segments
            seg_size = content_length // segments
            seg_files = []
            procs = []

            for i in range(segments):
                start = i * seg_size
                end = content_length - 1 if i == segments - 1 else (i + 1) * seg_size - 1
                seg_path = dest.parent / f"{dest.name}.seg{i}"
                seg_files.append(seg_path)
                proc = subprocess.Popen(
                    ["curl", "-sL", "-o", str(seg_path),
                     "-H", f"Range: bytes={start}-{end}", url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                procs.append(proc)

            # Wait for all segments
            for proc in procs:
                proc.wait(timeout=timeout)

            # Concatenate
            with open(dest, "wb") as out:
                for seg_path in seg_files:
                    with open(seg_path, "rb") as seg:
                        while True:
                            chunk = seg.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                    seg_path.unlink()

            return (True, None)

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(30 * (attempt + 1))
            else:
                return (False, str(e))

    return (False, "Max retries exceeded")


def _convert_ftp_to_https(url: str) -> str:
    """Convert ENA FTP URL to HTTPS."""
    if url.startswith("ftp://"):
        return url.replace("ftp://", "https://", 1)
    return url


def download_from_ena(
    r1_url: str,
    r2_url: Optional[str],
    output_dir: Path,
    sample_id: str,
    segments: int = 8,
) -> DownloadResult:
    """Download FASTQ files from ENA with parallel segments."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_url = _convert_ftp_to_https(r1_url)
    r1_dest = output_dir / f"{sample_id}_R1.fastq.gz"

    # Check cache
    if r1_dest.exists() and r1_dest.stat().st_size > 0:
        result = DownloadResult(success=True, r1_paths=[r1_dest], method="ena_cached")
        if r2_url:
            r2_dest = output_dir / f"{sample_id}_R2.fastq.gz"
            if r2_dest.exists() and r2_dest.stat().st_size > 0:
                result.r2_paths = [r2_dest]
        result.time_s = time.time() - t0
        return result

    ok, err = _download_parallel_segments(r1_url, r1_dest, segments=segments)
    if not ok:
        return DownloadResult(error=f"ENA R1 download failed: {err}", time_s=time.time() - t0)

    result = DownloadResult(success=True, r1_paths=[r1_dest], method="ena_direct")

    if r2_url:
        r2_url = _convert_ftp_to_https(r2_url)
        r2_dest = output_dir / f"{sample_id}_R2.fastq.gz"
        ok, err = _download_parallel_segments(r2_url, r2_dest, segments=segments)
        if not ok:
            return DownloadResult(error=f"ENA R2 download failed: {err}", time_s=time.time() - t0)
        result.r2_paths = [r2_dest]

    result.time_s = time.time() - t0
    return result


def download_from_sra(
    srr_accession: str,
    output_dir: Path,
    threads: int = 4,
) -> DownloadResult:
    """Download FASTQ from SRA using fasterq-dump as fallback."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["fasterq-dump", "--split-3", "-e", str(threads),
             "-O", str(output_dir), srr_accession],
            timeout=7200, check=True,
            capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return DownloadResult(
            error=f"fasterq-dump failed: {e}", time_s=time.time() - t0
        )

    # Compress with pigz
    for fq in output_dir.glob(f"{srr_accession}*.fastq"):
        try:
            subprocess.run(
                ["pigz", "-p", str(threads), str(fq)],
                timeout=1800, check=True,
            )
        except FileNotFoundError:
            subprocess.run(["gzip", str(fq)], timeout=1800, check=True)

    r1 = sorted(output_dir.glob(f"{srr_accession}*_1.fastq.gz"))
    r2 = sorted(output_dir.glob(f"{srr_accession}*_2.fastq.gz"))

    return DownloadResult(
        success=True,
        r1_paths=r1 if r1 else sorted(output_dir.glob(f"{srr_accession}*.fastq.gz")),
        r2_paths=r2,
        method="fasterq_dump",
        time_s=time.time() - t0,
    )


def download_fastq(
    sample_id: str,
    *,
    ena_r1_url: Optional[str] = None,
    ena_r2_url: Optional[str] = None,
    srr_accession: Optional[str] = None,
    output_dir: str | Path = "/tmp/singlet_download",
    prefer_ena: bool = True,
    segments: int = 8,
    sra_threads: int = 4,
) -> DownloadResult:
    """Download FASTQ files with automatic ENA → SRA fallback.

    Parameters
    ----------
    sample_id : str
        Sample identifier (e.g. GSM accession).
    ena_r1_url : str, optional
        ENA URL for R1 FASTQ.
    ena_r2_url : str, optional
        ENA URL for R2 FASTQ.
    srr_accession : str, optional
        SRR accession for SRA fallback.
    output_dir : str or Path
        Output directory for FASTQ files.
    prefer_ena : bool
        Try ENA first (faster parallel downloads).
    segments : int
        Number of parallel download segments for ENA.
    sra_threads : int
        Threads for fasterq-dump.

    Returns
    -------
    DownloadResult
        Download result with paths and metadata.
    """
    out = Path(output_dir)

    # Try ENA first
    if prefer_ena and ena_r1_url:
        result = download_from_ena(ena_r1_url, ena_r2_url, out, sample_id, segments)
        if result.success:
            return result
        logger.warning("ENA download failed for %s, trying SRA", sample_id)

    # SRA fallback
    if srr_accession:
        result = download_from_sra(srr_accession, out, sra_threads)
        if result.success:
            return result

    # If prefer_ena was False but ENA URL exists, try it now
    if not prefer_ena and ena_r1_url:
        result = download_from_ena(ena_r1_url, ena_r2_url, out, sample_id, segments)
        if result.success:
            return result

    return DownloadResult(error="All download methods failed")
