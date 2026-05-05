"""
Download module for FASTQ files from ENA/SRA.

Provides high-level functions for downloading single-cell RNA-seq FASTQ files
with automatic fallback strategies and parallel segment downloads.
"""
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def compute_download_timeout(read_count: float, min_timeout: int = 300, max_timeout: int = 7200) -> int:
    """Compute a download timeout scaled by expected file size.

    Heuristic: ~80 bytes per read (compressed FASTQ averaged over R1+R2),
    at ~4 MB/s minimum sustained throughput.  This gives ~250s per GB,
    plus a generous floor.

    Args:
        read_count: Total reads for the sample (used as size proxy).
            0 or negative → use default max_timeout.
        min_timeout: Minimum timeout in seconds (default: 5 min).
        max_timeout: Maximum timeout in seconds (default: 2 hours).

    Returns:
        Timeout in seconds, clamped to [min_timeout, max_timeout].
    """
    if not read_count or read_count <= 0:
        return max_timeout
    # ~80 bytes/read compressed (R1 ~30B + R2 ~50B avg) → estimate GB
    # ~250s/GB at ~4 MB/s sustained + 300s overhead
    estimated_gb = read_count * 80 / 1e9
    timeout = int(300 + estimated_gb * 250)
    return max(min_timeout, min(timeout, max_timeout))


@dataclass
class DownloadResult:
    """Result of a FASTQ download operation.
    
    Attributes:
        success: Whether download succeeded
        r1_paths: List of R1 FASTQ file paths
        r2_paths: List of R2 FASTQ file paths (empty for single-end)
        method: Download method used ("ena_direct", "s3_sra", "s3_sra_pending", "fasterq_dump")
        time_s: Download time in seconds
        error: Error message if failed
        fail_category: Standardized failure category for analysis
        sra_path: Path to a downloaded .sra file awaiting fasterq-dump on the quant node
            (only set when method=="s3_sra_pending")
    """
    success: bool = False
    r1_paths: List[Path] = None
    r2_paths: List[Path] = None
    method: str = ""
    time_s: float = 0.0
    error: str = ""
    fail_category: str = ""
    sra_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.r1_paths is None:
            self.r1_paths = []
        if self.r2_paths is None:
            self.r2_paths = []


def _construct_ena_urls(srr: str) -> Tuple[str, str]:
    """Construct ENA FTP URLs for an SRR accession.

    ENA path convention:
      - Prefix directory: first 6 characters of accession (e.g. "SRR147")
      - Suffix directory (for accessions >9 chars): chars beyond position 9,
        zero-padded to 3 digits (e.g. "038" for SRR14747538)

    Args:
        srr: SRR accession (e.g. "SRR14747538")

    Returns:
        (r1_url, r2_url) — ENA FTP URLs for the paired FASTQ files.
    """
    base = "ftp://ftp.sra.ebi.ac.uk/vol1/fastq"
    prefix = srr[:6]

    if len(srr) > 9:
        suffix = srr[9:].zfill(3)
        path = f"{base}/{prefix}/{suffix}/{srr}"
    else:
        path = f"{base}/{prefix}/{srr}"

    return f"{path}/{srr}_1.fastq.gz", f"{path}/{srr}_2.fastq.gz"


def download_parallel_segments(
    url: str,
    dest: Path,
    segments: int = 8,
    timeout: int = 3600,
    retries: int = 3
) -> Tuple[bool, Optional[str]]:
    """Download file using parallel HTTP/2 byte-range segments.
    
    Achieves ~3-4x speedup vs single connection by using multiple curl processes
    with HTTP byte-range requests.
    
    Args:
        url: Source URL (must support HTTP Range requests)
        dest: Destination file path
        segments: Number of parallel curl processes (default: 8)
        timeout: Timeout per segment in seconds
        retries: Number of retry attempts for failures
        
    Returns:
        (success, error_message)
        
    Example:
        >>> success, error = download_parallel_segments(
        ...     "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123_1.fastq.gz",
        ...     Path("/tmp/SRR123_1.fastq.gz"),
        ...     segments=8
        ... )
    """
    import tempfile
    import os
    
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(retries):
        try:
            # Step 1: Get file size via HEAD request
            head_cmd = ["curl", "-sI", "-L", "--max-time", "30", url]
            head_result = subprocess.run(head_cmd, capture_output=True, text=True, timeout=60)
            
            content_length = 0
            accepts_ranges = False
            for line in head_result.stdout.split("\n"):
                line_lower = line.strip().lower()
                if line_lower.startswith("content-length:"):
                    try:
                        content_length = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                if "accept-ranges: bytes" in line_lower:
                    accepts_ranges = True
            
            # If file is small or server doesn't support ranges, single download
            if content_length < 10_000_000 or not accepts_ranges or segments <= 1:
                cmd = [
                    "curl", "-sL", "-f",
                    "--max-time", str(timeout),
                    "--retry", "3",
                    "-o", str(dest),
                    url,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
                if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                    return True, None
                else:
                    error_msg = result.stderr.strip() or f"curl exit {result.returncode}"
                    if attempt < retries - 1:
                        wait_time = min(5 * (2 ** attempt), 300)  # Cap at 5 minutes
                        logger.warning(f"Download failed (attempt {attempt+1}/{retries}): {error_msg}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    return False, error_msg
            
            # Step 2: Parallel byte-range downloads
            segment_size = content_length // segments
            segment_files = []
            processes = []
            
            tmp_dir = dest.parent / f".dl_segments_{dest.stem}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                for i in range(segments):
                    start_byte = i * segment_size
                    end_byte = (i + 1) * segment_size - 1 if i < segments - 1 else content_length - 1
                    seg_file = tmp_dir / f"seg_{i:03d}"
                    segment_files.append(seg_file)
                    
                    cmd = [
                        "curl", "-sL", "-f",
                        "--max-time", str(timeout),
                        "-r", f"{start_byte}-{end_byte}",
                        "-o", str(seg_file),
                        url,
                    ]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    processes.append(proc)
                
                # Wait for all segments
                all_ok = True
                errors = []
                for i, proc in enumerate(processes):
                    try:
                        _, stderr = proc.communicate(timeout=timeout)
                        if proc.returncode != 0:
                            all_ok = False
                            errors.append(f"segment {i}: exit {proc.returncode}")
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        all_ok = False
                        errors.append(f"segment {i}: timeout")
                
                if not all_ok:
                    error_msg = "; ".join(errors)
                    if attempt < retries - 1:
                        wait_time = min(5 * (2 ** attempt), 300)  # Cap at 5 minutes
                        logger.warning(f"Parallel download failed (attempt {attempt+1}/{retries}): {error_msg}. Retrying...")
                        time.sleep(wait_time)
                        continue
                    return False, error_msg
                
                # Step 3: Concatenate segments to temp file, then atomic rename
                tmp_dest = dest.parent / f".{dest.name}.tmp"
                with open(tmp_dest, 'wb') as outf:
                    for seg_file in segment_files:
                        with open(seg_file, 'rb') as inf:
                            while True:
                                chunk = inf.read(1024 * 1024)  # 1 MB chunks
                                if not chunk:
                                    break
                                outf.write(chunk)
                
                if tmp_dest.exists() and tmp_dest.stat().st_size > 0:
                    tmp_dest.rename(dest)
                    return True, None
                else:
                    tmp_dest.unlink(missing_ok=True)
                    return False, "Concatenated file is empty"
                    
            finally:
                # Cleanup segment files
                import shutil
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                wait_time = min(5 * (2 ** attempt), 300)  # Cap at 5 minutes
                logger.warning(f"Download timeout (attempt {attempt+1}/{retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return False, f"Download timed out after {retries} attempts"
        except Exception as e:
            if attempt < retries - 1:
                wait_time = min(5 * (2 ** attempt), 300)  # Cap at 5 minutes
                logger.warning(f"Download exception (attempt {attempt+1}/{retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return False, f"Download failed after {retries} attempts: {e}"
    
    return False, "Download failed: max retries exceeded"


def download_from_ena(
    r1_url: str,
    r2_url: Optional[str],
    output_dir: Path,
    gsm_id: str,
    config,
    budget_remaining: Optional[int] = None,
) -> DownloadResult:
    """Download FASTQ files from ENA with parallel segments.
    
    Args:
        r1_url: ENA URL for R1 FASTQ (ftp:// or https://)
        r2_url: ENA URL for R2 FASTQ (None for single-end)
        output_dir: Directory to save files
        gsm_id: GSM accession for file naming
        config: Configuration object with download settings
        budget_remaining: Max seconds for this download (caps curl timeouts).
        
    Returns:
        DownloadResult with success status and file paths
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert FTP URLs to HTTPS
    def ftp_to_https(url: str) -> str:
        if not url or str(url) == "nan":
            return ""
        url = str(url).strip()
        if url.startswith("ftp://"):
            return "https://" + url[6:]
        return url
    
    r1_url = ftp_to_https(r1_url)
    r2_url = ftp_to_https(r2_url) if r2_url else None
    
    if not r1_url:
        return DownloadResult(
            success=False,
            error="No R1 URL provided",
            fail_category="no_url",
            time_s=time.time() - t0
        )
    
    # Check if FASTQs already exist (reuse previous downloads)
    r1_dest = output_dir / f"{gsm_id}_R1.fastq.gz"
    r2_dest = output_dir / f"{gsm_id}_R2.fastq.gz"
    
    if r1_dest.exists() and r1_dest.stat().st_size > 0:
        r1_paths = [r1_dest]
        r2_paths = [r2_dest] if r2_dest.exists() and r2_dest.stat().st_size > 0 else []
        elapsed = time.time() - t0
        logger.info(f"Reusing existing FASTQs for {gsm_id}: {len(r1_paths)} R1, {len(r2_paths)} R2")
        return DownloadResult(
            success=True,
            r1_paths=r1_paths,
            r2_paths=r2_paths,
            method="ena_cached",
            time_s=elapsed
        )
    
    def _capped_timeout() -> int:
        """Per-file timeout, capped at remaining budget."""
        base = config.download.timeout
        if budget_remaining is not None:
            remaining = max(30, int(budget_remaining - (time.time() - t0)))
            return min(base, remaining)
        return base

    # Download R1
    logger.info(f"Downloading R1 from ENA: {r1_url}")
    
    r1_success, r1_error = download_parallel_segments(
        r1_url,
        r1_dest,
        segments=config.download.segments,
        timeout=_capped_timeout(),
        retries=config.download.retries
    )
    
    if not r1_success:
        return DownloadResult(
            success=False,
            error=f"R1 download failed: {r1_error}",
            fail_category="ena_download_failed",
            time_s=time.time() - t0
        )
    
    r1_paths = [r1_dest]
    r2_paths = []
    
    # Download R2 if provided
    if r2_url:
        r2_dest = output_dir / f"{gsm_id}_R2.fastq.gz"
        logger.info(f"Downloading R2 from ENA: {r2_url}")
        
        r2_success, r2_error = download_parallel_segments(
            r2_url,
            r2_dest,
            segments=config.download.segments,
            timeout=_capped_timeout(),
            retries=config.download.retries
        )
        
        if r2_success:
            r2_paths = [r2_dest]
        else:
            logger.warning(f"R2 download failed: {r2_error}. Accepting R1-only.")
    
    elapsed = time.time() - t0
    logger.info(f"Download completed in {elapsed:.1f}s: {len(r1_paths)} R1, {len(r2_paths)} R2")
    
    return DownloadResult(
        success=True,
        r1_paths=r1_paths,
        r2_paths=r2_paths,
        method="ena_direct",
        time_s=elapsed
    )


def _resolve_s3_url(srr_accession: str, timeout: int = 30) -> Optional[str]:
    """Query NCBI SDL v2 API to find a public S3 URL for an SRR accession.

    Falls back to the well-known SRA public bucket URL pattern if the API
    response is unparseable.

    Args:
        srr_accession: SRR accession (e.g. "SRR12345678")
        timeout: curl timeout in seconds

    Returns:
        HTTPS URL for the .sra file, or None if unavailable.
    """
    import json

    sdl_url = (
        f"https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve"
        f"?acc={srr_accession}&filetype=sra"
    )
    try:
        result = subprocess.run(
            ["curl", "-sL", "-f", "--max-time", str(timeout), sdl_url],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            # SDL v2 response: {"result": [{..., "files": [{"locations": [...]}]}]}
            for entry in data.get("result", []):
                for f in entry.get("files", []):
                    for loc in f.get("locations", []):
                        if loc.get("service") == "s3" and loc.get("link"):
                            return loc["link"]
    except Exception as exc:
        logger.debug(f"SDL API parse error for {srr_accession}: {exc}")

    # Fallback: well-known public SRA S3 bucket URL (no trailing .sra extension)
    fallback = (
        f"https://sra-pub-run-odp.s3.amazonaws.com/sra"
        f"/{srr_accession}/{srr_accession}"
    )
    logger.debug(f"SDL returned no S3 link for {srr_accession} — using fallback: {fallback}")
    return fallback


def download_from_s3(
    srr_accession: str,
    output_dir: Path,
    config,
    budget_remaining: Optional[int] = None,
) -> DownloadResult:
    """Download .sra from NCBI public S3 via SDL API, then convert to FASTQ.

    Strategy:
    1. Resolve public S3 URL via NCBI SDL API (falls back to well-known pattern)
    2. curl-download the .sra file to node-local $SCRATCH/sra_staging
    3. Run fasterq-dump on the current SLURM compute node (output to output_dir on NFS)
    4. Optionally compress with pigz; clean up .sra file

    The download job runs inside an allocated SLURM compute node so fasterq-dump
    is safe to run here. output_dir is on NFS and accessible to all subsequent jobs.

    Args:
        srr_accession: SRR accession (e.g. "SRR12345678")
        output_dir: Directory into which FASTQ files are written (NFS)
        config: Configuration object (same as download_from_sra)
        budget_remaining: Max wall-clock seconds (caps both curl and fasterq-dump).
            None = no limit.

    Returns:
        DownloadResult with method="s3_sra" and r1_paths/r2_paths populated
    """
    import shutil  # used in error cleanup paths below

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_timeout = config.sra.timeout
    if budget_remaining is not None:
        total_timeout = min(total_timeout, max(60, budget_remaining))

    def _remaining() -> int:
        return max(30, total_timeout - int(time.time() - t0))

    # Step 1: Resolve S3 URL
    s3_url = _resolve_s3_url(srr_accession)
    if not s3_url:
        return DownloadResult(
            success=False,
            error=f"No S3 URL found for {srr_accession}",
            fail_category="s3_no_url",
            time_s=time.time() - t0,
        )
    logger.info(f"S3 URL for {srr_accession}: {s3_url}")

    # Step 2: Download .sra from S3
    config.sra.local_staging.mkdir(parents=True, exist_ok=True)
    prefetch_dir = config.sra.local_staging / srr_accession
    prefetch_dir.mkdir(parents=True, exist_ok=True)
    sra_file = prefetch_dir / f"{srr_accession}.sra"

    if not sra_file.exists():
        dl_timeout = _remaining()
        logger.info(f"Downloading .sra from S3 (timeout={dl_timeout}s): {s3_url}")
        dl_cmd = [
            "curl", "-sL", "-f",
            "--max-time", str(dl_timeout),
            "--speed-limit", "1048576",   # abort if < 1 MB/s ...
            "--speed-time", "60",          # ... for 60 consecutive seconds
            "--retry", "2",
            "--retry-delay", "5",
            "-o", str(sra_file),
            s3_url,
        ]
        try:
            dl_result = subprocess.run(
                dl_cmd, capture_output=True, text=True, timeout=dl_timeout + 30
            )
        except subprocess.TimeoutExpired:
            sra_file.unlink(missing_ok=True)
            shutil.rmtree(prefetch_dir, ignore_errors=True)
            return DownloadResult(
                success=False,
                error="S3 download timed out",
                fail_category="s3_timeout",
                time_s=time.time() - t0,
            )

        if dl_result.returncode != 0 or not sra_file.exists() or sra_file.stat().st_size == 0:
            err = dl_result.stderr.strip()[:500] if dl_result.stderr else f"exit {dl_result.returncode}"
            sra_file.unlink(missing_ok=True)
            shutil.rmtree(prefetch_dir, ignore_errors=True)
            return DownloadResult(
                success=False,
                error=f"S3 curl failed: {err}",
                fail_category="s3_download_failed",
                time_s=time.time() - t0,
            )
        logger.info(
            f"S3 download complete: {sra_file.stat().st_size / 1e9:.2f} GB "
            f"in {time.time() - t0:.0f}s"
        )
    else:
        logger.info(f"Using cached .sra file: {sra_file}")

    # Run fasterq-dump on the SLURM compute node where the download job runs.
    # The .sra file is node-local (/tmp/$SCRATCH) but fasterq-dump output goes
    # to output_dir which is on NFS and accessible by all subsequent jobs.
    remaining = total_timeout - int(time.time() - t0)
    if remaining < 30:
        sra_file.unlink(missing_ok=True)
        shutil.rmtree(prefetch_dir, ignore_errors=True)
        return DownloadResult(
            success=False,
            error="Insufficient time for fasterq-dump after S3 download",
            fail_category="s3_sra_timeout",
            time_s=time.time() - t0,
        )

    config.sra.temp_dir.mkdir(parents=True, exist_ok=True)
    fq_threads = str(getattr(config.compression, 'fasterq_threads', 8))
    fq_cmd = [
        "fasterq-dump",
        str(sra_file),
        "--outdir", str(output_dir),
        "--temp", str(config.sra.temp_dir),
        "--split-3",
        "--threads", fq_threads,
        "--mem", "3g",
    ]
    logger.info(f"Running fasterq-dump on S3 .sra: {sra_file.name} → {output_dir}")
    fq_result = subprocess.run(
        fq_cmd, capture_output=True, text=True, timeout=remaining
    )

    # Clean up the node-local .sra file regardless of fasterq-dump outcome
    try:
        sra_file.unlink(missing_ok=True)
        prefetch_dir.rmdir()
    except OSError:
        pass

    if fq_result.returncode != 0:
        return DownloadResult(
            success=False,
            error=f"fasterq-dump failed: {fq_result.stderr[:500]}",
            fail_category="s3_fasterq_failed",
            time_s=time.time() - t0,
        )

    srr_accession = sra_file.stem  # e.g. SRR12345678
    r1_path = output_dir / f"{srr_accession}_1.fastq"
    r2_path = output_dir / f"{srr_accession}_2.fastq"
    r1_paths = [r1_path] if r1_path.exists() else []
    r2_paths = [r2_path] if r2_path.exists() else []
    if not r1_paths:
        single_path = output_dir / f"{srr_accession}.fastq"
        if single_path.exists():
            r1_paths = [single_path]

    if not r1_paths:
        return DownloadResult(
            success=False,
            error="fasterq-dump on S3 .sra produced no output files",
            fail_category="s3_fasterq_no_output",
            time_s=time.time() - t0,
        )

    # Compress with pigz if configured
    if config.sra.compress_output:
        compressed_r1, compressed_r2 = [], []
        for path in r1_paths + r2_paths:
            gzip_path = path.parent / f"{path.name}.gz"
            try:
                subprocess.run(
                    ["pigz", "-p", fq_threads, str(path)],
                    check=True, timeout=7200
                )
                final = gzip_path if gzip_path.exists() else path
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning(f"pigz failed/timed out for {path}, keeping uncompressed")
                gzip_path.unlink(missing_ok=True)  # discard any partial .gz
                final = path if path.exists() else gzip_path  # pigz may have moved input
            if path in r1_paths:
                compressed_r1.append(final)
            else:
                compressed_r2.append(final)
        r1_paths = compressed_r1
        r2_paths = compressed_r2

    elapsed = time.time() - t0
    logger.info(f"S3 → fasterq-dump complete in {elapsed:.1f}s")
    return DownloadResult(
        success=True,
        r1_paths=r1_paths,
        r2_paths=r2_paths,
        method="s3_sra",
        time_s=elapsed,
    )


def _sra_single_attempt(
    srr_accession: str,
    output_dir: Path,
    config,
    sra_timeout: int,
    t0: float,
) -> DownloadResult:
    """Execute one prefetch + fasterq-dump attempt (no retries)."""
    # Ensure temp/staging dirs exist
    config.sra.temp_dir.mkdir(parents=True, exist_ok=True)
    config.sra.local_staging.mkdir(parents=True, exist_ok=True)

    # Step 1: prefetch the .sra file locally (avoids fasterq-dump segfault
    # on remote accession resolution seen with sra-tools 3.4.x)
    prefetch_dir = config.sra.local_staging / srr_accession
    prefetch_dir.mkdir(parents=True, exist_ok=True)
    sra_file = prefetch_dir / f"{srr_accession}.sra"

    if not sra_file.exists():
        prefetch_cmd = [
            config.sra.prefetch,
            srr_accession,
            "--output-directory", str(config.sra.local_staging),
            "--max-size", config.sra.max_size,
        ]
        logger.debug(f"Running: {' '.join(prefetch_cmd)}")
        prefetch_result = subprocess.run(
            prefetch_cmd,
            capture_output=True,
            text=True,
            timeout=sra_timeout,
        )
        if prefetch_result.returncode != 0:
            # Clean up partial downloads to avoid filling staging dir
            try:
                import shutil
                shutil.rmtree(prefetch_dir, ignore_errors=True)
            except OSError:
                pass
            return DownloadResult(
                success=False,
                error=f"prefetch failed: {prefetch_result.stderr[:500]}",
                fail_category="sra_prefetch_failed",
                time_s=time.time() - t0,
            )
        if not sra_file.exists():
            try:
                import shutil
                shutil.rmtree(prefetch_dir, ignore_errors=True)
            except OSError:
                pass
            return DownloadResult(
                success=False,
                error=f"prefetch produced no .sra file at {sra_file}",
                fail_category="sra_prefetch_no_output",
                time_s=time.time() - t0,
            )
        logger.info(
            f"Prefetch complete: {sra_file.stat().st_size / 1e9:.1f} GB "
            f"in {time.time() - t0:.0f}s"
        )
    else:
        logger.info(f"Using cached .sra file: {sra_file}")

    # Step 2: fasterq-dump on the local .sra file
    remaining = sra_timeout - int(time.time() - t0)
    if remaining < 30:
        try:
            import shutil
            shutil.rmtree(prefetch_dir, ignore_errors=True)
        except OSError:
            pass
        return DownloadResult(
            success=False,
            error=f"Insufficient time for fasterq-dump after prefetch ({remaining}s left)",
            fail_category="sra_timeout",
            time_s=time.time() - t0,
        )

    cmd = [
        "fasterq-dump",
        str(sra_file),
        "--outdir", str(output_dir),
        "--temp", str(config.sra.temp_dir),
        "--split-3",  # Split paired-end into _1 and _2
        "--threads", str(getattr(config.compression, 'fasterq_threads', 8)),
        "--mem", "3g",
    ]

    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=remaining,
    )

    # Clean up .sra file after conversion
    try:
        sra_file.unlink(missing_ok=True)
        prefetch_dir.rmdir()
    except OSError:
        pass

    if result.returncode != 0:
        return DownloadResult(
            success=False,
            error=f"fasterq-dump failed: {result.stderr[:500]}",
            fail_category="sra_fasterq_failed",
            time_s=time.time() - t0
        )

    # Find output files
    r1_path = output_dir / f"{srr_accession}_1.fastq"
    r2_path = output_dir / f"{srr_accession}_2.fastq"

    r1_paths = [r1_path] if r1_path.exists() else []
    r2_paths = [r2_path] if r2_path.exists() else []

    if not r1_paths:
        # Try single-end naming
        single_path = output_dir / f"{srr_accession}.fastq"
        if single_path.exists():
            r1_paths = [single_path]

    if not r1_paths:
        return DownloadResult(
            success=False,
            error="fasterq-dump produced no output files",
            fail_category="sra_no_output",
            time_s=time.time() - t0
        )

    # Compress with pigz if available
    if config.sra.compress_output:
        compressed_r1 = []
        compressed_r2 = []
        for path in r1_paths + r2_paths:
            gzip_path = path.parent / f"{path.name}.gz"
            try:
                subprocess.run(
                    ["pigz", "-p", str(getattr(config.compression, 'pigz_threads', 8)), str(path)],
                    check=True,
                    timeout=7200
                )
                # pigz replaces original with .gz automatically
                final = gzip_path if gzip_path.exists() else path
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning(f"pigz compression failed/timed out for {path}, keeping uncompressed")
                gzip_path.unlink(missing_ok=True)  # discard any partial .gz
                final = path if path.exists() else gzip_path  # pigz may have moved input

            if path in r1_paths:
                compressed_r1.append(final)
            else:
                compressed_r2.append(final)
        r1_paths = compressed_r1
        r2_paths = compressed_r2

    elapsed = time.time() - t0
    logger.info(f"SRA download completed in {elapsed:.1f}s")

    return DownloadResult(
        success=True,
        r1_paths=r1_paths,
        r2_paths=r2_paths,
        method="fasterq_dump",
        time_s=elapsed
    )


# Retry delays between SRA attempts (seconds)
_SRA_RETRY_DELAYS = (30, 90)


def download_from_sra(
    srr_accession: str,
    output_dir: Path,
    config,
    budget_remaining: Optional[int] = None,
) -> DownloadResult:
    """Download FASTQ from SRA using fasterq-dump as fallback.
    
    Retries up to 2 times (3 total attempts) on failure, with
    exponential backoff (30 s, 90 s).  Cached .sra files from a
    previous prefetch are reused across retries.
    
    Args:
        srr_accession: SRR accession (e.g., "SRR12345678")
        output_dir: Directory to save files
        config: Configuration object with SRA settings
        budget_remaining: Max seconds for this download (caps fasterq timeout).
        
    Returns:
        DownloadResult with success status and file paths
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sra_timeout = config.sra.timeout
    if budget_remaining is not None:
        sra_timeout = min(sra_timeout, max(30, budget_remaining))

    last_result: DownloadResult = DownloadResult(
        success=False, error="SRA not attempted", fail_category="sra_not_attempted"
    )

    max_attempts = 1 + len(_SRA_RETRY_DELAYS)  # 3

    for attempt in range(max_attempts):
        logger.info(
            f"SRA attempt {attempt + 1}/{max_attempts} for {srr_accession}"
        )

        try:
            # Recompute remaining budget each attempt
            attempt_timeout = sra_timeout - int(time.time() - t0)
            if attempt_timeout < 30:
                last_result = DownloadResult(
                    success=False,
                    error=f"SRA budget exhausted ({attempt_timeout}s left)",
                    fail_category="sra_timeout",
                    time_s=time.time() - t0,
                )
                break

            last_result = _sra_single_attempt(
                srr_accession, output_dir, config,
                sra_timeout=attempt_timeout, t0=t0,
            )
        except subprocess.TimeoutExpired:
            last_result = DownloadResult(
                success=False,
                error=f"fasterq-dump timed out after {config.sra.timeout}s",
                fail_category="sra_timeout",
                time_s=time.time() - t0,
            )
        except Exception as e:
            last_result = DownloadResult(
                success=False,
                error=f"SRA fallback failed: {e}",
                fail_category="sra_exception",
                time_s=time.time() - t0,
            )

        if last_result.success:
            return last_result

        # Don't retry if budget would be exhausted
        if attempt < len(_SRA_RETRY_DELAYS):
            delay = _SRA_RETRY_DELAYS[attempt]
            elapsed = time.time() - t0
            if budget_remaining is not None and (elapsed + delay) >= budget_remaining:
                logger.warning(
                    f"SRA retry skipped — budget would be exceeded "
                    f"({elapsed:.0f}s + {delay}s delay ≥ {budget_remaining}s)"
                )
                break
            logger.warning(
                f"SRA attempt {attempt + 1} failed: {last_result.error}. "
                f"Retrying in {delay}s..."
            )
            # Clean up partial output before retry
            for partial in output_dir.glob(f"{srr_accession}*"):
                try:
                    partial.unlink()
                except Exception:
                    pass
            time.sleep(delay)

    return last_result


def download_sample(
    gsm_id: str,
    ena_r1_url: Optional[str],
    ena_r2_url: Optional[str],
    srr_accession: Optional[str],
    output_dir: Path,
    config,
    prefer_ena: bool = True,
    sample_max_time: Optional[int] = None,
) -> DownloadResult:
    """Download FASTQ files for a sample with automatic fallback.
    
    Strategy:
    1. Try ENA direct download (fast, parallel segments)
    2. Fall back to SRA fasterq-dump if ENA fails
    3. Download additional SRR runs if present (multi-SRR samples)
    
    If *sample_max_time* is set, the entire download (including retries
    and fallbacks) is abandoned once the wall-clock budget is exceeded.
    This prevents a single slow sample from blocking an entire batch.
    
    Args:
        gsm_id: GSM accession
        ena_r1_url: ENA URL for R1 (or None)
        ena_r2_url: ENA URL for R2 (or None)
        srr_accession: SRR accession(s) — semicolon-separated for
            multi-run samples (e.g. "SRR123;SRR124;SRR125").
            Only the first SRR is used for SRA fallback; additional
            SRRs are downloaded after the primary succeeds.
        output_dir: Directory to save files
        config: Configuration object
        prefer_ena: If True, try ENA first; if False, go straight to SRA
        sample_max_time: Maximum wall-clock seconds for this sample's
            download (all methods combined). None = no limit.
        
    Returns:
        DownloadResult with success status and paths (may contain
        multiple R1/R2 paths from multiple SRR runs)
    """
    # Wall-clock budget for entire download
    _download_start = time.time()

    def _budget_exceeded() -> bool:
        if sample_max_time is None:
            return False
        return (time.time() - _download_start) >= sample_max_time

    def _remaining_budget() -> Optional[int]:
        """Seconds remaining in budget (None if no budget)."""
        if sample_max_time is None:
            return None
        return max(0, int(sample_max_time - (time.time() - _download_start)))

    # Parse SRR accessions (may be semicolon-separated)
    srr_list = []
    if srr_accession:
        srr_list = [s.strip() for s in str(srr_accession).split(';') if s.strip()]
    primary_srr = srr_list[0] if srr_list else None
    additional_srrs = srr_list[1:] if len(srr_list) > 1 else []

    # Create sample-specific directory
    sample_dir = Path(output_dir) / gsm_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing complete downloads (any .fastq.gz files)
    existing_r1 = sorted(sample_dir.glob(f"{gsm_id}_R1*.fastq.gz"))
    existing_r2 = sorted(sample_dir.glob(f"{gsm_id}_R2*.fastq.gz"))
    if existing_r1 and all(p.stat().st_size > 0 for p in existing_r1):
        r2_valid = [p for p in existing_r2 if p.stat().st_size > 0]
        # If R1 cached but R2 missing, try SRA to get paired reads
        if not r2_valid and config.sra.fallback_enabled and primary_srr:
            logger.info(f"Cached R1 found but no R2 for {gsm_id} — trying SRA fasterq-dump")
            sra_result = download_from_sra(
                primary_srr, sample_dir, config,
                budget_remaining=_remaining_budget(),
            )
            if sra_result.success and sra_result.r2_paths:
                return sra_result
            logger.info("SRA fallback also has no R2 — returning cached R1 only")
        # Include additional SRR files that may already be downloaded
        extra_r1 = sorted(sample_dir.glob("SRR*_1.fastq.gz"))
        extra_r2 = sorted(sample_dir.glob("SRR*_2.fastq.gz"))
        all_r1 = existing_r1 + [p for p in extra_r1 if p not in existing_r1 and p.stat().st_size > 0]
        all_r2 = r2_valid + [p for p in extra_r2 if p not in r2_valid and p.stat().st_size > 0]
        logger.info(f"Reusing existing downloads for {gsm_id} ({len(all_r1)} R1, {len(all_r2)} R2)")
        return DownloadResult(
            success=True,
            r1_paths=all_r1,
            r2_paths=all_r2,
            method="cached",
            time_s=0.0
        )
    
    # Transient error keywords that warrant a top-level retry
    _TRANSIENT_KEYWORDS = (
        "timeout", "timed out", "connection reset", "connection refused",
        "network unreachable", "name resolution", "ssl", "502", "503", "429",
        "broken pipe", "eof occurred",
    )
    MAX_OUTER_RETRIES = 2  # Up to 3 total attempts (1 initial + 2 retries)

    last_result: DownloadResult = DownloadResult(
        success=False, error="No download methods available", fail_category="download_exhausted"
    )

    for outer_attempt in range(MAX_OUTER_RETRIES + 1):
        # Check wall-clock budget before each attempt
        if _budget_exceeded():
            elapsed = time.time() - _download_start
            logger.warning(
                f"Download budget exhausted for {gsm_id} after {elapsed:.0f}s "
                f"(limit {sample_max_time}s) — skipping"
            )
            last_result = DownloadResult(
                success=False,
                error=f"Download budget {sample_max_time}s exceeded ({elapsed:.0f}s elapsed)",
                fail_category="download_timeout_budget",
            )
            break

        # Try S3 first (NCBI public S3 via SDL API — fastest, no rate limits)
        if primary_srr:
            last_result = download_from_s3(
                primary_srr, sample_dir, config,
                budget_remaining=_remaining_budget(),
            )
            if last_result.success:
                break
            logger.warning(f"S3 download failed: {last_result.error}")

        # Try ENA as fallback if URLs available
        if ena_r1_url:
            last_result = download_from_ena(
                ena_r1_url,
                ena_r2_url,
                sample_dir,
                gsm_id,
                config,
                budget_remaining=_remaining_budget(),
            )
            if last_result.success:
                # ENA succeeded but R2 missing — try SRA which splits paired reads
                if not last_result.r2_paths and config.sra.fallback_enabled and primary_srr:
                    logger.info(
                        f"ENA download has no R2 — trying SRA fasterq-dump for {primary_srr}"
                    )
                    sra_result = download_from_sra(
                        primary_srr, sample_dir, config,
                        budget_remaining=_remaining_budget(),
                    )
                    if sra_result.success and sra_result.r2_paths:
                        last_result = sra_result
                    else:
                        logger.info("SRA fallback also has no R2 — proceeding with R1 only")
                break
            logger.warning(f"ENA download failed: {last_result.error}")

        # Last resort: SRA prefetch+fasterq-dump (slow but reliable)
        if config.sra.fallback_enabled and primary_srr:
            last_result = download_from_sra(
                primary_srr, sample_dir, config,
                budget_remaining=_remaining_budget(),
            )
            if last_result.success:
                break

        # Check if the failure looks transient and worth retrying
        err_lower = (last_result.error or "").lower()
        is_transient = any(kw in err_lower for kw in _TRANSIENT_KEYWORDS)

        if is_transient and outer_attempt < MAX_OUTER_RETRIES:
            wait = 60 * (2 ** outer_attempt)  # 60 s, 120 s
            logger.warning(
                f"Transient download failure (attempt {outer_attempt + 1}/"
                f"{MAX_OUTER_RETRIES + 1}): {last_result.error}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
            # Clean up partial files before retrying
            for partial in sample_dir.glob("*.fastq.gz"):
                try:
                    partial.unlink()
                except Exception:
                    pass
            continue

        # Non-transient error or max retries exhausted
        break

    # ── Download additional SRR runs (multi-SRR samples) ──
    if last_result.success and additional_srrs:
        has_r2 = bool(last_result.r2_paths)
        n_extra_ok = 0
        for extra_srr in additional_srrs:
            # Respect download budget for additional SRRs
            if _budget_exceeded():
                logger.warning(f"Budget exhausted — skipping remaining {len(additional_srrs) - n_extra_ok} extra SRRs")
                break

            # Check if already downloaded
            extra_r1_dest = sample_dir / f"{extra_srr}_1.fastq.gz"
            extra_r2_dest = sample_dir / f"{extra_srr}_2.fastq.gz"
            if extra_r1_dest.exists() and extra_r1_dest.stat().st_size > 0:
                last_result.r1_paths.append(extra_r1_dest)
                if has_r2 and extra_r2_dest.exists() and extra_r2_dest.stat().st_size > 0:
                    last_result.r2_paths.append(extra_r2_dest)
                n_extra_ok += 1
                continue

            # Try S3 first for additional SRR (fast, no rate limits)
            s3_result = download_from_s3(
                extra_srr, sample_dir, config,
                budget_remaining=_remaining_budget(),
            )
            if s3_result.success:
                last_result.r1_paths.extend(s3_result.r1_paths)
                last_result.r2_paths.extend(s3_result.r2_paths)
                n_extra_ok += 1
                continue

            # Cap per-file timeout at remaining budget
            _br = _remaining_budget()
            _file_timeout = config.download.timeout
            if _br is not None:
                _file_timeout = min(_file_timeout, max(30, _br))

            # ENA fallback for additional SRR
            extra_r1_url, extra_r2_url = _construct_ena_urls(extra_srr)
            r1_ok, r1_err = download_parallel_segments(
                extra_r1_url.replace("ftp://", "https://"),
                extra_r1_dest,
                segments=config.download.segments,
                timeout=_file_timeout,
                retries=config.download.retries,
            )
            if r1_ok:
                last_result.r1_paths.append(extra_r1_dest)
                if has_r2:
                    # Recompute capped timeout for R2
                    _br2 = _remaining_budget()
                    _ft2 = config.download.timeout
                    if _br2 is not None:
                        _ft2 = min(_ft2, max(30, _br2))
                    r2_ok, _ = download_parallel_segments(
                        extra_r2_url.replace("ftp://", "https://"),
                        extra_r2_dest,
                        segments=config.download.segments,
                        timeout=_ft2,
                        retries=config.download.retries,
                    )
                    if r2_ok:
                        last_result.r2_paths.append(extra_r2_dest)
                    else:
                        # R2 failed — remove the dangling R1 and try SRA fallback
                        last_result.r1_paths.pop()
                        logger.warning(
                            f"ENA R2 failed for {extra_srr} — removing R1 and trying SRA"
                        )
                        if config.sra.fallback_enabled:
                            sra_result = download_from_sra(
                                extra_srr, sample_dir, config,
                                budget_remaining=_remaining_budget(),
                            )
                            if sra_result.success:
                                last_result.r1_paths.extend(sra_result.r1_paths)
                                last_result.r2_paths.extend(sra_result.r2_paths)
                                n_extra_ok += 1
                        continue
                n_extra_ok += 1
                continue

            # ENA failed for this SRR — try fasterq-dump
            if config.sra.fallback_enabled:
                sra_result = download_from_sra(
                    extra_srr, sample_dir, config,
                    budget_remaining=_remaining_budget(),
                )
                if sra_result.success:
                    last_result.r1_paths.extend(sra_result.r1_paths)
                    last_result.r2_paths.extend(sra_result.r2_paths)
                    n_extra_ok += 1
                    continue

            logger.warning(f"Failed to download additional SRR {extra_srr} — skipping")

        if n_extra_ok > 0:
            logger.info(
                f"Downloaded {n_extra_ok}/{len(additional_srrs)} additional SRR runs "
                f"(total: {len(last_result.r1_paths)} R1, {len(last_result.r2_paths)} R2)"
            )

    return last_result
