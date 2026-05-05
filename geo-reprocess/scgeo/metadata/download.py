"""Download supplementary files from GEO FTP/HTTPS.

Supports both GEO FTP URLs and their HTTPS equivalents, with automatic
protocol conversion and retry logic.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_supplementary_file(
    url: str,
    dest: Path,
    max_size_mb: int = 5000,
    timeout: int = 600,
    retries: int = 3,
) -> bool:
    """Download a supplementary file from GEO.

    Converts ``ftp://`` URLs to ``https://`` automatically.  Skips
    files larger than *max_size_mb*.  Retries on transient failures.

    Args:
        url: FTP or HTTPS URL to the supplementary file.
        dest: Local destination path.
        max_size_mb: Maximum file size in megabytes; larger files
            are skipped.
        timeout: HTTP timeout in seconds.
        retries: Number of retry attempts on transient failures.

    Returns:
        ``True`` on success, ``False`` on failure or skip.
    """
    # Normalise to HTTPS (NCBI FTP mirrors are accessible via HTTPS)
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
        url = url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
    elif url.startswith("ftp://"):
        url = url.replace("ftp://", "https://", 1)

    dest.parent.mkdir(parents=True, exist_ok=True)

    import time as _time
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cellarium/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                    logger.info(
                        "Skipping %s: too large (%d MB > %d MB limit)",
                        url, int(content_length) // (1024 * 1024), max_size_mb,
                    )
                    return False
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
            logger.debug("Downloaded %s -> %s", url, dest)
            return True
        except Exception as e:
            if dest.exists():
                dest.unlink()
            if attempt < retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                logger.debug("Download attempt %d failed for %s: %s (retrying in %ds)", attempt + 1, url, e, wait)
                _time.sleep(wait)
            else:
                logger.warning("Download failed for %s after %d attempts: %s", url, retries, e)
                return False
