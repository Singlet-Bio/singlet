# SPDX-License-Identifier: MIT
"""singlet.whitelists — Barcode whitelist loading with lazy download.

The small whitelists are shipped in the package under ``singlet/data/whitelists/``;
the four large 10x ones (3M-february-2018, 737K-august-2016, 737K-cratac-v1,
gex_737K-arc-v1) are downloaded on first use from Teichlab's public CDN
(scg_lib_structs) into ``~/.singlet/whitelists`` and then read from there.

Usage::

    from singlet.whitelists import load_whitelist, WHITELIST_DIR

    barcodes = load_whitelist("737K-august-2016")  # set of 16-bp barcodes
    path = ensure_whitelist("737K-august-2016")  # a Path, downloaded if needed
"""

from __future__ import annotations

import gzip
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Set

__all__ = [
    "WHITELIST_DIR",
    "load_whitelist",
    "ensure_whitelist",
    "list_available_whitelists",
]

# Package-bundled whitelists (preferred)
_PKG_DATA = Path(__file__).parent / "data" / "whitelists"

# User cache fallback (lazy download)
_CACHE_DIR = Path(os.environ.get("SINGLET_CACHE_DIR", Path.home() / ".singlet")) / "whitelists"

# Public CDN for 10x Genomics barcode whitelists (Teichlab scg_lib_structs)
_TEICHLAB_BASE = "https://teichlab.github.io/scg_lib_structs/data/10X-Genomics"

# Known whitelists with their Teichlab filenames (gzipped on CDN)
_TEICHLAB_MAP = {
    "737K-august-2016": "737K-august-2016.txt.gz",
    "737K-april-2014": None,  # Not on Teichlab, must be bundled
    "737K-cratac-v1": "737K-cratac-v1.txt.gz",
    "3M-february-2018": "3M-february-2018.txt.gz",
    "gex_737K-arc-v1": "gex_737K-arc-v1.txt.gz",
}


def _resolve_dir() -> Path:
    """Return the active whitelist directory (bundled or cache)."""
    if _PKG_DATA.is_dir():
        return _PKG_DATA
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


WHITELIST_DIR = _resolve_dir()


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _check_name(name: str) -> str:
    """A whitelist name is a bare file stem: no separators, no traversal."""
    if not isinstance(name, str) or not _NAME_RE.match(name) or ".." in name:
        raise ValueError(f"not a valid whitelist name: {name!r}")
    return name


def list_available_whitelists() -> list[str]:
    """List whitelist names usable here: bundled, already cached, or downloadable on first use."""
    local = {p.stem for d in (_PKG_DATA, _CACHE_DIR) if d.is_dir() for p in d.glob("*.txt")}
    downloadable = {n for n, f in _TEICHLAB_MAP.items() if f}
    return sorted(local | downloadable)


def _download_whitelist(name: str, dest: Path) -> None:
    """Download whitelist from Teichlab CDN."""
    name = _check_name(name)
    cdn_file = _TEICHLAB_MAP.get(name)
    if cdn_file is None:
        raise FileNotFoundError(
            f"Whitelist '{name}' is not available on the public CDN. "
            f"Ensure it is bundled with the package."
        )
    url = f"{_TEICHLAB_BASE}/{cdn_file}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".txt.part")
    req = urllib.request.Request(url, headers={"User-Agent": "singlet-whitelists/1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # Decompress gzip on the fly
            with gzip.open(resp, "rt") as gz, open(tmp, "w") as out:
                shutil.copyfileobj(gz, out)
        _check_downloaded(tmp, name)
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        raise FileNotFoundError(f"Failed to download {url}: HTTP {e.code}") from e
    tmp.replace(dest)


def _check_downloaded(path: Path, name: str) -> None:
    """A barcode whitelist is one A/C/G/T/N string per line; anything else is not one (an error page, a truncated gzip)."""
    n = 0
    problem = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not re.fullmatch(r"[ACGTN]+(-[0-9]+)?", s):
                problem = f"line {n + 1}: {s[:40]!r}"
                break
            n += 1
    if problem is None and n == 0:
        problem = "it is empty"
    if problem is not None:
        path.unlink(
            missing_ok=True
        )  # after the file is closed: Windows will not delete an open file
        raise ValueError(f"downloaded whitelist {name!r} is not a barcode list ({problem})")


def ensure_whitelist(name: str) -> Path:
    """Ensure whitelist file exists, downloading if necessary.

    Parameters
    ----------
    name : str
        Whitelist name without .txt extension (e.g., "737K-august-2016").

    Returns
    -------
    Path
        Path to the whitelist file.

    Raises
    ------
    FileNotFoundError
        If the whitelist cannot be found or downloaded.
    """
    name = _check_name(name)
    # Check package-bundled location first
    bundled = _PKG_DATA / f"{name}.txt"
    if bundled.is_file():
        return bundled

    # Check cache
    cached = _CACHE_DIR / f"{name}.txt"
    if cached.is_file():
        return cached

    # Attempt download
    _download_whitelist(name, cached)
    return cached


def load_whitelist(name: str) -> Set[str]:
    """Load a barcode whitelist as a set of strings.

    Parameters
    ----------
    name : str
        Whitelist name without .txt extension (e.g., "737K-august-2016").

    Returns
    -------
    set of str
        The barcode sequences (one per line in the whitelist file).
    """
    path = ensure_whitelist(name)
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}
