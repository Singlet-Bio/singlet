# SPDX-License-Identifier: MIT
"""singlet.fetch — Download canonical sample bundles from a remote host.

Sample directories are served as a flat collection of files under a
per-accession prefix. The required entry point is ``manifest.json``
which lists every file in the sample with its size and sha256 digest::

    {
      "schema": "singlet-sample-manifest/v1",
      "accession": "GSM3308814",
      "produced_by": {"tool": "singlet", "version": "0.3.0"},
      "files": [
        {"path": "summary.json",      "size": 4321,  "sha256": "..."},
        {"path": "counts.1pz",        "size": 12345, "sha256": "..."},
        {"path": "cell_meta.parquet", "size": 6789,  "sha256": "..."},
        ...
      ]
    }

Caching is path-based: a file already present on disk with a matching
sha256 is not re-downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["fetch", "default_cache_dir", "default_base_url"]


_DEFAULT_BASE_URL = "https://data.singlet.bio/v1"
_MANIFEST_NAME = "manifest.json"
_USER_AGENT = "singlet-fetch/1"


def default_base_url() -> str:
    """Base URL for hosted samples. Override with ``SINGLET_DATA_BASE``."""
    return os.environ.get("SINGLET_DATA_BASE", _DEFAULT_BASE_URL).rstrip("/")


def default_cache_dir() -> Path:
    """Local cache root. Override with ``SINGLET_CACHE_DIR``."""
    env = os.environ.get("SINGLET_CACHE_DIR")
    root = Path(env) if env else (Path.home() / ".singlet" / "data")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _http_get(url: str, dest: Path) -> None:
    """Stream URL → dest atomically. Uses a .part file then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        raise FileNotFoundError(f"{url} → HTTP {e.code}") from e
    tmp.replace(dest)


def _fetch_one(base_url: str, rel: str, expected_sha: Optional[str], out_dir: Path) -> Path:
    dest = out_dir / rel
    if expected_sha and dest.exists():
        if _sha256_of(dest) == expected_sha:
            return dest
    _http_get(f"{base_url}/{rel}", dest)
    if expected_sha:
        got = _sha256_of(dest)
        if got != expected_sha:
            dest.unlink(missing_ok=True)
            raise IOError(
                f"sha256 mismatch for {rel}: expected {expected_sha}, got {got}"
            )
    return dest


def fetch(
    accession: str,
    cache_dir: Optional[str | Path] = None,
    base_url: Optional[str] = None,
    files: Optional[Iterable[str]] = None,
    max_workers: int = 8,
) -> Path:
    """Download a hosted sample to the local cache; return its directory.

    Parameters
    ----------
    accession
        Sample accession (e.g. ``"GSM3308814"``) used as the remote prefix
        and the local cache subdirectory name.
    cache_dir
        Override the local cache root (default: ``~/.singlet/data`` or
        ``$SINGLET_CACHE_DIR``).
    base_url
        Override the remote base URL (default: ``$SINGLET_DATA_BASE``
        or ``https://data.singlet.bio/v1``).
    files
        Optional list of file paths (relative to the sample root) to fetch.
        Default: every file in the manifest.
    max_workers
        Parallel download workers.

    Returns
    -------
    pathlib.Path
        The local sample directory (containing ``manifest.json``,
        ``summary.json``, ``counts.1pz``, …).

    Notes
    -----
    Files already present on disk with the manifest-declared sha256 are not
    re-downloaded. Partial transfers go to ``*.part`` files and are renamed
    atomically on success.
    """
    base = (base_url or default_base_url()).rstrip("/")
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    out_dir = root / accession
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_base = f"{base}/{accession}"
    manifest_path = _fetch_one(sample_base, _MANIFEST_NAME, None, out_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)

    listing = manifest.get("files", [])
    if not isinstance(listing, list):
        raise ValueError(f"{manifest_path}: 'files' must be a list")

    want = set(files) if files is not None else None
    todo = [
        (entry["path"], entry.get("sha256"))
        for entry in listing
        if want is None or entry["path"] in want
    ]

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, sample_base, rel, sha, out_dir): rel
            for rel, sha in todo
        }
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 — bubble up after collecting all
                errors.append(f"{rel}: {exc}")

    if errors:
        raise RuntimeError(
            "fetch failed for {n} file(s):\n  - {joined}".format(
                n=len(errors), joined="\n  - ".join(errors)
            )
        )

    return out_dir
