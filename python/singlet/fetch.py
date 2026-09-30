# SPDX-License-Identifier: MIT
"""singlet.fetch — Download canonical sample directories from a mirror.

.. note::
   The public per-sample host this module was written for
   (``https://data.singlet.bio/v1``) was retired; public data is published
   as one ``.singlet`` bundle per study. Use :func:`singlet.load`,
   :func:`singlet.download` or :func:`singlet.open_bundle` for it, and
   :func:`singlet.find` to search. :func:`fetch` still works against a
   self-hosted mirror of sample directories, named by ``base_url`` or by
   ``$SINGLET_SAMPLE_MIRROR``; with neither it raises
   :class:`NotImplementedError`.

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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["fetch", "default_cache_dir", "default_base_url"]


_MANIFEST_NAME = "manifest.json"
_USER_AGENT = "singlet-fetch/1"

# The former public per-sample host. Retired; fetch() refuses it.
_RETIRED_SAMPLE_BASE = "https://data.singlet.bio/v1"
# Names a self-hosted mirror of sample directories. Separate from
# $SINGLET_DATA_BASE, which now names the .singlet bundle host singlet.load() uses.
_SAMPLE_MIRROR_ENV = "SINGLET_SAMPLE_MIRROR"
_LEGACY_ENV = "SINGLET_DATA_BASE"

_RETIRED_HOST_MSG = (
    "singlet.fetch() downloaded per-sample directories from https://data.singlet.bio/v1, "
    "which has been retired. Public data is now one .singlet bundle per study: use "
    "singlet.load('GSE…' or 'GSM…') for an AnnData, singlet.download('GSE…') for the file, "
    "or singlet.find('…') to search. To fetch from a self-hosted mirror of sample "
    "directories, pass base_url= or set $SINGLET_SAMPLE_MIRROR."
)


def _sample_mirror() -> tuple[Optional[str], Optional[str]]:
    """``(mirror_base_url, env_var_it_came_from)``, or ``(None, None)``.

    ``$SINGLET_SAMPLE_MIRROR`` names the mirror. Up to 2.0.0 :func:`fetch`
    read it from ``$SINGLET_DATA_BASE``; that variable now names the bundle
    host used by :func:`singlet.load`, but a value set there is still used
    here when ``$SINGLET_SAMPLE_MIRROR`` is unset, with a ``FutureWarning``.
    """
    mirror = os.environ.get(_SAMPLE_MIRROR_ENV, "").strip()
    if mirror:
        return mirror.rstrip("/"), _SAMPLE_MIRROR_ENV
    legacy = os.environ.get(_LEGACY_ENV, "").strip()
    if legacy:
        warnings.warn(
            f"singlet.fetch() is reading its sample mirror from ${_LEGACY_ENV}, which now "
            "names the .singlet bundle host used by singlet.load(). Set "
            f"${_SAMPLE_MIRROR_ENV} to your sample mirror instead.",
            FutureWarning,
            stacklevel=3,
        )
        return legacy.rstrip("/"), _LEGACY_ENV
    return None, None


def default_base_url() -> str:
    """Base URL of the sample mirror :func:`fetch` uses when no ``base_url`` is given.

    ``$SINGLET_SAMPLE_MIRROR`` (or, deprecated, ``$SINGLET_DATA_BASE``); with
    neither set, the retired public host ``https://data.singlet.bio/v1``,
    which :func:`fetch` refuses with :class:`NotImplementedError`. The
    ``.singlet`` bundle host used by :func:`singlet.load` is configured
    separately, with ``$SINGLET_DATA_BASE``.
    """
    return _sample_mirror()[0] or _RETIRED_SAMPLE_BASE


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
    """Download a sample directory from a mirror to the local cache.

    Parameters
    ----------
    accession
        Sample accession (e.g. ``"GSM3308814"``) used as the remote prefix
        and the local cache subdirectory name.
    cache_dir
        Override the local cache root (default: ``~/.singlet/data`` or
        ``$SINGLET_CACHE_DIR``).
    base_url
        Base URL of a self-hosted mirror of sample directories. Defaults to
        ``$SINGLET_SAMPLE_MIRROR`` (or, deprecated, ``$SINGLET_DATA_BASE``).
        The former public default (``https://data.singlet.bio/v1``) has been
        retired: with no mirror configured this raises
        :class:`NotImplementedError`. For public data use
        :func:`singlet.load` instead.
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
    source = None
    if not base_url:
        base_url, source = _sample_mirror()
    if not base_url or base_url.rstrip("/") == _RETIRED_SAMPLE_BASE:
        raise NotImplementedError(_RETIRED_HOST_MSG)
    base = base_url.rstrip("/")
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    out_dir = root / accession
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_base = f"{base}/{accession}"
    try:
        manifest_path = _fetch_one(sample_base, _MANIFEST_NAME, None, out_dir)
    except FileNotFoundError as e:
        if source != _LEGACY_ENV:
            raise
        raise FileNotFoundError(
            f"{e}. The sample mirror was taken from ${_LEGACY_ENV}, which now names "
            f"the .singlet bundle host; set ${_SAMPLE_MIRROR_ENV} to a mirror of sample "
            "directories, or use singlet.load() for bundles."
        ) from e
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
