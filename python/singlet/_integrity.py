# SPDX-License-Identifier: MIT
"""Integrity checks for downloaded ``.singlet`` bundles, and accession hygiene.

A bundle is a ZIP64 archive whose ``manifest.json`` lists the SHA-256 of every
other member. A download is only moved into the cache after it passes:

1. the byte count matches ``Content-Length`` (a connection that drops early is
   an error, not a short file);
2. the archive's central directory reads (a truncated file has none);
3. the manifest declares a schema this reader understands;
4. every member the manifest lists is present, its ZIP CRC-32 holds (checked
   by :mod:`zipfile` as the member is read), and its SHA-256 matches.

Member names are only ever used as dictionary keys and ZIP lookups, never as
file-system paths, but names that could escape a directory are rejected anyway
so a future extraction cannot be tricked by an old file.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

#: Schema major version this reader understands (``manifest.json`` ``schema_version``).
SUPPORTED_SCHEMA_MAJOR = 1

_ACCESSION_RE = re.compile(r"^GS[EM][0-9]{1,9}$")
_CHUNK = 1 << 20


class BundleIntegrityError(ValueError):
    """A ``.singlet`` file is truncated, corrupt, altered, or from an unsupported schema."""


def normalize_accession(value: object) -> str:
    """Upper-case and validate a GEO accession (``GSE…`` or ``GSM…``).

    Raises ``ValueError`` for anything else, including path separators,
    whitespace or URL syntax, so the result is safe to put in a file name and
    a URL.
    """
    acc = str(value).strip().upper()
    if not _ACCESSION_RE.match(acc):
        raise ValueError(f"not a GEO accession (expected GSE… or GSM… followed by digits): {value!r}")
    return acc


def _safe_member_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")) or "\\" in name or "\x00" in name:
        return False
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts


def schema_major(manifest: dict) -> int:
    """Major version of ``manifest["schema_version"]``; raises if absent or malformed."""
    raw = manifest.get("schema_version")
    m = re.match(r"^(\d+)(?:\.\d+)*$", str(raw)) if raw is not None else None
    if not m:
        raise BundleIntegrityError(f"manifest.json has no valid schema_version (got {raw!r})")
    return int(m.group(1))


def check_schema(manifest: dict) -> None:
    """Refuse a bundle written under a schema major this reader does not know."""
    major = schema_major(manifest)
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise BundleIntegrityError(
            f"this bundle uses schema {manifest.get('schema_version')!r}, but this version of singlet reads "
            f"schema {SUPPORTED_SCHEMA_MAJOR}.x. Upgrade singlet (pip install -U singlet-bio) to read it."
        )


def read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        with zf.open("manifest.json") as fh:
            manifest = json.load(fh)
    except KeyError as e:
        raise BundleIntegrityError("not a .singlet bundle: manifest.json is missing") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise BundleIntegrityError(f"manifest.json is not valid JSON: {e}") from e
    if not isinstance(manifest, dict):
        raise BundleIntegrityError("manifest.json must hold a JSON object")
    return manifest


def verify_bundle(
    path: str | Path,
    *,
    expected_size: Optional[int] = None,
    hashes: bool = True,
) -> dict:
    """Check a ``.singlet`` file and return its parsed manifest.

    Parameters
    ----------
    path
        The file to check.
    expected_size
        The size the server announced; a different size is an error.
    hashes
        Also re-hash every member against the manifest (reads the whole file
        once). With ``False`` only the size, the archive structure and the
        schema are checked, which is enough to catch a truncated file and is
        what a cache hit costs.

    Raises
    ------
    BundleIntegrityError
        On any failure. The message says what is wrong.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        raise BundleIntegrityError(f"cannot read {p}: {e}") from e
    if size == 0:
        raise BundleIntegrityError(f"{p.name} is empty")
    if expected_size is not None and size != expected_size:
        raise BundleIntegrityError(
            f"{p.name} is {size:,} bytes but the server announced {expected_size:,}: the download was cut short or altered"
        )
    try:
        zf = zipfile.ZipFile(p, "r")
    except zipfile.BadZipFile as e:
        raise BundleIntegrityError(f"{p.name} is not a complete .singlet archive ({e})") from e
    with zf:
        manifest = read_manifest(zf)
        check_schema(manifest)
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict) or not checksums:
            raise BundleIntegrityError("manifest.json lists no checksums")
        names = set(zf.namelist())
        for name in names:
            if not _safe_member_name(name):
                raise BundleIntegrityError(f"unsafe member name in archive: {name!r}")
        for name, digest in checksums.items():
            if not _safe_member_name(str(name)):
                raise BundleIntegrityError(f"unsafe member name in manifest: {name!r}")
            if name not in names:
                raise BundleIntegrityError(f"{name} is listed in manifest.json but missing from the archive")
            if not (isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
                raise BundleIntegrityError(f"manifest.json has a malformed checksum for {name}")
        if hashes:
            for name, digest in checksums.items():
                h = hashlib.sha256()
                try:
                    with zf.open(name) as fh:  # zipfile verifies the CRC-32 at end of member
                        for chunk in iter(lambda: fh.read(_CHUNK), b""):
                            h.update(chunk)
                except (zipfile.BadZipFile, OSError, RuntimeError) as e:
                    raise BundleIntegrityError(f"{name} is corrupt ({e})") from e
                if h.hexdigest() != digest.lower():
                    raise BundleIntegrityError(f"{name} does not match its checksum in manifest.json")
    return manifest
