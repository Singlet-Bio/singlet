# SPDX-License-Identifier: MIT
"""singlet.manifest_gen — Emit ``manifest.json`` for a sample directory.

Scans a canonical v2 sample directory and writes the v1 manifest schema
(see :mod:`singlet.fetch`). Every file except ``manifest.json`` itself is
listed with its size and sha256 digest.

CLI::

    python -m singlet.manifest_gen SAMPLE_DIR [--accession GSM...] [-o manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

__all__ = ["generate_manifest", "main"]


_SCHEMA = "singlet-sample-manifest/v1"
_TOOL_VERSION = "0.3.0"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def generate_manifest(sample_dir, accession: Optional[str] = None) -> dict:
    """Build a manifest dict for ``sample_dir``.

    Excludes ``manifest.json`` (the output itself) and ``*.part`` files.
    All other regular files are included with size and sha256.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise NotADirectoryError(sample_dir)

    files = []
    for p in sorted(sample_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(sample_dir).as_posix()
        if rel == "manifest.json" or rel.endswith(".part"):
            continue
        files.append({"path": rel, "size": p.stat().st_size, "sha256": _sha256(p)})

    return {
        "schema": _SCHEMA,
        "accession": accession or sample_dir.name,
        "produced_by": {"tool": "singlet", "version": _TOOL_VERSION},
        "files": files,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m singlet.manifest_gen")
    ap.add_argument("sample_dir")
    ap.add_argument("--accession", default=None)
    ap.add_argument("-o", "--output", default=None, help="Default: <sample_dir>/manifest.json")
    args = ap.parse_args(argv)

    manifest = generate_manifest(args.sample_dir, args.accession)
    out = Path(args.output) if args.output else Path(args.sample_dir) / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
