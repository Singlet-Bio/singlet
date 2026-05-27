#!/usr/bin/env python3
"""Verify reference bundle against state/reference-manifest-v1.yaml.

Usage:
    # Verify required files match recorded checksums.
    verify_reference.py --ref-base $SINGLET_REF_BASE

    # First-time setup: compute checksums and write them back into the manifest
    # (replaces <pending> entries). Required before tagging v0.3.0-pilot-freeze.
    verify_reference.py --ref-base $SINGLET_REF_BASE --record

Exit codes:
    0  all required files present + checksums match (or recorded successfully)
    1  missing required files
    2  checksum mismatch
    3  manifest unreadable / malformed
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    sys.stderr.write("error: PyYAML required (pip install pyyaml)\n")
    sys.exit(3)


CHUNK = 1 << 20  # 1 MB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        sys.stderr.write(f"error: manifest not found: {path}\n")
        sys.exit(3)
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def verify(ref_base: Path, manifest: dict, record: bool) -> int:
    files = manifest.get("files", [])
    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    new_records: list[tuple[int, str]] = []

    for idx, entry in enumerate(files):
        rel = entry["path"]
        full = ref_base / rel
        recorded = entry.get("sha256", "")
        required = entry.get("required", True)

        if not full.exists():
            if required:
                missing.append(rel)
            continue

        actual = sha256_of(full)

        if record or recorded == "<pending>":
            new_records.append((idx, actual))
            print(f"[record] {rel}: sha256={actual}")
            continue

        if actual != recorded:
            mismatched.append((rel, recorded, actual))
        else:
            print(f"[ok]     {rel}")

    if missing:
        sys.stderr.write("\nMISSING required files:\n")
        for m in missing:
            sys.stderr.write(f"  - {m}\n")
        return 1

    if mismatched:
        sys.stderr.write("\nCHECKSUM MISMATCH:\n")
        for rel, exp, act in mismatched:
            sys.stderr.write(f"  - {rel}\n    expected: {exp}\n    actual:   {act}\n")
        return 2

    if record and new_records:
        # Re-emit the YAML with checksums substituted.
        manifest_path = Path(__file__).parent.parent / "state" / "reference-manifest-v1.yaml"
        text = manifest_path.read_text()
        for idx, actual in new_records:
            entry = files[idx]
            # Replace the first <pending> occurrence after the file's path line.
            needle_path = f"path: {entry['path']}"
            pos = text.find(needle_path)
            if pos == -1:
                continue
            sha_pos = text.find("sha256: <pending>", pos)
            if sha_pos == -1:
                continue
            text = text[:sha_pos] + f"sha256: {actual}" + text[sha_pos + len("sha256: <pending>"):]
        manifest_path.write_text(text)
        print(f"\nUpdated {len(new_records)} entries in {manifest_path}")

    # Compute and print the manifest SHA256 (for embedding in summary.json).
    manifest_path = Path(__file__).parent.parent / "state" / "reference-manifest-v1.yaml"
    manifest_sha = sha256_of(manifest_path)
    print(f"\nreference_manifest_sha256: {manifest_sha}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-base", required=True, help="Path to SINGLET_REF_BASE")
    p.add_argument(
        "--manifest",
        default=str(Path(__file__).parent.parent / "state" / "reference-manifest-v1.yaml"),
        help="Path to reference manifest YAML",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="Compute checksums and write them into the manifest (first-time setup)",
    )
    args = p.parse_args()

    manifest = load_manifest(Path(args.manifest))
    return verify(Path(args.ref_base), manifest, args.record)


if __name__ == "__main__":
    sys.exit(main())
