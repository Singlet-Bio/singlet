# SPDX-License-Identifier: MIT
"""singlet.manifest — Validate that a v2 sample directory is complete.

A canonical singlet v2 sample directory is considered **complete** when
all required files exist and match the schema in
``docs/CANONICAL_OUTPUT_FORMAT.md``. This module provides one function:
:func:`validate_sample` — a fast, structural check used by orchestrators
(post-job, post-rsync, pre-publish) to gate downstream work.

CLI::

    python -m singlet.manifest SAMPLE_DIR              # one sample
    python -m singlet.manifest --batch DIR             # all subdirs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from singlet.pz_v2 import PzV2Error, read_pz_v2

__all__ = ["ValidationReport", "validate_sample", "main"]


REQUIRED_FILES = (
    "summary.json",
    "counts.1pz",
    "cell_meta.parquet",
)


OPTIONAL_FILES = (
    "snp.1pz",
    "mt.1pz",
    "saturation_curve.tsv",
    "star_Log.final.out",
    "nonhost.json",
    "nonhost_species.1pz",
)


@dataclass
class ValidationReport:
    """Outcome of a single-sample validation pass."""

    path: Path
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_cells: Optional[int] = None
    block_names: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def summary_line(self) -> str:
        status = "OK " if self.ok else "FAIL"
        return f"[{status}] {self.path} cells={self.n_cells} blocks={self.block_names}"


def validate_sample(path) -> ValidationReport:
    """Validate a single sample directory against the v2 spec.

    Parameters
    ----------
    path
        Directory to check.

    Returns
    -------
    ValidationReport
        ``.ok`` is ``True`` only if every required file exists and
        ``counts.1pz`` is readable with at least one block.
    """
    path = Path(path)
    rep = ValidationReport(path=path)

    if not path.is_dir():
        rep.ok = False
        rep.errors.append(f"not a directory: {path}")
        return rep

    for name in REQUIRED_FILES:
        if not (path / name).exists():
            rep.ok = False
            rep.errors.append(f"missing required file: {name}")

    # counts.1pz structural check
    counts = path / "counts.1pz"
    if counts.exists():
        try:
            with read_pz_v2(counts) as rd:
                rep.n_cells = rd.n_cells
                rep.block_names = list(rd.block_names)
                if not rep.block_names:
                    rep.ok = False
                    rep.errors.append("counts.1pz has no blocks")
        except (PzV2Error, OSError, ValueError) as exc:
            rep.ok = False
            rep.errors.append(f"counts.1pz unreadable: {exc}")

    # summary.json shape check
    summary_path = path / "summary.json"
    if summary_path.exists():
        try:
            json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rep.ok = False
            rep.errors.append(f"summary.json invalid: {exc}")

    # Cross-check optional pair: snp.1pz / mt.1pz should have two layers.
    for fname, want_block in (("snp.1pz", "snp"), ("mt.1pz", "mt")):
        p = path / fname
        if not p.exists():
            continue
        try:
            with read_pz_v2(p) as rd:
                if want_block in rd.block_names:
                    b = rd.block(want_block)
                    if b.n_data_layers != 2:
                        rep.warnings.append(
                            f"{fname}/{want_block} has {b.n_data_layers} data layers (expected 2)"
                        )
        except (PzV2Error, OSError, ValueError) as exc:
            rep.warnings.append(f"{fname} unreadable: {exc}")

    return rep


def _validate_batch(root: Path) -> List[ValidationReport]:
    return [validate_sample(p) for p in sorted(root.iterdir()) if p.is_dir()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m singlet.manifest",
        description="Validate v2 sample directory structure.",
    )
    p.add_argument("path", help="Sample directory (or batch root with --batch)")
    p.add_argument(
        "--batch",
        action="store_true",
        help="Treat PATH as a directory of sample dirs; validate each one.",
    )
    args = p.parse_args(argv)

    root = Path(args.path)
    reports = _validate_batch(root) if args.batch else [validate_sample(root)]
    failed = 0
    for r in reports:
        print(r.summary_line())
        for e in r.errors:
            print(f"  ERROR: {e}")
        for w in r.warnings:
            print(f"  WARN:  {w}")
        if not r.ok:
            failed += 1
    if failed:
        print(f"\n{failed}/{len(reports)} samples failed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
