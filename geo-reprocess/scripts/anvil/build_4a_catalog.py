#!/usr/bin/env python3
"""Build a filtered phase-4a catalog (mouse droplets only).

Reads the full processing_catalog.parquet and writes a filtered copy
containing only phase-4a eligible samples.  The smaller parquet avoids
OOM kills when 10+ concurrent daemon workers each load it on a login node.

Usage:
    python3 build_4a_catalog.py                          # default paths
    python3 build_4a_catalog.py --catalog-dir /path/to   # custom catalog dir
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path so grab_batch imports work
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from grab_batch import _is_eligible_phase4a


def main():
    parser = argparse.ArgumentParser(description="Build filtered phase-4a catalog")
    parser.add_argument(
        "--catalog-dir",
        default=os.environ.get("SCGEO_CATALOG_DIR", ""),
        help="Directory containing processing_catalog.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <catalog-dir>/phase4a/)",
    )
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir) if args.catalog_dir else Path(".")
    src = catalog_dir / "processing_catalog.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else catalog_dir / "phase4a"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "processing_catalog.parquet"

    print(f"Reading {src} ...")
    pc = pd.read_parquet(src)
    print(f"  Full catalog: {len(pc):,} rows  ({src.stat().st_size / 1e6:.1f} MB)")

    mask = _is_eligible_phase4a(pc)
    filtered = pc[mask].copy()
    print(f"  Phase 4a eligible: {filtered.shape[0]:,} rows")

    filtered.to_parquet(dst, index=False)
    print(f"  Written to {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
