#!/usr/bin/env python3
"""Convert singlet-pileup MTX output to .1pz format.

Usage:
    python mtx_to_1pz.py <pileup_output_dir> [--threads 8]

Converts all .mtx files in the directory to .1pz alongside their sidecar files.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

try:
    import singlepress
except ImportError:
    print("ERROR: singlepress not installed. Install with: pip install singlepress", file=sys.stderr)
    sys.exit(1)


def convert_mtx_to_1pz(mtx_path: Path, threads: int = 8) -> Path:
    """Convert an MTX + sidecar TSVs to .1pz format."""
    prefix = mtx_path.stem  # e.g., "snp_ad"
    out_dir = mtx_path.parent
    feat_path = out_dir / f"{prefix}_features.tsv"
    bc_path = out_dir / f"{prefix}_barcodes.tsv"
    pz_path = out_dir / f"{prefix}.1pz"

    # Read MTX (features × barcodes = rows × cols)
    mat = sio.mmread(str(mtx_path))
    if not sp.issparse(mat):
        mat = sp.csc_matrix(mat)
    else:
        mat = mat.tocsc()

    # Read names
    features = feat_path.read_text().strip().split("\n") if feat_path.exists() else None
    barcodes = bc_path.read_text().strip().split("\n") if bc_path.exists() else None

    print(f"  {prefix}: {mat.shape[0]} features × {mat.shape[1]} barcodes, {mat.nnz} nnz")

    # Write .1pz
    singlepress.write_1pz(
        str(pz_path),
        mat,
        rownames=features,
        colnames=barcodes,
        num_threads=threads,
        level=3,
    )

    # Write metadata parquets
    if features:
        pd.DataFrame({"feature_name": features}).to_parquet(
            out_dir / f"{prefix}_feature_metadata.parquet"
        )
    if barcodes:
        pd.DataFrame({"barcode": barcodes}).to_parquet(
            out_dir / f"{prefix}_cell_metadata.parquet"
        )

    print(f"  → {pz_path} ({pz_path.stat().st_size / 1e6:.1f} MB)")
    return pz_path


def main():
    parser = argparse.ArgumentParser(description="Convert singlet-pileup MTX to .1pz")
    parser.add_argument("dir", type=Path, help="Pileup output directory")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    mtx_files = sorted(args.dir.glob("*.mtx"))
    if not mtx_files:
        print(f"No .mtx files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(mtx_files)} matrices to .1pz format:")
    for mtx in mtx_files:
        convert_mtx_to_1pz(mtx, args.threads)

    print("Done!")


if __name__ == "__main__":
    main()
