#!/usr/bin/env python3
"""Filter pileup MTX output to only include cells in a filtered barcode list.

When running the streaming pipeline (STARsolo → singlet-pileup), we pileup
against the full 3.7M barcode whitelist. This script filters the output
matrices to only include cells that passed STARsolo's cell calling.

Usage:
    python filter_pileup.py <pileup_dir> <filtered_barcodes.tsv> [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def filter_matrix(mtx_path: Path, bc_path: Path, keep_indices: np.ndarray,
                  filtered_barcodes: list, out_dir: Path):
    """Filter a pileup .mtx to keep only specified barcode columns."""
    prefix = mtx_path.stem
    mat = sio.mmread(str(mtx_path))
    if sp.issparse(mat):
        mat = mat.tocsc()
    else:
        mat = sp.csc_matrix(mat)

    # Subset columns
    mat_filtered = mat[:, keep_indices]

    # Write filtered output
    out_mtx = out_dir / f"{prefix}.mtx"
    sio.mmwrite(str(out_mtx), mat_filtered)

    out_bc = out_dir / f"{prefix}_barcodes.tsv"
    out_bc.write_text("\n".join(filtered_barcodes) + "\n")

    # Copy features file unchanged
    feat_src = mtx_path.parent / f"{prefix}_features.tsv"
    if feat_src.exists():
        (out_dir / f"{prefix}_features.tsv").write_text(feat_src.read_text())

    nnz = mat_filtered.nnz if sp.issparse(mat_filtered) else np.count_nonzero(mat_filtered)
    print(f"  {prefix}: {mat.shape[1]} → {mat_filtered.shape[1]} cells, "
          f"{mat.nnz} → {nnz} nnz")


def main():
    parser = argparse.ArgumentParser(description="Filter pileup output by barcode list")
    parser.add_argument("pileup_dir", type=Path)
    parser.add_argument("barcodes", type=Path, help="Filtered barcode list (one per line)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.pileup_dir / "filtered")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read filtered barcodes
    filtered = set(args.barcodes.read_text().strip().split("\n"))
    print(f"Filtered barcodes: {len(filtered)}")

    # Find all MTX files in pileup dir
    mtx_files = sorted(args.pileup_dir.glob("*.mtx"))
    if not mtx_files:
        print(f"No .mtx files found in {args.pileup_dir}", file=sys.stderr)
        sys.exit(1)

    # Read barcode order from first matrix's sidecar
    first_prefix = mtx_files[0].stem
    bc_path = args.pileup_dir / f"{first_prefix}_barcodes.tsv"
    all_barcodes = bc_path.read_text().strip().split("\n")

    # Map filtered barcodes to column indices
    keep_indices = np.array([i for i, bc in enumerate(all_barcodes) if bc in filtered])
    keep_barcodes = [all_barcodes[i] for i in keep_indices]
    print(f"Matched: {len(keep_indices)} / {len(filtered)} barcodes in pileup output")

    for mtx in mtx_files:
        bc = args.pileup_dir / f"{mtx.stem}_barcodes.tsv"
        filter_matrix(mtx, bc, keep_indices, keep_barcodes, out_dir)

    # Copy stats unchanged
    stats = args.pileup_dir / "pileup_stats.json"
    if stats.exists():
        (out_dir / "pileup_stats.json").write_text(stats.read_text())

    print(f"Filtered output in: {out_dir}")


if __name__ == "__main__":
    main()
