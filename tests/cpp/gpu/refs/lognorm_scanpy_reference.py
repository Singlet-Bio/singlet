#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/lognorm_scanpy_reference.py
#
# Reference driver: run scanpy normalize_total + log1p on a matrix produced by
# the C++ test, write the expected result as a .npy float64 array.
#
# Environment: pip install scanpy scipy numpy anndata
#
# Usage:
#   python lognorm_scanpy_reference.py \
#       --input  <input.bin>   \  # dump_csc.h binary format
#       --output <expected.npy>   # numpy float32 array, shape (n_rows, n_cols)
#       [--target-sum <float>]    # optional; if omitted, scanpy uses median
#
# The C++ test calls this as a subprocess and then reads <expected.npy> back.

import argparse
import struct
import sys
import numpy as np
import scipy.sparse as sp

def read_csc_bin(path: str):
    """Read the compact binary produced by tests/refs/dump_csc.h."""
    with open(path, 'rb') as f:
        raw = f.read()

    offset = 0
    magic, n_rows, n_cols = struct.unpack_from('<III', raw, offset); offset += 12
    if magic != 0x43535343:
        raise ValueError(f"Bad magic: 0x{magic:08x} (expected 0x43535343)")
    (nnz,) = struct.unpack_from('<Q', raw, offset); offset += 8

    values  = np.frombuffer(raw, dtype=np.float32,
                            count=nnz, offset=offset).copy(); offset += nnz * 4
    indptr  = np.frombuffer(raw, dtype=np.int32,
                            count=n_cols + 1, offset=offset).copy(); offset += (n_cols + 1) * 4
    indices = np.frombuffer(raw, dtype=np.int32,
                            count=nnz, offset=offset).copy()

    mat = sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))
    return mat, n_rows, n_cols


def run_scanpy_reference(csc_mat, target_sum=None):
    """
    Apply scanpy normalize_total (library-size normalization to median or
    target_sum) followed by log1p, exactly as scanpy does internally.

    scanpy operates in fp64 internally for normalize_total; log1p also uses
    fp64.  We preserve fp64 here to serve as the high-precision reference for
    the fp32 GPU kernel.

    Returns: dense float64 array, shape (n_cells, n_genes)  [AnnData convention:
    rows = cells, cols = genes], then transposed back to (n_genes, n_cells) to
    match singlet-gpu's (rows=genes, cols=cells) CSC convention.
    """
    import anndata as ad
    import scanpy as sc

    # AnnData convention: rows = cells, cols = genes.  Our CSC is genes x cells.
    # Transpose before feeding to scanpy; transpose result back.
    csc_t = csc_mat.T.tocsc()  # cells x genes, still sparse

    adata = ad.AnnData(X=csc_t.astype(np.float64))

    # normalize_total with target_sum=None uses the median library size
    # (scanpy default).  Passing an explicit float uses that value directly —
    # this matches LogNormConfig::target_count in the C++ design.
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata)

    # Back to (n_genes x n_cells) to match GPU output convention.
    result = adata.X.T  # shape (n_genes, n_cells)
    if sp.issparse(result):
        result = result.toarray()
    return result.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="scanpy normalize_total + log1p reference for lognorm correctness test")
    parser.add_argument('--input',  required=True,
                        help='Path to dump_csc binary (.bin)')
    parser.add_argument('--output', required=True,
                        help='Path to write numpy float64 array (.npy)')
    parser.add_argument('--target-sum', type=float, default=None,
                        help='If set, use this as target_sum instead of median')
    args = parser.parse_args()

    csc_mat, n_rows, n_cols = read_csc_bin(args.input)
    print(f"[ref] Loaded CSC: {n_rows} rows x {n_cols} cols, nnz={csc_mat.nnz}",
          file=sys.stderr)

    result = run_scanpy_reference(csc_mat, target_sum=args.target_sum)
    print(f"[ref] Reference shape: {result.shape}, dtype={result.dtype}",
          file=sys.stderr)

    np.save(args.output, result)
    print(f"[ref] Saved to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
