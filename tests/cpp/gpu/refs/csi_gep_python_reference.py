#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/csi_gep_python_reference.py
#
# Reference implementation driver for cycle 28 CSI-GEP correctness harness.
#
# Loads a CSC binary file (dump_csc.h format) produced by the C++ test fixture,
# constructs an AnnData object, runs the CSI-GEP Python reference implementation
# (geeleherlab/CSI-GEP), and writes consensus programs + chosen_k to binary
# files for the C++ test to compare against.
#
# Usage (called by reduce_nmf_csi_gep_correctness.cpp via subprocess):
#   python3 csi_gep_python_reference.py \
#       --csc            <matrix.bin>        \
#       --k-range        3,4,5,6,7           \
#       --n-runs         10                  \
#       --seed           <S>                 \
#       --out-programs   <programs.bin>      \
#       --out-chosen-k   <chosen_k.bin>
#
# Exit codes:
#   0  success — programs.bin and chosen_k.bin written
#   2  csi-gep (or a required dependency) not installed → GTEST_SKIP
#   1  any other error → GTEST_FAIL
#
# Output files:
#   programs.bin   float32[n_genes * chosen_k], column-major (genes vary fastest)
#   chosen_k.bin   int32 scalar
#
# Dependencies (heavy install):
#   pip install csi-gep anndata numpy scipy
#   (csi-gep pulls in torchNMF + sklearn transitively)
#
# Note: CSI-GEP is run with reduced n_runs for test speed.  The C++ test
# passes n_runs=10 which is sufficient for correctness on planted-program data.
#
# Robustness: if csi_gep is importable but internal errors occur (e.g., torch
# not available at runtime), the script exits 1 (FAIL) not 2 (SKIP), so the
# test surfaces a real failure rather than silently skipping.

import sys
import argparse
import struct
import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# Availability check — must happen before heavy imports so exit(2) is fast.
# ---------------------------------------------------------------------------

def _check_dependencies():
    missing = []
    for pkg in ("csi_gep", "anndata", "numpy", "scipy"):
        try:
            __import__(pkg)
        except ImportError as e:
            missing.append(str(e))
    if missing:
        print(f"[csi_gep_python_reference] SKIP: {'; '.join(missing)}", file=sys.stderr)
        sys.exit(2)


_check_dependencies()


# ---------------------------------------------------------------------------
# Imports (only after availability check)
# ---------------------------------------------------------------------------

import anndata  # noqa: E402


# ---------------------------------------------------------------------------
# CSC binary reader (dump_csc.h format)
#
# Format (little-endian, no padding):
#   uint32  magic      = 0x43535343 ("CSCC")
#   uint32  n_rows     (genes)
#   uint32  n_cols     (cells)
#   uint64  nnz
#   float32[nnz]       values
#   int32[n_cols+1]    indptr
#   int32[nnz]         indices
# ---------------------------------------------------------------------------

def read_csc_bin(path: str) -> scipy.sparse.csc_matrix:
    with open(path, 'rb') as f:
        magic, n_rows, n_cols = struct.unpack('<III', f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic in {path}: 0x{magic:08X}")
        (nnz,) = struct.unpack('<Q', f.read(8))
        values  = np.frombuffer(f.read(nnz * 4),          dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4), dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),          dtype=np.int32).copy()
    return scipy.sparse.csc_matrix(
        (values, indices, indptr), shape=(n_rows, n_cols)
    )


# ---------------------------------------------------------------------------
# CSI-GEP wrapper
#
# The csi_gep package exposes:
#   csi_gep.csi_gep(adata, k_range, n_runs, subsample_frac, seed, ...)
#
# Returns an AnnData with:
#   adata.varm["csi_gep_programs"]  — genes × chosen_k  (ndarray, float32)
#   adata.uns["csi_gep_chosen_k"]   — int
#   adata.uns["csi_gep_repro"]      — reproducibility curve (list/array)
#
# We access via public API only; no internal imports.
# ---------------------------------------------------------------------------

def run_csi_gep(
    csc: scipy.sparse.csc_matrix,
    k_range: list,
    n_runs: int,
    seed: int,
) -> tuple:
    """
    Returns (programs_colmajor: np.ndarray[float32], chosen_k: int)
    programs_colmajor shape: (n_genes, chosen_k) — same layout as C++ output.
    """
    import csi_gep as cg  # noqa: PLC0415  (import inside function is intentional)

    # CSI-GEP expects cells × genes (transposed relative to our CSC).
    X = csc.T.toarray().astype(np.float32)  # cells × genes, dense
    adata = anndata.AnnData(X)

    # Run CSI-GEP.  reduced n_runs for test speed.
    cg.csi_gep(
        adata,
        k_range=k_range,
        n_runs=n_runs,
        seed=seed,
    )

    # Extract results from AnnData.
    programs = adata.varm.get("csi_gep_programs")
    chosen_k = adata.uns.get("csi_gep_chosen_k")

    if programs is None:
        raise RuntimeError("csi_gep did not write 'csi_gep_programs' to adata.varm")
    if chosen_k is None:
        raise RuntimeError("csi_gep did not write 'csi_gep_chosen_k' to adata.uns")

    programs = np.asarray(programs, dtype=np.float32)  # genes × chosen_k
    chosen_k = int(chosen_k)

    # Ensure shape is (n_genes, chosen_k).
    if programs.ndim != 2:
        raise RuntimeError(f"csi_gep_programs has unexpected ndim={programs.ndim}")
    if programs.shape[1] != chosen_k:
        # Some versions of csi-gep return (chosen_k, n_genes); transpose if so.
        if programs.shape[0] == chosen_k:
            programs = programs.T
        else:
            raise RuntimeError(
                f"csi_gep_programs shape {programs.shape} incompatible with chosen_k={chosen_k}"
            )

    return programs, chosen_k


# ---------------------------------------------------------------------------
# Binary writers
# ---------------------------------------------------------------------------

def write_float_bin(arr: np.ndarray, path: str):
    """Write a float32 ndarray as a flat little-endian binary (no header)."""
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(arr.tobytes())


def write_int32_scalar(value: int, path: str):
    """Write a single int32 in little-endian."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', value))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CSI-GEP Python reference for singlet-gpu cycle 28 tests"
    )
    parser.add_argument("--csc",          required=True,
                        help="Path to CSC binary (dump_csc.h format)")
    parser.add_argument("--k-range",      required=True,
                        help="Comma-separated k values, e.g. 3,4,5,6,7")
    parser.add_argument("--n-runs",       type=int, default=10,
                        help="Number of NMF runs per k (default 10)")
    parser.add_argument("--seed",         type=int, default=0xC0FFEE,
                        help="Random seed")
    parser.add_argument("--out-programs", required=True,
                        help="Output path for flat float32[n_genes*chosen_k]")
    parser.add_argument("--out-chosen-k", required=True,
                        help="Output path for int32 chosen_k scalar")
    args = parser.parse_args()

    k_range = [int(x.strip()) for x in args.k_range.split(",")]
    if not k_range:
        print("[csi_gep_python_reference] ERROR: empty k-range", file=sys.stderr)
        sys.exit(1)

    # Load matrix.
    csc = read_csc_bin(args.csc)
    print(
        f"[csi_gep_python_reference] Loaded matrix: {csc.shape[0]} genes × {csc.shape[1]} cells, "
        f"nnz={csc.nnz}",
        file=sys.stderr,
    )

    # Run CSI-GEP.
    print(
        f"[csi_gep_python_reference] Running CSI-GEP: k_range={k_range}, n_runs={args.n_runs}",
        file=sys.stderr,
    )
    programs, chosen_k = run_csi_gep(csc, k_range, args.n_runs, args.seed)
    print(
        f"[csi_gep_python_reference] Done: chosen_k={chosen_k}, "
        f"programs shape={programs.shape}",
        file=sys.stderr,
    )

    # Validate output.
    if not np.all(np.isfinite(programs)):
        print("[csi_gep_python_reference] ERROR: NaN/Inf in programs", file=sys.stderr)
        sys.exit(1)

    # Write outputs.
    write_float_bin(programs, args.out_programs)
    write_int32_scalar(chosen_k, args.out_chosen_k)

    print(
        f"[csi_gep_python_reference] Written: programs→{args.out_programs}, "
        f"chosen_k→{args.out_chosen_k}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
