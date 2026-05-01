#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/sparse_eig_reference.py
#
# Reference driver for core/sparse_eigensolver_correctness.cpp — Test 2
# (ScaleSmokeMedium, n=10000, K=15).
#
# Generates a random sparse symmetric positive-definite matrix (same construction
# as the C++ test helper build_random_sparse_symmetric), runs scipy ARPACK eigsh,
# and saves eigenvalues + eigenvectors + the CSR arrays to:
#
#   /tmp/singlet_gpu_sparse_eig_refs_tmp/sparse_eig_medium_ref.npz
#
# The C++ test loads this file via Python one-liner extraction; if the file is
# absent the test is GTEST_SKIP()'d with a descriptive message.
#
# Environment: pip install numpy scipy
#
# Usage:
#   python3 tests/refs/sparse_eig_reference.py
#   python3 tests/refs/sparse_eig_reference.py --n 10000 --k 15 --seed 0xC0FFEE
#   python3 tests/refs/sparse_eig_reference.py --output /custom/path/ref.npz

import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def build_sparse_symmetric(n: int, sparsity: float, diag_shift: float,
                            seed: int) -> csr_matrix:
    """
    Build a random sparse symmetric PD matrix matching the C++ helper:
      build_random_sparse_symmetric(n, sparsity, diag_shift, seed).

    Uses the same MT19937-64 bit-pattern via numpy (which also uses MT19937
    internally; the construction is order-identical to the C++ code's row-major
    upper-triangular scan with the same seed).

    NOTE: numpy's MT19937 seed path and C++ std::mt19937_64 are NOT
    bit-for-bit identical, so the matrix will differ in exact entries.
    The important invariant is that BOTH sides use the *same* npz file —
    the C++ test loads the CSR arrays from this script's output, so the
    matrix seen by LOBPCG and eigsh is identical.
    """
    rng = np.random.default_rng(seed)

    # Upper-triangular random edges.
    n_possible = n * (n - 1) // 2
    mask  = rng.random(n_possible) < sparsity
    vals  = rng.uniform(0.1, 1.0, n_possible).astype(np.float32)

    rows, cols, data = [], [], []
    idx = 0
    for r in range(n):
        for c in range(r + 1, n):
            if mask[idx]:
                v = float(vals[idx])
                rows += [r, c]
                cols += [c, r]
                data += [v, v]
            idx += 1

    # Add diagonal shift for PD.
    rows += list(range(n))
    cols += list(range(n))
    data += [diag_shift] * n

    A = csr_matrix((np.array(data, dtype=np.float32),
                    (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                   shape=(n, n))
    A.sort_indices()
    A.sum_duplicates()
    return A


def main():
    parser = argparse.ArgumentParser(
        description="scipy ARPACK eigsh reference for core/sparse_eigensolver Test 2")
    parser.add_argument("--n",       type=int,   default=10000,
                        help="Matrix dimension (default 10000)")
    parser.add_argument("--k",       type=int,   default=15,
                        help="Number of eigenpairs (default 15)")
    parser.add_argument("--sparsity", type=float, default=0.001,
                        help="Off-diagonal fill fraction (default 0.001 → ~100k nnz)")
    parser.add_argument("--diag-shift", type=float, default=10.0,
                        help="Diagonal shift for positive definiteness (default 10.0)")
    parser.add_argument("--seed",    type=lambda x: int(x, 0), default=0xC0FFEE,
                        help="RNG seed (default 0xC0FFEE)")
    parser.add_argument("--output",  type=str,
                        default="/tmp/singlet_gpu_sparse_eig_refs_tmp/"
                                "sparse_eig_medium_ref.npz",
                        help="Output .npz path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[ref] Building sparse symmetric matrix n={args.n}, "
          f"sparsity={args.sparsity}, diag_shift={args.diag_shift}, seed={args.seed:#x}",
          file=sys.stderr)
    t0 = time.time()
    A = build_sparse_symmetric(args.n, args.sparsity, args.diag_shift, args.seed)
    print(f"[ref] Matrix built: nnz={A.nnz} ({time.time()-t0:.1f}s)", file=sys.stderr)

    print(f"[ref] Running eigsh(k={args.k}, which='LA') ...", file=sys.stderr)
    t1 = time.time()
    eigenvalues, eigenvectors = eigsh(A, k=args.k, which='LA', tol=1e-10,
                                      maxiter=5000, return_eigenvectors=True)
    print(f"[ref] eigsh done ({time.time()-t1:.1f}s)", file=sys.stderr)

    # eigsh returns eigenvalues in ascending order; reverse so index 0 = largest.
    eigenvalues  = eigenvalues[::-1].astype(np.float32)
    eigenvectors = eigenvectors[:, ::-1].astype(np.float32)   # shape (n, k), col-major
    # Flatten col-major: the C++ side stores eigenvectors [n × K] col-major.
    # np stores arrays row-major by default; to get col-major flat layout:
    eigenvectors_flat = np.asfortranarray(eigenvectors).flatten(order='F')

    # CSR arrays for the exact same matrix (so C++ test uses identical A).
    csr_values  = A.data.astype(np.float32)
    csr_row_ptr = A.indptr.astype(np.int32)
    csr_col_idx = A.indices.astype(np.int32)

    np.savez(args.output,
             eigenvalues=eigenvalues,
             eigenvectors=eigenvectors_flat,
             csr_values=csr_values,
             csr_row_ptr=csr_row_ptr,
             csr_col_idx=csr_col_idx)

    print(f"[ref] Saved to {args.output}", file=sys.stderr)
    print(f"[ref] Top-3 eigenvalues: {eigenvalues[:3]}", file=sys.stderr)
    print(f"[ref] Min eigenvalue (K-th): {eigenvalues[-1]:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
