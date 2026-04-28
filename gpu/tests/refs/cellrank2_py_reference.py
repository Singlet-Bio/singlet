#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/cellrank2_py_reference.py
#
# Reference implementation for fate/cellrank2 correctness harness (cycle 43).
#
# Dispatch: called by fate_cellrank2_correctness.cpp via subprocess.
#
# Exit codes:
#   0  — success (reference written to --out_dir)
#   2  — CellRank 2 Python not installed; scipy.sparse.linalg.gmres fallback ran
#   1  — unexpected error
#
# Primary path: CellRank 2 Python package (Weiler et al. Nature Methods 2024).
#   pip install cellrank anndata scvelo
#   Computes absorption_prob via CellRank 2 GPCCA / absorption-probability API.
#
# Fallback path (always present): direct scipy sparse linear solve of
#   (I - Q) @ B = R
# using scipy.sparse.linalg.gmres for each terminal state RHS.
# This fallback is always available and produces the canonical reference.
#
# Input files (from --data_dir, written by the C++ test):
#   transition.bin    — sparse CSR transition matrix T (n_cells × n_cells):
#                       [uint32_t n_rows][uint32_t n_cols][uint64_t nnz]
#                       [float×nnz values][int32_t×(n+1) indptr][int32_t×nnz indices]
#   terminal_ids.bin  — uint32_t n_terminals + int32_t[n_terminals]
#
# Output files (to --out_dir):
#   py_absorption_prob.bin — [uint32_t n_cells][uint32_t n_terminals]
#                            [float × n_cells × n_terminals] (row-major)
#
# Usage:
#   python3 cellrank2_py_reference.py --task tiny_synthetic \
#       --data_dir /tmp/... --out_dir /tmp/...

import argparse
import struct
import sys
import os

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csr_bin(path):
    """Read CSR binary format written by C++ test."""
    with open(path, 'rb') as f:
        n_rows    = struct.unpack('<I', f.read(4))[0]
        n_cols    = struct.unpack('<I', f.read(4))[0]
        nnz       = struct.unpack('<Q', f.read(8))[0]
        values  = np.frombuffer(f.read(nnz * 4),           dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_rows + 1) * 4),  dtype=np.int32  ).copy()
        indices = np.frombuffer(f.read(nnz * 4),           dtype=np.int32  ).copy()
    return values, indptr, indices, n_rows, n_cols


def read_i32_vec(path):
    """Read [uint32_t n][int32_t × n]."""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        return np.frombuffer(f.read(n * 4), dtype=np.int32).copy()


def write_absorption_prob(path, prob):
    """Write [uint32_t n_cells][uint32_t n_terms][float × n_cells × n_terms]."""
    n_cells, n_terms = prob.shape
    prob32 = np.asarray(prob, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', n_cells))
        f.write(struct.pack('<I', n_terms))
        f.write(prob32.tobytes())


# ---------------------------------------------------------------------------
# Scipy GMRES fallback: solve (I - Q) @ B = R for each terminal RHS
#
# Inputs:
#   T            : scipy CSR (n_cells × n_cells) row-stochastic transition matrix
#   terminal_ids : 1-D int32 array of absorbing state indices (length n_terminals)
#
# Returns:
#   absorption_prob : (n_cells × n_terminals) float64 array
#                     Row sums ≈ 1; identity rows for absorbing cells.
# ---------------------------------------------------------------------------

def solve_absorption_gmres(T, terminal_ids):
    from scipy.sparse import eye as speye, diags
    from scipy.sparse.linalg import gmres

    n = T.shape[0]
    n_terms = len(terminal_ids)
    terminal_set = set(int(t) for t in terminal_ids)
    transient_idx = np.array([i for i in range(n) if i not in terminal_set],
                              dtype=np.int32)
    absorbing_idx = np.array(sorted(terminal_set), dtype=np.int32)

    # Map from original index → terminal column index (for R columns).
    term_col = {int(t): c for c, t in enumerate(absorbing_idx)}

    n_tr = len(transient_idx)
    absorption_prob = np.zeros((n, n_terms), dtype=np.float64)

    # Identity rows for absorbing states.
    for c, t in enumerate(absorbing_idx):
        # Find which column in absorption_prob this terminal owns.
        col = next(i for i, tid in enumerate(terminal_ids) if int(tid) == int(t))
        absorption_prob[int(t), col] = 1.0

    if n_tr == 0:
        return absorption_prob.astype(np.float32)

    # Q = T[transient, :][:, transient]  (n_tr × n_tr)
    Q = T[transient_idx, :][:, transient_idx]

    # R = T[transient, :][:, absorbing]  (n_tr × n_terms)
    R = T[transient_idx, :][:, absorbing_idx]

    # Coefficient matrix A = I - Q
    A = speye(n_tr, format='csr') - Q

    # Solve A @ b = R[:, k] for each terminal k.
    for k in range(n_terms):
        rhs = np.asarray(R[:, k].todense(), dtype=np.float64).ravel()
        x0  = np.zeros(n_tr, dtype=np.float64)
        sol, info = gmres(A, rhs, x0=x0, rtol=1e-6, maxiter=300,
                          restart=30)
        if info != 0:
            print(f"[cellrank2_py_reference] GMRES warning: terminal {k}, "
                  f"info={info} (non-convergence)", file=sys.stderr)
        # Map which original terminal_ids column this absorbing_idx[k] maps to.
        orig_col = next(i for i, tid in enumerate(terminal_ids)
                        if int(tid) == int(absorbing_idx[k]))
        absorption_prob[transient_idx, orig_col] = np.maximum(sol, 0.0)

    return absorption_prob.astype(np.float32)


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------

def run_task(args):
    data_dir = args.data_dir
    out_dir  = args.out_dir

    # Read inputs.
    values, indptr, indices, n_rows, n_cols = \
        read_csr_bin(os.path.join(data_dir, 'transition.bin'))
    terminal_ids = read_i32_vec(os.path.join(data_dir, 'terminal_ids.bin'))

    from scipy.sparse import csr_matrix
    T = csr_matrix((values, indices, indptr), shape=(n_rows, n_cols))

    # --- Try CellRank 2 Python first ---
    use_cr2 = False
    try:
        import cellrank as cr
        import anndata as ad
        import numpy as np

        # Build AnnData with the transition matrix stored as a connectivities
        # approximation (CellRank 2 accepts pre-built T matrices via
        # cr.kernels.PrecomputedKernel).
        adata = ad.AnnData(X=np.eye(n_rows, dtype=np.float32))
        kernel = cr.kernels.PrecomputedKernel(T, adata=adata)
        estimator = cr.estimators.GPCCA(kernel)

        n_terms = len(terminal_ids)
        estimator.compute_schur(n_components=min(n_terms + 2, n_rows - 1))
        estimator.predict_terminal_states(n_states=n_terms)
        estimator.compute_absorption_probabilities(solver='gmres',
                                                   tol=1e-6, maxiter=300)

        abs_prob = estimator.absorption_probabilities.values  # (n_cells, n_terms)
        absorption_prob = np.asarray(abs_prob, dtype=np.float32)
        use_cr2 = True
        print(f"[cellrank2_py_reference] CellRank 2 path: n_cells={n_rows}, "
              f"n_terminals={n_terms}, "
              f"row_sum range=[{absorption_prob.sum(axis=1).min():.4f}, "
              f"{absorption_prob.sum(axis=1).max():.4f}]")

    except (ImportError, Exception) as e:
        if not isinstance(e, ImportError):
            print(f"[cellrank2_py_reference] CellRank 2 failed ({e}); "
                  f"falling back to scipy GMRES", file=sys.stderr)

    if not use_cr2:
        print("[cellrank2_py_reference] CellRank 2 not available; "
              "using scipy GMRES fallback")
        absorption_prob = solve_absorption_gmres(T, terminal_ids)
        n_terms = len(terminal_ids)
        rs = absorption_prob.sum(axis=1)
        print(f"[cellrank2_py_reference] scipy GMRES fallback: n_cells={n_rows}, "
              f"n_terminals={n_terms}, "
              f"row_sum range=[{rs.min():.4f}, {rs.max():.4f}]")

    write_absorption_prob(os.path.join(out_dir, 'py_absorption_prob.bin'),
                          absorption_prob)

    rc = 0 if use_cr2 else 2
    return rc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CellRank 2 Python reference for singlet-gpu correctness harness')
    parser.add_argument('--task',     required=True,
                        choices=['tiny_synthetic', 'gsm_real'],
                        help='Which reference task to run')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing input binary files')
    parser.add_argument('--out_dir',  required=True,
                        help='Directory to write output binary files')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        rc = run_task(args)
        sys.exit(rc)
    except Exception as e:
        print(f"[cellrank2_py_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
