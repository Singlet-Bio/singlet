#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/cospar_py_reference.py
#
# Reference implementation for fate/cospar correctness harness (cycle 41).
#
# Dispatch: called by fate_cospar_correctness.cpp via subprocess.
#
# Exit codes:
#   0  — success (reference written to --out_dir)
#   2  — cospar Python not installed AND pure-Python fallback produced output
#         (used as the GTEST_SKIP signal when fallback itself fails)
#   1  — unexpected error
#
# Primary path: cospar Python package (Klein Lab, Nature Biotech 2022).
#   pip install cospar
#   Computes transition_map + fate_bias using cospar.tmap API.
#
# Fallback path (always present): pure-Python implementation of the core
#   iterative optimizer from the design doc §Algorithm equations 4 (grad step +
#   soft-threshold + simplex projection), coupled with a state similarity
#   kernel from PCA + Gaussian kNN. Does not require cospar to be installed.
#   This ensures the harness always has a reference.
#
# Input files (from --data_dir, written by the C++ test):
#   expression.bin   — HostCSC binary (CSCC magic, see dump_csc.h)
#   time_point.bin   — uint32_t n + int32_t[n]
#   clone_id.bin     — uint32_t n + int32_t[n]
#   fate_ind_t1.bin  — uint32_t n + float[n]
#
# Output files (to --out_dir):
#   py_fate_bias_t0_fateA.bin  — uint32_t n + float[n]  (fate-A bias for t=0 cells)
#
# Usage:
#   python3 cospar_py_reference.py --task tiny_synthetic \
#       --data_dir /tmp/... --out_dir /tmp/... [--n_iter 50] [--n_knn 10]

import argparse
import struct
import sys
import os
import math

import numpy as np

# ---------------------------------------------------------------------------
# I/O helpers (matching dump_csc.h binary format)
# ---------------------------------------------------------------------------

def read_csc_bin(path):
    """Read a CSCC-format binary → (values, indptr, indices, n_rows, n_cols)."""
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x43535343, f"Bad CSCC magic: {magic:#010x}"
        n_rows = struct.unpack('<I', f.read(4))[0]
        n_cols = struct.unpack('<I', f.read(4))[0]
        nnz    = struct.unpack('<Q', f.read(8))[0]
        values  = np.frombuffer(f.read(nnz * 4),        dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols+1) * 4), dtype=np.int32  ).copy()
        indices = np.frombuffer(f.read(nnz * 4),        dtype=np.int32  ).copy()
    return values, indptr, indices, n_rows, n_cols


def read_i32_vec(path):
    """Read uint32_t n + int32_t[n]."""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        return np.frombuffer(f.read(n * 4), dtype=np.int32).copy()


def read_f32_vec(path):
    """Read uint32_t n + float[n]."""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def write_f32_vec(path, arr):
    """Write uint32_t n + float[n]."""
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(arr)))
        f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Pure-Python fallback: core Cospar optimizer
# (from design doc §Algorithm, equations 4)
#
# Inputs (numpy arrays):
#   L : (n_t0 × n_t1) lineage coupling matrix (float32)
#   K : (n_cells × n_cells) state similarity kernel (float32, sparse OK)
#   lambda1 : float  — L1 sparsity
#   lambda2 : float  — state coherence
#   step    : float  — gradient step size
#   n_iter  : int    — number of outer iterations
#
# The optimizer implements:
#   T_{k+1} = Prox(T_k - step * (2*(T_k - L) + 2*lambda2 * K_t0t0 @ T_k))
# where Prox = soft-threshold(·, lambda1 * step) + simplex projection per row.
#
# K_t0t0 is the (n_t0 × n_t0) sub-kernel for t=0 cells (regulation term).
# ---------------------------------------------------------------------------

def _simplex_project_row(v):
    """Project vector v onto the probability simplex in O(n log n)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n+1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.zeros_like(v)
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _soft_threshold(T, threshold):
    """Elementwise soft-threshold: sign(T) * max(|T| - threshold, 0)."""
    return np.sign(T) * np.maximum(np.abs(T) - threshold, 0.0)


def run_cospar_optimizer(L, K_t0, lambda1=0.05, lambda2=0.1, step=0.01, n_iter=50):
    """
    Core iterative optimizer for the transition map.

    Args:
        L      : (n_t0, n_t1) lineage coupling matrix.
        K_t0   : (n_t0, n_t0) state similarity sub-kernel for t=0 cells.
        lambda1: L1 penalty weight.
        lambda2: state coherence penalty weight.
        step   : gradient step size.
        n_iter : number of outer iterations.

    Returns:
        T : (n_t0, n_t1) transition map satisfying T >= 0, row-sums <= 1.
    """
    n_t0, n_t1 = L.shape
    T = L.copy().astype(np.float64)

    L_f64     = L.astype(np.float64)
    K_t0_f64  = K_t0.astype(np.float64)

    for _ in range(n_iter):
        # Gradient: 2*(T - L) + 2*lambda2 * (K_t0 @ T)
        grad = 2.0 * (T - L_f64) + 2.0 * lambda2 * (K_t0_f64 @ T)
        # Gradient step.
        T_hat = T - step * grad
        # Soft-threshold (L1 prox).
        T_hat = _soft_threshold(T_hat, lambda1 * step)
        # Non-negativity + simplex projection per row (row-sum <= 1).
        for i in range(n_t0):
            T_hat[i] = _simplex_project_row(T_hat[i])
        T = T_hat

    return T.astype(np.float32)


# ---------------------------------------------------------------------------
# State similarity kernel: PCA + Gaussian kNN
# ---------------------------------------------------------------------------

def build_state_kernel(expr_dense, n_knn=10, n_pcs=10):
    """
    Build a (n_cells × n_cells) Gaussian state similarity kernel.

    Args:
        expr_dense: (n_genes, n_cells) float32 array.
        n_knn     : number of nearest neighbours per cell.
        n_pcs     : number of PCA components.

    Returns:
        K : (n_cells, n_cells) float32 symmetric Gaussian kernel (sparse-ish).
    """
    from scipy.sparse import csc_matrix
    import numpy.linalg as LA

    n_genes, n_cells = expr_dense.shape
    X = expr_dense.T  # (n_cells, n_genes)

    # Mean-center.
    X = X - X.mean(axis=0, keepdims=True)

    # Truncated SVD (PCA).
    k = min(n_pcs, min(n_cells, n_genes) - 1)
    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=k, random_state=42)
        X_pca = svd.fit_transform(X)  # (n_cells, k)
    except ImportError:
        # Fall back to numpy SVD.
        U, S, Vt = LA.svd(X, full_matrices=False)
        X_pca = U[:, :k] * S[:k]

    # Pairwise distances → kNN Gaussian kernel.
    diff = X_pca[:, None, :] - X_pca[None, :, :]  # (n, n, k)
    dists2 = np.sum(diff ** 2, axis=-1)            # (n, n)

    # Adaptive bandwidth: median kNN distance per cell.
    knn_dists = np.sort(dists2, axis=1)[:, 1:n_knn+1]  # exclude self
    sigma2 = np.median(knn_dists, axis=1) + 1e-8        # (n,)

    # Gaussian kernel.
    K = np.exp(-dists2 / (sigma2[:, None] + sigma2[None, :]))

    # Keep only kNN entries (zero-out non-neighbours to sparsify).
    knn_mask = np.argsort(dists2, axis=1)[:, :n_knn+1]
    K_sparse = np.zeros_like(K)
    for i in range(n_cells):
        K_sparse[i, knn_mask[i]] = K[i, knn_mask[i]]
    # Symmetrize.
    K_sparse = (K_sparse + K_sparse.T) / 2.0
    return K_sparse.astype(np.float32)


# ---------------------------------------------------------------------------
# Lineage coupling matrix from clone_id + time_point
# ---------------------------------------------------------------------------

def build_lineage_coupling(time_point, clone_id):
    """
    Build L (n_t0 × n_t1) from clone labels.

    L[i, j] = 1 if cell i (at t=0) and cell j (at t=1) share a clone, else 0.
    Rows/columns index cells sorted by time point.
    """
    t0_idx = np.where(time_point == 0)[0]
    t1_idx = np.where(time_point == 1)[0]

    n_t0, n_t1 = len(t0_idx), len(t1_idx)
    L = np.zeros((n_t0, n_t1), dtype=np.float32)

    cl_t0 = clone_id[t0_idx]
    cl_t1 = clone_id[t1_idx]

    for j in range(n_t1):
        L[:, j] = (cl_t0 == cl_t1[j]).astype(np.float32)

    # Row-normalize so the initial guess sums to <= 1.
    row_sums = L.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    L = L / row_sums
    return L, t0_idx, t1_idx


# ---------------------------------------------------------------------------
# Fate bias: forward iteration T^k @ terminal_indicator
# ---------------------------------------------------------------------------

def compute_fate_bias(T, t0_clone, t1_clone, n_clones, n_iter_fb=5):
    """
    Compute fate_bias for t=0 cells.

    Returns:
        fate_bias : (n_t0, n_clones) float32 per-cell per-fate probability.
    """
    n_t0, n_t1 = T.shape
    # Terminal indicator: per t=1 cell, per fate (clone).
    term_ind = np.zeros((n_t1, n_clones), dtype=np.float32)
    for c in range(n_clones):
        term_ind[:, c] = (t1_clone == c).astype(np.float32)
    # fate_bias = T @ terminal_indicator (one hop).
    fate_bias = T @ term_ind  # (n_t0, n_clones)
    # Row-normalize.
    row_sums = fate_bias.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (fate_bias / row_sums).astype(np.float32)


# ---------------------------------------------------------------------------
# Main task: tiny_synthetic
# ---------------------------------------------------------------------------

def task_tiny_synthetic(args):
    data_dir = args.data_dir
    out_dir  = args.out_dir
    n_iter   = args.n_iter
    n_knn    = args.n_knn

    # Read inputs.
    values, indptr, indices, n_genes, n_cells = \
        read_csc_bin(os.path.join(data_dir, 'expression.bin'))
    time_point = read_i32_vec(os.path.join(data_dir, 'time_point.bin'))
    clone_id   = read_i32_vec(os.path.join(data_dir, 'clone_id.bin'))

    # Reconstruct dense expression (genes × cells).
    from scipy.sparse import csc_matrix
    expr_csc   = csc_matrix((values, indices, indptr), shape=(n_genes, n_cells))
    expr_dense = expr_csc.toarray()  # (n_genes, n_cells)

    # --- Try cospar Python first ---
    try:
        import cospar as cs
        import anndata as ad

        # Build AnnData for cospar API.
        adata = ad.AnnData(X=expr_dense.T.astype(np.float32))  # cells × genes
        adata.obs['time_info']  = [str(t) for t in time_point]
        adata.obs['state_info'] = ['c' + str(c) for c in clone_id]

        cs.settings.verbosity = 0

        # Preprocess and compute transition map.
        cs.pp.initialize_adata_object(adata=adata)
        cs.tmap.infer_Tmap_from_clonal_info_alone(
            adata,
            selected_fates=['c0', 'c1', 'c2'],
            plot=False,
        )

        # Extract fate_bias for t=0 cells.
        t0_mask   = time_point == 0
        n_t0      = int(t0_mask.sum())
        n_clones  = int(clone_id.max()) + 1

        if hasattr(adata, 'obs') and 'fate_bias' in adata.obs.columns:
            # Use cospar's own fate_bias if available.
            fate_bias_col = 'fate_bias'
            fate_a = adata.obs.loc[t0_mask, fate_bias_col].values.astype(np.float32)
        else:
            # Fall back to extracting from tmap output manually.
            T = adata.uns.get('Tmap_cell_to_cell', None)
            if T is None:
                raise ValueError("cospar did not produce Tmap_cell_to_cell")
            T_arr = np.array(T)[np.ix_(np.where(t0_mask)[0],
                                         np.where(~t0_mask)[0])]
            n_t1 = int((~t0_mask).sum())
            cl_t1 = clone_id[~t0_mask]
            term = np.zeros((n_t1, n_clones), dtype=np.float32)
            for c in range(n_clones):
                term[:, c] = (cl_t1 == c).astype(np.float32)
            fb = T_arr @ term
            rs = fb.sum(axis=1, keepdims=True)
            rs = np.where(rs == 0, 1.0, rs)
            fb = fb / rs
            fate_a = fb[:, 0].astype(np.float32)

        write_f32_vec(os.path.join(out_dir, 'py_fate_bias_t0_fateA.bin'), fate_a)
        print(f"[cospar_py_reference] cospar Python: n_t0={n_t0}, "
              f"fate_a range=[{fate_a.min():.3f}, {fate_a.max():.3f}]")
        return 0

    except ImportError:
        pass  # cospar not installed; fall through to pure-Python fallback

    # --- Pure-Python fallback ---
    print("[cospar_py_reference] cospar not available; using pure-Python fallback")

    t0_idx = np.where(time_point == 0)[0]
    t1_idx = np.where(time_point == 1)[0]
    n_t0, n_t1 = len(t0_idx), len(t1_idx)
    n_clones = int(clone_id.max()) + 1

    # State similarity kernel for t=0 cells only (for the K_t0 term).
    expr_t0 = expr_dense[:, t0_idx]  # (n_genes, n_t0)
    K_t0 = build_state_kernel(expr_t0, n_knn=n_knn, n_pcs=min(10, n_t0-1))
    K_t0 = K_t0[:n_t0, :n_t0]       # (n_t0, n_t0)

    # Lineage coupling matrix L (n_t0 × n_t1).
    cl_t0 = clone_id[t0_idx]
    cl_t1 = clone_id[t1_idx]
    L = np.zeros((n_t0, n_t1), dtype=np.float32)
    for j in range(n_t1):
        L[:, j] = (cl_t0 == cl_t1[j]).astype(np.float32)
    rs_L = L.sum(axis=1, keepdims=True)
    rs_L = np.where(rs_L == 0, 1.0, rs_L)
    L = L / rs_L

    # Run optimizer.
    T = run_cospar_optimizer(L, K_t0,
                              lambda1=0.05, lambda2=0.1, step=0.01,
                              n_iter=n_iter)

    # Fate bias: T @ terminal_indicator.
    fate_bias = compute_fate_bias(T, cl_t0, cl_t1, n_clones)
    fate_a = fate_bias[:, 0]  # fate-A = clone 0

    write_f32_vec(os.path.join(out_dir, 'py_fate_bias_t0_fateA.bin'), fate_a)
    print(f"[cospar_py_reference] pure-Python fallback: n_t0={n_t0}, "
          f"fate_a range=[{fate_a.min():.3f}, {fate_a.max():.3f}]")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cospar Python reference for singlet-gpu correctness harness')
    parser.add_argument('--task',     required=True,
                        choices=['tiny_synthetic'],
                        help='Which reference task to run')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing input binary files')
    parser.add_argument('--out_dir',  required=True,
                        help='Directory to write output binary files')
    parser.add_argument('--n_iter',   type=int, default=50,
                        help='Number of optimizer iterations (default 50)')
    parser.add_argument('--n_knn',    type=int, default=10,
                        help='kNN for state kernel (default 10)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        if args.task == 'tiny_synthetic':
            rc = task_tiny_synthetic(args)
            sys.exit(rc)
        else:
            print(f"Unknown task: {args.task}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[cospar_py_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
