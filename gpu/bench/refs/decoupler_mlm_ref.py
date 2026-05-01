#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/decoupler_mlm_ref.py
#
# CYCLE-165 Phase E — manual numpy/scipy CPU baseline for enrich/decoupler_mlm.
#
# GPU kernel: 5 passes — cuBLAS Sgemm (W^T W), ridge kernel, cuSPARSE SpMM
#             (X^T · W), cuSOLVER Spotrf/Spotrs (Cholesky solve), cuBLAS Sgeam.
# Reference:  Badia-i-Mompel et al. (2022) decoupleR, Bioinformatics Advances 2:vbac016.
#
# MLM (Multivariate Linear Model): jointly regress all p pathways on each cell:
#
#   X[:, c] = W · β_c + ε    (over all m genes)
#
# Closed-form OLS (vectorized over all n_cells):
#
#   B = (W^T W + ridge*I)^{-1} (W^T X)
#
# where B is (p × n_cells); each column is the coefficient vector for one cell.
#
# CPU implementation using scipy Cholesky factorization:
#   1. Compute WtW = W.T @ W                          (p × p)
#   2. Add ridge to diagonal: WtW += ridge * I_p
#   3. Cholesky factor: (c_factor, lower) = scipy.linalg.cho_factor(WtW)
#   4. Compute RHS: WtX = W.T @ X.T                  (p × n_cells); X.T is (n_genes × n_cells)
#      — equivalently WtX = (X @ W).T since X is (n_cells × n_genes)
#   5. Solve: B = scipy.linalg.cho_solve((c_factor, lower), WtX)  → (p × n_cells)
#   6. Transpose B to (n_cells × p) for output convention matching GPU.
#
# NOTE: `decoupler` Python package is NOT installed on the cluster.
#       This is a MANUAL numpy/scipy implementation of decoupleR's MLM.
#
# Input generation:
#   X — scipy.sparse.random(n_cells, n_genes, density=0.05, format='csc',
#                           dtype=float32, random_state=42)
#       CSC layout (matches GPU bench input).
#       NOTE: X is (n_cells × n_genes) — use X @ W not X.T @ W.
#   W — np.random.default_rng(42).standard_normal(n_genes, n_pathways).astype(float32)
#       Dense col-major weight matrix (n_genes × n_pathways).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,density,n_pathways,wall_ms
#
# Usage: python3 decoupler_mlm_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_PATHWAYS   = 50
RIDGE        = 1e-6  # matches MlmConfig default

SCALES = [
    ("10k",  10000, 5000, 0.05),
    ("30k",  30000, 5000, 0.05),
]


def make_synthetic_inputs(n_cells: int, n_genes: int, density: float,
                           n_pathways: int, seed: int = 42):
    """Build synthetic sparse X (CSC, n_cells × n_genes) and dense W (n_genes × n_pathways).

    X: scipy.sparse.random with seed=42 for parity with GPU bench.
    W: standard normal via np.random.default_rng(42).

    NOTE: X is (n_cells × n_genes), W is (n_genes × n_pathways).
    SpMM equivalent: X @ W → (n_cells × n_pathways). NOT X.T @ W.
    """
    rng = np.random.default_rng(seed)
    # scipy.sparse.random random_state accepts int or np.random.Generator (scipy >=1.9).
    X = sp.random(n_cells, n_genes, density=density, format="csc",
                  dtype=np.float32, random_state=seed)

    W = rng.standard_normal((n_genes, n_pathways)).astype(np.float32)
    return X, W


def time_mlm(X, W):
    """2 warmup + 5 timed iters of closed-form MLM (scipy Cholesky).

    Computes multivariate OLS B = (W^T W + ridge*I)^{-1} W^T X
    for all cells simultaneously.

    Steps (matching GPU 5-pass kernel):
      1. WtW = W.T @ W                            (p × p) — Sgemm
      2. WtW += ridge * I                          — ridge kernel
      3. cho_factor(WtW)                           — Spotrf
      4. WtX = (X @ W).T                          (p × n_cells) — SpMM + transpose
      5. B   = cho_solve((c, lower), WtX)          — Spotrs
      6. scores = B.T                              (n_cells × p) — Sgeam

    Returns median wall_ms over TIMED_ITERS.
    """
    n_cells, n_genes = X.shape
    p = W.shape[1]

    # Pre-compute Cholesky factor once (A = W^T W + ridge*I).
    # In GPU bench these are inside the timed call; we time everything
    # including WtW + Cholesky + SpMM + solve each iteration.
    # (GPU bench also recomputes WtW each timed iteration.)

    def _run():
        # Pass 1+2: W^T W + ridge diagonal
        WtW = W.T @ W                                    # (p × p) float32
        WtW += RIDGE * np.eye(p, dtype=np.float32)

        # Pass 4 (Cholesky factor): A = L L^T
        c_factor = sla.cho_factor(WtW, lower=True)       # Spotrf equivalent

        # Pass 3 (SpMM): X @ W → (n_cells × p), then transpose to (p × n_cells)
        XW = X @ W                                       # (n_cells × p) sparse×dense
        if sp.issparse(XW):
            XW = XW.toarray()
        WtX = XW.T                                       # (p × n_cells) col-major

        # Pass 4b (Spotrs): solve (W^T W + ridge*I) B = W^T X
        B = sla.cho_solve(c_factor, WtX)                 # (p × n_cells)

        # Pass 5 (transpose): B^T → (n_cells × p) output convention
        scores = B.T                                     # (n_cells × p)
        return scores

    # Warmup
    for _ in range(WARMUP_ITERS):
        try:
            _run()
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _run()
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n = len(wall_times)
    return (wall_times[n // 2] if n % 2 == 1
            else 0.5 * (wall_times[n // 2 - 1] + wall_times[n // 2]))


def main():
    print("scale,n_cells,n_genes,density,n_pathways,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_genes} genes "
              f"density={density:.0%} n_pathways={N_PATHWAYS}...",
              file=sys.stderr, flush=True)

        X, W = make_synthetic_inputs(n_cells, n_genes, density, N_PATHWAYS, seed=42)

        print(f"[ref] Timing {scale_name} method='mlm'...",
              file=sys.stderr, flush=True)
        try:
            wall_ms = time_mlm(X, W)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},"
              f"{N_PATHWAYS},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
