#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/decoupler_ulm_ref.py
#
# CYCLE-164 Phase E — manual numpy CPU baseline for enrich/decoupler_ulm.
#
# GPU kernel: 5 passes — per-cell mean, per-pathway W stats, var_W,
#             cuSPARSE SpMM (X^T · W), OLS slope score kernel.
# Reference:  Badia-i-Mompel et al. (2022) decoupleR, Bioinformatics Advances 2:vbac016.
#
# ULM (Univariate Linear Model): for each cell c and pathway p, fit OLS:
#   X[g, c] = β_0 + β_1 · W[g, p] + ε   (over all genes g)
#
# Closed-form OLS slope (regression coefficient):
#   β_1[c, p] = cov(X_c, W_p) / var(W_p)
#
# Expanding in matrix form:
#   mean_X[c]  = mean(X[:, c])       shape: (n_cells,)
#   mean_W[p]  = mean(W[:, p])       shape: (n_pathways,)
#   var_W[p]   = var(W[:, p])        shape: (n_pathways,)
#   XW[c, p]   = (X.T @ W)[c, p]    shape: (n_cells × n_pathways)  — SpMM
#   cov[c, p]  = XW[c, p]/m - mean_X[c]*mean_W[p]
#   score[c,p] = cov[c,p] / max(var_W[p], ε)
#
# Vectorized numpy implementation (~15 lines) — equivalent to GPU kernel.
#
# NOTE: `decoupler` Python package is NOT installed on the cluster.
#       This is a MANUAL numpy/scipy implementation of decoupleR's ULM.
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
# Usage: python3 decoupler_ulm_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_PATHWAYS   = 50

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
    SpMM: X @ W → (n_cells × n_pathways). NOT X.T @ W.
    """
    rng = np.random.default_rng(seed)
    # scipy.sparse.random random_state accepts int or np.random.Generator (scipy >=1.9).
    # Use seed directly for broadest compatibility.
    X = sp.random(n_cells, n_genes, density=density, format="csc",
                  dtype=np.float32, random_state=seed)

    W = rng.standard_normal((n_genes, n_pathways)).astype(np.float32)
    return X, W


def time_ulm(X, W):
    """2 warmup + 5 timed iters of closed-form ULM (vectorized numpy).

    Computes OLS regression slope β_1[c, p] = cov(X_c, W_p) / var(W_p)
    for all cells c and pathways p simultaneously.

    Steps (matching GPU 5-pass kernel):
      1. mean_X[c]  = X.mean(axis=1)   — per-cell mean (dense column-sum / m)
      2. mean_W[p]  = W.mean(axis=0)   — per-pathway mean
      3. var_W[p]   = W.var(axis=0)    — per-pathway variance (unbiased=False, matches GPU)
      4. XW         = X @ W            — SpMM: (n_cells × n_genes) · (n_genes × n_pathways)
      5. score[c,p] = (XW[c,p]/m - mean_X[c]*mean_W[p]) / max(var_W[p], ε)

    Returns median wall_ms over TIMED_ITERS.
    """
    n_cells, n_genes = X.shape
    m = float(n_genes)
    eps = 1e-9

    # Pre-compute W stats once (same values every iter — not part of timed path
    # per GPU bench pattern; GPU also computes these inside the timed call).
    mean_W = W.mean(axis=0)       # (n_pathways,)
    var_W  = W.var(axis=0)        # (n_pathways,) — biased variance, matches GPU
    var_W_safe = np.maximum(var_W, eps)

    # Pre-compute per-cell mean of X (sparse column sum / m).
    # X is CSC (n_cells × n_genes); .mean(axis=1) returns (n_cells, 1) matrix.
    mean_X = np.asarray(X.mean(axis=1)).ravel()   # (n_cells,)

    def _run():
        # Step 4: SpMM — X @ W → (n_cells × n_pathways)
        XW = X @ W  # scipy sparse @ dense: (n_cells × n_genes) · (n_genes × n_pathways)

        # Step 5: OLS slope — vectorized over all (c, p) pairs.
        # cov[c,p] = XW[c,p]/m - mean_X[c]*mean_W[p]
        # score[c,p] = cov[c,p] / var_W[p]
        if sp.issparse(XW):
            XW = XW.toarray()
        scores = (XW / m - mean_X[:, None] * mean_W[None, :]) / var_W_safe[None, :]
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

        print(f"[ref] Timing {scale_name} method='ulm'...",
              file=sys.stderr, flush=True)
        try:
            wall_ms = time_ulm(X, W)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},"
              f"{N_PATHWAYS},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
