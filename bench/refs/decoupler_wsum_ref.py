#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/decoupler_wsum_ref.py
#
# CYCLE-163 Phase E — manual scipy/numpy CPU baseline for enrich/decoupler_wsum.
#
# GPU kernel: cuSPARSE SpMM (X^T · W) + per-pathway normalisation.
# Reference:  Badia-i-Mompel et al. (2022) decoupleR, Bioinformatics Advances 2:vbac016.
#
# Two decoupleR flavors benchmarked per scale (DUAL-baseline pattern):
#
#   method='wsum'  — WSUM: (X.T @ W) / max(|W|.sum(axis=0), 1e-8)
#                    normalises by L1 norm of each pathway's weights
#   method='wmean' — WMEAN: (X.T @ W) / max((W != 0).sum(axis=0), 1)
#                    normalises by count of nonzero weights per pathway
#
# NOTE: `decoupler` Python package is NOT installed on the cluster.
#       This is a MANUAL numpy/scipy implementation that is mathematically
#       equivalent to decoupleR's WSUM and WMEAN methods.
#
# Input generation:
#   X — scipy.sparse.random(n_cells, n_genes, density=0.05, format='csc',
#                           dtype=float32, random_state=42)
#       CSC layout (matches GPU bench input).
#   W — np.random.default_rng(42).standard_normal(n_genes, n_pathways).astype(float32)
#       Dense col-major weight matrix.
#
# Protocol: 2 warmup + 5 timed iterations per method per scale; median wall time.
#
# NOTE (CYCLE-158 lesson): If building AnnData (not needed here), set var_names
# via `adata.var_names = ...` (direct assignment), NOT `var={"col": ...}` constructor.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,density,n_pathways,method,wall_ms
#
# Usage: python3 decoupler_wsum_ref.py

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

METHODS = ["wsum", "wmean"]


def make_synthetic_inputs(n_cells: int, n_genes: int, density: float,
                           n_pathways: int, seed: int = 42):
    """Build synthetic sparse X (CSC, n_cells × n_genes) and dense W (n_genes × n_pathways).

    X: scipy.sparse.random with seed=42 for parity with GPU bench.
    W: standard normal via np.random.default_rng(42).
    """
    rng = np.random.default_rng(seed)
    # scipy.sparse.random random_state accepts int or np.random.Generator (scipy >=1.9).
    # Use seed directly for broadest compatibility.
    X = sp.random(n_cells, n_genes, density=density, format="csc",
                  dtype=np.float32, random_state=seed)

    W = rng.standard_normal((n_genes, n_pathways)).astype(np.float32)
    return X, W


def time_method(X, W, method: str):
    """2 warmup + 5 timed iters of WSUM or WMEAN.

    WSUM:  scores = X.T @ W, then divide each column p by max(|W[:,p]|.sum(), 1e-8).
    WMEAN: scores = X.T @ W, then divide each column p by max((W[:,p]!=0).sum(), 1).

    Returns median wall_ms over TIMED_ITERS.
    """
    # Pre-compute normaliser once (same value every iter — not part of the timed path
    # since the GPU bench also does this per-call; we measure the full call cost).
    if method == "wsum":
        norm = np.maximum(np.abs(W).sum(axis=0), 1e-8)  # shape (n_pathways,)
    elif method == "wmean":
        norm = np.maximum((W != 0).sum(axis=0), 1).astype(np.float32)  # shape (n_pathways,)
    else:
        raise ValueError(f"Unknown method: {method}")

    def _run():
        # Step 1: SpMM — X @ W → (n_cells × n_pathways)
        # X is CSC (n_cells × n_genes); W is (n_genes × n_pathways).
        scores = X @ W
        # Step 2: divide each column by its normaliser
        scores = scores / norm  # broadcasts over rows
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
    print("scale,n_cells,n_genes,density,n_pathways,method,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_genes} genes "
              f"density={density:.0%} n_pathways={N_PATHWAYS}...",
              file=sys.stderr, flush=True)

        X, W = make_synthetic_inputs(n_cells, n_genes, density, N_PATHWAYS, seed=42)

        for method in METHODS:
            print(f"[ref] Timing {scale_name} method='{method}'...",
                  file=sys.stderr, flush=True)
            try:
                wall_ms = time_method(X, W, method)
            except Exception as exc:
                print(f"[ref] ERROR {scale_name} method={method}: {exc}",
                      file=sys.stderr, flush=True)
                wall_ms = -1.0

            print(f"{scale_name},{n_cells},{n_genes},{density:.2f},"
                  f"{N_PATHWAYS},{method},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
