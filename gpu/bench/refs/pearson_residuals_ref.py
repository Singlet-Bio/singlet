#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/pearson_residuals_ref.py
#
# CYCLE-157 Phase E — scanpy CPU baseline for preprocess/pearson_residuals.
#
# Generates synthetic CSC count matrices at the same shapes as the GPU bench
# (using scipy.sparse.random with the same density / seed), then times
# scanpy.experimental.pp.highly_variable_genes(flavor='pearson_residuals').
#
# Wall time only — no bit-exactness with GPU output required.
#
# Output (stdout): CSV header + rows matching the GPU bench format:
#   scale,n_cells,n_genes,density,wall_ms
#
# Usage: python3 pearson_residuals_ref.py

import sys
import time
import tracemalloc

import numpy as np
import scipy.sparse as sp


WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k",  10000, 5000, 0.05),
    ("30k",  30000, 5000, 0.05),
]


def make_synthetic_adata(n_cells: int, n_genes: int, density: float, seed: int = 42):
    """Build an AnnData with a synthetic sparse count matrix (CSC, int32 UMI counts)."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    # scipy.sparse.random generates values in [0, 1]; scale to UMI range 1–15.
    X = sp.random(n_cells, n_genes, density=density, format="csc",
                  dtype=np.float32, random_state=rng)
    # Snap to integer counts in [1, 15] for nonzeros.
    X.data = np.clip(np.round(X.data * 15.0).astype(np.float32), 1.0, 15.0)
    return ad.AnnData(X=X)


def time_pearson_residuals(adata, theta: float = 100.0):
    """2 warmup + 5 timed iters of scanpy pearson_residuals HVG. Returns median ms."""
    import scanpy as sc

    def _run(ad):
        import copy
        a2 = copy.copy(ad)
        try:
            sc.experimental.pp.highly_variable_genes(
                a2, flavor="pearson_residuals", theta=theta,
                n_top_genes=min(2000, ad.n_vars))
        except AttributeError:
            # Newer scanpy (>=1.10) moved it to sc.pp.
            sc.pp.highly_variable_genes(
                a2, flavor="pearson_residuals", theta=theta,
                n_top_genes=min(2000, ad.n_vars))
        return a2

    # Warmup.
    for _ in range(WARMUP_ITERS):
        try:
            _run(adata)
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _run(adata)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n = len(wall_times)
    median_ms = (wall_times[n // 2] if n % 2 == 1
                 else 0.5 * (wall_times[n // 2 - 1] + wall_times[n // 2]))
    return median_ms


def main():
    import importlib.util
    if importlib.util.find_spec("anndata") is None:
        print("ERROR: anndata not installed", file=sys.stderr)
        sys.exit(1)
    if importlib.util.find_spec("scanpy") is None:
        print("ERROR: scanpy not installed", file=sys.stderr)
        sys.exit(1)

    print("scale,n_cells,n_genes,density,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_genes} genes "
              f"density={density:.0%}...", file=sys.stderr, flush=True)

        adata = make_synthetic_adata(n_cells, n_genes, density, seed=42)

        print(f"[ref] Timing scanpy pearson_residuals {scale_name}...",
              file=sys.stderr, flush=True)

        try:
            wall_ms = time_pearson_residuals(adata, theta=100.0)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
