#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/diffmap_ref.py
#
# CYCLE-159 Phase E — scanpy CPU baseline for embed/diffmap.
#
# Generates synthetic dense PCA embeddings (n_cells x n_pcs) matching the GPU bench,
# then times scanpy.tl.diffmap() after a pre-built neighbor graph.
#
# Protocol:
#   - sc.pp.neighbors(adata, n_neighbors=10, use_rep='X') ONCE per scale (untimed).
#   - sc.tl.diffmap(adata, n_comps=5) — 2 warmup + 5 timed iterations.
#   - Wall time covers only diffmap (not the neighbors step).
#
# NOTE: scanpy.tl.diffmap operates on adata.uns['neighbors'] built from adata.X
# (the PCA embedding in this case). No gene-name parity issue — diffmap uses
# the precomputed neighbor graph, not gene lists.
#
# Output (stdout): CSV header + rows matching the GPU bench format:
#   scale,n_cells,n_pcs,k,n_components,wall_ms
#
# Usage: python3 diffmap_ref.py

import sys
import time
import copy

import numpy as np

WARMUP_ITERS  = 2
TIMED_ITERS   = 5
N_NEIGHBORS   = 10
N_COMPS       = 5

SCALES = [
    ("10k",  10000, 50),
    ("30k",  30000, 50),
]


def make_synthetic_embedding(n_cells: int, n_pcs: int, seed: int = 42) -> np.ndarray:
    """Build a synthetic dense PCA embedding (n_cells x n_pcs, float32).

    Uses np.random.default_rng(seed).standard_normal to match a standardized
    PCA coordinate distribution. Same seed and shape as the GPU bench driver.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_cells, n_pcs)).astype(np.float32)


def build_adata(embedding: np.ndarray):
    """Wrap a numpy embedding array in an AnnData object with X = embedding."""
    import anndata as ad
    adata = ad.AnnData(X=embedding)
    return adata


def run_neighbors(adata, n_neighbors: int = N_NEIGHBORS):
    """Build the kNN neighbor graph (untimed warmup — not the kernel under test).

    sc.pp.neighbors with use_rep='X' builds the graph directly from adata.X
    (our synthetic PCA embedding). n_neighbors=10 matches the GPU bench.
    """
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')


def time_diffmap(adata) -> float:
    """2 warmup + 5 timed iters of scanpy.tl.diffmap. Returns median wall_ms.

    A fresh copy of adata is made per iteration so scanpy cannot reuse stale
    state between runs. The neighbors graph is already embedded in adata.uns.
    """
    import scanpy as sc

    def _run(ad_in):
        a = copy.copy(ad_in)
        sc.tl.diffmap(a, n_comps=N_COMPS)

    # Warmup iterations (populate JIT caches, pre-allocate LAPACK buffers).
    for _ in range(WARMUP_ITERS):
        try:
            _run(adata)
        except Exception as exc:
            print(f"[ref] warmup error (ignored): {exc}", file=sys.stderr, flush=True)

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

    import scanpy as sc
    # Silence scanpy verbose output to keep stdout clean (CSV only).
    sc.settings.verbosity = 0

    print("scale,n_cells,n_pcs,k,n_components,wall_ms", flush=True)

    for scale_name, n_cells, n_pcs in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells x {n_pcs} PCs "
              f"(dense, float32)...", file=sys.stderr, flush=True)

        embedding = make_synthetic_embedding(n_cells, n_pcs, seed=42)
        adata = build_adata(embedding)

        print(f"[ref] Building kNN graph n_neighbors={N_NEIGHBORS} for {scale_name} "
              f"(untimed)...", file=sys.stderr, flush=True)
        try:
            run_neighbors(adata, n_neighbors=N_NEIGHBORS)
        except Exception as exc:
            print(f"[ref] ERROR building neighbors for {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            print(f"{scale_name},{n_cells},{n_pcs},{N_NEIGHBORS},{N_COMPS},-1.0",
                  flush=True)
            continue

        print(f"[ref] Timing scanpy.tl.diffmap n_comps={N_COMPS} for {scale_name} "
              f"({WARMUP_ITERS}w + {TIMED_ITERS}t)...", file=sys.stderr, flush=True)

        try:
            wall_ms = time_diffmap(adata)
        except Exception as exc:
            print(f"[ref] ERROR timing diffmap for {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_pcs},{N_NEIGHBORS},{N_COMPS},{wall_ms:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
