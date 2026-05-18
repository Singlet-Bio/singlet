#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/score_genes_ref.py
#
# CYCLE-158 Phase E — scanpy CPU baseline for enrich/score_genes.
#
# Generates synthetic CSC expression matrices at the same shapes as the GPU bench
# (using scipy.sparse.random with the same density / seed=42), then times
# scanpy.tl.score_genes() in a loop over N_SETS=5 gene sets of 50 genes each.
#
# Wall time covers the full loop over all gene sets (matching the GPU batch call).
#
# Output (stdout): CSV header + rows matching the GPU bench format:
#   scale,n_cells,n_genes,density,n_sets,wall_ms
#
# Usage: python3 score_genes_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp


WARMUP_ITERS = 2
TIMED_ITERS  = 5

N_SETS   = 5
SET_SIZE = 50

SCALES = [
    ("10k",  10000, 5000, 0.05),
    ("30k",  30000, 5000, 0.05),
]


def make_synthetic_adata(n_cells: int, n_genes: int, density: float, seed: int = 42):
    """Build an AnnData with a synthetic sparse expression matrix (CSC, float32)."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, n_genes, density=density, format="csc",
                  dtype=np.float32, random_state=rng)
    # Snap to integer counts in [1, 15] for nonzeros (UMI-like).
    X.data = np.clip(np.round(X.data * 15.0).astype(np.float32), 1.0, 15.0)
    var_names = [f"gene_{i}" for i in range(n_genes)]
    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    return adata


def make_gene_lists(n_genes: int, n_sets: int, set_size: int, seed: int = 99):
    """Build n_sets gene name lists of size set_size, uniform indices."""
    rng = np.random.default_rng(seed)
    gene_lists = []
    for _ in range(n_sets):
        indices = rng.integers(0, n_genes, size=set_size)
        gene_lists.append([f"gene_{i}" for i in indices])
    return gene_lists


def time_score_genes(adata, gene_lists):
    """2 warmup + 5 timed iters of scanpy.tl.score_genes loop. Returns median ms."""
    import scanpy as sc

    def _run(ad_in):
        import copy
        a = copy.copy(ad_in)
        for s_idx, gene_list in enumerate(gene_lists):
            sc.tl.score_genes(a, gene_list=gene_list,
                               score_name=f"score_set_{s_idx}",
                               ctrl_size=50, n_bins=25, random_state=42)

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

    print("scale,n_cells,n_genes,density,n_sets,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells x {n_genes} genes "
              f"density={density:.0%}...", file=sys.stderr, flush=True)

        adata = make_synthetic_adata(n_cells, n_genes, density, seed=42)
        gene_lists = make_gene_lists(n_genes, N_SETS, SET_SIZE, seed=99)

        print(f"[ref] Timing scanpy.tl.score_genes loop ({N_SETS} sets x {SET_SIZE} genes) "
              f"{scale_name}...", file=sys.stderr, flush=True)

        try:
            wall_ms = time_score_genes(adata, gene_lists)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},{N_SETS},{wall_ms:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
