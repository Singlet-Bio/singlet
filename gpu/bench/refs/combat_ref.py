#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/combat_ref.py
#
# CYCLE-175 Phase E — scanpy.pp.combat CPU baseline for integrate/combat.
#
# GPU kernel (integrate::combat): 7 device passes, in-place Z-buffer reuse,
#   EB shrinkage at max_iter=2.  See include/singlet-gpu/integrate/combat.h.
# Reference: Johnson WE, Li C, Rabinovic A (2007) Biostatistics 8:118-127.
#
# CPU implementation: scanpy.pp.combat(adata, key='batch').
#   scanpy IS installed (confirmed CYCLE-157/158/162).
#   scanpy.pp.combat uses a vectorised NumPy + SciPy parametric EB implementation
#   (no per-gene Python loop in the inner EB step — closed-form method-of-moments
#   priors + two-pass update, all numpy arrays).  SOTA structure: vectorized numpy.
#   §J.7 prediction: class 2-3 (50–300×).  Could be class 1 if dense m×n ops
#   dominate equally on CPU and GPU.
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Only scanpy.pp.combat() is timed (adata construction is untimed — matches
#   GPU bench protocol where data upload and batch-label construction are untimed).
#   Note: scanpy.pp.combat works on dense adata.X (ndarray, not sparse).
#
# Scales:
#   10k: n_cells=10000, n_genes=5000, dense counts 0–20, 4 batches, max_iter default
#   30k: n_cells=30000, n_genes=5000, dense counts 0–20, 4 batches, max_iter default
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,n_batches,wall_ms
#
# Usage: python3 combat_ref.py

import sys
import time
import copy

import numpy as np
import anndata as ad
import scanpy as sc

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 5000, 4),
    ("30k", 30000, 5000, 4),
]


def make_dense_X(n_cells: int, n_genes: int, seed: int = 42) -> np.ndarray:
    """Synthetic dense expression matrix (n_cells × n_genes, float32).

    Integer counts 0–20 matching the C++ LCG synthetic data intent.
    Uses numpy for reproducibility (not bitwise identical to C++ LCG but
    same statistical character — dense, uniform counts in [0, 20]).
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 21, size=(n_cells, n_genes)).astype(np.float32)


def make_adata(X: np.ndarray, n_batches: int) -> ad.AnnData:
    """Build AnnData from dense X (n_cells × n_genes) with round-robin batch labels."""
    n_cells = X.shape[0]
    obs = {"batch": [str(c % n_batches) for c in range(n_cells)]}
    return ad.AnnData(X=X.copy(), obs=obs)


def time_combat(adata_template: ad.AnnData) -> float:
    """2 warmup + 5 timed iterations of scanpy.pp.combat.

    A fresh copy of adata is used each iteration (combat modifies X in-place).
    Returns median wall time in ms.
    """
    for _ in range(WARMUP_ITERS):
        adata_copy = adata_template.copy()
        try:
            sc.pp.combat(adata_copy, key="batch")
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        adata_copy = adata_template.copy()
        t0 = time.perf_counter()
        sc.pp.combat(adata_copy, key="batch")
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    return (wall_times[n_t // 2] if n_t % 2 == 1
            else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))


def main():
    print("scale,n_cells,n_genes,n_batches,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, n_batches in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_genes} genes × "
              f"{n_cells} cells, dense, n_batches={n_batches}...",
              file=sys.stderr, flush=True)

        X        = make_dense_X(n_cells, n_genes, seed=42)
        adata    = make_adata(X, n_batches)

        print(f"[ref] {scale_name}: X.shape={adata.X.shape}, "
              f"dtype={adata.X.dtype}. Timing scanpy.pp.combat...",
              file=sys.stderr, flush=True)

        try:
            wall_ms = time_combat(adata)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{n_batches},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
