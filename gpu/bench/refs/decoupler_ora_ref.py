#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/decoupler_ora_ref.py
#
# CYCLE-166 Phase E — manual numpy/scipy CPU baseline for enrich/decoupler_ora.
#
# GPU kernel: 4 passes — ora_topk_smem_kernel (64-bucket shared-mem top-K),
#             ora_build_T_kernel (binary top-K mask T, m × n col-major),
#             cuBLAS Sgemm (T^T · M → hits, n × n_sets col-major),
#             ora_hypergeo_kernel (log-sum-exp lgamma, -log10(p) per cell×set).
# Reference:  Badia-i-Mompel et al. (2022) decoupleR, Bioinformatics Advances 2:vbac016.
#
# ORA (Over-Representation Analysis): for each cell, identify the top-K expressed
# genes (top 5% by expression value) and test each gene set for over-representation
# using a one-tailed hypergeometric test.
#
# Setup per cell c:
#   Universe N   = n_genes (m)
#   Top-K    n   = ceil(m * 0.05)   [decoupleR default]
#   For each set s:
#     K   = |set s|   (population successes)
#     k   = |T_c ∩ set s|   (sample successes / hits)
#     p   = hypergeom.sf(k-1, N, K, n)
#          = P(X >= k)  where X ~ Hypergeometric(N, K, n)
#     score[c, s] = -log10(p)  (0 when p=1, i.e. no overlap)
#
# CPU implementation using numpy.argsort + scipy.stats.hypergeom.sf:
#   1. Build X (n_cells × n_genes) dense from synthetic CSC.
#      X convention: X @ W (rows=cells, cols=genes) — NOT X.T @ W.
#      GPU CSC has rows=genes, cols=cells. X here is the transpose for dense ops.
#   2. top_k = ceil(n_genes * 0.05)
#   3. For each cell: argsort -X[c, :] (descending), take first top_k indices → T_c.
#      Vectorised over all cells via numpy.argpartition + mask.
#   4. Build binary top-K mask T (n_cells × n_genes), T[c, g] = 1 if g in top_k of cell c.
#   5. Build binary gene-set matrix M (n_genes × n_sets), M[g, s] = 1 if g ∈ set s.
#   6. Hits = T @ M → (n_cells × n_sets) integer overlap counts.
#      Matches GPU Sgemm: hits = T^T · M_col where GPU T is m×n → our hits = (X@W) equiv.
#   7. For each (c, s): p = hypergeom.sf(k-1, N, K, n); score = -log10(p).
#      Vectorized using scipy.stats.hypergeom.sf with broadcast arrays.
#
# NOTE: `decoupler` Python package is NOT installed on the cluster.
#       This is a MANUAL numpy/scipy implementation of decoupleR's ORA.
#
# Input generation (mirrors GPU bench):
#   X — scipy.sparse.random(n_genes, n_cells, density=0.05, format='csc',
#                           dtype=float32, random_state=42)  ← CSC rows=genes
#       Converted to dense for argsort; X.T gives (n_cells × n_genes).
#   Gene sets — deterministic stride-7 walk per pathway (mirrors C++ make_gene_sets):
#       set_size = int(n_genes * 0.10), offset = (42 * (s+1) * LCG) % n_genes,
#       members  = sorted(unique((offset + i*7) % n_genes for i in range(set_size))).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,density,n_pathways,wall_ms
#
# Usage: python3 decoupler_ora_ref.py

import math
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.stats

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_PATHWAYS   = 50

SCALES = [
    ("10k",  10000, 5000, 0.05),
    ("30k",  30000, 5000, 0.05),
]


def make_gene_sets(n_genes: int, n_pathways: int, seed: int = 42):
    """Build deterministic synthetic gene sets mirroring C++ make_gene_sets.

    Each set has ~10% membership (stride-7 walk with per-pathway offset).
    Mirrors the C++ LCG: offset = (seed * (s+1) * 6364136223846793005) % n_genes
    (mod 2^64 then mod n_genes).
    """
    set_size = int(n_genes * 0.10)
    gene_sets = []
    for s in range(n_pathways):
        # Match C++ LCG offset exactly (uint64 overflow via mask).
        offset = int((seed * (s + 1) * 6364136223846793005) & 0xFFFFFFFFFFFFFFFF) % n_genes
        members = sorted(set((offset + i * 7) % n_genes for i in range(set_size)))
        if not members:
            members = [s % n_genes]
        gene_sets.append(members)
    return gene_sets


def make_synthetic_inputs(n_genes: int, n_cells: int, density: float,
                           n_pathways: int, seed: int = 42):
    """Build synthetic sparse X (CSC, rows=genes, cols=cells) and gene sets.

    X: scipy.sparse.random with seed=42 for parity with GPU bench.
    X is n_genes × n_cells CSC (matches GPU convention: rows=genes, cols=cells).
    Dense view for CPU ORA: X.T gives (n_cells × n_genes).

    NOTE: X @ W convention (cells × genes) for the "X @ M" hits computation
    requires X.T since X here is (n_genes × n_cells).
    Equivalently: hits = (X.T) @ M   or   hits = (X.T.toarray()) @ M_dense.
    """
    X = sp.random(n_genes, n_cells, density=density, format="csc",
                  dtype=np.float32, random_state=seed)
    gene_sets = make_gene_sets(n_genes, n_pathways, seed=seed)
    return X, gene_sets


def time_ora(X, gene_sets):
    """2 warmup + 5 timed iters of manual numpy/scipy ORA.

    Steps (matching GPU 4-pass kernel):
      1. Dense X^T (n_cells × n_genes): X_dense = X.T.toarray()
      2. top_k = ceil(n_genes * 0.05)
      3. Vectorised top-K mask: T (n_cells × n_genes) binary via argpartition.
      4. Build M (n_genes × n_sets) binary membership matrix.
      5. hits = T @ M → (n_cells × n_sets) float32 counts.   [X @ W equivalent]
      6. Vectorised hypergeom.sf: p = hypergeom.sf(k-1, N, K, n) per (c, s).
         score = -log10(p), clamped to [0, 300].

    Returns median wall_ms over TIMED_ITERS.
    """
    n_genes, n_cells = X.shape
    n_sets = len(gene_sets)
    top_k  = math.ceil(n_genes * 0.05)
    N      = n_genes

    # Build M (n_genes × n_sets) once outside the timed loop (GPU also builds
    # M host-side once; but we include it in timing for a conservative baseline).
    # Actually: GPU's Pass 2 H2D of M is inside the timed call; include here too.

    def _run():
        # Pass 1 — convert CSC to dense (n_cells × n_genes) = X.T.
        X_dense = X.T.toarray()   # (n_cells × n_genes) float32

        # Pass 2 — top-K mask T (n_cells × n_genes) binary.
        # argpartition is O(n_genes) per cell; argsort is O(n_genes log n_genes).
        # Use argpartition for speed; matches GPU bucket-threshold approach.
        part_idx = np.argpartition(-X_dense, top_k, axis=1)[:, :top_k]  # (n_cells × top_k)
        T = np.zeros((n_cells, n_genes), dtype=np.float32)
        row_idx = np.repeat(np.arange(n_cells), top_k)
        col_idx = part_idx.ravel()
        T[row_idx, col_idx] = 1.0

        # Pass 2b — M (n_genes × n_sets) binary.
        M = np.zeros((n_genes, n_sets), dtype=np.float32)
        for s_idx, s_genes in enumerate(gene_sets):
            M[s_genes, s_idx] = 1.0
        set_sizes = np.array([len(gs) for gs in gene_sets], dtype=np.int32)  # K per set

        # Pass 3 — hits = T @ M → (n_cells × n_sets) float32.
        hits = T @ M   # X @ W convention (T is n_cells × n_genes; M is n_genes × n_sets)

        # Pass 4 — vectorised hypergeometric -log10(p).
        # hits[c, s] = k;  K = set_sizes[s];  n = top_k;  N = n_genes.
        k_mat = hits.astype(np.int32)                           # (n_cells × n_sets)
        K_vec = set_sizes[np.newaxis, :]                        # (1 × n_sets) broadcast
        # p = P(X >= k) = hypergeom.sf(k-1, N, K, n)
        # scipy.stats.hypergeom.sf is vectorised; shape broadcasts.
        k_minus1 = np.maximum(k_mat - 1, 0)   # sf(k-1, ...) for k=0 → sf(-1) = 1 → score=0
        p_vals = scipy.stats.hypergeom.sf(k_minus1, N, K_vec, top_k)
        # score = -log10(p), cap at 300.
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(k_mat > 0,
                              np.minimum(-np.log10(np.clip(p_vals, 1e-300, 1.0)), 300.0),
                              0.0)
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

        X, gene_sets = make_synthetic_inputs(n_genes, n_cells, density,
                                              N_PATHWAYS, seed=42)

        print(f"[ref] Timing {scale_name} method='ora' (top_k=5%, scipy hypergeom.sf)...",
              file=sys.stderr, flush=True)
        try:
            wall_ms = time_ora(X, gene_sets)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},"
              f"{N_PATHWAYS},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
