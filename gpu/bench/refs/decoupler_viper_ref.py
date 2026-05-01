#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/decoupler_viper_ref.py
#
# CYCLE-167 Phase E — manual numpy/scipy CPU baseline for enrich/decoupler_viper.
#
# GPU kernel: 3 passes — viper_fill_dense + CUB segmented sort (rank transform),
#             viper_assign_t1_kernel (normcdfinvf qnorm per rank),
#             cuBLAS Sgemm (T1^T · W → NES, n × n_reg),
#             viper_l1_norm_kernel + viper_scale_kernel (L1 normalisation).
# Reference:  Alvarez MJ et al. (2016) Nat Genet 48:838-847.
#             VIPER/aREA v0 simplification: NES = (T1^T · W) / L1(W).
#
# VIPER algorithm (CPU equivalent):
#   For each cell c:
#     1. Extract expression column X[:, c] (m_genes values).
#     2. Rank transform: r[g] = rank of gene g among all m genes (ascending).
#     3. qnorm transform: T1[g, c] = scipy.stats.norm.ppf((rank + 1) / (m + 1))
#        Equivalent to normcdfinvf(p) on GPU.
#   4. NES = T1^T · W   (n_cells × n_reg)
#   5. Divide each column r by L1(W[:, r]) = sum(|W[:, r]|).
#
# CPU implementation:
#   1. Build X (m_genes × n_cells) dense from synthetic CSC.
#   2. Per-cell rankdata via scipy.stats.rankdata (vectorised axis=0):
#        ranks = scipy.stats.rankdata(X_dense, axis=0, method='ordinal')
#        shape (m × n), dtype float64 → cast to float32.
#   3. Vectorised qnorm: T1 = scipy.stats.norm.ppf((ranks) / (m + 1))
#      where ranks are 1-based (rankdata default is 1-based).
#   4. NES_raw = T1.T @ W   (n × n_reg)  — X @ W convention.
#   5. L1 norms: norm[r] = np.sum(np.abs(W[:, r]))
#   6. NES = NES_raw / norm[np.newaxis, :]
#
# NOTE: `decoupler` Python package is NOT installed on the cluster.
#       This is a MANUAL numpy/scipy implementation of VIPER/aREA.
#
# NOTE (CYCLE-163/164/165/166 INFRA pattern): if scipy not available in SLURM env
# (g008 ModuleNotFoundError), orchestrator runs the ref locally (scipy available)
# and pastes CSV manually per established workaround.
#
# Input generation (mirrors GPU bench):
#   X — scipy.sparse.random(n_genes, n_cells, density=0.05, format='csc',
#                           dtype=float32, random_state=42)  ← CSC rows=genes
#   W — dense (n_genes × n_reg) float32, random_state=99, values in [-1, 1].
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,density,n_reg,wall_ms
#
# Usage: python3 decoupler_viper_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.stats

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_REG        = 50

SCALES = [
    ("10k",  10000, 5000, 0.05),
    ("30k",  30000, 5000, 0.05),
]


def make_synthetic_inputs(n_genes: int, n_cells: int, density: float,
                           n_reg: int, seed: int = 42):
    """Build synthetic sparse X (CSC, rows=genes, cols=cells) and dense W.

    X: scipy.sparse.random, seed=42 for parity with GPU bench.
       n_genes × n_cells CSC (matches GPU convention: rows=genes, cols=cells).
    W: dense (n_genes × n_reg) float32, values in [-1, 1] (signed weights).
       Seeded separately (seed=99) to avoid correlation with X.
    """
    X = sp.random(n_genes, n_cells, density=density, format="csc",
                  dtype=np.float32, random_state=seed)
    rng = np.random.default_rng(seed=99)
    W = rng.uniform(-1.0, 1.0, size=(n_genes, n_reg)).astype(np.float32)
    return X, W


def time_viper(X, W):
    """2 warmup + 5 timed iters of manual numpy/scipy VIPER.

    Steps matching GPU 3-pass kernel:
      1. Dense X (m_genes × n_cells): X_dense = X.toarray()
      2. Per-cell rank transform (axis=0): ranks = rankdata(X_dense, axis=0)
         scipy.stats.rankdata with method='ordinal', axis=0 → shape (m × n),
         1-based ranks. One call for all cells (vectorised).
      3. qnorm: T1 = norm.ppf(ranks / (m + 1))   shape (m × n) float32.
         scipy.stats.norm.ppf is vectorised on the whole array → one call.
      4. NES_raw = T1.T @ W   (n × n_reg) — X @ W convention.
      5. L1 norms: norm_vec = np.abs(W).sum(axis=0)   (n_reg,)
      6. NES = NES_raw / norm_vec[np.newaxis, :]

    Returns median wall_ms over TIMED_ITERS.
    """
    m, n = X.shape   # m_genes × n_cells
    n_reg = W.shape[1]

    def _run():
        # Step 1: dense materialisation (m × n) float32.
        X_dense = X.toarray()   # (m_genes × n_cells) float32

        # Step 2: per-cell rank transform, vectorised over all cells at once.
        # scipy.stats.rankdata with axis=0 → ranks along gene axis per cell.
        # method='ordinal' matches GPU CUB ascending sort tie-breaking.
        ranks = scipy.stats.rankdata(X_dense, axis=0, method='ordinal')
        # ranks: (m × n) float64 (rankdata always returns float64), 1-based.

        # Step 3: qnorm  T1[g, c] = norm.ppf(rank[g,c] / (m + 1))
        # Vectorised single call — norm.ppf accepts arrays.
        p = ranks / (m + 1)
        T1 = scipy.stats.norm.ppf(p).astype(np.float32)   # (m × n) float32

        # Step 4: NES_raw = T1^T @ W   (n × n_reg).
        # T1 is (m × n) → T1.T is (n × m) → (n × m) @ (m × n_reg) = (n × n_reg).
        NES_raw = T1.T @ W   # X @ W convention (T1.T is cells × genes)

        # Step 5: per-regulon L1 norm of W.
        norm_vec = np.abs(W).sum(axis=0)   # (n_reg,) float32
        # Guard division by zero (epsilon = 1e-9 matching GPU ViperConfig).
        norm_vec = np.maximum(norm_vec, 1e-9)

        # Step 6: normalise.
        NES = NES_raw / norm_vec[np.newaxis, :]   # (n × n_reg)
        return NES

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
    n_t = len(wall_times)
    return (wall_times[n_t // 2] if n_t % 2 == 1
            else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))


def main():
    print("scale,n_cells,n_genes,density,n_reg,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_genes} genes "
              f"density={density:.0%} n_reg={N_REG}...",
              file=sys.stderr, flush=True)

        X, W = make_synthetic_inputs(n_genes, n_cells, density, N_REG, seed=42)

        print(f"[ref] Timing {scale_name} method='viper' "
              f"(scipy.stats.rankdata + norm.ppf vectorised)...",
              file=sys.stderr, flush=True)
        try:
            wall_ms = time_viper(X, W)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},"
              f"{N_REG},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
