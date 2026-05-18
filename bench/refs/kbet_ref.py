#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/kbet_ref.py
#
# CYCLE-171 Phase E — manual numpy CPU baseline for integrate/kbet.
#
# GPU kernel: one block per cell; shared-mem int local_count[n_batches];
#   thread 0 serially scans k neighbors, builds batch histogram, computes
#   chi2 = Σ_b (obs-exp)²/max(exp,1), p-value via Wilson-Hilferty cube-root
#   transform (df=n_batches-1; df==1 uses erfc exact).  kbet_reject_kernel
#   marks p<0.05 per cell.  cub::DeviceReduce::Sum for mean_chi2/reject_rate.
# Reference: Büttner M, Miao Z, Wolf FA, et al. (2019) Nat Methods 16:43-49.
#
# kBET formula (per-cell, full scan, no random subsetting — v0 per header):
#   For each cell c with k nearest neighbors {n_1,...,n_k} and global batch
#   counts global_cnt[b] (for b in 0..n_batches-1):
#     1. n_obs[b] = count of c's k neighbors with batch label b.
#     2. n_exp[b] = k * (global_cnt[b] / n_cells).
#     3. chi2[c]  = Σ_b (n_obs[b] - n_exp[b])^2 / max(n_exp[b], 1.0)
#                   (term skipped if n_exp[b] <= 0).
#     4. df = n_batches - 1.
#     5. p-value via Wilson-Hilferty cube-root transform:
#          z = ((chi2/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))
#          p = 1 - Φ(z) = norm.sf(z)   [scipy] or computed from erfc
#        df==1 exact: p = erfc(sqrt(chi2/2)) / 1   (chi2(1) ~ Z^2).
#        chi2==0 → p=1; guard p to [0,1].
#     6. reject[c] = 1 if p < 0.05.
#   kBET score = mean(reject) = reject_rate (lower = better mixing).
#   mean_chi2  = mean(chi2) over all cells.
#
# CPU implementation (vectorised numpy, per-cell loop-free):
#   Build embedding: random fp32 (n × n_pcs), seed=42 — same LCG-equivalent.
#   Build kNN: sklearn.neighbors.NearestNeighbors (brute, Euclidean, k=k).
#     Drop column 0 (self-loop) to get k true neighbors.
#   Batch labels: round-robin batch[c] = c % n_batches.
#   chi2 vectorised:
#     For each batch b:
#       n_obs[:, b] = (batch[neighbors] == b).sum(axis=1)   # (n_cells,)
#     n_exp[b] = k * (global_cnt[b] / n_cells)              # scalar per b
#     diff = n_obs - n_exp[np.newaxis, :]                   # (n_cells, n_batches)
#     denom = np.maximum(n_exp, 1.0)                        # (n_batches,)
#     chi2 = (diff**2 / denom).sum(axis=1)                  # (n_cells,)
#   p-value vectorised (Wilson-Hilferty, df = n_batches - 1):
#     if df == 1: p = scipy.special.erfc(np.sqrt(chi2 / 2))
#     else:       z = (chi2/df)^(1/3) - c1) / sd
#                 p = 0.5 * erfc(z / sqrt(2))   [i.e. norm.sf(z)]
#     p clipped to [0, 1]; chi2==0 → p=1.
#   reject = (p < 0.05).astype(float)
#   reject_rate = reject.mean(); mean_chi2 = chi2.mean()
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   kNN build is NOT timed (matches GPU bench: kNN is untimed warmup).
#   Only the kBET computation is timed.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_pcs,k,n_batches,wall_ms,mean_chi2,reject_rate
#
# Usage: python3 kbet_ref.py

import math
import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_BATCHES    = 4

SCALES = [
    ("10k", 10000, 50, 10),
    ("30k", 30000, 50, 15),
]


def make_synthetic_embedding(n_cells: int, n_pcs: int, seed: int = 42) -> np.ndarray:
    """Row-major float32 embedding (n_cells × n_pcs), seed=42.

    Uses numpy default_rng; different LCG than C++ bench but both produce
    random embeddings in [-1, 1] — acceptable for a CPU timing baseline.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1.0, 1.0, size=(n_cells, n_pcs)).astype(np.float32)


def build_knn(emb: np.ndarray, k: int):
    """Return neighbor_indices (n_cells × k), excluding self.

    sklearn NearestNeighbors with n_neighbors=k+1 (includes self at col 0).
    Brute-force Euclidean — matches GPU Exact backend (L2).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                          metric="euclidean", n_jobs=-1)
    nn.fit(emb)
    _, indices = nn.kneighbors(emb)
    return indices[:, 1:k + 1].astype(np.int32)


def compute_kbet_vectorized(
        neighbors: np.ndarray,
        batch: np.ndarray,
        n_batches: int) -> tuple:
    """Vectorised kBET over all cells.

    neighbors  : (n_cells, k) int32  — k nearest-neighbor indices per cell.
    batch      : (n_cells,)   int32  — batch label in [0, n_batches).
    n_batches  : int                 — number of distinct batches.

    Returns (mean_chi2: float, reject_rate: float, chi2_arr: ndarray,
             pvalue_arr: ndarray).
    """
    n_cells, k = neighbors.shape

    # Gather batch labels of each cell's k neighbors: (n_cells, k) int.
    batch_nbrs = batch[neighbors]   # advanced indexing

    # Global batch counts.
    global_cnt = np.bincount(batch, minlength=n_batches).astype(np.float32)

    # n_exp[b] = k * global_cnt[b] / n_cells.
    n_exp = (float(k) * global_cnt / float(n_cells)).astype(np.float32)

    # n_obs[c, b] = count of cell c's neighbors with batch b.
    # Build via one-hot indicator: (n_cells, k, n_batches) → sum axis=1.
    labels_b = np.arange(n_batches, dtype=np.int32)
    # broadcast: (n_cells, k, 1) == (1, 1, n_batches)
    one_hot = (batch_nbrs[:, :, np.newaxis] ==
               labels_b[np.newaxis, np.newaxis, :])     # (n_cells, k, n_batches)
    n_obs = one_hot.sum(axis=1).astype(np.float32)      # (n_cells, n_batches)

    # chi2[c] = Σ_b (n_obs[c,b] - n_exp[b])^2 / max(n_exp[b], 1.0)
    # Skip terms where n_exp[b] <= 0 (none here with round-robin labels).
    denom = np.maximum(n_exp, np.float32(1.0))          # (n_batches,)
    diff  = n_obs - n_exp[np.newaxis, :]                # (n_cells, n_batches)
    chi2  = (diff ** 2 / denom).sum(axis=1)             # (n_cells,)

    # p-value via Wilson-Hilferty cube-root transform.
    df = n_batches - 1
    pvalue = np.ones(n_cells, dtype=np.float32)

    nonzero = chi2 > 0.0

    if df == 1:
        # chi2(1) = Z^2 → exact: p = erfc(sqrt(chi2/2)).
        pvalue[nonzero] = (
            np.vectorize(math.erfc)(np.sqrt(chi2[nonzero] * 0.5))
        ).astype(np.float32)
    else:
        dff  = float(df)
        c1   = 1.0 - (2.0 / (9.0 * dff))
        sd   = math.sqrt(2.0 / (9.0 * dff))
        ratio = chi2[nonzero].astype(np.float64) / dff
        z     = (np.cbrt(ratio) - c1) / sd
        # norm.sf(z) = 0.5 * erfc(z / sqrt(2))
        pv_nz = (0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))).astype(np.float32)
        pvalue[nonzero] = pv_nz

    pvalue = np.clip(pvalue, 0.0, 1.0)

    reject      = (pvalue < 0.05).astype(np.float32)
    mean_chi2   = float(chi2.mean())
    reject_rate = float(reject.mean())

    return mean_chi2, reject_rate, chi2, pvalue


def time_kbet(neighbors: np.ndarray,
              batch: np.ndarray,
              n_batches: int) -> tuple:
    """2 warmup + 5 timed iterations of compute_kbet_vectorized.

    Returns (median_wall_ms, mean_chi2, reject_rate) from last timed run.
    """
    last_mean_chi2  = 0.0
    last_reject_rate = 0.0

    def _run():
        return compute_kbet_vectorized(neighbors, batch, n_batches)

    for _ in range(WARMUP_ITERS):
        try:
            _run()
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        mc, rr, _, _ = _run()
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)
        last_mean_chi2   = mc
        last_reject_rate = rr

    wall_times.sort()
    n_t = len(wall_times)
    median_ms = (wall_times[n_t // 2] if n_t % 2 == 1
                 else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))

    return median_ms, last_mean_chi2, last_reject_rate


def main():
    print("scale,n_cells,n_pcs,k,n_batches,wall_ms,mean_chi2,reject_rate",
          flush=True)

    for scale_name, n_cells, n_pcs, k in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_pcs} PCs "
              f"k={k} n_batches={N_BATCHES}...",
              file=sys.stderr, flush=True)

        emb = make_synthetic_embedding(n_cells, n_pcs, seed=42)

        print(f"[ref] Building kNN (sklearn brute, k={k})...",
              file=sys.stderr, flush=True)
        neighbors = build_knn(emb, k)   # (n_cells, k) — untimed

        # Round-robin batch labels: batch[c] = c % N_BATCHES.
        batch = (np.arange(n_cells, dtype=np.int32) % N_BATCHES)

        print(f"[ref] Timing {scale_name} kbet vectorized "
              f"(n_batches={N_BATCHES} k={k})...",
              file=sys.stderr, flush=True)

        try:
            wall_ms, mean_chi2, reject_rate = time_kbet(neighbors, batch, N_BATCHES)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0
            mean_chi2 = float("nan")
            reject_rate = float("nan")

        print(f"{scale_name},{n_cells},{n_pcs},{k},{N_BATCHES},"
              f"{wall_ms:.1f},{mean_chi2:.4f},{reject_rate:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
