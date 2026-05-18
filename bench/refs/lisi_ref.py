#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/lisi_ref.py
#
# CYCLE-169 Phase E — manual numpy CPU baseline for integrate/lisi.
#
# GPU kernel: one block per cell; shared-memory histogram of n_labels ints;
#   accumulates label counts for k neighbors; thread 0 computes
#   D = sum(p_l^2) and writes LISI = 1/D.
# Reference: Korsunsky I et al. (2019) Nat Methods 16:1289-1296.
#
# LISI formula (uniform neighbor weights):
#   For each cell c with k nearest neighbors {n_1,...,n_k} and labels label[c]:
#     1. p_l = count(label[n_i] == l) / k  for each label l.
#     2. D[c] = sum_l p_l^2   (Simpson's diversity index)
#     3. LISI[c] = 1 / D[c]
#
# CPU implementation (vectorised numpy):
#   Build embedding: random fp32 (n × n_pcs), seed=42 — same as GPU bench.
#   Build kNN: sklearn.neighbors.NearestNeighbors (brute, Euclidean, k=10).
#     NOTE: sklearn returns self as the first neighbor; we drop column 0
#           (index neighbor = self) to get k true neighbors, matching the GPU
#           bench which excludes the self-loop via FLT_MAX sentinel.
#   Batch labels: round-robin label[c] = c % n_batches.
#   LISI vectorised:
#     label_nbrs = label[neighbors]            # (n, k) int
#     For each cell c: count occurrences of each label in label_nbrs[c],
#       compute p_l = count / k, D = sum(p_l^2), LISI = 1/D.
#     Vectorised over cells via numpy: use np.apply_along_axis or
#     a label-hot approach with np.eye + matrix multiply.
#     Hot approach: one_hot[c, l] = count(label_nbrs[c] == l)
#       P = one_hot / k                         # (n, n_batches) float
#       D = (P**2).sum(axis=1)                  # (n,) float
#       LISI = 1 / np.maximum(D, 1e-9)          # (n,) float
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   kNN build is NOT timed (matches GPU bench: kNN is untimed warmup).
#   Only the LISI computation is timed.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_pcs,k,n_batches,wall_ms
#
# Usage: python3 lisi_ref.py

import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

WARMUP_ITERS = 2
TIMED_ITERS  = 5
K            = 10
N_BATCHES    = 4

SCALES = [
    ("10k", 10000, 50),
    ("30k", 30000, 50),
]


def make_synthetic_embedding(n_cells: int, n_pcs: int, seed: int = 42) -> np.ndarray:
    """Row-major float32 embedding (n_cells × n_pcs) matching GPU bench LCG seed=42.

    Uses numpy default_rng for reproducibility (different from C++ LCG but that
    is acceptable for a CPU baseline — both produce random embeddings in [-1, 1]).
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1.0, 1.0, size=(n_cells, n_pcs)).astype(np.float32)


def build_knn(emb: np.ndarray, k: int) -> np.ndarray:
    """Return neighbor indices (n_cells × k), excluding self (column 0).

    sklearn.NearestNeighbors with n_neighbors=k+1 (includes self at col 0).
    Brute-force Euclidean — matches GPU Exact backend (L2, k=10).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean",
                          n_jobs=-1)
    nn.fit(emb)
    _, indices = nn.kneighbors(emb)
    # Drop self (column 0 is always the query cell itself at distance 0).
    return indices[:, 1:k + 1].astype(np.int32)   # (n_cells, k)


def compute_lisi_vectorized(neighbors: np.ndarray,
                             labels: np.ndarray,
                             n_batches: int,
                             k: int) -> np.ndarray:
    """Vectorised LISI over all cells.

    neighbors : (n_cells, k) int32 — k nearest neighbor indices per cell.
    labels    : (n_cells,)   int32 — batch label per cell in [0, n_batches).
    Returns   : (n_cells,)   float32 — per-cell LISI score.
    """
    n_cells = neighbors.shape[0]

    # Gather labels of each cell's neighbors: (n_cells, k) int.
    label_nbrs = labels[neighbors]   # advanced indexing, shape (n_cells, k)

    # Count occurrences of each label per cell via one-hot + sum.
    # one_hot[c, l] = number of neighbors of cell c with label l.
    # Encoding: compare label_nbrs (n, k) to arange(n_batches) (1, 1, n_batches).
    # Expand dims: label_nbrs[:, :, None] == arange[None, None, :] → (n, k, n_batches)
    one_hot = (label_nbrs[:, :, np.newaxis] ==
               np.arange(n_batches, dtype=np.int32)[np.newaxis, np.newaxis, :])
    # Sum over k neighbors: (n, n_batches) int → float
    counts = one_hot.sum(axis=1).astype(np.float32)   # (n_cells, n_batches)

    # Proportions.
    P = counts / float(k)   # (n_cells, n_batches) float32

    # Simpson's D = sum_l p_l^2.
    D = (P * P).sum(axis=1)   # (n_cells,) float32

    # LISI = 1 / max(D, eps).
    eps = np.float32(1e-9)
    lisi = 1.0 / np.maximum(D, eps)
    return lisi


def time_lisi(neighbors: np.ndarray, labels: np.ndarray,
              n_batches: int, k: int) -> float:
    """2 warmup + 5 timed iterations of compute_lisi_vectorized.

    Returns median wall_ms.
    """
    def _run():
        return compute_lisi_vectorized(neighbors, labels, n_batches, k)

    # Warmup.
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
    print("scale,n_cells,n_pcs,k,n_batches,wall_ms", flush=True)

    for scale_name, n_cells, n_pcs in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_pcs} PCs "
              f"k={K} n_batches={N_BATCHES}...",
              file=sys.stderr, flush=True)

        emb = make_synthetic_embedding(n_cells, n_pcs, seed=42)

        print(f"[ref] Building kNN (sklearn brute, k={K})...",
              file=sys.stderr, flush=True)
        neighbors = build_knn(emb, K)   # (n_cells, K) — untimed (matches GPU bench)

        # Round-robin batch labels: label[c] = c % N_BATCHES.
        labels = (np.arange(n_cells, dtype=np.int32) % N_BATCHES)

        print(f"[ref] Timing {scale_name} lisi vectorized (n_batches={N_BATCHES} k={K})...",
              file=sys.stderr, flush=True)

        try:
            wall_ms = time_lisi(neighbors, labels, N_BATCHES, K)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_pcs},{K},{N_BATCHES},{wall_ms:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
