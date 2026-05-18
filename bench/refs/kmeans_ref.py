#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/kmeans_ref.py
#
# CYCLE-172 Phase E — sklearn KMeans CPU baseline for graph/kmeans.
#
# GPU kernel: Forgy init (host mt19937 random column copy); Lloyd loop:
#   Sgemm D = X^T·C, per-cell argmin, atomic-scatter centroid update.
#   D2H scalar change count per iter; final inertia via km_inertia_kernel.
# Reference: Lloyd (1982) "Least squares quantization in PCM." IEEE TIT 28:129-137.
#
# CPU implementation: sklearn.cluster.KMeans
#   init='random'  — Forgy-equivalent (random cell selection as initial centroids).
#   n_init=1       — single initialisation (matches GPU single-run bench).
#   max_iter=20    — matches GPU bench max_iter.
#   algorithm='lloyd' — explicit Lloyd iterations (sklearn default for small k).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Embedding synthesis is NOT timed (matches GPU bench: untimed build).
#   Only the KMeans.fit() call is timed.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_pcs,k,max_iter,wall_ms,inertia,n_iter
#
# Usage: python3 kmeans_ref.py

import sys
import time

import numpy as np
from sklearn.cluster import KMeans

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 50, 10),
    ("30k", 30000, 50, 10),
]

MAX_ITER = 20


def make_synthetic_embedding(n_cells: int, n_pcs: int, seed: int = 42) -> np.ndarray:
    """Row-major float32 embedding (n_cells × n_pcs), seed=42.

    Uses numpy default_rng; different LCG than C++ bench but both produce
    random embeddings in [-1, 1] — acceptable for a CPU timing baseline.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1.0, 1.0, size=(n_cells, n_pcs)).astype(np.float32)


def time_kmeans(emb: np.ndarray, k: int, max_iter: int) -> tuple:
    """2 warmup + 5 timed iterations of KMeans.fit().

    init='random' matches GPU Forgy (random cell column copy).
    n_init=1 matches single-run GPU bench.
    algorithm='lloyd' forces explicit Lloyd (no elkan shortcut).

    Returns (median_wall_ms, inertia, n_iter) from the last timed run.
    """
    last_inertia = 0.0
    last_n_iter  = 0

    def _run():
        km = KMeans(n_clusters=k, init='random', n_init=1,
                    max_iter=max_iter, algorithm='lloyd', random_state=42)
        km.fit(emb)
        return km.inertia_, km.n_iter_

    for _ in range(WARMUP_ITERS):
        try:
            _run()
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        inertia, n_iter = _run()
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)
        last_inertia = inertia
        last_n_iter  = n_iter

    wall_times.sort()
    n_t = len(wall_times)
    median_ms = (wall_times[n_t // 2] if n_t % 2 == 1
                 else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))

    return median_ms, last_inertia, last_n_iter


def main():
    print("scale,n_cells,n_pcs,k,max_iter,wall_ms,inertia,n_iter",
          flush=True)

    for scale_name, n_cells, n_pcs, k in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_pcs} PCs "
              f"k={k} max_iter={MAX_ITER}...",
              file=sys.stderr, flush=True)

        emb = make_synthetic_embedding(n_cells, n_pcs, seed=42)

        print(f"[ref] Timing {scale_name} KMeans(init='random', n_init=1, "
              f"max_iter={MAX_ITER})...",
              file=sys.stderr, flush=True)

        try:
            wall_ms, inertia, n_iter = time_kmeans(emb, k, MAX_ITER)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms  = -1.0
            inertia  = float("nan")
            n_iter   = -1

        print(f"{scale_name},{n_cells},{n_pcs},{k},{MAX_ITER},"
              f"{wall_ms:.1f},{inertia:.2f},{n_iter}",
              flush=True)


if __name__ == "__main__":
    main()
