#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/knn_ref.py
#
# Performance reference: cuML NearestNeighbors k=15 vs singlet-gpu kNN.
#
# Measures wall time for cuML's GPU kNN on a float32 embedding. Falls back to
# scikit-learn BruteForce if cuml is absent (CPU baseline).  Reports timing in
# a JSON file consumed by the C++ bench driver.
#
# Environment:
#   GPU path:  pip install cuml-cu12 numpy
#   CPU path:  pip install scikit-learn numpy  (no GPU required)
#
# Usage:
#   python knn_ref.py \
#       --input    <embedding.npy>    \   # float32 (N, D)
#       --k        <int>              \   # default: 15
#       --output   <neighbors.npy>   \   # int32 (N, k)  [optional]
#       --timing-json <result.json>  \   # timing output (required by bench driver)
#       [--metric  euclidean|cosine]
#
# JSON output schema:
#   { "wall_ms": float, "mem_mb": float, "impl": str, "n": int, "k": int }

import argparse
import json
import os
import sys
import time

import numpy as np


def _memory_delta_mb(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return (result, delta_mb) using pynvml or /proc."""
    # Try GPU memory via pynvml.
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        result = fn(*args, **kwargs)
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        return result, max(0.0, mem_after - mem_before)
    except Exception:
        pass
    # Fallback: RSS via /proc/self/status.
    def _rss():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
        return 0.0
    rss_before = _rss()
    result = fn(*args, **kwargs)
    return result, max(0.0, _rss() - rss_before)


def run_cuml_knn(X: np.ndarray, k: int, metric: str = "euclidean"):
    import cuml
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

    nn = cuNearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute",
                            output_type="numpy")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # Strip self (index 0, distance ≈ 0).
    return indices[:, 1:].astype(np.int32)


def run_sklearn_knn(X: np.ndarray, k: int, metric: str = "euclidean"):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices[:, 1:].astype(np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="cuML NearestNeighbors k=15 reference for singlet-gpu kNN bench")
    parser.add_argument("--input",       required=True,
                        help="float32 .npy embedding, shape (N, D)")
    parser.add_argument("--k",           type=int, default=15,
                        help="Number of nearest neighbors (excl. self)")
    parser.add_argument("--metric",      default="euclidean",
                        choices=["euclidean", "cosine"])
    parser.add_argument("--output",      default=None,
                        help="Optional path to write int32 neighbors .npy")
    parser.add_argument("--timing-json", required=True,
                        help="Path to write timing JSON")
    args = parser.parse_args()

    X = np.load(args.input).astype(np.float32)
    N, D = X.shape
    print(f"[knn_ref] Loaded embedding: N={N}, D={D}", file=sys.stderr)

    # Try cuML first, fall back to scikit-learn.
    has_cuml = False
    try:
        import cuml  # noqa: F401
        has_cuml = True
    except ImportError:
        print("[knn_ref] cuml not available — using scikit-learn (CPU)", file=sys.stderr)

    run_fn  = run_cuml_knn  if has_cuml else run_sklearn_knn
    impl    = "cuml-NearestNeighbors" if has_cuml else "sklearn-BruteForce"

    # Warmup run (discarded).
    _ = run_fn(X, args.k, args.metric)

    # 5 timed runs.
    wall_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        neighbors, mem_mb = _memory_delta_mb(run_fn, X, args.k, args.metric)
        wall_times.append((time.perf_counter() - t0) * 1000.0)

    wall_median_ms = float(sorted(wall_times)[len(wall_times) // 2])

    if args.output:
        np.save(args.output, neighbors)
        print(f"[knn_ref] Neighbors saved to {args.output}", file=sys.stderr)

    result = {
        "wall_ms": wall_median_ms,
        "mem_mb":  float(mem_mb),
        "impl":    impl,
        "n":       N,
        "k":       args.k,
    }
    with open(args.timing_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[knn_ref] {impl}: N={N} k={args.k} wall_ms={wall_median_ms:.1f} "
          f"mem_mb={mem_mb:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
