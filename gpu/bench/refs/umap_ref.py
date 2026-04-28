#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/umap_ref.py
#
# Performance reference: cuML UMAP (precomputed kNN path) vs singlet-gpu UMAP.
#
# Accepts precomputed kNN (indices + distances .npy), runs cuML UMAP.
# Falls back to umap-learn if cuml absent.
#
# Environment:
#   GPU path:  pip install cuml-cu12 numpy scipy
#   CPU path:  pip install umap-learn numpy scipy
#
# Usage:
#   python umap_ref.py \
#       --indices     <knn_indices.npy>   \   # int32  (N, k)
#       --distances   <knn_distances.npy> \   # float32 (N, k)
#       --n-neighbors <int>               \   # must match k; default: 15
#       --n-components <int>              \   # default: 2
#       --seed        <int>               \   # default: 42
#       --timing-json <result.json>           # required
#
# JSON output schema:
#   { "wall_ms": float, "mem_mb": float, "impl": str, "n": int, "n_components": int }

import argparse
import json
import sys
import time

import numpy as np


def _memory_delta_mb(fn, *args, **kwargs):
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        result = fn(*args, **kwargs)
        after  = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        return result, max(0.0, after - before)
    except Exception:
        pass

    def _rss():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
        return 0.0

    before = _rss()
    result = fn(*args, **kwargs)
    return result, max(0.0, _rss() - before)


def run_cuml_umap(indices: np.ndarray, distances: np.ndarray,
                  n_neighbors: int, n_components: int, seed: int,
                  min_dist: float = 0.5, spread: float = 1.0) -> np.ndarray:
    from cuml.manifold import UMAP as cuUMAP
    N = indices.shape[0]
    X_dummy = np.zeros((N, 1), dtype=np.float32)

    # cuML UMAP precomputed kNN: pass (indices, distances) as knn_graph.
    # cuML expects int64 indices; convert once here.
    knn_graph = (indices.astype(np.int64), distances.astype(np.float32))

    reducer = cuUMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        init="random",
        random_state=seed,
        min_dist=min_dist,
        spread=spread,
        verbose=False,
    )
    # cuML UMAP accepts precomputed_knn via fit_transform(X, knn_graph=...).
    embedding = reducer.fit_transform(X_dummy, knn_graph=knn_graph)
    return np.array(embedding, dtype=np.float32)


def run_umap_learn(indices: np.ndarray, distances: np.ndarray,
                   n_neighbors: int, n_components: int, seed: int,
                   min_dist: float = 0.5, spread: float = 1.0) -> np.ndarray:
    import umap
    N = indices.shape[0]
    X_dummy = np.zeros((N, 1), dtype=np.float32)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        init="random",
        random_state=seed,
        min_dist=min_dist,
        spread=spread,
        metric="precomputed",
        precomputed_knn=(indices, distances),
    )
    return reducer.fit_transform(X_dummy).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="cuML/umap-learn UMAP reference for singlet-gpu UMAP bench")
    parser.add_argument("--indices",      required=True,
                        help="int32 kNN indices .npy, shape (N, k)")
    parser.add_argument("--distances",    required=True,
                        help="float32 kNN distances .npy, shape (N, k)")
    parser.add_argument("--n-neighbors",  type=int, default=15,
                        help="Must match k in the kNN arrays")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--min-dist",     type=float, default=0.5)
    parser.add_argument("--spread",       type=float, default=1.0)
    parser.add_argument("--timing-json",  required=True,
                        help="Path to write timing JSON")
    args = parser.parse_args()

    indices   = np.load(args.indices).astype(np.int32)
    distances = np.load(args.distances).astype(np.float32)
    N, k = indices.shape
    print(f"[umap_ref] N={N}, k={k}", file=sys.stderr)

    has_cuml = False
    try:
        import cuml  # noqa: F401
        has_cuml = True
    except ImportError:
        print("[umap_ref] cuml not available — using umap-learn (CPU)", file=sys.stderr)

    run_fn = run_cuml_umap  if has_cuml else run_umap_learn
    impl   = "cuml-UMAP"    if has_cuml else "umap-learn"

    # Warmup.
    _ = run_fn(indices, distances, args.n_neighbors, args.n_components, args.seed,
               args.min_dist, args.spread)

    wall_times = []
    mem_mb_last = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        emb, mem_mb = _memory_delta_mb(run_fn, indices, distances,
                                       args.n_neighbors, args.n_components, args.seed,
                                       args.min_dist, args.spread)
        wall_times.append((time.perf_counter() - t0) * 1000.0)
        mem_mb_last = mem_mb

    wall_median_ms = float(sorted(wall_times)[len(wall_times) // 2])

    result = {
        "wall_ms":     wall_median_ms,
        "mem_mb":      float(mem_mb_last),
        "impl":        impl,
        "n":           N,
        "n_components": args.n_components,
    }
    with open(args.timing_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[umap_ref] {impl}: N={N} nc={args.n_components} "
          f"wall_ms={wall_median_ms:.1f} mem_mb={mem_mb_last:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
