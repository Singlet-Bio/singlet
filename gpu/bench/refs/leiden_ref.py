#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/leiden_ref.py
#
# Performance reference: cugraph Leiden vs singlet-gpu Leiden.
#
# Accepts precomputed kNN (indices + distances .npy), builds a weighted graph,
# runs cugraph.leiden.  Falls back to scanpy sc.tl.leiden if cugraph absent.
#
# Environment:
#   GPU path:  pip install cugraph-cu12 cudf-cu12 numpy scipy anndata scanpy
#   CPU path:  pip install scanpy leidenalg numpy scipy anndata
#
# Usage:
#   python leiden_ref.py \
#       --indices    <knn_indices.npy>   \   # int32 (N, k)
#       --distances  <knn_distances.npy> \   # float32 (N, k)
#       --resolution <float>             \   # default: 1.0
#       --seed       <int>               \   # default: 42
#       --timing-json <result.json>          # required
#
# JSON output schema:
#   { "wall_ms": float, "mem_mb": float, "impl": str, "n": int, "n_clusters": int }

import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp


def _memory_delta_mb(fn, *args, **kwargs):
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


def _build_csr_graph(indices: np.ndarray, distances: np.ndarray):
    """Build scipy CSR connectivity-weighted graph from kNN arrays."""
    N, k = indices.shape
    rows = np.repeat(np.arange(N, dtype=np.int32), k)
    cols = indices.ravel().astype(np.int32)
    dists = distances.ravel().astype(np.float32)
    valid = (cols >= 0) & np.isfinite(dists)
    rows, cols, dists = rows[valid], cols[valid], dists[valid]
    # Connectivity weight: 1 / (1 + d²) — matches singlet-gpu default.
    weights = 1.0 / (1.0 + dists.astype(np.float64) ** 2)
    return sp.csr_matrix((weights.astype(np.float32), (rows, cols)), shape=(N, N))


def run_cugraph_leiden(indices: np.ndarray, distances: np.ndarray,
                       resolution: float, seed: int):
    import cudf
    import cugraph

    csr = _build_csr_graph(indices, distances)
    # Convert to edge list for cugraph.
    coo = csr.tocoo()
    src = cudf.Series(coo.row.astype(np.int32))
    dst = cudf.Series(coo.col.astype(np.int32))
    wts = cudf.Series(coo.data.astype(np.float32))
    G = cugraph.Graph()
    G.from_cudf_edgelist(cudf.DataFrame({"src": src, "dst": dst, "weight": wts}),
                         source="src", destination="dst", edge_attr="weight",
                         renumber=False)
    parts, modularity = cugraph.leiden(G, resolution=resolution, random_state=seed)
    labels = parts.sort_values("vertex")["partition"].values.tolist()
    return np.array(labels, dtype=np.int32), modularity


def run_scanpy_leiden(indices: np.ndarray, distances: np.ndarray,
                      resolution: float, seed: int):
    import anndata
    import scanpy as sc

    N, k = indices.shape
    csr = _build_csr_graph(indices, distances)
    # Distance matrix (for scanpy compat).
    rows = np.repeat(np.arange(N, dtype=np.int32), k)
    cols = indices.ravel().astype(np.int32)
    dists = distances.ravel().astype(np.float32)
    valid = (cols >= 0) & np.isfinite(dists)
    dist_csr = sp.csr_matrix(
        (dists[valid], (rows[valid], cols[valid])), shape=(N, N), dtype=np.float32)

    X_dummy = sp.csr_matrix((N, 1), dtype=np.float32)
    adata = anndata.AnnData(X=X_dummy)
    adata.obsp["distances"]      = dist_csr
    adata.obsp["connectivities"] = csr
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key":      "distances",
        "params": {"n_neighbors": k, "method": "umap", "metric": "euclidean"},
    }
    sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                 key_added="leiden", directed=False)
    return adata.obs["leiden"].cat.codes.values.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="cugraph/scanpy Leiden reference for singlet-gpu Leiden bench")
    parser.add_argument("--indices",     required=True,
                        help="int32 kNN indices .npy, shape (N, k)")
    parser.add_argument("--distances",   required=True,
                        help="float32 kNN distances .npy, shape (N, k)")
    parser.add_argument("--resolution",  type=float, default=1.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--timing-json", required=True,
                        help="Path to write timing JSON")
    args = parser.parse_args()

    indices   = np.load(args.indices).astype(np.int32)
    distances = np.load(args.distances).astype(np.float32)
    N, k = indices.shape
    print(f"[leiden_ref] N={N}, k={k}", file=sys.stderr)

    has_cugraph = False
    try:
        import cugraph  # noqa: F401
        import cudf     # noqa: F401
        has_cugraph = True
    except ImportError:
        print("[leiden_ref] cugraph not available — using scanpy (CPU)", file=sys.stderr)

    run_fn = run_cugraph_leiden if has_cugraph else run_scanpy_leiden
    impl   = "cugraph-leiden" if has_cugraph else "scanpy-leiden"

    # Warmup.
    _ = run_fn(indices, distances, args.resolution, args.seed)

    wall_times = []
    mem_mb_last = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        labels, mem_mb = _memory_delta_mb(run_fn, indices, distances, args.resolution, args.seed)
        wall_times.append((time.perf_counter() - t0) * 1000.0)
        mem_mb_last = mem_mb

    # labels may be a tuple (cugraph returns labels, modularity) — handle both.
    if isinstance(labels, tuple):
        labels = labels[0]

    wall_median_ms = float(sorted(wall_times)[len(wall_times) // 2])
    n_clusters     = int(labels.max()) + 1

    result = {
        "wall_ms":    wall_median_ms,
        "mem_mb":     float(mem_mb_last),
        "impl":       impl,
        "n":          N,
        "n_clusters": n_clusters,
    }
    with open(args.timing_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[leiden_ref] {impl}: N={N} n_clusters={n_clusters} "
          f"wall_ms={wall_median_ms:.1f} mem_mb={mem_mb_last:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
