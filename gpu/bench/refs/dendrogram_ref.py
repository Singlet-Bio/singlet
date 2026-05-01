#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/dendrogram_ref.py
#
# CYCLE-173 Phase E — scipy/numpy CPU baseline for embed/dendrogram.
#
# GPU kernel (embed::dendrogram):
#   (a) atomic-scatter centroid (one block/cell over sparse nnz),
#   (b) centroid divide /= n_per_cluster,
#   (c) per-column mean + center,
#   (d) per-column L2 norm + normalize,
#   (e) cuBLAS Sgemm corr = μ_norm^T · μ_norm (k × k),
#   (f) d[i,j] = 1 - corr[i,j] (element-wise),
#   Host UPGMA O(k³).
# Reference: scanpy.tl.dendrogram (Wolf et al. 2018 Genome Biol 19:15).
#
# CPU implementation (scipy/numpy):
#   1. Centroid aggregation: np.add.at (row-indexed scatter sum) + divide by counts.
#   2. Center each centroid column: col -= col.mean().
#   3. Normalize each centroid column: col /= ||col||₂  (eps guard).
#   4. Pairwise correlation distance: scipy.spatial.distance.pdist(μ_norm.T, metric='cosine').
#      (cosine distance on unit-normalized vectors = 1 - correlation = our d[i,j])
#   5. UPGMA linkage: scipy.cluster.hierarchy.linkage(dist_vec, method='average').
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Embedding synthesis is NOT timed (matches GPU bench: untimed build).
#   All 5 steps (centroid → center → norm → pdist → linkage) are timed together.
#
# Scales:
#   10k: n_cells=10000, n_genes=50 (PCA dims), n_clusters=20, density=1.0
#   30k: n_cells=30000, n_genes=50, n_clusters=20, density=1.0
#
# Labels: round-robin  label[c] = c % n_clusters  (matches C++ bench).
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_pcs,n_clusters,wall_ms
#
# Usage: python3 dendrogram_ref.py

import sys
import time

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 50, 20),
    ("30k", 30000, 50, 20),
]


def make_synthetic_dense_matrix(n_genes: int, n_cells: int,
                                 seed: int = 42) -> np.ndarray:
    """Dense float32 matrix (n_genes × n_cells) with integer count values 0–10.

    Uses numpy default_rng(seed) with integer sampling to match the C++ LCG
    intent (counts-like synthetic data).  Row-major layout (n_genes rows).
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 11, size=(n_genes, n_cells)).astype(np.float32)


def make_roundrobin_labels(n_cells: int, n_clusters: int) -> np.ndarray:
    """Integer label array label[c] = c % n_clusters, shape (n_cells,)."""
    return np.arange(n_cells, dtype=np.int32) % n_clusters


def run_dendrogram(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> None:
    """Execute the full scipy dendrogram pipeline (all 5 CPU steps).

    Step 1: centroid scatter — np.add.at + divide by counts.
    Step 2: center each centroid column.
    Step 3: L2-normalize each centroid column (eps guard: skip zero vectors).
    Step 4: pairwise cosine distance via scipy.spatial.distance.pdist.
    Step 5: UPGMA linkage via scipy.cluster.hierarchy.linkage.

    Returns nothing (wall-time only benchmark).
    """
    n_genes, n_cells = X.shape

    # Step 1 — centroid aggregation (n_genes × n_clusters, col-major ≡ F-order).
    mu = np.zeros((n_genes, n_clusters), dtype=np.float32, order='F')
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    np.add.at(mu, (slice(None), labels), X)
    # Divide each column by cluster count (broadcast over rows).
    nonzero = counts > 0
    mu[:, nonzero] /= counts[nonzero]

    # Step 2 — center each centroid column.
    mu -= mu.mean(axis=0, keepdims=True)

    # Step 3 — L2-normalize each centroid column.
    norms = np.linalg.norm(mu, axis=0)
    nonzero_norm = norms > 1e-9
    mu[:, nonzero_norm] /= norms[nonzero_norm]

    # Step 4 — pairwise cosine distance on unit-normalized vectors.
    # pdist on (n_clusters × n_genes) row-major with metric='cosine'.
    dist_vec = pdist(mu.T, metric='cosine')   # mu.T is (n_clusters × n_genes)

    # Step 5 — UPGMA (average linkage) hierarchical clustering.
    _ = linkage(dist_vec, method='average')


def time_dendrogram(X: np.ndarray, labels: np.ndarray,
                    n_clusters: int) -> float:
    """2 warmup + 5 timed iterations.  Returns median wall time in ms."""
    for _ in range(WARMUP_ITERS):
        try:
            run_dendrogram(X, labels, n_clusters)
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        run_dendrogram(X, labels, n_clusters)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    return (wall_times[n_t // 2] if n_t % 2 == 1
            else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))


def main():
    print("scale,n_cells,n_pcs,n_clusters,wall_ms", flush=True)

    for scale_name, n_cells, n_pcs, n_clusters in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_pcs} genes(PCs) × "
              f"{n_cells} cells, k={n_clusters} clusters...",
              file=sys.stderr, flush=True)

        X      = make_synthetic_dense_matrix(n_pcs, n_cells, seed=42)
        labels = make_roundrobin_labels(n_cells, n_clusters)

        print(f"[ref] Timing {scale_name} scipy dendrogram "
              f"(n_clusters={n_clusters})...",
              file=sys.stderr, flush=True)

        try:
            wall_ms = time_dendrogram(X, labels, n_clusters)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_pcs},{n_clusters},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
