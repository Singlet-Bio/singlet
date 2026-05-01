#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/asw_ref.py
#
# CYCLE-170 Phase E — manual numpy CPU baseline for integrate/asw.
#
# GPU kernel: one block per cell; shared-mem float smem_sum[n_labels] and
#   int smem_cnt[n_labels]; thread 0 sweeps kNN row accumulating distances by
#   label; computes a (intra-cluster mean dist), b (min over other-cluster mean
#   dists), silhouette = (b - a) / max(a, b).
# Reference: Rousseeuw PJ (1987) J Comput Appl Math 20:53-65.
#            Luecken et al. (2022) Nature Methods 19:41-50 (scIB).
#
# ASW formula (kNN-approximated, scIB default):
#   For each cell c with k nearest neighbors {n_1,...,n_k}, distances {d_1,...,d_k}
#   and cluster label label[c]:
#     1. a(c) = mean(d_i for i where label[n_i] == label[c])
#               = 0 if no neighbors share the label (degenerate).
#     2. b(c) = min over l != label[c] of mean(d_i for i where label[n_i] == l)
#               = 0 if no other-cluster neighbor exists (degenerate).
#     3. silhouette(c) = (b - a) / max(a, b)    ∈ [-1, 1]
#               = 0 if max(a, b) == 0 (degenerate).
#   ASW = mean over all cells.
#
# CPU implementation (vectorised numpy, per-cell loop-free):
#   Build embedding: random fp32 (n × n_pcs), seed=42 — same as GPU bench.
#   Build kNN: sklearn.neighbors.NearestNeighbors (brute, Euclidean, k=k).
#     NOTE: sklearn returns self as the first neighbor; we drop column 0
#           (self-loop) to get k true neighbors — matches the GPU bench which
#           stamps self-distance to FLT_MAX before sort.
#   Cluster labels: round-robin label[c] = c % n_clusters.
#   ASW vectorised:
#     label_nbrs  = label[neighbors]      # (n, k) int
#     dist_nbrs   = distances[:, 1:]      # (n, k) float  — drop col 0 (self)
#     For each cell c, for each label l:
#       mask_l[c] = label_nbrs[c] == l    # bool (k,)
#       sum_l[c]  = dist_nbrs[c][mask_l[c]].sum()
#       cnt_l[c]  = mask_l[c].sum()
#       mean_l[c] = sum_l[c] / cnt_l[c] if cnt_l[c] > 0 else 0
#     Vectorised:
#       one_hot[c, l] = (label_nbrs[c, :] == l)          # (n, k) bool → (n, n_clusters) after sum
#       sum_dist[c, l] = sum of dist_nbrs[c] masked by label l
#         → broadcast: (n, k, 1) * (n, k, n_clusters bool) → sum axis=1 → (n, n_clusters)
#       cnt[c, l] = one_hot.sum(axis=1)                  # (n, n_clusters) int
#       mean_dist[c, l] = sum_dist[c, l] / max(cnt[c, l], 1)  # (n, n_clusters) float
#       self_labels = label                              # (n,) int
#       a[c] = mean_dist[c, self_labels[c]]
#       b[c] = min over l != self_labels[c] of mean_dist[c, l], where cnt[c, l] > 0
#            = masked min with large sentinel where cnt == 0 or l == self_label
#       sil[c] = (b[c] - a[c]) / max(a[c], b[c])  (0 if degenerate)
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   kNN build is NOT timed (matches GPU bench: kNN is untimed warmup).
#   Only the ASW computation is timed.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_pcs,k,n_clusters,wall_ms
#
# Usage: python3 asw_ref.py

import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_CLUSTERS   = 4

SCALES = [
    ("10k", 10000, 50, 10),
    ("30k", 30000, 50, 15),
]


def make_synthetic_embedding(n_cells: int, n_pcs: int, seed: int = 42) -> np.ndarray:
    """Row-major float32 embedding (n_cells × n_pcs) matching GPU bench LCG seed=42.

    Uses numpy default_rng for reproducibility (different from C++ LCG but that
    is acceptable for a CPU baseline — both produce random embeddings in [-1, 1]).
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1.0, 1.0, size=(n_cells, n_pcs)).astype(np.float32)


def build_knn(emb: np.ndarray, k: int):
    """Return (neighbor_indices, distances), each (n_cells × k), excluding self.

    sklearn.NearestNeighbors with n_neighbors=k+1 (includes self at col 0).
    Brute-force Euclidean — matches GPU Exact backend (L2).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean",
                          n_jobs=-1)
    nn.fit(emb)
    dists, indices = nn.kneighbors(emb)
    # Drop self (column 0 is always the query cell itself at distance 0).
    return indices[:, 1:k + 1].astype(np.int32), dists[:, 1:k + 1].astype(np.float32)


def compute_asw_vectorized(neighbors: np.ndarray,
                            distances: np.ndarray,
                            labels: np.ndarray,
                            n_clusters: int) -> float:
    """Vectorised ASW over all cells.  Returns scalar mean silhouette.

    neighbors : (n_cells, k) int32   — k nearest neighbor indices per cell.
    distances : (n_cells, k) float32 — corresponding L2 distances per cell.
    labels    : (n_cells,)   int32   — cluster label per cell in [0, n_clusters).
    Returns   : scalar float          — mean silhouette (ASW).
    """
    n_cells, k = neighbors.shape

    # Gather labels of each cell's neighbors: (n_cells, k) int.
    label_nbrs = labels[neighbors]   # advanced indexing

    # Indicator: (n_cells, k, n_clusters) bool
    # one_hot[c, i, l] = 1 iff neighbor i of cell c has label l.
    one_hot = (label_nbrs[:, :, np.newaxis] ==
               np.arange(n_clusters, dtype=np.int32)[np.newaxis, np.newaxis, :])
    # → (n_cells, n_clusters)
    cnt = one_hot.sum(axis=1).astype(np.float32)   # (n_cells, n_clusters)

    # Sum of distances per label per cell: (n_cells, n_clusters)
    # distances[:, :, None] * one_hot → (n_cells, k, n_clusters) → sum axis=1
    sum_dist = (distances[:, :, np.newaxis] * one_hot).sum(axis=1).astype(np.float32)

    # Mean distance per label per cell (0 where cnt == 0).
    # Avoid division by zero: replace cnt==0 with 1 for division, then zero out.
    safe_cnt = np.where(cnt > 0, cnt, np.float32(1.0))
    mean_dist = sum_dist / safe_cnt                # (n_cells, n_clusters)
    mean_dist = np.where(cnt > 0, mean_dist, np.float32(0.0))

    # a(c): intra-cluster mean distance.
    cell_idx = np.arange(n_cells, dtype=np.int32)
    a = mean_dist[cell_idx, labels]                # (n_cells,)

    # b(c): min mean distance to any OTHER cluster present in kNN.
    # Mask: True where this label is the cell's own label OR cnt==0 (no neighbors).
    self_mask = (np.arange(n_clusters, dtype=np.int32)[np.newaxis, :] ==
                 labels[:, np.newaxis])             # (n_cells, n_clusters) bool
    no_nbrs   = (cnt == 0)
    # Apply sentinel (large value) where masked so argmin skips them.
    INF = np.float32(1e30)
    mean_other = np.where(self_mask | no_nbrs, INF, mean_dist)  # (n_cells, n_clusters)

    # b = row-wise min of mean_other; if all entries are INF, b = 0 (degenerate).
    b = mean_other.min(axis=1)                     # (n_cells,)
    b = np.where(b >= INF, np.float32(0.0), b)

    # silhouette(c) = (b - a) / max(a, b); = 0 if degenerate (max == 0).
    denom = np.maximum(a, b)                       # (n_cells,)
    sil = np.where(denom > 0, (b - a) / denom, np.float32(0.0))

    return float(sil.mean())


def time_asw(neighbors: np.ndarray, distances: np.ndarray,
             labels: np.ndarray, n_clusters: int) -> float:
    """2 warmup + 5 timed iterations of compute_asw_vectorized.

    Returns median wall_ms.
    """
    def _run():
        return compute_asw_vectorized(neighbors, distances, labels, n_clusters)

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
    print("scale,n_cells,n_pcs,k,n_clusters,wall_ms", flush=True)

    for scale_name, n_cells, n_pcs, k in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_pcs} PCs "
              f"k={k} n_clusters={N_CLUSTERS}...",
              file=sys.stderr, flush=True)

        emb = make_synthetic_embedding(n_cells, n_pcs, seed=42)

        print(f"[ref] Building kNN (sklearn brute, k={k})...",
              file=sys.stderr, flush=True)
        neighbors, distances = build_knn(emb, k)   # (n_cells, k) — untimed

        # Round-robin cluster labels: label[c] = c % N_CLUSTERS.
        labels = (np.arange(n_cells, dtype=np.int32) % N_CLUSTERS)

        print(f"[ref] Timing {scale_name} asw vectorized (n_clusters={N_CLUSTERS} k={k})...",
              file=sys.stderr, flush=True)

        try:
            wall_ms = time_asw(neighbors, distances, labels, N_CLUSTERS)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_pcs},{k},{N_CLUSTERS},{wall_ms:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
