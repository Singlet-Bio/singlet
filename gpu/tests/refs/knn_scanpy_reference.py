#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/knn_scanpy_reference.py
#
# Reference driver: run scikit-learn brute-force kNN on a dense fp32 embedding
# and write (neighbors[N, k], distances[N, k]) to a .npz file.
#
# Environment: pip install scikit-learn numpy
#
# Usage:
#   python knn_scanpy_reference.py \
#       --input   <embedding.npy>    \   # float32 array, shape (N, d)
#       --output  <result.npz>       \   # keys: neighbors, distances (both [N, k])
#       --k       <int>              \
#       [--metric <euclidean|cosine|inner>]   # default: euclidean
#       [--return-squared]                    # return squared L2 instead of L2
#
# The C++ test calls this as a subprocess, then reads result.npz.
#
# Self-exclusion: scikit-learn's kneighbors(X) returns k+1 neighbors (the
# query itself at distance 0 is index 0).  We strip it so the output is exactly
# k non-self neighbors per row, sorted by (distance, neighbor_id).

import argparse
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def run_knn_reference(X: np.ndarray, k: int, metric: str = "euclidean",
                      return_squared: bool = False):
    """
    Run brute-force kNN via scikit-learn.

    Returns:
        neighbors  : int32 array shape (N, k), indices of k nearest non-self neighbors
        distances  : float32 array shape (N, k), L2 (or squared-L2, or cosine) distances
    """
    N = X.shape[0]
    # If k >= N we return all non-self neighbors (at most N-1).
    k_eff = min(k, N - 1)

    if metric == "cosine":
        # Normalize rows then use euclidean; L2 on unit vectors == sqrt(2(1-cos)).
        # We report cosine distance = 1 - cos_similarity.
        X_norm = normalize(X, norm="l2")
        sk_metric = "euclidean"
    elif metric == "inner":
        # Negative dot product; cosine-like but without normalization.
        # sklearn doesn't support inner product natively for BruteForce;
        # we use euclidean on the raw vectors and post-process.
        # Actually: inner-product NN == euclidean NN after centering only if
        # norms are equal. We implement it explicitly as -X @ X.T.
        X_norm = None
        sk_metric = None
    else:
        X_norm = X
        sk_metric = "euclidean"

    if sk_metric is not None:
        # k+1 because NearestNeighbors includes self
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=sk_metric,
                              algorithm="brute")
        nn.fit(X_norm)
        dists, inds = nn.kneighbors(X_norm)
        # Strip self (always index 0 for brute-force, distance ~0)
        dists = dists[:, 1:]   # shape (N, k_eff)
        inds  = inds[:, 1:]

        if metric == "cosine":
            # Convert euclidean-on-unit-vectors to cosine distance:
            # euclidean(u,v)^2 = 2(1 - cos) => cos_dist = eucl^2 / 2
            dists = (dists ** 2) / 2.0

    else:
        # Inner product: build full distance matrix D[i,j] = -X[i] @ X[j]
        D = -(X @ X.T)  # (N, N)
        inds  = np.argsort(D, axis=1)  # ascending (most negative = most similar)
        dists_full = D[np.arange(N)[:, None], inds]

        # Remove self from each row
        def strip_self(row_inds, row_dists, i):
            mask = row_inds != i
            return row_inds[mask][:k_eff], row_dists[mask][:k_eff]

        inds  = np.array([strip_self(inds[i], dists_full[i], i)[0]
                          for i in range(N)], dtype=np.int32)
        dists = np.array([strip_self(dists_full[i], dists_full[i], i)[1]   # noqa
                          for i in range(N)], dtype=np.float32)

    # Pad with -1 / inf if k_eff < k (edge case: n < k+1)
    if k_eff < k:
        pad_n = k - k_eff
        pad_i = np.full((N, pad_n), -1, dtype=np.int32)
        pad_d = np.full((N, pad_n), np.inf, dtype=np.float32)
        inds  = np.concatenate([inds,  pad_i], axis=1)
        dists = np.concatenate([dists, pad_d], axis=1)

    # Deterministic tie-break: sort each row by (distance, neighbor_id).
    for i in range(N):
        order = np.lexsort((inds[i], dists[i]))
        inds[i]  = inds[i][order]
        dists[i] = dists[i][order]

    if return_squared and metric == "euclidean":
        dists = dists ** 2

    return inds.astype(np.int32), dists.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="scikit-learn brute-force kNN reference for singlet-gpu correctness test")
    parser.add_argument("--input",  required=True,
                        help="Path to float32 embedding .npy, shape (N, d)")
    parser.add_argument("--output", required=True,
                        help="Path to output .npz with keys: neighbors, distances")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of nearest neighbors (excluding self)")
    parser.add_argument("--metric", choices=["euclidean", "cosine", "inner"],
                        default="euclidean")
    parser.add_argument("--return-squared", action="store_true",
                        help="Return squared L2 distances (euclidean only)")
    args = parser.parse_args()

    X = np.load(args.input).astype(np.float32)
    N, d = X.shape
    print(f"[ref] Loaded embedding: {N} x {d}, dtype={X.dtype}", file=sys.stderr)

    neighbors, distances = run_knn_reference(
        X, k=args.k, metric=args.metric, return_squared=args.return_squared)

    print(f"[ref] neighbors shape: {neighbors.shape}, distances shape: {distances.shape}",
          file=sys.stderr)
    np.savez(args.output, neighbors=neighbors, distances=distances)
    print(f"[ref] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
