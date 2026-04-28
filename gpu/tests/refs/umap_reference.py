#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/umap_reference.py
#
# Reference driver: run umap-learn on a precomputed kNN graph and write the
# resulting embedding to a .npy file.
#
# Environment: pip install umap-learn numpy scipy
#
# Usage:
#   python umap_reference.py \
#       --indices     <indices.npy>    \   # int32 array, shape (N, k)
#       --distances   <distances.npy>  \   # float32 array, shape (N, k)
#       --output      <embedding.npy>  \   # float32 array, shape (N, n_components)
#       [--n-components <int>]             # default: 2
#       [--n-neighbors  <int>]             # must match k from kNN; default: 15
#       [--seed         <int>]             # random_state; default: 42
#       [--min-dist     <float>]           # default: 0.5
#       [--spread       <float>]           # default: 1.0
#
# The C++ test calls this as a subprocess (via run_cmd), then loads embedding.npy.
#
# Note: umap-learn's UMAP(precomputed_knn=(indices, distances)) takes the indices
# and distances from a precomputed kNN graph and uses them directly without
# re-computing neighbors.  The 'X' argument to fit_transform is only used to
# obtain the number of samples; we pass a dummy zeros array.

import argparse
import sys
import numpy as np

try:
    import umap
except ImportError:
    print("[umap_reference] ERROR: umap-learn not installed. "
          "Run: pip install umap-learn numpy scipy", file=sys.stderr)
    sys.exit(1)


def run_umap_reference(
    indices: np.ndarray,
    distances: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    seed: int = 42,
    min_dist: float = 0.5,
    spread: float = 1.0,
) -> np.ndarray:
    """
    Run umap-learn on a precomputed kNN graph.

    Parameters
    ----------
    indices    : int32 (N, k) — kNN neighbor indices
    distances  : float32 (N, k) — kNN neighbor distances
    n_components: int — embedding dimensionality
    n_neighbors : int — must match k; passed to UMAP for graph construction params
    seed        : int — random_state for reproducibility
    min_dist    : float — umap min_dist parameter
    spread      : float — umap spread parameter

    Returns
    -------
    embedding : float32 (N, n_components)
    """
    N = indices.shape[0]
    print(f"[ref] N={N}, k={indices.shape[1]}, n_components={n_components}, seed={seed}",
          file=sys.stderr)

    # Dummy input array — only used for shape; actual data comes from precomputed_knn.
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
    embedding = reducer.fit_transform(X_dummy)
    return embedding.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="umap-learn reference driver for singlet-gpu UMAP correctness test")
    parser.add_argument("--indices",      required=True,
                        help="Path to int32 kNN indices .npy, shape (N, k)")
    parser.add_argument("--distances",    required=True,
                        help="Path to float32 kNN distances .npy, shape (N, k)")
    parser.add_argument("--output",       required=True,
                        help="Path to write float32 embedding .npy, shape (N, n_components)")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-neighbors",  type=int, default=15,
                        help="Must match k in the kNN arrays")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--min-dist",     type=float, default=0.5)
    parser.add_argument("--spread",       type=float, default=1.0)
    args = parser.parse_args()

    indices   = np.load(args.indices).astype(np.int32)
    distances = np.load(args.distances).astype(np.float32)

    print(f"[ref] Loaded indices: {indices.shape}, distances: {distances.shape}",
          file=sys.stderr)

    embedding = run_umap_reference(
        indices=indices,
        distances=distances,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
        min_dist=args.min_dist,
        spread=args.spread,
    )

    np.save(args.output, embedding)
    print(f"[ref] Saved embedding: shape={embedding.shape}, dtype={embedding.dtype} "
          f"-> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
