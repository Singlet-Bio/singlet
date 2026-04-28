#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/bbknn_python_reference.py
#
# Reference driver: run bbknn.bbknn on a dense fp32 embedding (wrapped in an
# AnnData object) and write the resulting neighbor graph to .npz files.
#
# Environment: pip install bbknn anndata scanpy numpy scipy
#
# Usage:
#   python bbknn_python_reference.py \
#       --embedding    <embedding.npy>       \   # float32 (N, d)
#       --batch-labels <batch_labels.npy>    \   # int32   (N,) in [0, B)
#       --out-neighbors <neighbors.npz>      \   # keys: indices (N, k), distances (N, k)
#       [--neighbors-within-batch <int>]     \   # default: 3
#       [--n-pcs <int>]                      \   # default: 50 (ignored if embedding given)
#
# Output .npz keys:
#   indices   : int32 (N, k_total)   — sorted by distance per row
#   distances : float32 (N, k_total) — Euclidean distances
#
# where k_total = neighbors_within_batch * n_batches.
#
# The C++ test calls this as a subprocess then reads neighbors.npz.

import argparse
import sys
import numpy as np


def run_bbknn_reference(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    neighbors_within_batch: int = 3,
):
    """
    Run bbknn on the embedding and return (indices, distances) arrays.

    indices   : int32 (N, k_total)
    distances : float32 (N, k_total)
    """
    try:
        import anndata as ad
        import bbknn
    except ImportError:
        print("[ref] ERROR: bbknn or anndata not installed. "
              "pip install bbknn anndata", file=sys.stderr)
        raise

    N, d = embedding.shape
    n_batches = int(batch_labels.max()) + 1
    k_total = neighbors_within_batch * n_batches

    print(f"[ref] embedding: {N} x {d}  n_batches={n_batches}  "
          f"neighbors_within_batch={neighbors_within_batch}  k_total={k_total}",
          file=sys.stderr)

    adata = ad.AnnData(X=np.zeros((N, 1), dtype=np.float32))
    adata.obsm["X_pca"] = embedding.astype(np.float32)
    adata.obs["batch"]  = batch_labels.astype(str)

    bbknn.bbknn(
        adata,
        batch_key="batch",
        neighbors_within_batch=neighbors_within_batch,
        use_rep="X_pca",
        n_pcs=d,
    )

    # adata.obsp["distances"] is a scipy CSR (N, N) sparse matrix.
    dist_csr = adata.obsp["distances"]

    # Convert to dense (N, k_total) arrays sorted by distance.
    indices   = np.full((N, k_total), -1, dtype=np.int32)
    distances = np.full((N, k_total), np.inf, dtype=np.float32)

    for i in range(N):
        row_start = dist_csr.indptr[i]
        row_end   = dist_csr.indptr[i + 1]
        row_inds  = dist_csr.indices[row_start:row_end].astype(np.int32)
        row_dists = np.array(dist_csr.data[row_start:row_end], dtype=np.float32)
        # Sort by distance ascending.
        order     = np.argsort(row_dists)
        row_inds  = row_inds[order]
        row_dists = row_dists[order]
        k_row     = min(len(row_inds), k_total)
        indices[i, :k_row]   = row_inds[:k_row]
        distances[i, :k_row] = row_dists[:k_row]

    print(f"[ref] output neighbors: {indices.shape}", file=sys.stderr)
    return indices, distances


def main():
    parser = argparse.ArgumentParser(
        description="bbknn Python reference driver for singlet-gpu correctness test")
    parser.add_argument("--embedding",    required=True,
                        help="float32 embedding .npy, shape (N, d)")
    parser.add_argument("--batch-labels", required=True,
                        help="int32 batch labels .npy, shape (N,)")
    parser.add_argument("--out-neighbors", required=True,
                        help="Output .npz: keys indices (N,k), distances (N,k)")
    parser.add_argument("--neighbors-within-batch", type=int, default=3)
    args = parser.parse_args()

    embedding    = np.load(args.embedding).astype(np.float32)
    batch_labels = np.load(args.batch_labels).astype(np.int32)

    indices, distances = run_bbknn_reference(
        embedding, batch_labels,
        neighbors_within_batch=args.neighbors_within_batch,
    )

    np.savez(args.out_neighbors, indices=indices, distances=distances)
    print(f"[ref] Saved neighbor graph to {args.out_neighbors}", file=sys.stderr)


if __name__ == "__main__":
    main()
