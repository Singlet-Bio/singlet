#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/leiden_scanpy_reference.py
#
# Reference driver: run scanpy sc.tl.leiden on a kNN graph described by
# pre-computed indices + distances, and write per-cell cluster labels to .npy.
#
# Environment: pip install scanpy leidenalg numpy scipy anndata
#
# Usage:
#   python leiden_scanpy_reference.py \
#       --indices    <knn_indices.npy>    \   # int32 array, shape (N, k)
#       --distances  <knn_distances.npy>  \   # float32 array, shape (N, k)
#       --output     <labels.npy>         \   # int32 array, shape (N,)
#       --resolution <float>              \   # default: 1.0
#       --seed       <int>                \   # default: 42
#
# The C++ test calls this as a subprocess, then reads labels.npy.
#
# Implementation notes:
#   - We build a synthetic AnnData (N cells, 1 dummy variable) and populate
#     adata.uns['neighbors'] / adata.obsp['distances'] / adata.obsp['connectivities']
#     from the pre-computed kNN result so sc.tl.leiden sees it without running
#     sc.pp.neighbors.
#   - Connectivities are built as Gaussian weights matching singlet-gpu's default
#     Connectivity weight function: w = 1 / (1 + d^2).
#   - Labels are returned as contiguous integer codes (0..n_clusters-1) — the
#     same semantics as singlet-gpu's output labels[i].

import argparse
import sys
import numpy as np
import scipy.sparse as sp


def build_anndata_with_knn(indices: np.ndarray, distances: np.ndarray):
    """
    Build a minimal AnnData with pre-computed kNN results loaded into the
    expected scanpy neighbor slots so sc.tl.leiden can proceed without
    re-running sc.pp.neighbors.

    Parameters
    ----------
    indices   : int32 (N, k)  — kNN neighbor indices
    distances : float32 (N, k) — corresponding distances

    Returns
    -------
    adata : anndata.AnnData with populated obsp['connectivities'],
            obsp['distances'], and uns['neighbors'].
    """
    import anndata
    N, k = indices.shape

    # Build a dummy obs × var AnnData (1 variable, all-zero counts).
    X_dummy = sp.csr_matrix((N, 1), dtype=np.float32)
    adata = anndata.AnnData(X=X_dummy)

    # ------------------------------------------------------------------ #
    # Sparse distance matrix (CSR, shape N×N)                             #
    # ------------------------------------------------------------------ #
    rows = np.repeat(np.arange(N, dtype=np.int32), k)
    cols = indices.ravel().astype(np.int32)
    dist_data = distances.ravel().astype(np.float32)

    # Remove any padding entries (indices == -1 or distances == inf).
    valid = (cols >= 0) & np.isfinite(dist_data)
    rows, cols, dist_data = rows[valid], cols[valid], dist_data[valid]

    dist_csr = sp.csr_matrix((dist_data, (rows, cols)), shape=(N, N), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Sparse connectivities: w = 1 / (1 + d^2)                           #
    # (Connectivity weight matching singlet-gpu's default LeidenWeight)   #
    # ------------------------------------------------------------------ #
    conn_data = 1.0 / (1.0 + dist_data ** 2)
    conn_csr = sp.csr_matrix((conn_data, (rows, cols)), shape=(N, N), dtype=np.float32)

    adata.obsp["distances"]      = dist_csr
    adata.obsp["connectivities"] = conn_csr

    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key":      "distances",
        "params": {
            "n_neighbors": k,
            "method":      "umap",
            "metric":      "euclidean",
            "use_rep":     "X",
        },
    }
    return adata


def run_leiden_reference(indices: np.ndarray, distances: np.ndarray,
                         resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Run scanpy sc.tl.leiden on a pre-computed kNN graph.

    Returns
    -------
    labels : int32 array of shape (N,) with contiguous cluster ids 0..n_clusters-1.
    """
    import scanpy as sc

    adata = build_anndata_with_knn(indices, distances)

    sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                 key_added="leiden", directed=False)

    # adata.obs['leiden'] is a Categorical; .cat.codes gives integer codes 0..
    labels = adata.obs["leiden"].cat.codes.values.astype(np.int32)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="scanpy sc.tl.leiden reference for singlet-gpu correctness test")
    parser.add_argument("--indices",   required=True,
                        help="int32 kNN indices .npy, shape (N, k)")
    parser.add_argument("--distances", required=True,
                        help="float32 kNN distances .npy, shape (N, k)")
    parser.add_argument("--output",    required=True,
                        help="Output int32 labels .npy, shape (N,)")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    indices   = np.load(args.indices).astype(np.int32)
    distances = np.load(args.distances).astype(np.float32)
    N, k = indices.shape
    print(f"[ref] Loaded kNN: N={N}, k={k}", file=sys.stderr)

    labels = run_leiden_reference(indices, distances,
                                  resolution=args.resolution, seed=args.seed)
    n_clusters = int(labels.max()) + 1
    print(f"[ref] Leiden (res={args.resolution}, seed={args.seed}): "
          f"{n_clusters} clusters", file=sys.stderr)

    np.save(args.output, labels)
    print(f"[ref] Saved labels to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
