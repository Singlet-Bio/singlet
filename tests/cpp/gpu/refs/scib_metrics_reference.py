#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/scib_metrics_reference.py
#
# Reference driver: compute scIB batch-integration metrics (iLISI, cLISI) on
# a dense fp32 embedding with batch labels and optional cell-type labels.
#
# Environment: pip install scib-metrics scanpy anndata numpy
#
# Usage:
#   python scib_metrics_reference.py \
#       --embedding      <embedding.npy>       \   # float32 (N, d)
#       --batch-labels   <batch_labels.npy>    \   # int32   (N,) in [0, B)
#       [--cell-type-labels <ct_labels.npy>]   \   # int32   (N,) in [0, C); required for cLISI
#       --out-metrics    <metrics.json>        \   # {"ilisi": float, "clisi": float}
#       [--n-neighbors <int>]                  \   # default: 15 (for kNN graph)
#
# Output JSON keys:
#   ilisi  : float  — mean iLISI (integration LISI), range [1, B]; closer to B is better
#   clisi  : float  — mean cLISI (cell-type LISI), range [1, C]; closer to 1 is better
#                     (present only when --cell-type-labels is given)
#
# iLISI close to n_batches  = good batch mixing.
# cLISI close to 1          = good cell-type preservation.
#
# The C++ test calls this as a subprocess before and after correction, then
# checks that iLISI improved by >= 0.1 and that cLISI did NOT degrade by > 5%.

import argparse
import json
import sys
import numpy as np


def build_adata(embedding, batch_labels, cell_type_labels=None):
    """Build an AnnData object ready for scib metrics."""
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        print("[ref] ERROR: anndata or scanpy not installed.", file=sys.stderr)
        raise

    N = embedding.shape[0]
    adata = ad.AnnData(X=np.zeros((N, 1), dtype=np.float32))
    adata.obsm["X_emb"] = embedding.astype(np.float32)
    adata.obs["batch"]  = batch_labels.astype(str)
    if cell_type_labels is not None:
        adata.obs["cell_type"] = cell_type_labels.astype(str)
    return adata


def compute_ilisi(adata, n_neighbors: int = 15) -> float:
    """Compute mean iLISI (integration LISI) using scib-metrics."""
    try:
        import scib_metrics
        from sklearn.neighbors import NearestNeighbors
        import pandas as pd
    except ImportError:
        print("[ref] ERROR: scib-metrics or sklearn not installed. "
              "pip install scib-metrics scikit-learn", file=sys.stderr)
        raise

    embedding = adata.obsm["X_emb"]
    batch_labels = adata.obs["batch"].values

    # Build kNN graph via sklearn for compatibility.
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean",
                          algorithm="brute")
    nn.fit(embedding)
    # distances matrix (N, N) as CSR — scib_metrics.ilisi_knn takes knn connectivity.
    dists, inds = nn.kneighbors(embedding)

    # Convert to sparse connectivity matrix (N, N) for scib_metrics.
    import scipy.sparse as sp
    N = embedding.shape[0]
    rows = np.repeat(np.arange(N), n_neighbors)
    cols = inds.ravel()
    data = np.ones(len(rows), dtype=np.float32)
    knn_conn = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Encode batch labels as integers.
    unique_batches, batch_int = np.unique(batch_labels, return_inverse=True)

    ilisi_scores = scib_metrics.ilisi_knn(knn_conn, batch_int)
    mean_ilisi   = float(np.nanmean(ilisi_scores))
    print(f"[ref] iLISI: mean={mean_ilisi:.4f}  n_batches={len(unique_batches)}",
          file=sys.stderr)
    return mean_ilisi


def compute_clisi(adata, n_neighbors: int = 15) -> float:
    """Compute mean cLISI (cell-type LISI) using scib-metrics."""
    try:
        import scib_metrics
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("[ref] ERROR: scib-metrics or sklearn not installed.", file=sys.stderr)
        raise

    if "cell_type" not in adata.obs:
        raise ValueError("cLISI requires --cell-type-labels")

    embedding       = adata.obsm["X_emb"]
    ct_labels       = adata.obs["cell_type"].values
    unique_cts, ct_int = np.unique(ct_labels, return_inverse=True)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean",
                          algorithm="brute")
    nn.fit(embedding)
    _, inds = nn.kneighbors(embedding)

    import scipy.sparse as sp
    N    = embedding.shape[0]
    rows = np.repeat(np.arange(N), n_neighbors)
    cols = inds.ravel()
    data = np.ones(len(rows), dtype=np.float32)
    knn_conn = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    clisi_scores = scib_metrics.clisi_knn(knn_conn, ct_int)
    mean_clisi   = float(np.nanmean(clisi_scores))
    print(f"[ref] cLISI: mean={mean_clisi:.4f}  n_cell_types={len(unique_cts)}",
          file=sys.stderr)
    return mean_clisi


def main():
    parser = argparse.ArgumentParser(
        description="scib metrics reference driver for singlet-gpu correctness test")
    parser.add_argument("--embedding",        required=True,
                        help="float32 embedding .npy, shape (N, d)")
    parser.add_argument("--batch-labels",     required=True,
                        help="int32 batch labels .npy, shape (N,)")
    parser.add_argument("--cell-type-labels", default=None,
                        help="int32 cell-type labels .npy, shape (N,) — for cLISI")
    parser.add_argument("--out-metrics",      required=True,
                        help="Output JSON: ilisi, (optionally) clisi")
    parser.add_argument("--n-neighbors", type=int, default=15)
    args = parser.parse_args()

    embedding    = np.load(args.embedding).astype(np.float32)
    batch_labels = np.load(args.batch_labels).astype(np.int32)
    ct_labels    = (np.load(args.cell_type_labels).astype(np.int32)
                    if args.cell_type_labels else None)

    N, d = embedding.shape
    n_batches = int(batch_labels.max()) + 1
    print(f"[ref] Loaded embedding: {N} x {d}  n_batches={n_batches}", file=sys.stderr)

    adata = build_adata(embedding, batch_labels, ct_labels)

    metrics: dict = {}
    metrics["ilisi"] = compute_ilisi(adata, n_neighbors=args.n_neighbors)
    if ct_labels is not None:
        metrics["clisi"] = compute_clisi(adata, n_neighbors=args.n_neighbors)

    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ref] Saved metrics to {args.out_metrics}: {metrics}", file=sys.stderr)


if __name__ == "__main__":
    main()
