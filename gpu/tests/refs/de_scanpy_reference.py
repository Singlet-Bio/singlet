#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/de_scanpy_reference.py
#
# Reference driver: run scanpy sc.tl.rank_genes_groups on a CSC count matrix
# with per-cell cluster labels, then write per-cluster DE results to a .npz.
#
# Environment: pip install scanpy numpy scipy anndata
#
# Usage:
#   python de_scanpy_reference.py \
#       --input    <matrix.bin>          # CSC binary (dump_csc.h format)
#       --labels   <labels.npy>          # int32 array, shape (n_cells,)
#       --output   <de_results.npz>      # per-cluster results
#       --method   wilcoxon | t-test     # default: wilcoxon
#       --top-n    <int>                 # default: 100
#
# Output .npz keys (all per-cluster, zero-indexed clusters 0..K-1):
#   cluster_ids       : int32 (K,)       — cluster ids present in labels
#   gene_indices_{c}  : int32 (top_n,)   — top_n gene indices, rank-ordered
#   scores_{c}        : float32 (top_n,) — z-scores / t-stats per top_n gene
#   pvals_{c}         : float32 (top_n,) — p-values per top_n gene
#   pvals_adj_{c}     : float32 (top_n,) — BH-adjusted p-values
#   logfoldchanges_{c}: float32 (top_n,) — log2 fold-changes
#
# The C++ tests call this as a subprocess for both Wilcoxon and t-test harnesses.
# The shared script avoids duplication and keeps reference behaviour identical.

import argparse
import sys
import struct
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Binary CSC reader (dump_csc.h format)
# Magic = 0x43535343 ("CSCC"), then n_rows, n_cols (uint32), nnz (uint64),
# float32 values[nnz], int32 indptr[n_cols+1], int32 indices[nnz].
# ---------------------------------------------------------------------------
def load_csc_bin(path: str) -> sp.csc_matrix:
    with open(path, "rb") as f:
        magic, n_rows, n_cols = struct.unpack("<III", f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic: {magic:#010x}, expected 0x43535343")
        nnz = struct.unpack("<Q", f.read(8))[0]
        values  = np.frombuffer(f.read(nnz * 4),            dtype=np.float32)
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4),   dtype=np.int32)
        indices = np.frombuffer(f.read(nnz * 4),            dtype=np.int32)
    mat = sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))
    print(f"[ref] Loaded CSC: {n_rows} genes × {n_cols} cells, nnz={nnz}",
          file=sys.stderr)
    return mat


# ---------------------------------------------------------------------------
# Run scanpy DE reference
# ---------------------------------------------------------------------------
def run_de_reference(mat: sp.csc_matrix,
                     labels: np.ndarray,
                     method: str,
                     top_n: int) -> dict:
    """
    Parameters
    ----------
    mat    : (genes × cells) CSC float32 sparse matrix — raw counts.
    labels : int32 (n_cells,) cluster labels 0..K-1.
    method : 'wilcoxon' or 't-test'.
    top_n  : number of top markers to record per cluster.

    Returns
    -------
    dict mapping string keys → numpy arrays, ready for np.savez.
    """
    import anndata
    import scanpy as sc

    n_genes, n_cells = mat.shape
    assert len(labels) == n_cells, \
        f"labels length {len(labels)} != n_cells {n_cells}"

    cluster_ids = np.unique(labels).astype(np.int32)
    n_clusters  = len(cluster_ids)
    print(f"[ref] method={method}, n_genes={n_genes}, n_cells={n_cells}, "
          f"n_clusters={n_clusters}, top_n={top_n}", file=sys.stderr)

    # Build AnnData.  scanpy wants cells × genes (CSR), so transpose.
    X_csr = mat.T.tocsr().astype(np.float32)
    adata = anndata.AnnData(X=X_csr)
    adata.obs["cluster"] = [str(c) for c in labels]
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    # Log-normalise to match the singlet-gpu test (which log-normalises before DE).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run DE (one-vs-rest).
    # Use_raw=False because we already preprocessed X in-place.
    n_genes_req = min(top_n, n_genes)
    sc.tl.rank_genes_groups(
        adata,
        groupby="cluster",
        method=method,
        n_genes=n_genes_req,
        use_raw=False,
        key_added="de_result",
    )

    rgg = adata.uns["de_result"]

    # Extract per-cluster arrays.
    out: dict = {"cluster_ids": cluster_ids}

    for c in cluster_ids:
        key = str(c)
        try:
            # .names is a recarray with dtype [('cluster0',…), ('cluster1',…), …]
            gene_names  = rgg["names"][key]        # top_n gene names (strings)
            scores      = rgg["scores"][key].astype(np.float32)
            pvals       = rgg["pvals"][key].astype(np.float32)
            pvals_adj   = rgg["pvals_adj"][key].astype(np.float32)
            lfcs        = rgg["logfoldchanges"][key].astype(np.float32)
        except Exception as e:
            print(f"[ref] WARNING cluster {c}: {e}", file=sys.stderr)
            gene_names  = np.array([], dtype=str)
            scores      = np.array([], dtype=np.float32)
            pvals       = np.array([], dtype=np.float32)
            pvals_adj   = np.array([], dtype=np.float32)
            lfcs        = np.array([], dtype=np.float32)

        # Convert string gene names back to integer indices (0-based).
        # scanpy stores names as strings from adata.var_names (which we leave
        # as default integer strings "0", "1", …).
        try:
            gene_indices = np.array([int(g) for g in gene_names], dtype=np.int32)
        except ValueError:
            # var_names were not plain integers — fall back to positional lookup.
            name_to_idx = {nm: idx for idx, nm in enumerate(adata.var_names)}
            gene_indices = np.array([name_to_idx.get(g, -1) for g in gene_names],
                                    dtype=np.int32)

        actual_n = len(gene_indices)
        # Pad to top_n if shorter (edge-case: fewer genes than top_n).
        if actual_n < top_n:
            pad = top_n - actual_n
            gene_indices = np.concatenate([gene_indices, np.full(pad, -1, np.int32)])
            scores       = np.concatenate([scores,     np.zeros(pad, np.float32)])
            pvals        = np.concatenate([pvals,      np.ones (pad, np.float32)])
            pvals_adj    = np.concatenate([pvals_adj,  np.ones (pad, np.float32)])
            lfcs         = np.concatenate([lfcs,       np.zeros(pad, np.float32)])

        out[f"gene_indices_{c}"]   = gene_indices[:top_n]
        out[f"scores_{c}"]         = scores[:top_n]
        out[f"pvals_{c}"]          = pvals[:top_n]
        out[f"pvals_adj_{c}"]      = pvals_adj[:top_n]
        out[f"logfoldchanges_{c}"] = lfcs[:top_n]

    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="scanpy rank_genes_groups reference for singlet-gpu DE tests")
    parser.add_argument("--input",   required=True,
                        help="CSC binary file (dump_csc.h format)")
    parser.add_argument("--labels",  required=True,
                        help="int32 .npy cluster labels, shape (n_cells,)")
    parser.add_argument("--output",  required=True,
                        help="Output .npz path")
    parser.add_argument("--method",  default="wilcoxon",
                        choices=["wilcoxon", "t-test"],
                        help="DE method (default: wilcoxon)")
    parser.add_argument("--top-n",   type=int, default=100,
                        help="Number of top marker genes per cluster (default: 100)")
    args = parser.parse_args()

    mat    = load_csc_bin(args.input)
    labels = np.load(args.labels).astype(np.int32)

    result = run_de_reference(mat, labels, args.method, args.top_n)

    np.savez(args.output, **result)
    print(f"[ref] Saved DE results to {args.output}", file=sys.stderr)
    cluster_ids = result["cluster_ids"]
    for c in cluster_ids:
        n_valid = int(np.sum(result[f"gene_indices_{c}"] >= 0))
        print(f"[ref]   cluster {c}: {n_valid} valid markers", file=sys.stderr)


if __name__ == "__main__":
    main()
