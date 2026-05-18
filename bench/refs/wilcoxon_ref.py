#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/wilcoxon_ref.py
#
# Performance reference: scanpy rank_genes_groups(method='wilcoxon') vs singlet-gpu.
#
# Accepts a CSC binary matrix (dump_csc.h format) + cluster labels (.npy),
# runs sc.tl.rank_genes_groups, reports wall time in a timing JSON.
#
# Environment: pip install scanpy anndata numpy scipy
#
# Usage:
#   python wilcoxon_ref.py \
#       --input       <matrix.bin>       \   # CSC binary (dump_csc.h format)
#       --labels      <labels.npy>       \   # int32 (n_cells,)
#       --top-n       <int>              \   # default: 100
#       --timing-json <result.json>          # required
#
# CSC binary format (matches tests/refs/de_scanpy_reference.py):
#   Magic   : uint32  0x43535343 ("CSCC")
#   n_rows  : uint32
#   n_cols  : uint32
#   nnz     : uint64
#   values  : float32[nnz]
#   indptr  : int32[n_cols+1]
#   indices : int32[nnz]
#
# JSON output schema:
#   { "wall_ms": float, "mem_mb": float, "impl": str, "n": int, "n_genes": int,
#     "n_clusters": int }

import argparse
import json
import struct
import sys
import time

import numpy as np
import scipy.sparse as sp


def _memory_delta_mb(fn, *args, **kwargs):
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


def load_csc_bin(path: str) -> sp.csc_matrix:
    with open(path, "rb") as f:
        magic, n_rows, n_cols = struct.unpack("<III", f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic: {magic:#010x}")
        nnz = struct.unpack("<Q", f.read(8))[0]
        values  = np.frombuffer(f.read(nnz * 4),            dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4),   dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),            dtype=np.int32).copy()
    return sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))


def run_scanpy_wilcoxon(mat: sp.csc_matrix, labels: np.ndarray, top_n: int):
    import anndata
    import scanpy as sc

    n_genes, n_cells = mat.shape
    # scanpy wants cells × genes (CSR).
    X_csr = mat.T.tocsr().astype(np.float32)
    adata = anndata.AnnData(X=X_csr)
    adata.obs["cluster"] = [str(c) for c in labels]
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_genes_req = min(top_n, n_genes)
    sc.tl.rank_genes_groups(
        adata, groupby="cluster", method="wilcoxon",
        n_genes=n_genes_req, use_raw=False, key_added="de")
    return adata.uns["de"]


def main():
    parser = argparse.ArgumentParser(
        description="scanpy Wilcoxon rank_genes_groups reference for singlet-gpu DE bench")
    parser.add_argument("--input",       required=True,
                        help="CSC binary (dump_csc.h format)")
    parser.add_argument("--labels",      required=True,
                        help="int32 .npy cluster labels, shape (n_cells,)")
    parser.add_argument("--top-n",       type=int, default=100)
    parser.add_argument("--timing-json", required=True,
                        help="Path to write timing JSON")
    args = parser.parse_args()

    mat    = load_csc_bin(args.input)
    labels = np.load(args.labels).astype(np.int32)
    n_genes, n_cells = mat.shape
    n_clusters = int(labels.max()) + 1
    print(f"[wilcoxon_ref] {n_genes} genes × {n_cells} cells, "
          f"{n_clusters} clusters", file=sys.stderr)

    # Warmup.
    _ = run_scanpy_wilcoxon(mat, labels, args.top_n)

    wall_times = []
    mem_mb_last = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        res, mem_mb = _memory_delta_mb(run_scanpy_wilcoxon, mat, labels, args.top_n)
        wall_times.append((time.perf_counter() - t0) * 1000.0)
        mem_mb_last = mem_mb

    wall_median_ms = float(sorted(wall_times)[len(wall_times) // 2])

    result = {
        "wall_ms":    wall_median_ms,
        "mem_mb":     float(mem_mb_last),
        "impl":       "scanpy-wilcoxon",
        "n":          n_cells,
        "n_genes":    n_genes,
        "n_clusters": n_clusters,
    }
    with open(args.timing_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[wilcoxon_ref] scanpy-wilcoxon: n={n_cells} wall_ms={wall_median_ms:.1f} "
          f"mem_mb={mem_mb_last:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
