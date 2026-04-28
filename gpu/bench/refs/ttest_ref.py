#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/ttest_ref.py
#
# Performance reference: scanpy rank_genes_groups(method='t-test') vs singlet-gpu.
# Mirror of wilcoxon_ref.py — same CSC binary format, same JSON output schema.
#
# Usage:
#   python ttest_ref.py \
#       --input       <matrix.bin>       \
#       --labels      <labels.npy>       \
#       --top-n       <int>              \
#       --timing-json <result.json>
#
# JSON output: { "wall_ms": float, "mem_mb": float, "impl": str, "n": int,
#                "n_genes": int, "n_clusters": int }

import argparse, json, struct, sys, time
import numpy as np
import scipy.sparse as sp


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def load_csc_bin(path):
    with open(path, "rb") as f:
        magic, n_rows, n_cols = struct.unpack("<III", f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic: {magic:#010x}")
        nnz = struct.unpack("<Q", f.read(8))[0]
        values  = np.frombuffer(f.read(nnz * 4),            dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4),   dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),            dtype=np.int32).copy()
    return sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))


def run_scanpy_ttest(mat, labels, top_n):
    import anndata, scanpy as sc
    n_genes, n_cells = mat.shape
    X_csr = mat.T.tocsr().astype(np.float32)
    adata = anndata.AnnData(X=X_csr)
    adata.obs["cluster"] = [str(c) for c in labels]
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_genes_req = min(top_n, n_genes)
    sc.tl.rank_genes_groups(adata, groupby="cluster", method="t-test",
                             n_genes=n_genes_req, use_raw=False, key_added="de")
    return adata.uns["de"]


def main():
    parser = argparse.ArgumentParser(description="scanpy t-test reference for singlet-gpu bench")
    parser.add_argument("--input",       required=True)
    parser.add_argument("--labels",      required=True)
    parser.add_argument("--top-n",       type=int, default=100)
    parser.add_argument("--timing-json", required=True)
    args = parser.parse_args()

    mat    = load_csc_bin(args.input)
    labels = np.load(args.labels).astype(np.int32)
    n_genes, n_cells = mat.shape
    n_clusters = int(labels.max()) + 1
    print(f"[ttest_ref] {n_genes} genes × {n_cells} cells, {n_clusters} clusters",
          file=sys.stderr)

    # Warmup
    _ = run_scanpy_ttest(mat, labels, args.top_n)

    wall_times, mem_mb_last = [], 0.0
    for _ in range(5):
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        run_scanpy_ttest(mat, labels, args.top_n)
        wall_times.append((time.perf_counter() - t0) * 1000.0)
        mem_mb_last = max(0.0, _rss_mb() - rss_before)

    wall_median_ms = float(sorted(wall_times)[len(wall_times) // 2])
    result = {"wall_ms": wall_median_ms, "mem_mb": float(mem_mb_last),
              "impl": "scanpy-ttest", "n": n_cells, "n_genes": n_genes,
              "n_clusters": n_clusters}
    with open(args.timing_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ttest_ref] scanpy-ttest: n={n_cells} wall_ms={wall_median_ms:.1f} "
          f"mem_mb={mem_mb_last:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
