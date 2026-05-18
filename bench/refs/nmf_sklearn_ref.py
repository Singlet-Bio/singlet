#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/nmf_sklearn_ref.py
#
# Cycle 86 Phase E — sklearn NMF CPU reference for ALL four ranks k∈{10,20,50,100}.
# Two scales: small (synthetic 500×200) + medium (GSM4037629 gene_counts.1pz).
#
# Outputs:
#   SKLEARN_WALL_MS=<scale>_k<k>=<ms>
# (one line per (scale, k) combination)
#
# Usage: python3 nmf_sklearn_ref.py
# Options:
#   --scale {small|medium|all}  (default: all)
#   --timeout-s N               (default: 600 per run; sklearn on medium/k=100 is slow)
#
# Notes:
# - NMF operates on raw non-negative counts; do NOT log-normalize.
# - max_iter=100 matches the C++ bench driver (cfg.max_iter=100).
# - init='nndsvd', solver='mu' (matches GPU MU solver for fair comparison).
# - seed=42 for reproducibility.
# - Memory reported as peak RSS (tracemalloc for Python heap + /proc/self/status
#   for C-heap numpy arrays).
# - Warmup: 1 run. Timed: 3 runs. Median of timed runs is reported.

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

CANONICAL_SAMPLE_DIR = Path(
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna"
    "/GSE127/GSE127918/GSM4037629"
)

RANKS = [10, 20, 50, 100]
WARMUP = 1
TIMED  = 3


def proc_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def run_sklearn_nmf_timed(X, k: int):
    """Runs NMF on X (cells×genes, float32) with max_iter=100, returns wall_ms."""
    from sklearn.decomposition import NMF

    tracemalloc.start()
    t0 = time.perf_counter()
    model = NMF(n_components=k, init="nndsvd", solver="mu",
                max_iter=100, random_state=42)
    W = model.fit_transform(X)
    H = model.components_
    wall_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_mb = proc_rss_mb()
    peak_mb = max(peak_heap / (1024**2), rss_mb)
    _ = W, H  # suppress unused warning
    return wall_ms, peak_mb


def load_small():
    """Synthetic 500 cells × 200 genes, density=0.10, seed=1234567."""
    import numpy as np
    rng = np.random.default_rng(1234567)
    X = rng.binomial(20, 0.10, size=(500, 200)).astype("float32")
    X = np.clip(X, 0, None)
    return X


def load_medium():
    """Load GSM4037629 gene_counts.1pz via scanpy (converts to AnnData)."""
    # We load the 10x-style MTX from the star_Solo.out directory or read
    # the .1pz via a custom loader if available.
    # Fallback: read the TSV gene expression summary or use the star_Solo.out.
    # Best path: read the h5ad from a cached conversion.
    import anndata
    import scipy.sparse as sp
    import numpy as np

    # Try the star_Solo.out/GeneFull/filtered path (standard singlet output).
    mtx_dir = CANONICAL_SAMPLE_DIR / "star_Solo.out" / "GeneFull" / "filtered"
    if not mtx_dir.exists():
        # Try star_Solo.out/Gene/filtered.
        mtx_dir = CANONICAL_SAMPLE_DIR / "star_Solo.out" / "Gene" / "filtered"

    if mtx_dir.exists():
        import scanpy as sc
        adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_ids", cache=False)
    else:
        # Fallback: look for a prebuilt h5ad cache.
        h5ad_cache = Path("/dev/shm/gsm4037629_gene_counts.h5ad")
        if h5ad_cache.exists():
            adata = anndata.read_h5ad(str(h5ad_cache))
        else:
            print(f"[nmf_sklearn_ref] ERROR: could not find medium-scale data "
                  f"at {mtx_dir} or {h5ad_cache}", file=sys.stderr)
            return None

    # Ensure raw count matrix, cells × genes.
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype("float32")
    print(f"[nmf_sklearn_ref] medium X shape: {X.shape}", file=sys.stderr)
    return X


def bench_scale(X, scale_name: str):
    if X is None:
        for k in RANKS:
            print(f"SKLEARN_WALL_MS={scale_name}_k{k}=-1.0")
        return

    print(f"[nmf_sklearn_ref] {scale_name}: X={X.shape}  dtype={X.dtype}",
          file=sys.stderr)

    for k in RANKS:
        # Warmup.
        for _ in range(WARMUP):
            try:
                run_sklearn_nmf_timed(X, k)
            except Exception as e:
                print(f"[nmf_sklearn_ref] warmup k={k} FAILED: {e}", file=sys.stderr)

        times = []
        for i in range(TIMED):
            try:
                wall_ms, mem_mb = run_sklearn_nmf_timed(X, k)
                times.append(wall_ms)
                print(f"[nmf_sklearn_ref] {scale_name} k={k} run{i+1}: "
                      f"{wall_ms:.0f}ms  {mem_mb:.0f}MB", file=sys.stderr)
            except Exception as e:
                print(f"[nmf_sklearn_ref] {scale_name} k={k} run{i+1} FAILED: {e}",
                      file=sys.stderr)

        if not times:
            print(f"SKLEARN_WALL_MS={scale_name}_k{k}=-1.0", flush=True)
        else:
            times.sort()
            n = len(times)
            median = times[n // 2] if n % 2 == 1 else 0.5 * (times[n//2-1] + times[n//2])
            print(f"SKLEARN_WALL_MS={scale_name}_k{k}={median:.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["small", "medium", "all"], default="all")
    args = p.parse_args()

    if args.scale in ("small", "all"):
        bench_scale(load_small(), "small")

    if args.scale in ("medium", "all"):
        bench_scale(load_medium(), "medium")


if __name__ == "__main__":
    main()
