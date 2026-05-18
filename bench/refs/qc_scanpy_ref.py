#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/qc_scanpy_ref.py
#
# Cycle 86 add-on — scanpy QC metrics CPU reference.
#
# Benchmarks scanpy.pp.calculate_qc_metrics() on:
#   small : synthetic 1,000 cells × 2,000 genes
#   medium: GSM4037629 gene_counts.1pz (~11,560 cells × ~38,606 genes)
#
# Outputs:
#   SCANPY_QC_WALL_MS=<scale>=<ms>
#   SCANPY_QC_MEM_MB=<scale>=<mb>
#   SCANPY_QC_CELLS=<scale>=<int>
#
# Usage: python3 qc_scanpy_ref.py [--scale {small|medium|all}]
#
# Notes:
# - mt gene detection: genes starting with "MT-" or index < 5% of n_genes
#   (deterministic gene flag matching C++ bench driver).
# - Warmup: 2 runs. Timed: 5 runs. Median reported.
# - Memory: tracemalloc peak + /proc RSS (for numpy C-heap).

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

CANONICAL_SAMPLE_DIR = Path(
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna"
    "/GSE127/GSE127918/GSM4037629"
)

WARMUP = 2
TIMED  = 5


def proc_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def run_scanpy_qc(adata, qc_vars):
    """Run scanpy.pp.calculate_qc_metrics, return (wall_ms, peak_mb)."""
    import scanpy as sc
    tracemalloc.start()
    t0 = time.perf_counter()
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_mb = proc_rss_mb()
    return wall_ms, max(peak_heap / (1024**2), rss_mb)


def make_small_adata():
    import numpy as np
    import anndata
    import scipy.sparse as sp

    n_cells = 1000
    n_genes = 2000
    rng = np.random.default_rng(9876543)
    X = rng.binomial(50, 0.15, size=(n_cells, n_genes)).astype("float32")
    X = np.clip(X, 0, None)

    # Gene names: first 5% are "MT-..." (matches C++ bench driver flag assignment).
    n_mt = max(1, n_genes // 20)
    gene_names = [f"MT-GENE{i}" for i in range(n_mt)] + \
                 [f"GENE{i}"   for i in range(n_genes - n_mt)]

    adata = anndata.AnnData(
        X=sp.csr_matrix(X),
        var={"gene_names": gene_names},
    )
    adata.var_names = gene_names
    adata.var["mt"] = [n.startswith("MT-") for n in gene_names]
    return adata


def load_medium_adata():
    """Load GSM4037629 gene_counts as AnnData."""
    import anndata
    import scipy.sparse as sp
    import scanpy as sc

    mtx_dir = CANONICAL_SAMPLE_DIR / "star_Solo.out" / "GeneFull" / "filtered"
    if not mtx_dir.exists():
        mtx_dir = CANONICAL_SAMPLE_DIR / "star_Solo.out" / "Gene" / "filtered"

    if mtx_dir.exists():
        adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_ids", cache=False)
    else:
        h5ad_cache = Path("/dev/shm/gsm4037629_gene_counts.h5ad")
        if h5ad_cache.exists():
            adata = anndata.read_h5ad(str(h5ad_cache))
        else:
            print(f"[qc_scanpy_ref] ERROR: could not find medium data at "
                  f"{mtx_dir} or {h5ad_cache}", file=sys.stderr)
            return None

    # Mark mt genes: index < 5% of n_genes (matches C++ bench driver).
    n_mt = max(1, adata.n_vars // 20)
    adata.var["mt"] = [i < n_mt for i in range(adata.n_vars)]
    print(f"[qc_scanpy_ref] medium adata: {adata.n_obs} cells × {adata.n_vars} genes",
          file=sys.stderr)
    return adata


def bench_scale(adata, scale_name: str):
    if adata is None:
        print(f"SCANPY_QC_WALL_MS={scale_name}=-1.0", flush=True)
        print(f"SCANPY_QC_MEM_MB={scale_name}=-1.0",  flush=True)
        print(f"SCANPY_QC_CELLS={scale_name}=-1",     flush=True)
        return

    qc_vars = ["mt"]

    print(f"[qc_scanpy_ref] {scale_name}: {adata.n_obs} cells × {adata.n_vars} genes",
          file=sys.stderr)

    # Warmup.
    for _ in range(WARMUP):
        import copy
        run_scanpy_qc(copy.copy(adata), qc_vars)

    times, mems = [], []
    for i in range(TIMED):
        import copy
        a = copy.copy(adata)
        wall_ms, mem_mb = run_scanpy_qc(a, qc_vars)
        times.append(wall_ms)
        mems.append(mem_mb)
        print(f"[qc_scanpy_ref] {scale_name} run{i+1}: "
              f"{wall_ms:.1f}ms  {mem_mb:.0f}MB", file=sys.stderr)

    times.sort()
    n = len(times)
    median_wall = times[n//2] if n % 2 == 1 else 0.5*(times[n//2-1]+times[n//2])
    median_mem  = sorted(mems)[n//2] if n % 2 == 1 else 0.5*(sorted(mems)[n//2-1]+sorted(mems)[n//2])

    print(f"SCANPY_QC_WALL_MS={scale_name}={median_wall:.2f}", flush=True)
    print(f"SCANPY_QC_MEM_MB={scale_name}={median_mem:.1f}",   flush=True)
    print(f"SCANPY_QC_CELLS={scale_name}={adata.n_obs}",       flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["small", "medium", "all"], default="all")
    args = p.parse_args()

    if args.scale in ("small", "all"):
        bench_scale(make_small_adata(), "small")

    if args.scale in ("medium", "all"):
        bench_scale(load_medium_adata(), "medium")


if __name__ == "__main__":
    main()
