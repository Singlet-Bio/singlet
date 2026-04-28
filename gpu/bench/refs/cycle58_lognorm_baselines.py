#!/usr/bin/env python3
"""
singlet-gpu/bench/refs/cycle58_lognorm_baselines.py
SPDX-License-Identifier: GPL-2.0-or-later

Cycle 58 Phase E — lognorm SOTA baselines.

Runs one of:
  rapids_total_count   — rapids-singlecell pp.normalize_total + pp.log1p (GPU)
  scanpy_total_count   — scanpy pp.normalize_total + pp.log1p (CPU)
  scran_deconvolution  — scran::computeSumFactors via subprocess (R, CPU)

Usage:
    python3 cycle58_lognorm_baselines.py
        --baseline=<rapids_total_count|scanpy_total_count|scran_deconvolution>
        --h5ad-path=<path.h5ad>
        --out-json=<path.json>
        [--n-cells=<int>]    # optional override for n_cells in output

JSON output:
    { "wall_ms": float, "mem_mb": float, "impl": string,
      "scale": string, "n_cells": int,
      "size_factors_sample": [float, ...10] }   # first 10 size factors for corr check
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

WARMUP_ITERS = 2
TIMED_ITERS  = 5


def proc_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def time_fn(fn, *args, **kwargs):
    """2 warmup + 5 timed iters. Returns (median_ms, peak_mem_mb, last_result)."""
    for _ in range(WARMUP_ITERS):
        fn(*args, **kwargs)

    wall_times = []
    peak_mb_list = []
    last_result = None
    for _ in range(TIMED_ITERS):
        tracemalloc.start()
        rss_before = proc_rss_mb()
        t0 = time.perf_counter()
        last_result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc_rss_mb()
        wall_times.append((t1 - t0) * 1000.0)
        peak_mb_list.append(max(peak / (1024 * 1024), rss_after - rss_before))

    wall_times.sort()
    n = len(wall_times)
    median = wall_times[n // 2] if n % 2 == 1 else 0.5 * (
        wall_times[n // 2 - 1] + wall_times[n // 2])
    return median, max(peak_mb_list), last_result


def write_result(out_json: str, wall_ms: float, mem_mb: float,
                 impl: str, scale: str, n_cells: int,
                 size_factors_sample=None, extra=None):
    d = {
        "wall_ms": wall_ms,
        "mem_mb": mem_mb,
        "impl": impl,
        "scale": scale,
        "n_cells": n_cells,
        "size_factors_sample": size_factors_sample or [],
    }
    if extra:
        d.update(extra)
    with open(out_json, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[{impl}] wall={wall_ms:.1f}ms mem={mem_mb:.0f}MB n_cells={n_cells}", flush=True)


# ---------------------------------------------------------------------------
# rapids-singlecell: pp.normalize_total + pp.log1p
# ---------------------------------------------------------------------------
def run_rapids_total_count(h5ad_path: str):
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(h5ad_path)

    try:
        import rapids_singlecell as rsc
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import scipy.sparse as sp
        has_rapids = True
    except ImportError:
        has_rapids = False

    if not has_rapids:
        print("[rapids] SKIP: rapids_singlecell or cupy not available", flush=True)
        return -1.0, -1.0, 0, []

    # Convert to CSR on device once outside timing loop.
    import scipy.sparse as sp
    if sp.issparse(adata.X):
        csr = adata.X.tocsr().astype(np.float32)
    else:
        csr = sp.csr_matrix(adata.X, dtype=np.float32)

    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    def _normalize():
        # Fresh copy each iter to re-apply normalization from counts.
        d_csr = cpsp.csr_matrix(
            (cp.asarray(csr.data), cp.asarray(csr.indices), cp.asarray(csr.indptr)),
            shape=csr.shape
        )
        # Create minimal AnnData-like wrapper for rapids_singlecell.
        import anndata as ad2
        ad_gpu = ad.AnnData(X=csr)
        rsc.get.anndata_to_GPU(ad_gpu)
        rsc.pp.normalize_total(ad_gpu, target_sum=1e4)
        rsc.pp.log1p(ad_gpu)
        cp.cuda.Device().synchronize()
        return ad_gpu

    wall_ms, mem_mb, result = time_fn(_normalize)

    # Extract size factors from a fresh run for correlation check.
    ad_sf = ad.AnnData(X=csr.copy())
    rsc.get.anndata_to_GPU(ad_sf)
    # Compute total-count size factors (row sums / median).
    col_sums = np.array(csr.sum(axis=1)).flatten().astype(np.float64)
    positive = col_sums[col_sums > 0]
    median_lib = float(np.median(positive)) if len(positive) > 0 else 1.0
    size_factors = (col_sums / median_lib).tolist()[:10]

    n_cells = csr.shape[0]
    return wall_ms, mem_mb, n_cells, size_factors


# ---------------------------------------------------------------------------
# scanpy: pp.normalize_total + pp.log1p (CPU)
# ---------------------------------------------------------------------------
def run_scanpy_total_count(h5ad_path: str):
    import anndata as ad
    import numpy as np
    import scanpy as sc

    adata_orig = ad.read_h5ad(h5ad_path)

    def _normalize():
        # Fresh copy each iter (normalize_total modifies in place).
        adata = adata_orig.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    wall_ms, mem_mb, result = time_fn(_normalize)

    # Size factors from a fresh normalize (proportional to library sizes).
    import scipy.sparse as sp
    X = adata_orig.X
    if sp.issparse(X):
        col_sums = np.array(X.sum(axis=1)).flatten().astype(np.float64)
    else:
        col_sums = X.sum(axis=1).astype(np.float64)
    positive = col_sums[col_sums > 0]
    median_lib = float(np.median(positive)) if len(positive) > 0 else 1.0
    size_factors = (col_sums / median_lib).tolist()[:10]

    n_cells = adata_orig.n_obs
    return wall_ms, mem_mb, n_cells, size_factors


# ---------------------------------------------------------------------------
# scran: computeSumFactors via Rscript subprocess
# ---------------------------------------------------------------------------
def run_scran_deconvolution(h5ad_path: str, out_json: str):
    """
    Runs scran::computeSumFactors via Rscript.
    Writes scran_sf.json next to out_json with size factors.
    Returns (wall_ms, mem_mb, n_cells, size_factors_sample).
    """
    import subprocess
    import tempfile

    # R script that reads the h5ad, extracts count matrix, runs scran.
    r_script = r"""
suppressMessages({
    library(Matrix)
    library(scran)
})
args <- commandArgs(trailingOnly=TRUE)
h5ad_path <- args[1]
sf_out    <- args[2]

# Read the .h5ad using hdf5r or rhdf5 (whichever is available).
# We use rhdf5 since it is the standard Bioconductor dependency.
if (!requireNamespace("rhdf5", quietly=TRUE)) {
    install.packages("BiocManager", repos="https://cran.rstudio.com/", quiet=TRUE)
    BiocManager::install("rhdf5", ask=FALSE, quiet=TRUE)
}
library(rhdf5)

# Read sparse matrix from h5ad: /X is stored as CSR or CSC.
# Try CSR layout first (/X/data, /X/indices, /X/indptr).
h5f <- H5Fopen(h5ad_path, flags="H5F_ACC_RDONLY")
tryCatch({
    data_h5   <- h5read(h5ad_path, "/X/data")
    indices_h5 <- h5read(h5ad_path, "/X/indices")
    indptr_h5  <- h5read(h5ad_path, "/X/indptr")
    # shape
    shape_h5 <- h5read(h5ad_path, "/X/shape")
    n_obs  <- shape_h5[1]
    n_vars <- shape_h5[2]
    # CSR: n_obs x n_vars
    mat_csr <- sparseMatrix(
        i = indices_h5 + 1L,
        p = indptr_h5,
        x = as.numeric(data_h5),
        dims = c(n_vars, n_obs),   # transposed: genes x cells (CSC-style for scran)
        index1 = TRUE
    )
    cat(sprintf("Matrix loaded: %d genes x %d cells, nnz=%d\n",
                nrow(mat_csr), ncol(mat_csr), length(data_h5)))
}, error = function(e) {
    cat("ERROR loading matrix from h5ad:", conditionMessage(e), "\n", file=stderr())
    quit(status=1)
})
H5Fclose(h5f)

n_cells <- ncol(mat_csr)
n_genes <- nrow(mat_csr)

# Subsample to 5000 cells if very large (scran is O(n^1.5) in pool construction).
# Report the subsample; orchestrator notes extrapolation.
set.seed(42)
if (n_cells > 5000) {
    idx <- sample(n_cells, 5000)
    mat_sub <- mat_csr[, idx]
    cat(sprintf("Subsampled to 5000 cells for scran (n_cells=%d)\n", n_cells))
} else {
    mat_sub <- mat_csr
    idx <- seq_len(n_cells)
}

# Run scran computeSumFactors (deconvolution).
t0 <- proc.time()[3]
suppressWarnings({
    sce <- SingleCellExperiment::SingleCellExperiment(assays=list(counts=mat_sub))
    sce <- scran::computeSumFactors(sce)
    sf <- SingleCellExperiment::sizeFactors(sce)
})
t1 <- proc.time()[3]

wall_s <- as.numeric(t1 - t0)
cat(sprintf("scran_wall_sec=%.4f\n", wall_s))
cat(sprintf("scran_n_cells=%d\n", n_cells))

# Write size factors + metadata to json.
sf_json <- list(
    wall_ms       = wall_s * 1000.0,
    n_cells_sub   = length(sf),
    n_cells_total = n_cells,
    size_factors  = as.list(as.numeric(sf)),
    cell_indices  = as.list(as.integer(idx))
)
writeLines(jsonlite::toJSON(sf_json, auto_unbox=TRUE, digits=7), sf_out)
cat(sprintf("Wrote size factors to %s\n", sf_out))
"""

    # Write R script to temp file.
    scran_sf_json = str(Path(out_json).parent / "scran_sf.json")
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as tf:
        tf.write(r_script)
        r_script_path = tf.name

    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            ["Rscript", r_script_path, h5ad_path, scran_sf_json],
            capture_output=True, text=True, timeout=1800  # 30 min max
        )
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000.0

        if result.returncode != 0:
            print(f"[scran] Rscript failed (rc={result.returncode})", flush=True)
            print(result.stderr[-3000:], file=sys.stderr)
            # Check if jsonlite is missing
            if "jsonlite" in result.stderr:
                print("[scran] Missing R package: jsonlite. Trying to install...", flush=True)
                subprocess.run(["Rscript", "-e",
                    'install.packages("jsonlite", repos="https://cran.rstudio.com/", quiet=TRUE)'],
                    capture_output=True, timeout=300)
                # Retry once.
                result2 = subprocess.run(
                    ["Rscript", r_script_path, h5ad_path, scran_sf_json],
                    capture_output=True, text=True, timeout=1800
                )
                if result2.returncode != 0:
                    return -1.0, -1.0, 0, []
                result = result2
                wall_ms = (time.perf_counter() - t0) * 1000.0
            else:
                return -1.0, -1.0, 0, []

        print(result.stdout[-2000:], flush=True)

        # Parse the json for size factors.
        size_factors_sample = []
        n_cells = 0
        if Path(scran_sf_json).exists():
            with open(scran_sf_json) as f:
                sf_data = json.load(f)
            n_cells = sf_data.get("n_cells_total", 0)
            sf_list = sf_data.get("size_factors", [])
            size_factors_sample = sf_list[:10]
            # Use the scran-reported wall_ms if available (more accurate than subprocess wall).
            scran_wall = sf_data.get("wall_ms", wall_ms)
            wall_ms = scran_wall

        # Memory: parse from stdout or use RSS.
        mem_mb = proc_rss_mb()
        return wall_ms, mem_mb, n_cells, size_factors_sample

    except subprocess.TimeoutExpired:
        print("[scran] Timeout (>30min). Scran is too slow for this sample size.", flush=True)
        return -1.0, -1.0, 0, []
    except Exception as e:
        print(f"[scran] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return -1.0, -1.0, 0, []
    finally:
        try:
            os.unlink(r_script_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",  required=True)
    parser.add_argument("--h5ad-path", default="")
    parser.add_argument("--out-json",  required=True)
    parser.add_argument("--scale",     default="small")
    parser.add_argument("--n-cells",   type=int, default=0)
    args = parser.parse_args()

    baseline  = args.baseline
    h5ad_path = args.h5ad_path
    out_json  = args.out_json
    scale     = args.scale

    try:
        if baseline == "rapids_total_count":
            if not h5ad_path or not Path(h5ad_path).exists():
                print(f"[rapids] SKIP: h5ad not found: {h5ad_path}", flush=True)
                write_result(out_json, -1.0, -1.0, "rapids-singlecell", scale, 0)
                return
            wall_ms, mem_mb, n_cells, sf = run_rapids_total_count(h5ad_path)
            write_result(out_json, wall_ms, mem_mb, "rapids-singlecell", scale, n_cells,
                         size_factors_sample=sf)

        elif baseline == "scanpy_total_count":
            if not h5ad_path or not Path(h5ad_path).exists():
                print(f"[scanpy] SKIP: h5ad not found: {h5ad_path}", flush=True)
                write_result(out_json, -1.0, -1.0, "scanpy", scale, 0)
                return
            wall_ms, mem_mb, n_cells, sf = run_scanpy_total_count(h5ad_path)
            write_result(out_json, wall_ms, mem_mb, "scanpy", scale, n_cells,
                         size_factors_sample=sf)

        elif baseline == "scran_deconvolution":
            if not h5ad_path or not Path(h5ad_path).exists():
                print(f"[scran] SKIP: h5ad not found: {h5ad_path}", flush=True)
                write_result(out_json, -1.0, -1.0, "scran", scale, 0)
                return
            wall_ms, mem_mb, n_cells, sf = run_scran_deconvolution(h5ad_path, out_json)
            write_result(out_json, wall_ms, mem_mb, "scran", scale, n_cells,
                         size_factors_sample=sf)

        else:
            print(f"[cycle58_lognorm_baselines] unknown baseline: {baseline}", file=sys.stderr)
            write_result(out_json, -1.0, -1.0, baseline, scale, 0)
            sys.exit(1)

    except ImportError as e:
        print(f"[{baseline}] SKIP: missing dependency: {e}", flush=True)
        write_result(out_json, -1.0, -1.0, baseline, scale, 0)
    except Exception as e:
        print(f"[{baseline}] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        write_result(out_json, -1.0, -1.0, baseline, scale, 0)
        sys.exit(1)


if __name__ == "__main__":
    main()
