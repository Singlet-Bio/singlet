#!/usr/bin/env python3
"""
singlet-gpu/bench/refs/cycle59_hvg_baselines.py
SPDX-License-Identifier: MIT

Cycle 59 Phase E — HVG SOTA baselines.

Runs one of:
  scanpy_seurat_v3         — scanpy pp.highly_variable_genes(flavor='seurat_v3') (CPU)
  scanpy_pearson_residuals — scanpy experimental pp.highly_variable_genes(flavor='pearson_residuals') (CPU)
  rapids_seurat_v3         — rapids-singlecell pp.highly_variable_genes (GPU) — skipped if absent
  seurat_v3_R              — Seurat FindVariableFeatures(method='vst') via Rscript subprocess

Usage:
    python3 cycle59_hvg_baselines.py
        --baseline=<scanpy_seurat_v3|scanpy_pearson_residuals|rapids_seurat_v3|seurat_v3_R>
        --h5ad-path=<path.h5ad>
        --out-json=<path.json>

JSON output:
    {
        "impl": str,
        "wall_ms": float,
        "mem_mb": float,
        "n_genes": int,
        "n_cells": int,
        "top2000_hvg": [gene_name, ...],        # top-2000 HVG gene names (seurat_v3)
        "status": "ok" | "skipped" | "error",
        "skip_reason": str                      # if status=="skipped"
    }
"""

import argparse
import json
import os
import subprocess
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
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
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
    median_ms = wall_times[n // 2] if n % 2 == 1 else 0.5 * (
        wall_times[n // 2 - 1] + wall_times[n // 2])
    return median_ms, max(peak_mb_list), last_result


def write_result(out_json, impl, wall_ms, mem_mb, n_genes, n_cells,
                 top2000_hvg=None, status="ok", skip_reason=""):
    d = {
        "impl": impl,
        "wall_ms": wall_ms,
        "mem_mb": mem_mb,
        "n_genes": n_genes,
        "n_cells": n_cells,
        "top2000_hvg": top2000_hvg or [],
        "status": status,
        "skip_reason": skip_reason,
    }
    with open(out_json, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[{impl}] wall={wall_ms:.1f}ms  mem={mem_mb:.0f}MB  n_genes={n_genes}  "
          f"n_top2000={len(d['top2000_hvg'])}  status={status}", flush=True)


def run_scanpy_seurat_v3(h5ad_path, out_json):
    import scanpy as sc
    import numpy as np
    adata = sc.read_h5ad(h5ad_path)
    # Need raw counts for seurat_v3; assume h5ad was stored as raw counts.
    # scanpy seurat_v3 operates on the raw matrix directly (not log-normalized).
    #
    # Near-singularity fix: GSM4037629 has 310k genes; many have mean<0.0125
    # (near-identical values), causing skmisc LOESS to raise ValueError.
    # Pre-filter to genes with mean >= 0.0125 (matching GPU kernel MIN_MEAN).
    # This is equivalent to what Seurat/scanpy recommend for large atlases.
    gene_means = np.array(adata.X.mean(axis=0)).ravel()
    keep = gene_means >= 0.0125
    print(f"  [seurat_v3] Pre-filter: {adata.n_vars} → {keep.sum()} genes (mean >= 0.0125)", flush=True)
    adata_filt = adata[:, keep].copy()

    def _run(ad):
        a2 = ad.copy()
        sc.pp.highly_variable_genes(a2, flavor="seurat_v3", n_top_genes=2000)
        return a2

    wall_ms, mem_mb, result = time_fn(_run, adata_filt)
    top2000 = []
    if result is not None and "highly_variable" in result.var.columns:
        hvg_mask = result.var["highly_variable"].values
        gene_names = list(result.var_names)
        top2000 = [gene_names[i] for i in range(len(gene_names)) if hvg_mask[i]][:2000]
    write_result(out_json, "scanpy_seurat_v3", wall_ms, mem_mb,
                 adata.n_vars, adata.n_obs, top2000)


def run_scanpy_pearson_residuals(h5ad_path, out_json):
    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)
    def _run(ad):
        import copy
        a2 = copy.copy(ad)
        try:
            sc.experimental.pp.highly_variable_genes(
                a2, flavor="pearson_residuals", n_top_genes=2000)
        except AttributeError:
            # Newer scanpy: sc.pp.highly_variable_genes with flavor=pearson_residuals
            sc.pp.highly_variable_genes(
                a2, flavor="pearson_residuals", n_top_genes=2000)
        return a2
    wall_ms, mem_mb, result = time_fn(_run, adata)
    top2000 = []
    if result is not None and "highly_variable" in result.var.columns:
        hvg_mask = result.var["highly_variable"].values
        gene_names = list(result.var_names)
        top2000 = [gene_names[i] for i in range(len(gene_names)) if hvg_mask[i]][:2000]
    write_result(out_json, "scanpy_pearson_residuals", wall_ms, mem_mb,
                 adata.n_vars, adata.n_obs, top2000)


def run_rapids_seurat_v3(h5ad_path, out_json):
    try:
        import rapids_singlecell as rsc
        import cupy as cp
        import anndata as ad
        import scanpy as sc
    except ImportError as e:
        write_result(out_json, "rapids_seurat_v3", -1.0, -1.0, 0, 0,
                     status="skipped", skip_reason=f"rapids-singlecell not available: {e}")
        return
    adata = sc.read_h5ad(h5ad_path)
    rsc.get.anndata_to_GPU(adata)
    def _run(ad):
        import copy
        a2 = ad.copy()
        rsc.pp.highly_variable_genes(a2, flavor="seurat_v3", n_top_genes=2000)
        return a2
    wall_ms, mem_mb, result = time_fn(_run, adata)
    top2000 = []
    if result is not None and "highly_variable" in result.var.columns:
        hvg_mask = result.var["highly_variable"].values
        gene_names = list(result.var_names)
        top2000 = [gene_names[i] for i in range(len(gene_names)) if hvg_mask[i]][:2000]
    write_result(out_json, "rapids_seurat_v3", wall_ms, mem_mb,
                 adata.n_vars, adata.n_obs, top2000)


def run_seurat_v3_R(h5ad_path, out_json):
    """Run Seurat FindVariableFeatures(method='vst') via Rscript subprocess."""
    rscript = "Rscript"
    # Check Rscript is available.
    try:
        subprocess.run([rscript, "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        write_result(out_json, "seurat_v3_R", -1.0, -1.0, 0, 0,
                     status="skipped", skip_reason=f"Rscript not available: {e}")
        return

    # Write a temp R script.
    r_script = f"""
suppressMessages(library(Seurat))
suppressMessages(library(Matrix))

# Load h5ad via SeuratDisk or native h5seurat
# Fallback: load h5ad as a Seurat object via Python-written MEX (if available)
# We use SeuratDisk if available, else skip.

tryCatch({{
    if (!requireNamespace("SeuratDisk", quietly=TRUE)) {{
        cat("SeuratDisk not available\\n")
        cat("STATUS=skipped REASON=SeuratDisk_missing\\n")
        quit(status=0)
    }}
    library(SeuratDisk)
    h5ad_path <- "{h5ad_path}"
    # Convert .h5ad -> .h5seurat
    h5seurat_path <- tempfile(fileext=".h5seurat")
    Convert(h5ad_path, dest="h5seurat", assay="RNA", overwrite=TRUE)
    obj <- LoadH5Seurat(h5seurat_path, verbose=FALSE)
    # Ensure assay has counts
    if (!("counts" %in% slotNames(obj[["RNA"]])) || is.null(obj[["RNA"]]@counts)) {{
        cat("STATUS=skipped REASON=no_raw_counts_in_h5seurat\\n")
        quit(status=0)
    }}
    # Bench: 2 warmup + 5 timed
    times <- c()
    for (i in 1:7) {{
        t0 <- proc.time()[3]
        obj2 <- FindVariableFeatures(obj, selection.method="vst", nfeatures=2000, verbose=FALSE)
        t1 <- proc.time()[3]
        if (i > 2) times <- c(times, (t1 - t0) * 1000.0)
    }}
    med_ms <- median(times)
    hvg_genes <- VariableFeatures(obj2)
    cat(sprintf("SEURAT_WALL_MS=%.2f\\n", med_ms))
    cat(sprintf("SEURAT_N_HVG=%d\\n", length(hvg_genes)))
    top2000_str <- paste(hvg_genes[1:min(2000, length(hvg_genes))], collapse=",")
    cat(sprintf("SEURAT_TOP2000=%s\\n", top2000_str))
    cat("STATUS=ok\\n")
}}, error=function(e) {{
    cat(sprintf("STATUS=error REASON=%s\\n", conditionMessage(e)))
}})
"""
    r_tmp = "/dev/shm/cycle59_seurat_hvg.R"
    with open(r_tmp, "w") as f:
        f.write(r_script)
    try:
        result = subprocess.run(
            [rscript, r_tmp], capture_output=True, text=True, timeout=600)
        out = result.stdout
        wall_ms = -1.0
        top2000 = []
        status = "error"
        skip_reason = ""
        n_genes = 0
        n_cells = 0
        for line in out.splitlines():
            if line.startswith("SEURAT_WALL_MS="):
                wall_ms = float(line.split("=", 1)[1])
            elif line.startswith("SEURAT_TOP2000="):
                top2000 = line.split("=", 1)[1].split(",")[:2000]
            elif line.startswith("STATUS=ok"):
                status = "ok"
            elif line.startswith("STATUS=skipped"):
                status = "skipped"
                skip_reason = line.split("REASON=", 1)[-1] if "REASON=" in line else ""
            elif line.startswith("STATUS=error"):
                status = "error"
                skip_reason = line.split("REASON=", 1)[-1] if "REASON=" in line else result.stderr[:200]
        if status == "error" and not skip_reason:
            skip_reason = result.stderr[:400]
        write_result(out_json, "seurat_v3_R", wall_ms, -1.0, n_genes, n_cells,
                     top2000, status=status, skip_reason=skip_reason)
    except subprocess.TimeoutExpired:
        write_result(out_json, "seurat_v3_R", -1.0, -1.0, 0, 0,
                     status="skipped", skip_reason="Rscript timed out (>600s)")
    finally:
        if os.path.exists(r_tmp):
            os.unlink(r_tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
        choices=["scanpy_seurat_v3", "scanpy_pearson_residuals",
                 "rapids_seurat_v3", "seurat_v3_R"])
    parser.add_argument("--h5ad-path", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    h5ad = args.h5ad_path
    out  = args.out_json

    if not Path(h5ad).exists():
        print(f"ERROR: h5ad not found: {h5ad}", file=sys.stderr)
        write_result(out, args.baseline, -1.0, -1.0, 0, 0,
                     status="error", skip_reason=f"h5ad not found: {h5ad}")
        sys.exit(1)

    dispatch = {
        "scanpy_seurat_v3":         run_scanpy_seurat_v3,
        "scanpy_pearson_residuals":  run_scanpy_pearson_residuals,
        "rapids_seurat_v3":          run_rapids_seurat_v3,
        "seurat_v3_R":               run_seurat_v3_R,
    }
    dispatch[args.baseline](h5ad, out)


if __name__ == "__main__":
    main()
