#!/usr/bin/env python3
"""
singlet-gpu/bench/refs/cycle57_baselines.py
SPDX-License-Identifier: MIT

Cycle 57 Phase E — baseline measurements for io/pz_device_loader benchmark.

Runs one of three baseline implementations and writes timing JSON to --out-json.

Baselines:
  scanpy      — scanpy.read_10x_h5() on a .h5 file (CPU → host CSR)
  anndata_gpu — anndata.read_h5ad() + cupy.sparse.csr_matrix() (CPU→GPU)
  factornet_spz — factornet io/spz_loader on a .spz file (skipped if no .spz)

Usage:
    python3 cycle57_baselines.py
        --baseline=<scanpy|anndata_gpu|factornet_spz>
        --scale=<tiny|small|medium>
        --pz-path=<path.1pz>
        --h5-path=<path.h5>
        --h5ad-path=<path.h5ad>
        --out-json=<path.json>

JSON output:
    { "wall_ms": float, "mem_mb": float, "pcie_mb": float,
      "impl": string, "scale": string, "n_cells": int }
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
    """Read VmRSS from /proc/self/status."""
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
    median = wall_times[n // 2] if n % 2 == 1 else 0.5 * (wall_times[n // 2 - 1] + wall_times[n // 2])
    return median, max(peak_mb_list), last_result


def write_result(out_json: str, wall_ms: float, mem_mb: float,
                 pcie_mb: float, impl: str, scale: str, n_cells: int):
    with open(out_json, "w") as f:
        json.dump({
            "wall_ms": wall_ms,
            "mem_mb": mem_mb,
            "pcie_mb": pcie_mb,
            "impl": impl,
            "scale": scale,
            "n_cells": n_cells,
        }, f)
    print(f"[{impl}] wall={wall_ms:.1f}ms mem={mem_mb:.0f}MB pcie={pcie_mb:.0f}MB n_cells={n_cells}",
          flush=True)


def run_scanpy(h5_path: str) -> tuple:
    """scanpy.read_10x_h5 — CPU baseline. Uses /proc/self/status for RSS (no psutil)."""
    import scanpy as sc

    def _load():
        return sc.read_10x_h5(h5_path, genome=None, gex_only=True)

    wall_ms, mem_mb, adata = time_fn(_load)
    n_cells = adata.n_obs if hasattr(adata, "n_obs") else adata.shape[0]
    # PCIe: zero (CPU-only — data stays on host).
    return wall_ms, mem_mb, 0.0, n_cells


def run_anndata_gpu(h5ad_path: str) -> tuple:
    """anndata.read_h5ad + cupy.sparse.csr_matrix — GPU baseline."""
    import anndata as ad
    import numpy as np

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        has_cupy = True
    except ImportError:
        has_cupy = False

    def _load():
        adata = ad.read_h5ad(h5ad_path)
        if has_cupy:
            import scipy.sparse as sp
            if sp.issparse(adata.X):
                csr = adata.X.tocsr()
            else:
                csr = sp.csr_matrix(adata.X)
            # Upload to GPU via cupy.
            gpu_mat = cpsp.csr_matrix(
                (cp.asarray(csr.data.astype(np.float32)),
                 cp.asarray(csr.indices.astype(np.int32)),
                 cp.asarray(csr.indptr.astype(np.int32))),
                shape=csr.shape
            )
            cp.cuda.stream.get_current_stream().synchronize()
            return gpu_mat
        return adata

    wall_ms, mem_mb, result = time_fn(_load)
    if has_cupy and hasattr(result, "shape"):
        n_cells = result.shape[0]
        # PCIe estimate: nnz * (4+4) bytes for data+indices + (n+1)*4 for indptr.
        # We use host CSR nnz if available.
        import anndata as ad
        adata_tmp = ad.read_h5ad(h5ad_path)
        import scipy.sparse as sp
        if sp.issparse(adata_tmp.X):
            nnz = adata_tmp.X.nnz
            n_cells_real = adata_tmp.n_obs
        else:
            nnz = 0
            n_cells_real = adata_tmp.n_obs
        pcie_mb = (nnz * 8 + (n_cells_real + 1) * 4) / (1024 * 1024)
    else:
        adata_tmp = result
        n_cells = adata_tmp.n_obs if hasattr(adata_tmp, "n_obs") else adata_tmp.shape[0]
        pcie_mb = 0.0

    return wall_ms, mem_mb, pcie_mb, n_cells


def run_factornet_spz(_pz_path: str) -> tuple:
    """
    factornet spz_loader — only works if a .spz version of the sample exists.
    .spz is factornet's own format; there is no .spz version of our GSM4037629
    sample. This baseline is SKIPPED and returns (-1, -1, -1, 0).
    """
    # The .spz format is produced by factornet's own encoder; singlet produces
    # .1pz only. There is no .spz version of any of our pipeline outputs.
    # Return sentinel values — the C++ driver treats wall_ms < 0 as "not run".
    print("[factornet_spz] SKIP: no .spz file available for pipeline outputs.",
          flush=True)
    return -1.0, -1.0, -1.0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",  required=True)
    parser.add_argument("--scale",     required=True)
    parser.add_argument("--pz-path",   default="")
    parser.add_argument("--h5-path",   default="")
    parser.add_argument("--h5ad-path", default="")
    parser.add_argument("--out-json",  required=True)
    args = parser.parse_args()

    baseline  = args.baseline
    scale     = args.scale
    out_json  = args.out_json
    h5_path   = args.h5_path
    h5ad_path = args.h5ad_path
    pz_path   = args.pz_path

    try:
        if baseline == "scanpy":
            if not h5_path or not Path(h5_path).exists():
                print(f"[scanpy] SKIP: h5 not found: {h5_path}", flush=True)
                write_result(out_json, -1.0, -1.0, -1.0, "scanpy", scale, 0)
                return
            wall_ms, mem_mb, pcie_mb, n_cells = run_scanpy(h5_path)
            write_result(out_json, wall_ms, mem_mb, pcie_mb, "scanpy/read_10x_h5", scale, n_cells)

        elif baseline == "anndata_gpu":
            if not h5ad_path or not Path(h5ad_path).exists():
                print(f"[anndata_gpu] SKIP: h5ad not found: {h5ad_path}", flush=True)
                write_result(out_json, -1.0, -1.0, -1.0, "anndata-gpu", scale, 0)
                return
            wall_ms, mem_mb, pcie_mb, n_cells = run_anndata_gpu(h5ad_path)
            write_result(out_json, wall_ms, mem_mb, pcie_mb, "anndata-gpu", scale, n_cells)

        elif baseline == "factornet_spz":
            wall_ms, mem_mb, pcie_mb, n_cells = run_factornet_spz(pz_path)
            write_result(out_json, wall_ms, mem_mb, pcie_mb, "factornet/spz_loader", scale, n_cells)

        else:
            print(f"[cycle57_baselines] unknown baseline: {baseline}", file=sys.stderr)
            write_result(out_json, -1.0, -1.0, -1.0, baseline, scale, 0)
            sys.exit(1)

    except ImportError as e:
        print(f"[{baseline}] SKIP: missing dependency: {e}", flush=True)
        write_result(out_json, -1.0, -1.0, -1.0, baseline, scale, 0)
    except Exception as e:
        print(f"[{baseline}] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        write_result(out_json, -1.0, -1.0, -1.0, baseline, scale, 0)
        sys.exit(1)


if __name__ == "__main__":
    main()
