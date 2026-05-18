#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/stagate_ref.py
#
# Performance reference: STAGATE Python vs singlet-gpu spatial/stagate.
#
# Accepts a CSC binary count matrix + float32 spatial coordinates (.npy),
# runs STAGATE (PyTorch + PyG-based) via the stagate_pyg package.
# Falls back gracefully if stagate_pyg or torch is absent — prints
# a NO_SOTA JSON and exits 0.
#
# Environment (GPU path):
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install torch_geometric
#   pip install stagate_pyg   # or: pip install STAGATE_pyG
#   pip install numpy scipy anndata scanpy
#
# Usage:
#   python stagate_ref.py \
#       --input      <matrix.bin>       \   # CSC binary (dump_csc format, genes × spots)
#       --coords     <coords.npy>       \   # float32 (n_spots, 2) spatial XY
#       --n-neighbors <int>             \   # spatial kNN k (default: 6)
#       --n-epochs    <int>             \   # training epochs (default: 20)
#       --timing-json <result.json>         # required
#
# CSC binary format:
#   Magic : uint32  0x43535343
#   n_rows: uint32  (= n_genes)
#   n_cols: uint32  (= n_spots)
#   nnz   : uint64
#   values: float32[nnz]
#   indptr: int32[n_cols+1]
#   indices: int32[nnz]
#
# JSON output schema:
#   { "wall_ms": float, "mem_mb": float, "impl": str, "n_spots": int, "n_epochs": int }
#   wall_ms = -1.0 and impl = "NO_SOTA" if STAGATE is not installed.

import argparse
import json
import os
import struct
import sys
import time

import numpy as np


def _read_csc_bin(path):
    """Read CSC binary file. Returns (values, indptr, indices, n_rows, n_cols, nnz)."""
    with open(path, "rb") as f:
        magic, = struct.unpack("<I", f.read(4))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic: 0x{magic:08x}")
        n_rows, n_cols = struct.unpack("<II", f.read(8))
        nnz, = struct.unpack("<Q", f.read(8))
        values  = np.frombuffer(f.read(nnz * 4),          dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4), dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),          dtype=np.int32).copy()
    return values, indptr, indices, int(n_rows), int(n_cols), int(nnz)


def _memory_delta_mb(fn, *args, **kwargs):
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        result = fn(*args, **kwargs)
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        return result, max(0.0, mem_after - mem_before)
    except Exception:
        pass

    def _rss():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
        return 0.0

    rss_before = _rss()
    result = fn(*args, **kwargs)
    return result, max(0.0, _rss() - rss_before)


def _run_stagate(X_dense, coords, n_neighbors, n_epochs):
    """Run STAGATE via stagate_pyg. Returns embedding array."""
    import torch
    import anndata
    import scanpy as sc
    import STAGATE_pyG as STAGATE

    n_spots, n_genes = X_dense.shape
    adata = anndata.AnnData(X=X_dense.astype(np.float32))
    adata.obsm["spatial"] = coords.astype(np.float64)

    # Build spatial graph (stagate API).
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=n_neighbors,
                             model="KNN", verbose=False)
    # Train STAGATE.
    STAGATE.train_STAGATE(adata, n_epochs=n_epochs, verbose=False)
    return adata.obsm["STAGATE"]


def main():
    parser = argparse.ArgumentParser(
        description="STAGATE Python reference for singlet-gpu spatial/stagate bench")
    parser.add_argument("--input",       required=True,
                        help="CSC binary matrix (genes × spots)")
    parser.add_argument("--coords",      required=True,
                        help="float32 .npy spatial coords (n_spots, 2)")
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--n-epochs",    type=int, default=20)
    parser.add_argument("--timing-json", required=True,
                        help="Path to write timing JSON")
    args = parser.parse_args()

    # Try to import STAGATE; if absent, write NO_SOTA and exit cleanly.
    try:
        import STAGATE_pyG  # noqa: F401
        has_stagate = True
    except ImportError:
        print("[stagate_ref] STAGATE_pyG not available — reporting NO_SOTA",
              file=sys.stderr)
        has_stagate = False

    if not has_stagate:
        result = {"wall_ms": -1.0, "mem_mb": -1.0, "impl": "NO_SOTA",
                  "n_spots": 0, "n_epochs": args.n_epochs}
        with open(args.timing_json, "w") as f:
            json.dump(result, f, indent=2)
        return

    # Load data.
    values, indptr, indices, n_genes, n_spots, nnz = _read_csc_bin(args.input)
    coords = np.load(args.coords).astype(np.float32)

    # Convert CSC (genes × spots) → dense (spots × genes) for STAGATE.
    import scipy.sparse as sp
    csc = sp.csc_matrix((values, indices, indptr), shape=(n_genes, n_spots))
    X_dense = csc.T.toarray()  # (spots × genes)

    print(f"[stagate_ref] n_spots={n_spots} n_genes={n_genes} n_neighbors={args.n_neighbors} "
          f"n_epochs={args.n_epochs}", file=sys.stderr)

    # Warmup run (1 iter — STAGATE is slow so just 1).
    _ = _run_stagate(X_dense, coords, args.n_neighbors, args.n_epochs)

    # 5 timed runs.
    wall_times = []
    mem_mb_last = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        _, mem_mb = _memory_delta_mb(_run_stagate, X_dense, coords,
                                      args.n_neighbors, args.n_epochs)
        wall_times.append((time.perf_counter() - t0) * 1000.0)
        mem_mb_last = mem_mb

    wall_median_ms = float(sorted(wall_times)[len(wall_times) // 2])
    print(f"[stagate_ref] STAGATE_pyG: n_spots={n_spots} wall_ms={wall_median_ms:.1f} "
          f"mem_mb={mem_mb_last:.1f}", file=sys.stderr)

    result = {
        "wall_ms":  wall_median_ms,
        "mem_mb":   float(mem_mb_last),
        "impl":     "STAGATE_pyG",
        "n_spots":  n_spots,
        "n_epochs": args.n_epochs,
    }
    with open(args.timing_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
