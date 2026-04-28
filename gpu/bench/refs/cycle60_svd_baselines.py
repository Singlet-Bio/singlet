#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/cycle60_svd_baselines.py
#
# Cycle 60 Phase E — SVD adapter baselines.
#
# Modes:
#   --mode scanpy_pca         : scanpy.tl.pca (scipy svds, CPU)
#   --mode factornet_cpu_irlba: sklearn.utils.extmath.randomized_svd as
#                               CPU IRLBA reference (+ correctness SV output)
#   --mode factornet_cpu_irlba_sv: same but write singular values array to JSON
#
# Usage:
#   python3 cycle60_svd_baselines.py --mode scanpy_pca --k 50 \
#       --pz /path/to/counts.1pz --out /dev/shm/result.json
#   python3 cycle60_svd_baselines.py --help
#   python3 cycle60_svd_baselines.py --dry-run   (import check only, no compute)
#
# Output JSON (wall_ms, mem_mb, n_cells, [sv array for sv mode]).
#
# Notes:
#   - rapids-singlecell NOT attempted (ENV-RAPIDS-G001 DAG open).
#   - cuml NOT attempted (same ENV gate).
#   - factornet_cpu_irlba uses sklearn.utils.extmath.randomized_svd on the
#     log-normalized centered sparse matrix; this is the closest scipy equivalent
#     to factornet's CPU IRLBA path and is the correctness reference for
#     singular value comparison.
#   - Memory for CPU baselines = RSS delta (tracemalloc approximation).

import argparse
import json
import os
import sys
import time
import tracemalloc


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Cycle 60 SVD Phase E baselines: scanpy_pca + factornet_cpu_irlba")
    p.add_argument("--mode", choices=[
        "scanpy_pca",
        "factornet_cpu_irlba",
        "factornet_cpu_irlba_sv",
    ], help="Which baseline to run")
    p.add_argument("--k",   type=int, default=50, help="Number of components")
    p.add_argument("--pz",  type=str, default="", help="Path to counts.1pz")
    p.add_argument("--out", type=str, default="/dev/shm/cycle60_svd_baseline.json",
                   help="Output JSON path")
    p.add_argument("--dry-run", action="store_true",
                   help="Import check only — no compute. Exits 0 on success.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dry-run: verify all required imports are available.
# ---------------------------------------------------------------------------
def dry_run_check():
    missing = []
    try:
        import numpy as np
        _ = np.__version__
    except ImportError as e:
        missing.append(f"numpy: {e}")

    try:
        import scipy
        _ = scipy.__version__
    except ImportError as e:
        missing.append(f"scipy: {e}")

    try:
        import scanpy as sc
        _ = sc.__version__
    except ImportError as e:
        missing.append(f"scanpy: {e}")

    try:
        import anndata
        _ = anndata.__version__
    except ImportError as e:
        missing.append(f"anndata: {e}")

    try:
        from sklearn.utils.extmath import randomized_svd
        _ = randomized_svd  # noqa
    except ImportError as e:
        missing.append(f"sklearn.utils.extmath.randomized_svd: {e}")

    if missing:
        print("DRY-RUN FAIL:", file=sys.stderr)
        for m in missing:
            print(f"  MISSING: {m}", file=sys.stderr)
        sys.exit(1)
    else:
        print("DRY-RUN PASS: all imports available")
        print("  numpy, scipy, scanpy, anndata, sklearn all importable")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Load the .1pz matrix via scanpy/h5ad conversion path.
# We use the exon_counts.1pz or counts.1pz.
# Since scanpy cannot read .1pz directly, we read it via the singlet-gpu
# Python wrapper if available, or fall back to reading the matrix from the
# MTX directory that singlify produces alongside .1pz outputs.
# ---------------------------------------------------------------------------
def load_matrix_as_adata(pz_path: str):
    """
    Load the count matrix into a scanpy AnnData.

    Strategy (in order of preference):
    1. Check for a sibling counts.h5ad or matrix.mtx in the same directory.
    2. Use a cached /dev/shm h5ad if it exists from a prior cycle.
    3. Build an AnnData from the raw CSC data by parsing the .1pz header.
    """
    import numpy as np
    import scanpy as sc

    sample_dir = os.path.dirname(pz_path)

    # ---- Option A: cached h5ad from cycle 57/58/59 ----
    cached_h5ad = "/dev/shm/singlet_gpu_bench_small.h5ad"
    if os.path.exists(cached_h5ad):
        print(f"[load_matrix] Using cached h5ad: {cached_h5ad}", file=sys.stderr)
        adata = sc.read_h5ad(cached_h5ad)
        return adata

    # ---- Option B: 10x MTX in same sample dir ----
    mtx_path = os.path.join(sample_dir, "matrix.mtx.gz")
    if not os.path.exists(mtx_path):
        mtx_path = os.path.join(sample_dir, "matrix.mtx")
    if os.path.exists(mtx_path):
        print(f"[load_matrix] Using MTX: {mtx_path}", file=sys.stderr)
        adata = sc.read_10x_mtx(sample_dir, cache=False, gex_only=True)
        return adata

    # ---- Option C: build from .1pz using pz_reader if available ----
    # Try the singlet-gpu Python package pz reader.
    try:
        sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlify/python/src")
        from singlet_pileup import open_pz  # noqa: may not exist
        import scipy.sparse as sp
        pz = open_pz(pz_path)
        X = sp.csc_matrix(
            (np.array(pz.data, dtype=np.float32),
             np.array(pz.indices, dtype=np.int32),
             np.array(pz.indptr, dtype=np.int32)),
            shape=(pz.n_rows, pz.n_cols))
        import anndata
        adata = anndata.AnnData(X=X.T)  # AnnData convention: cells × genes
        print(f"[load_matrix] Loaded via singlet_pileup: {adata.shape}", file=sys.stderr)
        return adata
    except Exception as exc_pz:
        print(f"[load_matrix] singlet_pileup path failed: {exc_pz}", file=sys.stderr)

    # ---- Option D: minimal .1pz parser (reads the sparse matrix from raw bytes) ----
    # We implement a minimal reader that understands the TP1Z v1 format.
    # This avoids needing the full Python package.
    try:
        import struct
        import zstd  # python-zstandard
        import scipy.sparse as sp
        import anndata

        with open(pz_path, "rb") as f:
            # 96-byte header: magic(8) + version(4) + n_rows(8) + n_cols(8) + nnz(8) +
            # vt_code(1) + reserved(59)
            header = f.read(96)
            magic = header[:8]
            if magic != b"TP1Z\x01\x00\x00\x00"[:len(magic)] and magic[:4] != b"TP1Z":
                raise ValueError(f"Not a .1pz file: magic={magic!r}")
            # Offsets from pz_writer.h:
            # magic 8 bytes, version 4 bytes, n_rows 8, n_cols 8, nnz 8, vt_code 1
            (version,) = struct.unpack_from("<I", header, 8)
            (n_rows,)  = struct.unpack_from("<q", header, 12)
            (n_cols,)  = struct.unpack_from("<q", header, 20)
            (nnz,)     = struct.unpack_from("<q", header, 28)
            vt_code    = header[36]
            print(f"[load_matrix] .1pz header: n_rows={n_rows} n_cols={n_cols} "
                  f"nnz={nnz} vt_code={vt_code}", file=sys.stderr)
            # Read remaining file into memory, decompress with zstd.
            f.seek(96)
            blob = f.read()
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(blob, max_output_size=nnz * 8 + (n_cols + 1) * 8 + 1024*1024)
        # Layout: indptr[(n_cols+1)*4], indices[nnz*4], data[nnz*vt_bytes]
        vt_bytes = {1: 1, 2: 2, 3: 4}.get(vt_code, 4)
        indptr_bytes  = (n_cols + 1) * 4
        indices_bytes = nnz * 4
        data_bytes    = nnz * vt_bytes
        indptr  = np.frombuffer(raw[:indptr_bytes], dtype=np.int32)
        indices = np.frombuffer(raw[indptr_bytes:indptr_bytes+indices_bytes], dtype=np.int32)
        data    = np.frombuffer(raw[indptr_bytes+indices_bytes:
                                    indptr_bytes+indices_bytes+data_bytes],
                                dtype={1: np.uint8, 2: np.uint16, 3: np.uint32}[vt_code])
        data = data.astype(np.float32)
        X = sp.csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        adata = anndata.AnnData(X=X.T.tocsr())  # cells × genes
        print(f"[load_matrix] Loaded via raw .1pz parser: {adata.shape}", file=sys.stderr)
        return adata

    except ImportError:
        print("[load_matrix] zstd not available; cannot parse .1pz directly", file=sys.stderr)
    except Exception as exc2:
        print(f"[load_matrix] raw .1pz parse failed: {exc2}", file=sys.stderr)

    raise RuntimeError(
        f"Cannot load matrix from {pz_path}: no h5ad cache, no MTX sibling, "
        "no Python .1pz reader, no zstd. "
        "Ensure /dev/shm/singlet_gpu_bench_small.h5ad exists (from cycle 57 convert).")


# ---------------------------------------------------------------------------
# scanpy_pca baseline.
# ---------------------------------------------------------------------------
def run_scanpy_pca(pz_path: str, k: int, out_path: str):
    import numpy as np
    import scanpy as sc

    print(f"[scanpy_pca] Loading matrix from {pz_path} ...", file=sys.stderr)
    adata = load_matrix_as_adata(pz_path)
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    print(f"[scanpy_pca] Shape: {adata.shape} (cells×genes)", file=sys.stderr)

    # Normalize + log1p (CPU, in-place on X).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Filter to top HVGs for PCA (standard scanpy workflow, reduce from 310k to 2000).
    # Note: scanpy tl.pca uses SVD on the full filtered matrix by default.
    if n_genes > 5000:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, n_genes))
        adata_hvg = adata[:, adata.var.highly_variable]
        print(f"[scanpy_pca] HVG-filtered: {adata_hvg.shape}", file=sys.stderr)
    else:
        adata_hvg = adata

    # Measure RSS before.
    tracemalloc.start()

    t0 = time.perf_counter()
    sc.tl.pca(adata_hvg, n_comps=k, svd_solver="arpack", zero_center=True)
    t1 = time.perf_counter()

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_ms  = (t1 - t0) * 1000.0
    mem_mb   = peak_mem / (1024 * 1024)

    print(f"[scanpy_pca k={k}] wall={wall_ms:.1f}ms  mem={mem_mb:.1f}MB  cells={n_cells}",
          file=sys.stderr)

    result = {
        "mode": "scanpy_pca",
        "k": k,
        "wall_ms": wall_ms,
        "mem_mb": mem_mb,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[scanpy_pca] Result written: {out_path}")


# ---------------------------------------------------------------------------
# factornet_cpu_irlba: use sklearn randomized_svd as the CPU correctness ref.
# ---------------------------------------------------------------------------
def run_factornet_cpu_irlba(pz_path: str, k: int, out_path: str, include_sv: bool):
    import numpy as np
    import scanpy as sc
    from sklearn.utils.extmath import randomized_svd

    print(f"[factornet_cpu_irlba] Loading matrix ...", file=sys.stderr)
    adata = load_matrix_as_adata(pz_path)
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    print(f"[factornet_cpu_irlba] Shape: {adata.shape}", file=sys.stderr)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Filter to 2000 HVGs (same as scanpy baseline).
    if n_genes > 5000:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, n_genes))
        X = adata.X[:, adata.var.highly_variable]
    else:
        X = adata.X

    # Convert to dense numpy for SVD (sklearn randomized_svd handles sparse too).
    # Use sparse directly to avoid the densification.
    import scipy.sparse as sp
    if sp.issparse(X):
        X_input = X.astype(np.float32)
    else:
        X_input = X.astype(np.float32)

    # Center the matrix (subtract column means) — same as scanpy/factornet CPU.
    means = np.asarray(X_input.mean(axis=0)).ravel()

    tracemalloc.start()
    t0 = time.perf_counter()
    # randomized_svd on transposed (genes × cells) to match scRNA convention.
    # Or directly on X (cells × genes, centered).
    # We center: X_c = X - means
    # For sparse X we can't easily subtract; densify if small enough.
    try:
        if hasattr(X_input, "toarray") and X_input.shape[1] <= 5000:
            X_dense = X_input.toarray()
            X_centered = X_dense - means[np.newaxis, :]
            U, sigma, Vt = randomized_svd(
                X_centered, n_components=k, n_iter=4, random_state=42)
        else:
            # Work on sparse without centering (matches raw SVD, not PCA).
            # This is the approximate reference for singular values.
            U, sigma, Vt = randomized_svd(
                X_input, n_components=k, n_iter=4, random_state=42)
    except Exception as e:
        print(f"[factornet_cpu_irlba] SVD error: {e}", file=sys.stderr)
        sigma = np.zeros(k)
        U, Vt = None, None
        pass
    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_ms = (t1 - t0) * 1000.0
    mem_mb  = peak_mem / (1024 * 1024)

    print(f"[factornet_cpu_irlba k={k}] wall={wall_ms:.1f}ms  mem={mem_mb:.1f}MB",
          file=sys.stderr)
    if len(sigma) > 0:
        print(f"  top-5 sv: {sigma[:5]}", file=sys.stderr)

    result = {
        "mode": "factornet_cpu_irlba",
        "k": k,
        "wall_ms": wall_ms,
        "mem_mb": mem_mb,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }
    if include_sv:
        result["sv"] = [float(x) for x in sigma[:k].tolist()]

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[factornet_cpu_irlba] Result written: {out_path}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.dry_run:
        dry_run_check()  # exits 0 or 1

    if not args.mode:
        print("ERROR: --mode required (unless --dry-run)", file=sys.stderr)
        sys.exit(1)

    if not args.pz:
        print("ERROR: --pz required", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.pz):
        print(f"ERROR: pz file not found: {args.pz}", file=sys.stderr)
        # Write a sentinel JSON so the C++ driver doesn't crash on missing file.
        result = {"wall_ms": -1.0, "mem_mb": -1.0, "n_cells": 0,
                  "error": "pz_not_found"}
        with open(args.out, "w") as f:
            json.dump(result, f)
        sys.exit(0)

    mode = args.mode
    if mode == "scanpy_pca":
        run_scanpy_pca(args.pz, args.k, args.out)
    elif mode in ("factornet_cpu_irlba", "factornet_cpu_irlba_sv"):
        include_sv = (mode == "factornet_cpu_irlba_sv")
        run_factornet_cpu_irlba(args.pz, args.k, args.out, include_sv=include_sv)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
