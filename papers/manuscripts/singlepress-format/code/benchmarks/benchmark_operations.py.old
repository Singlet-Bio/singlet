#!/usr/bin/env python3
"""
Benchmark common operations across single-cell data formats.

Operations benchmarked:
  1. Random 10% column subsample → deep copy (uniform random, seed=42)
  2. Column-range read (contiguous 10% slice from middle)
  3. On-the-fly log-normalization during read

Formats:
  .1pz   — singlepress native (open_1pz slicing, read_1pz_columns)
  H5AD   — anndata (backed mode for slicing, full read for norm)
  10x H5 — h5py CellRanger format
  scipy npz — scipy.sparse baseline

Usage:
    cd /tmp && srun --time=120 --mem=64G --cpus-per-task=8 \
        python3 -u /path/to/benchmark_operations.py
"""

import csv
import json
import os
import sys
import tempfile
import time
import copy
import gc

# Fix namespace-package shadowing
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as ss
import singlepress as sp

assert hasattr(sp, "write_1pz"), f"singlepress not loaded correctly"

# ── Configuration ────────────────────────────────────────────────
SEED = 42
FRAC = 0.10          # 10% subsample
N_TRIALS = 3
WARMUP = 1
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 15 datasets spanning a range of sizes (subset of the 25-dataset benchmark set)
# Excludes the two tiniest overhead-dominated datasets
DATASETS = {
    "GSE210261": {"protocol": "BD Rhapsody",     "notes": "small ~4.6M nnz"},
    "GSE123662": {"protocol": "10x Chromium v2",  "notes": "zebrafish ~7.5M nnz"},
    "GSE290216": {"protocol": "10x Chromium v2",  "notes": "human ~11M nnz"},
    "GSE209597": {"protocol": "10x Chromium v3",  "notes": "human ~13.2M nnz"},
    "GSE176269": {"protocol": "Seq-Well",          "notes": "human ~14M nnz"},
    "GSE216064": {"protocol": "10x Chromium v3",  "notes": "human ~27M nnz"},
    "GSE304926": {"protocol": "10x Chromium v4",  "notes": "frog ~29M nnz"},
    "GSE103976": {"protocol": "10x Chromium v3",  "notes": "mouse ~32M nnz"},
    "GSE297773": {"protocol": "10x Chromium v3",  "notes": "rat ~40M nnz"},
    "GSE248556": {"protocol": "10x Chromium v3 5'","notes": "human ~47M nnz"},
    "GSE136679": {"protocol": "Seq-Well",          "notes": "mouse ~57M nnz"},
    "GSE187438": {"protocol": "10x Chromium v3",  "notes": "human ~70M nnz"},
    "GSE278497": {"protocol": "DNBelab C4",        "notes": "human ~93M nnz"},
    "GSE306871": {"protocol": "DNBelab C4",        "notes": "mouse ~117M nnz"},
    "GSE203171": {"protocol": "10x Chromium v3",  "notes": "mouse ~136M nnz"},
}


def timer(func, n_trials=N_TRIALS, warmup=WARMUP):
    """Return median time and result of last trial."""
    times = []
    for i in range(warmup + n_trials):
        gc.collect()
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
        del result
    # Run one more to return the actual result
    gc.collect()
    t0 = time.perf_counter()
    result = func()
    t1 = time.perf_counter()
    return np.median(times), result


def write_h5ad(mat_csc, rownames, colnames, path):
    """Write matrix as H5AD (genes×cells CSC → cells×genes CSR for AnnData)."""
    import pandas as pd
    if os.path.exists(path):
        os.remove(path)
    adata = ad.AnnData(
        X=mat_csc.T.tocsr(),
        obs=pd.DataFrame(index=colnames),
        var=pd.DataFrame(index=rownames),
    )
    adata.write_h5ad(path, compression="gzip")


def write_10x_h5(mat_csc, rownames, colnames, path):
    """Write matrix as 10x CellRanger HDF5."""
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("data", data=mat_csc.data.astype(np.int32), compression="gzip")
        grp.create_dataset("indices", data=mat_csc.indices, compression="gzip")
        grp.create_dataset("indptr", data=mat_csc.indptr, compression="gzip")
        grp.attrs["shape"] = np.array(mat_csc.shape, dtype=np.int32)
        feat = grp.create_group("features")
        feat.create_dataset("id", data=np.array(rownames, dtype="S"))
        feat.create_dataset("name", data=np.array(rownames, dtype="S"))
        feat.create_dataset("feature_type",
                            data=np.array(["Gene Expression"] * len(rownames), dtype="S"))
        grp.create_dataset("barcodes", data=np.array(colnames, dtype="S"))


def write_npz(mat_csc, path):
    """Write matrix as scipy npz."""
    if os.path.exists(path):
        os.remove(path)
    ss.save_npz(path, mat_csc)


# ── Operation benchmarks ─────────────────────────────────────────

def bench_random_subsample_1pz(pz_path, col_indices):
    """Random 10% column subsample from .1pz via open_1pz + boolean mask."""
    def op():
        pz = sp.open_1pz(pz_path)
        mask = np.zeros(pz.shape[1], dtype=bool)
        mask[col_indices] = True
        sub = pz[:, mask]  # Returns scipy CSC
        out = sub.copy()   # Deep copy
        return out
    return timer(op)


def bench_random_subsample_h5ad(h5ad_path, col_indices):
    """Random 10% column subsample from H5AD via backed mode.
    AnnData stores cells×genes, so col_indices → obs indices."""
    sorted_idx = np.sort(col_indices)  # AnnData backed requires sorted
    def op():
        a = ad.read_h5ad(h5ad_path, backed="r")
        sub = a[sorted_idx].to_memory()
        mat = sub.X.copy()  # Deep copy of the sparse matrix
        a.file.close()
        return mat
    return timer(op)


def bench_random_subsample_10x(h5_path, mat_shape, col_indices):
    """Random 10% column subsample from 10x H5 — must read full, then slice."""
    sorted_idx = np.sort(col_indices)
    def op():
        with h5py.File(h5_path, "r") as f:
            grp = f["matrix"]
            data = grp["data"][:]
            indices = grp["indices"][:]
            indptr = grp["indptr"][:]
            shape = grp.attrs["shape"]
        full = ss.csc_matrix((data, indices, indptr), shape=shape)
        sub = full[:, sorted_idx].copy()
        return sub
    return timer(op)


def bench_random_subsample_npz(npz_path, col_indices):
    """Random 10% column subsample from scipy npz — full load then slice."""
    sorted_idx = np.sort(col_indices)
    def op():
        full = ss.load_npz(npz_path)
        sub = full[:, sorted_idx].copy()
        return sub
    return timer(op)


def bench_contiguous_slice_1pz(pz_path, col_start, col_end):
    """Contiguous column-range read from .1pz using read_1pz_columns."""
    def op():
        sub = sp.read_1pz_columns(pz_path, col_start, col_end)
        out = sub.copy()
        return out
    return timer(op)


def bench_contiguous_slice_h5ad(h5ad_path, col_start, col_end):
    """Contiguous column-range read from H5AD backed mode."""
    def op():
        a = ad.read_h5ad(h5ad_path, backed="r")
        sub = a[col_start:col_end].to_memory()
        mat = sub.X.copy()
        a.file.close()
        return mat
    return timer(op)


def bench_contiguous_slice_10x(h5_path, mat_shape, col_start, col_end):
    """Contiguous range from 10x H5 — must read full, then slice."""
    def op():
        with h5py.File(h5_path, "r") as f:
            grp = f["matrix"]
            data = grp["data"][:]
            indices = grp["indices"][:]
            indptr = grp["indptr"][:]
            shape = grp.attrs["shape"]
        full = ss.csc_matrix((data, indices, indptr), shape=shape)
        sub = full[:, col_start:col_end].copy()
        return sub
    return timer(op)


def bench_contiguous_slice_npz(npz_path, col_start, col_end):
    """Contiguous range from scipy npz — full load then slice."""
    def op():
        full = ss.load_npz(npz_path)
        sub = full[:, col_start:col_end].copy()
        return sub
    return timer(op)


def bench_normalize_1pz(pz_path):
    """Read with on-the-fly log-normalization."""
    def op():
        return sp.read_1pz(pz_path, normalize=True, scale=10000.0)
    return timer(op)


def bench_normalize_h5ad(h5ad_path):
    """Read H5AD + scanpy-style log-normalization."""
    import scanpy as sc
    def op():
        adata = ad.read_h5ad(h5ad_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata.X
    return timer(op)


# ── Main benchmark loop ──────────────────────────────────────────

def main():
    tmpdir = tempfile.mkdtemp(prefix="sp_ops_bench_")
    print(f"Temp directory: {tmpdir}")
    print(f"Benchmarking {len(DATASETS)} datasets, {FRAC*100:.0f}% subsample, seed={SEED}")

    all_results = []
    rng = np.random.RandomState(SEED)

    for gse_id, meta in DATASETS.items():
        pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if not os.path.isfile(pz_path):
            print(f"\n  {gse_id}: counts.1pz not found, skipping")
            continue

        pz_size = os.path.getsize(pz_path)
        print(f"\n{'='*60}")
        print(f"Dataset: {gse_id} ({meta['protocol']}, {meta['notes']})")

        # Get dimensions without full read
        info = sp.info_1pz(pz_path)
        nrows = info["m"]
        ncols = info["n"]
        nnz = info["nnz"]
        raw_int32_bytes = nnz * 8 + (ncols + 1) * 4  # data+indices + indptr

        print(f"  Shape: {nrows:,} × {ncols:,}, NNZ: {nnz:,}")

        # Generate random 10% column indices
        n_sample = max(1, int(ncols * FRAC))
        col_indices = rng.choice(ncols, size=n_sample, replace=False)
        col_indices_sorted = np.sort(col_indices)

        # Contiguous 10% from middle
        mid = ncols // 2
        half = n_sample // 2
        col_start = max(0, mid - half)
        col_end = min(ncols, col_start + n_sample)

        result = {
            "gse_id": gse_id,
            "protocol": meta["protocol"],
            "nrows": nrows,
            "ncols": ncols,
            "nnz": int(nnz),
            "pz_size_mb": pz_size / 1e6,
            "n_sample": n_sample,
            "raw_int32_bytes": raw_int32_bytes,
        }

        # ── Prepare format files (skip for large datasets) ──
        skip_heavy = nnz > 100_000_000  # >100M nnz
        h5ad_path = os.path.join(tmpdir, f"{gse_id}.h5ad")
        h5_path = os.path.join(tmpdir, f"{gse_id}.h5")
        npz_path = os.path.join(tmpdir, f"{gse_id}.npz")

        if not skip_heavy:
            print(f"  Preparing format files...")
            mat = sp.read_1pz(pz_path)
            csc = mat.tocsc()
            csc.data = csc.data.astype(np.int32)
            rownames = getattr(mat, 'rownames', None) or [f"gene_{i}" for i in range(nrows)]
            colnames = getattr(mat, 'colnames', None) or [f"cell_{i}" for i in range(ncols)]
            write_h5ad(csc, list(rownames), list(colnames), h5ad_path)
            write_10x_h5(csc, list(rownames), list(colnames), h5_path)
            write_npz(csc, npz_path)
            del mat, csc
            gc.collect()
            print(f"  Files ready")

        # ── 1. Random 10% subsample ──
        print(f"  --- Random {FRAC*100:.0f}% subsample ({n_sample:,} cols) ---")

        # .1pz
        try:
            t, sub = bench_random_subsample_1pz(pz_path, col_indices)
            result["rand_1pz_s"] = t
            print(f"    .1pz:    {t*1000:8.1f} ms  ({sub.shape[1]:,} cols, {sub.nnz:,} nnz)")
            del sub
        except Exception as e:
            print(f"    .1pz FAILED: {e}")
            result["rand_1pz_s"] = None

        if not skip_heavy:
            # H5AD
            try:
                t, sub = bench_random_subsample_h5ad(h5ad_path, col_indices_sorted)
                result["rand_h5ad_s"] = t
                print(f"    H5AD:    {t*1000:8.1f} ms")
                del sub
            except Exception as e:
                print(f"    H5AD FAILED: {e}")
                result["rand_h5ad_s"] = None

            # 10x H5
            try:
                t, sub = bench_random_subsample_10x(h5_path, (nrows, ncols), col_indices_sorted)
                result["rand_10x_s"] = t
                print(f"    10x H5:  {t*1000:8.1f} ms")
                del sub
            except Exception as e:
                print(f"    10x H5 FAILED: {e}")
                result["rand_10x_s"] = None

            # scipy npz
            try:
                t, sub = bench_random_subsample_npz(npz_path, col_indices_sorted)
                result["rand_npz_s"] = t
                print(f"    npz:     {t*1000:8.1f} ms")
                del sub
            except Exception as e:
                print(f"    npz FAILED: {e}")
                result["rand_npz_s"] = None
        else:
            result["rand_h5ad_s"] = None
            result["rand_10x_s"] = None
            result["rand_npz_s"] = None

        # ── 2. Contiguous 10% column-range ──
        print(f"  --- Contiguous 10% slice (cols {col_start}–{col_end}) ---")

        try:
            t, sub = bench_contiguous_slice_1pz(pz_path, col_start, col_end)
            result["contig_1pz_s"] = t
            print(f"    .1pz:    {t*1000:8.1f} ms  ({sub.shape[1]:,} cols)")
            del sub
        except Exception as e:
            print(f"    .1pz FAILED: {e}")
            result["contig_1pz_s"] = None

        if not skip_heavy:
            try:
                t, sub = bench_contiguous_slice_h5ad(h5ad_path, col_start, col_end)
                result["contig_h5ad_s"] = t
                print(f"    H5AD:    {t*1000:8.1f} ms")
                del sub
            except Exception as e:
                print(f"    H5AD FAILED: {e}")
                result["contig_h5ad_s"] = None

            try:
                t, sub = bench_contiguous_slice_10x(h5_path, (nrows, ncols), col_start, col_end)
                result["contig_10x_s"] = t
                print(f"    10x H5:  {t*1000:8.1f} ms")
                del sub
            except Exception as e:
                print(f"    10x H5 FAILED: {e}")
                result["contig_10x_s"] = None

            try:
                t, sub = bench_contiguous_slice_npz(npz_path, col_start, col_end)
                result["contig_npz_s"] = t
                print(f"    npz:     {t*1000:8.1f} ms")
                del sub
            except Exception as e:
                print(f"    npz FAILED: {e}")
                result["contig_npz_s"] = None
        else:
            result["contig_h5ad_s"] = None
            result["contig_10x_s"] = None
            result["contig_npz_s"] = None

        # ── 3. Normalized read ──
        print(f"  --- Normalized read (log1p, scale=10k) ---")

        try:
            t, _ = bench_normalize_1pz(pz_path)
            result["norm_1pz_s"] = t
            print(f"    .1pz:    {t*1000:8.1f} ms")
        except Exception as e:
            print(f"    .1pz FAILED: {e}")
            result["norm_1pz_s"] = None

        if not skip_heavy:
            try:
                t, _ = bench_normalize_h5ad(h5ad_path)
                result["norm_h5ad_s"] = t
                print(f"    H5AD:    {t*1000:8.1f} ms")
            except Exception as e:
                print(f"    H5AD FAILED: {e}")
                result["norm_h5ad_s"] = None
        else:
            result["norm_h5ad_s"] = None

        all_results.append(result)

        # Cleanup temp files to save disk
        for p in [h5ad_path, h5_path, npz_path]:
            if os.path.exists(p):
                os.remove(p)
        gc.collect()

    # ── Write JSON results ──
    out_path = os.path.join(SCRIPT_DIR, "operations_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")

    # ── Write CSV for R plotting ──
    csv_path = os.path.join(SCRIPT_DIR, "operations_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gse_id", "protocol", "nnz", "ncols", "n_sample",
                     "operation", "format", "time_s"])
        for r in all_results:
            base = [r["gse_id"], r["protocol"], r["nnz"], r["ncols"], r["n_sample"]]
            # Random subsample
            for fmt, key in [(".1pz", "rand_1pz_s"), ("H5AD", "rand_h5ad_s"),
                              ("10x H5", "rand_10x_s"), ("npz", "rand_npz_s")]:
                if r.get(key) is not None:
                    w.writerow(base + ["random_10pct", fmt, f"{r[key]:.6f}"])
            # Contiguous slice
            for fmt, key in [(".1pz", "contig_1pz_s"), ("H5AD", "contig_h5ad_s"),
                              ("10x H5", "contig_10x_s"), ("npz", "contig_npz_s")]:
                if r.get(key) is not None:
                    w.writerow(base + ["contiguous_10pct", fmt, f"{r[key]:.6f}"])
            # Normalized read
            for fmt, key in [(".1pz", "norm_1pz_s"), ("H5AD", "norm_h5ad_s")]:
                if r.get(key) is not None:
                    w.writerow(base + ["normalize", fmt, f"{r[key]:.6f}"])
    print(f"Wrote {csv_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    for r in all_results:
        gse = r["gse_id"]
        nnz = r["nnz"]
        rand_pz = r.get("rand_1pz_s")
        rand_h5 = r.get("rand_h5ad_s")
        contig_pz = r.get("contig_1pz_s")
        contig_h5 = r.get("contig_h5ad_s")
        norm_pz = r.get("norm_1pz_s")
        norm_h5 = r.get("norm_h5ad_s")
        print(f"  {gse} ({nnz/1e6:.0f}M nnz):")
        if rand_pz and rand_h5:
            print(f"    Random 10%:   .1pz {rand_pz*1000:.0f}ms  H5AD {rand_h5*1000:.0f}ms  ({rand_h5/rand_pz:.1f}x)")
        if contig_pz and contig_h5:
            print(f"    Contig 10%:   .1pz {contig_pz*1000:.0f}ms  H5AD {contig_h5*1000:.0f}ms  ({contig_h5/contig_pz:.1f}x)")
        if norm_pz and norm_h5:
            print(f"    Normalize:    .1pz {norm_pz*1000:.0f}ms  H5AD {norm_h5*1000:.0f}ms  ({norm_h5/norm_pz:.1f}x)")


if __name__ == "__main__":
    main()
