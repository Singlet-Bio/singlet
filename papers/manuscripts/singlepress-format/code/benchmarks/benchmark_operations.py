#!/usr/bin/env python3
"""
Benchmark common operations across single-cell data formats.
Produces data for Figure 2 panels (c) and (d).

Operations benchmarked:
  1. Random 10% column subsample (uniform random, seed=42)
  2. Contiguous 10% column-range read from middle
  3. On-the-fly log-normalization during read

Formats:
  .1pz   — singlepress (open_1pz slicing, read_1pz_columns)
  H5AD   — anndata (backed mode for slicing, full read for norm)
  10x H5 — h5py CellRanger format
  scipy npz — scipy.sparse baseline

Re-encodes each .1pz dataset with the CURRENT codec before benchmarking.

Output: code/data/operations_benchmark.csv

Usage:
    srun --time=120 --mem=64G --cpus-per-task=8 bash -c \\
        'source /mnt/home/debruinz/venv/bin/activate && cd /tmp && \\
         python3 -u <this_script>'
"""

import csv
import gc
import json
import os
import sys
import tempfile
import time

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if "" in sys.path:
    sys.path.remove("")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as ss
import singlepress as sp

assert hasattr(sp, "write_1pz"), "singlepress not loaded correctly"

# ── Configuration ────────────────────────────────────────────────
SEED = 42
FRAC = 0.10          # 10% subsample
N_TRIALS = 3
WARMUP = 1
QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TMPFS = "/dev/shm"

# 15 datasets spanning a range of sizes
DATASETS = {
    "GSE210261": {"protocol": "BD Rhapsody",      "notes": "small ~4.6M nnz"},
    "GSE123662": {"protocol": "10x Chromium v2",   "notes": "zebrafish ~7.5M nnz"},
    "GSE290216": {"protocol": "10x Chromium v2",   "notes": "human ~11M nnz"},
    "GSE209597": {"protocol": "10x Chromium v3",   "notes": "human ~13.2M nnz"},
    "GSE176269": {"protocol": "Seq-Well",           "notes": "human ~14M nnz"},
    "GSE216064": {"protocol": "10x Chromium v3",   "notes": "human ~27M nnz"},
    "GSE304926": {"protocol": "10x Chromium v4",   "notes": "frog ~29M nnz"},
    "GSE103976": {"protocol": "10x Chromium v3",   "notes": "mouse ~32M nnz"},
    "GSE297773": {"protocol": "10x Chromium v3",   "notes": "rat ~40M nnz"},
    "GSE248556": {"protocol": "10x Chromium v3 5'", "notes": "human ~47M nnz"},
    "GSE136679": {"protocol": "Seq-Well",           "notes": "mouse ~57M nnz"},
    "GSE187438": {"protocol": "10x Chromium v3",   "notes": "human ~70M nnz"},
    "GSE278497": {"protocol": "DNBelab C4",         "notes": "human ~93M nnz"},
    "GSE306871": {"protocol": "DNBelab C4",         "notes": "mouse ~117M nnz"},
    "GSE203171": {"protocol": "10x Chromium v3",   "notes": "mouse ~136M nnz"},
}


def reencode(gse_id):
    """Re-encode a production .1pz with the current codec to tmpfs."""
    src = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    dst = os.path.join(TMPFS, f"ops_bench_{gse_id}.1pz")
    if os.path.exists(dst):
        return dst
    print(f"  Re-encoding {gse_id}...", end="", flush=True)
    mat = sp.read_1pz(src)
    pz = sp.open_1pz(src)
    sp.write_1pz(dst, mat.tocsc(),
                  rownames=list(pz.rownames) if pz.rownames else [],
                  colnames=list(pz.colnames) if pz.colnames else [])
    del mat, pz; gc.collect()
    print(f" {os.path.getsize(dst)/1e6:.1f} MB")
    return dst


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
    gc.collect()
    t0 = time.perf_counter()
    result = func()
    t1 = time.perf_counter()
    return np.median(times), result


def write_h5ad(mat_csc, rownames, colnames, path):
    """Write matrix as H5AD."""
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
        grp.create_dataset("data", data=mat_csc.data.astype(np.int32),
                           compression="gzip")
        grp.create_dataset("indices", data=mat_csc.indices, compression="gzip")
        grp.create_dataset("indptr", data=mat_csc.indptr, compression="gzip")
        grp.attrs["shape"] = np.array(mat_csc.shape, dtype=np.int32)
        feat = grp.create_group("features")
        feat.create_dataset("id", data=np.array(rownames, dtype="S"))
        feat.create_dataset("name", data=np.array(rownames, dtype="S"))
        feat.create_dataset("feature_type",
                            data=np.array(["Gene Expression"] * len(rownames),
                                          dtype="S"))
        grp.create_dataset("barcodes", data=np.array(colnames, dtype="S"))


# ── Operation benchmarks ─────────────────────────────────────────

def bench_random_subsample_1pz(pz_path, col_indices):
    def op():
        pz = sp.open_1pz(pz_path)
        mask = np.zeros(pz.shape[1], dtype=bool)
        mask[col_indices] = True
        return pz[:, mask].copy()
    return timer(op)


def bench_random_subsample_h5ad(h5ad_path, col_indices):
    sorted_idx = np.sort(col_indices)
    def op():
        a = ad.read_h5ad(h5ad_path, backed="r")
        sub = a[sorted_idx].to_memory()
        mat = sub.X.copy()
        a.file.close()
        return mat
    return timer(op)


def bench_random_subsample_10x(h5_path, mat_shape, col_indices):
    sorted_idx = np.sort(col_indices)
    def op():
        with h5py.File(h5_path, "r") as f:
            grp = f["matrix"]
            data = grp["data"][:]
            indices = grp["indices"][:]
            indptr = grp["indptr"][:]
            shape = grp.attrs["shape"]
        full = ss.csc_matrix((data, indices, indptr), shape=shape)
        return full[:, sorted_idx].copy()
    return timer(op)


def bench_random_subsample_npz(npz_path, col_indices):
    sorted_idx = np.sort(col_indices)
    def op():
        full = ss.load_npz(npz_path)
        return full[:, sorted_idx].copy()
    return timer(op)


def bench_contiguous_slice_1pz(pz_path, col_start, col_end):
    def op():
        return sp.read_1pz_columns(pz_path, col_start, col_end).copy()
    return timer(op)


def bench_contiguous_slice_h5ad(h5ad_path, col_start, col_end):
    def op():
        a = ad.read_h5ad(h5ad_path, backed="r")
        sub = a[col_start:col_end].to_memory()
        mat = sub.X.copy()
        a.file.close()
        return mat
    return timer(op)


def bench_contiguous_slice_10x(h5_path, mat_shape, col_start, col_end):
    def op():
        with h5py.File(h5_path, "r") as f:
            grp = f["matrix"]
            data = grp["data"][:]
            indices = grp["indices"][:]
            indptr = grp["indptr"][:]
            shape = grp.attrs["shape"]
        full = ss.csc_matrix((data, indices, indptr), shape=shape)
        return full[:, col_start:col_end].copy()
    return timer(op)


def bench_contiguous_slice_npz(npz_path, col_start, col_end):
    def op():
        full = ss.load_npz(npz_path)
        return full[:, col_start:col_end].copy()
    return timer(op)


def bench_normalize_1pz(pz_path):
    def op():
        return sp.read_1pz(pz_path, normalize=True, scale=10000.0)
    return timer(op)


def bench_normalize_h5ad(h5ad_path):
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
    print(f"Benchmarking {len(DATASETS)} datasets, {FRAC*100:.0f}% subsample, "
          f"seed={SEED}")
    print(f"singlepress {getattr(sp, '__version__', 'dev')}")

    all_results = []
    rng = np.random.RandomState(SEED)

    for gse_id, meta in DATASETS.items():
        src_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if not os.path.isfile(src_path):
            print(f"\n  {gse_id}: counts.1pz not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {gse_id} ({meta['protocol']}, {meta['notes']})")

        # Re-encode with current codec
        pz_path = reencode(gse_id)
        pz_size = os.path.getsize(pz_path)

        info = sp.info_1pz(pz_path)
        nrows = info["m"]
        ncols = info["n"]
        nnz = info["nnz"]
        raw_int32_bytes = nnz * 8 + (ncols + 1) * 4

        print(f"  Shape: {nrows:,} x {ncols:,}, NNZ: {nnz:,}")

        # Column indices for subsample
        n_sample = max(1, int(ncols * FRAC))
        col_indices = rng.choice(ncols, size=n_sample, replace=False)
        col_indices_sorted = np.sort(col_indices)

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

        # Prepare format files (skip for large datasets)
        skip_heavy = nnz > 100_000_000
        h5ad_path = os.path.join(tmpdir, f"{gse_id}.h5ad")
        h5_path = os.path.join(tmpdir, f"{gse_id}.h5")
        npz_path = os.path.join(tmpdir, f"{gse_id}.npz")

        if not skip_heavy:
            print(f"  Preparing format files...")
            mat = sp.read_1pz(pz_path)
            csc = mat.tocsc()
            csc.data = csc.data.astype(np.int32)
            rownames = getattr(mat, "rownames", None) or \
                [f"gene_{i}" for i in range(nrows)]
            colnames = getattr(mat, "colnames", None) or \
                [f"cell_{i}" for i in range(ncols)]
            write_h5ad(csc, list(rownames), list(colnames), h5ad_path)
            write_10x_h5(csc, list(rownames), list(colnames), h5_path)
            ss.save_npz(npz_path, csc)
            del mat, csc; gc.collect()
            print(f"  Files ready")

        # ── 1. Random 10% subsample ──
        print(f"  --- Random {FRAC*100:.0f}% subsample ({n_sample:,} cols) ---")

        try:
            t, sub = bench_random_subsample_1pz(pz_path, col_indices)
            result["rand_1pz_s"] = t
            print(f"    .1pz:    {t*1000:8.1f} ms ({sub.shape[1]:,} cols)")
            del sub
        except Exception as e:
            print(f"    .1pz FAILED: {e}")
            result["rand_1pz_s"] = None

        if not skip_heavy:
            for name, bench_fn, args, key in [
                ("H5AD", bench_random_subsample_h5ad,
                 (h5ad_path, col_indices_sorted), "rand_h5ad_s"),
                ("10x H5", bench_random_subsample_10x,
                 (h5_path, (nrows, ncols), col_indices_sorted), "rand_10x_s"),
                ("npz", bench_random_subsample_npz,
                 (npz_path, col_indices_sorted), "rand_npz_s"),
            ]:
                try:
                    t, sub = bench_fn(*args)
                    result[key] = t
                    print(f"    {name:8s} {t*1000:8.1f} ms")
                    del sub
                except Exception as e:
                    print(f"    {name} FAILED: {e}")
                    result[key] = None
        else:
            result["rand_h5ad_s"] = None
            result["rand_10x_s"] = None
            result["rand_npz_s"] = None

        # ── 2. Contiguous 10% column-range ──
        print(f"  --- Contiguous 10% slice (cols {col_start}-{col_end}) ---")

        try:
            t, sub = bench_contiguous_slice_1pz(pz_path, col_start, col_end)
            result["contig_1pz_s"] = t
            print(f"    .1pz:    {t*1000:8.1f} ms ({sub.shape[1]:,} cols)")
            del sub
        except Exception as e:
            print(f"    .1pz FAILED: {e}")
            result["contig_1pz_s"] = None

        if not skip_heavy:
            for name, bench_fn, args, key in [
                ("H5AD", bench_contiguous_slice_h5ad,
                 (h5ad_path, col_start, col_end), "contig_h5ad_s"),
                ("10x H5", bench_contiguous_slice_10x,
                 (h5_path, (nrows, ncols), col_start, col_end), "contig_10x_s"),
                ("npz", bench_contiguous_slice_npz,
                 (npz_path, col_start, col_end), "contig_npz_s"),
            ]:
                try:
                    t, sub = bench_fn(*args)
                    result[key] = t
                    print(f"    {name:8s} {t*1000:8.1f} ms")
                    del sub
                except Exception as e:
                    print(f"    {name} FAILED: {e}")
                    result[key] = None
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

        # Cleanup
        for p in [h5ad_path, h5_path, npz_path]:
            if os.path.exists(p):
                os.remove(p)
        staged = os.path.join(TMPFS, f"ops_bench_{gse_id}.1pz")
        if os.path.exists(staged):
            os.remove(staged)
        gc.collect()

    # ── Write CSV for R plotting ──
    os.makedirs(CODE_DATA_DIR, exist_ok=True)
    csv_path = os.path.join(CODE_DATA_DIR, "operations_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gse_id", "protocol", "nnz", "ncols", "n_sample",
                     "operation", "format", "time_s"])
        for r in all_results:
            base = [r["gse_id"], r["protocol"], r["nnz"], r["ncols"],
                    r["n_sample"]]
            for fmt, key in [(".1pz", "rand_1pz_s"), ("H5AD", "rand_h5ad_s"),
                              ("10x H5", "rand_10x_s"), ("npz", "rand_npz_s")]:
                if r.get(key) is not None:
                    w.writerow(base + ["random_10pct", fmt, f"{r[key]:.6f}"])
            for fmt, key in [(".1pz", "contig_1pz_s"), ("H5AD", "contig_h5ad_s"),
                              ("10x H5", "contig_10x_s"), ("npz", "contig_npz_s")]:
                if r.get(key) is not None:
                    w.writerow(base + ["contiguous_10pct", fmt, f"{r[key]:.6f}"])
            for fmt, key in [(".1pz", "norm_1pz_s"), ("H5AD", "norm_h5ad_s")]:
                if r.get(key) is not None:
                    w.writerow(base + ["normalize", fmt, f"{r[key]:.6f}"])
    print(f"\nWrote {csv_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    for r in all_results:
        gse = r["gse_id"]
        nnz = r["nnz"]
        rp, rh = r.get("rand_1pz_s"), r.get("rand_h5ad_s")
        cp, ch = r.get("contig_1pz_s"), r.get("contig_h5ad_s")
        np_, nh = r.get("norm_1pz_s"), r.get("norm_h5ad_s")
        print(f"  {gse} ({nnz/1e6:.0f}M nnz):")
        if rp and rh:
            print(f"    Random:   .1pz {rp*1000:.0f}ms  H5AD {rh*1000:.0f}ms  "
                  f"({rh/rp:.1f}x)")
        if cp and ch:
            print(f"    Contig:   .1pz {cp*1000:.0f}ms  H5AD {ch*1000:.0f}ms  "
                  f"({ch/cp:.1f}x)")
        if np_ and nh:
            print(f"    Norm:     .1pz {np_*1000:.0f}ms  H5AD {nh*1000:.0f}ms  "
                  f"({nh/np_:.1f}x)")


if __name__ == "__main__":
    main()
