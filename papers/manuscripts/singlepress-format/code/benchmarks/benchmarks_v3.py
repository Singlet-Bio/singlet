#!/usr/bin/env python3
"""
Expanded benchmark script for SinglePress .1pz format paper (v3).
Tests 25 datasets across 8 droplet-based protocols, 7 species.
Includes USA-resolved vs summed analysis and R format benchmarks.

Usage:
    cd /tmp && srun --time=120 --mem=64G --cpus-per-task=8 \
        python3 -u /path/to/benchmarks_v3.py

Outputs: benchmark_results_v3.json (in script directory)
"""

import json
import os
import resource
import sys
import tempfile
import time
import gzip
import shutil
import traceback
import subprocess

# Fix namespace-package shadowing
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import anndata as ad
import h5py
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import singlepress as sp

assert hasattr(sp, "write_1pz"), f"singlepress not loaded correctly"

# ── Configuration ────────────────────────────────────────────────
N_TRIALS = 3
WARMUP = 1
WRITE_TRIALS = 3
WRITE_WARMUP = 1

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gene rows → species mapping (all USA-resolved: nrows = 3 × n_genes)
SPECIES_MAP = {
    115818: ("Homo sapiens", "GRCh38", 38606),
    171540: ("Mus musculus", "GRCm39", 57180),
    97560: ("Danio rerio", "GRCz11", 32520),
    91686: ("Rattus norvegicus", "mRatBN7.2", 30562),
    72834: ("Drosophila melanogaster", "BDGP6", 24278),
    106296: ("Macaca mulatta", "Mmul_10", 35432),
    90324: ("Gallus gallus", "bGalGal1", 30108),
    73842: ("Xenopus laevis", "v10.1", 24614),
}

# 25 datasets: diverse protocols, species, sizes
DATASETS = {
    # Human 10x Chromium v3 (3' GEX) — 5 datasets spanning 2 orders of magnitude
    "GSE209597": {"protocol": "10x Chromium v3", "notes": "small human 10xv3"},
    "GSE216064": {"protocol": "10x Chromium v3", "notes": "medium-small human 10xv3"},
    "GSE187438": {"protocol": "10x Chromium v3", "notes": "medium human 10xv3"},
    "GSE204704": {"protocol": "10x Chromium v3", "notes": "large human 10xv3"},
    "GSE288147": {"protocol": "10x Chromium v3", "notes": "very large human 10xv3"},
    # Human 10x v2
    "GSE290216": {"protocol": "10x Chromium v2", "notes": "human 10xv2"},
    # Human 10x 5' GEX
    "GSE248556": {"protocol": "10x Chromium v3 5'", "notes": "human 5prime"},
    # 10x v4
    "GSE286044": {"protocol": "10x Chromium v4", "notes": "mouse 10xv4"},
    # Mouse 10x v3
    "GSE150338": {"protocol": "10x Chromium v3", "notes": "mouse 10xv3 small"},
    "GSE103976": {"protocol": "10x Chromium v3", "notes": "mouse 10xv3 medium"},
    "GSE203171": {"protocol": "10x Chromium v3", "notes": "mouse 10xv3 large"},
    # Human Seq-Well
    "GSE176269": {"protocol": "Seq-Well", "notes": "human seqwell"},
    # Mouse Seq-Well
    "GSE136679": {"protocol": "Seq-Well", "notes": "mouse seqwell"},
    # Human Drop-seq
    "GSE155409": {"protocol": "Drop-seq", "notes": "human dropseq"},
    # Human inDrop
    "GSE174189": {"protocol": "inDrop", "notes": "human indrop"},
    # Human BD Rhapsody
    "GSE210261": {"protocol": "BD Rhapsody", "notes": "human bd_rhapsody"},
    # Human DNBelab
    "GSE278497": {"protocol": "DNBelab C4", "notes": "human dnbelab"},
    # Mouse DNBelab
    "GSE306871": {"protocol": "DNBelab C4", "notes": "mouse dnbelab"},
    # Danio rerio
    "GSE123662": {"protocol": "10x Chromium v2", "notes": "zebrafish"},
    "GSE241296": {"protocol": "10x Chromium v3", "notes": "zebrafish"},
    # Rattus norvegicus
    "GSE297773": {"protocol": "10x Chromium v3", "notes": "rat"},
    # Drosophila melanogaster
    "GSE126139": {"protocol": "10x Chromium v2", "notes": "fruit fly"},
    # Macaca mulatta
    "GSE245419": {"protocol": "10x Chromium v3", "notes": "rhesus macaque"},
    # Gallus gallus
    "GSE249652": {"protocol": "10x Chromium v3", "notes": "chicken, very large"},
    # Xenopus laevis
    "GSE304926": {"protocol": "10x Chromium v4", "notes": "frog 10xv4"},
}


def timer(func, n_trials=N_TRIALS, warmup=WARMUP):
    times = []
    for i in range(warmup + n_trials):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
    return np.median(times), result


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def file_size(path):
    if os.path.isdir(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total
    return os.path.getsize(path)


# ── Format Writers ───────────────────────────────────────────────

def write_1pz(mat, rownames, colnames, path):
    csc = mat.tocsc()
    if os.path.exists(path):
        os.remove(path)
    sp.write_1pz(path, csc, rownames=rownames, colnames=colnames)


def write_h5ad(mat, rownames, colnames, path, compression="gzip"):
    import pandas as pd
    if os.path.exists(path):
        os.remove(path)
    adata = ad.AnnData(
        X=mat.T.tocsr(),
        obs=pd.DataFrame(index=colnames),
        var=pd.DataFrame(index=rownames),
    )
    adata.write_h5ad(path, compression=compression)


def write_10x_h5(mat, rownames, colnames, path):
    csc = mat.tocsc()
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("data", data=csc.data.astype(np.int32), compression="gzip")
        grp.create_dataset("indices", data=csc.indices, compression="gzip")
        grp.create_dataset("indptr", data=csc.indptr, compression="gzip")
        grp.attrs["shape"] = np.array(csc.shape, dtype=np.int32)
        feat = grp.create_group("features")
        feat.create_dataset("id", data=np.array(rownames, dtype="S"))
        feat.create_dataset("name", data=np.array(rownames, dtype="S"))
        feat.create_dataset("feature_type",
                            data=np.array(["Gene Expression"] * len(rownames), dtype="S"))
        grp.create_dataset("barcodes", data=np.array(colnames, dtype="S"))


def write_npz(mat, rownames, colnames, path):
    if os.path.exists(path):
        os.remove(path)
    ss.save_npz(path, mat.tocsc())


def write_rds(mat, rownames, colnames, path):
    """Write as R dgCMatrix via subprocess R."""
    csc = mat.tocsc()
    # Write components as binary files for R to read
    prefix = path + ".tmp"
    np.save(prefix + "_i.npy", csc.indices.astype(np.int32))
    np.save(prefix + "_p.npy", csc.indptr.astype(np.int32))
    np.save(prefix + "_x.npy", csc.data.astype(np.float64))
    with open(prefix + "_rn.txt", "w") as f:
        f.write("\n".join(str(r) for r in rownames))
    with open(prefix + "_cn.txt", "w") as f:
        f.write("\n".join(str(c) for c in colnames))

    r_script = f"""
suppressMessages(library(Matrix))
# Read components saved by Python
i_raw <- readBin("{prefix}_i.npy", "raw", file.info("{prefix}_i.npy")$size)
p_raw <- readBin("{prefix}_p.npy", "raw", file.info("{prefix}_p.npy")$size)
x_raw <- readBin("{prefix}_x.npy", "raw", file.info("{prefix}_x.npy")$size)

# NumPy .npy format: 128-byte header then raw data
# Skip header (find header size from byte 9-10 as uint16 LE + 10)
parse_npy_int32 <- function(raw_bytes) {{
  # Header: magic(6) + version(2) + header_len(2) + header_str
  header_len <- as.integer(raw_bytes[9]) + 256L * as.integer(raw_bytes[10])
  offset <- 10L + header_len
  data_bytes <- raw_bytes[(offset+1):length(raw_bytes)]
  readBin(data_bytes, "integer", n=length(data_bytes)/4, size=4, endian="little")
}}
parse_npy_float64 <- function(raw_bytes) {{
  header_len <- as.integer(raw_bytes[9]) + 256L * as.integer(raw_bytes[10])
  offset <- 10L + header_len
  data_bytes <- raw_bytes[(offset+1):length(raw_bytes)]
  readBin(data_bytes, "double", n=length(data_bytes)/8, size=8, endian="little")
}}

i <- parse_npy_int32(i_raw)
p <- parse_npy_int32(p_raw)
x <- parse_npy_float64(x_raw)

rn <- readLines("{prefix}_rn.txt")
cn <- readLines("{prefix}_cn.txt")

m <- new("dgCMatrix", i=i, p=p, x=x,
         Dim=c({csc.shape[0]}L, {csc.shape[1]}L),
         Dimnames=list(rn, cn))
saveRDS(m, "{path}", compress="gzip")
cat("OK\\n")
"""
    r_file = prefix + ".R"
    with open(r_file, "w") as f:
        f.write(r_script)
    result = subprocess.run(
        ["bash", "-c", f"module load r/4.5.2 && Rscript {r_file}"],
        capture_output=True, timeout=600)
    # Cleanup
    for suffix in ["_i.npy", "_p.npy", "_x.npy", "_rn.txt", "_cn.txt", ".R"]:
        p = prefix + suffix
        if os.path.exists(p):
            os.remove(p)
    if result.returncode != 0:
        raise RuntimeError(f"R write failed: {result.stderr.decode()[:200]}")


def read_rds(path, n_trials=5):
    """Read RDS dgCMatrix via single R subprocess, return median wall-clock time."""
    r_script = f"""
suppressMessages(library(Matrix))
# Warmup
m <- readRDS("{path}")
rm(m); invisible(gc(verbose=FALSE, reset=FALSE))

# Timed trials
times <- numeric({n_trials})
for (i in 1:{n_trials}) {{
  invisible(gc(verbose=FALSE, reset=FALSE))
  t0 <- proc.time()
  m <- readRDS("{path}")
  t1 <- proc.time()
  times[i] <- (t1 - t0)[3]
  rm(m); invisible(gc(verbose=FALSE, reset=FALSE))
}}
cat(sprintf("%.6f\\n", median(times)))
"""
    r_file = path + ".read.R"
    with open(r_file, "w") as f:
        f.write(r_script)
    result = subprocess.run(
        ["bash", "-c", f"module load r/4.5.2 && Rscript {r_file}"],
        capture_output=True, timeout=600)
    os.remove(r_file)
    if result.returncode != 0:
        raise RuntimeError(f"R read failed: {result.stderr.decode()[:200]}")
    # Parse last numeric line (gc output may precede it)
    for line in reversed(result.stdout.decode().strip().split("\n")):
        line = line.strip()
        try:
            return float(line)
        except ValueError:
            continue
    raise RuntimeError(f"No numeric output from R read: {result.stdout.decode()[:200]}")


# ── Format Readers ───────────────────────────────────────────────

def read_1pz(path):
    return sp.read_1pz(path)

def read_h5ad(path):
    return ad.read_h5ad(path)

def read_10x_h5(path):
    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = grp.attrs["shape"]
        return ss.csc_matrix((data, indices, indptr), shape=shape)

def read_npz(path):
    return ss.load_npz(path)


# ── Benchmark Functions ─────────────────────────────────────────

def benchmark_single(gse_id, meta, tmpdir):
    """Benchmark a single dataset (read from original .1pz)."""
    pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    if not os.path.isfile(pz_path):
        print(f"  {gse_id}: counts.1pz not found, skipping")
        return None

    pz_size = os.path.getsize(pz_path)
    print(f"\n{'='*60}")
    print(f"Dataset: {gse_id} ({meta['protocol']}, {meta['notes']})")
    print(f"  .1pz file: {pz_size/1e6:.1f} MB")

    # Read the matrix
    try:
        mat = sp.read_1pz(pz_path)
    except Exception as e:
        print(f"  FAILED to read: {e}")
        return None

    nrows, ncols = mat.shape
    nnz = mat.nnz
    density = nnz / (nrows * ncols) if nrows * ncols > 0 else 0

    # Determine species
    sp_info = SPECIES_MAP.get(nrows, ("Unknown", "Unknown", nrows))
    species, genome, n_genes = sp_info
    is_usa = (nrows % 3 == 0) and (nrows // 3 == n_genes)

    print(f"  Shape: {nrows:,} × {ncols:,} (USA={is_usa}, {n_genes:,} genes)")
    print(f"  NNZ: {nnz:,}, Density: {density:.4%}")
    print(f"  Species: {species} ({genome})")

    csc = mat.tocsc()
    csc.data = csc.data.astype(np.int32)
    rownames = getattr(mat, 'rownames', [f"gene_{i}" for i in range(nrows)])
    colnames = getattr(mat, 'colnames', [f"cell_{i}" for i in range(ncols)])
    if not rownames:
        rownames = [f"gene_{i}" for i in range(nrows)]
    if not colnames:
        colnames = [f"cell_{i}" for i in range(ncols)]

    # int32 CSC baseline: indptr + indices + data (all int32)
    raw_int32_bytes = csc.indptr.nbytes + csc.indices.nbytes + nnz * 4

    result = {
        "gse_id": gse_id,
        "protocol": meta["protocol"],
        "species": species,
        "genome": genome,
        "n_genes": n_genes,
        "nrows": nrows,
        "ncols": ncols,
        "nnz": int(nnz),
        "density": density,
        "is_usa": is_usa,
        "raw_int32_bytes": raw_int32_bytes,
        "pz_file_size": pz_size,
        "formats": {},
    }

    # ── .1pz read from original file ──
    try:
        t_read, _ = timer(lambda: sp.read_1pz(pz_path))
        result["formats"]["1pz"] = {
            "size": pz_size,
            "ratio_vs_int32": raw_int32_bytes / pz_size,
            "read_s": t_read,
            "read_MBps_int32": raw_int32_bytes / t_read / 1e6,
        }
        print(f"  1pz:  {pz_size/1e6:8.1f} MB  ratio={raw_int32_bytes/pz_size:.1f}x  read={t_read*1000:.0f}ms  {raw_int32_bytes/t_read/1e6:.0f} MB/s")
    except Exception as e:
        print(f"  1pz read FAILED: {e}")

    # ── Write .1pz (may fail for large matrices with codec bug) ──
    pz_write_path = os.path.join(tmpdir, f"{gse_id}.1pz")
    try:
        t_write, _ = timer(lambda: write_1pz(csc, list(rownames), list(colnames), pz_write_path),
                           n_trials=WRITE_TRIALS, warmup=WRITE_WARMUP)
        result["formats"]["1pz"]["write_s"] = t_write
        result["formats"]["1pz"]["write_MBps_int32"] = raw_int32_bytes / t_write / 1e6
        print(f"        write={t_write*1000:.0f}ms  {raw_int32_bytes/t_write/1e6:.0f} MB/s")
    except Exception as e:
        print(f"        write FAILED (codec bug): {str(e)[:60]}")
        result["formats"]["1pz"]["write_s"] = None

    # ── H5AD (gzip) — write once for size, time reads only ──
    h5ad_path = os.path.join(tmpdir, f"{gse_id}.h5ad")
    skip_heavy_writes = nnz > 80_000_000  # >80M nnz → skip slow format writes
    try:
        if not skip_heavy_writes:
            write_h5ad(csc, list(rownames), list(colnames), h5ad_path, "gzip")
            sz = file_size(h5ad_path)
            t_read, _ = timer(lambda: read_h5ad(h5ad_path))
            result["formats"]["h5ad_gzip"] = {
                "size": sz, "ratio_vs_int32": raw_int32_bytes / sz,
                "read_s": t_read,
                "read_MBps_int32": raw_int32_bytes / t_read / 1e6,
            }
            print(f"  h5ad: {sz/1e6:8.1f} MB  ratio={raw_int32_bytes/sz:.1f}x  read={t_read*1000:.0f}ms")
        else:
            print(f"  h5ad: skipped (large dataset)")
    except Exception as e:
        print(f"  h5ad FAILED: {e}")

    # ── 10x HDF5 — write once for size, time reads only ──
    h5_path = os.path.join(tmpdir, f"{gse_id}.h5")
    try:
        if not skip_heavy_writes:
            write_10x_h5(csc, list(rownames), list(colnames), h5_path)
            sz = file_size(h5_path)
            t_read, _ = timer(lambda: read_10x_h5(h5_path))
            result["formats"]["10x_h5"] = {
                "size": sz, "ratio_vs_int32": raw_int32_bytes / sz,
                "read_s": t_read,
                "read_MBps_int32": raw_int32_bytes / t_read / 1e6,
            }
            print(f"  10xh5:{sz/1e6:8.1f} MB  ratio={raw_int32_bytes/sz:.1f}x  read={t_read*1000:.0f}ms")
        else:
            print(f"  10xh5:skipped (large dataset)")
    except Exception as e:
        print(f"  10x_h5 FAILED: {e}")

    # ── scipy npz — write once for size, time reads only ──
    npz_path = os.path.join(tmpdir, f"{gse_id}.npz")
    if not skip_heavy_writes:
        try:
            write_npz(csc, list(rownames), list(colnames), npz_path)
            sz = file_size(npz_path)
            t_read, _ = timer(lambda: read_npz(npz_path))
            result["formats"]["npz"] = {
                "size": sz, "ratio_vs_int32": raw_int32_bytes / sz,
                "read_s": t_read,
                "read_MBps_int32": raw_int32_bytes / t_read / 1e6,
            }
            print(f"  npz:  {sz/1e6:8.1f} MB  ratio={raw_int32_bytes/sz:.1f}x  read={t_read*1000:.0f}ms")
        except Exception as e:
            print(f"  npz FAILED: {e}")
    else:
        print(f"  npz:  skipped (large dataset)")

    # ── R RDS dgCMatrix — write once for size, time reads only ──
    rds_path = os.path.join(tmpdir, f"{gse_id}.rds")
    if not skip_heavy_writes:
        try:
            write_rds(csc, list(rownames), list(colnames), rds_path)
            sz = file_size(rds_path)
            t_read = read_rds(rds_path, n_trials=N_TRIALS)
            result["formats"]["rds_dgCMatrix"] = {
                "size": sz, "ratio_vs_int32": raw_int32_bytes / sz,
                "read_s": t_read,
                "read_MBps_int32": raw_int32_bytes / t_read / 1e6,
            }
            print(f"  rds:  {sz/1e6:8.1f} MB  ratio={raw_int32_bytes/sz:.1f}x  read={t_read*1000:.0f}ms")
        except Exception as e:
            print(f"  rds FAILED: {str(e)[:80]}")
    else:
        print(f"  rds:  skipped (large dataset)")

    # ── Partial column read ──
    if ncols >= 100:
        n_partial = 100
        try:
            t_partial_1pz, _ = timer(lambda: sp.read_1pz_columns(pz_path, 0, n_partial))
            t_partial_h5ad = None
            if os.path.exists(h5ad_path):
                def partial_h5ad():
                    a = ad.read_h5ad(h5ad_path, backed="r")
                    chunk = a[0:n_partial].to_memory()
                    a.file.close()
                    return chunk
                t_partial_h5ad, _ = timer(partial_h5ad)

            result["partial_read"] = {
                "n_cols": n_partial,
                "1pz_s": t_partial_1pz,
                "h5ad_s": t_partial_h5ad,
            }
            print(f"  partial({n_partial}): 1pz={t_partial_1pz*1000:.1f}ms" +
                  (f"  h5ad={t_partial_h5ad*1000:.1f}ms" if t_partial_h5ad else ""))
        except Exception as e:
            print(f"  partial FAILED: {e}")

    # ── USA-resolved vs summed analysis ──
    if is_usa:
        try:
            # The full USA matrix
            usa_size = pz_size
            usa_nnz = nnz
            usa_int32_bytes = raw_int32_bytes

            # Sum S+U+A to get total counts matrix (n_genes × ncols)
            genes_per_layer = n_genes
            S = csc[:genes_per_layer, :]
            U = csc[genes_per_layer:2*genes_per_layer, :]
            A = csc[2*genes_per_layer:, :]
            total = (S + U + A).tocsc()
            total.data = total.data.astype(np.int32)
            total_nnz = total.nnz
            total_int32_bytes = total.indptr.nbytes + total.indices.nbytes + total_nnz * 4

            # Write summed .1pz
            summed_path = os.path.join(tmpdir, f"{gse_id}_summed.1pz")
            rn_summed = list(rownames[:genes_per_layer]) if rownames else [f"g{i}" for i in range(genes_per_layer)]
            write_1pz(total, rn_summed, list(colnames), summed_path)
            summed_size = file_size(summed_path)

            # Read summed
            t_read_sum, _ = timer(lambda: sp.read_1pz(summed_path))

            result["usa_analysis"] = {
                "usa_nnz": usa_nnz,
                "usa_int32_bytes": usa_int32_bytes,
                "usa_pz_size": usa_size,
                "usa_ratio": usa_int32_bytes / usa_size,
                "summed_nnz": int(total_nnz),
                "summed_int32_bytes": total_int32_bytes,
                "summed_pz_size": summed_size,
                "summed_ratio": total_int32_bytes / summed_size,
                "summed_read_s": t_read_sum,
                "size_ratio_usa_vs_summed": usa_size / summed_size,
                "nnz_ratio_usa_vs_summed": usa_nnz / total_nnz,
            }
            print(f"  USA analysis:")
            print(f"    USA:    {usa_size/1e6:.1f}MB, {usa_nnz:,} nnz, ratio={usa_int32_bytes/usa_size:.1f}x")
            print(f"    Summed: {summed_size/1e6:.1f}MB, {total_nnz:,} nnz, ratio={total_int32_bytes/summed_size:.1f}x")
            print(f"    Size USA/summed: {usa_size/summed_size:.2f}x")
        except Exception as e:
            print(f"  USA analysis FAILED: {str(e)[:80]}")

    # Cleanup temp files
    for ext in [".1pz", ".h5ad", ".h5", ".npz", ".rds", "_summed.1pz"]:
        p = os.path.join(tmpdir, f"{gse_id}{ext}")
        if os.path.exists(p):
            os.remove(p)

    result["peak_rss_mb"] = peak_rss_mb()
    print(f"  Peak RSS: {result['peak_rss_mb']:.0f} MB")
    return result


def main():
    print(f"SinglePress v3 Benchmark — {len(DATASETS)} datasets")
    print(f"Trials: {N_TRIALS} (read), {WRITE_TRIALS} (write), warmup: {WARMUP}/{WRITE_WARMUP}")
    print(f"singlepress version: {sp.__version__ if hasattr(sp, '__version__') else 'unknown'}")

    # Check R availability
    try:
        r_check = subprocess.run(
            ["bash", "-c", "module load r/4.5.2 && Rscript -e 'library(Matrix); cat(\"OK\")'"],
            capture_output=True, timeout=30)
        has_r = r_check.returncode == 0 and b"OK" in r_check.stdout
        print(f"R available: {has_r}")
    except Exception:
        has_r = False
        print("R not available (skipping RDS benchmarks)")

    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gse_id, meta in DATASETS.items():
            try:
                result = benchmark_single(gse_id, meta, tmpdir)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"\n  {gse_id} CRASHED: {e}")
                traceback.print_exc()

    # Save results
    out_path = os.path.join(SCRIPT_DIR, "benchmark_results_v3.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total datasets benchmarked: {len(all_results)}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'GSE':<12} {'Protocol':<20} {'Species':<15} {'Cells':>8} {'NNZ':>12} "
          f"{'1pz(MB)':>8} {'Ratio':>6} {'Read(ms)':>9} {'MB/s':>7}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: x['nnz']):
        pz = r['formats'].get('1pz', {})
        print(f"{r['gse_id']:<12} {r['protocol']:<20} {r['species']:<15} {r['ncols']:>8,} "
              f"{r['nnz']:>12,} {pz.get('size',0)/1e6:>8.1f} {pz.get('ratio_vs_int32',0):>6.1f} "
              f"{pz.get('read_s',0)*1000:>9.0f} {pz.get('read_MBps_int32',0):>7.0f}")


if __name__ == "__main__":
    main()
