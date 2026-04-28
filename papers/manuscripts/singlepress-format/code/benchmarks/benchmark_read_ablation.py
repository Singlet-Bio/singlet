#!/usr/bin/env python3
"""
Read speed ablation study: decompose WHY .1pz reads faster than H5AD.

Experiments:
  1. Single-threaded .1pz vs single-threaded H5AD (isolate codec + format)
  2. Thread scaling 1→8 for .1pz (measure parallelism contribution)
  3. File size effect: measure raw sequential read speed (no decode)
  4. Codec race: decompress same raw data with zstd-3 vs gzip-4
  5. H5AD with alternative backends: uncompressed H5AD, lzf H5AD
  6. Memory footprint: peak RSS during read

Output: read_speed_ablation.csv

Usage:
    cd /tmp && srun --time=120 --mem=64G --cpus-per-task=8 \
        python3 -u /path/to/benchmark_read_ablation.py
"""
import csv
import gc
import os
import resource
import struct
import sys
import tempfile
import time
import zlib

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# Fix namespace-package shadowing
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if "" in sys.path:
    sys.path.remove("")

import numpy as np

import singlepress as sp

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "read_speed_ablation.csv")

# Representative datasets spanning range of sizes
DATASETS = [
    ("GSE155409", 122_930),        # tiny
    ("GSE210261", 4_575_353),      # small
    ("GSE189042", 20_016_693),     # medium
    ("GSE290932", 32_590_922),     # medium-large
    ("GSE142483", 50_452_624),     # large
    ("GSE207157", 73_987_183),     # large
    ("GSE248138", 112_694_589),    # very large
]

N_TRIALS = 5
WARMUP = 1

results = []


def get_rss_mb():
    """Peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timed(func, n_trials=N_TRIALS, warmup=WARMUP):
    """Median of n_trials after warmup. Returns (median_s, result)."""
    times = []
    result = None
    for i in range(warmup + n_trials):
        gc.collect()
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
    return float(np.median(times)), result


print("=" * 70)
print("READ SPEED ABLATION STUDY")
print("=" * 70)

for gse_id, expected_nnz in DATASETS:
    pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    if not os.path.isfile(pz_path):
        print(f"SKIP {gse_id}: file not found")
        continue

    pz_size = os.path.getsize(pz_path)
    info = sp.info_1pz(pz_path)
    nnz = info["nnz"]
    nrows = info["m"]
    ncols = info["n"]
    raw_bytes = nnz * 4  # int32

    print(f"\n{'─' * 60}")
    print(f"Dataset: {gse_id}  nnz={nnz:,}  shape=({nrows},{ncols})")
    print(f"  .1pz size: {pz_size / 1e6:.1f} MB  raw int32: {raw_bytes / 1e6:.1f} MB")

    rec = {
        "gse_id": gse_id,
        "nnz": nnz,
        "nrows": nrows,
        "ncols": ncols,
        "pz_bytes": pz_size,
        "raw_int32_bytes": raw_bytes,
    }

    # ─── Experiment 1: Raw file read (no decode) ───
    # Measures pure I/O: how fast can we just read bytes from disk?
    print("  [1] Raw file read (no decode)...")
    t_raw, _ = timed(lambda: open(pz_path, "rb").read())
    rec["raw_read_s"] = round(t_raw, 6)
    rec["raw_read_mbps"] = round(pz_size / t_raw / 1e6, 1)
    print(f"      {pz_size / t_raw / 1e6:.0f} MB/s ({t_raw * 1000:.1f} ms)")

    # ─── Experiment 2: .1pz read with 1 thread ───
    print("  [2] .1pz read, 1 thread...")
    t_1t, mat = timed(lambda: sp.read_1pz(pz_path, num_threads=1))
    rec["pz_1thread_s"] = round(t_1t, 6)
    rec["pz_1thread_mbps"] = round(raw_bytes / t_1t / 1e6, 1)
    print(f"      {raw_bytes / t_1t / 1e6:.0f} MB/s ({t_1t * 1000:.1f} ms)")

    # ─── Experiment 3: .1pz read with 2 threads ───
    print("  [3] .1pz read, 2 threads...")
    t_2t, _ = timed(lambda: sp.read_1pz(pz_path, num_threads=2))
    rec["pz_2thread_s"] = round(t_2t, 6)
    rec["pz_2thread_mbps"] = round(raw_bytes / t_2t / 1e6, 1)
    print(f"      {raw_bytes / t_2t / 1e6:.0f} MB/s ({t_2t * 1000:.1f} ms)")

    # ─── Experiment 4: .1pz read with 4 threads ───
    print("  [4] .1pz read, 4 threads...")
    t_4t, _ = timed(lambda: sp.read_1pz(pz_path, num_threads=4))
    rec["pz_4thread_s"] = round(t_4t, 6)
    rec["pz_4thread_mbps"] = round(raw_bytes / t_4t / 1e6, 1)
    print(f"      {raw_bytes / t_4t / 1e6:.0f} MB/s ({t_4t * 1000:.1f} ms)")

    # ─── Experiment 5: .1pz read with 8 threads (default) ───
    print("  [5] .1pz read, 8 threads (default)...")
    t_8t, _ = timed(lambda: sp.read_1pz(pz_path, num_threads=8))
    rec["pz_8thread_s"] = round(t_8t, 6)
    rec["pz_8thread_mbps"] = round(raw_bytes / t_8t / 1e6, 1)
    print(f"      {raw_bytes / t_8t / 1e6:.0f} MB/s ({t_8t * 1000:.1f} ms)")

    # Thread speedup
    if t_1t > 0:
        rec["thread_speedup_2"] = round(t_1t / t_2t, 2)
        rec["thread_speedup_4"] = round(t_1t / t_4t, 2)
        rec["thread_speedup_8"] = round(t_1t / t_8t, 2)
        print(f"      Thread speedup: 2T={t_1t/t_2t:.1f}x  4T={t_1t/t_4t:.1f}x  8T={t_1t/t_8t:.1f}x")

    # Prepare H5AD for comparison
    print("  [prep] Writing H5AD formats...")
    try:
        import anndata as ad
        import pandas as pd
        import scipy.sparse as ss

        csc = mat.tocsc()
        csc.data = csc.data.astype(np.int32)
        rownames = getattr(mat, "rownames", None) or [f"g{i}" for i in range(nrows)]
        colnames = getattr(mat, "colnames", None) or [f"c{i}" for i in range(ncols)]
        adata = ad.AnnData(
            X=csc.T.tocsr(),
            obs=pd.DataFrame(index=list(colnames)),
            var=pd.DataFrame(index=list(rownames)),
        )

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
            # Write H5AD with gzip (standard)
            h5ad_gzip_path = os.path.join(tmpdir, "gzip.h5ad")
            adata.write_h5ad(h5ad_gzip_path, compression="gzip")
            h5ad_gzip_size = os.path.getsize(h5ad_gzip_path)
            rec["h5ad_gzip_bytes"] = h5ad_gzip_size

            # Write H5AD with lzf (faster decompression)
            h5ad_lzf_path = os.path.join(tmpdir, "lzf.h5ad")
            adata.write_h5ad(h5ad_lzf_path, compression="lzf")
            h5ad_lzf_size = os.path.getsize(h5ad_lzf_path)
            rec["h5ad_lzf_bytes"] = h5ad_lzf_size

            # Write uncompressed H5AD
            h5ad_none_path = os.path.join(tmpdir, "none.h5ad")
            adata.write_h5ad(h5ad_none_path, compression=None)
            h5ad_none_size = os.path.getsize(h5ad_none_path)
            rec["h5ad_none_bytes"] = h5ad_none_size

            print(f"      H5AD sizes: gzip={h5ad_gzip_size/1e6:.1f}MB  lzf={h5ad_lzf_size/1e6:.1f}MB  none={h5ad_none_size/1e6:.1f}MB")

            # ─── Experiment 6: H5AD read (gzip, standard) ───
            print("  [6] H5AD read (gzip, standard)...")
            t_h5gzip, _ = timed(lambda: ad.read_h5ad(h5ad_gzip_path))
            rec["h5ad_gzip_read_s"] = round(t_h5gzip, 6)
            rec["h5ad_gzip_read_mbps"] = round(raw_bytes / t_h5gzip / 1e6, 1)
            print(f"      {raw_bytes / t_h5gzip / 1e6:.0f} MB/s ({t_h5gzip * 1000:.1f} ms)")

            # ─── Experiment 7: H5AD read (lzf, fast codec) ───
            print("  [7] H5AD read (lzf, fast codec)...")
            t_h5lzf, _ = timed(lambda: ad.read_h5ad(h5ad_lzf_path))
            rec["h5ad_lzf_read_s"] = round(t_h5lzf, 6)
            rec["h5ad_lzf_read_mbps"] = round(raw_bytes / t_h5lzf / 1e6, 1)
            print(f"      {raw_bytes / t_h5lzf / 1e6:.0f} MB/s ({t_h5lzf * 1000:.1f} ms)")

            # ─── Experiment 8: H5AD read (uncompressed) ───
            # This isolates HDF5/anndata Python overhead from decompression
            print("  [8] H5AD read (uncompressed)...")
            t_h5none, _ = timed(lambda: ad.read_h5ad(h5ad_none_path))
            rec["h5ad_none_read_s"] = round(t_h5none, 6)
            rec["h5ad_none_read_mbps"] = round(raw_bytes / t_h5none / 1e6, 1)
            print(f"      {raw_bytes / t_h5none / 1e6:.0f} MB/s ({t_h5none * 1000:.1f} ms)")

            # ─── Experiment 9: Codec microbenchmark ───
            # Compress same raw CSC data with zstd-3 and gzip-4, time decompression
            print("  [9] Codec microbenchmark (zstd-3 vs gzip-4)...")
            import zstandard as zstd

            raw_data = csc.data.tobytes() + csc.indices.tobytes()
            raw_size = len(raw_data)

            # zstd-3 compress then decompress
            cctx = zstd.ZstdCompressor(level=3)
            zstd_compressed = cctx.compress(raw_data)
            dctx = zstd.ZstdDecompressor()

            zstd_times = []
            for trial in range(WARMUP + N_TRIALS):
                gc.collect()
                t0 = time.perf_counter()
                _ = dctx.decompress(zstd_compressed, max_output_size=raw_size)
                t1 = time.perf_counter()
                if trial >= WARMUP:
                    zstd_times.append(t1 - t0)
            t_zstd = float(np.median(zstd_times))
            rec["zstd3_decompress_mbps"] = round(raw_size / t_zstd / 1e6, 0)
            print(f"      zstd-3 decompress: {raw_size / t_zstd / 1e6:.0f} MB/s  "
                  f"({len(zstd_compressed) / 1e6:.1f} MB compressed)")

            # gzip-4 compress then decompress
            gzip_compressed = zlib.compress(raw_data, 4)

            gzip_times = []
            for trial in range(WARMUP + N_TRIALS):
                gc.collect()
                t0 = time.perf_counter()
                _ = zlib.decompress(gzip_compressed)
                t1 = time.perf_counter()
                if trial >= WARMUP:
                    gzip_times.append(t1 - t0)
            t_gzip = float(np.median(gzip_times))
            rec["gzip4_decompress_mbps"] = round(raw_size / t_gzip / 1e6, 0)
            print(f"      gzip-4 decompress: {raw_size / t_gzip / 1e6:.0f} MB/s  "
                  f"({len(gzip_compressed) / 1e6:.1f} MB compressed)")

            rec["zstd_vs_gzip_codec_speedup"] = round(t_gzip / t_zstd, 2)
            print(f"      Codec speedup (zstd/gzip): {t_gzip / t_zstd:.1f}x")

            # Compressed sizes for reference
            rec["zstd3_compressed_bytes"] = len(zstd_compressed)
            rec["gzip4_compressed_bytes"] = len(gzip_compressed)
            rec["zstd3_ratio"] = round(raw_size / len(zstd_compressed), 2)
            rec["gzip4_ratio"] = round(raw_size / len(gzip_compressed), 2)

            del raw_data, zstd_compressed, gzip_compressed
            gc.collect()

    except Exception as e:
        import traceback
        print(f"      ERROR: {e}")
        traceback.print_exc()

    # ─── Speedup decomposition ───
    if "h5ad_gzip_read_s" in rec and rec.get("pz_8thread_s", 0) > 0:
        total_speedup = rec["h5ad_gzip_read_s"] / rec["pz_8thread_s"]
        # Decompose: 1pz-1T vs h5ad-gzip tells us format+codec advantage
        format_codec_speedup = rec["h5ad_gzip_read_s"] / rec["pz_1thread_s"]
        # 1pz-8T vs 1pz-1T tells us threading contribution
        thread_speedup = rec["pz_1thread_s"] / rec["pz_8thread_s"]
        # h5ad-none vs h5ad-gzip tells us how much gzip costs h5ad
        if "h5ad_none_read_s" in rec:
            gzip_overhead = rec["h5ad_gzip_read_s"] / rec["h5ad_none_read_s"]

        rec["total_speedup_vs_h5ad"] = round(total_speedup, 2)
        rec["format_codec_speedup_1t"] = round(format_codec_speedup, 2)
        rec["thread_contribution"] = round(thread_speedup, 2)
        if "h5ad_none_read_s" in rec:
            rec["h5ad_gzip_overhead"] = round(gzip_overhead, 2)

        print(f"\n  DECOMPOSITION:")
        print(f"    Total speedup (.1pz-8T vs h5ad-gzip): {total_speedup:.1f}x")
        print(f"    Format+codec (.1pz-1T vs h5ad-gzip):  {format_codec_speedup:.1f}x")
        print(f"    Threading (.1pz-8T vs .1pz-1T):       {thread_speedup:.1f}x")
        if "h5ad_none_read_s" in rec:
            print(f"    H5AD gzip overhead (gzip vs none):    {gzip_overhead:.1f}x")
            h5ad_overhead = rec["h5ad_none_read_s"] / rec["pz_1thread_s"]
            rec["h5ad_format_overhead_1t"] = round(h5ad_overhead, 2)
            print(f"    H5AD format overhead (h5ad-none vs .1pz-1T): {h5ad_overhead:.1f}x")

    del mat
    gc.collect()
    results.append(rec)

# ─── Write CSV ───
if results:
    fieldnames = list(results[0].keys())
    # Ensure all keys from all records
    for r in results:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {OUT}")

# ─── Print summary table ───
print("\n" + "=" * 90)
print("SUMMARY: Read speed ablation")
print("=" * 90)
print(f"{'Dataset':<14} {'nnz':>10} {'1pz-1T':>10} {'1pz-8T':>10} {'h5ad-gz':>10} "
      f"{'h5ad-lzf':>10} {'h5ad-raw':>10} {'Total':>7} {'Codec':>7} {'Thread':>7}")
print(f"{'':14} {'':>10} {'(MB/s)':>10} {'(MB/s)':>10} {'(MB/s)':>10} "
      f"{'(MB/s)':>10} {'(MB/s)':>10} {'(x)':>7} {'(x)':>7} {'(x)':>7}")
print("─" * 90)
for r in results:
    print(f"{r['gse_id']:<14} {r['nnz']:>10,} "
          f"{r.get('pz_1thread_mbps', ''):>10} "
          f"{r.get('pz_8thread_mbps', ''):>10} "
          f"{r.get('h5ad_gzip_read_mbps', ''):>10} "
          f"{r.get('h5ad_lzf_read_mbps', ''):>10} "
          f"{r.get('h5ad_none_read_mbps', ''):>10} "
          f"{r.get('total_speedup_vs_h5ad', ''):>7} "
          f"{r.get('fmt_codec_speedup_1t', ''):>7} "
          f"{r.get('thread_contribution', ''):>7}")

print("\n" + "=" * 90)
print("CODEC MICROBENCHMARK")
print("=" * 90)
print(f"{'Dataset':<14} {'nnz':>10} {'zstd-3':>12} {'gzip-4':>12} {'Speedup':>10}")
print(f"{'':14} {'':>10} {'(MB/s)':>12} {'(MB/s)':>12} {'(x)':>10}")
print("─" * 60)
for r in results:
    print(f"{r['gse_id']:<14} {r['nnz']:>10,} "
          f"{r.get('zstd3_decompress_mbps', ''):>12} "
          f"{r.get('gzip4_decompress_mbps', ''):>12} "
          f"{r.get('zstd_vs_gzip_codec_speedup', ''):>10}")

print(f"\nDone. Results in {OUT}")
