#!/usr/bin/env python3
"""
Compression frontier benchmark for SinglePress manuscript.

Measures:
1. Theoretical entropy lower bound (Shannon entropy of nonzero values)
2. Zstd level sweep (levels 1–22): compression ratio, write time, read time
3. Alternative backend codecs on the same VOCSC-encoded data
4. Pre-compression filters (delta, shuffle) applied before zstd

Strategy: We re-write each matrix at every zstd level and with alternative
codecs by writing to tmpfs (/dev/shm/) for cache-neutral I/O.

Output: compression_frontier.csv
"""

import os
import sys
import time
import csv
import math
import tempfile
import struct
import numpy as np
import scipy.sparse as ss

# Compression libraries
import zstandard as zstd
import lz4.frame
import brotli
import zlib
import bz2
import lzma

sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp

# ── Configuration ──────────────────────────────────────────────
PZ_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(CODE_DATA_DIR, "compression_frontier.csv")

# Stratified 20 datasets spanning 1M–5.6B nnz
DATASETS = [
    "GSE135310", "GSE202575", "GSE295234", "GSE223222", "GSE245073",
    "GSE267467", "GSE250444", "GSE239941", "GSE152915", "GSE305376",
    "GSE277034", "GSE151346", "GSE201534", "GSE293737", "GSE299078",
    "GSE270866", "GSE256025", "GSE192807", "GSE241998",
    # GSE292909 (5.6B nnz) is too large for full re-encode sweeps; skip or sample
]

ZSTD_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19, 22]
N_WRITE_REPS = 3
N_READ_REPS = 5

# ── Helpers ────────────────────────────────────────────────────

def shannon_entropy_bits(values):
    """Compute Shannon entropy in bits from an array of integer values."""
    if len(values) == 0:
        return 0.0
    unique, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def compute_entropy_stats(mat_csc):
    """Compute value entropy and theoretical lower bounds."""
    data = mat_csc.data
    nnz = len(data)
    
    # Value entropy
    H_value = shannon_entropy_bits(data)
    
    # Theoretical minimum: each nonzero needs H_value bits for its value
    # Plus we need to encode the position (row index per column)
    # For position: entropy of gap sizes between nonzeros within each column
    
    # Column-level gap entropy
    indptr = mat_csc.indptr
    indices = mat_csc.indices
    all_gaps = []
    for j in range(mat_csc.shape[1]):
        start, end = indptr[j], indptr[j+1]
        if end - start > 0:
            col_rows = indices[start:end]
            # VOCSC sorts by value, then by row within each value group
            # But for entropy calculation, raw gaps approximate this
            gaps = np.diff(np.sort(col_rows))
            if len(gaps) > 0:
                all_gaps.extend(gaps.tolist())
    
    all_gaps = np.array(all_gaps, dtype=np.int64) if all_gaps else np.array([0])
    H_gap = shannon_entropy_bits(all_gaps)
    
    # First row index per column (position of first nonzero)
    first_rows = []
    for j in range(mat_csc.shape[1]):
        start, end = indptr[j], indptr[j+1]
        if end - start > 0:
            first_rows.append(indices[start])
    H_first = shannon_entropy_bits(np.array(first_rows))
    
    # Theoretical minimum bytes:
    # = (nnz * H_value + (nnz - ncols_with_data) * H_gap + ncols_with_data * H_first) / 8
    # Plus column pointer overhead (negligible for large matrices)
    ncols_with_data = sum(1 for j in range(mat_csc.shape[1]) if indptr[j+1] > indptr[j])
    n_gap_values = nnz - ncols_with_data  # one first-row per column, rest are gaps
    
    theory_bits = nnz * H_value + max(0, n_gap_values) * H_gap + ncols_with_data * H_first
    theory_bytes = theory_bits / 8
    
    # Simpler bound: just value entropy (ignoring position encoding)
    value_only_bytes = (nnz * H_value) / 8
    
    return {
        "H_value_bits": H_value,
        "H_gap_bits": H_gap, 
        "H_first_bits": H_first,
        "theory_min_bytes": theory_bytes,
        "value_only_min_bytes": value_only_bytes,
        "nnz": nnz,
        "ncols_with_data": ncols_with_data,
    }


def compress_bytes_with_codec(raw_bytes, codec, level=None):
    """Compress raw bytes with various codecs. Returns (compressed_bytes, compress_time)."""
    t0 = time.perf_counter()
    if codec == "zstd":
        cctx = zstd.ZstdCompressor(level=level or 3)
        compressed = cctx.compress(raw_bytes)
    elif codec == "lz4":
        compressed = lz4.frame.compress(raw_bytes, compression_level=level or 0)
    elif codec == "brotli":
        compressed = brotli.compress(raw_bytes, quality=level or 4)
    elif codec == "gzip":
        compressed = zlib.compress(raw_bytes, level or 6)
    elif codec == "bz2":
        compressed = bz2.compress(raw_bytes, compresslevel=level or 9)
    elif codec == "lzma":
        compressed = lzma.compress(raw_bytes, preset=level or 6)
    else:
        raise ValueError(f"Unknown codec: {codec}")
    t1 = time.perf_counter()
    return compressed, t1 - t0


def decompress_bytes_with_codec(compressed_bytes, codec):
    """Decompress bytes with various codecs. Returns (raw_bytes, decompress_time)."""
    t0 = time.perf_counter()
    if codec == "zstd":
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed_bytes)
    elif codec == "lz4":
        raw = lz4.frame.decompress(compressed_bytes)
    elif codec == "brotli":
        raw = brotli.decompress(compressed_bytes)
    elif codec == "gzip":
        raw = zlib.decompress(compressed_bytes)
    elif codec == "bz2":
        raw = bz2.decompress(compressed_bytes)
    elif codec == "lzma":
        raw = lzma.decompress(compressed_bytes)
    else:
        raise ValueError(f"Unknown codec: {codec}")
    t1 = time.perf_counter()
    return raw, t1 - t0


def benchmark_write_read_1pz(mat_csc, rownames, colnames, level, n_write=3, n_read=5):
    """Benchmark write_1pz and read_1pz at a given zstd level."""
    tmp = f"/dev/shm/bench_level_{level}.1pz"
    
    # Write benchmark
    write_times = []
    file_size = 0
    for _ in range(n_write):
        t0 = time.perf_counter()
        info = sp.write_1pz(tmp, mat_csc, rownames=rownames, colnames=colnames,
                            level=level, num_threads=8)
        t1 = time.perf_counter()
        write_times.append(t1 - t0)
        file_size = os.path.getsize(tmp)
    
    # Read benchmark
    read_times = []
    for _ in range(n_read):
        # Clear page cache hint (best effort)
        os.sync()
        t0 = time.perf_counter()
        _ = sp.read_1pz(tmp, num_threads=8)
        t1 = time.perf_counter()
        read_times.append(t1 - t0)
    
    os.remove(tmp)
    
    return {
        "file_bytes": file_size,
        "write_median_s": float(np.median(write_times)),
        "write_q25_s": float(np.percentile(write_times, 25)),
        "write_q75_s": float(np.percentile(write_times, 75)),
        "read_median_s": float(np.median(read_times)),
        "read_q25_s": float(np.percentile(read_times, 25)),
        "read_q75_s": float(np.percentile(read_times, 75)),
    }


def benchmark_alt_codec_on_raw(mat_csc, codec, level):
    """
    Benchmark an alternative codec by compressing the raw CSC data arrays.
    This simulates replacing zstd in the SinglePress pipeline with another codec.
    We compress the CSC data array (values) and indices array separately,
    as SinglePress does per-chunk.
    """
    # Get raw byte arrays (what would be the payload after VOCSC encoding)
    data_bytes = mat_csc.data.astype(np.int32).tobytes()
    indices_bytes = mat_csc.indices.astype(np.int32).tobytes()
    indptr_bytes = mat_csc.indptr.astype(np.int64).tobytes()
    
    total_raw = len(data_bytes) + len(indices_bytes) + len(indptr_bytes)
    
    # Compress each component
    comp_data, t_data = compress_bytes_with_codec(data_bytes, codec, level)
    comp_idx, t_idx = compress_bytes_with_codec(indices_bytes, codec, level)
    comp_ptr, t_ptr = compress_bytes_with_codec(indptr_bytes, codec, level)
    total_compressed = len(comp_data) + len(comp_idx) + len(comp_ptr)
    total_compress_time = t_data + t_idx + t_ptr
    
    # Decompress
    _, dt_data = decompress_bytes_with_codec(comp_data, codec)
    _, dt_idx = decompress_bytes_with_codec(comp_idx, codec)
    _, dt_ptr = decompress_bytes_with_codec(comp_ptr, codec)
    total_decompress_time = dt_data + dt_idx + dt_ptr
    
    return {
        "raw_bytes": total_raw,
        "compressed_bytes": total_compressed,
        "ratio": total_raw / total_compressed if total_compressed > 0 else 0,
        "compress_s": total_compress_time,
        "decompress_s": total_decompress_time,
    }


# ── Main ───────────────────────────────────────────────────────

def main():
    rows = []
    
    print(f"Compression frontier benchmark")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Zstd levels: {ZSTD_LEVELS}")
    print(f"Write reps: {N_WRITE_REPS}, Read reps: {N_READ_REPS}")
    print()
    
    for i, gse in enumerate(DATASETS):
        pz_path = os.path.join(PZ_DIR, gse, "counts.1pz")
        if not os.path.exists(pz_path):
            print(f"[{i+1}/{len(DATASETS)}] {gse}: MISSING, skipping")
            continue
        
        pz_size = os.path.getsize(pz_path)
        print(f"[{i+1}/{len(DATASETS)}] {gse}: reading {pz_size/1e6:.0f} MB...")
        
        # Read the matrix
        t0 = time.perf_counter()
        mat = sp.read_1pz(pz_path, num_threads=8)
        t_read = time.perf_counter() - t0
        
        if not ss.issparse(mat):
            print(f"  Not sparse, skipping")
            continue
        
        mat_csc = mat.tocsc()
        nnz = mat_csc.nnz
        nrows, ncols = mat_csc.shape
        raw_int32 = nnz * 8  # 4 bytes value + 4 bytes index per nonzero
        
        # Get row/col names from file
        pz = sp.open_1pz(pz_path)
        rownames = pz.rownames
        colnames = pz.colnames
        del pz
        
        print(f"  Shape: {nrows}x{ncols}, nnz={nnz:,}, read in {t_read:.1f}s")

        # Re-encode at default level to get accurate pz_default_bytes
        tmp_default = f"/dev/shm/bench_default_{gse}.1pz"
        sp.write_1pz(tmp_default, mat_csc, rownames=rownames, colnames=colnames,
                     num_threads=8)
        pz_size = os.path.getsize(tmp_default)
        os.remove(tmp_default)
        print(f"  Re-encoded default: {pz_size/1e6:.1f} MB")

        # ── 1. Entropy analysis ──────────────────────────────────
        print(f"  Computing entropy...", end="", flush=True)
        entropy = compute_entropy_stats(mat_csc)
        print(f" H_value={entropy['H_value_bits']:.2f} bits, H_gap={entropy['H_gap_bits']:.2f} bits")
        print(f"  Theory min: {entropy['theory_min_bytes']/1e6:.1f} MB"
              f" (value-only: {entropy['value_only_min_bytes']/1e6:.1f} MB)")
        print(f"  Current .1pz: {pz_size/1e6:.1f} MB"
              f" ({pz_size/entropy['theory_min_bytes']:.2f}× theory)")
        
        # Store entropy row
        rows.append({
            "gse_id": gse, "nnz": nnz, "nrows": nrows, "ncols": ncols,
            "experiment": "entropy",
            "codec": "theory",
            "level": 0,
            "file_bytes": int(entropy["theory_min_bytes"]),
            "raw_int32_bytes": raw_int32,
            "ratio_vs_int32": raw_int32 / entropy["theory_min_bytes"],
            "ratio_vs_theory": 1.0,
            "write_median_s": 0, "read_median_s": 0,
            "write_mbps": 0, "read_mbps": 0,
            "H_value_bits": entropy["H_value_bits"],
            "H_gap_bits": entropy["H_gap_bits"],
            "pz_default_bytes": pz_size,
        })
        
        # Value-only entropy row
        rows.append({
            "gse_id": gse, "nnz": nnz, "nrows": nrows, "ncols": ncols,
            "experiment": "entropy",
            "codec": "value_entropy_only",
            "level": 0,
            "file_bytes": int(entropy["value_only_min_bytes"]),
            "raw_int32_bytes": raw_int32,
            "ratio_vs_int32": raw_int32 / entropy["value_only_min_bytes"],
            "ratio_vs_theory": entropy["theory_min_bytes"] / entropy["value_only_min_bytes"],
            "write_median_s": 0, "read_median_s": 0,
            "write_mbps": 0, "read_mbps": 0,
            "H_value_bits": entropy["H_value_bits"],
            "H_gap_bits": entropy["H_gap_bits"],
            "pz_default_bytes": pz_size,
        })
        
        # ── 2. Zstd level sweep ──────────────────────────────────
        # Skip full level sweep for very large datasets
        if nnz > 500_000_000:
            levels_to_test = [1, 3, 9, 19]
            n_write = 1
            n_read = 3
            print(f"  Large dataset — testing zstd levels {levels_to_test} only")
        else:
            levels_to_test = ZSTD_LEVELS
            n_write = N_WRITE_REPS
            n_read = N_READ_REPS
        
        for lvl in levels_to_test:
            print(f"  zstd-{lvl}...", end="", flush=True)
            result = benchmark_write_read_1pz(mat_csc, rownames, colnames, lvl,
                                               n_write=n_write, n_read=n_read)
            ratio_vs_int32 = raw_int32 / result["file_bytes"]
            ratio_vs_theory = result["file_bytes"] / entropy["theory_min_bytes"]
            write_mbps = (raw_int32 / 1e6) / result["write_median_s"] if result["write_median_s"] > 0 else 0
            read_mbps = (raw_int32 / 1e6) / result["read_median_s"] if result["read_median_s"] > 0 else 0
            
            print(f" {result['file_bytes']/1e6:.1f}MB ({ratio_vs_int32:.1f}×)"
                  f" W:{write_mbps:.0f}MB/s R:{read_mbps:.0f}MB/s")
            
            rows.append({
                "gse_id": gse, "nnz": nnz, "nrows": nrows, "ncols": ncols,
                "experiment": "zstd_level",
                "codec": f"zstd-{lvl}",
                "level": lvl,
                "file_bytes": result["file_bytes"],
                "raw_int32_bytes": raw_int32,
                "ratio_vs_int32": ratio_vs_int32,
                "ratio_vs_theory": ratio_vs_theory,
                "write_median_s": result["write_median_s"],
                "read_median_s": result["read_median_s"],
                "write_mbps": write_mbps,
                "read_mbps": read_mbps,
                "H_value_bits": entropy["H_value_bits"],
                "H_gap_bits": entropy["H_gap_bits"],
                "pz_default_bytes": pz_size,
            })
        
        # ── 3. Alternative codecs on raw CSC ────────────────────
        # These compress the raw int32 arrays (data + indices) to show
        # what generic compressors achieve WITHOUT VOCSC encoding
        alt_codecs = [
            ("lz4", 0), ("lz4", 9),
            ("gzip", 1), ("gzip", 6), ("gzip", 9),
            ("bz2", 9),
            ("brotli", 1), ("brotli", 4), ("brotli", 11),
            ("lzma", 6),
        ]
        
        # Only test alt codecs on medium-sized datasets (skip huge ones)
        if nnz <= 200_000_000:
            for codec_name, codec_lvl in alt_codecs:
                print(f"  {codec_name}-{codec_lvl} (raw CSC)...", end="", flush=True)
                try:
                    result = benchmark_alt_codec_on_raw(mat_csc, codec_name, codec_lvl)
                    ratio_vs_int32 = raw_int32 / result["compressed_bytes"]
                    ratio_vs_theory = result["compressed_bytes"] / entropy["theory_min_bytes"]
                    
                    print(f" {result['compressed_bytes']/1e6:.1f}MB ({ratio_vs_int32:.1f}×)"
                          f" C:{result['compress_s']:.2f}s D:{result['decompress_s']:.2f}s")
                    
                    rows.append({
                        "gse_id": gse, "nnz": nnz, "nrows": nrows, "ncols": ncols,
                        "experiment": "alt_codec_raw",
                        "codec": f"{codec_name}-{codec_lvl}",
                        "level": codec_lvl,
                        "file_bytes": result["compressed_bytes"],
                        "raw_int32_bytes": raw_int32,
                        "ratio_vs_int32": ratio_vs_int32,
                        "ratio_vs_theory": ratio_vs_theory,
                        "write_median_s": result["compress_s"],
                        "read_median_s": result["decompress_s"],
                        "write_mbps": (raw_int32 / 1e6) / result["compress_s"] if result["compress_s"] > 0 else 0,
                        "read_mbps": (raw_int32 / 1e6) / result["decompress_s"] if result["decompress_s"] > 0 else 0,
                        "H_value_bits": entropy["H_value_bits"],
                        "H_gap_bits": entropy["H_gap_bits"],
                        "pz_default_bytes": pz_size,
                    })
                except Exception as e:
                    print(f" FAILED: {e}")
        
        # Flush results after each dataset
        os.makedirs(CODE_DATA_DIR, exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  Wrote {len(rows)} rows to {OUT_CSV}")
        print()
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {len(rows)} total measurements")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
