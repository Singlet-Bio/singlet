#!/usr/bin/env python3
"""
Benchmark saveRDS(dgCMatrix) write speed for comparison with .1pz write.
Uses the same datasets as benchmark_results_v3.json.

Usage:
    srun --time=60 --mem=64G --cpus-per-task=8 \
        python3 -u /path/to/benchmark_write_rds.py

Output: r_write_benchmarks.csv
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import scipy.sparse as ss
import singlepress as sp

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
N_TRIALS = 3

# Load the dataset list from the v3 results to match exactly
with open(os.path.join(DATA_DIR, "benchmark_results_v3.json")) as f:
    v3_data = json.load(f)

datasets = [(r["gse_id"], r["nnz"], r["raw_int32_bytes"]) for r in v3_data]
datasets.sort(key=lambda x: x[1])

print(f"Benchmarking saveRDS write speed for {len(datasets)} datasets")
print(f"Trials per dataset: {N_TRIALS}")


def time_saverds(csc, rownames, colnames, rds_path, n_trials=N_TRIALS):
    """Time saveRDS(dgCMatrix) by exporting components and calling R."""
    prefix = rds_path + ".tmp"
    np.save(prefix + "_i.npy", csc.indices.astype(np.int32))
    np.save(prefix + "_p.npy", csc.indptr.astype(np.int32))
    np.save(prefix + "_x.npy", csc.data.astype(np.float64))
    with open(prefix + "_rn.txt", "w") as f:
        f.write("\n".join(str(r) for r in rownames) + "\n")
    with open(prefix + "_cn.txt", "w") as f:
        f.write("\n".join(str(c) for c in colnames) + "\n")

    r_script = f"""
suppressMessages(library(Matrix))

parse_npy_int32 <- function(raw_bytes) {{
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

i_raw <- readBin("{prefix}_i.npy", "raw", file.info("{prefix}_i.npy")$size)
p_raw <- readBin("{prefix}_p.npy", "raw", file.info("{prefix}_p.npy")$size)
x_raw <- readBin("{prefix}_x.npy", "raw", file.info("{prefix}_x.npy")$size)
i <- parse_npy_int32(i_raw)
p <- parse_npy_int32(p_raw)
x <- parse_npy_float64(x_raw)
rn <- readLines("{prefix}_rn.txt")
cn <- readLines("{prefix}_cn.txt")

m <- new("dgCMatrix", i=i, p=p, x=x,
         Dim=c({csc.shape[0]}L, {csc.shape[1]}L),
         Dimnames=list(rn, cn))

rm(i_raw, p_raw, x_raw, i, p, x, rn, cn)
invisible(gc(verbose=FALSE))

# Warmup
saveRDS(m, "{rds_path}", compress="gzip")

# Timed trials
times <- numeric({n_trials})
for (trial in 1:{n_trials}) {{
  if (file.exists("{rds_path}")) file.remove("{rds_path}")
  invisible(gc(verbose=FALSE))
  t0 <- proc.time()
  saveRDS(m, "{rds_path}", compress="gzip")
  t1 <- proc.time()
  times[trial] <- (t1 - t0)[3]
}}
cat(sprintf("%.6f\\n", median(times)))
cat(sprintf("SIZE:%d\\n", file.info("{rds_path}")$size))
"""
    r_file = prefix + ".R"
    with open(r_file, "w") as f:
        f.write(r_script)

    result = subprocess.run(
        ["bash", "-c", f"module load r/4.5.2 && Rscript {r_file}"],
        capture_output=True, timeout=600)

    # Cleanup temp files
    for suffix in ["_i.npy", "_p.npy", "_x.npy", "_rn.txt", "_cn.txt", ".R"]:
        p = prefix + suffix
        if os.path.exists(p):
            os.remove(p)

    if result.returncode != 0:
        raise RuntimeError(f"R failed (rc={result.returncode}): stderr={result.stderr.decode()[:500]}\nstdout={result.stdout.decode()[:200]}")

    lines = result.stdout.decode().strip().split("\n")
    median_time = None
    rds_size = None
    for line in lines:
        line = line.strip()
        if line.startswith("SIZE:"):
            rds_size = int(line[5:])
        else:
            try:
                median_time = float(line)
            except ValueError:
                pass

    if median_time is None:
        raise RuntimeError(f"No timing output: {result.stdout.decode()[:200]}")

    return median_time, rds_size


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gse_id, nnz, raw_int32_bytes in datasets:
            pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
            if not os.path.isfile(pz_path):
                print(f"  {gse_id}: not found, skipping")
                continue

            print(f"\n{gse_id}: nnz={nnz:,}")

            # Load matrix
            try:
                mat = sp.read_1pz(pz_path)
            except Exception as e:
                print(f"  FAILED to read: {e}")
                continue

            csc = mat.tocsc()
            csc.sort_indices()
            csc.data = csc.data.astype(np.int32)
            rownames = getattr(mat, 'rownames', None) or [f"gene_{i}" for i in range(mat.shape[0])]
            colnames = getattr(mat, 'colnames', None) or [f"cell_{i}" for i in range(mat.shape[1])]
            if not rownames:
                rownames = [f"gene_{i}" for i in range(mat.shape[0])]
            if not colnames:
                colnames = [f"cell_{i}" for i in range(mat.shape[1])]

            rds_path = os.path.join(tmpdir, f"{gse_id}.rds")

            try:
                t_write, rds_size = time_saverds(
                    csc, list(rownames), list(colnames), rds_path, N_TRIALS)
                write_mbps = raw_int32_bytes / t_write / 1e6
                print(f"  saveRDS: {t_write:.3f}s, {rds_size/1e6:.1f}MB, {write_mbps:.0f} MB/s")

                rows.append({
                    "gse_id": gse_id,
                    "nnz": nnz,
                    "raw_int32_bytes": raw_int32_bytes,
                    "rds_write_s": t_write,
                    "rds_bytes": rds_size,
                    "rds_write_mbps": write_mbps,
                })
            except Exception as e:
                print(f"  FAILED: {e}")

            # Cleanup
            if os.path.exists(rds_path):
                os.remove(rds_path)

    # Write CSV
    out_path = os.path.join(DATA_DIR, "r_write_benchmarks.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")
    else:
        print("No results!")


if __name__ == "__main__":
    main()
