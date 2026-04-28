#!/usr/bin/env python3
"""Threading benchmark with 100 replicates per configuration.

5 datasets × 6 thread levels × 100 reps = 3000 reads.
Output: threading_benchmark_v2.csv

Usage:
    srun --time=120 --mem=32G --cpus-per-task=32 bash -c \
        'source /mnt/home/debruinz/venv/bin/activate && cd /tmp && \
         python3 -u /path/to/benchmark_threading_v2.py'
"""
import csv
import os
import sys
import time
import gc

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Avoid namespace shadowing
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')

import numpy as np
import singlepress as sp

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "threading_benchmark_v2.csv")

# Same 5 datasets as original benchmark, spanning range of sizes
DATASETS = [
    ("GSE189042", 20_016_693),
    ("GSE290932", 32_590_922),
    ("GSE142483", 50_452_624),
    ("GSE207157", 73_987_183),
    ("GSE248138", 112_694_589),
]

THREADS = [1, 2, 4, 8, 16, 32]
N_REPS = 100
WARMUP = 3

results = []

for gse_id, nnz in DATASETS:
    pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    if not os.path.exists(pz_path):
        print(f"SKIP {gse_id}: no counts.1pz")
        continue

    print(f"\n{'='*60}")
    print(f"{gse_id}  nnz={nnz:,}")
    print(f"{'='*60}")

    for nt in THREADS:
        # Warmup
        for _ in range(WARMUP):
            _ = sp.read_1pz(pz_path, num_threads=nt)
            gc.collect()

        times = []
        for rep in range(N_REPS):
            gc.collect()
            t0 = time.perf_counter()
            _ = sp.read_1pz(pz_path, num_threads=nt)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        times = np.array(times)
        med = np.median(times)
        q25, q75 = np.percentile(times, [25, 75])
        raw_bytes = nnz * 4  # int32

        results.append({
            "gse_id": gse_id,
            "nnz": nnz,
            "n_threads": nt,
            "n_reps": N_REPS,
            "median_s": round(med, 4),
            "q25_s": round(q25, 4),
            "q75_s": round(q75, 4),
            "mean_s": round(np.mean(times), 4),
            "std_s": round(np.std(times), 4),
            "read_gbps": round(raw_bytes / med / 1e9, 3),
        })
        print(f"  threads={nt:>2}: median={med:.3f}s  IQR=[{q25:.3f}, {q75:.3f}]  "
              f"GB/s={raw_bytes/med/1e9:.3f}")

# Write CSV
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)

print(f"\nSaved {len(results)} rows to {OUT}")
