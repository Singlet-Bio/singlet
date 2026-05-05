#!/usr/bin/env python3
"""Thread-scaling benchmark for SinglePress manuscript Figure 2(b).

Measures decode throughput across 1–32 threads on 5 datasets spanning a
range of sizes, with 100 replicates per configuration for tight IQR.

Re-encodes each dataset with the CURRENT codec before benchmarking, so
results reflect the encoder version described in the manuscript.

Output: code/data/threading_benchmark_v2.csv

Usage:
    srun --time=120 --mem=32G --cpus-per-task=32 bash -c \\
        'source /mnt/home/debruinz/venv/bin/activate && cd /tmp && \\
         python3 -u <this_script>'
"""

import csv
import gc
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if "" in sys.path:
    sys.path.remove("")

import numpy as np
import singlepress as sp

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TMPFS = "/dev/shm"

# 5 datasets spanning a range of sizes
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


def reencode(gse_id):
    """Re-encode a production .1pz with the current codec to tmpfs."""
    src = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
    dst = os.path.join(TMPFS, f"thread_bench_{gse_id}.1pz")
    if os.path.exists(dst):
        return dst
    print(f"  Re-encoding {gse_id}...", end="", flush=True)
    mat = sp.read_1pz(src)
    pz = sp.open_1pz(src)
    sp.write_1pz(dst, mat.tocsc(),
                  rownames=list(pz.rownames) if pz.rownames else [],
                  colnames=list(pz.colnames) if pz.colnames else [])
    del mat, pz
    gc.collect()
    print(f" {os.path.getsize(dst)/1e6:.1f} MB")
    return dst


def main():
    print(f"Threading benchmark: {len(DATASETS)} datasets x {len(THREADS)} "
          f"thread counts x {N_REPS} reps")
    print(f"singlepress {getattr(sp, '__version__', 'dev')}")

    results = []

    for gse_id, nnz in DATASETS:
        src = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if not os.path.exists(src):
            print(f"SKIP {gse_id}: no counts.1pz")
            continue

        pz_path = reencode(gse_id)

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
            print(f"  threads={nt:>2}: median={med:.3f}s  "
                  f"IQR=[{q25:.3f}, {q75:.3f}]  GB/s={raw_bytes/med/1e9:.3f}")

        # Clean up re-encoded file
        staged = os.path.join(TMPFS, f"thread_bench_{gse_id}.1pz")
        if os.path.exists(staged):
            os.remove(staged)

    # Write CSV
    os.makedirs(CODE_DATA_DIR, exist_ok=True)
    out = os.path.join(CODE_DATA_DIR, "threading_benchmark_v2.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} rows to {out}")


if __name__ == "__main__":
    main()
