#!/usr/bin/env python3
"""
Fast read-only benchmarks: (1) .1pz read throughput across ~200 datasets,
(2) thread scaling on 10 representative datasets.
No file writing (H5AD, etc) — uses existing v3 benchmark data for format comparison.
"""
import csv, os, sys, time, resource
import numpy as np

_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _ws]
if '' in sys.path:
    sys.path.remove('')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import singlepress as sp

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_throughput_bench(n_sample=200, n_trials=3):
    """Benchmark .1pz read throughput on a stratified sample."""
    # Load survey for stratification
    import pandas as pd
    survey = pd.read_csv(os.path.join(SCRIPT_DIR, "all_datasets_survey.csv"))
    survey = survey.sort_values("nnz").reset_index(drop=True)
    # Filter to nnz > 100K and < 300M for reasonable timing
    survey = survey[(survey["nnz"] > 100_000) & (survey["nnz"] < 300_000_000)]
    step = max(1, len(survey) // n_sample)
    sample = survey.iloc[::step].head(n_sample).copy()
    print(f"Read throughput benchmark: {len(sample)} datasets")

    results = []
    for idx, row in sample.iterrows():
        gse_id = row["gse_id"]
        pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
        if not os.path.isfile(pz_path):
            continue
        nnz = int(row["nnz"])
        raw_bytes = int(row["raw_int32_bytes"])

        try:
            times = []
            for trial in range(n_trials + 1):
                t0 = time.perf_counter()
                mat = sp.read_1pz(pz_path)
                t1 = time.perf_counter()
                if trial > 0:
                    times.append(t1 - t0)
                del mat

            t = float(np.median(times))
            results.append({
                "gse_id": gse_id,
                "species": row["species"],
                "protocol": row["protocol"],
                "nrows": int(row["nrows"]),
                "ncols": int(row["ncols"]),
                "nnz": nnz,
                "raw_int32_bytes": raw_bytes,
                "pz_bytes": int(row["pz_bytes"]),
                "ratio": float(row["ratio"]),
                "read_s": round(t, 4),
                "read_gbps": round(raw_bytes / t / 1e9, 3),
                "read_mbps": round(raw_bytes / t / 1e6, 1),
            })
            if len(results) % 25 == 0:
                print(f"  [{len(results)}/{len(sample)}] last: {gse_id} {nnz:,} nnz → {results[-1]['read_gbps']:.2f} GB/s")
        except Exception as e:
            if len(results) < 5:
                print(f"  ERROR {gse_id}: {e}")

    out = os.path.join(SCRIPT_DIR, "read_throughput.csv")
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
    print(f"Read throughput saved: {out} ({len(results)} rows)")
    return results


def threading_bench(n_threads_list=(1, 2, 4, 8)):
    """Thread scaling on 10 representative datasets."""
    import pandas as pd
    survey = pd.read_csv(os.path.join(SCRIPT_DIR, "all_datasets_survey.csv"))
    # Pick 10 datasets: 5M-50M nnz, spread
    cands = survey[(survey["nnz"] > 5_000_000) & (survey["nnz"] < 50_000_000)].sort_values("nnz")
    step = max(1, len(cands) // 10)
    picked = cands.iloc[::step].head(10)
    print(f"\nThreading benchmark: {len(picked)} datasets × {len(n_threads_list)} thread configs")

    results = []
    for nt in n_threads_list:
        os.environ["OMP_NUM_THREADS"] = str(nt)
        # Force singlepress to pick up new thread count
        print(f"  OMP_NUM_THREADS={nt}")
        for _, row in picked.iterrows():
            gse_id = row["gse_id"]
            pz_path = os.path.join(QUANT_DIR, gse_id, "counts.1pz")
            nnz = int(row["nnz"])
            raw_bytes = int(row["raw_int32_bytes"])

            times = []
            for trial in range(4):
                t0 = time.perf_counter()
                sp.read_1pz(pz_path)
                t1 = time.perf_counter()
                if trial > 0:
                    times.append(t1 - t0)
            t = float(np.median(times))
            results.append({
                "gse_id": gse_id,
                "nnz": nnz,
                "n_threads": nt,
                "read_s": round(t, 4),
                "read_gbps": round(raw_bytes / t / 1e9, 3),
                "read_mbps": round(raw_bytes / t / 1e6, 1),
            })
        print(f"    Done: {[r['read_gbps'] for r in results[-len(picked):]]}")

    os.environ["OMP_NUM_THREADS"] = "8"
    out = os.path.join(SCRIPT_DIR, "threading_benchmarks.csv")
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
    print(f"Threading saved: {out} ({len(results)} rows)")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Fast Benchmarks")
    print("=" * 60)
    read_throughput_bench(n_sample=200, n_trials=3)
    threading_bench([1, 2, 4, 8])
    print(f"\nPeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
