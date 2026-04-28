#!/usr/bin/env python3
"""Extract write_benchmarks.csv from benchmark_results_v3.json.

Usage:
    python extract_write_benchmarks.py

Reads ../data/benchmark_results_v3.json and writes ../data/write_benchmarks.csv.
"""

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(DATA_DIR, "benchmark_results_v3.json")) as f:
    results = json.load(f)

rows = []
for entry in results:
    row = {
        "gse_id": entry["gse_id"],
        "species": entry["species"],
        "protocol": entry["protocol"],
        "nnz": entry["nnz"],
        "raw_int32_bytes": entry["raw_int32_bytes"],
    }
    fmt_map = {"1pz": "pz", "h5ad_gzip": "h5ad", "10x_h5": "h5", "npz": "npz"}
    for fmt_key, prefix in fmt_map.items():
        fmt = entry["formats"].get(fmt_key, {})
        row[f"{prefix}_bytes"] = fmt.get("size", "")
        row[f"{prefix}_write_s"] = fmt.get("write_s", "")
        if fmt.get("size") and fmt.get("write_s") and fmt["write_s"] > 0:
            row[f"{prefix}_write_mbps"] = entry["raw_int32_bytes"] / fmt["write_s"] / 1e6
        else:
            row[f"{prefix}_write_mbps"] = ""
    rows.append(row)

outpath = os.path.join(DATA_DIR, "write_benchmarks.csv")
fieldnames = list(rows[0].keys())
with open(outpath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {outpath}")
