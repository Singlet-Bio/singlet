#!/usr/bin/env python3
"""Aggregate the M8 JSONL run files into bench/gpu/results/m8.json and a markdown table.

Usage: make_m8_results.py OUT_JSON RUNS1.jsonl [RUNS2.jsonl ...]
Each input line is one JSON object: a pipeline run ("bench": "m8"), a copy, a bandwidth probe or a node-load line.
"""
import json
import statistics
import sys


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf8"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d["_src"] = p
                rows.append(d)
    return rows


def slim(r):
    """Per-run record without the per-batch arrays."""
    keep = ["label", "file", "threads", "spin_ns", "qual", "dev", "pairs", "batches", "payload_B_per_pair", "wall_s",
            "pairs_per_s_M_total", "pairs_per_s_M_steady", "gpu_idle_pct_steady", "kernel_ms_mean", "h2d_ms_mean", "h2d_GBps",
            "host_cores_avg", "host_core_s_per_Mpair", "decode_core_s_per_Mpair", "host_user_s", "host_sys_s", "thread_s",
            "per_thread_Mpairs_per_s_active", "per_thread_ns_per_pair", "file_B_per_pair", "file_read_MBps", "submitter_wait_s_steady",
            "cache_resident_before", "cache_resident_after_dontneed", "reads_with_N", "truncated", "unknown_bc",
            "verify_mismatch_batches", "epoch_start", "epoch_end", "crc", "prefetch", "dontneed"]
    return {k: r[k] for k in keep if k in r}


def main():
    out = sys.argv[1]
    rows = load(sys.argv[2:])
    runs = [r for r in rows if r.get("bench") == "m8"]
    res = {
        "runs": [slim(r) for r in runs],
        "copies": [r for r in rows if "copy_src" in r],
        "bandwidth": [r for r in rows if "bw_dev" in r],
        "node_load": [r for r in rows if "node_load" in r],
    }
    json.dump(res, open(out, "w", encoding="utf8"), indent=1)
    print(f"{len(runs)} runs, {len(res['copies'])} copies, {len(res['bandwidth'])} bw, {len(res['node_load'])} node-load lines -> {out}")


if __name__ == "__main__":
    main()
