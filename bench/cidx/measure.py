# SPDX-License-Identifier: MIT
"""Latency, reads, bytes and cost of the cell-index endpoints against a running data Worker (local workerd or data.singlet.bio).

    node <website>/workers/data/scripts/local.mjs <data dir with store/... and store/cidx/...> 8799      # local workerd + seeded R2
    python -P bench/cidx/measure.py http://127.0.0.1:8799 > bench/cidx/results_local_pilot.json

Each case runs ``repeat`` times; the first run is cold (empty Cache API and isolate memos for the objects it touches), the rest warm.
Every answer carries the Worker's worst-case ``estimate`` (all reads missing the cache) and its ``cost`` from the reads that did miss;
both must stay under the per-request budget ($0.01).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def post(base, path, body):
    t = time.perf_counter()
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
    return time.perf_counter() - t, raw


def main() -> None:
    base = sys.argv[1].rstrip("/")
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp()
    from singlet.store import atlas

    out = {"base": base, "queries": [], "gathers": []}
    queries = [
        ("count, summary (cell type)", {"where": {"cell_type": "T cell"}, "return": "count"}),
        ("count by cell type x sex, summary", {"where": {"organism": "Homo sapiens"}, "return": "count", "by": ["cell_type", "sex"]}),
        ("count, row scan (cell type + n_umi)", {"where": {"cell_type": "CL:0000625", "ranges": {"n_umi": [150, None]}}, "return": "count"}),
        ("count, feature range", {"where": {"ranges": {"hypoxia": [0.5, None], "malignancy": [None, 0.3]}}, "return": "count"}),
        ("sample 1,000 cells (cell type + score), uniform", {"where": {"cell_type": "CL:0000084", "ranges": {"cell_type_score": [0.5, None]}}, "return": "cells", "n": 1000, "seed": 1}),
        ("sample 1,000 cells, chunks", {"where": {"cell_type": "CL:0000084", "ranges": {"cell_type_score": [0.5, None]}}, "return": "cells", "n": 1000, "seed": 1, "sampling": "chunks"}),
        ("sample 10,000 of 40,614 cells", {"where": {}, "return": "cells", "n": 10000, "seed": 2}),
        ("dry run of a sample", {"where": {}, "return": "cells", "n": 10000, "dry_run": True}),
    ]
    for name, q in queries:
        runs = []
        for _k in range(repeat):
            dt, raw = post(base, "/store/cidx/query", q)
            r = json.loads(raw)
            runs.append({
                "ms_client": round(dt * 1000, 1), "ms_worker": r.get("ms"), "reads": r.get("reads"), "bytes_read": r.get("bytes_read"),
                "cache_hits": r.get("cache_hits"), "rows_scanned": r.get("rows_scanned"), "response_bytes": len(raw),
                "n": r.get("count", r.get("n")), "eligible": r.get("eligible"),
                "estimate_usd": (r.get("estimate") or (r.get("plan") or {}).get("estimate") or {}).get("usd"), "cost_usd": (r.get("cost") or {}).get("usd"),
            })
        out["queries"].append({"name": name, "query": q, "runs": runs})
        print(name, runs[0], runs[-1], file=sys.stderr)

    idx = atlas.Index.open(base=base)
    cases = [
        ("1 cell", {"cell_type": "CL:0000451", "ranges": {"n_umi": [1000, None]}}, 1, "uniform"),
        ("100 cells", {"cell_type": "CL:0000623"}, 100, "uniform"),
        ("1,000 cells, uniform", {"cell_type": "CL:0000084"}, 1000, "uniform"),
        ("1,000 cells, chunks", {"cell_type": "CL:0000084"}, 1000, "chunks"),
        ("10,000 cells, uniform", {}, 10000, "uniform"),
        ("10,000 cells, chunks", {}, 10000, "chunks"),
    ]
    for name, where, n, sampling in cases:
        df, _info = idx.sample_cells(where, n=n, seed=3, sampling=sampling)
        want = atlas._cells_by_sample(idx, df)
        ent = [idx.sample(s) for s in sorted(want)]
        cold_plan = atlas.plan_gather(ent, want, atlas.members_for(["counts"]))
        runs = []
        os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp()  # no prefixes on the client
        for k in range(repeat):
            t = time.perf_counter()
            a = atlas.gather(df, base=base, via="worker", index=idx)
            dt = time.perf_counter() - t
            g = a.uns["singlet_gather"]
            tr = g["trailer"]
            runs.append({"s_total": round(dt, 3), "s_fetch": g["seconds_fetch"], "worker_ms": tr.get("ms"), "reads": tr.get("reads"), "bytes_read": tr.get("bytes_read"),
                         "bytes_sent": tr.get("bytes_sent"), "cache_hits": tr.get("cache_hits"), "cost_usd": (tr.get("cost") or {}).get("usd"), "prefix_cached": k > 0})
        est = atlas.estimate_cost(reads=len(cold_plan["reads"]), mb=cold_plan["bytes"] / 1e6, do_calls=1, do_writes=1)
        out["gathers"].append({
            "name": name, "cells": int(len(df)), "samples": len(want), "sampling": sampling,
            "plan_no_prefix_cache": {"bytes": cold_plan["bytes"], "reads": len(cold_plan["reads"]), "payload": cold_plan["payload"]},
            "estimate_usd": est["usd"],
            "runs": runs,
        })
        print(name, len(df), runs[0], runs[-1], file=sys.stderr)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
