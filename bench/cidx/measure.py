# SPDX-License-Identifier: MIT
"""Latency, requests and bytes of the cell-index endpoints against a running data Worker (local workerd or data.singlet.bio).

    node <website>/workers/data-v2/scripts/local.mjs <data dir with v2/... and v2/cidx/...> 8799      # local workerd + seeded R2
    python -P bench/cidx/measure.py http://127.0.0.1:8799 > bench/cidx/results_local.json

Each case runs ``--repeat`` times; the first run is cold (empty Cache API and isolate memos for the objects it touches), the rest warm.
Gathers run with and without the client's prefix cache. For comparison, the same gathers are planned for a client that does its own
range requests (``via="direct"``: one request per coalesced read, from outside Cloudflare).
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
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
    return time.perf_counter() - t, raw


def main() -> None:
    base = sys.argv[1].rstrip("/")
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp()
    from singlet.v2 import atlas

    out = {"base": base, "queries": [], "gathers": []}
    queries = [
        ("count, summary (cell type)", {"where": {"cell_type": "T cell"}, "return": "count"}),
        (
            "count by cell type x sex, summary",
            {"where": {"organism": "Homo sapiens"}, "return": "count", "by": ["cell_type", "sex"]},
        ),
        (
            "count, row scan (cell type + n_umi)",
            {
                "where": {"cell_type": "CL:0000625", "ranges": {"n_umi": [150, None]}},
                "return": "count",
            },
        ),
        (
            "cells, 1000 (cell type + score)",
            {
                "where": {"cell_type": "CL:0000084", "ranges": {"cell_type_score": [0.5, None]}},
                "return": "cells",
                "limit": 1000,
            },
        ),
        (
            "cells, sample set (postings)",
            {
                "where": {
                    "samples": ["GSM4317426", "GSM4317430"],
                    "ranges": {"n_umi": [100, None]},
                },
                "return": "cells",
                "limit": 100000,
            },
        ),
        ("cells, everything (40k)", {"where": {}, "return": "cells", "limit": 100000}),
    ]
    for name, q in queries:
        runs = []
        for k in range(repeat):
            dt, raw = post(base, "/v2/cidx/query", q)
            r = json.loads(raw)
            runs.append(
                {
                    "ms_client": round(dt * 1000, 1),
                    "ms_worker": r.get("ms"),
                    "reads": r.get("reads"),
                    "bytes_read": r.get("bytes_read"),
                    "cache_hits": r.get("cache_hits"),
                    "rows_scanned": r.get("rows_scanned"),
                    "response_bytes": len(raw),
                    "n": r.get("count", r.get("n")),
                }
            )
        out["queries"].append({"name": name, "query": q, "runs": runs})
        print(name, runs[0], runs[-1], file=sys.stderr)

    idx = atlas.Index.open(base=base)
    cases = [
        ("1 cell", {"cell_type": "CL:0000451", "ranges": {"n_umi": [1000, None]}}, 1),
        ("100 cells, 14 samples", {"cell_type": "CL:0000623"}, 100),
        ("1,000 cells", {"cell_type": "CL:0000084"}, 1000),
        ("10,000 cells", {}, 10000),
        ("all 40,614 cells", {}, 100000),
    ]
    for name, where, lim in cases:
        df = idx.find(where, limit=None)
        if lim < len(df):
            df = df.sample(lim, random_state=0).sort_values(["so", "col"])
        want = atlas._cells_by_sample(idx, df)
        ent = [idx.sample(s) for s in sorted(want)]
        cold_plan = atlas.plan_gather(ent, want, atlas.members_for(["counts"]))
        warm_plan = atlas.plan_gather(
            ent,
            want,
            atlas.members_for(["counts"]),
            [m["digest"] for e in ent for m in e["members"].values()],
        )
        runs = []
        os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp()  # no prefixes on the client
        for k in range(repeat):
            t = time.perf_counter()
            a = atlas.gather(df, base=base, via="worker", index=idx)
            dt = time.perf_counter() - t
            tr = a.uns["singlet_gather"]["trailer"]
            runs.append(
                {
                    "s_total": round(dt, 3),
                    "s_fetch": a.uns["singlet_gather"]["seconds_fetch"],
                    "worker_ms": tr.get("ms"),
                    "reads": tr.get("reads"),
                    "bytes_read": tr.get("bytes_read"),
                    "bytes_sent": tr.get("bytes_sent"),
                    "cache_hits": tr.get("cache_hits"),
                    "prefix_cached": k > 0,
                }
            )
        out["gathers"].append(
            {
                "name": name,
                "cells": int(len(df)),
                "samples": len(want),
                "plan_no_prefix_cache": {
                    "bytes": cold_plan["bytes"],
                    "reads": len(cold_plan["reads"]),
                    "payload": cold_plan["payload"],
                },
                "plan_prefix_cached": {
                    "bytes": warm_plan["bytes"],
                    "reads": len(warm_plan["reads"]),
                    "payload": warm_plan["payload"],
                },
                "direct_client_requests": len(cold_plan["reads"])
                + len({e["so"] // 256 for e in ent}),
                "runs": runs,
            }
        )
        print(name, len(df), runs[0], runs[-1], file=sys.stderr)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
