# SPDX-License-Identifier: MIT
"""Streaming throughput of singlet.torch.AtlasCells from data.singlet.bio (HTTP range reads, nothing downloaded to disk).

For one atlas filter, plans the samples through the cell index, then streams a seeded random subset of them for each
``--workers`` (DataLoader worker processes) x ``--prefetch`` (reader threads per process) setting and records cells/s, MB/s, requests, bytes per cell and how many studies a batch mixes. Writes a JSON
report (``--out``). Every number is measured; the machine, Python, torch and the network location are recorded with it.

    python bench/stream/throughput.py --where '{"organism": "Homo sapiens"}' --samples 300 --workers 0 4 8 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _link import link_speed  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--where", default='{"organism": "Homo sapiens"}', help="cell-index filter (JSON)"
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=300,
        help="samples streamed per setting (seeded random subset of the plan)",
    )
    ap.add_argument(
        "--prefetch", type=int, nargs="+", default=[4], help="reader threads per process"
    )
    ap.add_argument(
        "--workers", type=int, nargs="+", default=[0, 4, 8], help="DataLoader worker processes"
    )
    ap.add_argument("--mix", type=int, default=16, help="mix_samples")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-cells-per-sample", type=int, default=None)
    ap.add_argument("--output", default="csr", choices=["dense", "csr", "scipy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from singlet.torch import AtlasCells, plan_samples

    where = json.loads(a.where)
    t = time.perf_counter()
    plan = plan_samples(where)
    t_plan = time.perf_counter() - t
    rng = np.random.default_rng(a.seed)
    report = {
        "where": where,
        "plan": {
            "seconds": round(t_plan, 2),
            "studies": int(plan["gse"].nunique()),
            "samples": int(len(plan)),
            "cells": int(plan["n_cells"].sum()),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "data_base": os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio"),
        },
        "settings": {
            "batch_size": a.batch,
            "mix_samples": a.mix,
            "output": a.output,
            "max_cells_per_sample": a.max_cells_per_sample,
        },
        "runs": [],
    }
    print(json.dumps(report["plan"]), flush=True)
    report["link_before"] = link_speed()
    print(json.dumps({"link": report["link_before"]}), flush=True)
    from torch.utils.data import DataLoader

    configs = [(w, pf) for w in a.workers for pf in a.prefetch]
    for i, (nw, pf) in enumerate(configs):
        # a different random subset per setting, so no run reads what an earlier one already fetched
        pick = rng.choice(len(plan), size=min(a.samples, len(plan)), replace=False)
        sub = plan.iloc[np.sort(pick)].reset_index(drop=True)
        ds = AtlasCells(
            where,
            plan=sub,
            batch_size=a.batch,
            mix_samples=a.mix,
            prefetch=pf,
            output=a.output,
            max_cells_per_sample=a.max_cells_per_sample,
            seed=a.seed + i,
        )
        _ = ds.gene_axis
        loader = DataLoader(ds, batch_size=None, num_workers=nw)
        mixes, io, cells, batches = [], {}, 0, 0
        t0 = time.perf_counter()
        first = None
        for b in loader:
            if first is None:
                first = time.perf_counter() - t0
            mixes.append(len(set(b["gse"])))
            io[b["stream"]["worker"]] = b["stream"]
            cells += len(b["gsm"])
            batches += 1
        wall = time.perf_counter() - t0
        req = sum(v["requests"] for v in io.values())
        byt = sum(v["bytes"] for v in io.values())
        n_s = sum(v["samples"] for v in io.values())
        run = {
            "workers": nw,
            "prefetch": pf,
            "samples": int(len(sub)),
            "studies": int(sub["gse"].nunique()),
            "samples_read": n_s,
            "samples_skipped": sum(v["samples_skipped"] for v in io.values()),
            "cells": cells,
            "batches": batches,
            "requests": req,
            "bytes": byt,
            "wall_seconds": round(wall, 2),
            "first_batch_seconds": round(first or float("nan"), 2),
            "cells_per_s": round(cells / wall, 1),
            "mb_per_s": round(byt / 1e6 / wall, 2),
            "bytes_per_cell": round(byt / max(1, cells), 1),
            "requests_per_sample": round(req / max(1, n_s), 2),
            "studies_per_batch_mean": round(float(np.mean(mixes)), 2) if mixes else None,
        }
        report["runs"].append(run)
        print(json.dumps(run), flush=True)
    report["link_after"] = link_speed()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
