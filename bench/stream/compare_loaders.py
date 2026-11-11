# SPDX-License-Identifier: MIT
"""Stream cells into a GPU training step with singlet.torch.AtlasCells or with the CELLxGENE Census loader (tiledbsoma-ml), on the
same machine, for a fixed time, and report cells/s.

``--loader singlet``: human cells of every tissue from data.singlet.bio (HTTP range reads); each batch arrives as a scipy CSR,
becomes a torch sparse CSR tensor on the device (``.to(device, non_blocking=True)``) and goes through ``torch.sparse.mm`` (cuSPARSE
SpMM on CUDA) into a two-layer model over the full gene axis, with backward and an optimizer step.

``--loader census``: primary human cells from the CELLxGENE Census (TileDB-SOMA on S3), ``tiledbsoma_ml.ExperimentDataset`` with
shuffling, raw counts; each batch is dense (the loader's form) and goes through the same model with a dense first layer.

``--step none`` measures the stream alone (no device work). Every run records the host, GPU, library versions and the link
speed measured next to it. The first ``--warmup`` seconds (connection set-up, the Census obs query, worker start) are reported
separately and not counted in the rate.

    python bench/stream/compare_loaders.py --loader singlet --seconds 600 --workers 8 --out singlet.json
    python bench/stream/compare_loaders.py --loader census --seconds 600 --workers 8 --out census.json
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


def model(n_in: int, device, sparse_in: bool):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.randn(n_in, 256, device=device) * 0.01)
            self.body = nn.Sequential(
                nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)
            )

        def forward(self, X):
            h = torch.sparse.mm(X, self.W) if sparse_in else X @ self.W
            return self.body(h)

    net = Net().to(device)
    return net, torch.optim.AdamW(net.parameters(), lr=1e-3)


def run_singlet(a, device):
    from singlet.torch import AtlasCells, plan_samples
    from torch.utils.data import DataLoader

    t = time.perf_counter()
    plan = plan_samples(json.loads(a.where))
    t_plan = time.perf_counter() - t
    sub = plan.sample(frac=1.0, random_state=a.seed).reset_index(
        drop=True
    )  # every sample, a random order
    ds = AtlasCells(
        plan=sub,
        batch_size=a.batch,
        mix_samples=16,
        prefetch=a.prefetch,
        output="csr",
        transform="log1p",
        max_cells_per_sample=a.cap,
        seed=a.seed,
    )
    n_in = len(ds.gene_axis)
    loader = DataLoader(ds, batch_size=None, num_workers=a.workers, persistent_workers=False)
    info = {
        "plan_seconds": round(t_plan, 2),
        "plan_samples": int(len(plan)),
        "plan_cells": int(plan["n_cells"].sum()),
        "genes": n_in,
        "x_format": "torch sparse CSR",
        "prefetch": a.prefetch,
        "max_cells_per_sample": a.cap,
    }
    io = {}

    def batches():
        for b in loader:
            io[b["stream"]["worker"]] = b["stream"]
            yield b["X"], len(b["gsm"]), len(set(b["gse"]))

    return (
        info,
        n_in,
        True,
        batches,
        lambda: {
            "requests": sum(v["requests"] for v in io.values()),
            "bytes": sum(v["bytes"] for v in io.values()),
            "samples": sum(v["samples"] for v in io.values()),
            "samples_skipped": sum(v["samples_skipped"] for v in io.values()),
        },
    )


def run_census(a, device):
    import cellxgene_census
    import tiledbsoma
    import tiledbsoma_ml as tml

    t = time.perf_counter()
    census = cellxgene_census.open_soma(census_version=a.census_version)
    exp = census["census_data"]["homo_sapiens"]
    query = exp.axis_query(
        measurement_name="RNA", obs_query=tiledbsoma.AxisQuery(value_filter=a.census_filter)
    )
    ds = tml.ExperimentDataset(
        query,
        layer_name="raw",
        obs_column_names=["dataset_id"],
        batch_size=a.batch,
        shuffle=True,
        seed=a.seed,
    )
    loader = tml.experiment_dataloader(ds, num_workers=a.workers)
    n_in = int(query.n_vars)
    info = {
        "census_version": a.census_version,
        "filter": a.census_filter,
        "query_cells": int(query.n_obs),
        "genes": n_in,
        "setup_seconds": round(time.perf_counter() - t, 2),
        "x_format": "dense float32 (the loader's batches)",
        "versions": {
            "cellxgene_census": cellxgene_census.__version__,
            "tiledbsoma": tiledbsoma.__version__,
            "tiledbsoma_ml": getattr(tml, "__version__", "?"),
        },
    }

    def batches():
        for X, obs in loader:
            yield X, int(X.shape[0]), int(obs["dataset_id"].nunique())

    return info, n_in, False, batches, lambda: {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--loader", choices=["singlet", "census"], required=True)
    ap.add_argument("--step", choices=["train", "none"], default="train")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--warmup", type=float, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cap", type=int, default=None, help="singlet: max cells per sample")
    ap.add_argument("--where", default='{"organism": "Homo sapiens"}')
    ap.add_argument("--census-version", default="stable")
    ap.add_argument("--census-filter", default="is_primary_data == True")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "loader": a.loader,
        "step": a.step,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "settings": {
            "seconds": a.seconds,
            "warmup": a.warmup,
            "workers": a.workers,
            "batch_size": a.batch,
        },
        "link_before": link_speed(),
    }
    print(json.dumps(report), flush=True)
    t0 = time.perf_counter()
    info, n_in, sparse_in, batches, io = (run_singlet if a.loader == "singlet" else run_census)(
        a, device
    )
    report["setup"] = info
    print(json.dumps(info), flush=True)
    net = opt = None
    if a.step == "train":
        net, opt = model(n_in, device, sparse_in)
    fmt = "csr"
    cells = warm_cells = nb = 0
    mix = []
    step_s = 0.0
    t_first = t_warm = None
    for X, n, studies in batches():
        now = time.perf_counter()
        if t_first is None:
            t_first = now - t0
        if a.step == "train":
            ts = time.perf_counter()
            if sparse_in:
                Xd = X.to(device, non_blocking=True)
                if fmt == "coo":
                    Xd = Xd.to_sparse_coo()
            else:
                Xd = torch.as_tensor(X, dtype=torch.float32).to(device, non_blocking=True)
                Xd = torch.log1p(Xd * (1e4 / Xd.sum(1, keepdim=True).clamp(min=1)))
            try:
                loss = (
                    net(Xd).logsumexp(1).mean()
                )  # a stand-in loss: the point is the data path and the step, not the task
                opt.zero_grad()
                loss.backward()
            except (
                RuntimeError,
                NotImplementedError,
            ) as e:  # SpMM backward with a CSR input is missing in some torch builds
                if not sparse_in or fmt == "coo":
                    raise
                fmt, report["csr_error"] = "coo", repr(e)[:300]
                Xd = Xd.to_sparse_coo()
                loss = net(Xd).logsumexp(1).mean()
                opt.zero_grad()
                loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_s += time.perf_counter() - ts
        if t_warm is None:
            if now - t0 >= a.warmup:
                t_warm = now
                warm_cells = cells
        else:
            mix.append(studies)
        cells += n
        nb += 1
        if t_warm is not None and now - t_warm >= a.seconds:
            break
    t_end = time.perf_counter()
    measured = (t_end - t_warm) if t_warm else float("nan")
    report["result"] = {
        "first_batch_seconds": round(t_first or float("nan"), 2),
        "warmup_cells": warm_cells,
        "measured_seconds": round(measured, 1),
        "measured_cells": cells - warm_cells,
        "cells_per_s": round((cells - warm_cells) / measured, 1) if t_warm else None,
        "batches": nb,
        "sparse_format_on_device": fmt if (sparse_in and a.step == "train") else None,
        "gpu_step_seconds_total": round(step_s, 1),
        "gpu_step_ms_per_batch": round(1e3 * step_s / max(1, nb), 2),
        "studies_per_batch_mean": round(float(np.mean(mix)), 2) if mix else None,
        **io(),
    }
    if report["result"].get("bytes"):
        report["result"]["mb_per_s_overall"] = round(
            report["result"]["bytes"] / 1e6 / (t_end - t0), 2
        )
        report["result"]["bytes_per_cell"] = round(report["result"]["bytes"] / max(1, cells), 1)
    report["link_after"] = link_speed()
    print(json.dumps(report["result"]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
