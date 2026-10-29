# SPDX-License-Identifier: MIT
"""Baseline: a cell-type classifier trained on cells streamed from data.singlet.bio, scored on held-out studies.

The cells come from :class:`singlet.torch.AtlasCells` (HTTP range reads; nothing is downloaded to disk). Splits are held out by
study (``study_split``): no study gives cells to both training and test. The model reads log1p(CP10k) values of a panel of the most
variable genes (chosen from a pilot of training samples only), then two hidden layers and one softmax head per Cell Ontology level
(1 compartment, 2 lineage, 3 cell type). Targets are the Research Alpha per-cell labels at each level (the CL path of the
labeller's call), so the scores measure how well a model trained on streamed atlas cells reproduces the atlas labels on studies it
never saw. They are not accuracy against author annotations; the labeller's own accuracy against those is documented separately.

Classes are numbered as the training stream meets them (heads have a fixed capacity); a test cell whose class never occurred in
training counts as an error, and the report gives that share. The majority-class rate of each level is reported for reference.

    python bench/stream/train_celltype.py --where '{"organism": "Homo sapiens"}' --train-cells 3000000 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _link import link_speed  # noqa: E402

CAPACITY = {1: 64, 2: 256, 3: 1024}


def pick_samples(plan, split: str, cells: int, cap: int, seed: int):
    """A seeded random subset of the split's samples whose (capped) cells add up to about ``cells``."""
    sub = plan[plan["split"] == split].reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(sub))
    n = np.minimum(sub["n_cells"].to_numpy()[order], cap)
    k = int(np.searchsorted(np.cumsum(n), cells)) + 1
    return sub.iloc[np.sort(order[:k])].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--where", default='{"organism": "Homo sapiens"}')
    ap.add_argument("--train-cells", type=int, default=3_000_000)
    ap.add_argument("--test-cells", type=int, default=300_000)
    ap.add_argument("--cap-train", type=int, default=1000, help="max cells per training sample")
    ap.add_argument("--cap-test", type=int, default=300, help="max cells per test sample")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--panel", type=int, default=4096, help="most variable genes the model reads")
    ap.add_argument(
        "--pilot-samples", type=int, default=40, help="training samples the panel is chosen from"
    )
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--prefetch", type=int, default=4, help="reader threads per worker process")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    ap.add_argument("--mix", type=int, default=16)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from singlet.torch import AtlasCells, plan_samples, study_split

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    where = json.loads(a.where)
    link_before = link_speed()
    t_start = time.perf_counter()
    plan = plan_samples(where)
    plan["split"] = [study_split(g, seed=a.split_seed) for g in plan["gse"]]
    by_split = {
        s: {
            "studies": int(d["gse"].nunique()),
            "samples": int(len(d)),
            "cells": int(d["n_cells"].sum()),
        }
        for s, d in plan.groupby("split")
    }
    train_plan = pick_samples(plan, "train", a.train_cells, a.cap_train, a.seed)
    test_plan = pick_samples(plan, "test", a.test_cells, a.cap_test, a.seed + 1)
    assert set(train_plan["gse"]).isdisjoint(test_plan["gse"])
    print(
        json.dumps(
            {"splits": by_split, "train_samples": len(train_plan), "test_samples": len(test_plan)}
        ),
        flush=True,
    )

    kw = dict(
        batch_size=a.batch,
        mix_samples=a.mix,
        prefetch=a.prefetch,
        output="scipy",
        levels=(1, 2, 3),
        transform=None,
    )
    train = AtlasCells(where, plan=train_plan, max_cells_per_sample=a.cap_train, seed=a.seed, **kw)
    n_genes = len(train.gene_axis)
    test = AtlasCells(
        where,
        plan=test_plan,
        max_cells_per_sample=a.cap_test,
        seed=a.seed,
        genes=train.gene_axis.ids,
        **kw,
    )

    # gene panel: the most variable genes (variance of log1p CP10k) in a pilot of training samples; test studies are not looked at
    from singlet.torch._stream import log1p_cp10k

    t_p = time.perf_counter()
    pilot = AtlasCells(
        where,
        plan=train_plan.head(a.pilot_samples),
        max_cells_per_sample=a.cap_train,
        seed=a.seed,
        **kw,
    )
    s1 = np.zeros(n_genes)
    s2 = np.zeros(n_genes)
    n_pilot = 0
    for b in pilot:
        Xp = log1p_cp10k(b["X"].astype(np.float32))
        s1 += np.asarray(Xp.sum(0)).ravel()
        s2 += np.asarray(Xp.multiply(Xp).sum(0)).ravel()
        n_pilot += Xp.shape[0]
    var = s2 / n_pilot - (s1 / n_pilot) ** 2
    panel = np.sort(np.argsort(-var)[: a.panel])
    pilot_info = {
        "samples": int(min(a.pilot_samples, len(train_plan))),
        "cells": n_pilot,
        "seconds": round(time.perf_counter() - t_p, 1),
        "bytes": pilot.stats()["bytes"],
    }
    print(json.dumps({"pilot": pilot_info}), flush=True)

    def prep(X):
        """log1p CP10k over all genes, then the panel columns, as a dense tensor."""
        X = log1p_cp10k(X.astype(np.float32))
        return torch.from_numpy(X[:, panel].toarray())

    classes = {k: {} for k in CAPACITY}

    def targets(b, grow: bool):
        out = {}
        for k in CAPACITY:
            ids = np.full(len(b["gsm"]), -100, np.int64)
            cmap = classes[k]
            for i, c in enumerate(b[f"cell_type_l{k}"]):
                if c is None:
                    continue
                j = cmap.get(c)
                if j is None and grow and len(cmap) < CAPACITY[k]:
                    j = cmap[c] = len(cmap)
                ids[i] = -1 if j is None else j  # -1: a class training never had (always wrong)
            out[k] = ids
        return out

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(len(panel), a.hidden),
                nn.LayerNorm(a.hidden),
                nn.ReLU(),
                nn.Linear(a.hidden, a.hidden),
                nn.ReLU(),
            )
            self.heads = nn.ModuleDict(
                {str(k): nn.Linear(a.hidden, c) for k, c in CAPACITY.items()}
            )

        def forward(self, X):
            h = self.body(X)
            return {k: self.heads[str(k)](h) for k in CAPACITY}

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    curve, seen, t0 = [], 0, time.perf_counter()
    from torch.utils.data import DataLoader

    class IO:
        """Requests, bytes and samples added up over the DataLoader worker processes (last report of each)."""

        def __init__(self):
            self.last, self.cells = {}, 0

        def add(self, b):
            self.last[b["stream"]["worker"]] = b["stream"]
            self.cells += len(b["gsm"])

        def stats(self):
            v = self.last.values()
            return {
                "cells": self.cells,
                "requests": sum(x["requests"] for x in v),
                "bytes": sum(x["bytes"] for x in v),
                "samples": sum(x["samples"] for x in v),
                "samples_skipped": sum(x["samples_skipped"] for x in v),
            }

    def loader(ds):
        return DataLoader(ds, batch_size=None, num_workers=a.workers)

    io_train, io_test = IO(), IO()
    for step, b in enumerate(loader(train)):
        io_train.add(b)
        y = targets(b, grow=True)
        out = net(prep(b["X"]))
        loss = sum(ce(out[k], torch.from_numpy(y[k])) for k in CAPACITY if (y[k] >= 0).any())
        opt.zero_grad()
        loss.backward()
        opt.step()
        seen += len(b["gsm"])
        if step % 200 == 0:
            el = time.perf_counter() - t0
            curve.append(
                {"cells": seen, "loss": round(float(loss.detach()), 4), "seconds": round(el, 1)}
            )
            print(json.dumps(curve[-1] | {"cells_per_s": round(seen / el, 1)}), flush=True)
    t_train = time.perf_counter() - t0
    train_stats = io_train.stats()

    # evaluation on the held-out studies
    net.eval()
    right = {k: 0 for k in CAPACITY}
    total = {k: 0 for k in CAPACITY}
    unseen = {k: 0 for k in CAPACITY}
    per_study = {k: defaultdict(lambda: [0, 0]) for k in CAPACITY}
    test_label_counts = {k: Counter() for k in CAPACITY}
    t1 = time.perf_counter()
    with torch.no_grad():
        for b in loader(test):
            io_test.add(b)
            y = targets(b, grow=False)
            out = net(prep(b["X"]))
            for k in CAPACITY:
                pred = out[k].argmax(1).numpy()
                lab = np.asarray(b[f"cell_type_l{k}"], dtype=object)
                m = y[k] != -100
                ok = (pred == y[k]) & (y[k] >= 0) & m
                right[k] += int(ok.sum())
                total[k] += int(m.sum())
                unseen[k] += int((y[k] == -1).sum())
                test_label_counts[k].update(lab[m].tolist())
                for g, mm, o in zip(b["gse"], m, ok):
                    if mm:
                        per_study[k][g][0] += int(o)
                        per_study[k][g][1] += 1
    t_test = time.perf_counter() - t1
    test_stats = io_test.stats()

    # per level: overall and per-study accuracy, and the share of the most common test label (a constant predictor's best case)
    levels = {}
    for k in CAPACITY:
        inv = {v: c for c, v in classes[k].items()}
        acc_s = [r / n for r, n in per_study[k].values() if n >= 20]
        maj = (
            max(test_label_counts[k].items(), key=lambda kv: kv[1])[0]
            if test_label_counts[k]
            else None
        )
        levels[str(k)] = {
            "name": {1: "compartment", 2: "lineage", 3: "cell type"}[k],
            "test_cells_with_label": total[k],
            "accuracy": round(right[k] / max(1, total[k]), 4),
            "per_study_accuracy_mean": round(float(np.mean(acc_s)), 4) if acc_s else None,
            "per_study_accuracy_median": round(float(np.median(acc_s)), 4) if acc_s else None,
            "studies_scored": len(acc_s),
            "classes_trained": len(inv),
            "test_cells_unseen_class": unseen[k],
            "test_majority_class_share": round(test_label_counts[k][maj] / max(1, total[k]), 4)
            if maj
            else None,
        }
    report = {
        "where": where,
        "target": "Research Alpha per-cell labels (CL path of the labeller's call), levels 1-3",
        "splits_by_study": by_split,
        "split_seed": a.split_seed,
        "train": {
            "studies": int(train_plan["gse"].nunique()),
            "samples": int(len(train_plan)),
            "cells": train_stats["cells"],
            "seconds": round(t_train, 1),
            "cells_per_s": round(train_stats["cells"] / t_train, 1),
            "requests": train_stats["requests"],
            "bytes": train_stats["bytes"],
            "mb_per_s": round(train_stats["bytes"] / 1e6 / t_train, 2),
            "samples_skipped": train_stats["samples_skipped"],
            "max_cells_per_sample": a.cap_train,
            "dataloader_workers": a.workers,
            "reader_threads_per_worker": a.prefetch,
        },
        "test": {
            "studies": int(test_plan["gse"].nunique()),
            "samples": int(len(test_plan)),
            "cells": test_stats["cells"],
            "seconds": round(t_test, 1),
            "requests": test_stats["requests"],
            "bytes": test_stats["bytes"],
            "samples_skipped": test_stats["samples_skipped"],
            "max_cells_per_sample": a.cap_test,
        },
        "levels": levels,
        "model": {
            "gene_axis": n_genes,
            "panel_genes": int(len(panel)),
            "panel": "highest variance of log1p(CP10k) over the pilot (training samples only)",
            "pilot": pilot_info,
            "input": "log1p(counts per 10k over all genes), panel columns",
            "hidden": a.hidden,
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "optimizer": f"AdamW lr={a.lr} wd=1e-4",
            "batch_size": a.batch,
            "epochs": 1,
        },
        "curve": curve,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "link": {"before": link_before, "after": link_speed()},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(levels, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
