# SPDX-License-Identifier: MIT
"""Collect the streaming benchmark results into one JSON for figures and reports (docs/ml-ready/preliminary.json).

Only copies and derives from the measured result files in bench/stream/results/; no number is typed in here.

    python bench/stream/summarize.py --out docs/ml-ready/preliminary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train", default=str(HERE / "results" / "train_celltype_human_laptop.json"))
    ap.add_argument("--throughput", default=str(HERE / "results" / "throughput_human_laptop.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tr = json.loads(Path(a.train).read_text(encoding="utf-8"))
    th = json.loads(Path(a.throughput).read_text(encoding="utf-8"))

    best = max(th["runs"], key=lambda r: r["cells_per_s"])
    out = {
        "what": "Singlet Research Alpha streamed into PyTorch by HTTP range reads (singlet.torch.AtlasCells); nothing downloaded to disk",
        "source": "https://data.singlet.bio (per-sample .s1 objects) and the atlas cell index (POST /s1/cidx/query)",
        "filter": tr["where"],
        "splits_by_study": {
            "rule": "SHA-256 of (split_seed, GSE) cut at 0.8 / 0.1 / 0.1; a study's cells are all in one split",
            "split_seed": tr["split_seed"],
            **tr["splits_by_study"],
        },
        "throughput": {
            "host": th["host"],
            "link_mb_per_s": {
                "before": th.get("link_before", {}).get("mb_per_s"),
                "after": th.get("link_after", {}).get("mb_per_s"),
            },
            "plan": th["plan"],
            "settings": th["settings"],
            "runs": th["runs"],
            "best": {
                k: best[k]
                for k in (
                    "workers",
                    "prefetch",
                    "cells_per_s",
                    "mb_per_s",
                    "bytes_per_cell",
                    "requests_per_sample",
                    "studies_per_batch_mean",
                )
            },
        },
        "baseline": {
            "task": "predict the atlas cell-type label at Cell Ontology levels 1-3 from log1p(CP10k) expression",
            "target": tr["target"],
            "train": tr["train"],
            "test": tr["test"],
            "model": tr["model"],
            "link_mb_per_s": {k: v.get("mb_per_s") for k, v in tr.get("link", {}).items()},
            "held_out_study_accuracy": [
                {
                    "level": int(k),
                    "name": v["name"],
                    "accuracy": v["accuracy"],
                    "per_study_mean": v["per_study_accuracy_mean"],
                    "per_study_median": v["per_study_accuracy_median"],
                    "studies": v["studies_scored"],
                    "cells": v["test_cells_with_label"],
                    "classes_trained": v["classes_trained"],
                    "unseen_class_cells": v["test_cells_unseen_class"],
                    "majority_class_share": v["test_majority_class_share"],
                }
                for k, v in sorted(tr["levels"].items(), key=lambda kv: int(kv[0]))
            ],
            "curve": tr["curve"],
        },
        "caveats": [
            "The targets are the Research Alpha labeller's calls, not author annotations: the scores measure how well a model "
            "trained on streamed atlas cells reproduces the atlas labels on studies it never saw. The labeller's own per-study "
            "accuracy against CELLxGENE author labels is 0.84 / 0.75 / 0.43 / 0.32 (levels 1-4; docs/cell_annotation.md).",
            "Throughput was measured from a laptop on a residential link; the link capacity measured next to each run is reported, "
            "and the stream ran at about that capacity. A cluster node was not used (Alpha jobs had priority on Clipper).",
            "A few samples the cell index lists are not in the released study manifests (withheld studies, excluded samples); "
            "the loader skips and counts them (samples_skipped).",
            "One pass over the training stream, one seed; no hyperparameter search.",
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
