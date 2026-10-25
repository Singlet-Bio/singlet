# SPDX-License-Identifier: MIT
"""Public HTO benchmark harness (results contract 1.0.0, capability ``hto``) on Howitt et al. 2023 (Zenodo 8304003).

    python -P bench/adt/hto_public_run.py --dataset howitt2023_bal1 --tool singlet --zenodo DIR --published PUBDIR \
        --out $WORK/bpub/artifacts/hto

``--tool prepare`` writes ``hto_<dataset>_truth.parquet`` (``barcode``, ``capture``, ``genetic_donor``, ``truth_hashtag``)
and ``hto_<dataset>_published.parquet`` (the paper's own calls of nine methods, one column each). Each method then runs on
each capture separately, on the called cells only (the paper's protocol), as its own process under ``/usr/bin/time -v``
and writes ``hto_<dataset>_<tool>.parquet`` (``barcode``, ``call``) and ``resources_hto_<dataset>_<tool>.json``.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATASETS = {"howitt2023_bal1": ("BAL", 1), "howitt2023_bal2": ("BAL", 2), "howitt2023_bal3": ("BAL", 3),
            "howitt2023_cl": ("CL", None)}
TOOLS = ("singlet", "htodemux", "multiseqdemux", "hashsolo")


def captures(z, ds):
    """[(capture, hto counts DataFrame (hashtags x cells), truth Series)] and the hashtag -> donor map."""
    import hto_zenodo as HZ

    kind, b = DATASETS[ds]
    name = f"BAL{b}" if kind == "BAL" else "CL"
    for n, caps, hmap in HZ.datasets(z):
        if n == name:
            return caps, hmap
    raise SystemExit(f"{ds}: not found in {z}")


def prepare(a):
    caps, hmap = captures(a.zenodo, a.dataset)
    inv = {v: k for k, v in hmap.items()}
    rows = []
    for cap, h, truth in caps:
        rows.append(pd.DataFrame({"barcode": h.columns, "capture": cap, "genetic_donor": truth.values,
                                  "truth_hashtag": truth.map(inv).fillna(truth).values}))
    T = pd.concat(rows, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    T.to_parquet(os.path.join(a.out, f"hto_{a.dataset}_truth.parquet"), index=False)
    kind, b = DATASETS[a.dataset]
    pub = os.path.join(a.published, ("lmo" if kind == "CL" else f"batch{b}") + "_published_calls.csv")
    P = pd.read_csv(pub)
    cols = {c: re.sub(r"_calls$", "", c) for c in P.columns if c.endswith("_calls")}
    out = P[["cell"] + list(cols)].rename(columns={"cell": "barcode", **cols})
    out.astype(str).to_parquet(os.path.join(a.out, f"hto_{a.dataset}_published.parquet"), index=False)
    json.dump({"hashtag_to_donor": hmap, "cells": int(len(T))}, open(os.path.join(a.out, f"hto_{a.dataset}_map.json"), "w"),
              indent=1)


def run_tool(a):
    import hto_methods as HM

    caps, _ = captures(a.zenodo, a.dataset)
    fn = HM.METHODS[a.tool] if a.tool != "singlet" else run_singlet_shipped
    parts = []
    for cap, h, _ in caps:
        wd = os.path.join(a.scratch, a.dataset, cap, a.tool)
        os.makedirs(wd, exist_ok=True)
        r = fn(h, None, workdir=wd)
        parts.append(pd.DataFrame({"barcode": r.index, "call": r["call"].values}))
    pd.concat(parts).to_parquet(os.path.join(a.out, f"hto_{a.dataset}_{a.tool}.parquet"), index=False)


def run_singlet_shipped(hto, empties=None, rna=None, workdir=None):
    """The caller Singlet ships: ``singlet.adt.hto.call_hashtags``."""
    import numpy as np
    from singlet.adt.hto import call_hashtags

    X = hto.to_numpy(dtype=float)
    amb = empties.to_numpy(dtype=float).sum(axis=1) if empties is not None and empties.shape[1] else None
    call, lt, _ = call_hashtags(X, amb)
    names = list(hto.index)
    lab = ["Negative" if c == -1 else "Doublet" if c == -2 else names[int(c)] for c in call]
    return pd.DataFrame({"call": lab, "score": np.asarray(lt, float)}, index=hto.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--tool", required=True, choices=("prepare",) + TOOLS)
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--inner", action="store_true")
    a = ap.parse_args()
    if a.tool == "prepare":
        return prepare(a)
    if a.inner:
        return run_tool(a)
    cmd = [sys.executable, "-P", os.path.abspath(__file__), "--inner"] + sys.argv[1:]
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True, text=True)
    wall = time.time() - t0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr)
    rec = {"tool": a.tool, "dataset": a.dataset, "wall_s": round(wall, 2),
           "max_rss_gb": round(int(m.group(1)) / 1024 ** 2, 3) if m else None, "threads": a.threads,
           "returncode": r.returncode, "command": " ".join(cmd), "singlet_commit": os.environ.get("SINGLET_COMMIT", ""),
           "stderr_tail": r.stderr[-1500:] if r.returncode else ""}
    with open(os.path.join(a.out, f"resources_hto_{a.dataset}_{a.tool}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "stderr_tail"}))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
