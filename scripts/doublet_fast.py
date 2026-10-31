#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fast sdfl + fixed-reference neighbour feature on the 17 evaluation datasets: features (keys sdfl_f, sdflc_f, lite_knn,
lite_entropy) and wall / CPU seconds per stage -> OUT/lite/feats_<tag>/.  Evaluate with scripts/doublet_lite.py eval."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import doublet_v2 as S
    from singlet.features import doublet as D
    from singlet.features import doublet_sdfl as DS
    from singlet.features.doublet_lite import Distilled

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="distill_aug_1000_256_32.npz")
    ap.add_argument("--variants", default="ac")
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--art-ratio", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--ref-cap", type=int, default=3000)
    ap.add_argument("--names", nargs="*")
    a = ap.parse_args()
    out = Path(a.out) / "lite"
    mdl = Distilled.load(out / a.model)
    p = replace(DS.FAST, cap=a.cap, art_ratio=a.art_ratio, max_iter=a.max_iter, folds=a.folds, rounds=a.rounds)
    dst = out / f"feats_{a.tag}"
    dst.mkdir(parents=True, exist_ok=True)
    cost = {}
    for n in a.names or list(D.STUDY):
        X, y, genes, species = S.load_counts(n)
        w0, c0 = time.perf_counter(), time.process_time()
        F = mdl.features(X, genes, species, ref_cap=a.ref_cap)
        w1, c1 = time.perf_counter(), time.process_time()
        sd = DS.score(X, tuple(a.variants), p=p)
        w2, c2 = time.perf_counter(), time.process_time()
        F["sdfl_f"] = sd["sdfl"] if "sdfl" in sd else sd["sdflc"]
        if "sdflc" in sd:
            F["sdflc_f"] = sd["sdflc"]
        cost[n] = dict(n_cells=int(X.shape[0]), atlas_wall=round(w1 - w0, 2), atlas_cpu=round(c1 - c0, 2),
                       sdfl_wall=round(w2 - w1, 2), sdfl_cpu=round(c2 - c1, 2))
        np.savez(dst / f"{n}.npz", **F)
        print(n, cost[n], flush=True)
    (dst / "cost.json").write_text(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
