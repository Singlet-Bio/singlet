# SPDX-License-Identifier: MIT
"""Reads-per-UMI histogram of the engine against Cell Ranger's molecule_info.h5 read counts.

The engine writes ``reads_per_umi_hist.tsv.gz`` (per barcode: molecules with 1..15 and 16+ reads). Cell Ranger's ``molecule_info.h5`` stores
one row per molecule (barcode, UMI, feature) with ``count`` = the reads behind it. Both are binned the same way and compared on the cells
both callers keep: the pooled histogram, each cell's mean reads per UMI and molecule count, and the sequencing saturation they imply.

    python bench/readlevel/rpu_vs_molinfo.py --hist OUT/reads_per_umi_hist.tsv.gz --molinfo pbmc_1k_v3_molecule_info.h5 \
        --cr-filtered filtered_feature_bc_matrix.h5 --calls OUT/cell_calls.tsv --out result.json
"""

from __future__ import annotations

import argparse
import gzip
import json

import h5py
import numpy as np

BINS = 16


def load_hist(path: str) -> dict:
    out = {}
    with gzip.open(path, "rt") as f:
        head = f.readline().rstrip("\n").split("\t")
        assert len(head) == BINS + 1, head
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            out[p[0]] = np.array(p[1:], dtype=np.int64)
    return out


def load_molinfo(path: str):
    with h5py.File(path, "r") as h:
        bcs = np.array([b.decode() for b in h["barcodes"][:]])
        bidx = h["barcode_idx"][:]
        gg = h["gem_group"][:]
        cnt = h["count"][:].astype(np.int64)
    binned = np.minimum(cnt, BINS) - 1
    hist = np.zeros((len(bcs), BINS), dtype=np.int64)
    np.add.at(hist, (bidx, binned), 1)
    reads = np.bincount(bidx, weights=cnt, minlength=len(bcs))
    suffix = f"-{int(gg[0])}" if len(set(gg.tolist())) == 1 else ""
    return {b + suffix: hist[i] for i, b in enumerate(bcs) if hist[i].sum() > 0}, {b + suffix: reads[i] for i, b in enumerate(bcs)}


def cr_filtered(path: str) -> set:
    with h5py.File(path, "r") as h:
        return {b.decode() for b in h["matrix/barcodes"][:]}


def engine_called(path: str) -> set:
    called = set()
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        ic = head.index("is_cell") if "is_cell" in head else 1
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if p[ic] in ("1", "true", "True", "TRUE"):
                called.add(p[0])
    return called


def norm(bc: str) -> str:
    return bc if "-" in bc else bc + "-1"


def mean_rpu(h: np.ndarray) -> float:
    # 16+ is counted as 16: a lower bound for the last bin, applied the same way to both sides
    return float((h * np.arange(1, BINS + 1)).sum() / max(h.sum(), 1))


def rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--molinfo", required=True)
    ap.add_argument("--cr-filtered", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    eng = {norm(k): v for k, v in load_hist(a.hist).items()}
    cr, cr_reads = load_molinfo(a.molinfo)
    cells = sorted(cr_filtered(a.cr_filtered) & {norm(b) for b in engine_called(a.calls)} & eng.keys() & cr.keys())
    E = np.stack([eng[c] for c in cells])
    R = np.stack([cr[c] for c in cells])
    pe, pr = E.sum(0) / E.sum(), R.sum(0) / R.sum()
    me, mr = np.array([mean_rpu(h) for h in E]), np.array([mean_rpu(h) for h in R])
    ne, nr = E.sum(1).astype(float), R.sum(1).astype(float)
    # saturation = 1 - molecules / reads, over the shared cells (16+ counted as 16 on both sides)
    re_, rr_ = (E * np.arange(1, BINS + 1)).sum(), (R * np.arange(1, BINS + 1)).sum()
    res = {
        "cells_shared": len(cells),
        "cells_engine_hist": len(eng),
        "cells_cr_filtered": len(cr_filtered(a.cr_filtered)),
        "molecules_engine": int(E.sum()),
        "molecules_cr": int(R.sum()),
        "pooled_fraction_engine": [round(float(x), 5) for x in pe],
        "pooled_fraction_cr": [round(float(x), 5) for x in pr],
        "pooled_tvd": round(float(0.5 * np.abs(pe - pr).sum()), 5),
        "singleton_fraction_engine": round(float(pe[0]), 5),
        "singleton_fraction_cr": round(float(pr[0]), 5),
        "mean_rpu_engine_median_cell": round(float(np.median(me)), 4),
        "mean_rpu_cr_median_cell": round(float(np.median(mr)), 4),
        "mean_rpu_pearson": round(float(np.corrcoef(me, mr)[0, 1]), 5),
        "mean_rpu_spearman": round(float(np.corrcoef(rank(me), rank(mr))[0, 1]), 5),
        "mean_rpu_median_abs_rel_diff": round(float(np.median(np.abs(me - mr) / mr)), 5),
        "molecules_per_cell_pearson_log": round(float(np.corrcoef(np.log1p(ne), np.log1p(nr))[0, 1]), 5),
        "molecules_per_cell_median_ratio": round(float(np.median(ne / np.maximum(nr, 1))), 5),
        "saturation_engine": round(float(1 - E.sum() / re_), 5),
        "saturation_cr": round(float(1 - R.sum() / rr_), 5),
        "per_cell_tvd_median": round(float(np.median(0.5 * np.abs(E / E.sum(1, keepdims=True) - R / R.sum(1, keepdims=True)).sum(1))), 5),
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
