# SPDX-License-Identifier: MIT
"""Collect every tool's output on one human + mouse library into the compact per-cell artifacts the notebook downloads.

    python -I barnyard_artifacts.py ID RUN_DIR WORK_DIR ARTIFACT_ROOT

writes (gzip TSV, one row per called cell, no raw counts of anything but per-species totals):
  ARTIFACT_ROOT/ambient-rna/<id>_cells.tsv.gz      species UMIs before and after each tool's correction, native rho estimates
  ARTIFACT_ROOT/ambient-rna/<id>_pool.json         species composition of the ambient pool (truth input)
  ARTIFACT_ROOT/doublets/<id>_doublets.tsv.gz      doublet score and call per tool, species UMIs (truth input)
  ARTIFACT_ROOT/<capability>/resources_<id>.json   wall time, peak memory, threads per tool (from GNU time -v files)
"""
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barnyard_prep import read_10x  # noqa: E402


def parse_time(path):
    """GNU `time -v` file -> (wall seconds, max RSS GB, cpu hours)."""
    t = open(path).read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
    parts = [float(x) for x in m.group(1).split(":")]
    wall = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 1024 ** 2
    cpu = float(re.search(r"User time \(seconds\): ([\d.]+)", t).group(1)) + float(re.search(r"System time \(seconds\): ([\d.]+)", t).group(1))
    return wall, rss, cpu / 3600


def main(ds, run, work, art):
    cells = pd.read_csv(f"{work}/cells.csv")
    genes = pd.read_csv(f"{work}/genes.csv")
    mouse_g = (genes.species == "mouse").values
    bc = cells.barcode.astype(str).values
    amb = cells.copy()
    dbl = cells.copy()

    def add_after(name, X, xbc):
        """X genes x barcodes (any order); per-species UMIs after correction for the called cells."""
        ix = pd.Series(np.arange(len(xbc)), index=xbc).reindex(bc)
        ok = ix.notna().values
        Xr = X.tocsr()
        M = np.asarray(Xr[mouse_g].sum(0)).ravel()
        Hh = np.asarray(Xr[~mouse_g].sum(0)).ravel()
        amb[f"{name}_human"] = np.where(ok, Hh[ix.fillna(0).astype(int).values], np.nan)
        amb[f"{name}_mouse"] = np.where(ok, M[ix.fillna(0).astype(int).values], np.nan)

    # Singlet: ambient_corrected_counts (gene level, integer, called cells only) and its per-cell rho
    C, cid, _, cbc = read_10x_named(f"{run}", "ambient_corrected_counts")
    assert (cid == genes.id.values).all(), "Singlet corrected-matrix features differ from the raw matrix"
    add_after("singlet", C, cbc)
    a = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t")
    a["barcode"] = a.barcode.astype(str).str.replace(r"-1$", "", regex=True)
    amb["singlet_rho"] = a.set_index("barcode").rho.reindex(bc).values

    for t in ("soupx", "decontx", "decontx_bg"):
        p = f"{work}/tools/{t}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p).set_index("barcode").reindex(bc)
            amb[f"{t}_human"] = d.human_after.values
            amb[f"{t}_mouse"] = d.mouse_after.values
            amb[f"{t}_rho"] = d.rho.values
        else:
            print(f"[warn] {t}: no output (tool failed); column absent")

    for cb in ("cellbender", "cellbender_rec"):
        cbp = f"{work}/{cb}/cb.h5"
        if not os.path.exists(cbp):
            print(f"[warn] {cb}: no output; column absent")
            continue
        import h5py
        with h5py.File(cbp, "r") as f:
            g = f["matrix"] if "matrix" in f else f
            cbb = np.array([b.decode().replace("-1", "") for b in g["barcodes"][:]])
            X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        if X.shape[0] != len(genes):
            X = X.T.tocsc()
        add_after(cb, X, cbb)
        print(f"{cb}: called cells missing from its output:", int(amb[f"{cb}_human"].isna().sum()))

    os.makedirs(f"{art}/ambient-rna", exist_ok=True)
    os.makedirs(f"{art}/doublets", exist_ok=True)
    amb.to_csv(f"{art}/ambient-rna/{ds}_cells.tsv.gz", sep="\t", index=False, float_format="%.6g")
    json.dump(json.load(open(f"{work}/pool.json")), open(f"{art}/ambient-rna/{ds}_pool.json", "w"), indent=1)

    # ---- doublets
    sd = pd.read_csv(f"{run}/doublet_scores.tsv", sep="\t")
    sd["barcode"] = sd.barcode.astype(str).str.replace(r"-1$", "", regex=True)
    sd = sd.set_index("barcode").reindex(bc)
    dbl["singlet_score"] = sd.doublet_score.values
    dbl["singlet_call"] = sd.is_doublet.astype(str).str.lower().eq("true").values.astype(int)
    for t in ("scrublet", "scdblfinder"):
        p = f"{work}/tools/{t}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p).set_index("barcode").reindex(bc)
            dbl[f"{t}_score"] = d.score.values
            dbl[f"{t}_call"] = d.call.astype(str).str.lower().eq("true").values.astype(int)
    dbl.to_csv(f"{art}/doublets/{ds}_doublets.tsv.gz", sep="\t", index=False, float_format="%.6g")

    # ---- resources (GNU time files)
    res_amb, res_dbl = [], []
    et, pt = parse_time(f"{run}/encode.time.txt"), parse_time(f"{run}/time.txt")
    log = open(f"{run}/run.log").read()
    step = lambda tag: float(re.search(r"\[%s\][^\n]*time=([\d.e+-]+)s" % tag, log).group(1))
    node = open(f"{run}/node.txt").read().strip()
    sing = dict(tool="singlet", dataset=ds, wall_s=round(et[0] + pt[0], 1), max_rss_gb=round(max(et[1], pt[1]), 2), threads=16,
                cpu_h=round(et[2] + pt[2], 3), node=node)
    res_amb.append({**sing, "notes": "whole pipeline (encode %.0f s + align/count/everything %.0f s); the ambient step itself took %.1f s" % (et[0], pt[0], step("ambient"))})
    res_dbl.append({**sing, "notes": "whole pipeline (encode %.0f s + align/count/everything %.0f s); the doublet step itself took %.1f s" % (et[0], pt[0], step("doublet"))})
    nodes = open(f"{work}/tools/node.txt").read().split("\n")
    for t, lst, thr in (("soupx", res_amb, 1), ("decontx", res_amb, 1), ("decontx_bg", res_amb, 1), ("scrublet", res_dbl, 1), ("scdblfinder", res_dbl, 1)):
        p = f"{work}/tools/{t}.time"
        if os.path.exists(p):
            w, r, c = parse_time(p)
            lst.append(dict(tool=t, dataset=ds, wall_s=round(w, 1), max_rss_gb=round(r, 2), threads=thr, cpu_h=round(c, 3), node=nodes[0]))
    for cb in ("cellbender", "cellbender_rec"):
        p = f"{work}/{cb}/time.txt"
        if os.path.exists(p) and os.path.exists(f"{work}/{cb}/cb.h5"):
            w, r, c = parse_time(p)
            gpu = open(f"{work}/{cb}/gpu.txt").read().strip().split(" (UUID")[0] if os.path.exists(f"{work}/{cb}/gpu.txt") else "GPU"
            res_amb.append(dict(tool=cb, dataset=ds, wall_s=round(w, 1), max_rss_gb=round(r, 2), threads=8, cpu_h=round(c, 3),
                                node=gpu, notes="runs on one GPU; wall time is dominated by training"))
    json.dump(res_amb, open(f"{art}/ambient-rna/resources_{ds}.json", "w"), indent=1)
    json.dump(res_dbl, open(f"{art}/doublets/resources_{ds}.json", "w"), indent=1)
    print("wrote artifacts for", ds, "cells", len(bc))


def read_10x_named(d, stem):
    """Singlet's `<stem>.mtx.gz`, `<stem>_features.tsv.gz`, `<stem>_barcodes.tsv.gz` (not a directory)."""
    with gzip.open(f"{d}/{stem}.mtx.gz", "rt") as f:
        X = scipy.io.mmread(f).tocsc()
    feats = pd.read_csv(f"{d}/{stem}_features.tsv.gz", sep="\t", header=None)
    bcs = pd.read_csv(f"{d}/{stem}_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).str.replace(r"-1$", "", regex=True).values
    return X, feats[0].astype(str).str.split("_").str[0].values, None, bcs  # rows are "<gene id>_<symbol>"


if __name__ == "__main__":
    main(*sys.argv[1:5])
