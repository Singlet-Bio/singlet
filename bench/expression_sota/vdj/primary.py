# SPDX-License-Identifier: MIT
"""Pre-registered primary metric (PREREG.md): clonotype pairing accuracy with bootstrap CIs, and pooled wrong-lineage.

    python -P primary.py --runs RUNS --data DATA --libs ctrl10k_t:TCR,ctrl10k_b:BCR,... [--singlet-dir singlet]
        [--vdj-ref ref.tsv.gz] --out primary.json

Loaders and filters are those of bench/vdj/compare_vdj.py (imported). Per reference-paired cell: correct when the method's top-UMI
productive contig of both chains of the reference pair (on called GEX barcodes, if the library has them) has the reference's cdr3_nt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

PAIRS = [("TRA", "TRB"), ("TRG", "TRD"), ("IGH", "IGK"), ("IGH", "IGL")]


def tops(t):
    t = t.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
    return {(b, c): n for b, c, n in zip(t.barcode, t.chain, t.cdr3_nt)}


def per_cell(ref, m, gex_cells):
    """barcode-ordered list of (barcode, correct) over reference-paired cells."""
    if gex_cells is not None:
        m = m[m["barcode"].isin(gex_cells)]
    rt, mt = tops(ref), tops(m)
    out = []
    for bc in sorted({b for b, _ in rt}):
        for x, y in PAIRS:
            if (bc, x) in rt and (bc, y) in rt:
                out.append((bc, int(mt.get((bc, x)) == rt[(bc, x)] and mt.get((bc, y)) == rt[(bc, y)])))
                break
    return out


def wrong_lineage(t, lab):
    ll = t.barcode.map(lab)
    known = ll.isin(["T", "B"])
    wrong = (t.chain.isin(CV.TCR) & (ll == "B")) | (t.chain.isin(CV.BCR) & (ll == "T"))
    return int(wrong.sum()), int(known.sum())


def boot(vecs, groups, n=2000, seed=0):
    """vecs: {method: {lib: array}} aligned per lib; groups: {name: [libs]}. Returns accuracy, CI and paired differences."""
    rng = np.random.default_rng(seed)
    libs = sorted({l for v in vecs.values() for l in v})
    idx = {l: [rng.integers(0, len(next(iter(vecs.values()))[l]), len(next(iter(vecs.values()))[l])) for _ in range(n)] for l in libs}
    res = {}
    for g, gl in groups.items():
        gl = [l for l in gl if l in libs]
        if not gl:
            continue
        stat = {}
        for meth, v in vecs.items():
            if not all(l in v for l in gl):
                continue
            tot = sum(len(v[l]) for l in gl)
            point = sum(int(v[l].sum()) for l in gl) / tot
            bs = np.array([sum(v[l][idx[l][b]].sum() for l in gl) / tot for b in range(n)])
            stat[meth] = (point, bs)
        r = {m: {"n_cells": sum(len(vecs[m][l]) for l in gl), "accuracy": p, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
             for m, (p, b) in stat.items()}
        if "singlet" in stat:
            for o in stat:
                if o != "singlet":
                    d = stat["singlet"][1] - stat[o][1]
                    r[f"diff_singlet_minus_{o}"] = {"diff": stat["singlet"][0] - stat[o][0],
                                                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        res[g] = r
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--libs", required=True, help="lib:TCR|BCR,...")
    ap.add_argument("--singlet-dir", default="singlet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    vecs = {"singlet": {}, "trust4": {}}
    wl = {"singlet": [0, 0], "cellranger": [0, 0]}
    per_lib = {}
    groups = {"TCR": [], "BCR": []}
    for item in a.libs.split(","):
        lib, chains = item.split(":")
        keep = CV.TCR if chains == "TCR" else CV.BCR
        d, r = os.path.join(a.data, lib), os.path.join(a.runs, lib)
        refp = os.path.join(d, "filtered_contig_annotations.csv")
        ref = CV.load_cellranger(refp)
        ref = ref[ref.chain.isin(keep)]
        gp = os.path.join(r, "gex_cells.txt")
        gex_cells = set(l.strip() for l in open(gp) if l.strip()) if os.path.exists(gp) else None
        sd = os.path.join(r, a.singlet_dir)
        s = CV.load_singlet(os.path.join(sd, "contigs.csv"), os.path.join(sd, "cells.csv"))
        s = s[s.chain.isin(keep)]
        t4p = os.path.join(r, "trust4", "TRUST_barcode_report.tsv")
        t4 = CV.load_trust4(t4p)
        t4 = t4[t4.chain.isin(keep)]
        pc = {}
        for name, m in (("singlet", s), ("trust4", t4)):
            pc[name] = per_cell(ref, m, gex_cells)
            vecs[name][lib] = np.array([c for _, c in pc[name]])
        assert [b for b, _ in pc["singlet"]] == [b for b, _ in pc["trust4"]]
        groups[chains].append(lib)
        per_lib[lib] = {"chains": chains, "reference_paired_cells": len(pc["singlet"])}
        lp = os.path.join(r, "gex_lineage.csv")
        if os.path.exists(lp):
            lab = dict(zip(*pd.read_csv(lp)[["barcode", "lineage"]].T.values))
            sm = s[s.barcode.isin(gex_cells)] if gex_cells is not None else s
            for name, t in (("singlet", sm), ("cellranger", ref)):
                w, k = wrong_lineage(t, lab)
                wl[name][0] += w
                wl[name][1] += k
                per_lib[lib][f"wrong_lineage_{name}"] = [w, k]
    groups["all"] = groups["TCR"] + groups["BCR"]
    res = {"libs": per_lib, "primary": boot(vecs, groups)}
    res["wrong_lineage_pooled"] = {k: {"wrong": v[0], "known": v[1], "rate": v[0] / max(1, v[1])} for k, v in wl.items()}
    res["wrong_lineage_pooled"]["singlet_minus_cellranger"] = (res["wrong_lineage_pooled"]["singlet"]["rate"]
                                                               - res["wrong_lineage_pooled"]["cellranger"]["rate"])
    # per-library accuracies with CIs
    res["per_library"] = boot(vecs, {l: [l] for l in per_lib})
    P = res["primary"]
    res["pass"] = {g: P[g]["diff_singlet_minus_trust4"]["diff"] >= -0.01 for g in P}
    res["pass"]["wrong_lineage"] = res["wrong_lineage_pooled"]["singlet_minus_cellranger"] <= 0.005
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("primary", "wrong_lineage_pooled", "pass")}, indent=1))


if __name__ == "__main__":
    main()
