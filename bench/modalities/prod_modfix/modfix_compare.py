# SPDX-License-Identifier: MIT
"""Gene-level and alignment-level differences between a reference run (singlet-prod) and a new run (singlet-modfix).

    python -P modfix_compare.py REF_RUN NEW_RUN --json OUT

Reports STAR splices / unique mapping (star_Log.final.out), the 5' clip decision, called cells, and on the cells called
in BOTH runs: total UMIs, median UMIs and genes per cell, per-gene total correlation and the share of genes (>= 100
UMIs) whose total moved by more than 10% / 2-fold. Plus the size of every output the new run wrote, public vs private
(pileup_stats.json provenance.outputs).
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp


def lines(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [x.rstrip("\n") for x in f]


def star_log(run):
    out = {}
    p = os.path.join(run, "star_Log.final.out")
    if not os.path.exists(p):
        return out
    for ln in lines(p):
        if "|" not in ln:
            continue
        k, v = [x.strip() for x in ln.split("|", 1)]
        if k in ("Number of input reads", "Uniquely mapped reads number", "Uniquely mapped reads %",
                 "Number of splices: Total", "Number of splices: Annotated (sjdb)", "Mismatch rate per base, %",
                 "Average mapped length"):
            out[k] = v
    return out


def called(run):
    p = os.path.join(run, "cell_calls.tsv")
    rows = lines(p)
    h = rows[0].split("\t")
    ic = h.index("is_cell") if "is_cell" in h else (h.index("called_v2") if "called_v2" in h else None)
    out = []
    for r in rows[1:]:
        f = r.split("\t")
        if ic is not None and f[ic] in ("1", "True", "true"):
            out.append(f[0])
    return out


def gene_matrix(run):
    d = os.path.join(run, "filtered_feature_bc_matrix")
    x = sp.csc_matrix(scipy.io.mmread(os.path.join(d, "matrix.mtx.gz")))
    feats = [l.split("\t")[0] for l in lines(os.path.join(d, "features.tsv.gz"))]
    bcs = lines(os.path.join(d, "barcodes.tsv.gz"))
    return x, feats, bcs


def clip(run):
    out = {}
    for ln in lines(os.path.join(run, "run.log")):
        if "constant 5' prefix" in ln or "5' prefix probe" in ln or "Auto-clip5p" in ln:
            out.setdefault("log", []).append(ln.strip()[:200])
    return out


def main():
    ref, new = sys.argv[1], sys.argv[2]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    R = {"ref": ref, "new": new, "star": {"ref": star_log(ref), "new": star_log(new)},
         "clip": {"ref": clip(ref), "new": clip(new)}}
    cr, cn = called(ref), called(new)
    R["called_cells"] = {"ref": len(cr), "new": len(cn), "both": len(set(cr) & set(cn))}
    try:
        xr, fr, br = gene_matrix(ref)
        xn, fn, bn = gene_matrix(new)
        both = sorted(set(cr) & set(cn) & set(br) & set(bn))
        ir = {b: i for i, b in enumerate(br)}
        inn = {b: i for i, b in enumerate(bn)}
        assert fr == fn, "feature lists differ"
        A = xr[:, [ir[b] for b in both]]
        B = xn[:, [inn[b] for b in both]]
        ga, gb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
        ua, ub = np.asarray(A.sum(0)).ravel(), np.asarray(B.sum(0)).ravel()
        na, nb = np.asarray((A > 0).sum(0)).ravel(), np.asarray((B > 0).sum(0)).ravel()
        keep = (ga >= 100) | (gb >= 100)
        ratio = (gb[keep] + 1) / (ga[keep] + 1)
        R["genes"] = {
            "cells_compared": len(both),
            "total_umi_ref": int(ga.sum()), "total_umi_new": int(gb.sum()),
            "total_umi_ratio": float(gb.sum() / max(ga.sum(), 1)),
            "median_umi_ref": float(np.median(ua)), "median_umi_new": float(np.median(ub)),
            "median_genes_ref": float(np.median(na)), "median_genes_new": float(np.median(nb)),
            "identical": bool((A != B).nnz == 0),
            "differing_entries": int((A != B).nnz),
            "pearson_log_gene_totals": float(np.corrcoef(np.log1p(ga), np.log1p(gb))[0, 1]),
            "genes_ge100": int(keep.sum()),
            "genes_changed_gt10pct": int(((ratio > 1.1) | (ratio < 1 / 1.1)).sum()),
            "genes_changed_gt2x": int(((ratio > 2) | (ratio < 0.5)).sum()),
            "top_up": [[fr[i], int(ga[i]), int(gb[i])] for i in np.argsort(-(gb - ga))[:8]],
            "top_down": [[fr[i], int(ga[i]), int(gb[i])] for i in np.argsort(gb - ga)[:8]],
        }
    except Exception as e:  # noqa: BLE001
        R["genes_error"] = repr(e)
    for run, key in ((ref, "ref"), (new, "new")):
        p = os.path.join(run, "sj_counts.mtx.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt") as f:
                for ln in f:
                    if not ln.startswith("%"):
                        nr, nc, nnz = ln.split()
                        break
            R.setdefault("sj", {})[key] = {"junctions": int(nr), "barcodes": int(nc), "nnz": int(nnz)}
    try:
        ps = json.load(open(os.path.join(new, "pileup_stats.json")))
        inv = ps.get("provenance", {}).get("outputs", {})
        R["outputs"] = {k: inv.get(k) for k in ("n_public", "n_private", "bytes_public", "bytes_private")}
        R["outputs"]["files"] = {f["path"]: [f["bytes"], f["visibility"]] for f in inv.get("files", [])
                                 if not f["path"].startswith("star_")}
        R["provenance_extra"] = {k: ps["provenance"].get(k) for k in ("clip5p", "snp_panel", "bam_sort_mode")}
    except Exception as e:  # noqa: BLE001
        R["outputs_error"] = repr(e)
    s = json.dumps(R, indent=1)
    print(s[:6000])
    if jout:
        with open(jout, "w") as f:
            f.write(s)


if __name__ == "__main__":
    main()
