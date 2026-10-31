# SPDX-License-Identifier: MIT
"""Development-library diagnostics (never run on held-out libraries): where a reference chain is lost, and gene disagreements.

    python -P diag2.py --runs RUNS --data DATA --lib LIB:TCR|BCR [--sdir singlet] [--params '{"cross_frac":0.1,...}'] [--out x.json]

Over the reference-paired cells of the primary metric (PREREG.md), every chain whose Singlet top contig is not the
reference's top junction is put in one class:

* ``no_barcode``: the barcode has no Singlet contig of any chain; ``no_chain``: none of this chain;
* ``exact_*``: Singlet has a contig with the reference junction, but it is not the reported top: ``exact_not_productive``,
  ``exact_low_support`` (not high-confidence: reads / read depth), ``exact_ambient`` (cross filter), ``exact_min_frac``,
  ``exact_variant`` (error-variant filter), ``exact_kept_not_top``;
* ``diff_*``: only other junctions: ``diff_cys`` (a 3-nt longer / shorter junction sharing the end: the Cys anchor),
  ``diff_sub`` (same length, <= 2 substitutions), ``diff_other``.

Then, on chains whose junction matches, the (reference, Singlet) V and J pairs that disagree, counting V genes of one
equivalence class as one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

PAIRS = [("TRA", "TRB"), ("TRG", "TRD"), ("IGH", "IGK"), ("IGH", "IGL")]


def tf(x):
    return str(x).strip().lower() in ("true", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--sdir", default="singlet")
    ap.add_argument("--vdj-ref", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lib, chains = a.lib.split(":")
    keep = CV.TCR if chains == "TCR" else CV.BCR
    ref = CV.load_cellranger(os.path.join(a.data, lib, "filtered_contig_annotations.csv"))
    ref = ref[ref.chain.isin(keep)]
    r = os.path.join(a.runs, lib)
    gp = os.path.join(r, "gex_cells.txt")
    gex = set(l.strip() for l in open(gp) if l.strip()) if os.path.exists(gp) else None
    sd = os.path.join(r, a.sdir)
    run = json.load(open(os.path.join(sd, "run.json")))
    P = run["params"]
    allc = pd.read_csv(os.path.join(sd, "contigs.csv"))
    allc = allc[allc.chain.isin(keep)]
    if gex is not None:
        allc = allc[allc.barcode.isin(gex)]
    for c in ("kept", "productive", "high_confidence"):
        allc[c] = allc[c].map(tf)
    allc["cdr3_nt"] = allc["cdr3_nt"].astype(str)
    jmax = allc.groupby(["chain", "cdr3_nt"])["umis"].transform("max")
    allc["ambient"] = (allc["umis"] < P["cross_frac"] * jmax) & (jmax >= P["cross_min_umis"])
    top_any = allc.groupby(["barcode", "chain"])["umis"].transform("max")
    allc["below_frac"] = allc["umis"] < P["min_frac"] * top_any
    kp = allc[allc.kept & allc.productive]
    mt = kp.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
    mtop = {(b, c): n for b, c, n in zip(mt.barcode, mt.chain, mt.cdr3_nt)}
    rt = ref.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
    rtop = {(b, c): (n, u) for b, c, n, u in zip(rt.barcode, rt.chain, rt.cdr3_nt, rt.umis)}
    by_bc = {b: g for b, g in allc.groupby("barcode")}
    cls = Counter()
    n_cells = n_ok = 0
    ex_rows = []
    miss_umis = []
    for bc in sorted({b for b, _ in rtop}):
        for x, y in PAIRS:
            if (bc, x) in rtop and (bc, y) in rtop:
                n_cells += 1
                ok = True
                for ch in (x, y):
                    rn, ru = rtop[(bc, ch)]
                    if mtop.get((bc, ch)) == rn:
                        continue
                    ok = False
                    g = by_bc.get(bc)
                    if g is None:
                        k = "no_barcode"
                    else:
                        gc = g[g.chain == ch]
                        if not len(gc):
                            k = "no_chain"
                        else:
                            e = gc[gc.cdr3_nt == rn]
                            if len(e):
                                e = e.sort_values("umis", ascending=False).iloc[0]
                                if not e.productive:
                                    k = "exact_not_productive"
                                elif e.kept:
                                    k = "exact_kept_not_top"
                                elif not e.high_confidence:
                                    k = "exact_low_support"
                                elif e.ambient:
                                    k = "exact_ambient"
                                elif e.below_frac:
                                    k = "exact_min_frac"
                                else:
                                    k = "exact_variant"
                                ex_rows.append({"chain": ch, "class": k, "umis": int(e.umis), "reads": int(e.reads),
                                                "junction_umis": int(e.junction_umis), "ref_umis": float(ru)})
                            else:
                                ms = gc.sort_values("umis", ascending=False)
                                k = "diff_other"
                                for n in ms.cdr3_nt:
                                    if abs(len(n) - len(rn)) == 3 and (n.endswith(rn) or rn.endswith(n)):
                                        k = "diff_cys"
                                        break
                                    if len(n) == len(rn) and sum(p != q for p, q in zip(n, rn)) <= 2:
                                        k = "diff_sub"
                                        break
                    cls[(ch, k)] += 1
                    miss_umis.append((k, float(ru)))
                n_ok += ok
                break
    out = {"lib": lib, "sdir": a.sdir, "reference_paired_cells": n_cells, "correct_pairing": n_ok / max(1, n_cells),
           "stats": {k: run["stats"].get(k) for k in ("read_pairs", "umis", "umi_read_threshold", "umi_reads_median",
                                                      "umis_low_support", "barcodes_assembled", "contigs", "contigs_kept")},
           "miss_classes": {f"{c}:{k}": v for (c, k), v in sorted(cls.items())},
           "miss_class_totals": dict(Counter(k for (_, k) in cls.elements())),
           }
    mu = pd.DataFrame(miss_umis, columns=["class", "ref_umis"])
    if len(mu):
        out["ref_umis_of_missed_by_class"] = mu.groupby("class").ref_umis.describe()[["count", "25%", "50%", "75%"]].to_dict("index")
    out["ref_umis_all_tops"] = rt.umis.describe()[["count", "25%", "50%", "75%"]].to_dict()
    if ex_rows:
        e = pd.DataFrame(ex_rows)
        out["exact_filtered_detail"] = e.groupby("class")[["umis", "reads", "junction_umis", "ref_umis"]].median().to_dict("index")
    # gene disagreements on exact junction matches (top contigs)
    m = CV.load_singlet(os.path.join(sd, "contigs.csv"))
    m = m[m.chain.isin(keep)]
    if gex is not None:
        m = m[m.barcode.isin(gex)]
    mtp = m.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    j = rt.merge(mtp, on=["barcode", "chain"], suffixes=("_r", "_m"))
    j = j[j.cdr3_nt_r == j.cdr3_nt_m]
    eq = {}
    if a.vdj_ref:
        from singlet.vdj import reference as VR
        eq = {CV.norm_gene(k): v for k, v in VR.equivalence(VR.load(a.vdj_ref)).items()}
    e_ = lambda g: eq.get(g, g)
    vbad = j[j.v_r.map(e_) != j.v_m.map(e_)]
    jbad = j[j.j_r != j.j_m]
    out["exact_matches"] = len(j)
    out["v_agreement_equiv"] = 1 - len(vbad) / max(1, len(j))
    out["v_agreement_raw"] = float((j.v_r == j.v_m).mean()) if len(j) else None
    out["j_agreement"] = 1 - len(jbad) / max(1, len(j))
    out["v_disagree_top"] = [[f"{p}->{q}", n] for (p, q), n in Counter(zip(vbad.v_r, vbad.v_m)).most_common(15)]
    out["j_disagree_top"] = [[f"{p}->{q}", n] for (p, q), n in Counter(zip(jbad.j_r, jbad.j_m)).most_common(15)]
    out["v_raw_disagree_top"] = [[f"{p}->{q}", n] for (p, q), n in
                                 Counter(zip(j[j.v_r != j.v_m].v_r, j[j.v_r != j.v_m].v_m)).most_common(15)]
    s = json.dumps(out, indent=1, default=str)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
