# SPDX-License-Identifier: MIT
"""BCR study, development libraries only (PREREG.md Addendum 3; never the spent or new held-out libraries):
where Singlet's BCR pairing loses to TRUST4, selection vs assembly, and where J calls differ.

    python -P diag_bcr.py --runs RUNS --data DATA --lib LIB --sdirs singlet_base,singlet_asmv3 [--vdj-ref ref.tsv.gz]
        [--out x.json]

Per Singlet run (``sdir``), over the reference-paired cells of the primary metric (GEX cells where GEX exists):

* ``crosstab``: Singlet correct / TRUST4 correct (the cells Singlet loses: S wrong and T right).
* ``loss_classes``: for each chain of a Singlet-wrong cell, why: ``assembly`` (no Singlet contig of the chain carries the
  reference junction: ``no_chain`` / ``diff_cys`` / ``diff_sub`` / ``diff_other``) or ``selection`` (the reference junction
  was assembled but is not the reported top: ``sel_kept_not_top`` (with the winner's support: UMI tie, more reads ...),
  ``sel_filtered_<reason>`` (not high-confidence / ambient / below min_frac / error variant), ``sel_not_productive``).
  The same split for the cells Singlet loses to TRUST4 only.
* ``rules``: pairing accuracy if Singlet's top contig per chain were chosen by another rule over the same contigs:
  ``umis`` (current: kept productive, most UMIs), ``reads``, ``umis_then_reads``, ``junction_umis``, ``hc_umis`` (all
  productive high-confidence contigs, ignoring the ambient / min_frac / variant filters), ``oracle_kept`` (correct if the
  reference junction is among the kept productive contigs), ``oracle_any`` (among all productive contigs).
* ``trust4_on_singlet``: TRUST4's rule (one heavy, one light chain per barcode: the most abundant; secondary chains kept)
  applied to Singlet's kept productive contigs, with abundance = reads (TRUST4's count is read-based) or UMIs.
* ``j``: on contigs whose junction equals the reference, J agreement of Singlet and TRUST4, and per Singlet J disagreement
  whether the junction's 3' end matches the reference's J germline better, equally or worse than Singlet's J
  (``equal`` = the junction cannot tell them apart: the call comes from the read vote past F/W118).
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

PAIRS = [("IGH", "IGK"), ("IGH", "IGL")]
LIGHT = ("IGK", "IGL")


def tf(x):
    return str(x).strip().lower() in ("true", "1")


def top_by(df, cols):
    t = df.sort_values(cols, ascending=[False] * len(cols), kind="mergesort").drop_duplicates(["barcode", "chain"])
    return {(b, c): n for b, c, n in zip(t.barcode, t.chain, t.cdr3_nt)}


def ref_pairs(ref):
    rt = ref.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
    rtop = {(b, c): n for b, c, n in zip(rt.barcode, rt.chain, rt.cdr3_nt)}
    out = []
    for bc in sorted({b for b, _ in rtop}):
        for x, y in PAIRS:
            if (bc, x) in rtop and (bc, y) in rtop:
                out.append((bc, x, y, rtop[(bc, x)], rtop[(bc, y)]))
                break
    return out, rt


def acc(pairs, top):
    return float(np.mean([top.get((b, x)) == nx and top.get((b, y)) == ny for b, x, y, nx, ny in pairs])) if pairs else None


def trust4_rule(df, ab):
    """TRUST4 barcode report: chain1 = most abundant heavy, chain2 = most abundant light (IGK or IGL); the other light
    chain is a secondary chain and is still reported (the metric then takes the top per chain)."""
    return top_by(df, [ab])


def j_end_match(nt, seg):
    """consecutive matching bases from the 3' end of the junction against the J germline up to the end of F/W118."""
    if seg is None or seg.anchor < 0:
        return -1
    g = seg.seq[: seg.anchor + 3]
    n = 0
    for p, q in zip(reversed(nt), reversed(g)):
        if p != q:
            break
        n += 1
    return n


def classify(bc, ch, rn, g, P, jmax_of):
    gc = g[g.chain == ch] if g is not None else None
    if gc is None or not len(gc):
        return "assembly:no_chain", None
    e = gc[gc.cdr3_nt == rn]
    if not len(e):
        for n in gc.sort_values("umis", ascending=False).cdr3_nt:
            if abs(len(n) - len(rn)) == 3 and (n.endswith(rn) or rn.endswith(n)):
                return "assembly:diff_cys", None
            if len(n) == len(rn) and sum(p != q for p, q in zip(n, rn)) <= 2:
                return "assembly:diff_sub", None
        return "assembly:diff_other", None
    e = e.sort_values("umis", ascending=False).iloc[0]
    if not e.productive:
        return "selection:not_productive", None
    if e.kept:
        tk = gc[gc.kept & gc.productive].sort_values(["umis", "reads"], ascending=False).iloc[0]
        why = ("umi_tie" if tk.umis == e.umis else "fewer_umis") + ("_more_reads" if e.reads > tk.reads else "")
        return "selection:kept_not_top", {"why": why, "umis": int(e.umis), "reads": int(e.reads), "top_umis": int(tk.umis),
                                          "top_reads": int(tk.reads), "top_c": str(tk.c_gene), "c": str(e.c_gene)}
    if not e.high_confidence:
        return "selection:filtered_low_support", None
    top_any = gc.umis.max()
    jm = jmax_of.get((ch, rn), 0)
    if e.umis < P["cross_frac"] * jm and jm >= P["cross_min_umis"]:
        return "selection:filtered_ambient", None
    if e.umis < P["min_frac"] * top_any:
        return "selection:filtered_min_frac", None
    return "selection:filtered_variant", None


def run_one(a, lib, sdir, ref, pairs, rt, gex, t4, segs):
    sd = os.path.join(a.runs, lib, sdir)
    P = json.load(open(os.path.join(sd, "run.json")))["params"]
    allc = pd.read_csv(os.path.join(sd, "contigs.csv"))
    allc = allc[allc.chain.isin(CV.BCR)]
    if gex is not None:
        allc = allc[allc.barcode.isin(gex)]
    for c in ("kept", "productive", "high_confidence"):
        allc[c] = allc[c].map(tf)
    allc["cdr3_nt"] = allc["cdr3_nt"].astype(str)
    jmax_of = allc.groupby(["chain", "cdr3_nt"])["umis"].max().to_dict()
    kp = allc[allc.kept & allc.productive]
    hc = allc[allc.high_confidence & allc.productive]
    pr = allc[allc.productive]
    t4top = top_by(t4, ["umis"])
    stop = top_by(kp, ["umis"])
    by_bc = {b: g for b, g in allc.groupby("barcode")}
    kp_set = set(zip(kp.barcode, kp.chain, kp.cdr3_nt))
    pr_set = set(zip(pr.barcode, pr.chain, pr.cdr3_nt))
    out = {"sdir": sdir, "cells": len(pairs)}
    out["rules"] = {
        "umis": acc(pairs, stop),
        "reads": acc(pairs, top_by(kp, ["reads"])),
        "umis_then_reads": acc(pairs, top_by(kp, ["umis", "reads"])),
        "junction_umis": acc(pairs, top_by(kp, ["junction_umis", "umis"])),
        "hc_umis": acc(pairs, top_by(hc, ["umis"])),
        "oracle_kept": float(np.mean([(b, x, nx) in kp_set and (b, y, ny) in kp_set for b, x, y, nx, ny in pairs])),
        "oracle_any": float(np.mean([(b, x, nx) in pr_set and (b, y, ny) in pr_set for b, x, y, nx, ny in pairs])),
        "trust4": acc(pairs, t4top),
    }
    out["trust4_on_singlet"] = {"reads": acc(pairs, trust4_rule(kp, "reads")), "umis": acc(pairs, trust4_rule(kp, "umis"))}
    ct = Counter()
    cls_all, cls_lost = Counter(), Counter()
    kn = []
    for b, x, y, nx, ny in pairs:
        s_ok = stop.get((b, x)) == nx and stop.get((b, y)) == ny
        t_ok = t4top.get((b, x)) == nx and t4top.get((b, y)) == ny
        ct[f"S{'+' if s_ok else '-'}T{'+' if t_ok else '-'}"] += 1
        if s_ok:
            continue
        for ch, rn in ((x, nx), (y, ny)):
            if stop.get((b, ch)) == rn:
                continue
            if gex is not None and b not in gex:
                k, d = "outside_gex:-", None
            else:
                k, d = classify(b, ch, rn, by_bc.get(b), P, jmax_of)
            cls_all[f"{ch}:{k}"] += 1
            if t_ok:
                cls_lost[f"{ch}:{k}"] += 1
                if d:
                    d.update({"chain": ch, "trust4_right": True})
                    kn.append(d)
    out["crosstab"] = dict(ct)
    out["loss_classes_all"] = dict(sorted(cls_all.items()))
    out["loss_classes_vs_trust4"] = dict(sorted(cls_lost.items()))
    tot = lambda c: dict(Counter(k.split(":")[1] for k in c.elements()))  # noqa: E731
    out["loss_split_vs_trust4"] = tot(cls_lost)
    if kn:
        k = pd.DataFrame(kn)
        out["kept_not_top_vs_trust4"] = {"why": k.why.value_counts().to_dict(),
                                         "median": k[["umis", "reads", "top_umis", "top_reads"]].median().to_dict(),
                                         "c_pairs": Counter(zip(k.c, k.top_c)).most_common(8)}
    # light chain: the reference's light chain type vs Singlet's / TRUST4's most abundant light chain
    lt = Counter()
    kl = kp[kp.chain.isin(LIGHT)].sort_values("umis", ascending=False).drop_duplicates("barcode")
    sl = dict(zip(kl.barcode, kl.chain))
    tl = t4[t4.chain.isin(LIGHT)].sort_values("umis", ascending=False).drop_duplicates("barcode")
    tl = dict(zip(tl.barcode, tl.chain))
    for b, x, y, nx, ny in pairs:
        lt[f"ref {y} / singlet top light {sl.get(b, '-')} / trust4 {tl.get(b, '-')}"] += 1
    out["light_type"] = dict(lt.most_common(10))
    # J calls on exact junction matches
    seg = {s.name: s for s in segs if s.region == "J"} if segs else {}
    mt = kp.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    j = rt.merge(mt[["barcode", "chain", "cdr3_nt", "j_gene"]], on=["barcode", "chain"], suffixes=("_r", "_s"))
    j = j[j.cdr3_nt_r == j.cdr3_nt_s]
    j["j_s"] = j.j_gene.map(CV.norm_gene)
    tj = t4.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    jt = rt.merge(tj[["barcode", "chain", "cdr3_nt", "j"]], on=["barcode", "chain"], suffixes=("_r", "_t"))
    jt = jt[jt.cdr3_nt_r == jt.cdr3_nt_t]
    bad = j[j.j != j.j_s]
    ev = Counter()
    for r in bad.itertuples():
        mr, ms = j_end_match(r.cdr3_nt_r, seg.get(r.j)), j_end_match(r.cdr3_nt_r, seg.get(r.j_s))
        ev["ref_J_better" if mr > ms else "equal" if mr == ms else "singlet_J_better"] += 1
    out["j"] = {"singlet_agreement": 1 - len(bad) / max(1, len(j)), "n": len(j),
                "trust4_agreement": float((jt.j_r == jt.j_t).mean()) if len(jt) else None, "n_trust4": len(jt),
                "disagreements_by_chain": bad.chain.value_counts().to_dict(),
                "junction_end_evidence": dict(ev),
                "top": [[f"{p}->{q}", n] for (p, q), n in Counter(zip(bad.j, bad.j_s)).most_common(10)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--sdirs", default="singlet_base")
    ap.add_argument("--vdj-ref", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lib = a.lib
    ref = CV.load_cellranger(os.path.join(a.data, lib, "filtered_contig_annotations.csv"))
    ref = ref[ref.chain.isin(CV.BCR)]
    r = os.path.join(a.runs, lib)
    gp = os.path.join(r, "gex_cells.txt")
    gex = set(l.strip() for l in open(gp) if l.strip()) if os.path.exists(gp) else None
    # as primary.py: reference pairs over all Cell Ranger cells, methods restricted to GEX cells
    pairs, rt = ref_pairs(ref)
    t4 = CV.load_trust4(os.path.join(r, "trust4", "TRUST_barcode_report.tsv"))
    t4 = t4[t4.chain.isin(CV.BCR)]
    if gex is not None:
        t4 = t4[t4.barcode.isin(gex)]
    segs = None
    if a.vdj_ref:
        from singlet.vdj import reference as VR
        segs = VR.load(a.vdj_ref)
    res = {"lib": lib, "reference_paired_cells": len(pairs),
           "runs": [run_one(a, lib, s, ref, pairs, rt, gex, t4, segs) for s in a.sdirs.split(",")]}
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
