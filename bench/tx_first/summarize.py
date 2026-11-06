#!/usr/bin/env python3
"""Overnight summary of the tx-first and multi-GPU studies.
Usage: summarize.py RESDIR [STUDY2_JOBID_DIR_GLOB]
Reads RESDIR/study1/*.json and RESDIR/study2/tp-*/results.tsv (+ the runs' star_Log.out) and writes RESDIR/overnight/summary.md and summary.json.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

res = sys.argv[1]
out = os.path.join(res, "overnight")
os.makedirs(out, exist_ok=True)
S = {"study1": {}, "study2": {}}
md = ["# Overnight summary: transcriptome-first concordance and multi-GPU", ""]


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "NA"


# ---------------------------------------------------------------- study 1
md += ["## Study 1: transcriptome-first concordance (CPU)", "",
       "Reads: cDNA read of the first lane, STAR 2.7.11b genome record as reference, STAR end-to-end transcriptome pass (at most 1 mismatch, one gene).", ""]
rows = []
for f in sorted(glob.glob(res + "/study1/*.json")):
    j = json.load(open(f))
    S["study1"][j["sample"]] = j
    n = j["reads"]
    cc = j["claimed_components"]
    claimed = j["claim_kind"].get("one_gene", 0)
    full = sum(j["fullmatch_by_class"].values())
    rows.append((j["sample"], n, claimed, cc.get("locus_cigar_equal", 0), full))
md += ["| sample | reads | claimed (1 gene) | locus and CIGAR equal | full match (also NH 1, AS equal) | full match of claimed |", "|---|---|---|---|---|---|"]
for s, n, cl, lc, fu in rows:
    md.append("| %s | %d | %s | %s | %s | %s |" % (s, n, pct(cl, n), pct(lc, n), pct(fu, n), pct(fu, cl)))
md += ["", "### Lane proxy: share of reads claimed, and share that match exactly", "",
       "| sample | class | reads | claimed | full match |", "|---|---|---|---|---|"]
for s, j in S["study1"].items():
    for c in ("light", "heavy_multi", "heavy_clip", "other_multisplice", "unmapped"):
        n = j["lane_class"].get(c, 0)
        md.append("| %s | %s | %s | %s | %s |" % (s, c, pct(n, j["reads"]), pct(j["claimed_by_class"].get(c, 0), n), pct(j["fullmatch_by_class"].get(c, 0), n)))
md += ["", "### Why claimed reads disagree with STAR (share of claimed reads)", "", "| sample | reason | share |", "|---|---|---|"]
for s, j in S["study1"].items():
    tot = sum(j["disagreement"].values())
    for k, v in sorted(j["disagreement"].items(), key=lambda x: -x[1]):
        md.append("| %s | %s | %s |" % (s, k, pct(v, tot)))
md += ["", "### Genomic fate of unclaimed, genome-mapped reads (share of all reads)", "", "| sample | class and fate | share |", "|---|---|---|"]
for s, j in S["study1"].items():
    for k, v in sorted(j["unclaimed_fate"].items(), key=lambda x: -x[1])[:8]:
        md.append("| %s | %s | %s |" % (s, k, pct(v, j["reads"])))

# ---------------------------------------------------------------- study 2
md += ["", "## Study 2: two GPUs of one node (binary b0, c3a6096c, pbmc1k_v3 level 5)", ""]
for d in sorted(glob.glob(res + "/study2/tp-*")):
    tsv = d + "/results.tsv"
    if not os.path.exists(tsv):
        md.append("%s: no results.tsv" % os.path.basename(d))
        continue
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    md += ["Job directory %s, node %s" % (os.path.basename(d), rows[0]["node"] if rows else "?"), "",
           "| point | k | slots | runs | failed | progress-weighted rate (M pairs/s) | window rate | cores (window) | lock wait frac | host peak GB | device peak GB | fallback runs |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        S["study2"][r["point"]] = r
        pw = None
        pj = glob.glob("%s/%s/point.json" % (d, r["point"]))
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["point"], r["k"], r["slots"], r["runs"], r["failed_runs"], r["span_rate"], r["win_rate"],
                  r["cores_win"], r["lock_wait_frac"], r["host_peak_gb"], r["dev_peak_gb"], r["fallback_runs"]))
    md.append("")
    # progress-weighted rate with the gpu/bold analyzer when present
    ap = "/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-bold/jobs/analyze_points.py"
    # lane kernel seconds of the runs (engine step timers)
    md += ["Lane timing per run (engine step timers, mean over runs; B reads from the E17 chunk lines):", "",
           "| point | laneA kernel s | laneM kernel s | laneB kernel s | B share of kernel time | B reads per run | distinct reads per run | B share of distinct reads |", "|---|---|---|---|---|---|---|---|"]
    for pdir in sorted(glob.glob(d + "/k*")):
        a, m, b, nb, nd = [], [], [], [], []
        for lg in glob.glob(pdir + "/c*/star_Log.out"):
            t = open(lg, errors="replace").read()
            mm = re.search(r"laneA\.kernels ([\d.]+).*?laneM\.kernels ([\d.]+).*?laneB\.kernels ([\d.]+)", t)
            if not mm:
                continue
            a.append(float(mm.group(1))); m.append(float(mm.group(2))); b.append(float(mm.group(3)))
            nb.append(sum(int(x) for x in re.findall(r" B (\d+) x \d+", t)))
            dm = re.search(r"reads, (\d+) distinct", t)
            nd.append(int(dm.group(1)) if dm else 0)
        if a:
            ka, km, kb = st.mean(a), st.mean(m), st.mean(b)
            S["study2"].setdefault("lanes", {})[os.path.basename(pdir)] = {"A": ka, "M": km, "B": kb, "B_reads": st.mean(nb), "distinct": st.mean(nd)}
            md.append("| %s | %.2f | %.2f | %.2f | %s | %d | %d | %s |" % (os.path.basename(pdir), ka, km, kb, pct(kb, ka + km + kb), st.mean(nb), st.mean(nd), pct(st.mean(nb), st.mean(nd))))
    md.append("")

# ---------------------------------------------------------------- recommendation
rec = []
s1 = S["study1"]
if s1:
    lf = [j["fullmatch_by_class"].get("light", 0) / max(1, j["lane_class"].get("light", 1)) for j in s1.values()]
    hv = [j["claimed_by_class"].get("heavy_multi", 0) / max(1, j["lane_class"].get("heavy_multi", 1)) for j in s1.values()]
    rec.append("Transcriptome-first: exact match covers %.0f%% to %.0f%% of light-proxy reads across samples. Heavy multimappers claimed: %.1f%% to %.1f%%. "
               "Because the engine must stay byte-identical, a fast path is only safe where a read matches exactly, and a cheap check is still needed to send other reads on."
               % (100 * min(lf), 100 * max(lf), 100 * min(hv), 100 * max(hv)))
two, one = S["study2"].get("k8t7s2r3_base"), S["study2"].get("k8t7s2r3_g1")
if two and one:
    try:
        rec.append("Second GPU: progress-weighted span rate %s versus %s M pairs/s on one GPU at the same k." % (two["span_rate"], one["span_rate"]))
    except KeyError:
        pass
md += ["## Recommendation", ""] + rec + ["", "Heavy reads on a second GPU: see docs/gpu-engine/bold/HARD-GPU.md (branch bench/tx-first)."]
S["recommendation"] = rec
open(out + "/summary.md", "w").write("\n".join(md) + "\n")
json.dump(S, open(out + "/summary.json", "w"), indent=1)
print("\n".join(md)[:3000])
