#!/usr/bin/env python3
"""Reads the per-library logs of run_panel.sbatch, writes calls.tsv and prints the confusion table.

A call is the protocol tag the engine settled on, or the gate's verdict when the gate stopped the run. Each tag is mapped to the class of
assay it names (CLASS below); a truth class and a call class agree when equal. Classes that no read statistic can separate share a name.
usage: score_panel.py JOBDIR
"""
import collections
import glob
import os
import re
import sys

CLASS = {
    "10x-3p-v1": "10x-3p", "10x-3p-v2": "10x-3p", "10x-3p-v3": "10x-3p", "10x-3p-v4": "10x-3p", "cite-seq-gex": "10x-3p",
    "10x-5p-v2": "10x-5p", "10x-5p-v3": "10x-5p", "10x-5p-v4": "10x-5p", "10x-arc-gex": "10x-multiome", "10x-visium": "visium",
    "dropseq": "dropseq", "seqwell": "dropseq", "celseq2": "celseq2", "celseq2-u6b8": "celseq2", "marsseq2": "marsseq", "scrbseq": "scrbseq",
    "icell8": "icell8", "ddseq": "ddseq", "surecell": "ddseq", "quartzseq2": "quartzseq2", "indrop": "indrop", "bd-rhapsody": "bd-rhapsody",
    "dnbelab-c4": "dnbelab-c4", "microwell-seq": "microwell-seq", "splitseq": "split-pool", "sci-rna-seq3": "split-pool", "strtseq": "strtseq",
}
ASSAY_CLASS = {"bulk-rna": "bulk-or-plate", "sc-rna-plate": "bulk-or-plate", "sc-atac": "atac", "cite-seq-adt": "adt",
               "sc-methylome": "bisulfite", "unknown": "no-call", "not-sequence": "no-call"}


def parse(log):
    t = open(log, errors="replace").read().replace("\r", "\n")
    g = re.findall(r"\[singlet-gate\] (REJECT|UNCERTAIN|PASS_10X|ROUTE)\b.*?tag=([A-Za-z0-9._-]+) confidence=(\d)", t)
    p = re.findall(r"\[singlet-download\] Protocol: ([A-Za-z0-9._-]+) \(confidence: (\d)", t)
    a = re.findall(r"\[1fq-encode\] Assay: ([a-z0-9-]+)", t)
    wl = re.findall(r"whitelist validation: \d+/\d+ match \((\d+)", t)
    reason = re.findall(r"reason=([a-z0-9_]+)", t)
    return dict(gate=g[-1][0] if g else "-", gate_tag=g[-1][1] if g else "-", proto=p[-1][0] if p else "-", conf=p[-1][1] if p else (g[-1][2] if g else "-"),
                assay=a[-1] if a else "-", wl=wl[-1] if wl else "-", reason=named[-1] if named else (reason[-1] if reason else "-"), incomplete="data_incomplete" in t)


def call_class(c):
    if c["reason"].startswith("unsupported_assay_"):
        return {"10x_flex": "flex", "singleron_gexscope": "singleron"}.get(c["reason"][len("unsupported_assay_"):], "named:" + c["reason"])
    if c["gate"] == "ROUTE" and c["proto"] == "-":
        return "routed-then-stopped:" + c["gate_tag"]
    if c["gate"] == "REJECT":
        return "rejected:" + c["gate_tag"]
    if c["assay"] == "cite-seq-adt":
        return "adt"
    if c["proto"] in CLASS and c["conf"] in ("2", "3", "4", "5"):
        return CLASS[c["proto"]]
    if c["assay"] in ASSAY_CLASS and ASSAY_CLASS[c["assay"]] != "no-call":
        return ASSAY_CLASS[c["assay"]]
    if c["proto"] in CLASS:
        return "low:" + CLASS[c["proto"]]
    return "no-call"


def main(d):
    rows = []
    for f in sorted(glob.glob(d + "/calls.part.*")):
        parts = open(f).read().rstrip("\n").split("\t")
        if len(parts) < 7:
            rows.append(parts + ["-"] * (13 - len(parts)))
            continue
        id_, group, truth, run, bn, rc, lens = parts[:7]
        c = parse(f"{d}/logs/{id_}.{bn}.log")
        rows.append([id_, group, truth, run, bn, rc, lens, c["gate"], c["gate_tag"], c["proto"], c["conf"], c["assay"], c["wl"], call_class(c)])
    hdr = ["id", "group", "truth", "run", "binary", "exit", "r1/r2", "gate", "gate_tag", "protocol", "conf", "assay", "wl_pct", "call"]
    with open(d + "/calls.tsv", "w") as o:
        o.write("\t".join(hdr) + "\n")
        for r in rows:
            o.write("\t".join(map(str, r)) + "\n")
    for bn in sorted({r[4] for r in rows}):
        sub = [r for r in rows if r[4] == bn and len(r) >= 14]
        ok = sum(r[13] == r[2] for r in sub)
        print(f"== binary {bn}: {ok} / {len(sub)} calls equal the truth class")
        conf = collections.Counter((r[2], r[13]) for r in sub)
        for (t, c), n in sorted(conf.items()):
            print(f"  {t:18s} -> {c:26s} {n}{'' if t == c else '   MISS'}")


if __name__ == "__main__":
    main(sys.argv[1])
