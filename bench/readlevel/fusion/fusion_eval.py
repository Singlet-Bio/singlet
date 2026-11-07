# SPDX-License-Identifier: MIT
"""Chimeric junctions of the engine against STAR-Fusion and Arriba on a droplet library of cell lines with known fusions.

Two levels, both on real data with literature-validated fusions as truth:

* Sample level: each caller's fusion calls (gene pairs, unordered) ranked by support. Recall = truth pairs called; false positives =
  other pairs called. Every caller's default call set is one point; the engine's ranked list gives a curve, read at each comparator's
  false-positive count ("recall at matched false positives").
* Cell level: cells are assigned to their line (``--labels``). For each truth fusion, recall = fraction of the positive line's cells with
  at least one supporting molecule; false-positive rate = the same fraction in the other lines' cells.

The comparators report the names of their supporting reads; ``readmap`` looks them up in R1 to get the cell barcode and UMI.

    python fusion_eval.py readmap --names names.txt --r1 L1_R1.fastq.gz,L2_R1.fastq.gz --cb-len 16 --umi-len 10 --out readmap.tsv
    python fusion_eval.py eval --engine chimeric_junctions.tsv.gz --starfusion star-fusion.fusion_predictions.tsv \
        --arriba fusions.tsv --readmap readmap.tsv --labels labels.tsv --truth "K562:BCR--ABL1,NUP214--XKR3;Kasumi-1:RUNX1--RUNX1T1" --out eval.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def pair(a: str, b: str) -> str:
    return "--".join(sorted((a, b)))


def first_gene(g: str) -> str:
    # Arriba writes "GENE1,GENE2" or "GENE(dist),GENE(dist)" for intergenic breakpoints
    return g.split(",")[0].split("(")[0]


# --------------------------------------------------------------------------- comparator outputs


def starfusion_calls(path):
    calls, reads = {}, {}
    with opener(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            a, b = row["#FusionName"].split("--")[:2]
            p = pair(a, b)
            sup = int(row["JunctionReadCount"]) + int(row["SpanningFragCount"])
            calls[p] = max(calls.get(p, 0), sup)
            names = [x for x in (row.get("JunctionReads", "") + "," + row.get("SpanningFrags", "")).split(",") if x and x != "."]
            reads.setdefault(p, set()).update(n.split("/")[0] for n in names)
    return calls, reads


def arriba_calls(path, confidences=("high", "medium", "low")):
    calls, reads = {}, {}
    with opener(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["confidence"] not in confidences:
                continue
            p = pair(first_gene(row["#gene1"]), first_gene(row["gene2"]))
            sup = int(row["split_reads1"]) + int(row["split_reads2"]) + int(row["discordant_mates"])
            calls[p] = max(calls.get(p, 0), sup)
            ids = row.get("read_identifiers", ".")
            if ids and ids != ".":
                reads.setdefault(p, set()).update(n.split("/")[0] for n in ids.split(","))
    return calls, reads


# --------------------------------------------------------------------------- engine table

MITO = {"chrM", "MT", "M"}


def engine_rows(path):
    with opener(path) as f:
        yield from csv.DictReader(f, delimiter="\t")


def engine_calls(path, cells=None, min_dist=1_000_000, canonical_only=False, allow_multimap=False):
    """Gene pairs of the engine's table after generic artifact filters (no catalogue): both sides in genes, different genes, not chrM,
    not a same-strand neighbour within ``min_dist`` on one chromosome (read-through), and optionally canonical-motif junctions and
    uniquely placed chimeric reads only. Support = molecules summed over cells and junctions; per-cell molecules are kept."""
    total = collections.Counter()
    per_cell = collections.defaultdict(collections.Counter)
    for r in engine_rows(path):
        if cells is not None and r["barcode"] not in cells:
            continue
        a, b = r["donor_gene"], r["acceptor_gene"]
        if a == "." or b == "." or a == b:
            continue
        if r["donor_chr"] in MITO or r["acceptor_chr"] in MITO:
            continue
        if r["donor_chr"] == r["acceptor_chr"] and abs(int(r["donor_pos"]) - int(r["acceptor_pos"])) < min_dist:
            continue
        if canonical_only and int(r["junction_type"]) < 1:
            continue
        if not allow_multimap and r["multimap"] == "1":
            continue
        p = pair(a, b)
        n = int(r["n_umis"])
        total[p] += n
        per_cell[r["barcode"]][p] += n
    return dict(total), per_cell


# --------------------------------------------------------------------------- read map and labels


def load_readmap(path):
    m = {}
    with opener(path) as f:
        for ln in f:
            name, cb, umi = ln.rstrip("\n").split("\t")
            m[name] = (cb, umi)
    return m


def load_labels(path):
    lab = {}
    with opener(path) as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if row and row[0] != "barcode":
                lab[row[0]] = row[1]
    return lab


def correct(cb, known):
    if cb in known:
        return cb
    hits = set()
    for i, c in enumerate(cb):
        for b in "ACGT":
            if b != c:
                t = cb[:i] + b + cb[i + 1 :]
                if t in known:
                    hits.add(t)
    return hits.pop() if len(hits) == 1 else None


def comparator_per_cell(reads, readmap, known):
    per_cell = collections.defaultdict(lambda: collections.defaultdict(set))
    for p, names in reads.items():
        for n in names:
            if n not in readmap:
                continue
            cb, umi = readmap[n]
            c = correct(cb, known)
            if c is not None:
                per_cell[c][p].add(umi)
    return {c: collections.Counter({p: len(u) for p, u in d.items()}) for c, d in per_cell.items()}


# --------------------------------------------------------------------------- metrics


def sample_point(calls, truth):
    called = set(calls)
    tp = sorted(called & truth)
    return {"n_calls": len(called), "truth_found": tp, "recall": len(tp) / len(truth), "false_positives": len(called - truth)}


def curve(calls, truth):
    """Recall and false positives at every support threshold, strongest calls first."""
    pts, tp, fp = [], 0, 0
    order = sorted(calls.items(), key=lambda kv: (-kv[1], kv[0]))
    i = 0
    while i < len(order):
        s = order[i][1]
        while i < len(order) and order[i][1] == s:
            if order[i][0] in truth:
                tp += 1
            else:
                fp += 1
            i += 1
        pts.append({"min_support": s, "recall": tp / len(truth), "false_positives": fp})
    return pts


def recall_at_fp(pts, fp_max):
    best = 0.0
    for p in pts:
        if p["false_positives"] <= fp_max:
            best = max(best, p["recall"])
    return best


def cell_metrics(per_cell, labels, truth_by_line):
    out = {}
    lines = collections.Counter(labels.values())
    for line, fus in truth_by_line.items():
        for p in fus:
            pos = [c for c, l in labels.items() if l == line]
            neg = [c for c, l in labels.items() if l != line]
            hit_pos = sum(1 for c in pos if per_cell.get(c, {}).get(p, 0) > 0)
            hit_neg = sum(1 for c in neg if per_cell.get(c, {}).get(p, 0) > 0)
            out[p] = {
                "line": line,
                "cells_line": len(pos),
                "cells_other": len(neg),
                "recall": round(hit_pos / max(len(pos), 1), 4),
                "fpr": round(hit_neg / max(len(neg), 1), 5),
                "cells_called_line": hit_pos,
                "cells_called_other": hit_neg,
            }
    out["_lines"] = dict(lines)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("names")
    a.add_argument("--starfusion")
    a.add_argument("--arriba")
    a.add_argument("--out", required=True)
    b = sub.add_parser("readmap")
    b.add_argument("--names", required=True)
    b.add_argument("--r1", required=True)
    b.add_argument("--cb-len", type=int, default=16)
    b.add_argument("--umi-len", type=int, default=10)
    b.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--engine", required=True)
    e.add_argument("--starfusion", required=True)
    e.add_argument("--arriba", required=True)
    e.add_argument("--readmap", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--truth", required=True)
    e.add_argument("--out", required=True)
    x = ap.parse_args()

    if x.cmd == "names":
        names = set()
        for path, fn in ((x.starfusion, starfusion_calls), (x.arriba, arriba_calls)):
            if path:
                for v in fn(path)[1].values():
                    names |= v
        with open(x.out, "w") as f:
            f.write("".join(n + "\n" for n in sorted(names)))
        print(len(names), "read names")
        return
    if x.cmd == "readmap":
        want = {ln.strip() for ln in open(x.names) if ln.strip()}
        n = 0
        with open(x.out, "w") as o:
            for fq in x.r1.split(","):
                with opener(fq) as f:
                    while True:
                        h = f.readline()
                        if not h:
                            break
                        s = f.readline().strip()
                        f.readline()
                        f.readline()
                        name = h[1:].split()[0].split("/")[0]
                        if name in want:
                            o.write(f"{name}\t{s[: x.cb_len]}\t{s[x.cb_len : x.cb_len + x.umi_len]}\n")
                            n += 1
        print(n, "of", len(want), "names found", file=sys.stderr)
        return

    truth_by_line = {}
    for part in x.truth.split(";"):
        line, fus = part.split(":")
        truth_by_line[line] = [pair(*f.split("--")) for f in fus.split(",")]
    truth = {p for v in truth_by_line.values() for p in v}
    labels = load_labels(x.labels)
    cells = set(labels)
    rm = load_readmap(x.readmap)

    sf, sf_reads = starfusion_calls(x.starfusion)
    ar, ar_reads = arriba_calls(x.arriba)
    ar_hm = {p: s for p, s in arriba_calls(x.arriba, ("high", "medium"))[0].items()}
    res = {"truth": sorted(truth), "cells_labelled": len(cells)}
    variants = {
        "engine_all_barcodes": engine_calls(x.engine),
        "engine_cells": engine_calls(x.engine, cells),
        "engine_cells_canonical": engine_calls(x.engine, cells, canonical_only=True),
    }
    res["sample"] = {"starfusion": sample_point(sf, truth), "arriba": sample_point(ar, truth), "arriba_high_medium": sample_point(ar_hm, truth)}
    res["curves"] = {"starfusion": curve(sf, truth), "arriba": curve(ar, truth)}
    for k, (calls, _) in variants.items():
        pts = curve(calls, truth)
        res["curves"][k] = pts[:200]
        res["sample"][k + "_min2"] = sample_point({p: s for p, s in calls.items() if s >= 2}, truth)
        res["sample"][k + "_recall_at_fp"] = {
            c: {"fp": res["sample"][c]["false_positives"], "recall_engine": recall_at_fp(pts, res["sample"][c]["false_positives"]),
                "recall_comparator": res["sample"][c]["recall"]}
            for c in ("starfusion", "arriba", "arriba_high_medium")
        }
    res["cell"] = {
        "engine": cell_metrics(variants["engine_cells"][1], labels, truth_by_line),
        "starfusion": cell_metrics(comparator_per_cell(sf_reads, rm, cells), labels, truth_by_line),
        "arriba": cell_metrics(comparator_per_cell(ar_reads, rm, cells), labels, truth_by_line),
    }
    res["support_truth"] = {
        "starfusion": {p: sf.get(p, 0) for p in truth},
        "arriba": {p: ar.get(p, 0) for p in truth},
        "engine_cells": {p: variants["engine_cells"][0].get(p, 0) for p in truth},
    }
    with open(x.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("sample", "cell", "support_truth")}, indent=1))


if __name__ == "__main__":
    main()
