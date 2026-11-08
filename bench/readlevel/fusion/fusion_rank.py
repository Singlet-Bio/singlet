# SPDX-License-Identifier: MIT
"""B6 ranking evidence: rescore the engine's chimeric gene pairs by breakpoint consistency across cells and a known-fusion prior,
then apply the committed B6 rule (ORACLES_B6_B8.md, 6de5821a) unchanged. The formula and weights are fixed in ORACLES_B6_B8.md
("B6 ranking evidence: pre-registered score") before any mix is scored.

Per gene pair p (unordered), from the PASS rows of chimeric_junctions.tsv.gz in labelled cells, after fusion_eval's generic filters:

  M  molecules: n_umis summed over cells and junctions (the old ranking)
  clusters: rows sorted by (donor_chr, donor_pos, acceptor_chr, acceptor_pos); a row joins the first cluster whose first row has the
     same chromosomes and both positions within 5 bp, else it starts a cluster. The top cluster has the most distinct cells (ties:
     most molecules, then lowest key).
  C  distinct cells (barcodes) in the top cluster. Each cell's molecules there are distinct barcode+UMI pairs.
  f  molecules in the top cluster / M (agreement of the split-read position)
  s  0.5 * [donor position within 2 bp of an annotated exon boundary] + 0.5 * [acceptor position within 2 bp of one], from the top
     cluster's first row (canonical splice-site use at both ends; the motif is already GT/AG for every PASS row)
  K  1 when the pair is a fusion (A::B, either order) in the Mitelman Database MCGENE table, else 0

Scores (higher first; ties broken by M, remaining ties ranked together):
  baseline   M
  cells      C
  agreement  M * f
  splice     M * (1 + s)
  breakpoint C * f * (1 + s)            (feature 1 combined)
  known      M * (1 + 4K)               (feature 2 alone)
  combined   C * f * (1 + s) * (1 + 4K)
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_eval as fe  # noqa: E402

TOL_BP = 5
TOL_EXON = 2
W_KNOWN = 4.0
VARIANTS = ("baseline", "cells", "agreement", "splice", "breakpoint", "known", "combined")


def load_exon_boundaries(gtf):
    b = collections.defaultdict(set)
    with fe.opener(gtf) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            t = ln.split("\t", 5)
            if len(t) > 4 and t[2] == "exon":
                b[t[0]].add(int(t[3]))
                b[t[0]].add(int(t[4]))
    return {c: sorted(v) for c, v in b.items()}


def near(bounds, chrom, pos, tol=TOL_EXON):
    v = bounds.get(chrom)
    if not v:
        return False
    i = bisect.bisect_left(v, pos - tol)
    return i < len(v) and v[i] <= pos + tol


def load_mitelman(mcgene):
    pairs = set()
    with open(mcgene, encoding="utf-8", errors="replace") as f:
        next(f)
        for ln in f:
            t = ln.rstrip("\n").split("\t")
            if len(t) > 5 and "::" in t[5]:
                g = [x.strip() for x in t[5].split("::") if x.strip()]
                for a, c in zip(g, g[1:]):
                    if a != c:
                        pairs.add(fe.pair(a, c))
    return pairs


def pair_rows(path, cells, min_dist=1_000_000):
    rows = collections.defaultdict(list)
    for r in fe.engine_rows(path):
        if r["barcode"] not in cells:
            continue
        a, b = r["donor_gene"], r["acceptor_gene"]
        if a == "." or b == "." or a == b or r["donor_chr"] in fe.MITO or r["acceptor_chr"] in fe.MITO:
            continue
        if r["donor_chr"] == r["acceptor_chr"] and abs(int(r["donor_pos"]) - int(r["acceptor_pos"])) < min_dist:
            continue
        if r["multimap"] == "1" or r.get("filter", "PASS") != "PASS":
            continue
        rows[fe.pair(a, b)].append((r["donor_chr"], int(r["donor_pos"]), r["acceptor_chr"], int(r["acceptor_pos"]), r["barcode"], int(r["n_umis"])))
    return rows


def features(rows, bounds, known_pairs):
    out = {}
    for p, rs in rows.items():
        rs = sorted(rs)
        clusters = []  # [first_row, cells(set), molecules]
        for r in rs:
            for c in clusters:
                h = c[0]
                if h[0] == r[0] and h[2] == r[2] and abs(h[1] - r[1]) <= TOL_BP and abs(h[3] - r[3]) <= TOL_BP:
                    c[1].add(r[4]); c[2] += r[5]
                    break
            else:
                clusters.append([r, {r[4]}, r[5]])
        top = min(clusters, key=lambda c: (-len(c[1]), -c[2], c[0][:4]))
        M = sum(r[5] for r in rs)
        h = top[0]
        s = 0.5 * near(bounds, h[0], h[1]) + 0.5 * near(bounds, h[2], h[3])
        out[p] = {"M": M, "C": len(top[1]), "f": round(top[2] / M, 4), "s": s, "K": int(p in known_pairs),
                  "n_clusters": len(clusters), "top": f"{h[0]}:{h[1]}>{h[2]}:{h[3]}"}
    return out


def score(v, x):
    M, C, f, s, K = x["M"], x["C"], x["f"], x["s"], x["K"]
    return {"baseline": M, "cells": C, "agreement": M * f, "splice": M * (1 + s), "breakpoint": C * f * (1 + s),
            "known": M * (1 + W_KNOWN * K), "combined": C * f * (1 + s) * (1 + W_KNOWN * K)}[v]


def curve(keys, truth):
    order = sorted(keys.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
    pts, tp, fp, i = [], 0, 0, 0
    while i < len(order):
        k = order[i][1]
        while i < len(order) and order[i][1] == k:
            tp += order[i][0] in truth
            fp += order[i][0] not in truth
            i += 1
        pts.append({"score": round(k[0], 4), "recall": tp / len(truth), "false_positives": fp})
    return pts, [p for p, _ in order]


def main():
    ap = argparse.ArgumentParser()
    for a in ("engine", "starfusion", "arriba", "readmap", "labels", "truth", "gtf", "mitelman", "out"):
        ap.add_argument("--" + a, required=True)
    x = ap.parse_args()
    truth_by_line = {}
    for part in x.truth.split(";"):
        line, fus = part.split(":")
        truth_by_line[line] = [fe.pair(*f.split("--")) for f in fus.split(",")]
    truth = {p for v in truth_by_line.values() for p in v}
    labels = fe.load_labels(x.labels)
    cells = set(labels)
    sf, sf_reads = fe.starfusion_calls(x.starfusion)
    ar, ar_reads = fe.arriba_calls(x.arriba)
    comps = {"starfusion": fe.sample_point(sf, truth), "arriba": fe.sample_point(ar, truth)}

    bounds = load_exon_boundaries(x.gtf)
    known_pairs = load_mitelman(x.mitelman)
    feats = features(pair_rows(x.engine, cells), bounds, known_pairs)
    res = {"truth": sorted(truth), "mitelman_pairs": len(known_pairs), "truth_in_mitelman": {p: p in known_pairs for p in sorted(truth)},
           "n_pairs": len(feats), "pairs_in_mitelman": sum(v["K"] for v in feats.values()),
           "truth_features": {p: feats.get(p) for p in sorted(truth)}, "comparators": comps, "variants": {}}
    for v in VARIANTS:
        keys = {p: (score(v, f), f["M"]) for p, f in feats.items()}
        pts, order = curve(keys, truth)
        rat = {c: {"fp": comps[c]["false_positives"], "recall_engine": fe.recall_at_fp(pts, comps[c]["false_positives"]),
                   "recall_comparator": comps[c]["recall"]} for c in comps}
        for c in rat:
            rat[c]["pass"] = rat[c]["recall_engine"] >= rat[c]["recall_comparator"]
        res["variants"][v] = {"recall_at_fp": rat, "part1_pass": all(r["pass"] for r in rat.values()),
                              "truth_rank": {p: (order.index(p) + 1 if p in order else None) for p in sorted(truth)},
                              "top15": [[p, round(keys[p][0], 3), feats[p]] for p in order[:15]]}
    # rule part 2: cell recall per truth fusion (>= 1 supporting molecule per cell); the ranking does not enter it
    rm = fe.load_readmap(x.readmap)
    eng_cells = fe.cell_metrics(fe.engine_calls(x.engine, cells)[1], labels, truth_by_line)
    cm = {"starfusion": fe.cell_metrics(fe.comparator_per_cell(sf_reads, rm, cells), labels, truth_by_line),
          "arriba": fe.cell_metrics(fe.comparator_per_cell(ar_reads, rm, cells), labels, truth_by_line)}
    part2 = {}
    for p in sorted(truth):
        best = max(cm, key=lambda c: cm[c][p]["recall"])
        e = eng_cells[p]
        part2[p] = {"engine_cells": e["cells_called_line"], "engine_fpr": e["fpr"], "best": best, "best_cells": cm[best][p]["cells_called_line"],
                    "pass": e["recall"] >= cm[best][p]["recall"] and e["fpr"] <= max(cm[best][p]["fpr"], 0.005)}
    res["part2"] = part2
    res["part2_pass"] = all(v["pass"] for v in part2.values())
    with open(x.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"truth_in_mitelman": res["truth_in_mitelman"], "truth_features": res["truth_features"],
                      "part1": {v: (res["variants"][v]["part1_pass"], res["variants"][v]["truth_rank"],
                                    {c: r["recall_engine"] for c, r in res["variants"][v]["recall_at_fp"].items()}) for v in VARIANTS},
                      "part2": part2}, indent=1))


if __name__ == "__main__":
    main()
