#!/usr/bin/env python3
"""Compares the calls of two binaries (rows 'before' and 'after', or the first two binary names) on the same runs.

A call is the protocol tag of the 'Protocol:' line, or gate:<tag> when the gate stopped the run, or '-' when neither was logged. Prints the
runs whose call differs, grouped by the stated assay, and the number of runs with an identical call.
usage: compare_b20.py calls.tsv
"""
import collections
import csv
import sys


def main(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    bins = sorted({r["binary"] for r in rows if r["binary"] not in ("-", "")})
    a, b = ("base", "new") if {"base", "new"} <= set(bins) else bins[:2]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["run"]][r["binary"]] = r
    same = diff = 0
    changes = collections.defaultdict(list)
    for run, d in sorted(by.items()):
        if a not in d or b not in d:
            continue
        ta, tb = d[a]["tag"], d[b]["tag"]
        ca, cb = d[a]["confidence"], d[b]["confidence"]
        if (ta, ca) == (tb, cb):
            same += 1
        else:
            diff += 1
            changes[d[a]["assay"]].append(f"{run}\t{ta}({ca}) -> {tb}({cb})")
    print(f"binaries {a} vs {b}: {same} runs with the same call and confidence, {diff} with a different one")
    tenx = {k: v for k, v in changes.items() if k.startswith(("10x", "visium"))}
    print(f"changed among the stated 10x and Visium assays: {sum(len(v) for v in tenx.values())}")
    for k in sorted(changes):
        print(f"== stated {k}: {len(changes[k])} changed")
        for line in changes[k]:
            print("  ", line)


if __name__ == "__main__":
    main(sys.argv[1])
