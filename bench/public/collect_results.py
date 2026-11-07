#!/usr/bin/env python3
"""Collect the newest results/<capability>.json from every remote branch into one directory (for release_gate.py).

    python bench/public/collect_results.py OUT_DIR [branch-regex]

Capability owners deliver on their own branches; this picks, per capability, the file from the branch whose last commit is
newest and prints where each came from. Results from different commits are mixed, so the gate's `result_commits` lists them.
"""
import os
import re
import subprocess
import sys

out = sys.argv[1]
rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"bench/|bench-public|cell-type|vdj|atac|adt|mtdna")
os.makedirs(out, exist_ok=True)
best = {}
for b in subprocess.check_output(["git", "branch", "-r"], text=True).split():
    if not rx.search(b):
        continue
    date = int(subprocess.check_output(["git", "log", "-1", "--format=%ct", b], text=True))
    for f in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", b], text=True).split("\n"):
        m = re.fullmatch(r"bench/public/results/([a-z-]+)\.json", f)
        if m and m.group(1) != "index" and (m.group(1) not in best or date > best[m.group(1)][0]):
            best[m.group(1)] = (date, b, f)
for c, (_, b, f) in sorted(best.items()):
    with open(os.path.join(out, c + ".json"), "wb") as fh:
        fh.write(subprocess.check_output(["git", "show", f"{b}:{f}"]))
    print(c, b)
