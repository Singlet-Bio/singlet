"""Standing STARsolo gate: is there a passing gate record for the current counting sources?

  python bench/count_parity/gate_check.py hash [ROOT]    print the counting-source hash of a checkout
  python bench/count_parity/gate_check.py check [ROOT]   exit 1 unless gate_results/<hash16>.json exists, matches and says PASS

The hash is sha256 over the files in counting_sources.txt (path, NUL, content with CRLF normalized to LF, NUL), in list order,
so Windows and Linux checkouts agree. gate.sbatch writes the record on Clipper after it has run a binary built from the commit
on the gate datasets and scored it against STARsolo (parity_score.py). Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def sources(root=ROOT):
    with open(os.path.join(root, "bench", "count_parity", "counting_sources.txt")) as f:
        return [ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()]


def source_hash(root=ROOT):
    h = hashlib.sha256()
    for p in sources(root):
        h.update(p.encode() + b"\0")
        fp = os.path.join(root, p)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                h.update(f.read().replace(b"\r\n", b"\n"))
        else:
            h.update(b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


def check(root=ROOT):
    sh = source_hash(root)
    rec = os.path.join(root, "bench", "count_parity", "gate_results", sh[:16] + ".json")
    if not os.path.exists(rec):
        print(f"FAIL: no STARsolo gate record for counting sources {sh[:16]}.\n"
              "A file in bench/count_parity/counting_sources.txt changed. Build this commit, run bench/count_parity/gate.sbatch\n"
              f"on Clipper with that binary, and commit its record as bench/count_parity/gate_results/{sh[:16]}.json.")
        return 1
    with open(rec) as f:
        r = json.load(f)
    if r.get("source_hash") != sh:
        print(f"FAIL: {rec} is for source hash {r.get('source_hash')}, this checkout is {sh}")
        return 1
    bad = [f"{ds}:{ax}" for ds, d in r.get("datasets", {}).items() for ax, v in d.get("axes", {}).items() if v.get("verdict") == "FAIL"]
    if r.get("gate") != "PASS" or bad or not r.get("datasets"):
        print(f"FAIL: gate record {rec} says {r.get('gate')}; axes worse than STARsolo beyond their CI: {', '.join(bad) or 'none listed'}")
        return 1
    print(f"PASS: counting sources {sh[:16]} gated against STARsolo at commit {r.get('commit', '?')[:12]} ({r.get('date', '?')})")
    for ds, d in r["datasets"].items():
        for ax, v in d["axes"].items():
            lo, hi = v["diff_ci"]
            print(f"  {ds:16s} {ax:16s} singlet {v['singlet']:.6g}  starsolo {v['starsolo']:.6g}  diff CI [{lo:+.3g}, {hi:+.3g}]  {v['verdict']}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    root = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else ROOT
    if cmd == "hash":
        print(source_hash(root))
    else:
        sys.exit(check(root))
