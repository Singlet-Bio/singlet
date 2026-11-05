#!/usr/bin/env python3
"""Copy the raw point.json, results.tsv and runs.tsv of every leg of a finished fast lane run from Clipper into its out directory.
Usage: fetch_points.py OUT_DIR     (reads state.json for the remote run directory). Small files only. Reads the cluster, writes nothing there."""
import json
import os
import subprocess
import sys

CL = [sys.executable, os.path.join(os.path.expanduser("~"), ".claude", "skills", "clipper", "clipper.py")]


def run(cmd):
    return subprocess.run(CL + ["run", cmd], capture_output=True, text=True).stdout


def main(out):
    st = json.load(open(os.path.join(out, "state.json")))
    res = st["run"] + "/res"
    legs = [l for l in run(f"ls {res}").split() if l.startswith("leg")]
    for l in legs:
        d = os.path.join(out, "legs", l)
        os.makedirs(d, exist_ok=True)
        for f in ("point.json", "results.tsv", "runs.tsv", "fallback.txt", "cotenants.log"):
            subprocess.run(CL + ["get", f"{res}/{l}/{f}", os.path.abspath(os.path.join(d, f))], capture_output=True)
    print(f"{out}: {len(legs)} legs copied")


if __name__ == "__main__":
    main(sys.argv[1])
