#!/usr/bin/env python3
"""Which legs shared the node with another job. Reads the legs/<leg>/cotenants.log files that fetch_points.py copies.

Usage: cotenants.py OUT_DIR [OUT_DIR ...]
For every leg: the window of its runs, and each other job seen in the snapshots with the first and last time seen.
The snapshot is the squeue of the node at the start, once a minute and at the end of the leg.
A job seen in any snapshot of a leg counts as overlapping that leg.
"""
import datetime as dt
import glob
import json
import os
import re
import sys


def parse(path, own=None):
    """-> list of (timestamp string, [(jobid, user, name, cpus, gres)]) in file order."""
    snaps = []
    for l in open(path, errors="replace"):
        l = l.rstrip("\n")
        if l.startswith("##"):
            snaps.append((l.split(None, 2)[2] if len(l.split(None, 2)) > 2 else "", []))
        elif l.strip() and not l.startswith("JOBID") and snaps:
            f = l.split(None, 4)
            if len(f) >= 3:
                snaps[-1][1].append(tuple(f) + ("",) * (5 - len(f)))
    return snaps


def others(snaps, own):
    seen = {}
    for ts, rows in snaps:
        for r in rows:
            jid = r[0].split("_")[0]
            if jid == own:
                continue
            e = seen.setdefault(jid, {"name": r[2], "cpus": r[3], "gres": r[4], "first": ts, "last": ts})
            e["last"] = ts
    return seen


def leg_window(leg_dir):
    p = os.path.join(leg_dir, "runs.tsv")
    if not os.path.exists(p):
        return None
    rows = [l.split("\t") for l in open(p).read().split("\n") if l and l[0].isdigit()]
    if not rows:
        return None
    return min(float(r[3]) for r in rows), max(float(r[4]) for r in rows)


def report(out_dir, tz_hours=-5, own=None, legs_root=None):
    """out_dir holds state.json and legs/ on the laptop. On Clipper pass legs_root=<run>/res and own=<job id>."""
    if own is None:
        own = str(json.load(open(os.path.join(out_dir, "state.json"))).get("job"))
    legs_root = legs_root or os.path.join(out_dir, "legs")
    tz = dt.timezone(dt.timedelta(hours=tz_hours))
    fmt = lambda x: dt.datetime.fromtimestamp(x, tz).strftime("%H:%M:%S")  # noqa: E731
    rows = []
    for leg in sorted(glob.glob(os.path.join(legs_root, "leg*")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        name = os.path.basename(leg)
        w = leg_window(leg)
        cp = os.path.join(leg, "cotenants.log")
        oth = others(parse(cp), own) if os.path.exists(cp) else None
        rows.append((name, fmt(w[0]) + " to " + fmt(w[1]) if w else "?", oth))
    return rows


def main(dirs):
    for d in dirs:
        print(f"== {os.path.basename(d.rstrip('/'))}")
        for name, win, oth in report(d):
            if oth is None:
                print(f"  {name} {win}: no cotenants.log copied")
            elif not oth:
                print(f"  {name} {win}: clean")
            else:
                print(f"  {name} {win}: CO-TENANT " + "; ".join(
                    f"job {j} {e['name']} {e['cpus']} cpus, seen {e['first'][11:19]} to {e['last'][11:19]}" for j, e in oth.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
