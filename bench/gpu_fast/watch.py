#!/usr/bin/env python3
"""Early-stop watcher of fast lane tier 2. Runs once per call and exits.

Usage: watch.py LEG_RES_DIR LEG_SCR_DIR
Looks at runs that finished at least GRACE seconds ago and have not been looked at. When one has a non-zero exit, no level 5, no GPU pileup,
or a disallowed fallback line (bench/gpu_batch1/fallback.py), it creates LEG_SCR_DIR/STOP and writes the reason to LEG_RES_DIR/STOPPED.
Workers stop taking new runs when STOP exists. Runs in flight finish. tp_point.py does the final count of fallback runs.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpu_batch1"))
import fallback  # noqa: E402

GRACE = 5.0


def problems(run_dir, rc):
    why = []
    if rc != 0:
        why.append(f"exit {rc}")
    sl = os.path.join(run_dir, "star_Log.out")
    lg = os.path.join(run_dir, "run.log.ts")
    s = open(sl, errors="replace").read() if os.path.exists(sl) else ""
    g = open(lg, errors="replace").read() if os.path.exists(lg) else ""
    if "E11: Solo on the device" not in s:
        why.append("level 5 NO")
    if "[gpu_pileup] on:" not in g:
        why.append("pileup NO")
    bad, _ = fallback.scan(run_dir)
    why += [f"{src}: {l[:140]}" for src, l in bad[:2]]
    return why


def main(res, scr, now=None):
    now = time.time() if now is None else now
    seen_path = os.path.join(res, ".watched")
    seen = set(open(seen_path).read().split()) if os.path.exists(seen_path) else set()
    runs = os.path.join(res, "runs.tsv")
    if not os.path.exists(runs):
        return None
    new = []
    for l in open(runs):
        f = l.rstrip("\n").split("\t")
        if len(f) < 6 or f[0] == "item" or f[0] in seen:
            continue
        if now - float(f[4]) < GRACE:
            continue
        new.append((f[0], int(f[5])))
    for i, rc in new:
        seen.add(i)
        why = problems(os.path.join(res, f"c{i}"), rc)
        if why:
            reason = f"fallback in run c{i}: " + " | ".join(why)
            os.makedirs(scr, exist_ok=True)
            open(os.path.join(scr, "STOP"), "w").write(reason + "\n")
            open(os.path.join(res, "STOPPED"), "w").write(reason + "\n")
            break
    open(seen_path, "w").write(" ".join(sorted(seen)))
    return open(os.path.join(res, "STOPPED")).read().strip() if os.path.exists(os.path.join(res, "STOPPED")) else None


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    if r:
        print(r)
