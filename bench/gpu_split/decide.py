#!/usr/bin/env python3
"""Decider of the gpu/bold-split chain: the best non-default lane B step budget of the first timing job.
Usage: decide.py TP_RESULTS_DIR OUT_DIR ANALYZE_POINTS_PY
Candidates are the arms with cfg s512, s128 and sauto. The one with the highest progress-weighted rate wins.
Without a usable rate the decision is auto. Writes OUT_DIR/decision.env (one NAME=value line for the MPS arm) and OUT_DIR/decision.md."""
import os
import subprocess
import sys

res, outd, ap = sys.argv[1:4]
VAL = {"s512": "512", "s128": "128", "sauto": "auto"}
rates = {}
if os.path.isdir(res):
    r = subprocess.run([sys.executable, ap, res], capture_output=True, text=True)
    for l in r.stdout.splitlines():
        c = [x.strip() for x in l.split("|")]
        if len(c) > 10 and c[4] in ("base", *VAL):
            try:
                rates[c[4]] = float(c[9].strip("*"))
            except ValueError:
                pass
cand = {k: v for k, v in rates.items() if k in VAL}
if cand:
    best = max(cand, key=cand.get)
    why = "highest progress-weighted rate among the non-default STEPS_B arms"
else:
    best = "sauto"
    why = "no usable rate in the first timing job, so auto"
os.makedirs(outd, exist_ok=True)
open(os.path.join(outd, "decision.env"), "w").write(f"SINGLET_ENGINE_GPU_STEPS_B={VAL[best]}\n")
open(os.path.join(outd, "decision.md"), "w").write(
    f"MPS arm uses SINGLET_ENGINE_GPU_STEPS_B={VAL[best]} ({why}).\nRates (M pairs/s) by arm: {rates}\n")
print(open(os.path.join(outd, "decision.md")).read())
