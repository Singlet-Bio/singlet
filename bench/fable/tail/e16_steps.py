#!/usr/bin/env python3
"""E16 (docs/gpu-engine/fable/E16-tail.md): the level-5 tail as a per-step table, from one gate run dir (run.log.ts).
Usage: e16_steps.py RUN_DIR
Reads the "[e16] <epoch> <pid> <text>" marks that STAR's child and the pipeline write at SINGLET_ENGINE_GPU_E2E=5 (include/singlet/tail_trace.h),
prints every mark (seconds after the prefill's device release, the step since the previous mark of the same process), then the tail's steps
(prefill end -> GPU pileup applied, the 34 s of E11's gate 796433) and the timers that the device passes append to their logs."""
import sys, re
d = sys.argv[1]
marks, extra = [], []
for l in open(f"{d}/run.log.ts", errors="replace"):
    i = l.find("[e16] ")
    if i >= 0:
        p = l[i + 6:].rstrip("\n").split(" ", 2)
        try: marks.append((float(p[0]), int(p[1]), p[2] if len(p) > 2 else ""))
        except ValueError: pass
        continue
    for pat in ("e16 steps (s):", "e16: context", "e16: merge"):
        j = l.find(pat)
        if j >= 0: extra.append(l[j:].rstrip("\n")[:600])
if not marks:
    print("e16: no marks in run.log.ts (not level 5, or SINGLET_E16_TRACE=0)"); sys.exit(0)
def first(pfx):
    return next((t for t, _, s in marks if s.startswith(pfx)), None)
t0 = first("prefill: mapDistinct returned") or marks[0][0]
child = {pid for _, pid, s in marks if s.startswith(("STAR", "prefill", "after mapping"))}
print(f"e16 marks (t = s after the prefill's device release, epoch {t0:.3f}; dt = since the previous mark of the same process)")
last = {}
for t, pid, s in marks:
    who = "child " if pid in child else "parent"
    dt = t - last.get(pid, t); last[pid] = t
    print(f"  {t - t0:8.3f} {dt:7.3f} {who} {s[:300]}")
for e in extra: print("  log: " + e)
# the tail's steps: (name, from mark prefix, to mark prefix)
S = [("prefill tail (cache, summaries, region, Unmapped.out, read source)", "prefill: mapDistinct returned", "prefill: end"),
     ("STAR: prefill end -> threads start", "prefill: end", "STAR: loop:"),
     ("STAR: loop (threads joined)", "prefill: end", "STAR: threads joined"),
     ("STAR: threads joined -> BAM stream closed", "STAR: threads joined", "STAR: BAM stream closed"),
     ("STAR: after-mapping steps (junctions, Solo on the device)", "after mapping: start", "STAR: after-mapping steps done"),
     ("STAR: cache + genome free", "STAR: after-mapping steps done", "STAR: genome freed"),
     ("STAR: SJ.out.tab", "STAR: genome freed", "STAR: SJ.out.tab written"),
     ("STAR: Solo constructor", "STAR: SJ.out.tab written", "STAR: Solo constructed"),
     ("STAR: Solo constructed -> files closed", "STAR: Solo constructed", "STAR: files closed"),
     ("STAR: files closed -> _exit", "STAR: files closed", "STAR child: _exit"),
     ("parent: first BAM record -> stream drained", "parent: first BAM record", "parent: stream drained"),
     ("parent: BAM stream closed (child) -> drained", "STAR: BAM stream closed", "parent: stream drained"),
     ("parent: hand-off pass", "parent: stream drained", "parent: hand-off pass done"),
     ("  of which e2ePass", "parent: hand-off pass: device lock taken", "parent: hand-off pass: e2ePass returned"),
     ("parent: build input", "parent: hand-off pass done", "parent: pileupRun starts"),
     ("parent: pileupRun", "parent: pileupRun starts", "parent: pileupRun returned"),
     ("parent: results into the accumulators", "parent: pileupRun returned", "parent: pileup results put"),
     ("parent: host pass", "parent: pileup results put", "parent: host pass done"),
     ("parent: GPU pileup applied (end of the tail)", "parent: host pass done", "parent: GPU pileup applied"),
     ("parent: finalize (collapse)", "parent: GPU pileup applied", "parent: finalize done"),
     ("parent: STAR reaped", "parent: finalize done", "parent: STAR reaped")]
print("e16 steps (s)")
for name, a, b in S:
    ta, tb = first(a), first(b)
    if ta is None or tb is None: print(f"  {name:62s}       -"); continue
    print(f"  {name:62s} {tb - ta:7.3f}   ({ta - t0:7.3f} -> {tb - t0:7.3f})")
te = first("parent: GPU pileup applied")
if te: print(f"  {'TAIL: prefill device release -> GPU pileup applied':62s} {te - t0:7.3f}")
tf = first("parent: finalize done")
if tf: print(f"  {'TAIL + collapse: prefill device release -> finalize done':62s} {tf - t0:7.3f}")
