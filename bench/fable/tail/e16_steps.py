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
# the run's STAR child (the prefill's process) and the pipeline; other processes (the CR4 probe's two STAR runs, forked by the child) left out
main = next((pid for _, pid, s in marks if s.startswith("prefill: mapDistinct returned")), None)
par = {pid for _, pid, s in marks if s.startswith(("parent:", "collector:"))}
nOther = sum(1 for m in marks if m[1] != main and m[1] not in par)
if main is not None: marks = [m for m in marks if m[1] == main or m[1] in par]
child = {main} if main is not None else {pid for _, pid, s in marks if s.startswith(("STAR", "prefill", "after mapping"))}
if nOther: print(f"e16: {nOther} marks of other processes left out (the CR4 probe's STAR runs)")
print(f"e16 marks (t = s after the prefill's device release, epoch {t0:.3f}; dt = since the previous mark of the same process)")
last = {}
for t, pid, s in marks:
    who = "child " if pid in child else "parent"
    dt = t - last.get(pid, t); last[pid] = t
    print(f"  {t - t0:8.3f} {dt:7.3f} {who} {s[:300]}")
for e in extra: print("  log: " + e)
# the tail's steps: (name, from mark prefix, to mark prefix)
S = [("prefill tail (cache, summaries, region, Unmapped.out, read source)", "prefill: mapDistinct returned", "prefill: end"),
     ("STAR: prefill end -> prefill returned", "prefill: end", "STAR: prefill returned"),
     ("STAR: chunks constructed", "STAR: prefill returned", "STAR: "),
     ("STAR: chunks constructed -> threads joined (loop)", "STAR: ", "STAR: threads joined"),
     ("STAR: threads joined -> BAM stream closed", "STAR: threads joined", "STAR: BAM stream closed"),
     ("STAR: after-mapping steps (junctions, Solo on the device)", "after mapping: start", "STAR: after-mapping steps done"),
     ("STAR: cache + genome free", "STAR: after-mapping steps done", "STAR: genome freed"),
     ("STAR: SJ.out.tab", "STAR: genome freed", "STAR: SJ.out.tab written"),
     ("STAR: Solo constructor", "STAR: SJ.out.tab written", "STAR: Solo constructed"),
     ("STAR: Solo constructed -> files closed", "STAR: Solo constructed", "STAR: files closed"),
     ("STAR: files closed -> _exit", "STAR: files closed", "STAR child: _exit"),
     ("parent: BAM header read -> references resolved", "parent: BAM header read", "parent: references resolved"),
     ("parent: first BAM record -> stream drained", "parent: first BAM record", "parent: stream drained"),
     ("parent: BAM stream closed (child) -> drained", "STAR: BAM stream closed", "parent: stream drained"),
     ("parent: hand-off pass", "parent: stream drained", "parent: hand-off pass done"),
     ("  of which e2ePass", "parent: hand-off pass: device lock taken", "parent: hand-off pass: e2ePass returned"),
     ("  of which waiting for the early pass (join)", "parent: stream drained", "parent: hand-off pass starts"),
     ("  early pass: region mapped -> e2ePass returned", "parent: E16 early hand-off pass: region complete", "parent: hand-off pass: e2ePass returned"),
     ("parent: build input", "parent: hand-off pass done", "parent: pileupRun starts"),
     ("parent: pileupRun", "parent: pileupRun starts", "parent: pileupRun returned"),
     ("parent: results into the accumulators", "parent: pileupRun returned", "parent: pileup results put"),
     ("parent: host pass", "parent: pileup results put", "parent: host pass done"),
     ("parent: GPU pileup applied (end of the tail)", "parent: host pass done", "parent: GPU pileup applied"),
     ("parent: host pass done -> rRNA detector joined (rrnabg)", "parent: host pass done", "parent: rRNA detector joined"),
     ("parent: finalize (collapse)", "parent: GPU pileup applied", "parent: finalize done"),
     ("parent: STAR reaped", "parent: finalize done", "parent: STAR reaped")]
print("e16 steps (s)")
import re
def firstx(pfx):
    if pfx == "STAR: ": return next((t for t, _, s in marks if re.match(r"STAR: \d+ chunks constructed", s)), None)
    return first(pfx)
for name, a, b in S:
    ta, tb = firstx(a), firstx(b)
    if ta is None or tb is None: print(f"  {name:62s}       -"); continue
    print(f"  {name:62s} {tb - ta:7.3f}   ({ta - t0:7.3f} -> {tb - t0:7.3f})")
lp = next((s for _, _, s in marks if s.startswith("STAR: loop:")), None)
m = re.search(r"threads start ([\d.]+), first chunk ([\d.]+)\.\.([\d.]+), last chunk mapped ([\d.]+), last thread end ([\d.]+)", lp or "")
if m:
    ts, f0, f1, lm, le = (float(x) for x in m.groups())
    print(f"  {'  loop: threads start -> last chunk mapped':62s} {lm - ts:7.3f}   ({ts - t0:7.3f} -> {lm - t0:7.3f})")
    print(f"  {'  loop: last chunk mapped -> last thread end':62s} {le - lm:7.3f}   ({lm - t0:7.3f} -> {le - t0:7.3f})")
te = first("parent: GPU pileup applied")
if te: print(f"  {'TAIL: prefill device release -> GPU pileup applied':62s} {te - t0:7.3f}")
tf = first("parent: finalize done")
if tf: print(f"  {'TAIL + collapse: prefill device release -> finalize done':62s} {tf - t0:7.3f}")
