#!/usr/bin/env python3
"""Is the low-quality mask of the GPU pileup active, and how wide is it?

Usage: maskstatus.py LABEL RUN_DIR [ENV_OBSERVED_FILE]
RUN_DIR holds run.log.ts and star_Log.final.out. The engine logs the mask only when it is used
("[gpu_pileup] per-base quality: low-quality masks for N reads, W words each").
When the line is absent the engine kept useQual = 0 (constant qualities, or qualities tied to N).
The read-length bound and SINGLET_HANDOFF_QWORDS are not logged. The optional file holds values of
SINGLET_HANDOFF_QWORDS that the job's watcher saw in /proc/<pid>/environ, one per line.
"""
import os
import re
import sys

label, d = sys.argv[1], sys.argv[2]
envf = sys.argv[3] if len(sys.argv) > 3 else None
log = open(os.path.join(d, "run.log.ts"), errors="replace").read() if os.path.exists(os.path.join(d, "run.log.ts")) else ""
fin = open(os.path.join(d, "star_Log.final.out"), errors="replace").read() if os.path.exists(os.path.join(d, "star_Log.final.out")) else ""
m = re.search(r"\[gpu_pileup\] per-base quality: low-quality masks for (\d+) reads, (\d+) words each( \(forced\))?", log)
pileup_on = "[gpu_pileup] on:" in log
unavail = "per-base quality needed but unavailable" in log
declined = re.search(r"SINGLET_GPU_PILEUP ignored: reads over the per-base-quality mask", log)
obs = sorted(set(open(envf).read().split())) if envf and os.path.exists(envf) else []
avg = re.search(r"Average input read length \|\s+(\d+)", fin)
if m:
    n, w = int(m.group(1)), int(m.group(2))
    use = f"useQual=1 ({n} reads, {w} words per read, mask covers {64 * w} bases{', forced' if m.group(3) else ''})"
    wide = "yes" if w >= 3 else "no (2 words, up to 128 bases)"
elif unavail:
    use, wide = "useQual=requested but unavailable (the masks were missing or too narrow)", "no"
elif declined:
    use, wide = "GPU pileup declined: reads over 512 bases", "no"
elif pileup_on:
    use, wide = "useQual=0 (no mask line: constant qualities or qualities tied to N)", "no (no mask is used)"
else:
    use, wide = "GPU pileup not on", "no"
print(f"MASK {label}: {use}; SINGLET_HANDOFF_QWORDS observed in /proc: {','.join(obs) if obs else 'not observed'} (not logged by the engine); "
      f"STAR average input read length, both mates: {avg.group(1) if avg else '?'}; >128-base mask exercised: {wide}")
