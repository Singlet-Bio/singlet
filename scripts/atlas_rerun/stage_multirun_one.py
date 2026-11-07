#!/usr/bin/env python3
"""A multi-run sample through the two fetch-tier paths from the same local .sra files (S9); prints one JSON line.

    stage_multirun_one.py EXPERIMENT RUN1,RUN2[,RUN3] WORKDIR BINDIR WHITELIST_DIR STAGE_BIN

  B two-step : singlet download RUN -> RUN.1fq (each), then singlet-stage RUN1.1fq RUN2.1fq ... -> b.1fg2   (one histogram, one table, pooled)
  C parts    : singlet download RUN --stage-out part_i --stage-part (each, no .1fq), then singlet-stage --merge-parts -> c.1fg2
Same order-independent digest required; CPU of each step (children rusage); bytes; part sizes.
"""
import json
import os
import resource
import subprocess
import sys
import time


def run(cmd):
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return p, round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 1), round(time.time() - t, 1)


def rm(*paths):
    for f in paths:
        try:
            os.remove(f)
        except OSError:
            pass


EXP, RUNS, B, BIN, WL, SSB = sys.argv[1:7]
runs = RUNS.split(",")
sing = BIN + "/singlet-cpu"
out = {"experiment": EXP, "runs": runs}
fqs, A_cpu = [], 0.0
for r in runs:
    p, c, w = run([sing, "download", "%s/%s.sra" % (B, r), "--quality", "binned", "--vdb-threads", "2", "--no-profile", "--whitelist-dir", WL, "-o", "%s/%s.1fq" % (B, r)])
    A_cpu += c
    if p.returncode != 0:
        out["skipped"] = "%s decode rc=%d" % (r, p.returncode)
        print(json.dumps(out), flush=True)
        sys.exit(0)
    fqs.append("%s/%s.1fq" % (B, r))
out.update(A_cpu=round(A_cpu, 1), fq_bytes=sum(os.path.getsize(f) for f in fqs))
p, c, w = run([SSB, "--whitelist-dir", WL, "--accessions", RUNS, "--scratch", B, "-o", B + "/b.1fg2"] + fqs)
out.update(B_stage_rc=p.returncode, B_stage_cpu=c, B_err=(p.stderr[-300:] if p.returncode else ""))
if p.returncode == 0:
    out["B_report"] = json.loads(p.stdout.strip().splitlines()[-1])
    out["B_bytes"] = os.path.getsize(B + "/b.1fg2")
rm(*fqs)
parts, C1 = [], 0.0
for i, r in enumerate(runs):
    part = "%s/part%d" % (B, i)
    p, c, w = run([sing, "download", "%s/%s.sra" % (B, r), "--quality", "binned", "--vdb-threads", "2", "--no-profile", "--whitelist-dir", WL,
                   "--stage-out", part, "--stage-part", "-o", B + "/unused.1fq"])
    C1 += c
    if p.returncode != 0:
        out["C_err"] = "%s part rc=%d %s" % (r, p.returncode, p.stderr[-200:])
        break
    parts.append(part)
if len(parts) == len(runs):
    out["part_bytes"] = sum(os.path.getsize(f) for f in parts)
    p, c, w = run([SSB, "--whitelist-dir", WL, "--scratch", B, "-o", B + "/c.1fg2", "--merge-parts"] + parts)
    out.update(C_parts_cpu=round(C1, 1), C_merge_cpu=c, C_total_cpu=round(C1 + c, 1), C_merge_rc=p.returncode, C_merge_err=(p.stderr[-300:] if p.returncode else ""))
    if p.returncode == 0:
        out["C_report"] = json.loads(p.stdout.strip().splitlines()[-1])
        out["C_bytes"] = os.path.getsize(B + "/c.1fg2")
        if os.path.exists(B + "/b.1fg2"):
            da = subprocess.run([SSB, "--digest", B + "/b.1fg2"], capture_output=True, text=True).stdout.strip()
            dc = subprocess.run([SSB, "--digest", B + "/c.1fg2"], capture_output=True, text=True).stdout.strip()
            out.update(digest_B=da, digest_C=dc, digest_equal=bool(da) and da == dc)
rm(B + "/b.1fg2", B + "/c.1fg2", B + "/unused.1fq", *parts)
print(json.dumps(out), flush=True)
