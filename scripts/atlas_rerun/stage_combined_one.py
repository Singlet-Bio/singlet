#!/usr/bin/env python3
"""One run through the three fetch-tier paths from the same local .sra (S9); prints one JSON line.

    stage_combined_one.py RUN WORKDIR BINDIR WHITELIST_DIR STAGE_BIN

  A baseline  singlet download --quality binned -o a.1fq                       (today: SRA -> barcode-sorted .1fq)
  B two-step  singlet-stage a.1fq -> b.1fg2                                    (input barcode-sorted)
  C combined  singlet download --stage-out c.1fg2                              (SRA order, no .1fq, no sort)
CPU seconds are children rusage deltas. B and C must have the same order-independent digest (singlet-stage --digest).
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


R, B, BIN, WL, SSB = sys.argv[1:6]
sra = "%s/%s.sra" % (B, R)
out = {"run": R, "sra_bytes": os.path.getsize(sra)}
sing = BIN + "/singlet-cpu"
p, c, w = run([sing, "download", sra, "--quality", "binned", "--vdb-threads", "2", "--no-profile", "--whitelist-dir", WL, "-o", B + "/a.1fq"])
out.update(A_rc=p.returncode, A_cpu=c, A_wall=w)
if p.returncode == 0:
    out["fq_bytes"] = os.path.getsize(B + "/a.1fq")
    p, c, w = run([SSB, "--whitelist-dir", WL, "--accessions", R, "--scratch", B, "-o", B + "/b.1fg2", B + "/a.1fq"])
    out.update(B_stage_rc=p.returncode, B_stage_cpu=c, B_stage_wall=w, B_err=(p.stderr[-300:] if p.returncode else ''))
    if p.returncode == 0:
        out["B_report"] = json.loads(p.stdout.strip().splitlines()[-1])
        out["B_bytes"] = os.path.getsize(B + "/b.1fg2")
    rm(B + "/a.1fq")
p, c, w = run([sing, "download", sra, "--quality", "binned", "--vdb-threads", "2", "--no-profile", "--stage-out", B + "/c.1fg2",
               "--whitelist-dir", WL, "-o", B + "/unused.1fq"])
out.update(C_rc=p.returncode, C_cpu=c, C_wall=w, C_err=(p.stderr[-300:] if p.returncode else ""))
if p.returncode == 0 and os.path.exists(B + "/c.1fg2"):
    out["C_bytes"] = os.path.getsize(B + "/c.1fg2")
    try:
        out["C_report"] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        pass
    if os.path.exists(B + "/b.1fg2"):
        da = subprocess.run([SSB, "--digest", B + "/b.1fg2"], capture_output=True, text=True).stdout.strip()
        dc = subprocess.run([SSB, "--digest", B + "/c.1fg2"], capture_output=True, text=True).stdout.strip()
        out.update(digest_B=da, digest_C=dc, digest_equal=bool(da) and da == dc)
rm(B + "/b.1fg2", B + "/c.1fg2", B + "/unused.1fq")
print(json.dumps(out), flush=True)
