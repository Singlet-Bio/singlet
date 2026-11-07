#!/usr/bin/env python3
"""Measure the fetch-tier steps for ONE run (S9): fetch (full .sra, shared rate cap) -> singlet download --quality binned (.1fq) -> singlet-stage (.1fg2).

    stage_one.py RUN --stage-bin PATH --singlet PATH --whitelist-dir D --scratch DIR --cap-stage DIR [--verify]

Prints one JSON line: bytes of the .sra, the .1fq and the .1fg2, CPU seconds of each step (children rusage), the singlet-stage report. Everything is deleted.
"""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_worker as F  # noqa: E402
import ratecap  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stage-bin", required=True)
    ap.add_argument("--singlet", required=True)
    ap.add_argument("--whitelist-dir", required=True)
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--cap-stage", required=True, help="directory holding stage.json (global rate cap) and rate/ heartbeats")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--streams", type=int, default=8)
    a = ap.parse_args()
    d = os.path.join(a.scratch, "s9-stageone-%s-%d" % (a.run, os.getpid()))
    os.makedirs(d)
    out = {"run": a.run}
    try:
        lim = ratecap.SharedRate(a.cap_stage, "stageone-%s-%d" % (a.run, os.getpid()))
        lim.set_active(True)
        t = time.time()
        sra, info = F.fetch_run(a.run, d, a.streams, lambda m: None, lim)
        lim.set_active(False)
        out.update(sra_bytes=info["bytes"], fetch_s=round(time.time() - t, 1))
        fq = os.path.join(d, a.run + ".1fq")
        env = dict(os.environ, SINGLET_WHITELIST_DIR=a.whitelist_dir)
        c0, t = cpu(), time.time()
        rc = subprocess.run([a.singlet, "download", sra, "--quality", "binned", "--vdb-threads", "2", "--no-profile", "-o", fq], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        out.update(decode_rc=rc, decode_cpu_s=round(cpu() - c0, 1), decode_wall_s=round(time.time() - t, 1))
        os.remove(sra)
        if rc != 0:
            out["skipped"] = "decode gate rc=%d" % rc
            print(json.dumps(out), flush=True)
            return 0
        out["fq_bytes"] = os.path.getsize(fq)
        g = os.path.join(d, a.run + ".1fg2")
        cmd = [a.stage_bin, "--whitelist-dir", a.whitelist_dir, "--accessions", a.run, "--scratch", d, "-o", g] + (["--verify"] if a.verify else []) + [fq]
        c0, t = cpu(), time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        out.update(stage_rc=p.returncode, stage_wall_s=round(time.time() - t, 1), stage_cpu_s_incl_verify=round(cpu() - c0, 1))
        if p.returncode in (0, 7) and p.stdout.strip():
            rep = json.loads(p.stdout.strip().splitlines()[-1])
            out["stage"] = rep
            out["g_bytes"] = os.path.getsize(g)
            out["verify"] = ("ok" if p.returncode == 0 else "MISMATCH") if a.verify else "not run"
        else:
            out["stage_err"] = p.stderr[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
