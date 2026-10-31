#!/usr/bin/env python3
"""Launcher (S9): keep fetch workers alive, start GPU/CPU drain workers when the stage has enough backlog.

    launcher.py --stage DIR --config config.json [--pipeline pipeline.json] [--once] [--dry-run]

It is itself a small Slurm job (1 core): sbatch works from compute nodes on Clipper (checked, job s9-slurmcheck), so no
login-node process and no scrontab are needed. Before it loops it queues its own successor with --dependency=afterany, like
the orchestrator's controller, so it survives the 4 h partition limit with no gap.

Policy (all numbers in pipeline.json, defaults below; derivations in docs/research-beta/download-and-staging.md §7):
  backlog_gpu_h  = ready pairs / gpu_Mpairs_per_s / 3600          the GPU-hours of work sitting in ready/
  * GPU worker  : submitted when backlog_gpu_h >= gpu_start_h (X) and fewer than max_gpu workers are running or pending. A
                  worker drains until the stage is empty for idle_exit_min, so a start is a *burst* of about
                  max(X, backlog) GPU-hours, never an idle wait for data.
  * CPU workers : the part of the backlog the GPU workers (running + pending) would not finish within cpu_overflow_h hours is
                  spread over short-cpu jobs of cpu_cores cores, up to max_cpu.
  * fetch workers: a fixed count (fetch_jobs) of low-core jobs while todo/ is not empty; each pauses itself at the stage cap.
Everything submitted carries the job names s9-fetch / s9-drain-gpu / s9-drain-cpu; the launcher only counts its own names.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stagequeue as Q  # noqa: E402

DEFAULTS = {
    "gpu_Mpairs_per_s": 10.0,       # S8 target for one H100 (node GPU bound 8.7-11.2 M pairs/s); measured single-sample today: ~0.17
    "cpu_Mpairs_per_core_s": 0.025, # CPU engine: 38.6-44.6 host core-s per M reads measured (docs/research-beta/gpu-cr4-clip.md)
    "gpu_start_h": 1.0,             # X: start a GPU worker when this many GPU-hours are waiting
    "max_gpu": 1,                   # one H100 at a time (single-GPU rule)
    "cpu_overflow_h": 3.0,          # backlog the GPUs cannot clear within this many hours goes to CPU workers
    "cpu_cores": 16,
    "cpu_job_h": 3.9,
    "max_cpu": 8,
    "fetch_jobs": 12,               # 12 x 4 cores = 48 cores of decode: 2.6 core-s/M x 18 M pairs/s (docs §3)
    "fetch_streams": 8,             # per job; 12 x 8 = 96 streams in total reaches the shared 0.9 GB/s
    "fetch_cpus": 4,
    "idle_exit_min": 10,
    "stage_cap_tb": 4,
    "interval_s": 120,
    "gpu_partition": "short-gpu",
    "gpu_gres": "gpu:nvidia_h100_nvl:1",
    "gpu_cpus": 16,
    "gpu_time": "03:59:00",
    "cpu_partition": "short-cpu",
    "fetch_partition": "short-cpu",
    "fetch_time": "03:59:00",
}


def sh(cmd, stdin=None):
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def count_jobs(user, name):
    rc, out = sh(["squeue", "-h", "-u", user, "-n", name, "-o", "%T"])
    states = out.split() if rc == 0 else []
    return len(states)


def plan(p, bl, have):
    """Pure decision function -> dict of how many of each worker to submit. bl = stagequeue.backlog(); have = running+pending."""
    gpu_rate = p["gpu_Mpairs_per_s"] * 1e6
    cpu_rate = p["cpu_Mpairs_per_core_s"] * 1e6 * p["cpu_cores"]  # per CPU worker, pairs/s
    pairs = bl["pairs"]
    backlog_h = pairs / gpu_rate / 3600
    out = {"backlog_gpu_h": round(backlog_h, 2), "gpu": 0, "cpu": 0, "fetch": 0}
    if backlog_h >= p["gpu_start_h"] and have["gpu"] < p["max_gpu"]:
        out["gpu"] = p["max_gpu"] - have["gpu"]
    gpu_cap_pairs = (have["gpu"] + out["gpu"]) * gpu_rate * p["cpu_overflow_h"] * 3600
    excess = max(0.0, pairs - gpu_cap_pairs)
    want_cpu = math.ceil(excess / (cpu_rate * p["cpu_job_h"] * 3600)) if excess > 0 else 0
    out["cpu"] = max(0, min(want_cpu, p["max_cpu"]) - have["cpu"])
    if bl["todo"] + bl["fetching"] > 0:
        out["fetch"] = max(0, min(p["fetch_jobs"], bl["todo"] + bl["fetching"]) - have["fetch"])
    return out


def sbatch_script(kind, a, p):
    base = ["#!/bin/bash", "#SBATCH --account=debruinz", "#SBATCH --job-name=s9-%s" % kind,
            "#SBATCH --output=%s/s9-%%x-%%j.out" % a.log_dir, "set -uo pipefail"]
    here = os.path.dirname(os.path.abspath(__file__))
    py = "python3 -u"
    common = "--stage %s --config %s" % (a.stage, a.config)
    if kind == "fetch":
        hdr = ["#SBATCH --partition=%s" % p["fetch_partition"], "#SBATCH --cpus-per-task=%d" % p["fetch_cpus"], "#SBATCH --mem=16G",
               "#SBATCH --time=%s" % p["fetch_time"]]
        body = ["%s %s/fetch_worker.py %s --cap-tb %s --streams %d --decode-threads %d --idle-exit-min 20" %
                (py, here, common, p["stage_cap_tb"], p["fetch_streams"], max(1, p["fetch_cpus"] // 2))]
    elif kind == "drain-gpu":
        hdr = ["#SBATCH --partition=%s" % p["gpu_partition"], "#SBATCH --gres=%s" % p["gpu_gres"],
               "#SBATCH --cpus-per-task=%d" % p["gpu_cpus"], "#SBATCH --mem=128G", "#SBATCH --time=%s" % p["gpu_time"]]
        body = ["%s %s/drain_worker.py %s --engine gpu --threads %d --idle-exit-min %s" % (py, here, common, p["gpu_cpus"], p["idle_exit_min"])]
    else:
        hdr = ["#SBATCH --partition=%s" % p["cpu_partition"], "#SBATCH --cpus-per-task=%d" % p["cpu_cores"],
               "#SBATCH --mem=%dG" % (p["cpu_cores"] * 6), "#SBATCH --time=%02d:%02d:00" % (int(p["cpu_job_h"]), round(p["cpu_job_h"] % 1 * 60))]
        body = ["%s %s/drain_worker.py %s --engine cpu --threads %d --idle-exit-min %s" % (py, here, common, p["cpu_cores"], p["idle_exit_min"])]
    return "\n".join(base[:2] + hdr + base[2:] + body) + "\n"


def submit(kind, n, a, p, dry):
    ids = []
    for _ in range(n):
        if dry:
            print("[dry-run] would sbatch s9-%s" % kind)
            continue
        rc, out = sh(["sbatch", "--parsable"], stdin=sbatch_script(kind, a, p))
        print("[launcher] submitted s9-%s: %s" % (kind, out))
        ids.append(out)
    return ids


def requeue_self(a):
    """Queue the next launcher job to start when this one ends (afterany), so the loop spans the 4 h limit."""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid or a.no_successor:
        return
    script = "\n".join(["#!/bin/bash", "#SBATCH --account=debruinz", "#SBATCH --job-name=s9-launcher", "#SBATCH --partition=short-cpu",
                        "#SBATCH --cpus-per-task=1", "#SBATCH --mem=2G", "#SBATCH --time=03:55:00", "#SBATCH --dependency=afterany:%s" % jid,
                        "#SBATCH --output=%s/s9-launcher-%%j.out" % a.log_dir,
                        "python3 -u %s %s" % (os.path.abspath(__file__), " ".join(sys.argv[1:]))]) + "\n"
    print("[launcher] successor:", sh(["sbatch", "--parsable"], stdin=script)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pipeline", help="JSON overriding DEFAULTS")
    ap.add_argument("--log-dir", default=os.path.expandvars("$WORK/logs") if os.environ.get("WORK") else "/tmp")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-successor", action="store_true")
    a = ap.parse_args()
    p = dict(DEFAULTS)
    if a.pipeline:
        p.update(json.load(open(a.pipeline)))
    user = os.environ.get("USER", "debruinz")
    Q.init(a.stage)
    if not a.once:
        requeue_self(a)
    end = time.time() + 3.8 * 3600
    while True:
        Q.reap(a.stage, log=print)
        if os.path.exists(os.path.join(a.stage, "PAUSE")):
            print("[launcher] PAUSE flag: nothing submitted")
        else:
            bl = Q.backlog(a.stage)
            have = {k: count_jobs(user, "s9-" + n) for k, n in (("gpu", "drain-gpu"), ("cpu", "drain-cpu"), ("fetch", "fetch"))}
            d = plan(p, bl, have)
            print("[launcher] %s backlog=%s have=%s -> %s" % (time.strftime("%H:%M:%S"), bl, have, d), flush=True)
            for kind, key in (("fetch", "fetch"), ("drain-gpu", "gpu"), ("drain-cpu", "cpu")):
                submit(kind, d[key], a, p, a.dry_run)
        if a.once or time.time() > end:
            return 0
        time.sleep(p["interval_s"])


if __name__ == "__main__":
    sys.exit(main())
