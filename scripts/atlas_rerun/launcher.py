#!/usr/bin/env python3
"""Launcher (S9): keep fetch workers alive, start GPU/CPU drain workers when the stage has enough backlog.

    launcher.py --stage DIR --config config.json [--pipeline pipeline.json] [--once] [--dry-run]

It is itself a small Slurm job (1 core): sbatch works from compute nodes on Clipper (checked, job s9-slurmcheck), so no
login-node process and no scrontab are needed. Before it loops it queues its own successor with --dependency=afterany, like
the orchestrator's controller, so it survives the 4 h partition limit with no gap.

Worker pools (decision 2026-10-30): GPU drain workers run mainly on the preemptible `scavenger` partition, one H100 per job over all
3 H100 nodes (max_h100 = 6); the V100S nodes are the alternate GPU pool; CPU workers (same drain, --engine cpu) are the last resort.
Every pool is preemption-safe (drain_worker returns a claim on SIGTERM; a SIGKILLed one is reaped after its heartbeat expires), so
workers are fungible and nothing is pinned to a long scavenger wait.

Policy (numbers in pipeline.json, defaults below; derivation in docs/research-beta/download-and-staging.md §7):
  backlog_h = ready pairs / (H100 pairs/s) / 3600           H100-hours of work in ready/
  * H100 workers : want min(max_h100, floor(backlog_h / X)); a worker is only started when it would have >= X hours of work (X = gpu_start_h).
  * Scavenger waits: a pending H100 job older than fallback_wait_h counts as no capacity. Capacity = running workers + young pending.
                   The pairs the capacity cannot clear within overflow_h hours go first to V100S workers (only while the H100 pool is
                   starved, up to max_v100), then to CPU workers (up to max_cpu, short-cpu so they start). Fallback workers are not
                   cancelled when an H100 starts: they exit by themselves after idle_exit_min once the stage is drained.
  * fetch workers: fetch_jobs low-core jobs while todo/ is not empty; each pauses itself at the stage cap, and all share the global
                   rate cap in <stage>/stage.json (ratecap.py).
Job names: s9-fetch, s9-drain-h100, s9-drain-v100, s9-drain-cpu (only these are counted).
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
    "gpu_Mpairs_per_s": 10.0,       # one H100: S8's target (node GPU bound 8.7-11.2 M pairs/s); measured end to end today: ~4 per node, host-bound
    "v100_rel": 0.3,                # V100S speed relative to an H100 (ASSUMPTION, not measured; needs a V100S-capable CUDA build)
    "cpu_Mpairs_per_core_s": 0.025,  # CPU engine: 38.6-44.6 host core-s per M reads measured (docs/research-beta/gpu-cr4-clip.md)
    "gpu_start_h": 1.0,             # X: start an H100 worker only if it would have this many hours of work
    "max_h100": 6,                  # 3 H100 nodes x 2 GPUs; single GPU per job
    "max_v100": 4,
    "fallback_wait_h": 1.0,         # a pending H100 job older than this counts as no capacity
    "overflow_h": 3.0,              # pairs the capacity cannot clear in this many hours go to the next pool
    "cpu_cores": 16,
    "cpu_job_h": 3.9,
    "max_cpu": 8,
    "fetch_jobs": 6,                # 6 x 4 cores = 24 cores of decode: 2.6 core-s/M x 8.2 M pairs/s (0.6 GB/s cap of full .sra at 73 B/pair)
    "fetch_streams": 16,
    "fetch_cpus": 4,
    "idle_exit_min": 10,
    "stage_cap_tb": 2,              # a few hours of data: 3 h x 8.2 M pairs/s x ~30 B/pair = 0.9 TB, plus a drain hiccup
    "interval_s": 120,
    "scav_partition": "scavenger",
    "h100_gres": "gpu:nvidia_h100_nvl:1",
    "v100_gres": "gpu:tesla_v100s:1",
    "gpu_cpus": 16,
    "gpu_time": "03:59:00",
    "cpu_partition": "short-cpu",
    "fetch_partition": "short-cpu",
    "fetch_time": "03:59:00",
}


def sh(cmd, stdin=None):
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def count_jobs(user, name, now=None):
    """-> {'run': n, 'pend': n, 'ages': [seconds each pending job has waited]} for jobs called `name`."""
    rc, out = sh(["squeue", "-h", "-u", user, "-n", name, "-o", "%T|%V"])
    res = {"run": 0, "pend": 0, "ages": []}
    now = now or time.time()
    for line in out.splitlines() if rc == 0 else []:
        st, _, sub = line.partition("|")
        if st == "RUNNING":
            res["run"] += 1
        elif st in ("PENDING", "CONFIGURING"):
            res["pend"] += 1
            try:
                res["ages"].append(now - time.mktime(time.strptime(sub, "%Y-%m-%dT%H:%M:%S")))
            except ValueError:
                res["ages"].append(0)
    return res


def plan(p, bl, have):
    """Pure decision function. bl = stagequeue.backlog(); have['h100'|'v100'|'cpu'] = count_jobs(...); have['fetch'] = int.
    -> how many of each worker to submit."""
    h_rate = p["gpu_Mpairs_per_s"] * 1e6
    v_rate = h_rate * p["v100_rel"]
    c_rate = p["cpu_Mpairs_per_core_s"] * 1e6 * p["cpu_cores"]
    pairs, horizon = bl["pairs"], p["overflow_h"] * 3600
    backlog_h = pairs / h_rate / 3600
    out = {"backlog_h100_h": round(backlog_h, 2), "h100": 0, "v100": 0, "cpu": 0, "fetch": 0}
    h, v, c = have["h100"], have["v100"], have["cpu"]
    want = min(p["max_h100"], int(backlog_h // p["gpu_start_h"]))
    out["h100"] = max(0, want - h["run"] - h["pend"])
    young = sum(1 for a in h["ages"] if a < p["fallback_wait_h"] * 3600)
    old = h["pend"] - young
    cap_pairs = (h["run"] + young + out["h100"]) * h_rate * horizon
    excess = max(0.0, pairs - cap_pairs)
    if excess > 0 and old > 0:  # the scavenger pool is not delivering: alternate GPU pool
        n_v = min(p["max_v100"], math.ceil(excess / (v_rate * horizon)))
        out["v100"] = max(0, n_v - v["run"] - v["pend"])
        excess = max(0.0, excess - (v["run"] + v["pend"] + out["v100"]) * v_rate * horizon)
    else:
        excess = max(0.0, excess - v["run"] * v_rate * horizon)
    if excess > 0:
        n_c = min(p["max_cpu"], math.ceil(excess / (c_rate * p["cpu_job_h"] * 3600)))
        out["cpu"] = max(0, n_c - c["run"] - c["pend"])
    todo = bl["todo"] + bl["fetching"]
    if todo > 0:
        out["fetch"] = max(0, min(p["fetch_jobs"], todo) - have["fetch"])
    return out


def sbatch_script(kind, a, p):
    head = ["#!/bin/bash", "#SBATCH --account=debruinz", "#SBATCH --job-name=s9-%s" % kind,
            "#SBATCH --output=%s/s9-%%x-%%j.out" % a.log_dir, "#SBATCH --no-requeue"]
    here = os.path.dirname(os.path.abspath(__file__))
    py = "python3 -u"
    common = "--stage %s --config %s" % (a.stage, a.config)
    if kind == "fetch":
        hdr = ["#SBATCH --partition=%s" % p["fetch_partition"], "#SBATCH --cpus-per-task=%d" % p["fetch_cpus"], "#SBATCH --mem=16G",
               "#SBATCH --time=%s" % p["fetch_time"], "#SBATCH --signal=B:TERM@60"]
        body = ["%s %s/fetch_worker.py %s --cap-tb %s --streams %d --decode-threads %d --idle-exit-min 20 &" %
                (py, here, common, p["stage_cap_tb"], p["fetch_streams"], max(1, p["fetch_cpus"] // 2))]
    elif kind in ("drain-h100", "drain-v100"):
        gres = p["h100_gres"] if kind == "drain-h100" else p["v100_gres"]
        hdr = ["#SBATCH --partition=%s" % p["scav_partition"], "#SBATCH --gres=%s" % gres,
               "#SBATCH --cpus-per-task=%d" % p["gpu_cpus"], "#SBATCH --mem=128G", "#SBATCH --time=%s" % p["gpu_time"],
               "#SBATCH --signal=B:TERM@60"]
        body = ["%s %s/drain_worker.py %s --engine gpu --threads %d --idle-exit-min %s --scratch /dev/shm &" %
                (py, here, common, p["gpu_cpus"], p["idle_exit_min"])]
    else:
        hdr = ["#SBATCH --partition=%s" % p["cpu_partition"], "#SBATCH --cpus-per-task=%d" % p["cpu_cores"],
               "#SBATCH --mem=%dG" % (p["cpu_cores"] * 6), "#SBATCH --time=%02d:%02d:00" % (int(p["cpu_job_h"]), round(p["cpu_job_h"] % 1 * 60)),
               "#SBATCH --signal=B:TERM@60"]
        body = ["%s %s/drain_worker.py %s --engine cpu --threads %d --idle-exit-min %s &" % (py, here, common, p["cpu_cores"], p["idle_exit_min"])]
    # `--signal=B:TERM@60` signals the batch shell only; the worker runs in the background so the trap can forward SIGTERM to it
    # and wait for its clean-up (claim returned) before the 60 s grace ends.
    tail = ["pid=$!", "trap 'kill -TERM $pid 2>/dev/null; wait $pid' TERM", "wait $pid"]
    return "\n".join(head + hdr + ["set -uo pipefail"] + body + tail) + "\n"


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
            have = {"h100": count_jobs(user, "s9-drain-h100"), "v100": count_jobs(user, "s9-drain-v100"),
                    "cpu": count_jobs(user, "s9-drain-cpu")}
            f = count_jobs(user, "s9-fetch")
            have["fetch"] = f["run"] + f["pend"]
            d = plan(p, bl, have)
            print("[launcher] %s backlog=%s have=%s -> %s" % (time.strftime("%H:%M:%S"), bl, have, d), flush=True)
            for kind, key in (("fetch", "fetch"), ("drain-h100", "h100"), ("drain-v100", "v100"), ("drain-cpu", "cpu")):
                submit(kind, d[key], a, p, a.dry_run)
        if a.once or time.time() > end:
            return 0
        time.sleep(p["interval_s"])


if __name__ == "__main__":
    sys.exit(main())
