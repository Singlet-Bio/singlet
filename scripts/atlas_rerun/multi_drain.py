#!/usr/bin/env python3
"""Multi-sample GPU drain worker (S14): one Slurm job = one H100 + ~28 cores + ~370 GB, running N samples at once from the stage queue.

    multi_drain.py --stage DIR --config config.json [--cluster clipper] [--concurrency 8] [--threads 12] [--scratch /tmp]
                   [--mem-gb 370] [--mem-frac 0.85] [--route fq|1fg2] [--publish dry-run|none|s1-dry|s1|CMD ...]

Why: one sample per H100 job reaches ~0.27 M pairs/s per GPU; ~8 samples sharing one GPU reach ~1.0 (S8). The account holds 3 H100s at once.

Admission (Budget): a claimed sample is started only while
  * the genome held once in shared memory (--genome-shared on, 17.7 GB) + the sum of the running samples' private estimates (common.gpu_private_gb: largest
    process + overhead, minus the shared genome) + this one <= mem_frac (0.85) of the job's cgroup memory limit (anon + shmem; page cache is reclaimable), and
  * the sum of the estimated node-local scratch bytes + this one fits in the free space of --scratch minus a margin, and
  * fewer than --concurrency samples run.
A sample whose own estimate cannot fit an empty worker is never claimed: it waits for the whole-node exclusive class (drain_worker.py,
--exclusive --mem=0), which takes anything. A smaller sample may overtake an older one that does not fit yet, but only for HOL_WAIT_S;
after that nothing newer is admitted until the older one fits (no starvation of big samples).

Claims, heartbeats, publish and cleanup are those of drain_worker.py (atomic rename claims, Heart, publish hooks, Q.finish).
Failure: a sample that fails or is killed is requeued ONCE (attempt 1 -> ready/), then moved to failed/ with its reason in manifest.json
(`failed_reason`) and a `failed` event with final=true. An OOM-killed attempt is requeued with a memory floor (1.5x the estimate) and
runs alone on the retry. SIGTERM (preemption, time limit) kills the children and returns every claim with the attempt refunded.
The measured peak RSS (/usr/bin/time -v) and scratch bytes of every sample go into its `done` event: the data for refitting common.MEM_MODEL.
The engine process starts per sample (index load is part of its start-up; the page cache keeps the shared index file hot between samples).
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import drain_worker as D  # noqa: E402
import stagequeue as Q  # noqa: E402

HOL_WAIT_S = 1200        # an older sample that does not fit yet may be overtaken by newer ones for this long
MEM_FRAC = 0.85
TMP_MARGIN_GB = 20.0
OOM_RETRY_FACTOR = 1.5
OOM_RCS = (-9, 137)


def log(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _read_bytes_limit(path):
    try:
        v = open(path).read().strip()
    except OSError:
        return None
    if v in ("", "max"):
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n < (1 << 60) else None


def cgroup_limit_gb(root="/sys/fs/cgroup", proc_cgroup="/proc/self/cgroup"):
    """The job's memory limit in GB from its cgroup (v2 memory.max, or v1 memory.limit_in_bytes), the smallest along the path to the root;
    None when no limit is visible."""
    rel = ""
    try:
        for line in open(proc_cgroup):
            parts = line.strip().split(":", 2)
            if len(parts) == 3 and (parts[0] == "0" or "memory" in parts[1].split(",")):
                rel = parts[2]
                break
    except OSError:
        pass
    best = None
    for sub in ("", "memory"):
        d = os.path.join(root, sub, rel.lstrip("/")) if sub or rel else os.path.join(root, sub)
        while True:
            for f in ("memory.max", "memory.limit_in_bytes"):
                v = _read_bytes_limit(os.path.join(d, f))
                if v is not None and (best is None or v < best):
                    best = v
            if len(d) <= len(os.path.join(root, sub)) or not d.startswith(root):
                break
            d = os.path.dirname(d)
    return best / 1e9 if best else None


class Budget:
    """Memory/scratch/slot accounting of the running samples. Pure bookkeeping (no I/O): the free-scratch figure is passed in."""

    def __init__(self, mem_gb, tmp_gb, slots, mem_frac=MEM_FRAC, tmp_margin_gb=TMP_MARGIN_GB, shared_gb=0.0):
        self.cap_mem, self.cap_tmp, self.slots = mem_gb * mem_frac, max(0.0, tmp_gb - tmp_margin_gb), slots
        self.shared = shared_gb   # the genome in shared memory: held once while any sample runs; samples are charged their private part
        self.run = {}    # gsm -> (private mem_gb, tmp_gb)

    @property
    def mem_used(self):
        return sum(m for m, _ in self.run.values()) + (self.shared if self.run else 0.0)

    @property
    def tmp_used(self):
        return sum(t for _, t in self.run.values())

    def ever_fits(self, mem):
        """Could this sample run on an empty worker? If not it belongs to the whole-node class."""
        return self.shared + mem <= self.cap_mem

    def fits(self, mem, tmp, solo=False):
        if solo:
            return not self.run and self.ever_fits(mem) and tmp <= self.cap_tmp
        need = mem + (0.0 if self.run else self.shared)
        return (len(self.run) < self.slots and self.mem_used + need <= self.cap_mem and self.tmp_used + tmp <= self.cap_tmp)

    def add(self, gsm, mem, tmp):
        self.run[gsm] = (mem, tmp)

    def drop(self, gsm):
        self.run.pop(gsm, None)


def estimate(man, route, shared=False):
    """-> (private_mem_gb, scratch_gb) of a staged sample: what it adds to a job already running samples (common.gpu_private_gb; with `shared` the genome
    held once in shared memory is not charged). A retry after an OOM kill carries a `mem_floor_gb`."""
    r = man.get("pairs", 0)
    rt = "1fg2" if man.get("format") == "1fg2" and route == "auto" else ("fq" if route == "auto" else route)
    return max(C.gpu_private_gb(r, rt, shared), man.get("mem_floor_gb", 0.0)), C.gpu_tmp_gb(r, rt)


class Admitter:
    """claim_ready's `accept` predicate for one claim attempt: static size filters, then the dynamic budget, with head-of-line protection."""

    def __init__(self, budget, route, free_tmp_gb, hol, min_pairs=0, max_pairs=0, accept_1fg2=False):
        self.b, self.route, self.free, self.hol = budget, route, free_tmp_gb, hol
        self.min_pairs, self.max_pairs, self.accept_1fg2 = min_pairs, max_pairs, accept_1fg2
        self.seen_head = False  # the first static-acceptable sample seen is the oldest one this worker could ever run
        self.blocked = False   # the head does not fit now

    def __call__(self, man):
        p = man.get("pairs", 0)
        if man.get("format") == "1fg2" and not self.accept_1fg2:
            return False
        if (self.min_pairs and p < self.min_pairs) or (self.max_pairs and p > self.max_pairs):
            return False
        mem, tmp = estimate(man, self.route, bool(self.b.shared))
        if not self.b.ever_fits(mem) or tmp > self.b.cap_tmp:
            return False       # too big for a shared worker: the whole-node class takes it
        solo = bool(man.get("solo"))
        ok = self.b.fits(mem, tmp, solo) and tmp <= self.free - self.b.tmp_used
        if not self.seen_head:
            self.seen_head = True
            if not ok:
                self.blocked = True
                self.hol.setdefault("since", time.time())
            else:
                self.hol.pop("since", None)
            return ok
        if self.blocked and time.time() - self.hol.get("since", time.time()) > HOL_WAIT_S:
            return False       # the head has waited long enough: drain until it fits
        return ok


def parse_time_v(path):
    """-> peak RSS in GB from `/usr/bin/time -v` output, or None."""
    try:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", open(path).read())
        return int(m.group(1)) / 1e6 if m else None
    except OSError:
        return None


def dir_bytes(path):
    n = 0
    for r, _, names in os.walk(path):
        for x in names:
            try:
                n += os.path.getsize(os.path.join(r, x))
            except OSError:
                pass
    return n


def settle_failure(a, root, gsm, claim, man, reason, detail="", oom_est=None):
    """Requeue the claim once; on the last attempt move it to failed/ with the reason. Never silent: an event is written either way."""
    attempts = man.get("attempts", 1)
    final = attempts >= a.max_attempts
    try:
        if os.path.isdir(claim):
            man = json.load(open(os.path.join(claim, "manifest.json")))
            if final:
                man["failed_reason"] = reason
                man["failed_detail"] = detail[-1500:]
            else:
                man["solo"] = True        # the retry runs alone: a second failure then is the sample's own
                if oom_est:
                    man["mem_floor_gb"] = round(oom_est * OOM_RETRY_FACTOR, 1)
            with open(os.path.join(claim, "manifest.json"), "w") as f:
                json.dump(man, f)
            Q.release(a.stage, gsm, claim, failed=final)
    finally:
        C.emit(root, "failed" if final else "requeued", gsm, attempt=attempts, stage="compute", reason=reason, detail=detail[-1500:],
               final=final or None, via="multi")
    return "failed (final): %s" % reason if final else "requeued after %s" % reason


class Run(threading.Thread):
    """One sample, claim to cleanup: singlet -> publish -> Q.finish. Never raises; its result string is logged by the main loop."""

    def __init__(self, a, cfg, ccfg, env, gsm, claim, man, mem_est, stopping):
        super().__init__(daemon=True)
        self.a, self.cfg, self.ccfg, self.env = a, cfg, ccfg, env
        self.gsm, self.claim, self.man, self.mem_est, self.stopping = gsm, claim, man, mem_est, stopping
        self.proc, self.result, self.t0 = {}, None, time.time()
        self.heart = D.Heart(claim, self.proc)

    def run(self):
        a, root, gsm, claim, man = self.a, self.ccfg["root"], self.gsm, self.claim, self.man
        scr = os.path.join(a.scratch, "s9-drain-%s-%d" % (gsm, os.getpid()))
        shutil.rmtree(scr, ignore_errors=True)
        os.makedirs(os.path.join(scr, "out"))
        keep_claim = False
        self.heart.start()
        try:
            C.emit(root, "running", gsm, attempt=man.get("attempts", 1), stage="compute", engine="gpu", via="multi")
            cmd = D.singlet_cmd(a, self.ccfg, self.cfg, claim, man, scr)
            with open(os.path.join(scr, "singlet.log"), "w") as fh:
                self.proc["p"] = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=self.env, cwd=scr)
                rc = self.proc["p"].wait()
            compute_s = time.time() - self.t0
            if self.stopping.is_set():
                keep_claim = True        # the main loop returns it with the attempt refunded
                self.result = "stopped"
                return
            if self.heart.lost or not os.path.isdir(claim):
                self.result = "claim lost (reaped by another worker): nothing published, nothing deleted"
                return
            peak = parse_time_v(os.path.join(scr, "time.txt"))
            if rc != 0:
                tail = open(os.path.join(scr, "singlet.log"), errors="replace").read()[-1500:]
                oom = rc in OOM_RCS or "bad_alloc" in tail or "Cannot allocate memory" in tail
                self.result = settle_failure(a, root, gsm, claim, man, "oom" if oom else "rc_%d" % rc, tail, self.mem_est if oom else None)
                return
            tmp_b = dir_bytes(scr)
            up = D.publish(a, man, os.path.join(scr, "out"), self.cfg, self.ccfg, scr, claim)
            pub, _n, _priv = D.checksum_outputs(os.path.join(scr, "out"))
            if self.heart.lost or not os.path.isdir(claim):
                self.result = "claim lost during publish: staged reads left to the new owner"
                return
            self.heart.stop.set()
            Q.finish(a.stage, gsm, claim)
            pairs = man.get("pairs", 0)
            C.emit(root, "done", gsm, attempt=man.get("attempts", 1), stage="compute", engine="gpu", compute_s=round(compute_s, 1),
                   wall_s=round(time.time() - self.t0, 1), pairs=pairs, staged_bytes=man.get("bytes"), out_public_bytes=pub, publish=up,
                   peak_rss_gb=peak, est_peak_gb=round(self.mem_est, 1), scratch_bytes=tmp_b, via="multi", concurrent=man.get("_n_running"))
            self.result = "done %.0fs %.3f Mpairs/s own, peak %s GB (est %.0f), scratch %.1f GB" % (
                compute_s, pairs / 1e6 / max(compute_s, 1e-9), "%.1f" % peak if peak else "?", self.mem_est, tmp_b / 1e9)
        except Q.Preempted:
            keep_claim = True
            self.result = "stopped"
        except Exception as e:  # noqa: BLE001
            self.result = settle_failure(a, root, gsm, claim, man, "exception", repr(e))
        finally:
            self.heart.stop.set()
            shutil.rmtree(scr, ignore_errors=True)
            if not keep_claim and not self.result:
                self.result = "?"


def free_tmp_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cluster", default="clipper")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12, help="per sample (S8: 12 at 8 per GPU was best, 12 per GPU worse)")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--mem-gb", type=float, default=-1, help="job memory in GB when no cgroup limit is visible (default: Slurm allocation)")
    ap.add_argument("--mem-frac", type=float, default=MEM_FRAC)
    ap.add_argument("--route", choices=("fq", "1fg2", "auto"), default="fq", help="memory/scratch model: staged .1fq (temp FASTQ) or the .1fg2 table-fed route")
    ap.add_argument("--max-attempts", type=int, default=2, help="claims per sample before it is marked failed (default: requeued once)")
    ap.add_argument("--idle-exit-min", type=float, default=10.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--publish", nargs="+", default=["dry-run"])
    ap.add_argument("--s1-repo")
    ap.add_argument("--s1-runtime")
    ap.add_argument("--accept-1fg2", action="store_true")
    ap.add_argument("--extra-args", default="--genome-shared on", help="extra engine arguments as one string; the genome is attached from shared memory "
                    "(one resident copy per node, S8's benchmark setting) so concurrent samples do not each map the index")
    ap.add_argument("--engine-env", nargs="*", default=[], metavar="K=V", help="extra environment for every engine process (e.g. the shared device-lock permit)")
    a = ap.parse_args()
    a.engine = "gpu"
    a.extra = shlex.split(a.extra_args)
    cfg = C.load_config(a.config)
    ccfg = C.cluster_cfg(cfg, a.cluster)
    env = dict(os.environ)
    if ccfg.get("whitelist_dir"):
        env["SINGLET_WHITELIST_DIR"] = ccfg["whitelist_dir"]
    env["SINGLET_REF_BASE"] = ccfg["ref_base"]
    env.update(kv.split("=", 1) for kv in a.engine_env)
    os.makedirs(a.scratch, exist_ok=True)
    Q.init(a.stage)
    Q.sweep_scratch(a.scratch, "s9-drain-", log)
    Q.install_sigterm()

    mem = cgroup_limit_gb() or (a.mem_gb if a.mem_gb > 0 else C.node_mem_gb())
    if a.mem_gb > 0:
        mem = min(mem, a.mem_gb) if mem else a.mem_gb
    genome_shared = "--genome-shared" in a.extra and a.extra[a.extra.index("--genome-shared") + 1:][:1] == ["on"]
    budget = Budget(mem, free_tmp_gb(a.scratch), a.concurrency, a.mem_frac, shared_gb=C.GENOME_SHARED_GB if genome_shared else 0.0)
    log("[multi] memory limit %.0f GB -> admit up to %.0f GB estimated peak; scratch %s %.0f GB free -> %.0f GB; %d slots x %d threads; route %s; "
        "a sample above %.0f M pairs goes to the whole-node class" % (
            mem, budget.cap_mem, a.scratch, free_tmp_gb(a.scratch), budget.cap_tmp, a.concurrency, a.threads, a.route,
            max([p for p in range(10_000_000, 3_000_000_000, 10_000_000) if C.gpu_cgroup_gb(p, "fq" if a.route == "auto" else a.route) - (0 if genome_shared else 0) <= budget.cap_mem] or [0]) / 1e6))

    running, hol, stopping = {}, {}, threading.Event()
    n_done, idle_since, last_head_log = 0, time.time(), 0.0

    def reap_runs():
        nonlocal n_done
        for gsm, r in list(running.items()):
            if not r.is_alive():
                running.pop(gsm)
                budget.drop(gsm)
                n_done += 1
                log("[multi] %s: %s  (running %d, mem %.0f/%.0f GB)" % (gsm, r.result, len(running), budget.mem_used, budget.cap_mem))

    try:
        while True:
            Q.reap(a.stage, log=log)
            reap_runs()
            c = None
            if len(running) < a.concurrency and not (a.max_samples and n_done + len(running) >= a.max_samples):
                adm = Admitter(budget, a.route, free_tmp_gb(a.scratch), hol, a.min_pairs, a.max_pairs, a.accept_1fg2)

                def accept(m, adm=adm):
                    m.setdefault("gsm", m.get("gsm"))
                    return adm(m)
                c = Q.claim_ready(a.stage, accept=accept)
                if not c and adm.blocked and time.time() - last_head_log > 300:
                    last_head_log = time.time()
                    log("[multi] oldest ready sample does not fit yet (running %d, mem %.0f/%.0f GB, scratch %.0f/%.0f GB)" % (
                        len(running), budget.mem_used, budget.cap_mem, budget.tmp_used, budget.cap_tmp))
            if c:
                idle_since = time.time()
                gsm, claim, man = c
                man["gsm"] = gsm
                est_mem, est_tmp = estimate(man, a.route, bool(budget.shared))
                budget.add(gsm, est_mem, est_tmp)
                man["_n_running"] = len(running) + 1
                r = Run(a, cfg, ccfg, env, gsm, claim, man, est_mem, stopping)
                running[gsm] = r
                r.start()
                log("[multi] start %s: %.0f M pairs, est %.0f GB / %.0f GB scratch (running %d, mem %.0f/%.0f GB)" % (
                    gsm, man.get("pairs", 0) / 1e6, est_mem, est_tmp, len(running), budget.mem_used, budget.cap_mem))
                time.sleep(2)     # let the engine start before the next claim is judged
                continue
            if running:
                idle_since = time.time()
            elif a.max_samples and n_done >= a.max_samples:
                break
            elif time.time() - idle_since > a.idle_exit_min * 60:
                log("[multi] idle %.0f min: exit after %d samples" % (a.idle_exit_min, n_done))
                break
            time.sleep(5 if running else 15)
        return 0
    except Q.Preempted:
        stopping.set()
        for gsm, r in list(running.items()):
            p = r.proc.get("p")
            if p and p.poll() is None:
                p.kill()
        for gsm, r in list(running.items()):
            r.join(timeout=20)
            r.heart.stop.set()
            if os.path.isdir(r.claim):
                Q.release(a.stage, gsm, r.claim, refund=True)
            C.emit(ccfg["root"], "requeued", gsm, attempt=r.man.get("attempts", 1), stage="compute", reason="preempted", via="multi")
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Q.Preempted:
        log("[multi] SIGTERM: claims returned to ready/, exiting")
        sys.exit(143)
