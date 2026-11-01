#!/usr/bin/env python3
"""Drain worker (S9): take staged samples off the queue, run singlet, upload, delete, repeat; exit when idle.

    drain_worker.py --stage DIR --engine gpu|cpu --config config.json [--cluster clipper] [--threads 8]
                    [--idle-exit-min 10] [--publish dry-run|none|CMD ...] [--max-samples N] [--max-pairs N]

One worker = one Slurm job (one H100 for --engine gpu, N cores for --engine cpu). GPU and CPU workers claim from the same
queue (stagequeue.claim_ready); a CPU worker can be told to skip samples above --max-pairs, a GPU worker to skip below --min-pairs.

Per sample: claim -> heartbeat thread -> run `singlet` on the staged .1fq (zstd-decoded on the CPU inside the reader; no nvCOMP)
with --out-prefix in node-local /tmp -> checksum the outputs -> publish hook -> delete outputs and staged reads -> event.
The reference/index stays resident for GPU workers because the process lives for the whole job: the binary is started once per
sample today (index load is part of its start-up), so a `--session` mode of the engine would remove that per-sample cost; the
worker measures and records `startup_s` so the cost is visible (see docs/research-beta/download-and-staging.md §6).

Nothing here writes to Cloudflare unless --publish is given a command; the default `dry-run` only records the sizes/hashes that
would be uploaded. Genotype-bearing `private/` outputs never leave the node: they are deleted with the rest.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stagequeue as Q  # noqa: E402


def log(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


class Heart(threading.Thread):
    """Heartbeats the claim every HB_STALE_S/5 and notices when the claim has been taken away (`lost`): a worker that was presumed
    dead (e.g. frozen node) must not publish or delete anything when it wakes up, the sample belongs to whoever reaped it."""

    def __init__(self, path, proc_holder):
        super().__init__(daemon=True)
        self.path, self.stop, self.lost, self.proc = path, threading.Event(), False, proc_holder

    def run(self):
        while not self.stop.wait(max(5, Q.HB_STALE_S / 5)):
            if not os.path.isdir(self.path):
                self.lost = True
                p = self.proc.get("p")
                if p and p.poll() is None:
                    p.kill()
                return
            Q.heartbeat(self.path)


def singlet_cmd(a, ccfg, cfg, claim, man, out_tmp):
    binary = ccfg.get("binary") or cfg["binary"]["path"]
    fq = [os.path.join(claim, r + ".1fq") for r in man["runs"]]
    screen = os.path.join(claim, "screen.json")
    if man.get("format") == "1fg2":
        reads = [os.path.join(claim, man["files"][0])]   # one pooled .1fg2; the engine reads it by magic (S8's reader)
    else:
        reads = [fq[0]] if len(fq) == 1 and not os.path.exists(screen) else ["--runs-dir", claim]
    cmd = ["/usr/bin/time", "-v", "-o", os.path.join(out_tmp, "time.txt"), binary] + reads
    cmd += ["--ref-base", ccfg["ref_base"]]
    ref = C.resolve_reference(man.get("organism"), cfg.get("references", {}))
    if ref is None:   # never species auto-detection: it needs Bloom filters that are not installed (6 of 12 pilot runs came back 'unknown')
        raise C.OrganismUnmapped("organism %r maps to no reference in config `references`" % (man.get("organism"),))
    d = os.path.join(ccfg["ref_base"], ref)
    cmd += ["--genome-dir", os.path.join(d, "star_2.7.11b"), "--exons", os.path.join(d, "genes", "genes.gtf")]
    cmd += ["--out-prefix", os.path.join(out_tmp, "out") + "/", "--threads", str(a.threads), "--pipeline", "--output-format", "1pz",
            "--engine", a.engine]
    if man.get("pairs"):
        cmd += ["--memory-tier", C.memory_tier(man["pairs"])]   # the .s1 holds summary.json: its memory_tier comes from the sample, not from this process
    if os.path.exists(screen):
        cmd += ["--run-screen", screen]
    return cmd + list(cfg.get("pipeline", {}).get("extra_args", [])) + a.extra


def checksum_outputs(out_dir):
    """-> (public_bytes, n_files, private_bytes). Private (genotype) files are counted, never listed for upload."""
    pub = priv = n = 0
    for r, _, names in os.walk(out_dir):
        for x in names:
            s = os.path.getsize(os.path.join(r, x))
            if "/private" in r.replace("\\", "/") or x.endswith(".vcf") or x.endswith(".vcf.gz"):
                priv += s
            else:
                pub += s
                n += 1
    return pub, n, priv


def publish_s1(a, cfg, ccfg, man, out_tmp, scr, claim, log_path, dry):
    """Per-sample Beta publish (sample_publish.py): collect the public outputs exactly as the orchestrator does (private/ split off and
    dropped with the scratch), build the `.s1` objects with s1_producer.py, upload, extend the study manifest."""
    import types

    import sample_publish as SP
    import stages
    ctx = types.SimpleNamespace(out_root=os.path.join(scr, "oroot"), private_root=os.path.join(scr, "priv"))
    gse = man.get("gse") or "NA"
    final, _priv, _sha, _man = stages.collect_outputs(ctx, gse, man["gsm"], out_tmp, claim, log_path)
    res = SP.publish_sample(cfg, ccfg, gse, man["gsm"], final, os.path.join(scr, "pub"), dry_run=dry, s1_repo=a.s1_repo,
                            runtime=a.s1_runtime, log=log)
    if res["status"] not in ("published", "dry_run_ok"):
        raise RuntimeError("beta publish: %s" % json.dumps(res)[:800])
    return {"mode": "s1-dry" if dry else "s1", **{k: res.get(k) for k in ("status", "objects", "rev", "rev_was", "study_samples", "bytes_sent",
                                                                          "uploaded", "present", "upload_s", "d1_rows_written")}}


def publish(a, man, out_dir, cfg=None, ccfg=None, scr=None, claim=None):
    """The upload hook. dry-run: size and hash of what a publish would send. s1 / s1-dry: per-sample Beta publish. Any other
    value is a command run with {gsm} {gse} {out}."""
    if a.publish in (["s1"], ["s1-dry"]):
        return publish_s1(a, cfg, ccfg, man, out_dir, scr, claim, os.path.join(scr, "singlet.log"), a.publish == ["s1-dry"])
    if a.publish == ["none"]:
        return {"mode": "none"}
    if a.publish == ["dry-run"]:
        pub, n, priv = checksum_outputs(out_dir)
        return {"mode": "dry-run", "files": n, "public_bytes": pub, "private_bytes_not_uploaded": priv}
    cmd = [x.format(gsm=man["gsm"], gse=man.get("gse") or "", out=out_dir) for x in a.publish]
    t0 = time.time()
    for i in range(3):
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            return {"mode": "cmd", "rc": 0, "upload_s": round(time.time() - t0, 1)}
        time.sleep(30 * (i + 1))
    raise RuntimeError("publish failed rc=%s" % rc)


class Job:
    """One claimed sample between the end of its singlet run and the end of its publish."""

    def __init__(self, gsm, claim, man, scr, heart, proc, t0, compute_s):
        self.gsm, self.claim, self.man, self.scr = gsm, claim, man, scr
        self.heart, self.proc, self.t0, self.compute_s = heart, proc, t0, compute_s


def compute_phase(a, cfg, ccfg, env, gsm, claim, man):
    """Claim -> singlet run. -> (message, Job|None). The GPU is busy only here; the publish of the previous sample runs meanwhile."""
    root = ccfg["root"]
    scr = os.path.join(a.scratch, "s9-drain-%s-%d" % (gsm, os.getpid()))
    shutil.rmtree(scr, ignore_errors=True)
    os.makedirs(os.path.join(scr, "out"))
    proc = {}
    heart = Heart(claim, proc)
    heart.start()
    t0 = time.time()
    C.emit(root, "running", gsm, attempt=man.get("attempts", 1), stage="compute", engine=a.engine, via="drain")
    keep = False
    try:
        cmd = singlet_cmd(a, ccfg, cfg, claim, man, scr)
        with open(os.path.join(scr, "singlet.log"), "w") as fh:
            proc["p"] = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=scr)
            rc = proc["p"].wait()
        compute_s = time.time() - t0
        if heart.lost or not os.path.isdir(claim):
            return "claim lost (reaped by another worker): nothing published, nothing deleted", None
        if rc != 0:
            tail = open(os.path.join(scr, "singlet.log")).read()[-1500:]
            C.emit(root, "failed", gsm, attempt=man.get("attempts", 1), stage="compute", reason="rc_%d" % rc, detail=tail)
            Q.release(a.stage, gsm, claim, failed=man.get("attempts", 1) >= Q.MAX_ATTEMPTS)
            return "failed", None
        keep = True
        return "computed %.0fs %.3f Mpairs/s" % (compute_s, man.get("pairs", 0) / 1e6 / max(compute_s, 1e-9)), \
            Job(gsm, claim, man, scr, heart, proc, t0, compute_s)
    except C.OrganismUnmapped as e:
        C.emit(root, "failed", gsm, attempt=man.get("attempts", 1), stage="compute", reason="organism_unmapped", detail=str(e), final=True, via="drain")
        man["failed_reason"] = "organism_unmapped: %s" % e
        with open(os.path.join(claim, "manifest.json"), "w") as f:
            json.dump(man, f)
        Q.release(a.stage, gsm, claim, failed=True)   # a retry would fail the same way
        return "failed (final): organism unmapped", None
    except Q.Preempted:
        # Slurm preempted this job or hit the time limit: stop the child, give the sample back with the attempt refunded.
        heart.stop.set()
        p = proc.get("p")
        if p and p.poll() is None:
            p.kill()
        if os.path.isdir(claim):
            Q.release(a.stage, gsm, claim, refund=True)
        C.emit(root, "requeued", gsm, attempt=man.get("attempts", 1), stage="compute", reason="preempted", via="drain")
        raise
    except Exception as e:  # noqa: BLE001
        C.emit(root, "failed", gsm, attempt=man.get("attempts", 1), stage="compute", reason="exception", detail=repr(e))
        if os.path.isdir(claim):
            Q.release(a.stage, gsm, claim, failed=man.get("attempts", 1) >= Q.MAX_ATTEMPTS)
        return "failed: %r" % e, None
    finally:
        if not keep:
            heart.stop.set()
            shutil.rmtree(scr, ignore_errors=True)


def finish_phase(a, cfg, ccfg, job):
    """Publish (per-sample Beta upload), then delete the staged reads and the scratch. Runs in a thread beside the next compute."""
    root, gsm, claim, man, scr, heart = ccfg["root"], job.gsm, job.claim, job.man, job.scr, job.heart
    try:
        out_dir = os.path.join(scr, "out")
        up = publish(a, man, out_dir, cfg, ccfg, scr, claim)
        pub, n, priv = checksum_outputs(out_dir)
        if heart.lost or not os.path.isdir(claim):
            return "claim lost during publish: staged reads left to the new owner"
        heart.stop.set()
        Q.finish(a.stage, gsm, claim)  # staged reads deleted only now: outputs are uploaded (or hooked) and checked
        C.emit(root, "done", gsm, attempt=man.get("attempts", 1), stage="compute", engine=a.engine, compute_s=round(job.compute_s, 1),
               wall_s=round(time.time() - job.t0, 1), pairs=man.get("pairs"), staged_bytes=man.get("bytes"), out_public_bytes=pub,
               publish=up, via="drain")
        return "done (publish %.0fs after compute)" % (time.time() - job.t0 - job.compute_s)
    except Exception as e:  # noqa: BLE001
        C.emit(root, "failed", gsm, attempt=man.get("attempts", 1), stage="compute", reason="exception", detail=repr(e)[:1500])
        if os.path.isdir(claim):
            Q.release(a.stage, gsm, claim, failed=man.get("attempts", 1) >= Q.MAX_ATTEMPTS)
        return "failed: %r" % e
    finally:
        heart.stop.set()
        shutil.rmtree(scr, ignore_errors=True)


def accept_sample(m, min_pairs=0, max_pairs=0, mem_gb=0, accept_1fg2=False, engine="gpu"):
    """May a worker with these limits take this staged sample? A GPU worker never claims a sample whose estimated memory (common.gpu_job_mem_gb)
    exceeds its own allocation: it would be OOM-killed, and the sample would burn its attempts. Such a sample waits for a larger worker."""
    p = m.get("pairs", 0)
    if m.get("format") == "1fg2" and not accept_1fg2:
        return False   # the engine's .1fg2 reader is S8's; until its identity gate passes, such samples wait for a worker started with --accept-1fg2
    if mem_gb and engine == "gpu" and mem_gb < C.GPU_EXCLUSIVE_GB and C.gpu_job_mem_gb(p) > mem_gb:
        return False   # a whole-node worker (mem_gb >= GPU_EXCLUSIVE_GB) takes anything: it is the largest there is
    return (not min_pairs or p >= min_pairs) and (not max_pairs or p <= max_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", choices=("gpu", "cpu"), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cluster", default="clipper")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--idle-exit-min", type=float, default=10.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=-1, help="memory of this job in GB (default: the Slurm allocation, 0 = unknown: no memory filter); "
                    "a GPU worker skips samples whose estimate (common.gpu_job_mem_gb) is larger")
    ap.add_argument("--publish", nargs="+", default=["dry-run"], help="dry-run | none | s1-dry | s1 | a command with {gsm} {gse} {out}")
    ap.add_argument("--s1-repo", help="built copy of the s1 code (compiled singlet._pz/_pz2) for s1_producer.py, e.g. $WORK/jobs/s6/s1repo")
    ap.add_argument("--s1-runtime", help="in_runtime.sh to source: the producer then runs inside the runtime image + venv (Clipper)")
    ap.add_argument("--accept-1fg2", action="store_true", help="also claim samples staged as .1fg2 (engine with the .1fg2 reader); .1fq samples are always accepted")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    ccfg = C.cluster_cfg(cfg, a.cluster)
    C.require_whitelists(ccfg.get("binary") or cfg["binary"]["path"])
    env = dict(os.environ)
    if ccfg.get("whitelist_dir"):
        env["SINGLET_WHITELIST_DIR"] = ccfg["whitelist_dir"]
    env["SINGLET_REF_BASE"] = ccfg["ref_base"]
    Q.init(a.stage)
    Q.sweep_scratch(a.scratch, "s9-drain-", log)
    Q.install_sigterm()

    mem_gb = a.mem_gb if a.mem_gb >= 0 else (C.node_mem_gb() if os.environ.get("SLURM_MEM_PER_NODE") else 0)
    if mem_gb:
        log("[drain] memory %.0f GB: samples above %d M pairs are left for larger workers" % (
            mem_gb, max([p for p in range(10_000_000, 2_000_000_000, 10_000_000) if C.gpu_job_mem_gb(p) <= mem_gb] or [0]) // 1_000_000))

    def accept(m):
        return accept_sample(m, a.min_pairs, a.max_pairs, mem_gb, a.accept_1fg2, a.engine)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inflight = []  # [(future, job)]: at most one publish runs beside the next sample's compute

    def settle(timeout=None):
        for fut, job in inflight:
            log("[drain] %s %s: %s" % (a.engine, job.gsm, fut.result(timeout=timeout)))
        inflight.clear()

    idle_since, n = time.time(), 0
    try:
        while True:
            Q.reap(a.stage, log=log)
            c = Q.claim_ready(a.stage, accept=accept)
            if c:
                idle_since = time.time()
                gsm, claim, man = c
                man["gsm"] = gsm
                msg, job = compute_phase(a, cfg, ccfg, env, gsm, claim, man)
                log("[drain] %s %s: %s" % (a.engine, gsm, msg))
                settle()  # the previous publish (it ran while this sample computed)
                if job:
                    inflight.append((pool.submit(finish_phase, a, cfg, ccfg, job), job))
                n += 1
                if a.max_samples and n >= a.max_samples:
                    break
                continue
            settle()
            if time.time() - idle_since > a.idle_exit_min * 60:
                log("[drain] idle %.0f min: exit after %d samples" % (a.idle_exit_min, n))
                break
            time.sleep(15)
        settle()
        return 0
    except Q.Preempted:
        # let a publish that is nearly done finish (it is idempotent anyway); otherwise give its sample back too
        for fut, job in inflight:
            try:
                log("[drain] %s %s: %s" % (a.engine, job.gsm, fut.result(timeout=40)))
            except concurrent.futures.TimeoutError:
                job.heart.stop.set()
                if os.path.isdir(job.claim):
                    Q.release(a.stage, job.gsm, job.claim, refund=True)
                C.emit(ccfg["root"], "requeued", job.gsm, attempt=job.man.get("attempts", 1), stage="compute", reason="preempted_in_publish", via="drain")
        raise
    finally:
        pool.shutdown(wait=False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Q.Preempted:
        log("[drain] SIGTERM: claim returned to ready/, exiting")
        sys.exit(143)
