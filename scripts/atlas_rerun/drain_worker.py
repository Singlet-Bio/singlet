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
    reads = [fq[0]] if len(fq) == 1 and not os.path.exists(screen) else ["--runs-dir", claim]
    cmd = ["/usr/bin/time", "-v", "-o", os.path.join(out_tmp, "time.txt"), binary] + reads
    cmd += ["--ref-base", ccfg["ref_base"]]
    ref = cfg.get("references", {}).get(man.get("organism") or "", cfg.get("references", {}).get("*", "auto"))
    if ref and ref != "auto":
        d = os.path.join(ccfg["ref_base"], ref)
        cmd += ["--genome-dir", os.path.join(d, "star_2.7.11b"), "--exons", os.path.join(d, "genes", "genes.gtf")]
    cmd += ["--out-prefix", os.path.join(out_tmp, "out") + "/", "--threads", str(a.threads), "--pipeline", "--output-format", "1pz",
            "--engine", a.engine]
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


def publish(a, man, out_dir):
    """The upload hook. dry-run: size and hash of what a publish would send. A command is run with {gsm} {gse} {out}."""
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


def drain_one(a, cfg, ccfg, env, gsm, claim, man):
    root = ccfg["root"]
    scr = os.path.join(a.scratch, "s9-drain-%s-%d" % (gsm, os.getpid()))
    shutil.rmtree(scr, ignore_errors=True)
    os.makedirs(os.path.join(scr, "out"))
    proc = {}
    heart = Heart(claim, proc)
    heart.start()
    t0 = time.time()
    C.emit(root, "running", gsm, attempt=man.get("attempts", 1), stage="compute", engine=a.engine, via="drain")
    try:
        cmd = singlet_cmd(a, ccfg, cfg, claim, man, scr)
        with open(os.path.join(scr, "singlet.log"), "w") as fh:
            proc["p"] = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=scr)
            rc = proc["p"].wait()
        compute_s = time.time() - t0
        if heart.lost or not os.path.isdir(claim):
            return "claim lost (reaped by another worker): nothing published, nothing deleted"
        if rc != 0:
            tail = open(os.path.join(scr, "singlet.log")).read()[-1500:]
            C.emit(root, "failed", gsm, attempt=man.get("attempts", 1), stage="compute", reason="rc_%d" % rc, detail=tail)
            Q.release(a.stage, gsm, claim, failed=man.get("attempts", 1) >= Q.MAX_ATTEMPTS)
            return "failed"
        out_dir = os.path.join(scr, "out")
        up = publish(a, man, out_dir)
        pub, n, priv = checksum_outputs(out_dir)
        if heart.lost or not os.path.isdir(claim):
            return "claim lost during publish: staged reads left to the new owner"
        heart.stop.set()
        Q.finish(a.stage, gsm, claim)  # staged reads deleted only now: outputs are uploaded (or hooked) and checked
        C.emit(root, "done", gsm, attempt=man.get("attempts", 1), stage="compute", engine=a.engine, compute_s=round(compute_s, 1),
               wall_s=round(time.time() - t0, 1), pairs=man.get("pairs"), staged_bytes=man.get("bytes"), out_public_bytes=pub,
               publish=up, via="drain")
        return "done %.0fs %.3f Mpairs/s" % (compute_s, man.get("pairs", 0) / 1e6 / max(compute_s, 1e-9))
    except Exception as e:  # noqa: BLE001
        heart.stop.set()
        C.emit(root, "failed", gsm, attempt=man.get("attempts", 1), stage="compute", reason="exception", detail=repr(e))
        if os.path.isdir(claim):
            Q.release(a.stage, gsm, claim, failed=man.get("attempts", 1) >= Q.MAX_ATTEMPTS)
        return "failed: %r" % e
    finally:
        heart.stop.set()
        shutil.rmtree(scr, ignore_errors=True)


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
    ap.add_argument("--publish", nargs="+", default=["dry-run"], help="dry-run | none | a command with {gsm} {gse} {out}")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    ccfg = C.cluster_cfg(cfg, a.cluster)
    env = dict(os.environ)
    if ccfg.get("whitelist_dir"):
        env["SINGLET_WHITELIST_DIR"] = ccfg["whitelist_dir"]
    env["SINGLET_REF_BASE"] = ccfg["ref_base"]
    Q.init(a.stage)
    Q.sweep_scratch(a.scratch, "s9-drain-", log)

    def accept(m):
        p = m.get("pairs", 0)
        return (not a.min_pairs or p >= a.min_pairs) and (not a.max_pairs or p <= a.max_pairs)

    idle_since, n = time.time(), 0
    while True:
        Q.reap(a.stage, log=log)
        c = Q.claim_ready(a.stage, accept=accept)
        if c:
            idle_since = time.time()
            gsm, claim, man = c
            man["gsm"] = gsm
            log("[drain] %s %s: %s" % (a.engine, gsm, drain_one(a, cfg, ccfg, env, gsm, claim, man)))
            n += 1
            if a.max_samples and n >= a.max_samples:
                return 0
            continue
        if time.time() - idle_since > a.idle_exit_min * 60:
            log("[drain] idle %.0f min: exit after %d samples" % (a.idle_exit_min, n))
            return 0
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
