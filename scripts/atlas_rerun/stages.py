"""What runs inside the jobs: the I/O stage, the compute stage and the node packer.

Every stage writes its progress only as events (common.emit) and its files only under its own sample's directories:

    <stage_root>/<GSM>/            staged reads: per-run .1fq, screen.json, sample.1fq (deleted once the sample is done)
    <out_root>/<GSE>/<GSM>/        public outputs (no BAM, no reads) + MANIFEST.sha256 + screen.json + run.log
    <private_root>/<GSE>/<GSM>/    the pipeline's private/ dir (SNP allele counts): never uploaded publicly; mode 0700
    <root>/logs/<GSM>/             stage logs of every attempt

Exit codes: 0 ok; 1 failed (the controller retries once, then parks); 3 parked by the stage itself (nothing to run, e.g.
no run passed the screen); a non-zero I/O exit cancels the chained compute job (--kill-on-invalid-dep=yes).
"""

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from common import emit, job_slots, node_mem_gb, now, resolve_reference, sha256_file, staged_reads

HEAD_READS = 1000000
HEAD_TIMEOUT_S = 900


class StageError(Exception):
    def __init__(self, reason, detail=""):
        Exception.__init__(self, reason)
        self.reason = reason
        self.detail = detail[-2000:] if detail else ""


class Parked(Exception):
    def __init__(self, reason, detail=""):
        Exception.__init__(self, reason)
        self.reason = reason
        self.detail = detail


# ── environment of a stage ──────────────────────────────────────────────────────────────────────────────────────────


class Ctx:
    def __init__(self, cfg, ccfg):
        self.cfg, self.c = cfg, ccfg
        self.root = ccfg["root"]
        self.stage_root = ccfg.get("stage_root") or os.path.join(self.root, "stage")
        self.out_root = ccfg.get("out_root") or os.path.join(self.root, "out")
        self.private_root = ccfg.get("private_root") or os.path.join(self.root, "private")
        self.bin = ccfg.get("binary") or os.path.join(self.root, "bin", "singlet-" + cfg["binary"]["sha256"][:12])
        self.sha = cfg["binary"]["sha256"]
        self.ref_base = ccfg["ref_base"]
        self.scratch_base = os.path.expandvars(ccfg.get("scratch", "/tmp"))
        p = cfg.get("pipeline", {})
        self.nonhost = bool(ccfg.get("nonhost", p.get("nonhost", True)))
        self.extra = list(p.get("extra_args", []))
        # Legacy path, kept for one release: pool a multi-run sample into one sample.1fq (`singlet encode`) instead of
        # handing the engine the per-run .1fq files (PR #82's --runs-dir).
        self.pooled_input = bool(p.get("pooled_input", False))
        self.env = dict(os.environ)
        if ccfg.get("sra_tools"):
            self.env["PATH"] = ccfg["sra_tools"] + ":" + self.env.get("PATH", "")
        if ccfg.get("ncbi_home"):
            self.env["HOME"] = ccfg["ncbi_home"]  # sra-tools reads its config (and cache settings) under $HOME
        if ccfg.get("whitelist_dir"):
            self.env["SINGLET_WHITELIST_DIR"] = ccfg["whitelist_dir"]
        self.env["SINGLET_STALL_TIMEOUT"] = str(p.get("stall_timeout_s", 10800))
        if self.nonhost:
            self.env["SINGLET_REF_BASE"] = self.ref_base
        else:
            self.env.pop("SINGLET_REF_BASE", None)

    def reference(self, organism):
        name = resolve_reference(organism, self.cfg.get("references", {}))
        if name is None:
            return None
        d = os.path.join(self.ref_base, name)
        return {"name": name, "genome_dir": os.path.join(d, "star_2.7.11b"), "exons": os.path.join(d, "genes", "genes.gtf")}

    def scratch(self, tag):
        base = self.scratch_base
        os.makedirs(base, exist_ok=True)
        return tempfile.mkdtemp(prefix="ar-%s-%s-" % (tag, os.environ.get("SLURM_JOB_ID", "x")), dir=base)

    def log_dir(self, gsm):
        d = os.path.join(self.root, "logs", gsm)
        os.makedirs(d, exist_ok=True)
        return d


def run(cmd, log, env=None, cwd=None, timeout=None):
    """Run cmd, appending its stdout+stderr to `log`. Returns (rc, seconds, tail of output)."""
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write("\n$ %s\n" % " ".join(cmd))
        fh.flush()
        try:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        except OSError as e:
            fh.write("exec failed: %s\n" % e)
            rc = 127
        fh.write("\n[rc=%s, %.1f s]\n" % (rc, time.time() - t0))
    return rc, time.time() - t0, tail(log)


def tail(path, n=4000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def dir_bytes(path):
    total = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def verify_binary(ctx):
    if not os.path.exists(ctx.bin):
        raise StageError("binary_missing", ctx.bin)
    got = sha256_file(ctx.bin)
    if got != ctx.sha:
        raise StageError("binary_mismatch", "%s has %s, config pins %s" % (ctx.bin, got, ctx.sha))


def read_header_reads(ctx, onefq, log):
    """Reads and protocol from `singlet decode --header` (the .1fq footer); (None, None) if unreadable."""
    try:
        p = subprocess.run([ctx.bin, "decode", onefq, "--header"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, env=ctx.env, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    m = re.search(r"Reads: (\d+)", p.stdout)
    pr = re.search(r"Protocol: (\S+)", p.stdout)
    return (int(m.group(1)) if m else None), (pr.group(1) if pr else None)


# ── I/O stage: head screen -> prefetch of kept runs -> local decode -> authoritative screen -> pool ─────────────────


def io_stage(ctx, row, attempt):
    gsm, root = row["gsm"], ctx.root
    log = os.path.join(ctx.log_dir(gsm), "io.a%d.%s.log" % (attempt, os.environ.get("SLURM_JOB_ID", "x")))
    emit(root, "downloading", gsm, attempt=attempt, stage="io")
    t_start = time.time()
    scr = ctx.scratch("io")
    sdir = os.path.join(ctx.stage_root, gsm)
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir)
    try:
        verify_binary(ctx)
        runs = row["runs"]
        if not runs:
            raise Parked("no_runs_known", "ENA lists no run for this sample")
        planned = dict(zip(runs, row.get("run_reads") or [0] * len(runs)))
        # 1. screen the first 1M read pairs of every run, streamed (nothing of a dropped run is downloaded). A run
        #    whose remote head cannot be read (slow or flaky VDB streaming) is not dropped: it is prefetched and judged by
        #    the full-run screen in step 3 instead.
        hdir = os.path.join(scr, "head")
        os.makedirs(hdir)
        heads, pregate, no_head, t0 = [], {}, [], time.time()
        for r in runs:
            out = os.path.join(hdir, r + ".1fq")
            rc, _ = retry(lambda: run([ctx.bin, "download", r, "--max-reads", str(HEAD_READS), "--vdb-threads", "2",
                                       "--no-profile", "-o", out], log, env=ctx.env, cwd=scr, timeout=HEAD_TIMEOUT_S),
                          tries=2)
            if rc == 2:
                pregate[r] = "non10x_pregate"  # `singlet download`'s non-10x gate: a run of another chemistry
            elif rc != 0 or not os.path.exists(out):
                no_head.append(r)
            else:
                heads.append(out)
        head_s = time.time() - t0
        if not heads and not no_head:
            raise Parked("non10x", "every run rejected by the download gate: %s" % pregate)
        hs, kept, dropped = {}, [], sorted(pregate)
        hjson = os.path.join(scr, "head_screen.json")
        if heads:
            rc, _, out = run([ctx.bin, "screen-runs", "--json", hjson] + heads, log, env=ctx.env, cwd=scr)
            hs = load_json_safe(hjson)
            if rc == 6 and not no_head:
                keep_copy(hjson, os.path.join(ctx.log_dir(gsm), "head_screen.json"))
                raise Parked("no_gex_runs", "screen kept no run")
            if rc not in (0, 6) or hs is None:
                raise StageError("screen_failed", out)
            kept = list(hs.get("runs_kept") or [])
            dropped += list(hs.get("runs_dropped") or [])
        kept += no_head
        run_rows = [{"run": x.get("run_id"), "keep": bool(x.get("keep")), "reason": x.get("reason"),
                     "chemistry": x.get("chemistry"), "best_rate": x.get("best_rate"),
                     "planned_reads": planned.get(x.get("run_id"))} for x in (hs or {}).get("runs", [])]
        run_rows += [{"run": r, "keep": False, "reason": why, "planned_reads": planned.get(r)} for r, why in pregate.items()]
        run_rows += [{"run": r, "keep": True, "reason": "head_unavailable: judged on the full run",
                      "planned_reads": planned.get(r)} for r in no_head]
        emit(root, "screened", gsm, attempt=attempt, stage="io", runs_kept=kept, runs_dropped=dropped,
             chemistry=(hs or {}).get("majority_chemistry"), runs=run_rows, head_s=round(head_s, 1),
             head_unavailable=no_head or None)
        if heads:
            keep_copy(hjson, os.path.join(sdir, "head_screen.json"))
        # 2. prefetch each kept run to node-local scratch, decode it locally to the shared stage dir
        sra_total, prefetch_s, decode_s, per_run = 0, 0.0, 0.0, []
        for r in kept:
            sdir_sra = os.path.join(scr, "sra")
            rc, t_pf = retry(lambda: run(["prefetch", "--max-size", "u", "-O", sdir_sra, r], log, env=ctx.env, cwd=scr,
                                   timeout=6 * 3600), tries=3)
            if rc != 0:
                raise StageError("prefetch_failed", "%s rc=%s %s" % (r, rc, tail(log, 1500)))
            sra = find_sra(sdir_sra, r)
            if not sra:
                raise StageError("prefetch_no_file", r)
            b = os.path.getsize(sra)
            out = os.path.join(sdir, r + ".1fq")
            rc, t_dec, txt = run([ctx.bin, "download", sra, "--vdb-threads", str(ctx.c.get("io", {}).get("cpus", 1)),
                                  "--no-profile", "-o", out], log, env=ctx.env, cwd=scr)
            shutil.rmtree(os.path.join(sdir_sra, r), ignore_errors=True)
            if rc != 0:
                raise StageError("decode_failed", "%s rc=%s %s" % (r, rc, txt))
            reads, _ = read_header_reads(ctx, out, log)
            sra_total += b
            prefetch_s += t_pf
            decode_s += t_dec
            per_run.append({"run": r, "sra_bytes": b, "total_reads": reads, "prefetch_s": round(t_pf, 1),
                            "decode_s": round(t_dec, 1)})
        # 3. the authoritative screen, on the full local runs (its JSON is the sample's runs_used provenance)
        fjson = os.path.join(sdir, "screen.json")
        rc, screen_s, out = run([ctx.bin, "screen-runs", "--json", fjson] +
                                [os.path.join(sdir, r + ".1fq") for r in kept], log, env=ctx.env, cwd=scr)
        fs = load_json_safe(fjson)
        if rc == 6:
            raise Parked("no_gex_runs_full", "full-run screen kept no run")
        if rc != 0 or fs is None:
            raise StageError("screen_failed", out)
        final = list(fs.get("runs_kept") or [])
        for r in kept:
            if r not in final:
                os.remove(os.path.join(sdir, r + ".1fq"))
        # 4. one run: it is the sample's input. Several runs: the engine reads the per-run .1fq files as one stream
        #    (--runs-dir + screen.json, PR #82); `pipeline.pooled_input` keeps the old re-encoded pool.
        sample = os.path.join(sdir, "sample.1fq")
        pool_s = 0.0
        if len(final) == 1:
            os.replace(os.path.join(sdir, final[0] + ".1fq"), sample)
            reads_used, _ = read_header_reads(ctx, sample, log)
        elif ctx.pooled_input:
            pool_s = pool_runs(ctx, [os.path.join(sdir, r + ".1fq") for r in final], sample, scr, log)
            for r in final:
                os.remove(os.path.join(sdir, r + ".1fq"))
            reads_used, _ = read_header_reads(ctx, sample, log)
        else:
            reads_used = sum(read_header_reads(ctx, os.path.join(sdir, r + ".1fq"), log)[0] or 0 for r in final)
        emit(root, "downloaded", gsm, attempt=attempt, stage="io", runs_kept=final, sra_bytes=sra_total,
             staged_bytes=dir_bytes(sdir), prefetch_s=round(prefetch_s, 1), decode_s=round(decode_s, 1),
             screen_s=round(screen_s, 1), pool_s=round(pool_s, 1), io_s=round(time.time() - t_start, 1),
             reads_used=reads_used, runs=per_run, screen_changed=(sorted(final) != sorted(kept)) or None)
        return 0
    except Parked as p:
        shutil.rmtree(sdir, ignore_errors=True)
        emit(root, "parked", gsm, attempt=attempt, stage="io", reason=p.reason, detail=p.detail)
        return 3
    except StageError as e:
        shutil.rmtree(sdir, ignore_errors=True)
        emit(root, "failed", gsm, attempt=attempt, stage="io", reason=e.reason, detail=e.detail)
        return 1
    except Exception as e:  # noqa: BLE001 - anything else is a failure of this attempt, never a silent success
        shutil.rmtree(sdir, ignore_errors=True)
        emit(root, "failed", gsm, attempt=attempt, stage="io", reason="exception", detail=repr(e))
        return 1
    finally:
        shutil.rmtree(scr, ignore_errors=True)


def retry(fn, tries=3, wait=30):
    """Call fn() -> (rc, seconds, text) up to `tries` times until rc is 0 or 2 (2 = the download's non-10x gate, a
    verdict, not an error). Returns (rc, seconds of all tries)."""
    rc, total = None, 0.0
    for i in range(tries):
        rc, secs, _ = fn()
        total += secs
        if rc in (0, 2):
            return rc, total
        if i + 1 < tries:
            time.sleep(wait * (i + 1))
    return rc, total


def find_sra(d, r):
    for pat in ("%s/%s/%s.sra" % (d, r, r), "%s/%s/%s.sralite" % (d, r, r), "%s/%s.sra" % (d, r), "%s/%s/*.sra" % (d, r)):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def pool_runs(ctx, inputs, out, scr, log):
    """Several kept runs -> one .1fq: decode each to FASTQ on node-local disk, then one `singlet encode` with the
    options that match `singlet download`'s defaults (no dedup, no re-trim, no quality). Interim until the engine reads
    several runs itself (a multi-input `singlet download A.sra B.sra -o x.1fq`)."""
    t0 = time.time()
    r1, r2 = os.path.join(scr, "pool_R1.fq"), os.path.join(scr, "pool_R2.fq")
    protos = set()
    for f in inputs:
        _, proto = read_header_reads(ctx, f, log)
        protos.add(proto)
        t1, t2 = os.path.join(scr, "t_R1.fq"), os.path.join(scr, "t_R2.fq")
        rc, _, txt = run([ctx.bin, "decode", f, "--r1", t1, "--r2", t2, "--threads", "1"], log, env=ctx.env, cwd=scr)
        if rc != 0:
            raise StageError("pool_decode_failed", txt)
        for src, dst in ((t1, r1), (t2, r2)):
            with open(dst, "ab") as o, open(src, "rb") as i:
                shutil.copyfileobj(i, o, 16 << 20)
            os.remove(src)
    # no --protocol: the .1fq header names protocols differently from encode's tags (10xv3-5p vs 10x-5p-v3), and
    # encode's own detection on the pooled reads is the one `download` ran on each run
    cmd = [ctx.bin, "encode", "--reads", r1, r2, "-o", out, "--no-dedup", "--no-trim", "--quality", "none"]
    if len(protos) > 1:
        with open(log, "a") as fh:
            fh.write("[pool] WARNING: runs carry different protocols in their headers: %s\n" % sorted(map(str, protos)))
    rc, _, txt = run(cmd, log, env=ctx.env, cwd=scr)
    for f in (r1, r2):
        if os.path.exists(f):
            os.remove(f)
    if rc != 0:
        raise StageError("pool_encode_failed", txt)
    return time.time() - t0


def load_json_safe(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def keep_copy(src, dst):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    except OSError:
        pass


# ── compute stage: the pinned binary on one staged sample ───────────────────────────────────────────────────────────

# never copied to out/: alignments, reads (STAR's unmapped mates are ~40% of a sample's raw bytes, the non-host read
# dump is reads too), temp dirs and core files.
EXCLUDE_OUT = re.compile(r"(\.bam$|\.bai$|\.csi$|_STARtmp|\.1fq$|^Aligned|\.sam$|core\.\d+$|Unmapped\.out\.mate\d$|"
                         r"\.f(ast)?q(\.gz)?$)")


# Engine exit 8: the unsorted stream's strand check failed (the reads contradict the protocol orientation).
STRAND_CONTRADICTION = 8


def strand_retry_cmd(cmd):
    """The compute command for the strand retry: the same run on the sorted-BAM path (idempotent)."""
    return cmd if "--sorted-bam" in cmd else list(cmd) + ["--sorted-bam"]


def classify_failure(rc, text, maxrss_gb=None):
    t = text.lower()
    if rc == STRAND_CONTRADICTION:   # still contradicting after the sorted-path retry: needs a look, not a retry
        return "strand_contradiction"
    if rc in (137, -9) or "out of memory" in t or "oom" in t or "std::bad_alloc" in t:
        return "oom"
    if rc == 124 or "stall" in t:
        return "stall_or_timeout"
    if "no space left" in t:
        return "disk_full"
    if "shared memory" in t or "genome" in t and "not found" in t:
        return "genome"
    return "exit_%s" % rc


def compute_stage(ctx, row, attempt, threads, shared_genome=False):
    gsm, gse, root = row["gsm"], row["gse"], ctx.root
    log = os.path.join(ctx.log_dir(gsm), "compute.a%d.%s.log" % (attempt, os.environ.get("SLURM_JOB_ID", "x")))
    emit(root, "running", gsm, attempt=attempt, stage="compute", threads=threads)
    sdir = os.path.join(ctx.stage_root, gsm)
    scr = ctx.scratch("cmp")
    t0 = time.time()
    try:
        verify_binary(ctx)
        onefq, screen = os.path.join(sdir, "sample.1fq"), os.path.join(sdir, "screen.json")
        if not staged_reads(sdir):
            raise StageError("staged_reads_missing", sdir)
        out_tmp = os.path.join(scr, "out")
        os.makedirs(out_tmp)
        # one single/pooled file, or the per-run files of a multi-run sample (the runs are named by screen.json)
        reads_args = [onefq] if os.path.exists(onefq) else ["--runs-dir", sdir]
        if reads_args[0] == "--runs-dir" and not os.path.exists(screen):
            raise StageError("staged_reads_missing", "per-run reads without %s" % screen)
        cmd = ["/usr/bin/time", "-v", "-o", os.path.join(scr, "time.txt"), ctx.bin] + reads_args + [
            "--ref-base", ctx.ref_base]
        ref = ctx.reference(row.get("organism"))
        if ref:
            cmd += ["--genome-dir", ref["genome_dir"], "--exons", ref["exons"]]
        cmd += ["--out-prefix", out_tmp + "/", "--threads", str(threads), "--pipeline", "--output-format", "1pz",
                "--bam-sort-mode", "disk", "--genome-shared", "on" if shared_genome else "off"]
        if os.path.exists(screen):
            cmd += ["--run-screen", screen]
        cmd += ctx.extra
        rc, secs, txt = run(cmd, log, env=ctx.env, cwd=scr)
        if rc == STRAND_CONTRADICTION:
            # The unsorted stream counted in the protocol orientation, and the reads contradict it. A pipe cannot be
            # re-read, so rerun once on the sorted-BAM path, whose strand safety net re-reads the BAM and recounts with
            # the opposite orientation (one orientation per library, so no mixed-strand mode).
            emit(root, "strand_retry", gsm, attempt=attempt, stage="compute", first_s=round(secs, 1))
            shutil.rmtree(out_tmp, ignore_errors=True)
            os.makedirs(out_tmp)
            rc2, secs2, txt = run(strand_retry_cmd(cmd), log, env=ctx.env, cwd=scr)
            rc, secs = rc2, secs + secs2
        maxrss = parse_time_v(os.path.join(scr, "time.txt"))
        if rc != 0:
            raise StageError(classify_failure(rc, txt), txt)
        out_dir, priv_dir, man_sha, out_bytes = collect_outputs(ctx, gse, gsm, out_tmp, sdir, log)
        stats = load_json_safe(os.path.join(out_dir, "pileup_stats.json")) or {}
        emit(root, "done", gsm, attempt=attempt, stage="compute", compute_s=round(secs, 1),
             compute_maxrss_gb=maxrss, pileup_total_reads=first_key(stats, ("total_reads", "reads_total", "n_reads")),
             n_cells=first_key(stats, ("n_called_cells", "n_cells_called", "cells_called", "n_cells")),
             binary_sha256=ctx.sha, out_dir=out_dir, private_dir=priv_dir, manifest_sha256=man_sha,
             out_bytes=out_bytes, wall_s=round(time.time() - t0, 1))
        shutil.rmtree(sdir, ignore_errors=True)
        return 0
    except StageError as e:
        emit(root, "failed", gsm, attempt=attempt, stage="compute", reason=e.reason, detail=e.detail)
        return 1
    except Exception as e:  # noqa: BLE001
        emit(root, "failed", gsm, attempt=attempt, stage="compute", reason="exception", detail=repr(e))
        return 1
    finally:
        shutil.rmtree(scr, ignore_errors=True)


def first_key(d, keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def parse_time_v(path):
    try:
        with open(path) as fh:
            for line in fh:
                if "Maximum resident set size" in line:
                    return round(int(line.split(":")[-1]) / 1e6, 2)  # kB -> GB
    except (OSError, ValueError):
        pass
    return None


def collect_outputs(ctx, gse, gsm, out_tmp, sdir, log):
    """private/ -> private_root (0700); everything else except BAM/reads -> out_root via a temp dir and a rename;
    MANIFEST.sha256 lists (sha256, bytes, path) of every public file; PRIVATE.sha256 the private ones."""
    final = os.path.join(ctx.out_root, gse, gsm)
    tmp = final + ".tmp-%s" % os.environ.get("SLURM_JOB_ID", "x")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    priv_src = os.path.join(out_tmp, "private")
    priv_dir = None
    if os.path.isdir(priv_src):
        priv_dir = os.path.join(ctx.private_root, gse, gsm)
        for d in (ctx.private_root, os.path.join(ctx.private_root, gse)):
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o700)
        shutil.rmtree(priv_dir, ignore_errors=True)
        shutil.copytree(priv_src, priv_dir)
        os.chmod(priv_dir, 0o700)
        write_manifest(priv_dir, "PRIVATE.sha256")
        shutil.rmtree(priv_src)
    for dp, dn, fn in os.walk(out_tmp):
        dn[:] = [d for d in dn if not EXCLUDE_OUT.search(d)]
        rel = os.path.relpath(dp, out_tmp)
        for f in fn:
            if EXCLUDE_OUT.search(f):
                continue
            dst = os.path.join(tmp, rel, f) if rel != "." else os.path.join(tmp, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(dp, f), dst)
    for name in ("screen.json", "head_screen.json"):
        if os.path.exists(os.path.join(sdir, name)):
            shutil.copyfile(os.path.join(sdir, name), os.path.join(tmp, name))
    shutil.copyfile(log, os.path.join(tmp, "run.log"))
    man = write_manifest(tmp, "MANIFEST.sha256")
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final, priv_dir, sha256_file(os.path.join(final, "MANIFEST.sha256")), man


def write_manifest(d, name):
    lines, total = [], 0
    for dp, dn, fn in os.walk(d):
        for f in sorted(fn):
            if f == name:
                continue
            p = os.path.join(dp, f)
            b = os.path.getsize(p)
            total += b
            lines.append("%s  %d  %s" % (sha256_file(p), b, os.path.relpath(p, d).replace(os.sep, "/")))
    with open(os.path.join(d, name), "w", newline="\n") as fh:
        fh.write("\n".join(sorted(lines, key=lambda ln: ln.split("  ", 2)[2])) + "\n")
    return total


# ── node packer (Anvil whole nodes; also usable on a Clipper slab for tests) ────────────────────────────────────────


def node_stage(ctx, batch_rows, batch_path, slots_cores, deadline, orchestrator, config_path, cluster):
    """Run the batch's staged samples on this node: one genome per reference in /dev/shm, samples started longest
    first while cores are free and the predicted finish (x1.3) is before the job's deadline. Samples never started are
    released back to `downloaded` for the next node job."""
    root = ctx.root
    refs = {}
    for r in batch_rows:
        ref = ctx.reference(r.get("organism"))
        if ref:
            refs[ref["genome_dir"]] = ref
    loaded = []

    def unload(*_a):
        for g in loaded:
            subprocess.run([ctx.bin, "genome", "unload", g], env=ctx.env)
        del loaded[:]

    signal.signal(signal.SIGTERM, lambda *a: (unload(), sys.exit(143)))
    try:
        verify_binary(ctx)
        for g in refs:
            rc = subprocess.run([ctx.bin, "genome", "load", g], env=ctx.env).returncode
            if rc != 0:
                for r in batch_rows:
                    emit(root, "released", r["gsm"], attempt=r["attempt"], stage="node", reason="genome_load_failed")
                return 1
            loaded.append(g)
        queue = sorted(range(len(batch_rows)), key=lambda i: -float(batch_rows[i].get("est_wall_s") or 0))
        running, free = {}, slots_cores
        # memory packing: the resident index plus every job's private peak must fit the node (config clusters.<c>.node)
        nc = ctx.c.get("node", {})
        node_gb = node_mem_gb()
        max_jobs = job_slots(node_gb, slots_cores, int(batch_rows[0].get("threads") or 4) if batch_rows else 4,
                             float(nc.get("job_peak_gb", 33.4)), float(nc.get("shared_index_gb", 17.8)),
                             float(nc.get("mem_fill", 0.9)))
        print("[node] %.0f GB, %d cores -> at most %d concurrent jobs" % (node_gb, slots_cores, max_jobs), flush=True)
        while queue or running:
            started = True
            while started and queue:
                started = False
                for qi, i in enumerate(queue):
                    r = batch_rows[i]
                    t = int(r.get("threads") or 4)
                    if t > free or len(running) >= max_jobs:
                        continue
                    if now() + 1.3 * float(r.get("est_wall_s") or 3600) > deadline:
                        continue
                    p = subprocess.Popen([sys.executable, orchestrator, "compute-stage", "--config", config_path,
                                          "--cluster", cluster, "--batch", batch_path, "--index", str(i),
                                          "--shared-genome"])
                    running[p.pid] = (p, i, t)
                    free -= t
                    queue.pop(qi)
                    started = True
                    break
            for pid, (p, i, t) in list(running.items()):
                if p.poll() is not None:
                    free += t
                    del running[pid]
            if queue and not running and free == slots_cores:
                break  # nothing left fits before the deadline
            time.sleep(10)
        for i in queue:
            r = batch_rows[i]
            emit(root, "released", r["gsm"], attempt=r["attempt"], stage="node", reason="deadline")
        return 0
    finally:
        unload()
