#!/usr/bin/env python3
"""Atlas re-run orchestrator: one plan, two clusters (Clipper, Anvil), a ledger per cluster merged on the laptop.

Design (docs/atlas-rerun-orchestration.md has the operator's guide):

* **Static partition by study.** `plan` assigns every study (all of its samples) to exactly one cluster, so the clusters
  never compete for a sample and a study is always produced by one cluster with one binary (per-study publishing stays
  simple). Moving a study is an explicit two-step `reassign` (release on the old cluster, then assign on the new one).
* **One controller per cluster, the only writer of that cluster's ledger.** It runs as a small self-renewing Slurm job
  (`controller`), takes a lease (lock/), folds the events jobs wrote into inbox/ into the append-only events.jsonl,
  reconciles with squeue/sacct, retries once then parks, submits new work in plan order and writes snapshots.
* **Jobs never touch shared state**: they write one event file per state change (atomic rename into inbox/).
* **The laptop** pulls each cluster's small snapshot files (`pull`) and merges them (`status`). Nothing on the laptop is
  needed for the campaign to keep running.

Subcommands
  plan            universe + run tables -> plan.tsv.gz (+ .json): order, cluster assignment, per-sample estimates
  dryrun          plan everything for a cluster without submitting (fake Slurm) and print the projection
  tick            one controller pass (normally run by `controller`)
  controller      loop of ticks inside a Slurm job; queues its successor (afterany) so it survives time limits
  status          progress per wave and cluster, throughput, failures, SU, ETA (from one or more ledger dirs)
  pull            laptop: fetch a cluster's ledger snapshots (Clipper through the clipper skill)
  reassign        move studies between clusters (release, then assign; refuses studies with work in flight)
  unpark          return parked samples to the queue (operator decision)
  rebuild         refold the ledger from events.jsonl
  io-stage / compute-stage / node-stage   job entry points (one array task = one sample)
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402
from ledger import ACTIVE, TERMINAL, Ledger, read_tsv  # noqa: E402
from slurm import FakeSlurm, RealSlurm, parse_elapsed, su_of_job  # noqa: E402

ORCH = os.path.abspath(__file__)
TEMPLATES = os.path.join(C.HERE, "templates")
WAVES = ("a_empty", "b_strand", "c_skipped10x", "d_rest")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# plan
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


def accession_key(a):
    a = str(a)
    digits = "".join(ch for ch in a if ch.isdigit())
    return (a.rstrip("0123456789"), len(digits), digits)


def assign_clusters(studies, policy):
    """studies: list of dicts (gse, wave, est_su, max_reads, n) in global order. Returns {gse: cluster}.

    Clipper gets everything by default. If policy.anvil_su_budget > 0, Anvil gets whole wave-d studies, taken in wave-d
    order (largest first), while their summed SU estimate fits the budget; studies with a sample beyond
    anvil_max_sample_reads stay on Clipper (they need a 5-day partition)."""
    out = {s["gse"]: "clipper" for s in studies}
    budget = float(policy.get("anvil_su_budget", 0) or 0)
    if budget <= 0:
        return out
    waves = set(policy.get("anvil_waves", ["d_rest"]))
    max_reads = float(policy.get("anvil_max_sample_reads", 1.3e9))
    min_n = int(policy.get("anvil_min_study_samples", 1))
    used = 0.0
    for s in studies:
        if s["wave"] not in waves or s["n"] < min_n or (s["max_reads"] or 0) > max_reads:
            continue
        if used + s["est_su"] <= budget:
            out[s["gse"]] = "anvil"
            used += s["est_su"]
    return out


def cmd_plan(a):
    import pandas as pd  # the only subcommand that needs pandas; it runs where the universe tables are

    cfg = C.load_config(a.config)
    pol = cfg["policy"]
    nonhost = cfg.get("pipeline", {}).get("nonhost", True)
    u = pd.read_parquet(a.universe)
    u = u[u.route == "gex"].copy()
    runs = pd.concat([pd.read_parquet(p) for p in a.runs], ignore_index=True)
    runs = runs.drop_duplicates("run_accession").dropna(subset=["gsm"])
    runs = runs[runs.gsm.isin(set(u.gsm_id))]
    runs["_k"] = runs.run_accession.map(accession_key)
    runs = runs.sort_values(["gsm", "_k"])
    per = runs.groupby("gsm").agg(runs=("run_accession", lambda x: ",".join(x)),
                                  run_reads=("read_count", lambda x: ",".join(str(int(v)) for v in x)),
                                  reads=("read_count", "sum"), bases=("base_count", "sum"),
                                  n_runs=("run_accession", "size"))
    u = u.drop(columns=[c for c in ("reads", "bases", "n_runs") if c in u.columns]).merge(
        per, left_on="gsm_id", right_index=True, how="left")
    u["runs"] = u.runs.fillna("")
    u["run_reads"] = u.run_reads.fillna("")
    u["n_runs"] = u.n_runs.fillna(0).astype(int)
    u["depth_unknown"] = (u.reads.fillna(0) <= 0).astype(int)
    # a study runs (and is handed off) as one unit, in the earliest wave any of its samples belongs to
    wrank = {w: i for i, w in enumerate(WAVES)}
    u["_w"] = u.wave.map(lambda w: wrank.get(w, len(WAVES)))
    # (policy.study_unit = false keeps each sample in its own wave; a study is still handed off only when complete)
    first = u.groupby("gse_id")._w.transform("min") if pol.get("study_unit", True) else u._w
    u["study_wave"] = first.map(lambda i: WAVES[i] if i < len(WAVES) else "z_other")
    if a.only_gse:
        u = u[u.gse_id.isin(set(a.only_gse.split(",")))]
    if a.only_gsm:
        u = u[u.gsm_id.isin(set(a.only_gsm.split(",")))]

    refs = cfg.get("references", {})
    rows = []
    for r in u.itertuples(index=False):
        reads = int(r.reads) if r.reads == r.reads and r.reads > 0 else None
        bases = int(r.bases) if r.bases == r.bases and r.bases > 0 else None
        rows.append({
            "gsm": r.gsm_id, "gse": r.gse_id, "wave": r.wave, "study_wave": r.study_wave, "organism": r.organism or "",
            "reference": refs.get(r.organism or "", refs.get("*", "auto")), "reads_available": reads,
            "depth_unknown": int(r.depth_unknown), "bases": bases, "n_runs": int(r.n_runs), "runs": r.runs,
            "run_reads": r.run_reads, "study_cells": float(r.study_cells or 0),
            "est_wall_s": round(C.compute_wall_s(reads, nonhost), 1),
            "est_core_h": round(C.compute_core_h(reads, nonhost), 4),
            "est_su": round(C.anvil_su(reads, nonhost), 4), "est_sra_bytes": int(C.sra_bytes(reads, bases)),
            "size_class": size_class(reads, cfg),
        })
    # order: waves a -> d; studies within a wave by the wave's rule; samples by accession
    order = pol.get("order_within", {})
    by_study = defaultdict(list)
    for r in rows:
        by_study[(r["study_wave"], r["gse"])].append(r)

    def study_key(k):
        wave, gse = k
        rs = by_study[k]
        rule = order.get(wave, "accession")
        w = WAVES.index(wave) if wave in WAVES else len(WAVES)
        if rule == "study_cells_desc":
            return (w, -rs[0]["study_cells"], accession_key(gse))
        if rule == "reads_asc":
            return (w, sum(x["reads_available"] or C.MEDIAN_READS for x in rs), accession_key(gse))
        return (w, accession_key(gse))

    skeys = sorted(by_study, key=study_key)
    studies = []
    for k in skeys:
        rs = by_study[k]
        studies.append({"gse": k[1], "wave": k[0], "n": len(rs), "est_su": sum(x["est_su"] for x in rs),
                        "max_reads": max((x["reads_available"] or 0) for x in rs)})
    assign = assign_clusters(studies, pol)
    out, pri = [], 0
    for si, k in enumerate(skeys):
        for sj, r in enumerate(sorted(by_study[k], key=lambda x: accession_key(x["gsm"]))):
            r.update({"priority": pri, "study_rank": si, "sample_rank": sj, "cluster": assign[k[1]]})
            out.append(r)
            pri += 1
    meta = {"created": C.iso(), "universe": os.path.abspath(a.universe), "universe_sha256": C.sha256_file(a.universe),
            "runs_tables": {os.path.basename(p): C.sha256_file(p) for p in a.runs}, "policy": pol,
            "n_samples": len(out), "n_studies": len(skeys), "cost_model": cost_model_meta(nonhost)}
    meta = C.write_plan(a.out, out, meta)
    print("plan %s: %d samples, %d studies -> %s" % (meta["plan_id"], len(out), len(skeys), a.out))
    print_plan_summary(out)


def size_class(reads, cfg):
    r = reads or C.MEDIAN_READS
    if r > float(cfg["policy"].get("xl_reads", 1.3e9)):
        return "XL"
    if r > 6e8:
        return "L"
    if r > 2.5e8:
        return "M"
    return "S"


def cost_model_meta(nonhost):
    return {"fixed_s": C.FIXED_S, "slope_s_per_M": C.SLOPE_S_PER_M[bool(nonhost)], "star_speedup": C.STAR_SPEEDUP,
            "pack": C.PACK, "retry": C.RETRY, "nonhost": bool(nonhost)}


def print_plan_summary(rows):
    agg = defaultdict(lambda: Counter())
    gses = defaultdict(set)
    for r in rows:
        k = (r["study_wave"], r["cluster"])
        agg[k]["samples"] += 1
        agg[k]["reads"] += r["reads_available"] or C.MEDIAN_READS
        agg[k]["core_h"] += r["est_core_h"]
        agg[k]["su"] += r["est_su"]
        agg[k]["sra"] += r["est_sra_bytes"]
        agg[k]["xl"] += r["size_class"] == "XL"
        gses[k].add(r["gse"])
    print("%-14s %-8s %8s %8s %10s %10s %10s %9s %5s" % ("wave", "cluster", "studies", "samples", "Gpairs",
                                                          "core-h", "SU(anvil)", "SRA", "XL"))
    for k in sorted(agg):
        g = agg[k]
        print("%-14s %-8s %8d %8d %10.0f %10.0f %10.0f %9s %5d" % (k[0], k[1], len(gses[k]), g["samples"],
                                                                   g["reads"] / 1e9, g["core_h"], g["su"],
                                                                   C.fmt_bytes(g["sra"]), g["xl"]))


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# controller
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


class Controller:
    def __init__(self, cfg, cluster, slurm, plan_path=None, root=None, log=print):
        self.cfg = cfg
        self.c = C.cluster_cfg(cfg, cluster)
        if root:
            self.c["root"] = root
        self.cluster = cluster
        self.root = self.c["root"]
        self.slurm = slurm
        self.log = log
        self.plan_path = plan_path or os.path.join(self.root, "plan", "plan.tsv.gz")
        self.plan = C.read_plan(self.plan_path, cluster=cluster)
        self.plan_meta = C.read_plan_meta(self.plan_path)
        self.by_gsm = {r["gsm"]: r for r in self.plan}
        self.L = Ledger(self.root)
        self.new_events = []
        self.stats = Counter()

    # -- helpers ----------------------------------------------------------------------------------------------------
    def event(self, ev, gsm=None, **f):
        rec = {"t": round(C.now(), 3), "ts": C.iso(), "event": ev, "cluster": self.cluster, "by": "controller"}
        if gsm:
            rec["gsm"] = gsm
        rec.update({k: v for k, v in f.items() if v is not None})
        self.L.append([rec])
        self.L.apply(rec)
        self.new_events.append(rec)
        return rec

    def flag(self, name):
        return os.path.exists(os.path.join(self.root, name))

    def install(self):
        for d in ("inbox", "batches", "slurm", "logs", "ledger", "handoff", "plan", "bin"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        b = self.cfg["binary"]
        dst = os.path.join(self.root, "bin", "singlet-" + b["sha256"][:12])
        if not os.path.exists(dst) and not self.slurm.dry:
            src = self.c.get("binary_source") or b["path"]
            got = C.sha256_file(src)
            if got != b["sha256"]:
                raise SystemExit("binary %s has sha256 %s, config pins %s" % (src, got, b["sha256"]))
            shutil.copyfile(src, dst + ".tmp")
            os.chmod(dst + ".tmp", 0o555)
            os.replace(dst + ".tmp", dst)

    # -- one pass ---------------------------------------------------------------------------------------------------
    def tick(self):
        t0 = time.time()
        self.install()
        self.L.ensure(self.plan)
        self.stats["ingested"] = self.L.ingest_inbox()
        self.stats["folded"] = self.L.fold()
        self.reconcile()
        self.handle_failures()
        paused = self.flag("PAUSE") or self.flag("STOP")
        if paused:
            self.stats["paused"] = 1
        elif self.c.get("enabled") is False:
            self.stats["disabled"] = 1
        else:
            if self.c.get("mode", "per_sample") == "node":
                self.submit_node_mode()
            else:
                self.submit_per_sample()
        self.handoffs()
        status = self.status()
        status["tick_s"] = round(time.time() - t0, 1)
        self.L.save()
        self.L.write_snapshots(status)
        return status

    # -- reconcile with Slurm ---------------------------------------------------------------------------------------
    def expected_job(self, s):
        st = s["state"]
        if st in ("queued", "downloading", "screened") and s.get("io_job"):
            return "io", s["io_job"]
        if st in ("downloaded", "running"):
            if s.get("node_job"):
                return "node", s["node_job"]
            if s.get("compute_job"):
                return "compute", s["compute_job"]
        return None, None

    def reconcile(self):
        live = [(g, s) + self.expected_job(s) for g, s in self.L.samples.items()]
        live = [x for x in live if x[3]]
        if not live or self.slurm.dry:
            return
        q = self.slurm.queue()
        missing = [x for x in live if x[3] not in q]
        if not missing:
            for g, s, _, _ in live:
                s.pop("missing_ticks", None)
            return
        acct = self.slurm.acct([x[3] for x in missing])
        for g, s, stage, job in missing:
            a = acct.get(job, {})
            state = a.get("state", "")
            if state in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED"):
                continue
            # an io job that completed hands over to the chained compute job (or to a node batch on Anvil)
            if stage == "io" and state == "COMPLETED" and s["state"] == "downloaded":
                continue
            s["missing_ticks"] = s.get("missing_ticks", 0) + 1
            if s["missing_ticks"] < 2:
                continue  # one tick of grace: the job's last event may not be visible on the shared FS yet
            reason = {"TIMEOUT": "timeout", "OUT_OF_MEMORY": "oom", "NODE_FAIL": "node_fail", "PREEMPTED": "preempted",
                      "CANCELLED": "cancelled", "FAILED": "failed_no_event", "COMPLETED": "completed_no_event",
                      "BOOT_FAIL": "node_fail", "DEADLINE": "timeout"}.get(state, "lost_" + (state or "unknown"))
            if stage == "node" and reason in ("timeout", "cancelled", "node_fail", "preempted"):
                self.event("released", g, attempt=s.get("attempt"), stage="node", reason=reason)
                continue
            self.event("lost", g, attempt=s.get("attempt"), stage=stage, reason=reason, job=job,
                       maxrss_gb=a.get("maxrss_gb"), elapsed_s=a.get("elapsed_s"))
            s.pop("missing_ticks", None)

    def handle_failures(self):
        rmax = int(self.cfg["policy"].get("retry_max", 1))
        for g, s in self.L.samples.items():
            if s["state"] != "failed":
                continue
            stale = [j for j in (s.get("io_job"), s.get("compute_job")) if j]
            if s["failures"] > rmax:
                self.slurm.cancel(stale)
                self.event("parked", g, attempt=s.get("attempt"), stage="controller",
                           reason="retries_exhausted(%s)" % s.get("last_reason"))
                continue
            self.slurm.cancel(stale)
            reason = s.get("last_reason", "")
            escalate = "timeout" if "timeout" in reason else ("oom" if "oom" in reason else None)
            staged = os.path.exists(os.path.join(self.c.get("stage_root") or os.path.join(self.root, "stage"), g,
                                                 "sample.1fq"))
            from_stage = "compute" if reason.startswith("compute") and staged else "io"
            self.event("requeued", g, attempt=int(s.get("attempt", 0)) + 1, escalate=escalate, from_stage=from_stage)

    # -- capacity ---------------------------------------------------------------------------------------------------
    def counts(self):
        io_inflight = sum(1 for s in self.L.samples.values()
                          if (s["state"] in ("downloading", "screened")) or (s["state"] == "queued" and s.get("io_job")))
        staged = sum(1 for s in self.L.samples.values() if s["state"] in ("downloaded", "running"))
        staged_bytes = sum(float(s.get("staged_bytes") or 0) for s in self.L.samples.values()
                           if s["state"] in ("downloaded", "running"))
        compute_inflight = sum(1 for s in self.L.samples.values()
                               if s["state"] not in TERMINAL and s["state"] != "failed" and
                               (s.get("compute_job") or s.get("node_job")))
        return io_inflight, staged, staged_bytes, compute_inflight

    def choose_class(self, row, s):
        classes = self.c["compute_classes"]
        safety = float(self.c.get("wall_safety", 1.6))
        reads = row.get("reads_available") or C.MEDIAN_READS
        nonhost = self.c.get("nonhost", self.cfg.get("pipeline", {}).get("nonhost", True))
        idx = None
        if reads > float(self.cfg["policy"].get("xl_reads", 1.3e9)):
            idx = len(classes) - 1
        else:
            for i, cl in enumerate(classes):
                if C.compute_wall_s(reads, nonhost, cl["threads"], packed=False) * safety <= parse_elapsed(cl["time"]):
                    idx = i
                    break
            if idx is None:
                idx = len(classes) - 1
        cl = dict(classes[idx])
        esc = s.get("escalate")
        if esc == "timeout" and idx + 1 < len(classes):
            cl = dict(classes[idx + 1])
        if esc == "oom":
            cl["mem_gb"] = int(cl["mem_gb"] * 1.5)
            cl["name"] += "+mem"
        return cl

    # -- per-sample mode (Clipper): io array + compute array chained task-by-task (aftercorr) ----------------------
    def next_queued(self, limit):
        out = []
        for r in self.plan:
            s = self.L.samples[r["gsm"]]
            if s["state"] == "queued" and not s.get("io_job"):
                out.append(r)
                if len(out) >= limit:
                    break
        return out

    def submit_per_sample(self):
        io_inflight, staged, staged_bytes, compute_inflight = self.counts()
        io_cap = int(self.c["io"]["max_inflight"])
        cmp_cap = int(self.c.get("max_compute_inflight", 64))
        stage_cap = float(self.c.get("staging_cap_tb", 20)) * 1e12
        # 1. compute-only retries (staged reads kept)
        retry_rows = [self.by_gsm[g] for g, s in self.L.samples.items()
                      if s["state"] == "downloaded" and not s.get("compute_job") and g in self.by_gsm]
        if retry_rows:
            self.submit_batches(retry_rows, with_io=False)
        # 2. new samples
        room = min(io_cap - io_inflight, cmp_cap - compute_inflight)
        if staged_bytes >= stage_cap:
            room = 0
            self.stats["staging_full"] = 1
        if room <= 0:
            return
        rows = self.next_queued(room)
        if rows:
            self.submit_batches(rows, with_io=True)

    def io_profile(self, r):
        """The I/O job shape for a sample: the default (4 h partition) unless its predicted I/O time x3 exceeds it."""
        io = dict(self.c["io"])
        io["name"] = "io"
        if self.c.get("io_long") and C.io_wall_s(r.get("reads_available"), r.get("bases")) * 3 > parse_elapsed(
                io.get("time", "03:59:00")):
            io.update(self.c["io_long"])
            io["name"] = "io_long"
        return io

    def submit_batches(self, rows, with_io):
        groups = defaultdict(list)
        for r in rows:
            s = self.L.samples[r["gsm"]]
            cl = self.choose_class(r, s)
            groups[(cl["name"], self.io_profile(r)["name"])].append((r, s, cl))
        bs = int(self.c.get("batch_size", 16))
        for key, items in groups.items():
            for i in range(0, len(items), bs):
                self.submit_one_batch(items[i:i + bs], with_io)

    def write_batch(self, items, kind):
        bid = "%s-%s-%d-%s" % (time.strftime("%Y%m%d-%H%M%S"), kind, os.getpid(), items[0][0]["gsm"])
        path = os.path.join(self.root, "batches", bid + ".jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="\n") as fh:
            for r, s, cl in items:
                rec = {k: r.get(k) for k in ("gsm", "gse", "wave", "organism", "reference", "runs", "run_reads",
                                             "reads_available", "est_wall_s")}
                rec.update({"attempt": int(s.get("attempt", 0)), "threads": cl.get("threads", 4),
                            "compute_class": cl.get("name")})
                fh.write(json.dumps(rec) + "\n")
        return bid, path

    def sbatch_common(self, name, part, cpus, mem_gb, time_s, array=None):
        o = ["--job-name=" + name, "--partition=" + part, "--cpus-per-task=%d" % cpus, mem_opt(mem_gb),
             "--time=" + time_s, "--output=" + os.path.join(self.root, "slurm", "%x-%A_%a.out"),
             "--export=ALL,AR_PYTHON=" + self.c.get("python", "python3")]
        if array:
            o.append("--array=" + array)
        o += list(self.c.get("sbatch_extra", []))
        return o

    def submit_one_batch(self, items, with_io):
        n = len(items)
        bid, path = self.write_batch(items, "b")
        cl = items[0][2]
        args = [ORCH, self.cfg["_path"], self.cluster, path]
        io_job = None
        if with_io:
            io = self.io_profile(items[0][0])
            thr = int(io.get("array_throttle", n))
            opts = self.sbatch_common("ar-io-" + self.cluster, io["partition"], int(io.get("cpus", 1)),
                                      float(io.get("mem_gb", 8)), io.get("time", "03:59:00"),
                                      "0-%d%%%d" % (n - 1, max(1, min(thr, n))))
            if io.get("requeue"):
                opts.append("--requeue")
            io_job = self.slurm.submit(os.path.join(TEMPLATES, "io.sbatch"), args, opts)
        opts = self.sbatch_common("ar-cmp-%s-%s" % (self.cluster, cl["name"]), cl["partition"], int(cl["threads"]),
                                  float(cl["mem_gb"]), cl["time"], "0-%d" % (n - 1))
        if cl.get("requeue"):
            opts.append("--requeue")
        if io_job:
            opts += ["--dependency=aftercorr:" + io_job, "--kill-on-invalid-dep=yes"]
        cmp_job = self.slurm.submit(os.path.join(TEMPLATES, "compute.sbatch"), args, opts)
        for i, (r, s, cl_) in enumerate(items):
            if io_job:
                self.event("submitted", r["gsm"], stage="io", job="%s_%d" % (io_job, i), attempt=s.get("attempt", 0),
                           batch=bid)
            self.event("submitted", r["gsm"], stage="compute", job="%s_%d" % (cmp_job, i),
                       attempt=s.get("attempt", 0), compute_class=cl_["name"], threads=cl_["threads"],
                       partition=cl_["partition"], batch=bid)
        self.stats["submitted_io"] += n if io_job else 0
        self.stats["submitted_compute"] += n
        self.log("[tick] %s batch %s: %d samples, io=%s compute=%s (%s, %d threads)" % (
            self.cluster, bid, n, io_job, cmp_job, cl["partition"], cl["threads"]))

    # -- node mode (Anvil): io arrays on `shared`, then whole-node packer jobs over staged samples ------------------
    def submit_node_mode(self):
        io_inflight, staged, staged_bytes, compute_inflight = self.counts()
        guard = self.budget_guard()
        if guard.get("stop"):
            self.stats["budget_stop"] = 1
            if not self.L.campaign.get("budget_stop"):
                self.event("budget_stop", reason=guard.get("why"))
            return
        io = self.c["io"]
        stage_cap = float(self.c.get("staging_cap_tb", 50)) * 1e12
        room = int(io["max_inflight"]) - io_inflight
        if staged_bytes < stage_cap and room > 0:
            rows = self.next_queued(room)
            bs = int(self.c.get("batch_size", 32))
            for i in range(0, len(rows), bs):
                chunk = rows[i:i + bs]
                items = [(r, self.L.samples[r["gsm"]], {"name": "io", "threads": 4}) for r in chunk]
                bid, path = self.write_batch(items, "io")
                n = len(items)
                opts = self.sbatch_common("ar-io-" + self.cluster, io["partition"], int(io.get("cpus", 1)),
                                          float(io.get("mem_gb", 2)), io.get("time", "06:00:00"),
                                          "0-%d%%%d" % (n - 1, max(1, min(int(io.get("array_throttle", n)), n))))
                if not self.charge_ok(guard, self.io_su(io, n)):
                    break
                job = self.slurm.submit(os.path.join(TEMPLATES, "io.sbatch"),
                                        [ORCH, self.cfg["_path"], self.cluster, path], opts)
                self.reserve(job, self.io_su(io, n))
                for k, (r, s, _) in enumerate(items):
                    self.event("submitted", r["gsm"], stage="io", job="%s_%d" % (job, k), attempt=s.get("attempt", 0),
                               batch=bid)
                self.stats["submitted_io"] += n
        # node jobs over staged samples not yet in a node job
        node = self.c["node"]
        ready = [self.by_gsm[g] for g, s in self.L.samples.items()
                 if s["state"] == "downloaded" and not s.get("node_job") and g in self.by_gsm]
        ready.sort(key=lambda r: r["priority"])
        hours = parse_elapsed(node["time"]) / 3600.0
        cap_core_h = int(node["slot_cores"]) * hours * float(node.get("fill", 0.8))
        nodes_running = len(set(s.get("node_job") for s in self.L.samples.values()
                                if s.get("node_job") and s["state"] in ("downloaded", "running")))
        drain = io_inflight == 0 and not self.next_queued(1)
        while ready and nodes_running < int(node.get("max_nodes", 1)):
            batch, core_h = [], 0.0
            for r in list(ready):
                s = self.L.samples[r["gsm"]]
                cl = self.choose_class(r, s)
                ch = C.compute_core_h(r.get("reads_available"), self.c.get("nonhost", True), cl["threads"])
                if core_h + ch > cap_core_h and batch:
                    continue
                batch.append((r, s, cl))
                core_h += ch
                ready.remove(r)
            if not batch or (core_h < 0.5 * cap_core_h and not drain):
                break  # wait for more staged samples rather than run a half-empty node
            su = int(node.get("nodes", 1)) * C.ANVIL_NODE_CORES * hours
            if not self.charge_ok(guard, su):
                break
            bid, path = self.write_batch(batch, "node")
            opts = ["--job-name=ar-node-" + self.cluster, "--partition=" + node["partition"], "--nodes=1",
                    "--ntasks=1", "--exclusive", "--time=" + node["time"], "--signal=B:TERM@600",
                    "--output=" + os.path.join(self.root, "slurm", "%x-%j.out"),
                    "--export=ALL,AR_PYTHON=%s,AR_TIME=%s" % (self.c.get("python", "python3"), node["time"])
                    ] + list(self.c.get("sbatch_extra", []))
            job = self.slurm.submit(os.path.join(TEMPLATES, "node.sbatch"),
                                    [ORCH, self.cfg["_path"], self.cluster, path, str(node["slot_cores"])], opts)
            self.reserve(job, su)
            for r, s, cl in batch:
                self.event("submitted", r["gsm"], stage="node", job=job, attempt=s.get("attempt", 0),
                           compute_class=cl["name"], threads=cl["threads"], partition=node["partition"], batch=bid)
            nodes_running += 1
            self.stats["submitted_node"] += 1

    # -- Anvil budget guard -----------------------------------------------------------------------------------------
    def io_su(self, io, n):
        cores = max(int(io.get("cpus", 1)), float(io.get("mem_gb", 2)) / 1.9)
        return n * cores * parse_elapsed(io.get("time", "06:00:00")) / 3600.0

    def reserve(self, job, su):
        self.L.campaign.setdefault("reservations", {})[str(job)] = {"su": su, "t": C.now()}

    def budget_guard(self):
        """Spent SU (sacct over every campaign job since the campaign start) + SU reserved by jobs still queued or
        running (their full time limit) must stay under su_cap; `mybalance` must stay above su_floor."""
        cap = float(self.c.get("su_cap", 0) or 0)
        g = {"cap": cap, "spent": 0.0, "reserved": 0.0, "balance": None}
        if self.c.get("budget") != "su":
            return g
        if cap <= 0:
            g.update(stop=True, why="su_cap is 0 (Anvil not approved)")
            return g
        since = self.c.get("campaign_start", "2026-10-01")
        rows = self.slurm.campaign_usage(since)
        spent = sum(su_of_job(r) for r in rows)
        q = self.slurm.queue() if not self.slurm.dry else {}
        live = set(j.split("_")[0] for j in q)
        res = self.L.campaign.get("reservations", {})
        reserved = sum(v["su"] for j, v in res.items() if j in live)
        for j in [j for j in res if j not in live and C.now() - res[j]["t"] > 3600]:
            del res[j]  # finished: its real charge is in sacct now
        bal = self.slurm.balance(self.c["account"])
        g.update(spent=spent, reserved=reserved, balance=bal)
        if spent + reserved >= cap:
            g.update(stop=True, why="spent %.0f + reserved %.0f >= cap %.0f" % (spent, reserved, cap))
        floor = float(self.c.get("su_floor", 5000))
        if bal is not None and bal < floor:
            g.update(stop=True, why="balance %.0f below floor %.0f" % (bal, floor))
        self.L.campaign["su"] = {k: g[k] for k in ("cap", "spent", "reserved", "balance")}
        return g

    def charge_ok(self, guard, su):
        if self.c.get("budget") != "su":
            return True
        if guard["spent"] + guard["reserved"] + su > guard["cap"]:
            self.stats["budget_block"] += 1
            return False
        guard["reserved"] += su
        return True

    # -- per-study hand-off -----------------------------------------------------------------------------------------
    def handoffs(self):
        by_study = defaultdict(list)
        for s in self.L.samples.values():
            by_study[s["gse"]].append(s)
        for gse, ss in by_study.items():
            if gse in self.L.studies and self.L.studies[gse].get("handoff"):
                continue
            if not all(s["state"] in TERMINAL for s in ss):
                continue
            path = write_handoff(self, gse, ss)
            self.event("study_handoff", gse=gse, path=path, n_done=sum(s["state"] == "done" for s in ss),
                       n_parked=sum(s["state"] == "parked" for s in ss))

    def status(self):
        st = summarize(list(self.L.samples.values()), self.cluster)
        st["controller"] = {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "t": C.iso(),
                            "plan_id": self.plan_meta.get("plan_id"), "stats": dict(self.stats),
                            "paused": self.flag("PAUSE"), "stopped": self.flag("STOP")}
        if self.L.campaign.get("su"):
            st["su"] = dict(self.L.campaign["su"])
            remaining = [self.by_gsm[g]["est_su"] for g, s in self.L.samples.items()
                         if s["state"] not in TERMINAL and g in self.by_gsm]
            st["su"]["projected_remaining"] = sum(x or 0 for x in remaining)
        return st


def mem_opt(gb):
    gb = float(gb)
    return "--mem=%dG" % gb if gb == int(gb) else "--mem=%dM" % round(gb * 1024)


def write_handoff(ctl, gse, samples):
    """The v2 packer's input for one study: every sample's outputs, provenance and state (packing is not done here)."""
    done = [s for s in samples if s["state"] == "done"]
    man = {
        "schema": "singlet.atlas_rerun.handoff/1", "gse": gse, "cluster": ctl.cluster, "created": C.iso(),
        "plan_id": ctl.plan_meta.get("plan_id"), "binary_sha256": ctl.cfg["binary"]["sha256"],
        "binary_commit": ctl.cfg["binary"].get("commit"), "pipeline": ctl.cfg.get("pipeline", {}),
        "ready_for_packing": bool(done),
        "samples": [{k: s.get(k) for k in ("gsm", "wave", "state", "out_dir", "manifest_sha256", "out_bytes",
                                           "reads_available", "reads_used", "runs_planned", "runs_kept",
                                           "runs_dropped", "chemistry", "n_cells", "binary_sha256", "attempt")}
                    for s in sorted(samples, key=lambda x: accession_key(x["gsm"]))],
        "parked": [{"gsm": s["gsm"], "reason": s.get("last_reason")} for s in samples if s["state"] == "parked"],
        "private": {"root": os.path.join(ctl.c.get("private_root") or os.path.join(ctl.root, "private"), gse),
                    "note": "private/ (SNP allele counts) must never be uploaded to a public bucket"},
    }
    path = os.path.join(ctl.root, "handoff", gse + ".json")
    C.write_json(path, man)
    return path


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# status (works on the controller's samples and on pulled snapshots)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(samples, cluster=None, window_h=24.0):
    t = C.now()
    out = {"cluster": cluster, "t": C.iso(t), "waves": {}, "failures": {}, "throughput": {}}
    by_wave = defaultdict(Counter)
    reasons = Counter()
    done_recent, bytes_recent, done_all, first_t = 0, 0.0, 0, None
    ratios, est_rem_core_h = [], 0.0
    for s in samples:
        st = s.get("state", "queued")
        w = s.get("wave", "?")
        by_wave[w]["n"] += 1
        by_wave[w][st] += 1
        if st in ("failed", "parked") and s.get("last_reason"):
            reasons["%s (%s)" % (s["last_reason"], st)] += 1
        td = _f(s.get("t_done"))
        if st == "done" and td:
            done_all += 1
            first_t = min(first_t or td, _f(s.get("t_submitted")) or td)
            if t - td < window_h * 3600:
                done_recent += 1
            cs, ew = _f(s.get("compute_s")), _f(s.get("est_wall_s"))
            if cs and ew:
                ratios.append(cs / (ew / C.PACK))
        tdl = _f(s.get("t_downloaded"))
        if tdl and t - tdl < window_h * 3600:
            bytes_recent += _f(s.get("sra_bytes")) or 0
        if st not in TERMINAL:
            est_rem_core_h += 4 * (_f(s.get("est_wall_s")) or 0) / 3600
    for w, c in sorted(by_wave.items()):
        out["waves"][w] = dict(c)
    out["failures"] = dict(reasons.most_common())
    rate = done_recent / window_h
    remaining = sum(c["n"] - c.get("done", 0) - c.get("parked", 0) for c in by_wave.values())
    out["throughput"] = {"samples_per_h_24h": round(rate, 2), "tb_per_day_24h": round(bytes_recent / 1e12 *
                                                                                     (24.0 / window_h), 3),
                         "done_total": done_all, "remaining": remaining,
                         "eta_days_at_24h_rate": round(remaining / rate / 24, 1) if rate > 0 else None,
                         "remaining_est_core_h": round(est_rem_core_h, 0),
                         "compute_vs_model_median": round(sorted(ratios)[len(ratios) // 2], 3) if ratios else None}
    return out


def print_status(st_list, out=sys.stdout):
    states = ("queued", "downloading", "screened", "downloaded", "running", "done", "failed", "parked")
    w = out.write
    w("%-10s %-14s %7s " % ("cluster", "wave", "n") + " ".join("%11s" % s for s in states) + "\n")
    for st in st_list:
        for wave, c in sorted(st["waves"].items()):
            w("%-10s %-14s %7d " % (st["cluster"], wave, c.get("n", 0)) +
              " ".join("%11d" % c.get(s, 0) for s in states) + "\n")
    w("\n")
    for st in st_list:
        tp = st["throughput"]
        w("%s: %.2f samples/h (24 h), %.2f TB/day downloaded, %d done, %d remaining, ETA %s days at the 24 h rate; "
          "model core-h left %.0f; compute time / model %s\n" % (
              st["cluster"], tp["samples_per_h_24h"], tp["tb_per_day_24h"], tp["done_total"], tp["remaining"],
              tp["eta_days_at_24h_rate"], tp["remaining_est_core_h"], tp["compute_vs_model_median"]))
        if st.get("su"):
            su = st["su"]
            w("  SU: spent %.0f, reserved %.0f, cap %.0f, balance %s, projected for the rest %.0f\n" % (
                su.get("spent", 0), su.get("reserved", 0), su.get("cap", 0), su.get("balance"),
                su.get("projected_remaining", 0)))
        ctl = st.get("controller") or {}
        if ctl:
            w("  controller %s job %s at %s, plan %s, paused=%s stopped=%s, last tick %s\n" % (
                ctl.get("host"), ctl.get("job"), ctl.get("t"), ctl.get("plan_id"), ctl.get("paused"),
                ctl.get("stopped"), ctl.get("stats")))
        if st["failures"]:
            w("  failures / parked by reason:\n")
            for k, v in list(st["failures"].items())[:15]:
                w("    %5d  %s\n" % (v, k))
    w("\n")


def cmd_status(a):
    sts = []
    for d in a.ledger:
        status_json = os.path.join(d, "status.json")
        samples_tsv = os.path.join(d, "samples.tsv")
        if not os.path.exists(samples_tsv):
            print("no samples.tsv in %s" % d)
            continue
        rows = read_tsv(samples_tsv)
        cluster = rows[0]["cluster"] if rows else os.path.basename(os.path.normpath(d))
        st = summarize(rows, cluster)
        old = C.read_json(status_json, {}) or {}
        for k in ("su", "controller"):
            if k in old:
                st[k] = old[k]
        sts.append(st)
    # cross-cluster check: a sample must be active on at most one cluster
    if len(a.ledger) > 1:
        seen = defaultdict(list)
        for d in a.ledger:
            p = os.path.join(d, "samples.tsv")
            if os.path.exists(p):
                for r in read_tsv(p):
                    if r["state"] != "queued" or r.get("io_job"):
                        seen[r["gsm"]].append(r["cluster"])
        dup = {g: c for g, c in seen.items() if len(set(c)) > 1}
        if dup:
            print("WARNING: %d samples have work on more than one cluster, e.g. %s" % (len(dup), list(dup.items())[:5]))
    if a.json:
        print(json.dumps(sts, indent=1))
    else:
        print_status(sts)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# dry run
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


def cmd_dryrun(a):
    import tempfile

    cfg = C.load_config(a.config)
    root = a.root or tempfile.mkdtemp(prefix="ar-dry-")
    slurm = FakeSlurm(cfg["clusters"][a.cluster].get("account", "dry"))
    ctl = Controller(cfg, a.cluster, slurm, plan_path=a.plan, root=root, log=lambda m: None)
    if a.enable:
        ctl.c["enabled"] = True
    if ctl.c.get("budget") == "su" and a.su_cap is not None:
        ctl.c["su_cap"] = a.su_cap
    rows = ctl.plan
    print("dry run, cluster %s, plan %s (%d samples assigned), root %s" % (a.cluster, ctl.plan_meta.get("plan_id"),
                                                                          len(rows), root))
    # 1. what the first tick would submit
    st = ctl.tick()
    print("\nfirst tick would submit %d sbatch calls (%s):" % (len(slurm.submitted), dict(ctl.stats)))
    for s in slurm.submitted[: a.show]:
        print("  " + " ".join(s["cmd"]))
    if len(slurm.submitted) > a.show:
        print("  ... (%d more)" % (len(slurm.submitted) - a.show))
    # 2. whole-campaign projection for this cluster
    c = ctl.c
    nonhost = c.get("nonhost", cfg.get("pipeline", {}).get("nonhost", True))
    agg = defaultdict(Counter)
    cls_count = Counter()
    for r in rows:
        cl = ctl.choose_class(r, {"escalate": None})
        cls_count[cl["name"]] += 1
        k = r.get("study_wave") or r["wave"]
        agg[k]["n"] += 1
        agg[k]["core_h"] += C.compute_core_h(r["reads_available"], nonhost, cl["threads"]) * C.RETRY
        agg[k]["io_core_h"] += C.io_wall_s(r["reads_available"], r["bases"]) * int(c["io"].get("cpus", 1)) / 3600
        agg[k]["sra"] += r["est_sra_bytes"] * C.RETRY
        agg[k]["su"] += r["est_su"]
    cores = float(c.get("cores", 300))
    bw = float(c.get("bandwidth_MBps", 150)) * 1e6
    print("\nprojection (cost_model_v2 constants, STAR x%.2f, non-host %s, +20%% retries), %d cores, %.0f MB/s:" % (
        C.STAR_SPEEDUP, "on" if nonhost else "off", cores, bw / 1e6))
    print("%-14s %8s %10s %10s %10s %9s %9s %9s" % ("wave", "samples", "core-h", "io core-h", "SRA", "compute d",
                                                     "network d", "SU"))
    tot = Counter()
    for k in sorted(agg):
        g = agg[k]
        cd, nd = g["core_h"] / cores / 24, g["sra"] / bw / 86400
        print("%-14s %8d %10.0f %10.0f %10s %9.1f %9.1f %9.0f" % (k, g["n"], g["core_h"], g["io_core_h"],
                                                                 C.fmt_bytes(g["sra"]), cd, nd, g["su"]))
        tot.update(g)
    cd, nd = tot["core_h"] / cores / 24, tot["sra"] / bw / 86400
    print("%-14s %8d %10.0f %10.0f %10s %9.1f %9.1f %9.0f   -> calendar ~%.0f days (%s-bound)" % (
        "total", tot["n"], tot["core_h"], tot["io_core_h"], C.fmt_bytes(tot["sra"]), cd, nd, tot["su"], max(cd, nd),
        "network" if nd > cd else "compute"))
    print("compute classes: %s" % dict(cls_count))
    if not a.root:
        shutil.rmtree(root, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# other subcommands
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


def acquire_lease(root, ttl):
    d = os.path.join(root, "lock")
    hb = os.path.join(d, "owner.json")
    me = {"host": socket.gethostname(), "pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID"), "t": C.now()}
    try:
        os.makedirs(root, exist_ok=True)
        os.mkdir(d)
    except FileExistsError:
        cur = C.read_json(hb, {}) or {}
        if cur and C.now() - float(cur.get("t", 0)) < ttl and (cur.get("job"), cur.get("host")) != (me["job"],
                                                                                                    me["host"]):
            return False
    C.write_json(hb, me)
    return True


def release_lease(root):
    shutil.rmtree(os.path.join(root, "lock"), ignore_errors=True)


def cmd_tick(a):
    cfg = C.load_config(a.config)
    slurm = RealSlurm(cfg["clusters"][a.cluster]["account"])
    root = cfg["clusters"][a.cluster]["root"]
    if not acquire_lease(root, ttl=a.lease):
        print("another controller holds the lease (%s); not ticking" % os.path.join(root, "lock"))
        return 2
    ctl = Controller(cfg, a.cluster, slurm, plan_path=a.plan)
    st = ctl.tick()
    print_status([st])
    if a.release:
        release_lease(root)
    return 0


def cmd_controller(a):
    """Loop of ticks for the life of this Slurm job; first queue the successor so the loop outlives the time limit."""
    cfg = C.load_config(a.config)
    c = cfg["clusters"][a.cluster]
    root = c["root"]
    if c.get("enabled") is False:
        print("cluster %s is disabled in %s; the controller does nothing" % (a.cluster, cfg["_path"]))
        return 0
    slurm = RealSlurm(c["account"])
    ctl_cfg = c.get("controller", {})
    job = os.environ.get("SLURM_JOB_ID")
    limit_s = parse_elapsed(ctl_cfg.get("time", "03:55:00"))
    if job and not os.path.exists(os.path.join(root, "STOP")) and not a.once:
        opts = ["--job-name=ar-ctl-" + a.cluster, "--partition=" + ctl_cfg.get("partition", "short-cpu"),
                "--cpus-per-task=1", mem_opt(ctl_cfg.get("mem_gb", 4)), "--time=" + ctl_cfg.get(
                    "time", "03:55:00"), "--dependency=afterany:" + job,
                "--output=" + os.path.join(root, "slurm", "%x-%j.out"),
                "--export=ALL,AR_PYTHON=" + c.get("python", "python3")] + list(c.get("sbatch_extra", []))
        succ = slurm.submit(os.path.join(TEMPLATES, "controller.sbatch"), [ORCH, cfg["_path"], a.cluster], opts)
        print("successor controller queued: %s" % succ)
    t_end = time.time() + limit_s - 300
    interval = int(ctl_cfg.get("interval_s", 600))
    while True:
        if os.path.exists(os.path.join(root, "STOP")):
            print("STOP file present: exiting (no successor will be queued by the next one)")
            break
        if acquire_lease(root, ttl=3 * interval):
            try:
                ctl = Controller(C.load_config(a.config), a.cluster, slurm)
                st = ctl.tick()
                print_status([st])
            except Exception as e:  # noqa: BLE001 - keep the loop alive; the error is in the log
                print("tick failed: %r" % e)
        else:
            print("lease held by another controller; waiting")
        if a.once or time.time() + interval > t_end:
            break
        time.sleep(interval)
    return 0


def load_batch_row(a):
    with open(a.batch) as fh:
        lines = fh.read().splitlines()
    return json.loads(lines[int(a.index)])


def cmd_io_stage(a):
    from stages import Ctx, io_stage

    cfg = C.load_config(a.config)
    ctx = Ctx(cfg, C.cluster_cfg(cfg, a.cluster))
    row = load_batch_row(a)
    return io_stage(ctx, row, int(row.get("attempt", 0)))


def cmd_compute_stage(a):
    from stages import Ctx, compute_stage

    cfg = C.load_config(a.config)
    ctx = Ctx(cfg, C.cluster_cfg(cfg, a.cluster))
    row = load_batch_row(a)
    return compute_stage(ctx, row, int(row.get("attempt", 0)), int(a.threads or row.get("threads") or 4),
                         shared_genome=a.shared_genome)


def cmd_node_stage(a):
    from stages import Ctx, node_stage

    cfg = C.load_config(a.config)
    ctx = Ctx(cfg, C.cluster_cfg(cfg, a.cluster))
    with open(a.batch) as fh:
        rows = [json.loads(x) for x in fh.read().splitlines() if x.strip()]
    end = float(os.environ.get("AR_JOB_END", 0)) or (time.time() + parse_elapsed(a.time_limit) - 900)
    return node_stage(ctx, rows, a.batch, int(a.slot_cores), end, ORCH, cfg["_path"], a.cluster)


def cmd_reassign(a):
    """Two-step move of whole studies. Step `release` marks them 'none' in the plan and must be pushed to the old
    cluster and ticked there; it refuses studies with any sample past `queued` or with a job. Step `assign` gives them
    to the new cluster; run it only after the old cluster's ledger shows the studies released."""
    gses = set(a.gse.split(","))
    rows = C.read_plan(a.plan)
    meta = C.read_plan_meta(a.plan)
    if a.step == "release" and a.ledger:
        busy = [r["gsm"] for r in read_tsv(os.path.join(a.ledger, "samples.tsv"))
                if r["gse"] in gses and (r["state"] != "queued" or r.get("io_job"))]
        if busy:
            raise SystemExit("refusing: %d samples of these studies have work on their cluster (%s...)" % (
                len(busy), busy[:5]))
    n = 0
    for r in rows:
        if r["gse"] in gses:
            r["cluster"] = "none" if a.step == "release" else a.to
            n += 1
    for r in rows:
        r["runs"] = ",".join(r["runs"])
        r["run_reads"] = ",".join(str(x) for x in r["run_reads"])
    meta.setdefault("reassignments", []).append({"t": C.iso(), "step": a.step, "gse": sorted(gses), "to": a.to})
    meta = C.write_plan(a.plan, rows, meta)
    print("%s: %d samples of %d studies -> %s; new plan %s" % (a.step, n, len(gses),
                                                               "none" if a.step == "release" else a.to,
                                                               meta["plan_id"]))


def cmd_unpark(a):
    cfg = C.load_config(a.config)
    root = cfg["clusters"][a.cluster]["root"]
    L = Ledger(root)
    want = set(a.gsm.split(",")) if a.gsm else None
    n = 0
    for g, s in L.samples.items():
        if s["state"] == "parked" and (want is None and a.reason and a.reason in (s.get("last_reason") or "") or
                                       want and g in want):
            C.emit(root, "unpark", g, by="operator", note=a.note)
            n += 1
    print("unparked %d samples (applied at the next tick)" % n)


def cmd_rebuild(a):
    cfg = C.load_config(a.config)
    root = cfg["clusters"][a.cluster]["root"]
    plan = C.read_plan(a.plan or os.path.join(root, "plan", "plan.tsv.gz"), cluster=a.cluster)
    L = Ledger(root)
    L.rebuild(plan)
    L.save()
    L.write_snapshots()
    print("rebuilt %d samples from %s" % (len(L.samples), L.events_path))


def cmd_pull(a):
    """Laptop side: copy a cluster's ledger snapshots into <dest>/<cluster>/ (small files only)."""
    cfg = C.load_config(a.config)
    c = cfg["clusters"][a.cluster]
    dest = os.path.join(a.dest, a.cluster)
    os.makedirs(dest, exist_ok=True)
    files = ["samples.tsv", "runs.tsv", "studies.tsv", "status.json"]
    if a.cluster == "clipper":
        tool = os.path.expanduser(a.clipper_tool)
        for f in files:
            subprocess.run([sys.executable, tool, "get", c["root"] + "/ledger/" + f, os.path.join(dest, f)], check=False)
    else:
        if not c.get("enabled"):
            raise SystemExit("cluster %s is disabled; nothing to pull" % a.cluster)
        for f in files:
            subprocess.run(["scp", "-o", "BatchMode=yes", "%s:%s/ledger/%s" % (c["ssh"], c["root"], f),
                            os.path.join(dest, f)], check=False)
    print("pulled %s -> %s" % (a.cluster, dest))


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd")
    default_cfg = os.path.join(C.HERE, "config.json")

    q = sp.add_parser("plan")
    q.add_argument("--config", default=default_cfg)
    q.add_argument("--universe", required=True)
    q.add_argument("--runs", nargs="+", required=True, help="runs.parquet runs_extra.parquet runs_extra2.parquet")
    q.add_argument("--out", required=True)
    q.add_argument("--only-gse", help="comma list: a plan restricted to these studies (tests, smoke runs)")
    q.add_argument("--only-gsm", help="comma list: a plan restricted to these samples (smoke runs)")

    q = sp.add_parser("dryrun")
    q.add_argument("--config", default=default_cfg)
    q.add_argument("--plan", required=True)
    q.add_argument("--cluster", default="clipper")
    q.add_argument("--root", help="keep the dry ledger here (default: a temp dir, removed)")
    q.add_argument("--show", type=int, default=6)
    q.add_argument("--enable", action="store_true", help="treat the cluster as enabled (to see Anvil's submissions)")
    q.add_argument("--su-cap", type=float, help="dry-run SU cap for an SU-budgeted cluster")

    for name in ("tick", "controller"):
        q = sp.add_parser(name)
        q.add_argument("--config", default=default_cfg)
        q.add_argument("--cluster", required=True)
        if name == "tick":
            q.add_argument("--plan")
            q.add_argument("--lease", type=float, default=1800)
            q.add_argument("--release", action="store_true")
        else:
            q.add_argument("--once", action="store_true")

    q = sp.add_parser("status")
    q.add_argument("ledger", nargs="+", help="ledger dirs (<root>/ledger on a cluster, or pulled copies)")
    q.add_argument("--json", action="store_true")

    q = sp.add_parser("pull")
    q.add_argument("--config", default=default_cfg)
    q.add_argument("--cluster", required=True)
    q.add_argument("--dest", default="campaign_ledgers")
    q.add_argument("--clipper-tool", default="~/.claude/skills/clipper/clipper.py")

    q = sp.add_parser("reassign")
    q.add_argument("--plan", required=True)
    q.add_argument("--gse", required=True)
    q.add_argument("--step", choices=("release", "assign"), required=True)
    q.add_argument("--to", choices=("clipper", "anvil"))
    q.add_argument("--ledger", help="pulled ledger dir of the old cluster (release refuses busy studies)")

    q = sp.add_parser("unpark")
    q.add_argument("--config", default=default_cfg)
    q.add_argument("--cluster", required=True)
    q.add_argument("--gsm")
    q.add_argument("--reason", help="unpark every parked sample whose reason contains this text")
    q.add_argument("--note", default="")

    q = sp.add_parser("rebuild")
    q.add_argument("--config", default=default_cfg)
    q.add_argument("--cluster", required=True)
    q.add_argument("--plan")

    for name in ("io-stage", "compute-stage", "node-stage"):
        q = sp.add_parser(name)
        q.add_argument("--config", default=default_cfg)
        q.add_argument("--cluster", required=True)
        q.add_argument("--batch", required=True)
        if name != "node-stage":
            q.add_argument("--index", required=True)
        if name == "compute-stage":
            q.add_argument("--threads", type=int)
            q.add_argument("--shared-genome", action="store_true")
        if name == "node-stage":
            q.add_argument("--slot-cores", default=112)
            q.add_argument("--time-limit", default="24:00:00")

    a = p.parse_args(argv)
    fn = {"plan": cmd_plan, "dryrun": cmd_dryrun, "tick": cmd_tick, "controller": cmd_controller,
          "status": cmd_status, "pull": cmd_pull, "reassign": cmd_reassign, "unpark": cmd_unpark,
          "rebuild": cmd_rebuild, "io-stage": cmd_io_stage, "compute-stage": cmd_compute_stage,
          "node-stage": cmd_node_stage}.get(a.cmd)
    if fn is None:
        p.print_help()
        return 1
    return fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
