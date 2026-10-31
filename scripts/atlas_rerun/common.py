"""Shared pieces of the atlas re-run orchestrator: config, plan file, events, cost model.

Standard library only (Python >= 3.8), so it runs with the system python3 of a compute node on Clipper or Anvil.
"""

import gzip
import hashlib
import itertools
import json
import math
import os
import socket
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def node_mem_gb():
    """Memory of this node in GB: the job's allocation when Slurm sets one, else the machine's MemTotal."""
    m = os.environ.get("SLURM_MEM_PER_NODE")
    if m and m.isdigit():
        return int(m) / 1024.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError):
        pass
    return 0.0


def gpu_peak_gb(reads):
    """Largest-process RSS of one `--engine gpu` sample (the STAR child; GPU pileup on): 17.5 GB + 0.39 GB per million reads.
    Fitted to four measured runs (S8, 1 H100, quiet node): 11.3 M -> 22.8 GB, 13.2 M -> 22.1, 66.6 M -> 43.5, 263.4 M -> 120.4 (all within 3%).
    Beyond 263 M reads it is an extrapolation."""
    return 17.5 + 0.39 * reads / 1e6


def gpu_job_mem_gb(reads, margin=1.2, extra=20, step=8):
    """What a job needs to run a sample of `reads` reads: the largest process with `margin`, plus the parent process, /dev/shm temp files and
    page cache (`extra` GB), rounded up to `step`. A 128 GB job holds samples up to ~190 M reads."""
    return int(math.ceil((margin * gpu_peak_gb(reads) + extra) / step) * step)


GPU_EXCLUSIVE_GB = 700   # a worker with at least this much memory owns a whole node (--exclusive, --mem=0): the "too big for a shared worker" class


def job_slots(node_gb, node_cores, threads, peak_gb=33.4, shared_gb=17.8, fill=0.9):
    """Concurrent sample jobs one node holds. Memory: the shared genome index is resident once per node (shared_gb), every
    job adds its private peak (peak_gb: deep GSM7036403 with the shared index measured 33.4 GB, index 17.8 GB), and
    `fill` of the node is usable, so 512 GB gives 13. Cores: threads per job. The smaller of the two; at least 1."""
    by_mem = int((node_gb * fill - shared_gb) // peak_gb) if node_gb else 0
    by_cpu = int(node_cores // max(1, threads))
    return max(1, min(by_mem, by_cpu)) if by_mem > 0 else max(1, by_cpu)


def staged_reads(sdir):
    """True when a sample's staged reads are on disk: the pooled sample.1fq or the per-run <run>.1fq files."""
    if os.path.exists(os.path.join(sdir, "sample.1fq")):
        return True
    try:
        return any(f.endswith(".1fq") for f in os.listdir(sdir))
    except OSError:
        return False

# ── cost model (bench/atlas_rerun/cost_model_v2.py, measured on PBMC 3' v3, Xeon Gold 6230R, 4 threads) ──────────────
FIXED_S = 300.0  # genome/reference load, index, export: seconds per sample
SLOPE_S_PER_M = {True: 16.8, False: 12.3}  # seconds per million read pairs, non-host screen on / off
MAPPING_S_PER_M = 10.8  # the STAR mapping share of that slope
STAR_SPEEDUP = 1.54  # PR #34 (sequence cache, lazy window table, huge pages, PGO)
PACK = 1.15  # 4 concurrent samples per 16 cores ran 15% slower than one alone
RETRY = 1.20  # retry margin used for budgets only
# wall time relative to 4 threads, from core-h per sample 1.24 (4t) / 1.59 (8t) / ~2.1 (16t) (atlas-rerun-plan §3)
THREAD_WALL = {4: 1.0, 8: 1.59 / 1.24 / 2.0, 16: 2.1 / 1.24 / 4.0}
SRA_BYTES_PER_BASE = 0.44
STREAM_BPS = 20e6  # one prefetch stream under aggregate load
DECODE_S_PER_M = 1.0  # local `singlet download <.sra>`
MEDIAN_READS = 250e6  # GEX universe median, used for samples whose depth ENA does not report
MEDIAN_BASES_PER_READ = 2 * 150.0

ANVIL_NODE_CORES = 128
ANVIL_NODE_BUSY = 112  # 28 four-thread samples; the rest of the cores carry I/O and the node's own work


def compute_wall_s(reads, nonhost=True, threads=4, speedup=STAR_SPEEDUP, packed=True):
    """Predicted wall seconds of the compute stage for one sample of `reads` read pairs."""
    u = (reads if reads and reads > 0 else MEDIAN_READS) / 1e6
    slope = SLOPE_S_PER_M[bool(nonhost)] - MAPPING_S_PER_M * (1.0 - 1.0 / speedup)
    wall = FIXED_S + slope * u
    wall *= THREAD_WALL.get(threads, 4.0 / threads)
    return wall * (PACK if packed else 1.0)


def compute_core_h(reads, nonhost=True, threads=4):
    return threads * compute_wall_s(reads, nonhost, threads) / 3600.0


def sra_bytes(reads, bases):
    r = reads if reads and reads > 0 else MEDIAN_READS
    b = bases if bases and bases > 0 else r * MEDIAN_BASES_PER_READ
    return b * SRA_BYTES_PER_BASE


def io_wall_s(reads, bases):
    r = reads if reads and reads > 0 else MEDIAN_READS
    return sra_bytes(reads, bases) / STREAM_BPS + DECODE_S_PER_M * r / 1e6


def anvil_su(reads, nonhost=True, threads=4):
    """SU on whole Anvil nodes, charged all 128 cores for 112 busy ones, retries included."""
    return compute_core_h(reads, nonhost, threads) * RETRY * ANVIL_NODE_CORES / ANVIL_NODE_BUSY


# ── config ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    cfg["_path"] = os.path.abspath(path)
    return cfg


def cluster_cfg(cfg, cluster):
    if cluster not in cfg["clusters"]:
        raise SystemExit("unknown cluster %r (config has %s)" % (cluster, ", ".join(cfg["clusters"])))
    c = dict(cfg["clusters"][cluster])
    c["name"] = cluster
    return c


# ── plan file: one row per GEX sample ────────────────────────────────────────────────────────────────────────────

PLAN_COLUMNS = [
    "priority", "gsm", "gse", "wave", "study_wave", "study_rank", "sample_rank", "organism", "reference", "reads_available",
    "depth_unknown", "bases", "n_runs", "runs", "run_reads", "est_wall_s", "est_core_h", "est_su", "est_sra_bytes",
    "size_class", "cluster",
]
INT_COLS = {"priority", "study_rank", "sample_rank", "reads_available", "depth_unknown", "bases", "n_runs",
            "est_sra_bytes"}
FLOAT_COLS = {"est_wall_s", "est_core_h", "est_su"}


def _open_text(path, mode="rt", gz=None):
    if path.endswith(".gz") if gz is None else gz:
        return gzip.open(path, mode, encoding="utf-8", newline="\n")
    return open(path, mode, encoding="utf-8", newline="\n")


def write_plan(path, rows, meta):
    """rows: list of dicts with PLAN_COLUMNS. A sidecar <path>.json carries the policy and the plan id."""
    tmp = path + ".tmp"
    with _open_text(tmp, "wt", gz=path.endswith(".gz")) as fh:
        fh.write("\t".join(PLAN_COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join("" if r.get(c) is None else str(r[c]) for c in PLAN_COLUMNS) + "\n")
    os.replace(tmp, path)
    meta = dict(meta)
    meta["plan_sha256"] = sha256_file(path)
    meta["plan_id"] = meta["plan_sha256"][:12]
    write_json(plan_meta_path(path), meta)
    return meta


def plan_meta_path(path):
    base = path[:-3] if path.endswith(".gz") else path
    return base + ".json"


def read_plan(path, cluster=None):
    rows = []
    with _open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            vals = line.rstrip("\n").split("\t")
            r = dict(zip(header, vals))
            for c in INT_COLS:
                if c in r:
                    r[c] = int(float(r[c])) if r[c] not in ("", "nan", "None") else None
            for c in FLOAT_COLS:
                if c in r:
                    r[c] = float(r[c]) if r[c] not in ("", "nan", "None") else None
            r["runs"] = [x for x in r.get("runs", "").split(",") if x]
            r["run_reads"] = [int(x) if x else 0 for x in r.get("run_reads", "").split(",")] if r.get("run_reads") else []
            if cluster is None or r.get("cluster") == cluster:
                rows.append(r)
    rows.sort(key=lambda r: r["priority"])
    return rows


def read_plan_meta(path):
    p = plan_meta_path(path)
    return read_json(p) if os.path.exists(p) else {}


# ── small io helpers ─────────────────────────────────────────────────────────────────────────────────────────────


def now():
    return time.time()


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if t is None else t))


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    """Atomic within a directory (write a temp file, then rename), LF line ends."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".%s.%s.%d.tmp" % (os.path.basename(path), socket.gethostname(), os.getpid()))
    with open(tmp, "w", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def sha256_file(path, block=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def fmt_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1000 or unit == "PB":
            return "%.1f %s" % (n, unit)
        n /= 1000.0
    return "%.1f PB" % n


def fmt_dur(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "-"
    s = float(s)
    if s < 120:
        return "%.0fs" % s
    if s < 7200:
        return "%.0fm" % (s / 60)
    if s < 172800:
        return "%.1fh" % (s / 3600)
    return "%.1fd" % (s / 86400)


# ── events: the only thing jobs write ─────────────────────────────────────────────────────────────────────────────
#
# A job never edits shared state. It writes one small JSON file per event into <root>/inbox/ with write-to-temp +
# rename (atomic on NFS and Lustre within one directory), named so that names never collide across nodes or requeues.
# The cluster's controller is the only process that reads the inbox, appends the events to <root>/events.jsonl (its
# single writer) and deletes the inbox files it has folded.


_SEQ = itertools.count()  # orders this process's events when two land in the same clock tick


def emit(root, event, gsm=None, **fields):
    t_ns = time.time_ns()
    rec = {"t": round(t_ns / 1e9, 3), "ts": iso(), "event": event}
    if gsm:
        rec["gsm"] = gsm
    job = os.environ.get("SLURM_JOB_ID")
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        job = "%s_%s" % (os.environ["SLURM_ARRAY_JOB_ID"], os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if job and "job" not in fields:
        rec["job"] = job
    rec["host"] = socket.gethostname().split(".")[0]
    rec.update({k: v for k, v in fields.items() if v is not None})
    inbox = os.path.join(root, "inbox")
    os.makedirs(inbox, exist_ok=True)
    # zero-padded ns time, then a per-process sequence: inbox files sort in emission order even within one tick
    name = "%020d-%08d-%s-%s-%d-%s.json" % (t_ns, next(_SEQ), (gsm or "campaign"), rec["host"], os.getpid(), event)
    tmp = os.path.join(inbox, "." + name + ".tmp")
    with open(tmp, "w", newline="\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    os.replace(tmp, os.path.join(inbox, name))
    return rec
