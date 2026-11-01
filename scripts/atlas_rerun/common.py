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


# Peak-RSS model of the largest process of one `--engine gpu` sample (the STAR child; GPU pileup on): intercept + slope * M reads.
# route "fq" (staged .1fq / temp FASTQ): 17.5 GB + 0.39 GB per M reads, fitted to four measured runs (S8, 1 H100, quiet node):
#   11.3 M -> 22.8 GB, 13.2 M -> 22.1, 66.6 M -> 43.5, 263.4 M -> 120.4 (all within 3%). Most of the slope is a transient spike in the FASTQ read pass.
# route "1fg2" (table-fed prefill from the staged .1fg2, no temp FASTQ, #126): no spike. 17.5 GB + 0.26 GB per M reads from two points (S8: 11.3 M -> 22.2 GB, 263.4 M -> 84.4 GB; the slope is an estimate, applies once #126 is merged;
#   scripts/atlas_rerun/multi_drain.py logs the measured peak of every sample as `peak_rss_gb`: refit from the `done` events).
# Node-local temp bytes per sample: the old route wrote ~320 B per read of temp FASTQ; the .1fg2 route writes none. Outputs add ~15 B per read (S8; a nuclei sample with 97% unmapped reads writes ~290 B per unmapped read, 34 GB at 119 M reads: the 20 GB scratch margin and the measured `scratch_bytes` of each done event cover it).
MEM_MODEL = {"fq": {"base_gb": 17.5, "gb_per_m": 0.39, "tmp_gb_per_m": 0.335, "tmp_base_gb": 2.0, "ovh_base_gb": 4.0, "ovh_gb_per_m": 0.05},
             "1fg2": {"base_gb": 17.5, "gb_per_m": 0.26, "tmp_gb_per_m": 0.015, "tmp_base_gb": 2.0, "ovh_base_gb": 4.0, "ovh_gb_per_m": 0.03}}


def gpu_peak_gb(reads, route="fq"):
    """Largest-process RSS of one `--engine gpu` sample, GB. Beyond 263 M reads it is an extrapolation."""
    m = MEM_MODEL[route]
    return m["base_gb"] + m["gb_per_m"] * reads / 1e6


GENOME_SHARED_GB = 17.7   # the GRCh38 index attached from shared memory (`--genome-shared on`): resident once per node, counted in every process's RSS


def gpu_cgroup_gb(reads, route="fq"):
    """Largest process (gpu_peak_gb) plus the rest of the sample's share of the job (parent, pinned buffers): what a sample needs when it runs alone.
    S8 measured the whole-job cgroup peak of the 263.4 M-read sample: 180.9 GB (FASTQ route) and 100.6 GB (.1fg2 route, anon 85.0 + page cache 15.4). The
    cgroup figure includes reclaimable page cache (temp FASTQ, staged reads), which the 85% admission margin leaves room for; the model is for anon + shmem.
    Gate part 2 (13 samples, 4 concurrent, publish dry-run): anon+shmem of the whole job peaked at 80 GB while page cache filled the 370 GB limit."""
    m = MEM_MODEL[route]
    return gpu_peak_gb(reads, route) + m["ovh_base_gb"] + m["ovh_gb_per_m"] * reads / 1e6


def gpu_private_gb(reads, route="fq", shared=True):
    """What one more sample adds to a job that already runs samples. With the genome attached from shared memory (shared=True) every process's RSS
    contains the one shared copy (GENOME_SHARED_GB), which the job holds once: the sample adds its RSS minus that. Without sharing, all of it."""
    return gpu_cgroup_gb(reads, route) - (GENOME_SHARED_GB if shared else 0.0)


def gpu_tmp_gb(reads, route="fq"):
    """Node-local scratch (/tmp) one sample holds at its peak: temp FASTQ (route "fq") plus outputs, GB."""
    m = MEM_MODEL[route]
    return m["tmp_base_gb"] + m["tmp_gb_per_m"] * reads / 1e6


def memory_tier(reads):
    """summary.json's memory_tier of a sample from its read count, not from the process that ran it (the field is a member of the .s1, so it must not
    depend on the engine or the node): the singlet thresholds on the expected peak resident set, <8 GB Small, <64 Medium, <256 Large, else XLarge."""
    gb = gpu_peak_gb(reads)
    return "Small" if gb < 8 else "Medium" if gb < 64 else "Large" if gb < 256 else "XLarge"


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


def whitelist_dir_for(binary):
    """Where singlet looks for its barcode whitelists: <real path of the binary>/../whitelists (it resolves /proc/self/exe)."""
    return os.path.join(os.path.dirname(os.path.realpath(binary)), "..", "whitelists")


def require_whitelists(binary, names=("3M-february-2018.txt", "737K-august-2016.txt")):
    """Fail fast when the engine cannot find its whitelists. Without <binary dir>/../whitelists the run does not stop: it silently takes the sorted-BAM
    pileup (pileup_threads > 1) instead of the stream / GPU pileup, a different and slower path with different memory and thread behaviour
    (S8, 2026-10-31). $S9_NO_WL_CHECK=1 skips the check (tests with a fake engine). Raises SystemExit(3)."""
    if os.environ.get("S9_NO_WL_CHECK"):
        return
    d = whitelist_dir_for(binary)
    missing = [n for n in names if not os.path.isfile(os.path.join(d, n))]
    if missing:
        raise SystemExit("[atlas_rerun] FATAL: %s not found in %s (the engine %s resolves <binary dir>/../whitelists): the campaign would silently run "
                         "the sorted-BAM pileup. Put the binary where ../whitelists exists, or symlink the directory." % (", ".join(missing), os.path.normpath(d), binary))


def species_check(log_path):
    """-> dict(verdict, line) from the engine's "[species-check] ... AGREE|DISAGREE|UNKNOWN" log line (singlet logs it when $SINGLET_SPECIES_FILTERS is set and a genome was
    given), or None. Species detection only confirms the catalogue organism: a DISAGREE is recorded and flagged, never acted on."""
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                if line.startswith("[species-check]"):
                    return {"verdict": line.strip().rsplit(" ", 1)[-1], "line": line.strip()[:300]}
    except OSError:
        pass
    return None


class OrganismUnmapped(Exception):
    """The catalogue organism of a sample maps to no reference: the sample cannot be aligned (and must not fall back to species auto-detection).
    .reason is the failed_reason the worker records: "organism_unmapped" or "mixed_species_unsupported"."""

    def __init__(self, msg, reason="organism_unmapped"):
        super().__init__(msg)
        self.reason = reason


# catalogue organism strings are ';'-separated lists of whatever GEO says; these tokens name no organism that was sequenced
ORGANISM_NOISE = {"", "blank sample", "synthetic construct", "unidentified", "unclassified sequences"}
ORGANISM_ALIASES = {"homo": "Homo sapiens", "human": "Homo sapiens", "mouse": "Mus musculus", "mus": "Mus musculus"}
# "Homo sapiens/Mus musculus xenograft" names both
ORGANISM_SPLIT = {"homo sapiens/mus musculus xenograft": ("Homo sapiens", "Mus musculus")}
# Organisms next to human that are NOT mammals (pathogens, plasmids, birds, fish, reptiles, amphibians, insects): their reads are picked up by the non-host screen and the
# sample is aligned to the human (or mouse) reference alone. Every other organism (primates, rodents, pigs, ... and any name not listed here) is "another mammal or unknown":
# aligning those reads to GRCh38 alone would give confident wrong counts.
NON_MAMMAL_MARKERS = ("virus", "phage", "plasmid", "herpes", "coronavirus", "bacteri", "vector", "construct")
NON_MAMMAL_SPECIES = {
    "Escherichia coli", "Toxoplasma gondii", "Plasmodium vivax", "Plasmodium falciparum", "Encephalitozoon intestinalis", "Gallus gallus", "Danio rerio",
    "Melopsittacus undulatus", "Ctenopharyngodon idella", "Oreochromis mossambicus", "Ambystoma mexicanum", "Serinus canaria", "Channa argus", "Pantherophis guttatus",
    "Anas platyrhynchos", "Pogona vitticeps", "Trachemys scripta elegans", "Drosophila melanogaster", "Xenopus laevis", "Xenopus tropicalis", "Caenorhabditis elegans",
    "Saccharomyces cerevisiae", "Candida albicans", "Staphylococcus aureus", "Salmonella enterica", "Mycobacterium tuberculosis"}
CORE_ORGANISMS = frozenset({"Homo sapiens", "Mus musculus"})


def _organism_tokens(organism):
    out = set()
    for t in (organism or "").split(";"):
        t = t.strip()
        if t.lower() in ORGANISM_NOISE:
            continue
        if t.lower() in ORGANISM_SPLIT:
            out.update(ORGANISM_SPLIT[t.lower()])
        else:
            out.add(ORGANISM_ALIASES.get(t.lower(), t))
    return out


def _is_non_mammal(t):
    return t in NON_MAMMAL_SPECIES or any(m in t.lower() for m in NON_MAMMAL_MARKERS)


def organism_set(organism):
    """-> frozenset of organism names in a catalogue organism string: 'Mus musculus; Homo sapiens' and 'Homo sapiens; Mus musculus' are the same set,
    'blank sample; Homo sapiens' is {'Homo sapiens'}, 'Homo; Homo sapiens' is {'Homo sapiens'}."""
    return frozenset(_organism_tokens(organism))


def reference_status(organism, refs):
    """-> (reference name, None) or (None, reason). Order: an exact key of `refs`; otherwise the organism set with the non-mammal tokens (pathogens, plasmids, birds,
    fish, ...) removed is compared with the set of every key. Another mammal next to human or mouse (or a name that is not known to be a non-mammal) gives
    (None, "mixed_species_unsupported"); no human or mouse at all, or a set no key covers, gives (None, "organism_unmapped"). The `*` default ("auto") is never used: the
    worker does not rely on species auto-detection (S14: its Bloom filters were not installed, 6 of 12 pilot runs came back 'unknown')."""
    refs = refs or {}
    o = (organism or "").strip()
    if o in refs and refs[o] != "auto":
        return refs[o], None
    tokens = _organism_tokens(o)
    core = frozenset(tokens & CORE_ORGANISMS)
    if any(not _is_non_mammal(t) for t in tokens - CORE_ORGANISMS):
        return None, "mixed_species_unsupported"
    if not core:
        return None, "organism_unmapped"
    for k, v in refs.items():
        if k != "*" and v != "auto" and organism_set(k) == core:
            return v, None
    return None, "organism_unmapped"


def resolve_reference(organism, refs):
    """The reference directory name (a value of config `references`) for a catalogue organism string, or None (see reference_status for why)."""
    return reference_status(organism, refs)[0]


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
    "size_class", "cluster", "protocol_inferred",
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
