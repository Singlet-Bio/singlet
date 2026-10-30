"""The per-cluster ledger: an append-only event log folded into one row per sample (and per run).

Layout under a cluster's campaign root (all on that cluster's project storage):

    inbox/            events written by jobs, one small JSON file each (atomic rename); only the controller reads it
    events.jsonl      append-only log, single writer = the controller holding lock/; the source of truth
    state.json        fold of events.jsonl up to a byte offset (a cache: `rebuild` recreates it from the log)
    ledger/samples.tsv, ledger/runs.tsv, ledger/studies.tsv, ledger/status.json
                      snapshots for people and for the laptop (`pull`), rewritten each tick

States of a sample:

    queued -> downloading -> screened -> downloaded -> running -> done
         \\-> failed (n failures so far; the controller requeues once, then parks) -> queued | parked
    parked is terminal until an operator `unpark`s it.

The screen runs on the first 1M read pairs of every run *before* the full prefetch, so a run that is dropped never costs
bandwidth; the I/O stage then fetches only the kept runs (atlas-rerun-plan §5/§6 item 3).
"""

import json
import os

from common import iso, read_json, write_json

STATES = ("queued", "downloading", "screened", "downloaded", "running", "done", "failed", "parked")
TERMINAL = ("done", "parked")
ACTIVE = ("downloading", "screened", "downloaded", "running")

SAMPLE_COLUMNS = [
    "gsm", "gse", "wave", "cluster", "state", "attempt", "failures", "last_reason", "io_job", "compute_job",
    "node_job", "compute_class", "threads", "partition", "t_submitted", "t_io_start", "t_screened", "t_downloaded",
    "t_compute_start", "t_done", "io_s", "head_s", "prefetch_s", "decode_s", "screen_s", "pool_s", "compute_s",
    "compute_maxrss_gb", "io_maxrss_gb", "reads_available", "reads_used", "pileup_total_reads", "runs_planned", "runs_kept",
    "runs_dropped", "chemistry", "sra_bytes", "staged_bytes", "binary_sha256", "out_dir", "private_dir",
    "manifest_sha256", "out_bytes", "n_cells", "node", "est_wall_s", "est_su",
]
RUN_COLUMNS = ["gsm", "run", "planned_reads", "keep", "reason", "chemistry", "best_rate", "total_reads", "sra_bytes",
               "prefetch_s", "decode_s"]


def new_sample(plan_row):
    return {
        "gsm": plan_row["gsm"], "gse": plan_row["gse"], "wave": plan_row.get("study_wave") or plan_row["wave"],
        "sample_wave": plan_row["wave"], "cluster": plan_row["cluster"],
        "state": "queued", "attempt": 0, "failures": 0, "reads_available": plan_row.get("reads_available"),
        "runs_planned": ",".join(plan_row.get("runs") or []), "est_wall_s": plan_row.get("est_wall_s"),
        "est_su": plan_row.get("est_su"), "history": [],
    }


class Ledger:
    def __init__(self, root):
        self.root = root
        self.events_path = os.path.join(root, "events.jsonl")
        self.state_path = os.path.join(root, "state.json")
        st = read_json(self.state_path, None) or {}
        self.offset = st.get("offset", 0)
        self.samples = st.get("samples", {})
        self.runs = st.get("runs", {})
        self.studies = st.get("studies", {})
        self.campaign = st.get("campaign", {})

    # -- folding ----------------------------------------------------------------------------------------------------
    def ensure(self, plan_rows):
        for r in plan_rows:
            if r["gsm"] not in self.samples:
                self.samples[r["gsm"]] = new_sample(r)

    def ingest_inbox(self, max_files=200000):
        """Move every inbox event into events.jsonl (append, fsync), then delete the inbox files. Returns n."""
        inbox = os.path.join(self.root, "inbox")
        if not os.path.isdir(inbox):
            return 0
        names = sorted(n for n in os.listdir(inbox) if n.endswith(".json") and not n.startswith("."))[:max_files]
        if not names:
            return 0
        lines = []
        for n in names:
            try:
                with open(os.path.join(inbox, n)) as fh:
                    txt = fh.read().strip()
                json.loads(txt)
                lines.append(txt)
            except (OSError, ValueError):
                bad = os.path.join(self.root, "inbox.bad")
                os.makedirs(bad, exist_ok=True)
                os.replace(os.path.join(inbox, n), os.path.join(bad, n))
        self.append(lines, raw=True)
        for n in names:
            try:
                os.remove(os.path.join(inbox, n))
            except OSError:
                pass
        return len(lines)

    def append(self, events, raw=False):
        if not events:
            return
        with open(self.events_path, "a", newline="\n") as fh:
            for e in events:
                fh.write((e if raw else json.dumps(e, sort_keys=True, default=str)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def fold(self):
        """Apply events.jsonl from the saved offset."""
        if not os.path.exists(self.events_path):
            return 0
        n = 0
        with open(self.events_path, "rb") as fh:
            fh.seek(self.offset)
            for raw in fh:
                if not raw.endswith(b"\n"):
                    break  # a partial last line: read it next time
                self.offset += len(raw)
                try:
                    self.apply(json.loads(raw.decode("utf-8")))
                    n += 1
                except ValueError:
                    continue
        return n

    def apply(self, e):
        ev = e.get("event")
        if ev == "study_handoff":
            self.studies.setdefault(e["gse"], {}).update({"handoff": e.get("path"), "t_handoff": e["t"],
                                                         "n_done": e.get("n_done"), "n_parked": e.get("n_parked")})
            return
        if ev in ("campaign_paused", "campaign_resumed", "budget_stop"):
            self.campaign[ev] = e["t"]
            return
        gsm = e.get("gsm")
        s = self.samples.get(gsm)
        if s is None:
            return  # not in this cluster's plan (or plan changed): kept in the log, ignored in the fold
        att = e.get("attempt")
        if att is not None and ev not in ("submitted", "requeued", "unpark") and int(att) < int(s.get("attempt", 0)):
            return  # a late event of an older attempt (requeued or lost job): never overrides the newer attempt
        t = e["t"]
        s.setdefault("history", []).append([round(t), ev, e.get("reason") or e.get("stage") or ""])
        del s["history"][:-20]
        if ev == "submitted":
            st = e.get("stage")
            s["attempt"] = int(e.get("attempt", s.get("attempt", 0)))
            if st == "io":
                s["io_job"] = e.get("job")
                s["t_submitted"] = t
                s["state"] = "queued" if s["state"] in ("queued", "failed") else s["state"]
            elif st == "compute":
                s["compute_job"] = e.get("job")
            elif st == "node":
                s["node_job"] = e.get("job")
            for k in ("compute_class", "threads", "partition"):
                if e.get(k) is not None:
                    s[k] = e[k]
        elif ev == "downloading":
            s["state"] = "downloading"
            s["t_io_start"] = t
        elif ev == "screened":
            s["state"] = "screened"
            s["t_screened"] = t
            s["runs_kept"] = ",".join(e.get("runs_kept") or [])
            s["runs_dropped"] = ",".join(e.get("runs_dropped") or [])
            s["chemistry"] = e.get("chemistry")
            s["head_s"] = e.get("head_s")
            for r in e.get("runs") or []:
                key = gsm + ":" + r["run"]
                rr = self.runs.setdefault(key, {"gsm": gsm, "run": r["run"]})
                rr.update({k: r.get(k) for k in ("keep", "reason", "chemistry", "best_rate", "planned_reads")})
        elif ev == "downloaded":
            s["state"] = "downloaded"
            s["t_downloaded"] = t
            for k in ("sra_bytes", "staged_bytes", "prefetch_s", "decode_s", "screen_s", "pool_s", "io_s",
                      "io_maxrss_gb", "reads_used"):
                if e.get(k) is not None:
                    s[k] = e[k]
            if e.get("runs_kept"):
                s["runs_kept"] = ",".join(e["runs_kept"])
            for r in e.get("runs") or []:
                key = gsm + ":" + r["run"]
                rr = self.runs.setdefault(key, {"gsm": gsm, "run": r["run"]})
                rr.update({k: r.get(k) for k in ("total_reads", "sra_bytes", "prefetch_s", "decode_s") if k in r})
        elif ev == "running":
            s["state"] = "running"
            s["t_compute_start"] = t
            s["node"] = e.get("host")
            if e.get("threads"):
                s["threads"] = e["threads"]
        elif ev == "done":
            s["state"] = "done"
            s["t_done"] = t
            for k in ("compute_s", "compute_maxrss_gb", "pileup_total_reads", "binary_sha256", "out_dir", "private_dir",
                      "manifest_sha256", "out_bytes", "n_cells"):
                if e.get(k) is not None:
                    s[k] = e[k]
        elif ev in ("failed", "lost"):
            if s["state"] in TERMINAL:
                return
            s["state"] = "failed"
            s["failures"] = int(s.get("failures", 0)) + 1
            s["last_reason"] = "%s:%s" % (e.get("stage", "?"), e.get("reason", "?"))
            s["t_failed"] = t
        elif ev == "parked":
            s["state"] = "parked"
            s["last_reason"] = "%s:%s" % (e.get("stage", "?"), e.get("reason", "?"))
            s["t_parked"] = t
        elif ev == "requeued":
            # from_stage=compute: the staged reads are kept and only the compute stage runs again
            s["state"] = "downloaded" if e.get("from_stage") == "compute" else "queued"
            s["attempt"] = int(e.get("attempt", int(s.get("attempt", 0)) + 1))
            s["escalate"] = e.get("escalate")
            for k in ("io_job", "compute_job", "node_job"):
                s.pop(k, None)
        elif ev == "released":
            if s["state"] in ("running", "downloaded"):
                s["state"] = "downloaded"
                s.pop("compute_job" if e.get("stage") == "compute" else "node_job", None)
        elif ev == "unpark":
            s["state"] = "downloaded" if e.get("from_stage") == "compute" else "queued"
            s["failures"] = 0
            s["attempt"] = int(s.get("attempt", 0)) + 1
            for k in ("io_job", "compute_job", "node_job", "escalate"):
                s.pop(k, None)

    # -- persistence ------------------------------------------------------------------------------------------------
    def save(self):
        write_json(self.state_path, {"offset": self.offset, "samples": self.samples, "runs": self.runs,
                                     "studies": self.studies, "campaign": self.campaign, "saved": iso()})

    def write_snapshots(self, status=None):
        d = os.path.join(self.root, "ledger")
        os.makedirs(d, exist_ok=True)
        _write_tsv(os.path.join(d, "samples.tsv"), SAMPLE_COLUMNS, self.samples.values())
        _write_tsv(os.path.join(d, "runs.tsv"), RUN_COLUMNS, self.runs.values())
        studies = study_table(self.samples, self.studies)
        _write_tsv(os.path.join(d, "studies.tsv"), list(STUDY_COLUMNS), studies)
        if status is not None:
            write_json(os.path.join(d, "status.json"), status)

    def rebuild(self, plan_rows):
        self.offset, self.samples, self.runs, self.studies, self.campaign = 0, {}, {}, {}, {}
        self.ensure(plan_rows)
        self.fold()


STUDY_COLUMNS = ("gse", "wave", "cluster", "n", "done", "parked", "failed", "active", "queued", "handoff")


def study_table(samples, studies):
    agg = {}
    for s in samples.values():
        a = agg.setdefault(s["gse"], {"gse": s["gse"], "wave": s["wave"], "cluster": s["cluster"], "n": 0, "done": 0,
                                      "parked": 0, "failed": 0, "active": 0, "queued": 0})
        a["n"] += 1
        st = s["state"]
        if st in ("done", "parked", "failed", "queued"):
            a[st] += 1
        else:
            a["active"] += 1
    for g, a in agg.items():
        a["handoff"] = (studies.get(g) or {}).get("handoff", "")
    return sorted(agg.values(), key=lambda a: a["gse"])


def _write_tsv(path, cols, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(_cell(r.get(c)) for c in cols) + "\n")
    os.replace(tmp, path)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "%.3f" % v if abs(v) >= 1e5 else "%.6g" % v  # epoch times keep their seconds
    return str(v).replace("\t", " ").replace("\n", " ")


def read_tsv(path):
    rows = []
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(cols, line.rstrip("\n").split("\t"))))
    return rows
