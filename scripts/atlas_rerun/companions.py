#!/usr/bin/env python3
"""Companion libraries (ADT, HTO, CRISPR, V(D)J) of a GEX sample: which they are, and fetching their reads as FASTQ.

The companion table is `companions.tsv` of `companion_libs.py` (gex_gsm, lib_type, lib_gsm, gse, stem, lib_runs, lib_reads). The GEX
sample is staged exactly as before; its companions land inside its own ready dir, `companions/<lib_type>/<lib_gsm>/<run>_R1.fastq.gz`
and `_R2.fastq.gz`, so the claim / reap / delete / stage-cap accounting of stagequeue covers them (S14's contract). A companion that
cannot be fetched is recorded as `status: failed` with a reason and never fails the GEX sample.
"""
import gzip
import os
import shutil
import subprocess

COMPANION_TYPES = ("adt", "hto", "crispr", "vdj")
R1_LENGTHS = {24, 26, 28}   # 10x cell barcode + UMI read lengths (adt.chemistry geometries; 5' v1 is 26, 3' v3 is 28)
INDEX_MAX = 10              # reads this short are sample-index reads, not R1 / R2


def load_table(path):
    """-> {gex_gsm: [{lib_type, lib_gsm, runs: [SRR], reads}]}; companion types only."""
    out = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            if r.get("lib_type") not in COMPANION_TYPES:
                continue
            runs = [x for x in (r.get("lib_runs") or "").replace(";", ",").split(",") if x]
            if not runs:
                continue
            out.setdefault(r["gex_gsm"], []).append(
                {"lib_type": r["lib_type"], "lib_gsm": r["lib_gsm"], "runs": runs, "reads": int(float(r.get("lib_reads") or 0))})
    return out


def read_lengths(path, n=200):
    """Distinct read lengths of the first n reads of a FASTQ(.gz)."""
    op = gzip.open if path.endswith(".gz") else open
    lens = set()
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                lens.add(len(line.rstrip("\n")))
                if len(lens) > 3 or i >= 4 * n:
                    break
    return lens


def pick_reads(files):
    """-> (r1, r2) from the FASTQ files fasterq-dump wrote for one run. Index reads (<= INDEX_MAX nt) are dropped; R1 is the remaining
    file of constant length in R1_LENGTHS (barcode + UMI), R2 the other. Falls back to file order when both or neither qualify."""
    cand = []
    for p in sorted(files):
        lens = read_lengths(p)
        if lens and max(lens) > INDEX_MAX:
            cand.append((p, lens))
    if len(cand) < 2:
        raise ValueError("need two non-index read files, got %d of %d" % (len(cand), len(files)))
    cand = cand[:3]
    r1 = [p for p, l in cand if len(l) == 1 and next(iter(l)) in R1_LENGTHS]
    if len(r1) == 1:
        r1 = r1[0]
        r2 = next(p for p, _ in cand if p != r1)
    else:
        r1, r2 = cand[0][0], cand[1][0]
    return r1, r2


def _gzip(path, threads, env):
    exe = shutil.which("pigz", path=env.get("PATH")) if env else shutil.which("pigz")
    cmd = [exe, "-1", "-p", str(threads), "-f", path] if exe else ["gzip", "-1", "-f", path]
    subprocess.run(cmd, check=True, env=env)
    return path + ".gz"


def fetch_library(lib, dest, scr, fetch_run, env, log, threads=2, **fr_kw):
    """Fetch every run of one companion library into dest as <run>_R1.fastq.gz / <run>_R2.fastq.gz. -> list of file names."""
    os.makedirs(dest, exist_ok=True)
    names = []
    for run in lib["runs"]:
        sra, _info = fetch_run(run, scr, log=log, **fr_kw)
        tmp = os.path.join(scr, "fq_" + run)
        os.makedirs(tmp, exist_ok=True)
        try:
            subprocess.run(["fasterq-dump", "--split-files", "--include-technical", "-e", str(threads), "-O", tmp, sra],
                           check=True, env=env, capture_output=True)
            files = [os.path.join(tmp, x) for x in os.listdir(tmp) if x.endswith(".fastq")]
            r1, r2 = pick_reads(files)
            for tag, src in (("R1", r1), ("R2", r2)):
                name = "%s_%s.fastq.gz" % (run, tag)
                shutil.move(_gzip(src, threads, env), os.path.join(dest, name))
                names.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                os.remove(sra)
            except OSError:
                pass
    return names


def fetch_all(table, gsm, inc, scr, fetch_run, env, log, heartbeat=lambda: None, **kw):
    """Fetch the companions of one GEX sample into inc/companions/. -> manifest records, one per library (failures included)."""
    recs = []
    for lib in table.get(gsm, []):
        dest = os.path.join(inc, "companions", lib["lib_type"], lib["lib_gsm"])
        rec = {"lib_type": lib["lib_type"], "lib_gsm": lib["lib_gsm"], "runs": lib["runs"], "reads": lib["reads"]}
        try:
            heartbeat()
            rec["files"] = fetch_library(lib, dest, scr, fetch_run, env, log, **kw)
            rec["status"] = "ok"
            rec["bytes"] = sum(os.path.getsize(os.path.join(dest, x)) for x in rec["files"])
        except Exception as e:  # noqa: BLE001  a companion never fails the GEX sample
            shutil.rmtree(dest, ignore_errors=True)
            rec.update(status="failed", reason=("%s: %s" % (type(e).__name__, e))[:300])
            log("[companion] %s %s FAILED: %s" % (lib["lib_type"], lib["lib_gsm"], rec["reason"]))
        recs.append(rec)
    return recs


def pairs_total(recs):
    """Read pairs the ok companions add (for the drain worker's admission estimate: `companion_pairs`)."""
    return sum(r.get("reads", 0) for r in recs if r.get("status") == "ok")
