"""Capability `download`: throughput and completeness of Singlet's SRA download orchestration vs the alternatives.

Four 10x 3' GEX runs (runs.sh of the harness; 32 to 36 M read pairs, 1.2 to 3.4 GB as .sra) are fetched with every tool as four
concurrent streams (one run each) from a Clipper compute node, twice, in alternating order. Truth: NCBI's own decode of the
archive (prefetch + fasterq-dump --split-files of the run).

Artifacts (made by bench/public/harness/download/collect.py):
  streams.tsv    rep, mode, srr, start_s, end_s, rc, payload_bytes (srr = ALL is the makespan of the four streams)
  memory.tsv     rep, mode, peak_rss_gb (summed RSS of the streams' process tree)
  digests.tsv    rep, mode, srr, file, spots, read_len, seq_digest, qual_digest of the decoded files of each tool
  reference.tsv  the same for the truth decode; sra_files.tsv: srr, .sra bytes, sha256

Throughput is *effective*: the bytes of the run as an .sra file (the same for every tool; the size the archive stores) divided by the
seconds until the tool has delivered what it delivers (a decoded read set, or for AWS S3 the file). Completeness: spot count of the
decoded output equals the archive's, and the order-independent digest of the decoded sequences equals the reference's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _perf as C  # noqa: E402

CAPABILITY = "download"
CODE_COMMIT = "0000000000000000000000000000000000000000"
DS = "sra_download_set"
# harness mode -> tool name in the document
TOOL = {"singlet_orch_io": "singlet", "singlet_prefetch_local": "singlet-prefetch-local", "singlet_remote": "singlet-remote",
        "fasterq_dump": "fasterq-dump", "prefetch_fasterq_dump": "prefetch+fasterq-dump", "aws_s3": "aws-s3", "ena_https": "ena-https"}
COMPETITORS = [t for m, t in TOOL.items() if t != "singlet"]
PRODUCES_READS = {"singlet", "singlet-prefetch-local", "singlet-remote", "fasterq-dump", "prefetch+fasterq-dump", "ena-https"}


def _metric(mid, name, unit, cat, direction, s, comps, rule, notes=None, ci=None):
    m = {"id": mid, "name": name, "unit": unit, "category": cat, "dataset": DS, "direction": direction,
         "singlet": {"value": C.finite(s)}, "competitors": [{"tool": t, "value": C.finite(v)} for t, v in comps], "rule": rule}
    if ci:
        m["singlet"]["ci"] = [float(ci[0]), float(ci[1])]
    if notes:
        m["notes"] = notes
    return m


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    st = pd.read_csv(art / "streams.tsv", sep="\t", dtype={"rep": str})
    mem = pd.read_csv(art / "memory.tsv", sep="\t", dtype={"rep": str})
    dig = pd.read_csv(art / "digests.tsv", sep="\t", dtype={"rep": str})
    ref = pd.read_csv(art / "reference.tsv", sep="\t")
    sra = pd.read_csv(art / "sra_files.tsv", sep="\t", header=None, names=["srr", "bytes", "sha256"]).set_index("srr")
    runs = list(sra.index)
    ref["srr"] = ref.file.str.split("_").str[0]
    spots = ref.groupby("srr").spots.max()
    total_bytes, total_spots = float(sra.bytes.sum()), float(spots.sum())

    # per (mode, rep): makespan, per-stream walls
    rows = []
    for (mode, rep), g in st.groupby(["mode", "rep"]):
        allr = g[g.srr == "ALL"].iloc[0]
        s = g[g.srr != "ALL"].copy()
        s["wall"] = s.end_s - s.start_s
        rows.append({"mode": mode, "rep": rep, "makespan": allr.end_s - allr.start_s, "failed": int((s.rc != 0).sum()),
                     "stream_mb_s": float(np.median([sra.bytes[r] / w / 1e6 for r, w in zip(s.srr, s.wall)])),
                     "stream_reads_s": float(np.median([spots[r] / w for r, w in zip(s.srr, s.wall)]))})
    t = pd.DataFrame(rows)
    t["agg_mb_s"] = total_bytes / t.makespan / 1e6
    t["agg_reads_s"] = total_spots / t.makespan
    t["tool"] = t["mode"].map(TOOL)

    def val(tool, col):
        v = t[t.tool == tool][col]
        return float(v.median()) if len(v) else None

    def spread(tool, col):  # relative rep-to-rep spread of a tool's own measurements
        v = t[t.tool == tool][col].to_numpy(float)
        return float((v.max() - v.min()) / v.mean()) if len(v) > 1 else 0.0

    def tput(mid, name, unit, col, tools, notes):
        comp = [(x, val(x, col)) for x in COMPETITORS if x in tools]
        best = max((c for c in comp if c[1] is not None), key=lambda c: c[1])
        tol = round(spread(best[0], col) + 1e-3, 3)
        return _metric(mid, name, unit, "resources", "higher", val("singlet", col), comp,
                       {"type": "match_or_beat", "against": "best", "tolerance": {"rel": tol},
                        "rationale": f"rep-to-rep spread of the best competitor ({best[0]}) on the same four runs: {tol:.1%}"}, notes)

    metrics = [
        tput("aggregate_mb_s", "Aggregate throughput, 4 streams", "MB/s", "agg_mb_s", set(COMPETITORS),
             "Archive bytes (.sra) of the four runs over the makespan of the four concurrent streams; median of 2 repetitions. AWS S3 and prefetch only "
             "move the file; every other tool also decodes, and Singlet's orchestrator stage also screens the reads."),
        tput("per_stream_mb_s", "Per-stream throughput", "MB/s", "stream_mb_s", set(COMPETITORS), "Median over the four streams, median of 2 repetitions."),
        tput("aggregate_reads_s", "Aggregate decoded reads per second", "reads/s", "agg_reads_s", PRODUCES_READS - {"singlet"},
             "Read pairs delivered per second (AWS S3 delivers a file, not reads)."),
        tput("per_stream_reads_s", "Per-stream decoded reads per second", "reads/s", "stream_reads_s", PRODUCES_READS - {"singlet"}, ""),
    ]

    # completeness (first repetition that was scored)
    def scored(mode):
        d = dig[dig["mode"] == mode]
        return d[d.rep == d.rep.min()] if len(d) else d

    def spot_match(mode) -> float | None:
        d = scored(mode)
        if not len(d):
            return None
        return float(np.mean([int(d[d.srr == r].spots.max()) == int(spots[r]) if (d.srr == r).any() else False for r in runs]))

    def seq_match(mode) -> float | None:
        d = scored(mode)
        if not len(d):
            return None
        if mode == "aws_s3":  # a file: identical iff the sha256 equals the prefetched archive
            return float(np.mean([(d[d.srr == r].seq_digest.astype(str) == sra.sha256[r]).any() for r in runs]))
        ok = []
        for r in runs:
            rr = set(ref[ref.srr == r].seq_digest.astype(str))
            dd = d[d.srr == r].seq_digest.astype(str)
            ok.append(len(dd) > 0 and all(x in rr for x in dd))
        return float(np.mean(ok))

    sm = {m: spot_match(m) for m in TOOL}
    cm = {m: seq_match(m) for m in TOOL}
    cm["singlet_orch_io"] = cm.get("singlet_lossless")  # the lossless configuration of the same code path is scored for identity
    comps_spot = [(TOOL[m], sm[m]) for m in TOOL if TOOL[m] != "singlet" and sm[m] is not None]
    comps_seq = [(TOOL[m], cm[m]) for m in TOOL if TOOL[m] != "singlet" and cm[m] is not None]
    metrics.append(_metric("spot_count_match", "Runs whose decoded spot count equals the archive's", "fraction", "coverage", "higher",
                           sm["singlet_orch_io"], comps_spot,
                           {"type": "threshold", "value": 1.0, "rationale": "every spot of every run must be delivered"},
                           "Singlet's default path (the orchestrator's I/O stage)."))
    metrics.append(_metric("read_checksum_match", "Runs whose decoded reads are identical to the archive's", "fraction", "accuracy", "higher",
                           cm["singlet_orch_io"], comps_seq,
                           {"type": "threshold", "value": 1.0, "rationale": "decoded reads must be bit-identical to NCBI's decode of the archive"},
                           "Scored for Singlet with --no-trim --quality full: its default path trims poly(A)/adapters and drops base qualities by design."))
    dflt = scored("singlet_orch_io")
    if len(dflt):
        r2 = dflt.seq_digest.astype(str)
        ident = float(np.mean([x in set(ref.seq_digest.astype(str)) for x in r2]))
        metrics.append(_metric("default_path_files_identical", "Singlet default path: decoded files identical to the archive's", "fraction",
                               "accuracy", "none", ident, [], {"type": "informational"},
                               "R1 (barcode and UMI) is unchanged; R2 differs where the default path trims poly(A) and adapters."))

    resources = []
    for m, tool in TOOL.items():
        tt = t[t["mode"] == m]
        mm = mem[mem["mode"] == m]
        if tt.empty:
            continue
        resources.append({"tool": tool, "dataset": DS, "wall_s": round(float(tt.makespan.median()), 1),
                          "max_rss_gb": round(float(mm.peak_rss_gb.max()), 2) if len(mm) else 0.0, "threads": 4,
                          "notes": "4 concurrent streams, one run each; wall = makespan; RSS summed over the streams' processes"})

    fig = {"id": "throughput_by_tool", "title": "Aggregate throughput of four concurrent streams", "kind": "bar",
           "caption": "Archive bytes per second until the tool has delivered its output; median of 2 repetitions.",
           "x": {"label": "tool", "categories": [TOOL[m] for m in TOOL]}, "y": {"label": "MB/s", "unit": "MB/s", "min": 0},
           "series": [{"name": "aggregate", "x": [TOOL[m] for m in TOOL], "y": [round(val(TOOL[m], "agg_mb_s") or 0, 2) for m in TOOL]},
                      {"name": "per stream", "x": [TOOL[m] for m in TOOL], "y": [round(val(TOOL[m], "stream_mb_s") or 0, 2) for m in TOOL]}]}

    prov = json.loads((art / "provenance.json").read_text(encoding="utf-8"))
    commit = prov["bench_commit"]

    def g(tool, col):
        return val(tool, col)

    def f(x):
        return "n/a" if x is None else f"{x:,.0f}"

    conclusion = (f"Four concurrent streams of four 10x runs (132 M read pairs): Singlet's orchestrator I/O stage reaches {f(g('singlet', 'agg_mb_s'))} MB/s "
                  f"effective ({f(g('singlet', 'agg_reads_s'))} read pairs/s) including two read screens, against {f(g('singlet-remote', 'agg_mb_s'))} MB/s for the remote "
                  f"singlet download, {f(g('fasterq-dump', 'agg_mb_s'))} for fasterq-dump, {f(g('prefetch+fasterq-dump', 'agg_mb_s'))} for prefetch plus fasterq-dump, "
                  f"{f(g('aws-s3', 'agg_mb_s'))} for AWS S3 (file only) and {f(g('ena-https', 'agg_mb_s'))} for ENA. Completeness is scored against NCBI's decode.")
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAPABILITY,
        "title": "SRA download orchestration", "conclusion": conclusion, "methods": prov["methods"], "status": prov.get("status", "final"),
        "generated_at": prov["generated_at"], "owner": "pipeline-benchmark", "singlet": prov["singlet"], "tools": prov["tools"],
        "datasets": prov["datasets"], "metrics": metrics, "resources": resources, "figures": [fig],
        "artifacts": C.artifact_entries(art, CAPABILITY, {"reference": "truth", "sra_files": "truth", "streams": "derived", "digests": "derived",
                                                          "memory": "derived", "provenance": "derived"}),
        "code": C.code_entries(commit, [
            ("bench/public/capabilities/download.py", "metric"), ("bench/public/harness/download/dl_bench.sbatch", "harness"),
            ("bench/public/harness/download/dl_reference.sbatch", "harness"), ("bench/public/harness/download/fq_digest.py", "metric"),
            ("bench/public/harness/download/collect.py", "harness")]),
        "reproduce": {"module": "bench/public/capabilities/download.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "sbatch bench/public/harness/download/dl_bench.sbatch <rep> \"<modes>\""},
        "caveats": prov["caveats"],
    }
    return doc


if __name__ == "__main__":
    C.cli(compute)
