"""Capability `alignment-performance`: wall time, CPU-hours and peak memory of Singlet vs STARsolo and alevin-fry.

Same inputs for every tool (the same two FASTQ files, copied to node-local disk before the clock starts), the same reference (10x
2024-A, GENCODE v44 / M33), 16 threads, 128 GB memory limit, an exclusive node of one CPU type (Xeon Gold 6248), two repetitions in
alternating tool order. Cell Ranger is not installed on Clipper (it needs an EULA acceptance and a download, which this benchmark
does not do), so it is not in the comparison.

Artifacts (made by bench/public/harness/alignment_performance/collect.py):
  runs.tsv      one row per tool x dataset x repetition: wall_s (for Singlet: `singlet encode` + the pipeline), cpu_s (user + system),
                max_rss_gb (summed RSS of the process tree, sampled at 1 Hz), host, cpu, reads_in, uniq_pct
  opt_runs.tsv  the optimization experiments on PBMC 1k v3 (kind star: PR #34 builds, 4 threads; kind pipe: job-shape settings of the campaign binary)
  reported.tsv  the prior measurements the optimizations are documented with (PR #34, docs/atlas-rerun-plan.md)

Run:  python bench/public/capabilities/alignment_performance.py --artifacts DIR [--out bench/public/results/alignment-performance.json]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

CAPABILITY = "alignment-performance"
CODE_COMMIT = "0000000000000000000000000000000000000000"
DATASETS = ["pbmc1k_v3", "nuclei2k_v2", "gsm5815984"]
FAMILIES = {  # suffix -> competitors compared against
    "starsolo": ["starsolo"],
    "alevinfry": ["alevinfry"],
}
SHOWN = {"starsolo": ["starsolo", "starsolo-bam"], "alevinfry": ["alevinfry"]}
RES = [("wall", "Wall time", "s", "wall_s"), ("cpu", "CPU time", "core-hours", "cpu_h"), ("rss", "Peak memory", "GB", "max_rss_gb")]


def _med(df, tool, ds, col):
    v = df[(df.tool == tool) & (df.dataset == ds)][col].astype(float)
    return float(v.median()) if len(v) else None


def _spread(df, tool, ds, col):
    v = df[(df.tool == tool) & (df.dataset == ds)][col].astype(float).to_numpy()
    return float((v.max() - v.min()) / np.median(v)) if len(v) > 1 else 0.0


def _fmt(x, nd=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    runs = pd.read_csv(art / "runs.tsv", sep="\t", dtype={"rep": str})
    runs = runs[runs.exit.astype(str).isin(["0", "0.0"])].copy()
    runs["cpu_h"] = runs.cpu_s / 3600.0
    opt = pd.read_csv(art / "opt_runs.tsv", sep="\t", dtype={"rep": str})
    rep = pd.read_csv(art / "reported.tsv", sep="\t")
    dsets = [d for d in DATASETS if (runs.dataset == d).any()]

    metrics = []
    for ds in dsets:
        for fam, against in FAMILIES.items():
            for rid, rname, unit, col in RES:
                s = _med(runs, "singlet", ds, col)
                comps = [(t, _med(runs, t, ds, col)) for t in SHOWN[fam] if _med(runs, t, ds, col) is not None]
                if s is None or not comps or any(_med(runs, t, ds, col) is None for t in against):
                    continue
                tol = round(max(_spread(runs, t, ds, col) for t in against) + 1e-3, 3)
                label = {"starsolo": "STARsolo", "alevinfry": "alevin-fry"}[fam]
                metrics.append({
                    "id": f"{rid}_vs_{fam}", "name": f"{rname} vs {label}", "unit": unit, "category": "resources", "dataset": ds,
                    "direction": "lower", "singlet": {"value": C.finite(s)},
                    "competitors": [{"tool": t, "value": C.finite(v)} for t, v in comps],
                    "rule": {"type": "match_or_beat", "against": against, "tolerance": {"rel": tol},
                             "rationale": f"rep-to-rep spread of {', '.join(against)} on this dataset: {tol:.1%}"},
                    "notes": ("Singlet wall time includes `singlet encode` (FASTQ to .1fq). Singlet runs the campaign pipeline (counts, introns, "
                              "junctions, sorted BAM, donor demux, mtDNA, ambient, doublets, cell calling); STARsolo and alevin-fry produce counts only"
                              + (" (STARsolo with Velocyto and SJ)." if fam == "starsolo" else "."))})

    # ------------------------------------------------------------------ the optimization experiments
    star = opt[opt.kind == "star"]
    pipe = opt[opt.kind == "pipe"]

    def o(df, name, col):
        v = df[df.name == name][col].replace("", np.nan).astype(float)
        return float(v.median()) if v.notna().any() else None

    def same_outputs(df, a, b):
        sa = df[df.name == a].set_index("rep").outputs_sig
        sb = df[df.name == b].set_index("rep").outputs_sig
        common = sa.index.intersection(sb.index)
        ok = [str(sa[r]) == str(sb[r]) and str(sa[r]) not in ("", "nan") for r in common]
        return float(np.mean(ok)) if ok else None

    ident = same_outputs(star, "campaign", "campaign-nocache")
    if ident is not None:
        metrics.append({"id": "seqcache_outputs_identical", "name": "Sequence cache: every count matrix identical with the cache off", "unit": "fraction",
                        "category": "robustness", "dataset": "pbmc1k_v3", "direction": "higher", "singlet": {"value": ident}, "competitors": [],
                        "rule": {"type": "threshold", "value": 1.0, "rationale": "the cache is exact by construction: matrices and junction tables must be byte-identical"},
                        "notes": "md5 of the content of exon/intron/ambiguous counts, STARsolo raw matrices and star_SJ.out.tab, cache 2 GB vs off, same binary and node."})
    if len(pipe):
        base = pipe[pipe.name == "t4"].set_index("rep").outputs_sig
        others = [n for n in pipe.name.unique() if n != "t4" and n != "t4_nonhost"]
        oks = []
        for n in others:
            sn = pipe[pipe.name == n].set_index("rep").outputs_sig
            for r in sn.index.intersection(base.index):
                oks.append(str(sn[r]) == str(base[r]) and str(sn[r]) not in ("", "nan"))
        if oks:
            metrics.append({"id": "job_shape_outputs_identical", "name": "Threads, shared genome, sort mode and packing: outputs identical", "unit": "fraction",
                            "category": "robustness", "dataset": "pbmc1k_v3", "direction": "higher", "singlet": {"value": float(np.mean(oks))},
                            "competitors": [],
                            "rule": {"type": "threshold", "value": 1.0, "rationale": "the pipeline is deterministic: 4, 8 and 16 threads, a shared genome, memory or disk sort and four concurrent samples give the same matrices"},
                            "notes": f"configurations compared with t4: {', '.join(sorted(others))}."})
    for nm_, a_, b_, mid, mname in (("campaign", "pr22", None, "star_mapping_speedup_vs_pr22", "STAR mapping speedup of the campaign binary vs the PR #22 binary"),
                                   ("campaign-nocache", "campaign", None, "seqcache_mapping_speedup", "STAR mapping speedup from the sequence cache")):
        num, den = (o(star, "pr22", "mapping_s"), o(star, "campaign", "mapping_s")) if mid.startswith("star") else (o(star, "campaign-nocache", "mapping_s"), o(star, "campaign", "mapping_s"))
        if num and den:
            metrics.append({"id": mid, "name": mname, "unit": "ratio", "category": "resources", "dataset": "pbmc1k_v3", "direction": "none",
                            "singlet": {"value": round(num / den, 3)}, "competitors": [], "rule": {"type": "informational"},
                            "notes": "Median of 2 paired repetitions, 4 threads, same node type. PR #34 reported 1.54x for the whole of its changes."
                                     + (" The PR #22 binary clips 11 bp from PBMC R2 reads (the 5' prefix false positive fixed since), so it maps shorter reads: the speedup is understated." if mid.startswith("star") else "")})

    def row(label, level, reported, by, measured, same):
        return [label, level, reported, by, measured, same]

    def mt(name, df=star):
        w, m = o(df, name, "wall_s"), o(df, name, "mapping_s")
        return w, m

    tab = []
    w_c, m_c = mt("campaign")
    w_n, m_n = mt("campaign-nocache")
    w_p, m_p = mt("pr22")
    w_s, m_s = mt("stock-star")
    w_l, m_l = mt("lazy-huge-cache1")
    w_g, m_g = mt("pgo-cache2")
    hit = o(star, "campaign", "cache_hit_rate")
    idc = same_outputs(star, "campaign", "campaign-nocache")
    tab.append(row("Sequence-to-alignment cache (2 GB), replaces the adjacent-read dedup", "STAR",
                   "hit rate 44-51% on PBMC at 1-4 GB; 66% simulated at 489 M reads; ~35% less mapping time at 48% hits", "PR #34",
                   f"mapping {_fmt(m_n)} s without, {_fmt(m_c)} s with ({_fmt((m_n or np.nan) / (m_c or np.nan), 2)}x); wall {_fmt(w_n)} to {_fmt(w_c)} s"
                   + (f"; hit rate {hit:.1%}" if hit else ""),
                   "identical to cache off" if idc == 1.0 else ("DIFFERS" if idc is not None else "not checked")))
    tab.append(row("Lazy window table + huge pages + faster read parsing (PR #34 build with a 1 GB cache)", "STAR",
                   "mapping 793 to 586 s with the 1 GB cache (Xeon 6226); 10 M subset: lazy 133 to 117.5 s, huge pages 111.5 s, both 107.5 s", "PR #34",
                   f"mapping {_fmt(m_p)} s (PR #22 binary) to {_fmt(m_l)} s; wall {_fmt(w_p)} to {_fmt(w_l)} s", "identical to stock STAR (reported)"))
    tab.append(row("Profile-guided build (PGO), 2 GB cache", "STAR", "mapping 586 to 515 s; whole run 912 to 845 s; 10 M subset 94.5 to 85 s", "PR #34",
                   f"mapping {_fmt(m_l)} s to {_fmt(m_g)} s; wall {_fmt(w_l)} to {_fmt(w_g)} s", "identical to stock STAR (reported)"))
    tab.append(row("All STAR changes: the pinned campaign binary vs the PR #22 binary", "STAR",
                   "mapping 793 to 515 s (1.54x); whole run 1,138 to 845 s (1.35x); stock STAR 863 s mapping (1.68x)", "PR #34",
                   f"mapping {_fmt(m_p)} to {_fmt(m_c)} s ({_fmt((m_p or np.nan) / (m_c or np.nan), 2)}x); wall {_fmt(w_p)} to {_fmt(w_c)} s "
                   f"({_fmt((w_p or np.nan) / (w_c or np.nan), 2)}x); stock STAR {_fmt(m_s)} s mapping, {_fmt(w_s)} s wall",
                   "identical to stock STAR (reported); PR #22 binary differs on ~300 reads"))

    def c_h(df, name):
        v = df[df.name == name]
        if v.empty:
            return None
        thr = {"t4": 4, "t8": 8, "t16": 16}.get(name, 4)
        return float(v.wall_s.median()) * thr / 3600.0

    t4w, t8w, t16w = o(pipe, "t4", "wall_s"), o(pipe, "t8", "wall_s"), o(pipe, "t16", "wall_s")
    tab.append(row("4 threads per sample instead of 8 or 16", "pipeline", "core-hours per sample 1.24 (4 threads), 1.59 (8), 1.85-2.41 (16)", "atlas-rerun plan section 3",
                   f"wall {_fmt(t4w)} / {_fmt(t8w)} / {_fmt(t16w)} s at 4 / 8 / 16 threads = {_fmt(c_h(pipe, 't4'), 2)} / {_fmt(c_h(pipe, 't8'), 2)} / {_fmt(c_h(pipe, 't16'), 2)} reserved core-hours",
                   "identical across thread counts" if all(str(x) == "1.0" or x == 1.0 for x in [same_outputs(pipe, 't4', 't8'), same_outputs(pipe, 't4', 't16')]) else "see job_shape_outputs_identical"))
    tab.append(row("Genome in shared memory (--genome-shared on)", "pipeline", "-1% time; one genome per node instead of one per job", "atlas-rerun plan section 3",
                   f"wall {_fmt(t4w)} s to {_fmt(o(pipe, 't4_shm', 'wall_s'))} s; peak RSS {_fmt(o(pipe, 't4', 'max_rss_gb'), 1)} to {_fmt(o(pipe, 't4_shm', 'max_rss_gb'), 1)} GB (shared pages are counted per process)",
                   "identical" if same_outputs(pipe, "t4", "t4_shm") == 1.0 else "see job_shape_outputs_identical"))
    tab.append(row("Disk-backed BAM sort (the campaign default) instead of an in-RAM sort", "pipeline", "no time change; peak RSS 18.6 GB instead of a buffer sized to the job's memory", "atlas-rerun plan section 3",
                   f"wall {_fmt(o(pipe, 't4_mem', 'wall_s'))} s (memory sort) vs {_fmt(t4w)} s (disk); peak RSS {_fmt(o(pipe, 't4_mem', 'max_rss_gb'), 1)} vs {_fmt(o(pipe, 't4', 'max_rss_gb'), 1)} GB",
                   "identical" if same_outputs(pipe, "t4", "t4_mem") == 1.0 else "see job_shape_outputs_identical"))
    x4 = pipe[pipe.name == "x4_lean"]
    if len(x4):
        per_sample_ch = float(x4.wall_s.median()) * 16 / 4 / 3600.0
        tab.append(row("Four samples packed on 16 cores with one shared genome and disk sort", "pipeline", "+15% per sample vs one alone (memory bandwidth); +19% in the lean-packing run", "atlas-rerun plan section 3",
                       f"four samples in {_fmt(float(x4.wall_s.median()))} s ({_fmt(float(x4.wall_s.median()) / t4w, 2) if t4w else 'n/a'}x one alone); {per_sample_ch:.2f} core-hours per sample vs {_fmt(c_h(pipe, 't4'), 2)}; peak RSS {_fmt(float(x4.max_rss_gb.astype(float).max()), 1)} GB (sum over the four, shared genome pages counted four times)",
                       "identical" if same_outputs(pipe, "t4", "x4_lean") == 1.0 else "see job_shape_outputs_identical"))
    tab.append(row("Non-host (viral, bacterial, fungal) screen on, the campaign default", "pipeline", "+27% time and +35 GB RSS vs off; indices cached in /dev/shm give no gain", "atlas-rerun plan section 3",
                   f"wall {_fmt(t4w)} s (off) to {_fmt(o(pipe, 't4_nonhost', 'wall_s'))} s (on); peak RSS {_fmt(o(pipe, 't4', 'max_rss_gb'), 1)} to {_fmt(o(pipe, 't4_nonhost', 'max_rss_gb'), 1)} GB",
                   "count matrices identical; non-host members added"))
    tab.append(row("Download as a separate 1-core stage (prefetch, then singlet download of the local .sra)", "pipeline", "163 s vs 1,054 s for one 1.15 GB run; -45% core-hours at full depth; decoded reads identical", "atlas-rerun plan section 3",
                   "measured in the download capability (bench/public/results/download.json)", "decoded reads identical (reported)"))

    cols = ["Optimization", "Level", "Reported", "Reported by", "Measured here (campaign binary, PBMC 1k v3, Xeon Gold 6248)", "Output vs unoptimized"]
    fig_tab = {"id": "optimizations", "title": "Each optimization, its reported effect and what this benchmark measures", "kind": "table",
               "caption": "Mapping time is STAR 'Finished loading the genome' to 'finished mapping' in the run log. Reported numbers are quoted from the PR #34 description and docs/atlas-rerun-plan.md "
                          "(Xeon Gold 6226 for the STAR rows); the measured column uses 4 threads on an exclusive Xeon Gold 6248, median of 2 repetitions.",
               "columns": cols, "rows": tab}

    # per-tool figures
    figs = [fig_tab]
    for rid, rname, unit, col in RES:
        series = []
        for tool in ("singlet", "starsolo", "starsolo-bam", "alevinfry", "singlet-core"):
            ys = [_med(runs, tool, d, col) for d in dsets]
            if any(y is not None for y in ys):
                series.append({"name": tool, "tool": tool, "x": dsets, "y": [None if y is None else round(y, 3) for y in ys]})
        figs.append({"id": f"{rid}_by_dataset", "title": f"{rname} by dataset", "kind": "grouped_bar",
                     "caption": "Median of 2 repetitions; 16 threads, identical FASTQ input, exclusive node. Singlet's wall time includes the FASTQ encoding step.",
                     "x": {"label": "dataset", "categories": dsets}, "y": {"label": rname, "unit": unit, "scale": "linear", "min": 0}, "series": series})

    resources = []
    for (ds, tool), g in runs.groupby(["dataset", "tool"]):
        resources.append({"tool": tool, "dataset": ds, "wall_s": round(float(g.wall_s.median()), 1), "max_rss_gb": round(float(g.max_rss_gb.astype(float).median()), 2),
                          "threads": 16, "cpu_h": round(float(g.cpu_h.median()), 3), "node": f"{g.host.iloc[0]} ({g.cpu.iloc[0]})",
                          "notes": f"median of {len(g)} repetitions; 128 GB limit"})
    for (kind, name), g in opt.groupby(["kind", "name"]):
        resources.append({"tool": "singlet", "dataset": "pbmc1k_v3", "wall_s": round(float(g.wall_s.median()), 1),
                          "max_rss_gb": round(float(g.max_rss_gb.replace("", np.nan).astype(float).median()), 2), "threads": 16 if name.startswith("t16") else (8 if name.startswith("t8") else 4),
                          "cpu_h": round(float(g.cpu_s.median()) / 3600.0, 3), "node": f"{g.host.iloc[0]} ({g.cpu.iloc[0]})",
                          "notes": f"optimization experiment {kind}/{name}: " + ("4 samples concurrently" if g.copies.iloc[0] == 4 else "one sample")})

    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAPABILITY,
        "title": "Alignment performance", "conclusion": "", "methods": "", "status": "final", "generated_at": "", "owner": "pipeline-benchmark",
        "singlet": C.SINGLET, "tools": [], "datasets": [], "metrics": metrics, "resources": resources, "figures": figs,
        "artifacts": C.artifact_entries(art, CAPABILITY, {"runs": "derived", "opt_runs": "derived", "reported": "derived"}),
        "code": C.code_entries(CODE_COMMIT, [
            ("bench/public/capabilities/alignment_performance.py", "metric"), ("bench/public/harness/alignment_performance/ap_run.sbatch", "harness"),
            ("bench/public/harness/alignment_performance/opt_run.sbatch", "harness"), ("bench/public/harness/alignment_performance/prep_deep.sbatch", "harness"),
            ("bench/public/harness/alignment_performance/collect.py", "harness"), ("bench/public/harness/memwatch.py", "harness")]),
        "reproduce": {"module": "bench/public/capabilities/alignment_performance.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "sbatch bench/public/harness/alignment_performance/ap_run.sbatch <dataset> <rep>"},
        "caveats": [],
    }
    return doc


if __name__ == "__main__":
    C.cli(compute)
