"""Turn the raw timing directories of ap_run.sbatch and opt_run.sbatch into the small artifact tables.

    python collect.py BP_DIR OUT_DIR

BP_DIR   $BP (…/bench-public): reads perf/<dataset>/<tool>_r<rep>/ and perf_opt/<kind>/<name>_r<rep>/
OUT_DIR  $ART/alignment-performance/: writes runs.tsv (one row per tool x dataset x repetition) and opt_runs.tsv
Standard library only.
"""
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def gnu_time(p: Path) -> dict:
    d = {}
    if not p.exists():
        return d
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock)"):
            t = ln.split(": ", 1)[1].split(":")
            d["wall_s"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
        elif ln.startswith("User time"):
            d["user_s"] = float(ln.split(": ")[1])
        elif ln.startswith("System time"):
            d["sys_s"] = float(ln.split(": ")[1])
        elif ln.startswith("Maximum resident"):
            d["time_rss_gb"] = int(ln.split(": ")[1]) / 1024 ** 2
        elif ln.startswith("Exit status"):
            d["exit"] = int(ln.split(": ")[1])
    return d


def mem(p: Path) -> float | None:
    try:
        return json.loads(p.read_text())["peak_rss_gb"]
    except (OSError, ValueError, KeyError):
        return None


def node_info(d: Path) -> tuple[str, str]:
    t = (d / "node.txt").read_text(errors="replace") if (d / "node.txt").exists() else ""
    m = re.search(r"\] (.*?) (\S+) 20\d\d-", t)
    return (m.group(2), m.group(1)) if m else ("", "")


def log_stats(p: Path) -> dict:
    """Reads and unique-mapping rate from a STAR Log.final.out."""
    d = {}
    if not p.exists():
        return d
    for ln in p.read_text(errors="replace").splitlines():
        if "Number of input reads" in ln:
            d["reads_in"] = int(ln.split("|")[1])
        elif "Uniquely mapped reads %" in ln:
            d["uniq_pct"] = float(ln.split("|")[1].strip().rstrip("%"))
    return d


STAMP = re.compile(r"([A-Z][a-z]{2}) +(\d+) (\d\d:\d\d:\d\d)")


def stamp(line: str):
    m = STAMP.search(line)
    return datetime.strptime(f"2026 {m.group(1)} {m.group(2)} {m.group(3)}", "%Y %b %d %H:%M:%S") if m else None


def star_phases(p: Path) -> dict:
    """Seconds from 'Finished loading the genome' to 'finished mapping', to Solo counting, and the BAM sort, from star_Log.out."""
    d = {}
    if not p.exists():
        return d
    t_gen = t_map = t_solo = t_sort = None
    cache = ""
    for ln in p.read_text(errors="replace").splitlines():
        if ln.startswith("Finished loading the genome") and t_gen is None:
            t_gen = stamp(ln)
        elif "finished mapping" in ln and t_map is None:
            t_map = stamp(ln)
        elif "finished Solo counting" in ln and t_solo is None:
            t_solo = stamp(ln)
        elif "started sorting BAM" in ln and t_sort is None:
            t_sort = stamp(ln)
        elif "SEQCACHE" in ln:
            cache = ln.strip()
    if t_gen and t_map:
        d["mapping_s"] = (t_map - t_gen).total_seconds()
    if t_map and t_solo:
        d["solo_s"] = (t_solo - t_map).total_seconds()
    d["seqcache"] = cache
    m = re.search(r"hits?[ =:]+(\d+)\D+(?:lookups?|total|of)[ =:]+(\d+)", cache)
    if m and int(m.group(2)):
        d["cache_hit_rate"] = int(m.group(1)) / int(m.group(2))
    return d


def collect_runs(bp: Path, out: Path):
    rows = []
    for dsdir in sorted((bp / "perf").glob("*")):
        if not dsdir.is_dir():
            continue
        for d in sorted(dsdir.glob("*_r[0-9]*")):
            tool, rep = d.name.rsplit("_r", 1)
            host, cpu = node_info(d)
            run, enc = gnu_time(d / "run" / "time.txt"), gnu_time(d / "encode" / "time.txt")
            if not run:
                continue
            ex = (d / "exit.txt").read_text().split()[-1] if (d / "exit.txt").exists() else ""
            wall = run["wall_s"] + enc.get("wall_s", 0.0)
            cpu_s = run.get("user_s", 0) + run.get("sys_s", 0) + enc.get("user_s", 0) + enc.get("sys_s", 0)
            rss = max(filter(None, [mem(d / "run" / "mem.json"), mem(d / "encode" / "mem.json")]), default=None)
            st = log_stats(d / "Log.final.out")
            ph = star_phases(d / "seqcache.txt") if (d / "seqcache.txt").exists() else {}
            st["cache_hit_rate"] = round(ph["cache_hit_rate"], 4) if "cache_hit_rate" in ph else ""
            st["seqcache_line"] = ph.get("seqcache", "")[:200]
            rows.append({"dataset": dsdir.name, "tool": tool, "rep": rep, "host": host, "cpu": cpu, "threads": 16,
                         "wall_s": round(wall, 2), "encode_s": round(enc.get("wall_s", 0.0), 2), "run_s": round(run["wall_s"], 2),
                         "cpu_s": round(cpu_s, 1), "max_rss_gb": round(rss, 3) if rss else "",
                         "time_rss_gb": round(max(run.get("time_rss_gb", 0), enc.get("time_rss_gb", 0)), 3),
                         "exit": ex, "reads_in": st.get("reads_in", ""), "uniq_pct": st.get("uniq_pct", ""),
                         "cache_hit_rate": st["cache_hit_rate"], "seqcache_line": st["seqcache_line"]})
    cols = ["dataset", "tool", "rep", "host", "cpu", "threads", "wall_s", "encode_s", "run_s", "cpu_s", "max_rss_gb", "time_rss_gb",
            "exit", "reads_in", "uniq_pct", "cache_hit_rate", "seqcache_line"]
    with open(out / "runs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"runs.tsv: {len(rows)} rows")


def collect_opt(bp: Path, out: Path):
    rows = []
    for kdir in sorted((bp / "perf_opt").glob("*")):
        if not kdir.is_dir():
            continue
        for d in sorted(kdir.glob("*_r[0-9]*")):
            if not d.is_dir() or d.name.endswith(".keep"):
                continue
            name, rep = d.name.rsplit("_r", 1)
            host, cpu = node_info(d)
            keep = kdir / (d.name + ".keep")
            if name.startswith("x4"):
                parts = [gnu_time(c / "run" / "time.txt") for c in sorted(d.glob("c[0-9]"))]
                m = re.search(r"wall_all=(\d+)", (d / "node.txt").read_text()) if (d / "node.txt").exists() else None
                if not parts or not m:
                    continue
                wall = float(m.group(1))
                cpu_s = sum(p.get("user_s", 0) + p.get("sys_s", 0) for p in parts)
                rss = max(filter(None, [mem(c / "run" / "mem.json") for c in sorted(d.glob("c[0-9]"))]), default=None)
                copies = 4
            else:
                t = gnu_time(d / "run" / "time.txt")
                if not t:
                    continue
                wall, cpu_s, rss, copies = t["wall_s"], t.get("user_s", 0) + t.get("sys_s", 0), mem(d / "run" / "mem.json"), 1
            ph = star_phases(keep / "star_Log.out")
            cs = keep / "checksums.txt"
            sig = hashlib.md5(cs.read_bytes()).hexdigest()[:12] if cs.exists() and cs.stat().st_size else ""
            rows.append({"outputs_sig": sig,"kind": kdir.name, "name": name, "rep": rep, "host": host, "cpu": cpu, "copies": copies,
                         "wall_s": round(wall, 2), "cpu_s": round(cpu_s, 1), "max_rss_gb": round(rss, 3) if rss else "",
                         "mapping_s": ph.get("mapping_s", ""), "solo_s": ph.get("solo_s", ""),
                         "cache_hit_rate": round(ph["cache_hit_rate"], 4) if "cache_hit_rate" in ph else "",
                         "seqcache_line": ph.get("seqcache", "")[:200]})
    cols = ["kind", "name", "rep", "host", "cpu", "copies", "wall_s", "cpu_s", "max_rss_gb", "mapping_s", "solo_s",
            "cache_hit_rate", "outputs_sig", "seqcache_line"]
    with open(out / "opt_runs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"opt_runs.tsv: {len(rows)} rows")


METHODS = """**Inputs.** The same two FASTQ files per dataset (R1 barcode and UMI, R2 cDNA, lanes concatenated and copied to node-local disk before the clock starts) for every tool: PBMC 1k v3 (66.6 M read pairs), mouse brain nuclei 2k v2 (3' v2) and GSM5815984, a deep human testis library (GSE193672 Hs#14; 489 M read pairs, 3' v3, both runs, all reads). Reference: 10x 2024-A GRCh38 / GRCm39 (GENCODE v44 / M33); the STAR and alevin-fry indexes were built beforehand and are not timed.

**Hardware.** 16 threads and a 128 GB memory limit for every tool, on an exclusive Xeon Gold 6248 node, two repetitions with the tool order reversed in the second. Wall time is GNU time elapsed (Singlet: `singlet encode` plus the pipeline); CPU time is user plus system; peak memory is the RSS summed over the process tree, sampled at 1 Hz.

**Tools.** Singlet: the pinned campaign binary with the campaign flags (counts, introns, junctions, sorted BAM, donor demux, mtDNA, ambient, doublets, cell calling), non-host screen off. STARsolo 2.7.11b with Gene, GeneFull, Velocyto and SJ and Cell Ranger 4+ settings, without a BAM (starsolo) and with a coordinate-sorted BAM carrying CB/UB tags (starsolo-bam). alevin-fry through simpleaf with a spliced+unspliced index (cr-like resolution). Cell Ranger is not run; its published matrices are the comparator elsewhere.

**Exact and pseudo alignment.** Singlet aligns every read to the genome (STAR, exact seed-and-extend with splicing), which is what lets it count introns, junctions, SNPs, mtDNA and non-host reads and write a BAM. alevin-fry is a pseudoaligner: it maps reads to a spliced+unspliced transcriptome index, which is much cheaper but sees only what that index contains. The plot places each tool on cost (CPU-hours) against quality; the reach matrix lists what each tool can produce at all.

**What is gated.** Runtime is shown, not gated: wall time, CPU-hours and memory are in the resources table and figures. Singlet does more work than a counting pipeline and alevin-fry is a pseudoaligner, so the page places each tool on a quality-versus-CPU plot. The gated metrics check that the alignments are the same work (unique-mapping share equals STARsolo's) and that the optimizations leave every count matrix and the junction table unchanged (md5 of the matrices, with the cache off or on, and across threads, shared genome, sort mode and packing).

**Optimizations.** The table lists each Singlet STAR and job-shape optimization with its reported speedup (PR #34, docs/atlas-rerun-plan.md) next to what was re-measured here on PBMC 1k v3 at 4 threads. Cache hit rates are measured from the run logs."""


def write_provenance(bp: Path, out: Path):
    import subprocess
    src = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()

    def ver(cmd):
        try:
            return (subprocess.run(cmd, capture_output=True, text=True).stdout or "?").strip().splitlines()[0]
        except (OSError, IndexError):
            return "?"

    sv = {"STAR": ver(["STAR", "--version"]), "simpleaf": ver(["simpleaf", "--version"]), "alevin-fry": ver(["alevin-fry", "--version"])}
    tools = [
        {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": "singlet encode --reads R1.fastq.gz R2.fastq.gz -o s.1fq; singlet s.1fq --ref-base <ref> --genome-dir <STAR index> --exons genes.gtf --threads 16 --pipeline --output-format mtx --raw-matrix"},
        {"name": "singlet-core", "version": "1.0.0", "role": "helper", "params": "as singlet without --pipeline"},
        {"name": "starsolo", "version": sv["STAR"], "role": "competitor", "url": "https://github.com/alexdobin/STAR", "citation": "Kaminow, Yunusov, Dobin 2021 bioRxiv",
         "params": "STAR --runThreadN 16 --readFilesIn R2 R1 --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist WL --soloCBlen 16 --soloUMIlen 12|10 --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None"},
        {"name": "starsolo-bam", "version": sv["STAR"], "role": "competitor", "params": "as starsolo with --outSAMtype BAM SortedByCoordinate --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM --limitBAMsortRAM 40000000000"},
        {"name": "alevinfry", "version": f"{sv['simpleaf']}; alevin-fry {sv['alevin-fry']}", "role": "competitor", "url": "https://github.com/COMBINE-lab/alevin-fry",
         "citation": "He et al. 2022 Nat Methods", "params": "simpleaf quant --reads1 R1 --reads2 R2 --index <spliced+unspliced index> --t2g-map t2g_3col.tsv --chemistry 10xv3|10xv2 --expected-ori fw --resolution cr-like --unfiltered-pl WL --threads 16"}]
    tools.append({"name": "cellranger", "version": "3.0.0 (pbmc_1k_v3), 2.1.0 (nuclei_2k_v2)", "role": "truth",
                  "url": "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0",
                  "params": "not run: 10x's published filtered matrices are the comparator for accuracy and agreement; Cell Ranger is left out of the timing"})

    def ds(i, t, acc):
        return {"id": i, "title": t, "accessions": acc, "truth": "none: timing and output agreement on identical reads"}

    prov = {
        "bench_commit": commit, "status": "final",
        "generated_at": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%S+00:00"], text=True).strip(),
        "singlet": {"version": "1.0.0", "commit": "02aa9b938b54e0f92bba00c44c07b883a8660aa2", "branch": "prod/campaign", "binary": "singlet-campaign",
                    "binary_sha256": "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c", "build_flags": "PGO, -march=x86-64-v3"},
        "methods": METHODS, "tools": tools,
        "datasets": [
            ds("pbmc1k_v3", "10x PBMC 1k, 3' v3", {"url": "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0"}),
            ds("nuclei2k_v2", "10x mouse brain nuclei 2k, 3' v2", {"url": "https://www.10xgenomics.com/datasets/2-k-brain-nuclei-from-an-e18-mouse-2-standard-2-1-0"}),
            ds("gsm5815984", "GSM5815984 (GSE193672 Hs#14) human testis, 10x 3' v3, all 489 M read pairs", {"gse": "GSE193672", "gsm": ["GSM5815984"], "srr": ["SRR17615087", "SRR17615088"]})],
        "caveats": [
            "Singlet runs more modules than the counting pipelines it is timed against (sorted BAM, junctions, donor demux, mtDNA, ambient, doublets, cell calling), and alevin-fry is a pseudoaligner that does not align to the genome: wall time and CPU-hours are not a like-for-like speed contest and are not gated.",
            "Singlet's wall time includes the FASTQ-to-.1fq encoding (a one-off per sample in the atlas pipeline).",
            "Cell Ranger is not in the timing: 10x software is neither installed nor run here; 10x's published matrices stand in as the comparator for accuracy and agreement (see gex-counts).",
            "STARsolo reads the gzip FASTQs through zcat (its documented usage); the STAR and alevin-fry indexes are prebuilt and untimed.",
            "The PR #22 and PR #34 binaries clip 11 bp from PBMC R2 reads (a 5' prefix false positive fixed in the campaign binary), so optimization rows of the star group are comparable with each other but not with the campaign rows.",
            "Peak memory sums RSS over the process tree, so pages of a genome shared by several processes (the packed and shared-genome runs) are counted once per process.",
            "Two repetitions per configuration."]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    bp, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    collect_runs(bp, out)
    collect_opt(bp, out)
    write_provenance(bp, out)
