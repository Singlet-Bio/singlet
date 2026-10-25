"""Build the non-host artifact tables from the raw runs of 10_singlet / 20_kraken / 30_oracle.

    python collect.py BP_DIR OUT_DIR

BP_DIR   $BP (…/bench-public): nonhost/<dataset>/{singlet,kraken2_pluspf08,kraken2_viral,oracle}/ and nonhost/cells/<dataset>.txt
OUT_DIR  $ART/non-host/: percell_<dataset>.tsv.gz, species.tsv, meta.tsv, resources.tsv (formats: bench/public/capabilities/non_host.py)
Needs pandas and numpy (the Clipper env). The definition of a called species is non_host.floor_reads, imported from the metric module.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "capabilities"))
import non_host as NH  # noqa: E402

DATASETS = [NH.DPI2, NH.MOCK, NH.PBMC]
RUN_DIR = {NH.DPI2: "gse166766_hbec_dpi2", NH.MOCK: "gse166766_hbec_mock", NH.PBMC: "pbmc1k_v3"}  # run directory names (10_singlet.sbatch LABEL)
KR = {NH.TOOL_KP: "kraken2_pluspf08", NH.TOOL_KV: "kraken2_viral"}


def gnu(p: Path) -> dict:
    d = {}
    if not p.exists():
        return d
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock)"):
            t = ln.split(": ", 1)[1].split(":")
            d["wall"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
        elif ln.startswith("User time"):
            d["cpu"] = d.get("cpu", 0) + float(ln.split(": ")[1])
        elif ln.startswith("System time"):
            d["cpu"] = d.get("cpu", 0) + float(ln.split(": ")[1])
    return d


def rss(p: Path) -> float:
    try:
        return json.loads(p.read_text())["peak_rss_gb"]
    except (OSError, ValueError, KeyError):
        return 0.0


def read_nodes(p: Path) -> str:
    t = p.read_text(errors="replace").splitlines()[0] if p.exists() and p.stat().st_size else ""
    n = re.search(r"node=([^\s.]+)", t)
    c = re.search(r"cpu=(.*?)(?: 20\d\d-\S+)?$", t)
    return f"{n.group(1)} ({c.group(1).strip()})" if n and c else (n.group(1) if n else "")


METHODS = """**Data.** SARS-CoV-2-infected (2 days post infection) and mock human bronchial epithelial cells (GSE166766, 10x 3' v3, the first 50 M read pairs of each run) and the clean PBMC 1k v3 sample. Every tool is scored on the same cells: the cells called in an earlier STARsolo run of each sample (11,567 infected, 14,013 mock, 1,136 PBMC). The mock and PBMC samples carry no virus. For the infected sample the truth is a minimap2 alignment of the host-unmapped reads to NC_045512.2 (at least 60 nt aligned at 90% identity), counted per cell as distinct (cell, UMI) molecules; it is a read-level oracle, not an independent cell annotation, and not every infected cell carries virus, which bounds the best infected-vs-mock AUROC.

**Singlet.** The pinned 1.0.0 binary with the campaign settings and the screen on: reads STAR cannot map to the host go to viral, bacterial and fungal k-mer screens behind a host Bloom filter, an EM over species keeps the confirmed ones, and the per-cell table lists confirmed species only. 16 threads.

**Kraken2.** Run at its defaults (confidence 0) on the same host-unmapped reads, with two databases: the standard PlusPF index capped at 8 GB (k2_pluspf_08gb_20250402, contains human) and the viral-only index of the first benchmark (k2_viral 2024-06-05). Reads are rolled up to species; reads assigned above species rank and human reads are dropped. Per-cell attribution uses the barcode mate with one-mismatch correction to the called cells.

**Calls.** A species is called in a sample when it reaches Singlet's own reporting floor, max(500, 0.5% of screened reads) reads. Singlet applies it internally; here it is applied to Kraken2 as well. A cell is positive for a tool when it has at least one UMI of a called species.

**Metrics.** Tolerances are measured: the half-width of the paired bootstrap 95% CI of the difference to the best Kraken2 (200 resamples of cells), or the disagreement between Kraken2 and the oracle or between the two Kraken2 databases. Resources are for the whole Singlet pipeline from FASTQ against Kraken2 on the unmapped reads alone, so they are not comparable."""


def write_provenance(bp: Path, out: Path):
    import subprocess
    src = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    d0 = bp / "nonhost" / RUN_DIR[NH.DPI2]
    kv = re.search(r"Kraken version (\S+)", (d0 / "kraken2_pluspf08" / "node.txt").read_text())
    mm = re.search(r"minimap2 (\S+)", (d0 / "oracle" / "node.txt").read_text())
    dbv = (bp / "db" / "k2_pluspf08" / "VERSION").read_text().strip()
    acc = lambda gsm, srr: {"gse": "GSE166766", "gsm": [gsm], "srr": [srr]}
    prov = {
        "bench_commit": commit, "status": "final",
        "generated_at": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%S+00:00"], text=True).strip(),
        "singlet": NH.C.SINGLET, "methods": METHODS,
        "tools": [
            {"name": NH.TOOL_S, "version": "1.0.0", "role": "singlet", "url": "https://github.com/Singlet-Bio/singlet",
             "params": "singlet s.1fq --ref-base <reference root> --genome-dir <GRCh38-2024-A STAR index> --exons genes.gtf --threads 16 --pipeline --output-format mtx --raw-matrix, with SINGLET_REF_BASE set (non-host screen on)"},
            {"name": NH.TOOL_KP, "version": kv.group(1) if kv else "?", "role": "competitor", "url": "https://github.com/DerrickWood/kraken2",
             "citation": "Wood, Lu, Langmead 2019 Genome Biol", "params": f"kraken2 --db {dbv} --threads 16 --report R --output O star_Unmapped.out.mate1 (confidence 0.0)"},
            {"name": NH.TOOL_KV, "version": kv.group(1) if kv else "?", "role": "competitor", "url": "https://github.com/DerrickWood/kraken2",
             "params": "kraken2 --db k2_viral (Langmead prebuilt, 2024-06-05) --threads 16 --report R --output O star_Unmapped.out.mate1 (confidence 0.0)"},
            {"name": NH.TOOL_O, "version": mm.group(1) if mm else "?", "role": "truth", "url": "https://github.com/lh3/minimap2",
             "params": "minimap2 -x sr NC_045512.2.fa star_Unmapped.out.mate1; a read counts when >= 60 nt aligned at >= 90% identity"}],
        "datasets": [
            {"id": NH.DPI2, "title": "Ravindra 2021 bronchial epithelium, SARS-CoV-2 2 dpi (first 50M spots)", "accessions": acc("GSM5082291", "SRR13711615"),
             "truth": "SARS-CoV-2 infected; minimap2 oracle per read and cell", "n_cells": 11567},
            {"id": NH.MOCK, "title": "Ravindra 2021 bronchial epithelium, mock (first 50M spots)", "accessions": acc("GSM5082289", "SRR13711613"),
             "truth": "uninfected", "n_cells": 14013},
            {"id": NH.PBMC, "title": "10x PBMC 1k, 3' v3", "accessions": {"url": "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0"},
             "truth": "healthy donor PBMC, no non-host organism expected", "n_cells": 1136}],
        "caveats": [
            "The infected sample is bounded by biology: only about 42% of its cells carry a SARS-CoV-2 read, so the infected-vs-mock AUROC cannot exceed the oracle's.",
            "Cell-level truth does not exist; the oracle is a read-level alignment.",
            "Only SARS-CoV-2 was tested for correct identification; bacterial and fungal paths were exercised only as false-positive checks. There is no sensitivity titration: Singlet's reporting floor, max(500, 0.5% of screened reads), hides an infection below it.",
            "Kraken2 runs on the reads Singlet screens (host-unmapped), at its defaults; a user running Kraken2 on all reads, or with a confidence threshold, would get different numbers. The reporting floor is applied to Kraken2 here to make sample-level calls comparable.",
            "One replicate per sample; the first 50 M read pairs of each GSE166766 run (about 600 M each).",
            "Resource rows are not like for like: Singlet's row is the whole pipeline from FASTQ, Kraken2's the classification of the unmapped reads only. Nodes differ between rows (see node)."]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


def main():
    bp, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sp_rows, meta_rows, res_rows = [], [], []
    for ds in DATASETS:
        d = bp / "nonhost" / RUN_DIR[ds]
        cells = [x.split("\t")[0].strip() for x in (bp / "nonhost" / "cells" / f"{RUN_DIR[ds]}.txt").read_text().split("\n") if x.strip()]
        base = pd.DataFrame({"barcode": cells}).set_index("barcode")
        # ---- Singlet
        sd = d / "singlet" / "nonhost"
        summ = json.loads((sd / "nonhost_summary.json").read_text())
        screened = int(summ["screened_reads"])
        em = pd.read_csv(sd / "nonhost_em_abundance.tsv", sep="\t") if (sd / "nonhost_em_abundance.tsv").exists() else pd.DataFrame()
        for _, r in em.iterrows():
            sp_rows.append({"dataset": ds, "tool": NH.TOOL_S, "taxid": r.species_id, "name": r.species_name, "domain": r.kingdom,
                            "reads": int(r.reads_assigned)})
        confirmed = set(em.species_id.astype(int)) if len(em) else set()
        pcf = sd / "nonhost_per_cell.tsv"
        pcs = pd.read_csv(pcf, sep="\t") if pcf.exists() and pcf.stat().st_size > 0 else pd.DataFrame(columns=["barcode", "species_name", "species_taxon_id", "n_umis"])
        if "confirmed" in pcs.columns:
            pcs = pcs[pcs.confirmed.astype(bool)]
        elif len(pcs):
            pcs = pcs[pcs.species_taxon_id.astype(int).isin(confirmed)]
        pcs = pcs[pcs.barcode.isin(base.index)]
        base["singlet_called_umi"] = pcs.groupby("barcode").n_umis.sum().reindex(base.index).fillna(0).astype(int)
        s2 = pcs[pcs.species_name.map(NH.is_sars2)]
        base["singlet_sars2_umi"] = s2.groupby("barcode").n_umis.sum().reindex(base.index).fillna(0).astype(int)
        # ---- Kraken2
        for tool, sub in KR.items():
            kd = d / sub
            ks = pd.read_csv(kd / "species.tsv", sep="\t")
            for _, r in ks.iterrows():
                if r.reads >= 50:  # keep the table small: species with a handful of reads are far below the floor
                    sp_rows.append({"dataset": ds, "tool": tool, "taxid": r.taxid, "name": r["name"], "domain": r.domain, "reads": int(r.reads)})
            fl = NH.floor_reads(screened)
            called = set(ks[ks.reads >= fl].taxid.astype(int))
            kp = pd.read_csv(kd / "percell.tsv.gz", sep="\t")
            kp = kp[kp.barcode.isin(base.index)]
            col = NH.COL[tool]
            base[f"{col}_called_umi"] = kp[kp.taxid.isin(called)].groupby("barcode").umis.sum().reindex(base.index).fillna(0).astype(int)
            base[f"{col}_sars2_umi"] = kp[kp["name"].map(NH.is_sars2)].groupby("barcode").umis.sum().reindex(base.index).fillna(0).astype(int)
            kt = gnu(kd / "run" / "time.txt")
            res_rows.append({"tool": tool, "dataset": ds, "wall_s": kt.get("wall", 0), "max_rss_gb": rss(kd / "run" / "mem.json"), "threads": 16,
                             "cpu_s": kt.get("cpu", 0), "node": read_nodes(kd / "node.txt"),
                             "note": "kraken2 classification of the host-unmapped reads only (excludes STAR and per-cell attribution)"})
        # ---- oracle
        orc = pd.read_csv(d / "oracle" / "percell.tsv", sep="\t")
        orc = orc[orc.barcode.isin(base.index)]
        base["oracle_sars2_umi"] = orc.groupby("barcode").umis.sum().reindex(base.index).fillna(0).astype(int)
        m = re.search(r"reads aligned: (\d+)", (d / "oracle" / "percell.log").read_text())
        sp_rows.append({"dataset": ds, "tool": NH.TOOL_O, "taxid": 2697049, "name": "Severe acute respiratory syndrome coronavirus 2 (NC_045512.2)",
                        "domain": "Viruses", "reads": int(m.group(1)) if m else 0})
        ot = gnu(d / "oracle" / "run" / "time.txt")
        res_rows.append({"tool": NH.TOOL_O, "dataset": ds, "wall_s": ot.get("wall", 0), "max_rss_gb": rss(d / "oracle" / "run" / "mem.json"),
                         "threads": 8, "cpu_s": ot.get("cpu", 0), "node": "", "note": "minimap2 -x sr of the host-unmapped reads against NC_045512.2"})
        # ---- Singlet resources: encode + full pipeline
        e, r_ = gnu(d / "singlet" / "encode" / "time.txt"), gnu(d / "singlet" / "run" / "time.txt")
        res_rows.append({"tool": NH.TOOL_S, "dataset": ds, "wall_s": e.get("wall", 0) + r_.get("wall", 0),
                         "max_rss_gb": max(rss(d / "singlet" / "encode" / "mem.json"), rss(d / "singlet" / "run" / "mem.json")), "threads": 16,
                         "cpu_s": e.get("cpu", 0) + r_.get("cpu", 0), "node": read_nodes(d / "singlet" / "node.txt"),
                         "note": "the whole pipeline from FASTQ (encode, STAR, pileup, counts, the viral/bacterial/fungal screen and every other module), not the screen alone"})
        meta_rows.append({"dataset": ds, "screened_reads": screened, "total_unmapped_reads": int(summ.get("total_unmapped_reads", screened)),
                          "n_cells": len(cells), "singlet_em_species": int(summ.get("em_species_count", len(em))),
                          "viral_fraction_raw": summ.get("viral_fraction"), "bacterial_fraction_raw": summ.get("bacterial_fraction"),
                          "fungal_fraction_raw": summ.get("fungal_fraction")})
        base.reset_index().to_csv(out / f"percell_{ds}.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(sp_rows).to_csv(out / "species.tsv", sep="\t", index=False)
    pd.DataFrame(meta_rows).to_csv(out / "meta.tsv", sep="\t", index=False)
    pd.DataFrame(res_rows).to_csv(out / "resources.tsv", sep="\t", index=False)
    write_provenance(bp, out)
    print("non-host tables written to", out)


if __name__ == "__main__":
    main()
