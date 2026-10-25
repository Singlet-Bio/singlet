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
    t = p.read_text(errors="replace") if p.exists() else ""
    m = re.search(r"node=(\S+?)(?:\.clipper\S*)? cpu=(.*?) 20\d\d-", t)
    return f"{m.group(1)} ({m.group(2)})" if m else ""


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
    print("non-host tables written to", out)


if __name__ == "__main__":
    main()
