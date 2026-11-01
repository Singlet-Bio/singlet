"""Collect the donor-demux artifacts (labels only) from the run directories on Clipper.

Usage: export_demux.py RUNS TRUTH_DIR OUT_DIR
Writes OUT_DIR/donor_demux_cells.tsv.gz, donor_demux_pools.tsv, donor_demux_resources.tsv.
Per-cell donor labels are written; no genotype, allele count or SNP is.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "capabilities"))
import pools as P  # noqa: E402
import demux_eval as DE  # noqa: E402

DEV = P.DEV


def parse_time(f: Path):
    """(wall seconds, max RSS GB) from /usr/bin/time -v output."""
    if not f.exists():
        return None, None
    txt = f.read_text()
    m = re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", txt)
    parts = [float(x) for x in m.group(1).split(":")] if m else []
    wall = sum(v * 60 ** i for i, v in enumerate(reversed(parts))) if parts else None
    r = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    return wall, (int(r.group(1)) / 1048576 if r else None)


def main():
    runs, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    cells, pools, res = [], [], []
    for pool in P.all_pools():
        rd = P.rundir(runs, pool)
        if not (rd / "singlet_auto" / "donor" / "donor_assignments.tsv").exists():
            print("skip", pool); continue
        T = P.load_truth(truth_dir, pool)
        labs = {"singlet_auto": DE.read_singlet(rd / "singlet_auto")["label"]}
        # the cells every tool was given: Singlet's called cells (cell_calls.tsv, is_cell); donor_assignments.tsv also labels
        # lower-confidence barcodes that the competitors never saw
        cc = pd.read_csv(rd / "singlet_auto" / "cell_calls.tsv", sep="\t", usecols=["barcode", "is_cell"])
        cellset = pd.Index(cc.loc[cc["is_cell"].astype(str).str.upper() == "TRUE", "barcode"].map(DE.norm_bc))
        if T is None:   # single-donor pool: every called cell is the donor
            T = pd.DataFrame({"ind": "patient", "multiplets": "singlet"}, index=cellset)
        T = T[T["ind"] != "<NA>"]
        kt = P.pool_k(truth_dir, pool)
        told = rd / f"singlet_k{kt}"
        if (told / "donor" / "donor_assignments.tsv").exists():
            labs["singlet_told"] = DE.read_singlet(told)["label"]
        for s in (1, 2, 3):
            d = rd / "vireo" / f"told_s{s}"
            if (d / "donor_ids.tsv").exists():
                labs[f"vireo_told_s{s}"] = DE.read_vireo(d)["label"]
        if (rd / "vireo" / "n12" / "donor_ids.tsv").exists():
            labs["vireo_n12"] = DE.read_vireo(rd / "vireo" / "n12")["label"]
        if (rd / "souporcell" / "out" / "clusters.tsv").exists():
            labs["souporcell"] = DE.read_souporcell(rd / "souporcell" / "out")["label"]
        df = T.rename(columns={"ind": "truth_donor", "multiplets": "truth_class"}).copy()
        called = cellset.intersection(labs["singlet_auto"].index)
        df = df.loc[df.index.intersection(called)]
        for k, v in labs.items():
            df[k] = v.reindex(df.index).fillna("not_called")
        df.insert(0, "pool", pool)
        df.index.name = "barcode"
        cells.append(df.reset_index())
        k_sel, k_lab = DE.singlet_k_from_log(rd / "singlet_auto")
        row = dict(pool=pool, group="dev" if pool == DEV else "heldout", k_true=kt,
                   truth_cells=int(len(T)), truth_singlets=int((T["multiplets"] == "singlet").sum()),
                   singlet_k_selected=k_sel, singlet_k_labels=k_lab)
        for fr, nm in ((0.005, "05"), (0.01, "1"), (0.02, "2")):
            row[f"vireo_n12_khat_{nm}pct"] = DE.vireo_khat(rd / "vireo" / "n12", fr) if "vireo_n12" in labs else None
        pools.append(row)
        for tool, f, thr in (("singlet", rd / "singlet_auto" / "time.txt", 16), ("singlet_told", told / "time.txt", 16),
                             ("cellsnp-lite", rd / "vireo" / "cellsnp.time.txt", 16),
                             ("vireo_told", rd / "vireo" / f"vireo_k{kt}.s1.time.txt", 16),
                             ("vireo_n12", rd / "vireo" / "vireo_n12.time.txt", 16),
                             ("souporcell", rd / "souporcell" / "soup.time.txt", 16),
                             ("mgatk", rd / "mgatk" / "mgatk.time.txt", 8)):
            w, r = parse_time(f)
            if w is not None:
                res.append(dict(tool=tool, pool=pool, wall_s=round(w, 1), max_rss_gb=round(r or 0, 2), threads=thr))
    pd.concat(cells).to_csv(out / "donor_demux_cells.tsv.gz", sep="\t", index=False)
    pd.DataFrame(pools).to_csv(out / "donor_demux_pools.tsv", sep="\t", index=False)
    pd.DataFrame(res).to_csv(out / "donor_demux_resources.tsv", sep="\t", index=False)
    print(pd.DataFrame(pools).to_string())


if __name__ == "__main__":
    main()
