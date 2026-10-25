"""Rebuild the Multiome ATAC->GEX barcode pairs and the Visium v1 spot list from 10x public datasets (CC BY 4.0).

Inputs (download.sbatch): <data>/arc_pbm/*_per_barcode_metrics.csv, <data>/pbmc_granulocyte_sorted_3k/..._per_barcode_metrics.csv,
<data>/V1_*/*_spatial.tar.gz. Outputs in <out>: 737K-arc-v1-atac-gex.tsv.gz, visium-v1.txt, visium-v1_coordinates.tsv,
lists_report.json (coverage: leave-one-dataset-out fraction of called cells whose pair is known; coordinate consistency).
"""
import glob, gzip, io, json, os, sys, tarfile
import pandas as pd

data, out, wl_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rep = {}

# ── Multiome pairs ──
pbm = sorted(glob.glob(f"{data}/arc_pbm/*_per_barcode_metrics.csv") +
             glob.glob(f"{data}/pbmc_granulocyte_sorted_3k/*_per_barcode_metrics.csv"))
tabs = {}
for f in pbm:
    name = os.path.basename(f).replace("_per_barcode_metrics.csv", "")
    d = pd.read_csv(f, usecols=["gex_barcode", "atac_barcode", "is_cell"])
    d["gex_barcode"] = d.gex_barcode.str.replace("-1", "", regex=False)
    d["atac_barcode"] = d.atac_barcode.str.replace("-1", "", regex=False)
    tabs[name] = d
allp = pd.concat(tabs.values()).drop_duplicates(["atac_barcode", "gex_barcode"])
conf_a = allp.atac_barcode.duplicated(keep=False).sum()
conf_g = allp.gex_barcode.duplicated(keep=False).sum()
rep["pairs"] = {"datasets": {k: int(len(v)) for k, v in tabs.items()}, "union_pairs": int(len(allp)),
                "atac_with_two_gex": int(conf_a), "gex_with_two_atac": int(conf_g)}
loo = {}
for k, v in tabs.items():
    others = set(pd.concat([w for kk, w in tabs.items() if kk != k]).atac_barcode)
    cells = v[v.is_cell == 1]
    loo[k] = {"cells": int(len(cells)), "pair_known_from_others": float(cells.atac_barcode.isin(others).mean())}
rep["pairs"]["leave_one_out"] = loo
# Is the ATAC barcode a function of the GEX barcode's line in the GEX list (same line of another known list)?
gexwl = [l.strip() for l in open(f"{wl_dir}/gex_737K-arc-v1.txt") if l.strip()]
idx = {b: i for i, b in enumerate(gexwl)}
allp["gex_line"] = allp.gex_barcode.map(idx)
rep["pairs"]["gex_in_gex_list"] = float(allp.gex_line.notna().mean())
crat = [l.strip() for l in open(f"{wl_dir}/737K-cratac-v1.txt") if l.strip()]
rc = lambda s: s[::-1].translate(str.maketrans("ACGT", "TGCA"))
m = allp.dropna(subset=["gex_line"]).copy()
m["gex_line"] = m.gex_line.astype(int)
rep["pairs"]["atac_eq_cratac_same_line"] = float((m.atac_barcode == m.gex_line.map(lambda i: crat[i] if i < len(crat) else "")).mean())
rep["pairs"]["atac_eq_revcomp_gex"] = float((m.atac_barcode == m.gex_barcode.map(rc)).mean())
with gzip.open(f"{out}/737K-arc-v1-atac-gex.tsv.gz", "wt") as fo:
    fo.write("# ATAC barcode <TAB> GEX barcode, pairs observed in 10x Genomics public Cell Ranger ARC outputs (CC BY 4.0)\n")
    for a, g in allp.sort_values("atac_barcode")[["atac_barcode", "gex_barcode"]].itertuples(index=False):
        fo.write(f"{a}\t{g}\n")

# ── Visium v1 ──
coords = {}
for t in sorted(glob.glob(f"{data}/V1_*/*_spatial.tar.gz")):
    with tarfile.open(t) as tf:
        mem = [m for m in tf.getmembers() if m.name.endswith("tissue_positions_list.csv") or m.name.endswith("tissue_positions.csv")][0]
        txt = tf.extractfile(mem).read().decode()
    hdr = txt.startswith("barcode")
    d = pd.read_csv(io.StringIO(txt), header=0 if hdr else None)
    d.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    d["barcode"] = d.barcode.str.replace("-1", "", regex=False)
    coords[os.path.basename(t).replace("_spatial.tar.gz", "")] = d.set_index("barcode")[["array_row", "array_col"]]
names = list(coords)
ref = coords[names[0]]
rep["visium"] = {"datasets": {k: int(len(v)) for k, v in coords.items()},
                 "identical_to_first": {k: bool(v.sort_index().equals(ref.sort_index())) for k, v in coords.items()}}
ref = ref.sort_values(["array_row", "array_col"])
with open(f"{out}/visium-v1.txt", "w") as fo:
    fo.write("\n".join(ref.index) + "\n")
ref.to_csv(f"{out}/visium-v1_coordinates.tsv", sep="\t")
json.dump(rep, open(f"{out}/lists_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
