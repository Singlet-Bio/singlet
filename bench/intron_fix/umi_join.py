"""Per-UMI join of Singlet's molecule labels with STARsolo Velocyto's (real data).

    python umi_join.py PB DS SINGLET_TAG GTF OUT.json

Singlet: runs/<ds>/<tag>/molecules.tsv.gz (SINGLET_MOLECULE_DUMP: barcode, gene_id, umi, label, flags, diag, reads).
STARsolo: runs/<ds>/starsolo-velodump/velo_umis.tsv.gz (CB, UMI, gene|-, S/U/A, or M = multi-gene UMI, E = empty
transcript intersection; both not counted). Cells: STARsolo EmptyDrops_CR calls (starsolo-public-cr Gene/filtered).
Molecules are matched on (cell barcode, UMI): Singlet collapses 1-Hamming UMIs per gene (directional), STARsolo corrects
UMIs per gene (1MM_CR) only for exonic reads, so a small share of molecules fails to match for that reason alone.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PB, DS, TAG, GTF, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
RUNS = PB / "runs" / DS
DIAG = {1: "multi", 2: "spliced", 4: "flank", 8: "othergene", 16: "antisense", 32: "outside"}
GENE_RE = re.compile(r"(ENS(?:MUS)?G\d{11}(?:_PAR_Y)?)")


def canon(g: str) -> str:
    m = GENE_RE.search(g)
    return m.group(1) if m else g


def bare(b: str) -> str:
    return b.split("-")[0]


def gene_table(gtf: Path) -> pd.DataFrame:
    """Gene body, strand and overlaps with other genes (same / opposite strand) from the reference GTF."""
    rows = []
    op = gzip.open if str(gtf).endswith(".gz") else open
    with op(gtf, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t", 9)
            if len(p) < 9 or p[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            nm = re.search(r'gene_name "([^"]+)"', p[8])
            gt = re.search(r'gene_(?:bio)?type "([^"]+)"', p[8])
            rows.append((canon(gid), p[0], int(p[3]), int(p[4]), p[6], nm.group(1) if nm else gid, gt.group(1) if gt else ""))
    g = pd.DataFrame(rows, columns=["gene", "chrom", "start", "end", "strand", "name", "type"]).drop_duplicates("gene")
    g["length"] = g["end"] - g["start"] + 1
    same = np.zeros(len(g), bool)
    anti = np.zeros(len(g), bool)
    for chrom, sub in g.groupby("chrom"):
        sub = sub.sort_values("start")
        idx, st, en, sd = sub.index.to_numpy(), sub["start"].to_numpy(), sub["end"].to_numpy(), sub["strand"].to_numpy()
        for i in range(len(sub)):
            j = i + 1
            while j < len(sub) and st[j] <= en[i]:
                if sd[i] == sd[j]:
                    same[g.index.get_loc(idx[i])] = same[g.index.get_loc(idx[j])] = True
                else:
                    anti[g.index.get_loc(idx[i])] = anti[g.index.get_loc(idx[j])] = True
                j += 1
    g["overlap_same"] = same
    g["overlap_anti"] = anti
    return g.set_index("gene")


def main():
    cells = {bare(l.strip()) for l in open(RUNS / "starsolo-public-cr" / "Solo.out" / "Gene" / "filtered" / "barcodes.tsv")}
    s = pd.read_csv(RUNS / TAG / "molecules.tsv.gz", sep="\t", dtype={"barcode": str, "gene_id": str, "umi": str})
    s["bc"] = s["barcode"].map(bare)
    s = s[s["bc"].isin(cells)].copy()
    s["gene"] = s["gene_id"].map(canon)
    v = pd.read_csv(RUNS / "starsolo-velodump" / "velo_umis.tsv.gz", sep="\t", header=None, names=["bc", "umi", "gene", "label"],
                    dtype=str)
    v["bc"] = v["bc"].map(bare)
    v = v[v["bc"].isin(cells)].copy()
    v["gene"] = v["gene"].map(lambda x: canon(x) if x != "-" else "-")
    ulen = int(v["umi"].str.len().mode()[0])
    s["umi"] = s["umi"].str[:ulen]  # Singlet prints config umi_len (12) bases; 10x v2 UMIs are 10
    genes = gene_table(GTF)

    # genes per (bc, umi) in Singlet (a UMI in more than one gene of the cell)
    s["n_genes_umi"] = s.groupby(["bc", "umi"])["gene"].transform("size")
    vk = v.drop_duplicates(["bc", "umi"]).rename(columns={"gene": "gene_star", "label": "label_star"})
    m = s.merge(vk, on=["bc", "umi"], how="left")

    def star_class(r_label, r_gene, s_gene, star_label):
        if pd.isna(star_label):
            return "star_none"
        if star_label in ("M", "E"):
            return f"star_{star_label}"
        if r_gene != s_gene:
            return "star_other_gene"
        return f"star_{star_label}"

    m["cls"] = [star_class(a, b, c, d) for a, b, c, d in zip(m["label"], m["gene_star"], m["gene"], m["label_star"])]
    afp = RUNS / "alevinfry-umis" / "af_umis.tsv.gz"
    if afp.exists():
        a = pd.read_csv(afp, sep="	", header=None, names=["bc", "umi", "gene", "label"], dtype=str)
        a["gene"] = a["gene"].map(lambda x: canon(x) if x != "-" else "-")
        ak = a.drop_duplicates(["bc", "umi"]).rename(columns={"gene": "gene_af", "label": "label_af"})
        m = m.merge(ak, on=["bc", "umi"], how="left")
        m["cls_af"] = [star_class(x, b, c, d).replace("star_", "af_") for x, b, c, d in
                       zip(m["label"], m["gene_af"], m["gene"], m["label_af"])]
    else:
        a = None
    res = {"dataset": DS, "n_cells": len(cells), "singlet_molecules": int(len(s)), "star_umis": int(len(v)),
           "star_labels": v["label"].value_counts().to_dict(), "singlet_labels": s["label"].value_counts().to_dict()}
    # where Singlet's labels land in STARsolo
    res["cross"] = {lab: m[m["label"] == lab]["cls"].value_counts().to_dict() for lab in ("S", "U", "A", "D")}

    def diag_breakdown(sub):
        d = sub["diag"].to_numpy(int)
        out = {name: int(((d & bit) != 0).sum()) for bit, name in DIAG.items()}
        out["none"] = int((d == 0).sum())
        out["umi_in_2plus_genes"] = int((sub["n_genes_umi"] > 1).sum())
        gi = genes.reindex(sub["gene"])
        out["gene_overlap_same"] = int(gi["overlap_same"].fillna(False).astype(bool).sum())
        out["gene_overlap_anti"] = int(gi["overlap_anti"].fillna(False).astype(bool).sum())
        out["gene_len_ge_100kb"] = int((gi["length"].fillna(0) >= 100_000).sum())
        out["reads_1"] = int((sub["reads"] == 1).sum())
        out["n"] = int(len(sub))
        return out

    if a is not None:
        res["af_labels"] = a["label"].value_counts().to_dict()
        res["cross_af"] = {lab: m[m["label"] == lab]["cls_af"].value_counts().to_dict() for lab in ("S", "U", "A", "D")}
        Uj = m[m["label"] == "U"]
        res["singlet_U_star_x_af"] = pd.crosstab(Uj["cls"], Uj["cls_af"]).to_dict()
        # both others give no unspliced molecule of this gene for this (cell, UMI)
        notU = Uj[(Uj["cls"] != "star_U") & (Uj["cls_af"] != "af_U")]
        res["singlet_U_neither_other_U"] = int(len(notU))
        res["af_U_total_in_cells"] = int((a["label"] == "U").sum())
        res["star_U_total_in_cells"] = int((v["label"] == "U").sum())
    U = m[m["label"] == "U"]
    res["singlet_U_by_class"] = {c: diag_breakdown(U[U["cls"] == c]) for c in sorted(U["cls"].unique())}
    if a is not None:
        res["singlet_U_by_af_class"] = {c: diag_breakdown(U[U["cls_af"] == c]) for c in sorted(U["cls_af"].unique())}
        res["singlet_U_neither"] = diag_breakdown(U[(U["cls"] != "star_U") & (U["cls_af"] != "af_U")])
        res["singlet_U_both_agree"] = diag_breakdown(U[(U["cls"] == "star_U") & (U["cls_af"] == "af_U")])
    # agreed U as reference for the diag rates
    # STARsolo U that Singlet does not call U for the same gene
    vs = v[v["label"] == "U"].merge(s[["bc", "umi", "gene", "label", "diag", "n_genes_umi"]], on=["bc", "umi"], how="left",
                                    suffixes=("", "_s"))
    vs["same_gene"] = vs["gene"] == vs["gene_s"]
    agg = vs.groupby(["bc", "umi"]).agg(any_same=("same_gene", "max"), lab=("label_s", lambda x: ",".join(sorted(set(map(str, x))))))
    sameg = vs[vs["same_gene"]]
    res["star_U_in_singlet"] = {"n": int(len(agg)), "singlet_none": int(vs.groupby(["bc", "umi"])["gene_s"].apply(lambda x: x.isna().all()).sum()),
                                "same_gene_label": sameg["label_s"].value_counts().to_dict(),
                                "other_gene_only": int((~agg["any_same"] & (agg["lab"] != "nan")).sum())}
    # genes carrying the excess: Singlet U molecules with no STARsolo U for that (cell, UMI, gene)
    ex = U[U["cls"] != "star_U"]
    top = ex.groupby("gene").size().sort_values(ascending=False).head(25)
    res["top_excess_genes"] = [(g, genes["name"].get(g, g), int(n), int(genes["length"].get(g, 0)),
                                bool(genes["overlap_same"].get(g, False)), bool(genes["overlap_anti"].get(g, False))) for g, n in top.items()]
    # excess share in genes by length / overlap category
    gi = genes.reindex(U["gene"])
    U = U.assign(long=(gi["length"].to_numpy() >= 100_000), ovs=gi["overlap_same"].fillna(False).to_numpy(bool),
                 ova=gi["overlap_anti"].fillna(False).to_numpy(bool), agree=(U["cls"] == "star_U").to_numpy())
    res["U_agreement_by_gene_class"] = {
        k: {"n": int(len(x)), "agree_frac": float(x["agree"].mean()) if len(x) else None}
        for k, x in (("all", U), ("long_ge_100kb", U[U["long"]]), ("short", U[~U["long"]]), ("overlap_same", U[U["ovs"]]),
                     ("overlap_anti", U[U["ova"]]), ("no_overlap", U[~U["ovs"] & ~U["ova"]]),
                     ("multi", U[(U["diag"] & 1) != 0]), ("unique", U[(U["diag"] & 1) == 0]),
                     ("spliced_read", U[(U["diag"] & 2) != 0]), ("flank", U[(U["diag"] & 4) != 0]),
                     ("antisense_overlap", U[(U["diag"] & 16) != 0]), ("umi_2plus_genes", U[U["n_genes_umi"] > 1]))}
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
