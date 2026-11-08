# SPDX-License-Identifier: MIT
"""Cell-line labels for the B6 held-out library (GSE329598) from hashtag reads, named by the PacBio BCR::ABL1 table.
Rules as registered in ORACLES_B6_B8.md ("B6 held-out run: registration"): distinct UMIs per exact whitelist barcode for each
hashtag (searched in the first 40 bases of the feature read), a cell takes a hashtag holding >= 70% of its hashtag UMIs and >= 10
UMIs; the hashtag whose cells more often carry >= 1 BCR::ABL1 long read is Ku812. Stops (exit 3) when fewer than half of the
feature reads contain a hashtag.

    python hto_labels.py --r1 a_1.fastq.gz,b_1.fastq.gz --r2 a_2.fastq.gz,b_2.fastq.gz --whitelist 3M-february-2018.txt \
        --pacbio long_read_count_table.csv.gz --out labels.tsv --summary labels.json
"""
import argparse, collections, csv, gzip, io, json, re, sys

HTO = {"B0253": "TTCCGCCTCTCTTTG", "B0254": "AGTAAGTTCAGCGTA"}


def fq_pairs(r1s, r2s):
    for a, b in zip(r1s, r2s):
        with gzip.open(a, "rt") as f1, gzip.open(b, "rt") as f2:
            while True:
                h = f1.readline()
                if not h:
                    return
                s1 = f1.readline(); f1.readline(); f1.readline()
                f2.readline(); s2 = f2.readline(); f2.readline(); f2.readline()
                yield s1, s2


def pacbio_bcr_abl(path):
    """Barcodes with >= 1 BCR::ABL1 long read. The table layout is read from its header: a barcode column (16 ACGT, optional -1)
    and the columns whose name mentions both BCR and ABL1."""
    with gzip.open(path, "rt") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    fus = [i for i, h in enumerate(head) if re.search("bcr", h, re.I) and re.search("abl", h, re.I)]
    bc = None
    for i in range(len(head)):
        if all(re.fullmatch(r"[ACGT]{16}(-1)?", r[i] or "") for r in rows[1:20] if len(r) > i):
            bc = i; break
    if bc is None or not fus:
        # long format: one row per barcode and feature
        print("pacbio header:", head[:12], "first row:", rows[1][:12] if len(rows) > 1 else None, file=sys.stderr)
        fcol = next((i for i, h in enumerate(head) if re.search("feature|gene|fusion|target|allele", h, re.I)), None)
        ccol = next((i for i, h in enumerate(head) if re.search("count|reads|umi|n$", h, re.I)), None)
        if bc is None or fcol is None:
            sys.exit("pacbio table: no barcode or BCR::ABL1 column found")
        pos = set()
        for r in rows[1:]:
            if re.search("bcr", r[fcol], re.I) and re.search("abl", r[fcol], re.I) and (ccol is None or float(r[ccol] or 0) > 0):
                pos.add(r[bc].split("-")[0])
        return pos, {"layout": "long", "barcode_col": head[bc], "feature_col": head[fcol]}
    pos = {r[bc].split("-")[0] for r in rows[1:] if any(float(r[i] or 0) > 0 for i in fus)}
    return pos, {"layout": "wide", "barcode_col": head[bc], "bcr_abl_cols": [head[i] for i in fus]}


def main():
    ap = argparse.ArgumentParser()
    for a in ("r1", "r2", "whitelist", "pacbio", "out", "summary"):
        ap.add_argument("--" + a, required=True)
    x = ap.parse_args()
    wl = {l.strip() for l in open(x.whitelist) if l.strip()}
    umis = collections.defaultdict(lambda: {k: set() for k in HTO})
    n = hit = 0
    for s1, s2 in fq_pairs(x.r1.split(","), x.r2.split(",")):
        n += 1
        w = s2[:40]
        for k, seq in HTO.items():
            if seq in w:
                hit += 1
                cb = s1[:16]
                if cb in wl:
                    umis[cb][k].add(s1[16:28])
                break
    frac = hit / max(n, 1)
    summ = {"feature_reads": n, "with_hashtag": hit, "fraction": round(frac, 4)}
    if frac < 0.5:
        json.dump(summ, open(x.summary, "w"), indent=1); print(summ); sys.exit(3)
    lab = {}
    for cb, d in umis.items():
        c = {k: len(v) for k, v in d.items()}
        tot = sum(c.values()); k, top = max(c.items(), key=lambda kv: kv[1])
        if tot >= 10 and top >= 0.7 * tot:
            lab[cb] = k
    pos, layout = pacbio_bcr_abl(x.pacbio)
    rate = {k: (sum(1 for c, v in lab.items() if v == k and c in pos), sum(1 for v in lab.values() if v == k)) for k in HTO}
    ku = max(HTO, key=lambda k: rate[k][0] / max(rate[k][1], 1))
    name = {k: ("Ku812" if k == ku else "HMC1.2") for k in HTO}
    with open(x.out, "w") as f:
        f.write("barcode\tline\n")
        for cb, k in sorted(lab.items()):
            f.write(f"{cb}\t{name[k]}\n")
    with open(x.out.replace(".tsv", "_pacbio_bcr_abl.txt"), "w") as f:
        f.write("".join(c + "\n" for c in sorted(pos)))
    summ.update({"cells_labelled": len(lab), "per_hashtag": {k: {"line": name[k], "cells": rate[k][1], "cells_with_bcr_abl_long_read": rate[k][0]} for k in HTO},
                 "pacbio_bcr_abl_barcodes": len(pos), "pacbio_layout": layout})
    json.dump(summ, open(x.summary, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
