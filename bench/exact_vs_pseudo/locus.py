"""Inspect the reads behind a cross-species hot spot (default: human LINC00486 in mouse cells, hgmm1k_v3).

Input: a STAR coordinate-sorted BAM with CB (STARsolo --outSAMattributes NH HI AS nM CB UB), the published Cell Ranger h5
(species of each cell), and the gene-lines GTF. For each reads-in-locus group (cells that are mouse / human) it reports how
many reads and molecules, how unique, how many mismatches, how much of the locus is exonic, and writes the mouse-cell reads
as FASTQ so they can be re-mapped to the mouse genome alone (locus.sbatch).

  python locus.py --bam B --h5 H --gtf G --gene LINC00486 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loaders as L  # noqa: E402


def species_sets(h5):
    cr = L.load_cr_h5(h5)
    ish = np.array([g.startswith("ENSG") for g in cr.genes])
    tot = np.asarray(cr.X.sum(1)).ravel()
    fh = np.asarray(cr.X[:, ish].sum(1)).ravel() / np.maximum(tot, 1)
    ok = tot >= 500
    return set(cr.barcodes[ok & (fh >= 0.9)]), set(cr.barcodes[ok & (fh <= 0.1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True); ap.add_argument("--h5", required=True); ap.add_argument("--gtf", required=True)
    ap.add_argument("--gene", default="LINC00486"); ap.add_argument("--out", required=True)
    ap.add_argument("--samtools", default="samtools")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    human, mouse = species_sets(a.h5)
    # locus and exons from the GTF
    gid = chrom = None; start = end = 0; exons = []
    for ln in open(a.gtf):
        p = ln.split("\t")
        if len(p) < 9:
            continue
        if p[2] == "gene" and f'gene_name "{a.gene}"' in p[8]:
            gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1); chrom = p[0]; start, end = int(p[3]), int(p[4])
    if gid is None:
        sys.exit("gene not found")
    # exons are not in the gene-lines GTF: ask the full GTF through STAR's exonInfo instead? use gene body only here
    reg = f"{chrom}:{start}-{end}"
    cmd = [a.samtools, "view", a.bam, reg]
    stats = {"human_cells": Counter(), "mouse_cells": Counter(), "other": Counter()}
    nm_sum = {"human_cells": 0.0, "mouse_cells": 0.0}; alen_sum = {"human_cells": 0, "mouse_cells": 0}
    umis = {"human_cells": set(), "mouse_cells": set()}
    pos_bins = {"human_cells": Counter(), "mouse_cells": Counter()}
    fq = open(os.path.join(a.out, f"{a.gene}_mouse_cells.fastq"), "w")
    seen = set()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as pr:
        for ln in pr.stdout:
            f = ln.rstrip("\n").split("\t")
            flag = int(f[1])
            if flag & 0x100:
                continue
            tags = {t[:2]: t[5:] for t in f[11:]}
            cb = tags.get("CB", "-")
            cb = cb.split("-")[0]
            grp = "human_cells" if cb in human else "mouse_cells" if cb in mouse else "other"
            s = stats[grp]
            s["reads"] += 1
            nh = int(tags.get("NH", "1"))
            s["NH1" if nh == 1 else "NHgt1"] += 1
            if grp == "other":
                continue
            cig = re.findall(r"(\d+)([MIDNS])", f[5])
            alen = sum(int(n) for n, o in cig if o == "M")
            nsp = sum(1 for n, o in cig if o == "N")
            nm_sum[grp] += int(tags.get("nM", 0)); alen_sum[grp] += alen
            if nsp:
                s["spliced"] += 1
            s["mapq255" if int(f[4]) == 255 else "mapq_lt255"] += 1
            if tags.get("UB", "-") not in ("-", "0"):
                umis[grp].add((cb, tags["UB"]))
            pos_bins[grp][int(f[3]) // 5000 * 5000] += 1
            if grp == "mouse_cells" and f[0] not in seen:
                seen.add(f[0])
                seq, qual = f[9], f[10]
                if flag & 0x10:  # restore the sequenced orientation
                    comp = str.maketrans("ACGTN", "TGCAN")
                    seq = seq.translate(comp)[::-1]; qual = qual[::-1]
                fq.write(f"@{f[0]}\n{seq}\n+\n{qual}\n")
    fq.close()
    res = {"gene": a.gene, "gene_id": gid, "region": reg, "length_bp": end - start + 1}
    for g in ("human_cells", "mouse_cells"):
        res[g] = {**dict(stats[g]), "molecules": len(umis[g]),
                  "mismatches_per_aligned_base": nm_sum[g] / max(alen_sum[g], 1),
                  "top_5kb_bins": pos_bins[g].most_common(5)}
    res["other_cells_reads"] = stats["other"]["reads"]
    json.dump(res, open(os.path.join(a.out, f"{a.gene}_locus.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
