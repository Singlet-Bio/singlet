# SPDX-License-Identifier: MIT
"""Diagnostic for B6: re-name the genes of the engine's chimeric_junctions.tsv.gz with the transcript strand instead of the read strand.

The engine names each junction side by the gene on the strand STAR reports, which is the read's strand. On an antisense library
(10x 5' R2) that prefers an antisense lncRNA over the real gene (RUNX1 -> ENSG00000286153). This rewrites donor_gene / acceptor_gene
with the strand the splice motif implies (junction_type 1: as reported; 2: opposite) and, for non-canonical junctions (type 0), the
library strand (--antisense). Same tie rules as chimeric_table.h gene_at (same-strand genes first, then the shortest span).

    python reannotate_strand.py --table chimeric_junctions.tsv.gz --gtf genes.gtf --antisense --out fixed.tsv.gz
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import re


def load_genes(gtf):
    idx = collections.defaultdict(list)
    with (gzip.open(gtf, "rt") if gtf.endswith(".gz") else open(gtf)) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8]) or re.search(r'gene_id "([^"]+)"', p[8])
            idx[p[0]].append((int(p[3]), int(p[4]), p[6], m.group(1)))
    out = {}
    for c, g in idx.items():
        g.sort()
        out[c] = (g, [x[0] for x in g], max(x[1] - x[0] for x in g))
    return out


def gene_at(idx, chr_, pos, strand):
    if chr_ not in idx:
        return "."
    g, starts, maxlen = idx[chr_]
    lo = bisect.bisect_left(starts, pos - maxlen)
    hi = bisect.bisect_right(starts, pos)
    best = None
    for s, e, st, name in g[lo:hi]:
        if pos < s or pos > e:
            continue
        key = (st != strand, e - s, name)
        if best is None or key < best[0]:
            best = (key, name)
    return best[1] if best else "."


def flip(s):
    return "-" if s == "+" else "+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--antisense", action="store_true", help="library reads are antisense to the transcript (10x 5' R2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = load_genes(a.gtf)
    changed = n = 0
    with gzip.open(a.table, "rt") as f, gzip.open(a.out, "wt") as o:
        head = f.readline()
        o.write(head)
        h = head.rstrip("\n").split("\t")
        c = {k: i for i, k in enumerate(h)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            jt = int(p[c["junction_type"]])
            ds, as_ = p[c["donor_strand"]], p[c["acceptor_strand"]]
            if jt == 2 or (jt == 0 and a.antisense):
                ds, as_ = flip(ds), flip(as_)
            dg = gene_at(idx, p[c["donor_chr"]], int(p[c["donor_pos"]]), ds)
            ag = gene_at(idx, p[c["acceptor_chr"]], int(p[c["acceptor_pos"]]), as_)
            n += 1
            if dg != p[c["donor_gene"]] or ag != p[c["acceptor_gene"]]:
                changed += 1
            p[c["donor_gene"]], p[c["acceptor_gene"]] = dg, ag
            p[c["fusion"]] = f"{dg}--{ag}"
            o.write("\t".join(p) + "\n")
    print(f"{changed} of {n} rows renamed")


if __name__ == "__main__":
    main()
