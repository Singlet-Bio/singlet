# SPDX-License-Identifier: MIT
"""Subfamily abundance of transposable elements from nanopore reads of the same library (scNanoGPS A375, GSE212945).

Long reads place a TE-derived transcript without the ambiguity short reads have in young, repetitive subfamilies, so their pseudobulk
subfamily counts are the truth for the short-read counters. A read counts when its primary alignment has MAPQ >= --min-mapq, overlaps
no annotated gene exon (the rule all three counters share: exonic reads belong to genes), and has at least --min-te-bases of its aligned
bases inside TE loci; it goes to the subfamily covering most of those bases (ties dropped).

    python te_longread_truth.py --bam ont.bam --te-gtf GRCh38_GENCODE_rmsk_TE.gtf.gz --gene-gtf genes.gtf --out truth.tsv
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import re

import pysam


def opener(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


class Intervals:
    """Sorted, possibly overlapping intervals per contig; query by a window of max length."""

    def __init__(self):
        self.d = collections.defaultdict(list)

    def add(self, chrom, s, e, v):
        self.d[chrom].append((s, e, v))

    def freeze(self):
        self.idx = {}
        for c, lst in self.d.items():
            lst.sort()
            starts = [x[0] for x in lst]
            maxlen = max(x[1] - x[0] for x in lst)
            self.idx[c] = (starts, lst, maxlen)

    def query(self, chrom, s, e):
        if chrom not in self.idx:
            return
        starts, lst, maxlen = self.idx[chrom]
        i = bisect.bisect_left(starts, s - maxlen)
        while i < len(lst) and lst[i][0] < e:
            a, b, v = lst[i]
            if b > s:
                yield a, b, v
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True)
    ap.add_argument("--te-gtf", required=True)
    ap.add_argument("--gene-gtf", required=True)
    ap.add_argument("--min-mapq", type=int, default=20)
    ap.add_argument("--min-te-bases", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gid = re.compile(r'gene_id "([^"]+)"')
    te = Intervals()
    with opener(a.te_gtf) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t")
            te.add(p[0], int(p[3]) - 1, int(p[4]), gid.search(p[8]).group(1))
    te.freeze()
    ex = Intervals()
    with opener(a.gene_gtf) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t")
            if len(p) > 8 and p[2] == "exon":
                ex.add(p[0], int(p[3]) - 1, int(p[4]), 1)
    ex.freeze()

    counts = collections.Counter()
    stats = collections.Counter()
    with pysam.AlignmentFile(a.bam) as bam:
        for r in bam:
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            stats["primary"] += 1
            if r.mapping_quality < a.min_mapq:
                stats["low_mapq"] += 1
                continue
            blocks = r.get_blocks()
            chrom = r.reference_name
            if any(True for s, e in blocks for _ in ex.query(chrom, s, e)):
                stats["exonic"] += 1
                continue
            per = collections.Counter()
            for s, e in blocks:
                for ts, tend, sub in te.query(chrom, s, e):
                    per[sub] += min(e, tend) - max(s, ts)
            if not per or sum(per.values()) < a.min_te_bases:
                stats["no_te"] += 1
                continue
            top = per.most_common(2)
            if len(top) > 1 and top[0][1] == top[1][1]:
                stats["tie"] += 1
                continue
            counts[top[0][0]] += 1
            stats["assigned"] += 1
    with open(a.out, "w") as o:
        o.write("subfamily\tlong_reads\n")
        for k, v in sorted(counts.items()):
            o.write(f"{k}\t{v}\n")
    print(dict(stats))


if __name__ == "__main__":
    main()
