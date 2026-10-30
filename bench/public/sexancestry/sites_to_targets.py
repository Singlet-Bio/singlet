#!/usr/bin/env python3
"""peddy's GRCH38.sites (one 'chrom:pos:ref:alt' per line, chrom with or without 'chr') -> a bcftools -T targets file
(chrom<TAB>pos) named like the 10x GRCh38-2024-A FASTA ('chr1').

    python sites_to_targets.py GRCH38.sites targets.tsv
"""
import sys

src, out = sys.argv[1:3]
n = 0
with open(src) as fh, open(out, "w") as o:
    for line in fh:
        parts = line.strip().replace("\t", ":").split(":")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        c = parts[0] if parts[0].startswith("chr") else "chr" + parts[0]
        o.write(f"{c}\t{parts[1]}\n")
        n += 1
print(n, "peddy sites")
