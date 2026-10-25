# SPDX-License-Identifier: MIT
"""Print the GTF gene lines of the given gene names (why a segment is missing from the reference)."""
import gzip
import sys

names = set(sys.argv[2:])
with gzip.open(sys.argv[1], "rt") as f:
    for ln in f:
        if ln.startswith("#"):
            continue
        p = ln.split("\t")
        if p[2] not in ("gene", "transcript", "exon", "CDS"):
            continue
        for n in names:
            if f'gene_name "{n}"' in p[8]:
                print(p[0], p[2], p[3], p[4], p[6], p[8][:300].strip())
