#!/usr/bin/env python3
"""Build a transcriptome FASTA and a transcript table from a genome FASTA and the GTF of the STAR index.
Usage: build_tx.py GENOME.fa GENES.gtf OUTDIR
Writes OUTDIR/tx.fa (transcript sequences, 5' to 3') and OUTDIR/tx.tsv:
  tx  chrom  strand  gene  biotype  exons (genomic ascending, 1-based inclusive, "s-e,s-e")
"""
import re
import sys
from collections import defaultdict

genome, gtf, out = sys.argv[1:4]
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
seqs, name, buf = {}, None, []
for l in open(genome):
    if l.startswith(">"):
        if name:
            seqs[name] = "".join(buf)
        name, buf = l[1:].split()[0], []
    else:
        buf.append(l.strip())
seqs[name] = "".join(buf)

attr = re.compile(r'(\w+) "([^"]*)"')
ex = defaultdict(list)
meta = {}
for l in open(gtf):
    if l.startswith("#"):
        continue
    f = l.rstrip("\n").split("\t")
    if f[2] != "exon":
        continue
    a = dict(attr.findall(f[8]))
    t = a["transcript_id"]
    ex[t].append((int(f[3]), int(f[4])))
    meta[t] = (f[0], f[6], a.get("gene_id", ""), a.get("gene_type", a.get("gene_biotype", "NA")))

with open(out + "/tx.fa", "w") as fa, open(out + "/tx.tsv", "w") as tsv:
    for t, e in ex.items():
        chrom, strand, gene, bt = meta[t]
        if chrom not in seqs:
            continue
        e.sort()
        s = "".join(seqs[chrom][a - 1:b] for a, b in e).upper()
        if strand == "-":
            s = s.translate(COMP)[::-1]
        fa.write(">%s\n%s\n" % (t, s))
        tsv.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (t, chrom, strand, gene, bt, ",".join("%d-%d" % x for x in e)))
print("transcripts", len(ex))
