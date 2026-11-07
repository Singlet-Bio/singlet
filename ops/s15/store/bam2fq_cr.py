#!/usr/bin/env python3
"""Rebuild 10x-style reads from `samtools fastq -T CR,CQ,UR,UQ` records on stdin: R1 = CR+UR (barcode+UMI) with CQ+UQ, R2 = the cDNA read.
usage: samtools fastq -F 2304 -T CR,CQ,UR,UQ X.bam | bam2fq_cr.py OUT_1.fastq.gz OUT_2.fastq.gz"""
import sys, gzip
o1 = gzip.open(sys.argv[1], "wt", compresslevel=1); o2 = gzip.open(sys.argv[2], "wt", compresslevel=1)
n = 0; seen = 0
while True:
    h = sys.stdin.readline()
    if not h: break
    sq = sys.stdin.readline(); sys.stdin.readline(); ql = sys.stdin.readline(); seen += 1
    f = h.rstrip("\n").split("\t"); nm = f[0]; t = {x[:2]: x[5:] for x in f[1:]}
    cr, cq, ur, uq = t.get("CR"), t.get("CQ"), t.get("UR"), t.get("UQ")
    if cr and cq and ur and uq:
        o1.write("%s\n%s%s\n+\n%s%s\n" % (nm, cr, ur, cq, uq)); o2.write("%s\n%s+\n%s" % (nm, sq, ql)); n += 1
o1.close(); o2.close(); sys.stderr.write("%d\n" % n); print("records %d kept %d" % (seen, n))
