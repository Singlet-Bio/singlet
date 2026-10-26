# SPDX-License-Identifier: MIT
"""Where observed protospacer windows sit relative to sgRNA coordinates (to pin the id convention and genome build)."""
import gzip
import re
import sys
from collections import Counter

import anndata as ad

d, genome = sys.argv[1], sys.argv[2]
_C = str.maketrans("ACGTN", "TGCAN")
seen = Counter()
with gzip.open(f"{d}/fastqs/SRR19331011_S1_L001_R2_001.fastq.gz", "rt") as f:
    for i, ln in enumerate(f):
        if i % 4 == 1:
            for anc in ("GTTTCAGAGC", "GTTTAAGAGC"):
                j = ln.find(anc)
                if j >= 20:
                    seen[ln[j - 19:j]] += 1
        if i > 1_600_000:
            break
top = {w for w, c in seen.most_common(20000)}
obs = ad.read_h5ad(f"{d}/K562_essential_raw_singlecell_01.h5ad", backed="r").obs
ids = []
for ab in obs["sgID_AB"].astype(str).unique()[:400]:
    for sg in ab.split("|"):
        m = re.match(r"(.+)_([+-])_(\d+)\.(\d+)-(.+)$", sg)
        if m:
            ids.append((m.group(1), m.group(2), int(m.group(3))))
genes = {g for g, _, _ in ids}
gtf = sys.argv[3]
chrom_of = {}
with gzip.open(gtf, "rt") as f:
    for ln in f:
        p = ln.split("\t", 9)
        if len(p) > 8 and p[2] == "gene":
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if m and m.group(1) in genes:
                chrom_of[m.group(1)] = p[0]
want = set(chrom_of.values())
seqs, cur, buf = {}, None, []
with gzip.open(genome, "rt") as f:
    for ln in f:
        if ln[0] == ">":
            if cur:
                seqs[cur] = "".join(buf).upper()
            cur = ln[1:].split()[0]
            cur = cur if cur in want else None
            buf = []
        elif cur:
            buf.append(ln.strip())
if cur:
    seqs[cur] = "".join(buf).upper()
rel = Counter()
n = 0
for g, s, c in ids:
    ch = seqs.get(chrom_of.get(g))
    if ch is None:
        continue
    n += 1
    lo, hi = max(0, c - 3000), c + 3000
    reg = ch[lo:hi]
    for k in range(len(reg) - 19):
        w = reg[k:k + 19]
        if w in top:
            rel[("+", s, k + lo - c)] += 1
        rw = w.translate(_C)[::-1]
        if rw in top:
            rel[("-", s, k + lo - c)] += 1
print(n, "ids scanned;", "top relative positions (window strand, id strand, start - coord):")
for k, v in rel.most_common(15):
    print(k, v)
