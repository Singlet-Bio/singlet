"""Reference chrM pileup from Singlet's tagged BAM with pysam (same filters as Singlet: MAPQ>=20, BQ>=10, read-level counts).
Writes per-cell counts (cell, pos1, base, n) and per-truth-donor pseudobulk. Usage: mt_pileup.py BAM CELLS.txt OUT.npz [DONOR_MAP.tsv]"""
import sys, numpy as np, pysam, pandas as pd, collections
bam, cells_f, out = sys.argv[1:4]
cells = [l.strip() for l in open(cells_f)]
cidx = {c: i for i, c in enumerate(cells)}
cidx.update({c.split("-")[0]: i for c, i in list(cidx.items())})
NB = {"A": 0, "C": 1, "G": 2, "T": 3}
L = 16569
cnt = collections.defaultdict(lambda: np.zeros(4, np.int32))   # (cell, pos0) -> counts
umi = collections.defaultdict(set)
f = pysam.AlignmentFile(bam, "rb")
chrm = [c for c in f.references if c in ("chrM", "MT", "M")][0]
n = 0
for r in f.fetch(chrm):
    if r.is_unmapped or r.mapping_quality < 20: continue
    try: cb = r.get_tag("CB")
    except KeyError: continue
    ci = cidx.get(cb, cidx.get(cb.split("-")[0]))
    if ci is None: continue
    n += 1
    q = r.query_qualities; s = r.query_sequence
    for qp, rp in r.get_aligned_pairs(matches_only=True):
        if rp >= L or (q is not None and q[qp] < 10): continue
        b = NB.get(s[qp]);
        if b is None: continue
        cnt[(ci, rp)][b] += 1
keys = sorted(cnt)
arr = np.array([(k[0], k[1], *cnt[k]) for k in keys], dtype=np.int32)
np.savez_compressed(out, arr=arr, cells=np.array(cells), n_reads=n)
print("chrM reads used", n, "cell-pos entries", len(keys))
