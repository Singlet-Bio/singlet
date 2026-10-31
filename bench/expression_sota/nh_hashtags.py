# SPDX-License-Identifier: MIT
"""Discover the hashtag sequences of the hashtag library (R1 = 16 bp barcode + 10 bp UMI, R2 = antibody oligo read) and count them per
cell barcode. python nh_hashtags.py R1.gz R2.gz CALLS_BARCODES.tsv OUT_PREFIX [N_READS]
Step 1: tally the first 10-mers of R2 over reads whose barcode is in the whitelist-like set (barcodes seen in the RNA raw matrix).
Step 2: pick the top tags (>= 2% of reads each, allowing 1 mismatch merges), count per barcode with UMI dedup."""
import gzip, sys, collections
r1, r2, bcf, out = sys.argv[1:5]; N = int(sys.argv[5]) if len(sys.argv) > 5 else 40_000_000
bcs = set(l.split("\t")[0].strip().split("-")[0] for l in open(bcf))
def reads(p):
    with gzip.open(p, "rt") as f:
        while True:
            h = f.readline()
            if not h: return
            s = f.readline().rstrip(); f.readline(); f.readline(); yield s
tally = collections.Counter(); n = nin = 0; keep = []
for a, b in zip(reads(r1), reads(r2)):
    n += 1
    if n > N: break
    if a[:16] in bcs:
        nin += 1; tally[b[:10]] += 1; keep.append((a[:16], a[16:26], b[:10]))
print("reads", n, "in barcode set", nin)
top = [(k, v) for k, v in tally.most_common(30)]
print("top R2 10-mers:", top[:20])
tot = sum(tally.values())
tags = [k for k, v in top if v / tot > 0.02]
def near(x):
    for t in tags:
        if sum(1 for u, w in zip(x, t) if u != w) <= 1: return t
    return None
cnt = collections.defaultdict(lambda: collections.defaultdict(set))
for cb, umi, t in keep:
    m = near(t)
    if m: cnt[cb][m].add(umi)
with open(out + ".counts.tsv", "w") as f:
    f.write("barcode\t" + "\t".join(tags) + "\n")
    for cb, d in cnt.items(): f.write(cb + "\t" + "\t".join(str(len(d.get(t, ()))) for t in tags) + "\n")
print("tags", tags, "barcodes with a tag", len(cnt))
