#!/usr/bin/env python3
"""t2-fallzone: depth histogram of the campaign manifest (corpus-beta samples_pairs.csv.gz): how many eligible samples, and what share of the
read pairs, lie in the zone 70-160 M pairs. Reads the columns eligible, read_pairs_gex_corrected / read_pairs_corrected / read_pairs, bases,
mux_queue_skip, assay_class only; prints counts. usage: t2-fallzone-depth.py samples_pairs.csv.gz"""
import csv, gzip, sys
from collections import defaultdict

EDGES = [0, 20e6, 50e6, 70e6, 100e6, 130e6, 160e6, 300e6, 600e6, 1.4e9, 1e18]
csv.field_size_limit(1 << 30)
cnt = defaultdict(int); prs = defaultdict(float); lc = defaultdict(int)
tot_n = tot_p = 0
with gzip.open(sys.argv[1], "rt", newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        if (r.get("eligible") or "").strip().lower() not in ("true", "1", "yes", "y"):
            continue
        if (r.get("mux_queue_skip") or "").strip().lower() in ("true", "1", "yes", "y"):
            continue
        try:
            p = float(r.get("read_pairs_gex_corrected") or r.get("read_pairs_corrected") or r.get("read_pairs") or 0)
        except ValueError:
            continue
        if p <= 0:
            continue
        try:
            b = float(r.get("bases") or 0)
        except ValueError:
            b = 0
        long_ = b / p >= 200 if b > 0 else False     # both mates, about 28 + 91 = 119 for 3' v3 at 90 bp; 28 + 150 and up for 150 bp
        for i in range(len(EDGES) - 1):
            if EDGES[i] <= p < EDGES[i + 1]:
                cnt[i] += 1; prs[i] += p
                if long_:
                    lc[i] += 1
                break
        tot_n += 1; tot_p += p
print("eligible samples (not mux-skipped) with a depth: %d, %.2f T pairs" % (tot_n, tot_p / 1e12))
print("bin_M_pairs\tsamples\tshare_samples_pct\tpairs_G\tshare_pairs_pct\tsamples_bases_per_pair_ge_200")
for i in range(len(EDGES) - 1):
    hi = "inf" if EDGES[i + 1] > 1e15 else "%g" % (EDGES[i + 1] / 1e6)
    print("%g-%s\t%d\t%.2f\t%.1f\t%.2f\t%d" % (EDGES[i] / 1e6, hi, cnt[i], 100.0 * cnt[i] / tot_n, prs[i] / 1e9, 100.0 * prs[i] / tot_p, lc[i]))
z = [3, 4, 5]
zn = sum(cnt[i] for i in z); zp = sum(prs[i] for i in z)
print("zone 70-160 M: %d samples (%.2f %%), %.1f G pairs (%.2f %% of all pairs)" % (zn, 100.0 * zn / tot_n, zp / 1e9, 100.0 * zp / tot_p))
