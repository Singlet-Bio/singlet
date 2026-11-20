#!/usr/bin/env python3
"""t2-fallzone (follow-up): eligible samples and read pairs per assay label of the corpus table (samples_pairs.csv.gz), and for the zone 70-160 M.
Same filter as t2-fallzone-depth.py. Reads assay, assay_class, eligible, mux_queue_skip, read_pairs_*corrected. usage: t2-fallzone-assays.py samples_pairs.csv.gz"""
import csv, gzip, sys
from collections import defaultdict

csv.field_size_limit(1 << 30)
n = defaultdict(int); p = defaultdict(float); cls = {}
tn = tp = 0
with gzip.open(sys.argv[1], "rt", newline="") as f:
    for r in csv.DictReader(f):
        if (r.get("eligible") or "").strip().lower() not in ("true", "1", "yes", "y"):
            continue
        if (r.get("mux_queue_skip") or "").strip().lower() in ("true", "1", "yes", "y"):
            continue
        try:
            d = float(r.get("read_pairs_gex_corrected") or r.get("read_pairs_corrected") or r.get("read_pairs") or 0)
        except ValueError:
            continue
        if d <= 0:
            continue
        a = r.get("assay") or "(blank)"
        n[a] += 1; p[a] += d; cls[a] = r.get("assay_class") or ""
        tn += 1; tp += d
print("assay label\tassay_class\tsamples\tshare_samples_pct\tpairs_G\tshare_pairs_pct")
for a in sorted(n, key=lambda k: -p[k]):
    print("%s\t%s\t%d\t%.2f\t%.1f\t%.2f" % (a, cls[a], n[a], 100.0 * n[a] / tn, p[a] / 1e9, 100.0 * p[a] / tp))
print("total\t\t%d\t100\t%.1f\t100" % (tn, tp / 1e9))
