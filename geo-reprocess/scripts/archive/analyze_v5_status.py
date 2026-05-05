#!/usr/bin/env python3
"""Analyze v5 pipeline processing status from batch result files."""
import csv
import os
import glob
from collections import Counter

results_dir = '/mnt/projects/debruinz_project/cellarium/pipeline/batches_v5_production'
status_counts = Counter()
error_counts = Counter()
protocol_counts = Counter()

files = sorted(glob.glob(os.path.join(results_dir, '*_results.csv')))
print(f"Total result files: {len(files)}")

for fpath in files[:500]:
    try:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                st = row.get('status', '')
                status_counts[st] += 1
                if st == 'failed':
                    err = row.get('error', 'unknown')
                    if err:
                        err = err[:80]
                    error_counts[err] += 1
                prot = row.get('protocol', '')
                if prot:
                    protocol_counts[prot] += 1
    except Exception as e:
        pass

print(f"\n=== Status (sample of {min(500,len(files))} batches) ===")
for st, cnt in status_counts.most_common():
    print(f"  {cnt:6d}  {st}")

print(f"\n=== Top failure errors ===")
for err, cnt in error_counts.most_common(15):
    print(f"  {cnt:5d}  {err}")

print(f"\n=== Top protocols ===")
for p, cnt in protocol_counts.most_common(10):
    print(f"  {cnt:5d}  {p}")
