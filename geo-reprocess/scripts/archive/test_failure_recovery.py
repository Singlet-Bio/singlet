#!/usr/bin/env python3
"""Quick test of failure recovery on compute node.

Tests whether the top failure modes are now handled by the current codebase.
"""
import sys
import json
import os

sys.path.insert(0, '/mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess')

from scgeo.config import get_config

config = get_config()
fail_dir = '/mnt/projects/debruinz_project/cellarium/pipeline/failures/'

# Collect one example of each failure type
by_cat = {}
for fn in sorted(os.listdir(fail_dir)):
    if not fn.endswith('.json'):
        continue
    try:
        with open(os.path.join(fail_dir, fn)) as f:
            d = json.load(f)
        cat = d.get('fail_category', 'unknown')
        if cat not in by_cat:
            by_cat[cat] = d
    except:
        pass

# Focus on the most common: cannot_determine_protocol
print("=== Testing protocol detection on previously failed samples ===\n")
proto_failures = []
for fn in sorted(os.listdir(fail_dir)):
    if not fn.endswith('.json'):
        continue
    try:
        with open(os.path.join(fail_dir, fn)) as f:
            d = json.load(f)
        if d.get('fail_category') == 'cannot_determine_protocol':
            proto_failures.append(d)
        if len(proto_failures) >= 10:
            break
    except:
        pass

# Check what the detection detail says
from collections import Counter
proto_detail_types = Counter()
for fn in os.listdir(fail_dir):
    if not fn.endswith('.json'):
        continue
    try:
        with open(os.path.join(fail_dir, fn)) as f:
            d = json.load(f)
        if d.get('fail_category') == 'cannot_determine_protocol':
            detail = d.get('detail', '')
            if 'Could not read R1' in detail:
                proto_detail_types['r1_read_failed'] += 1
            elif 'no R2' in detail:
                proto_detail_types['no_r2'] += 1
            elif 'ambiguous' in detail:
                proto_detail_types['ambiguous_lengths'] += 1
            elif 'R1=' in detail and 'R2=' in detail:
                prot = d.get('protocol', {})
                r1 = prot.get('r1_len', 0) if isinstance(prot, dict) else 0
                r2 = prot.get('r2_len', 0) if isinstance(prot, dict) else 0
                proto_detail_types[f'r1={r1}_r2={r2}'] += 1
            else:
                proto_detail_types['other: ' + detail[:60]] += 1
    except:
        pass

print("Sub-categories of cannot_determine_protocol:")
for dt, cnt in proto_detail_types.most_common():
    print(f"  {cnt:5d}  {dt}")

# Check the current v5 batch files for retry tracking
print("\n=== Checking if failed samples are queued for retry ===")
batch_dir = '/mnt/projects/debruinz_project/cellarium/pipeline/batches_v5_production'
import csv
batch_gsms = set()
count = 0
for fn in sorted(os.listdir(batch_dir)):
    if not fn.endswith('.csv') or '_results' in fn:
        continue
    count += 1
    try:
        with open(os.path.join(batch_dir, fn)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch_gsms.add(row.get('gsm_id', ''))
    except:
        pass

print(f"  Total batch files: {count}")
print(f"  Unique GSMs in batches: {len(batch_gsms):,}")

# Count how many failure GSMs are in batch files
fail_gsms = set()
for fn in os.listdir(fail_dir):
    if fn.endswith('.json'):
        try:
            with open(os.path.join(fail_dir, fn)) as f:
                d = json.load(f)
            fail_gsms.add(d.get('gsm_id', ''))
        except:
            pass

in_batch = fail_gsms & batch_gsms
not_in_batch = fail_gsms - batch_gsms
print(f"  Failed GSMs: {len(fail_gsms):,}")
print(f"  In current batches: {len(in_batch):,}")
print(f"  NOT in any batch: {len(not_in_batch):,}")

# Check the rescue/retry SLURM scripts
print("\n=== Rescue/retry script check ===")
for script in ['slurm_rescue_bigmem.sh', 'slurm_rescue_cpu.sh', 'slurm_scgeo-test.sh']:
    spath = os.path.join('/mnt/projects/debruinz_project/cellarium/pipeline', script)
    if os.path.exists(spath):
        with open(spath) as f:
            first_lines = f.readline() + f.readline()
        print(f"  {script}: exists")
    else:
        print(f"  {script}: NOT FOUND")
