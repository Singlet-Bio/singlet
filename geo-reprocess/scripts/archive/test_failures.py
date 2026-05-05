#!/usr/bin/env python3
"""Test that previously-failed samples can now succeed with updated pipeline.

Pick a few representative failures and attempt to reprocess them.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Analyze failures from the failures directory
fail_dir = '/mnt/projects/debruinz_project/cellarium/pipeline/failures/'
cats = Counter()
examples = {}  # cat -> list of (gsm_id, gse_id, detail)
total = 0

for fn in os.listdir(fail_dir):
    if not fn.endswith('.json'):
        continue
    total += 1
    try:
        with open(os.path.join(fail_dir, fn)) as f:
            d = json.load(f)
        cat = d.get('fail_category', 'unknown')
        cats[cat] += 1
        if cat not in examples or len(examples[cat]) < 3:
            examples.setdefault(cat, []).append({
                'gsm': d.get('gsm_id', ''),
                'gse': d.get('gse_id', ''),
                'detail': str(d.get('detail', ''))[:120],
            })
    except Exception:
        cats['parse_error'] += 1

print(f"Total failure files: {total}")
print(f"\n=== Failure category breakdown ===")
for cat, cnt in cats.most_common():
    print(f"\n  {cnt:5d}  {cat}")
    for ex in examples.get(cat, [])[:2]:
        print(f"         {ex['gse']}/{ex['gsm']}: {ex['detail']}")

# Now check if any failed samples have been subsequently re-processed successfully
print(f"\n=== Checking if failed samples were retried ===")
quant_dir = '/mnt/projects/debruinz_project/cellarium/pipeline/quant'
retried = 0
still_failed = 0
now_success = 0
checked = 0

for fn in sorted(os.listdir(fail_dir))[-500:]:
    if not fn.endswith('.json'):
        continue
    try:
        with open(os.path.join(fail_dir, fn)) as f:
            d = json.load(f)
        gsm = d.get('gsm_id', '')
        gse = d.get('gse_id', '')
        if not gsm or not gse:
            continue
        checked += 1
        manifest_path = os.path.join(quant_dir, gse, gsm, 'sample_manifest.json')
        if os.path.exists(manifest_path):
            retried += 1
            with open(manifest_path) as f:
                m = json.load(f)
            if m.get('status') in ('success', 'qc_pass', 'qc_warn'):
                now_success += 1
            elif m.get('status') == 'failed':
                still_failed += 1
    except Exception:
        pass

print(f"  Checked: {checked} recent failures")
print(f"  Had quant manifest: {retried}")
print(f"  Now succeeded: {now_success}")
print(f"  Still failed: {still_failed}")

# Check the pipeline module versions / recent fixes
print(f"\n=== Pipeline module check ===")
try:
    sys.path.insert(0, '/mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess')
    from scgeo.pipeline import api as pipeline_api
    from scgeo.pipeline import detect, download, quantify
    print(f"  pipeline.api loaded: {pipeline_api.__file__}")
    print(f"  pipeline.detect loaded: {detect.__file__}")
    print(f"  pipeline.download loaded: {download.__file__}")
    print(f"  pipeline.quantify loaded: {quantify.__file__}")
    
    # Check if key fixes are present
    import inspect
    detect_src = inspect.getsource(detect)
    if 'ambiguous' in detect_src.lower():
        print(f"  detect.py: has ambiguous length handling")
    if 'fallback' in detect_src.lower():
        print(f"  detect.py: has fallback detection")
        
    download_src = inspect.getsource(download)
    if 'retry' in download_src.lower():
        print(f"  download.py: has retry logic")
    if 'segment' in download_src.lower():
        print(f"  download.py: has segmented download")
        
except Exception as e:
    print(f"  Error loading pipeline: {e}")
