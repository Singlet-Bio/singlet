#!/usr/bin/env python3
"""Tier C Batch: Download and process GSE-level h5ad files.

Reads tier_c_targets.json and processes each GSE-level target:
1. Download h5ad from GEO (skip if already exists)
2. Run tier_c_apply_gse_h5ad.py
3. Log results

Usage:
    python tier_c_batch.py <targets_json> <dataset_dir> <catalog_path> [--work-dir DIR] [--max-size-gb N]
"""
import sys
import json
import os
import subprocess
import time

def main():
    targets_path = sys.argv[1]
    dataset_dir = sys.argv[2]
    catalog_path = sys.argv[3]

    work_dir = None
    max_size_gb = 10  # Skip h5ad files > this size
    for i, a in enumerate(sys.argv):
        if a == "--work-dir" and i + 1 < len(sys.argv):
            work_dir = sys.argv[i + 1]
        if a == "--max-size-gb" and i + 1 < len(sys.argv):
            max_size_gb = float(sys.argv[i + 1])

    if work_dir is None:
        work_dir = os.path.dirname(targets_path)

    os.makedirs(work_dir, exist_ok=True)

    with open(targets_path) as f:
        targets = json.load(f)

    # Filter to GSE-level targets only
    gse_level = [t for t in targets if t["is_gse_level"]]
    print(f"Processing {len(gse_level)} GSE-level targets")

    apply_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_c_apply_gse_h5ad.py")

    results = []
    for i, t in enumerate(gse_level):
        gse_id = t["gse_id"]
        n_gsms = t["n_gsms"]
        urls = t["urls"]

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(gse_level)}] {gse_id} ({n_gsms} GSMs)")

        if not urls:
            print(f"  SKIP: No h5ad URLs")
            results.append(dict(gse_id=gse_id, status="skip_no_url"))
            continue

        # Pick first URL
        url = urls[0]
        filename = url.split("/")[-1]
        local_path = os.path.join(work_dir, filename)

        # Download if not exists
        if os.path.exists(local_path):
            size_gb = os.path.getsize(local_path) / 1e9
            print(f"  Already downloaded: {filename} ({size_gb:.1f} GB)")
        else:
            # Check size via HEAD request first
            print(f"  Downloading: {filename}")
            try:
                result = subprocess.run(
                    ["wget", "-q", "--show-progress", "-O", local_path, url],
                    capture_output=False, timeout=1800,  # 30 min max per download
                )
                if result.returncode != 0:
                    print(f"  FAIL: Download failed (rc={result.returncode})")
                    results.append(dict(gse_id=gse_id, status="download_failed"))
                    continue
            except subprocess.TimeoutExpired:
                print(f"  FAIL: Download timeout")
                results.append(dict(gse_id=gse_id, status="download_timeout"))
                if os.path.exists(local_path):
                    os.remove(local_path)
                continue

        size_gb = os.path.getsize(local_path) / 1e9
        if size_gb > max_size_gb:
            print(f"  SKIP: {size_gb:.1f} GB > {max_size_gb} GB limit")
            results.append(dict(gse_id=gse_id, status="skip_too_large", size_gb=size_gb))
            continue

        # Process
        print(f"  Processing {filename} ({size_gb:.1f} GB)...")
        t0 = time.time()
        try:
            result = subprocess.run(
                ["python3", apply_script, local_path, gse_id, dataset_dir, catalog_path,
                 "--source", filename],
                capture_output=True, text=True, timeout=600,  # 10 min max
            )
            elapsed = time.time() - t0
            output = result.stdout + result.stderr
            print(f"  {output.strip()}")
            
            if result.returncode == 0:
                results.append(dict(gse_id=gse_id, status="success", time_s=elapsed))
            else:
                results.append(dict(gse_id=gse_id, status="process_failed", error=output[-200:]))
        except subprocess.TimeoutExpired:
            print(f"  FAIL: Processing timeout")
            results.append(dict(gse_id=gse_id, status="process_timeout"))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    for r in results:
        print(f"  {r['gse_id']}: {r['status']}")

    success = sum(1 for r in results if r["status"] == "success")
    print(f"\nSuccess: {success}/{len(gse_level)}")

    with open(os.path.join(work_dir, "tier_c_batch_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
