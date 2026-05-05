#!/usr/bin/env python3
"""Tier C Batch 2: Download and process ALL remaining h5ad targets.

Handles:
- Multi-URL GSEs (process each h5ad file)
- Per-GSM-pattern GSEs (actually GSE-level files)
- Retries for zero-match GSEs

Usage:
    python tier_c_batch2.py <targets_json> <dataset_dir> <catalog_path> [--work-dir DIR] [--max-size-gb N] [--download-timeout S]
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
    max_size_gb = 10
    download_timeout = 1800
    for i, a in enumerate(sys.argv):
        if a == "--work-dir" and i + 1 < len(sys.argv):
            work_dir = sys.argv[i + 1]
        if a == "--max-size-gb" and i + 1 < len(sys.argv):
            max_size_gb = float(sys.argv[i + 1])
        if a == "--download-timeout" and i + 1 < len(sys.argv):
            download_timeout = int(sys.argv[i + 1])

    if work_dir is None:
        work_dir = os.path.dirname(targets_path)

    os.makedirs(work_dir, exist_ok=True)

    with open(targets_path) as f:
        targets = json.load(f)

    apply_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_c_apply_gse_h5ad.py")

    # Check which GSEs already have good results (skip those)
    import pandas as pd
    cat = pd.read_parquet(catalog_path)

    # Skip GSEs already processed with decent rates
    skip_gses = set()
    for t in targets:
        gse = t["gse_id"]
        subset = cat[cat["gse_id"] == gse]
        if len(subset) > 0:
            mean_rate = subset["stage2a_match_rate"].mean()
            above50 = (subset["stage2a_match_rate"] >= 0.5).sum()
            # Skip if already >50% mean or sufficient >50% coverage
            if mean_rate > 0.5 or above50 > len(subset) * 0.5:
                skip_gses.add(gse)

    results = []
    for tidx, t in enumerate(targets):
        gse_id = t["gse_id"]
        n_gsms = t["n_gsms"]
        urls = t.get("urls", [])

        if gse_id in skip_gses:
            print(f"\n[{tidx+1}/{len(targets)}] SKIP {gse_id}: already has decent match rates")
            results.append(dict(gse_id=gse_id, status="skip_already_good"))
            continue

        if not urls:
            results.append(dict(gse_id=gse_id, status="skip_no_url"))
            continue

        print(f"\n{'='*60}")
        print(f"[{tidx+1}/{len(targets)}] {gse_id} ({n_gsms} GSMs, {len(urls)} files)")

        gse_success = False
        for uidx, url in enumerate(urls):
            filename = url.split("/")[-1]
            local_path = os.path.join(work_dir, filename)

            # Download if not exists
            if os.path.exists(local_path):
                size_gb = os.path.getsize(local_path) / 1e9
                print(f"  [{uidx+1}/{len(urls)}] Exists: {filename} ({size_gb:.1f} GB)")
            else:
                print(f"  [{uidx+1}/{len(urls)}] Downloading: {filename}")
                try:
                    result = subprocess.run(
                        ["wget", "-q", "-O", local_path, url],
                        capture_output=True, timeout=download_timeout,
                    )
                    if result.returncode != 0:
                        print(f"    Download failed (rc={result.returncode})")
                        if os.path.exists(local_path):
                            os.remove(local_path)
                        continue
                except subprocess.TimeoutExpired:
                    print(f"    Download timeout ({download_timeout}s)")
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    continue

            size_gb = os.path.getsize(local_path) / 1e9
            if size_gb > max_size_gb:
                print(f"    Skip: {size_gb:.1f} GB > {max_size_gb} GB limit")
                continue
            if size_gb < 0.001:
                print(f"    Skip: file too small ({size_gb*1024:.1f} MB)")
                continue

            # Process
            print(f"    Processing ({size_gb:.2f} GB)...")
            t0 = time.time()
            try:
                result = subprocess.run(
                    ["python3", apply_script, local_path, gse_id, dataset_dir, catalog_path,
                     "--source", f"tierc:{filename}"],
                    capture_output=True, text=True, timeout=600,
                )
                elapsed = time.time() - t0
                output = result.stdout + result.stderr

                # Extract key stats from output
                for line in output.split("\n"):
                    if "Match rates:" in line or "Mapped:" in line or "No GSMs" in line or "Catalog updated:" in line:
                        print(f"    {line.strip()}")

                if result.returncode == 0:
                    gse_success = True
                else:
                    # Show last error line
                    err_lines = [l for l in output.split("\n") if "ERROR" in l or "Error" in l]
                    if err_lines:
                        print(f"    FAIL: {err_lines[-1][:100]}")
            except subprocess.TimeoutExpired:
                print(f"    Processing timeout")

        results.append(dict(gse_id=gse_id, status="success" if gse_success else "failed",
                           n_files=len(urls)))

    # Summary
    print(f"\n{'='*60}")
    print("BATCH 2 SUMMARY")
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"].startswith("skip"))
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"  Success: {success}, Skipped: {skipped}, Failed: {failed}")
    for r in results:
        print(f"  {r['gse_id']}: {r['status']}")

    with open(os.path.join(work_dir, "tier_c_batch2_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
