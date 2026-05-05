#!/usr/bin/env python3
"""Scan catalog for Tier C targets: GSEs with GSE-level h5ad and <50% match.

Outputs a JSON manifest of targets with download URLs.
"""
import pandas as pd
import json
import sys

catalog_path = sys.argv[1]
geo_catalog_path = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else "tier_c_targets.json"

cat = pd.read_parquet(catalog_path)
gcat = pd.read_parquet(geo_catalog_path)

rates = cat["stage2a_match_rate"].fillna(0)
low = cat[rates < 0.5]

targets = []
for gse_id, group in low.groupby("gse_id"):
    fmts = json.loads(group["stage0_formats"].iloc[0])
    h5ad_count = fmts.get("h5ad", 0)
    if h5ad_count == 0:
        continue
    n_gsms = len(group)
    is_gse_level = h5ad_count > n_gsms

    # Get supplementary file URLs
    gse_rows = gcat[gcat["gse_id"] == gse_id]
    supp_urls = set()
    for _, r in gse_rows.iterrows():
        sf = str(r["supplementary_files"]) if pd.notna(r["supplementary_files"]) else ""
        for url in sf.split(";"):
            url = url.strip()
            if url.lower().endswith(".h5ad"):
                supp_urls.add(url)

    if not supp_urls:
        continue

    # Convert FTP to HTTPS
    https_urls = []
    for u in supp_urls:
        u = u.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
        https_urls.append(u)

    targets.append(dict(
        gse_id=gse_id,
        n_gsms=n_gsms,
        h5ad_count=h5ad_count,
        is_gse_level=is_gse_level,
        urls=https_urls[:5],
        mean_current_rate=round(rates[group.index].mean(), 4),
    ))

targets.sort(key=lambda x: -x["n_gsms"])

print(f"Total targets: {len(targets)}")
gse_level = [t for t in targets if t["is_gse_level"]]
per_gsm = [t for t in targets if not t["is_gse_level"]]
print(f"  GSE-level: {len(gse_level)} GSEs, {sum(t['n_gsms'] for t in gse_level)} GSMs")
print(f"  Per-GSM: {len(per_gsm)} GSEs, {sum(t['n_gsms'] for t in per_gsm)} GSMs")

print(f"\nGSE-level targets:")
for t in gse_level:
    fn = t["urls"][0].split("/")[-1][:60] if t["urls"] else "N/A"
    print(f"  {t['gse_id']}: {t['n_gsms']} GSMs, {fn}")

with open(output_path, "w") as f:
    json.dump(targets, f, indent=2)
print(f"\nSaved to {output_path}")
