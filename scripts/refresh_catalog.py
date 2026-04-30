#!/usr/bin/env python3
"""Refresh embedded catalog parquet files from Supabase + pipeline results.

Merges Supabase metadata (titles, QC metrics) with pipeline result JSONs.
Writes to python/singlet/data/sample_index.parquet and catalog_v1.parquet.

Usage:
    python scripts/refresh_catalog.py

Requires: SUPABASE_URL, SUPABASE_SERVICE_KEY environment variables.
Optional: Access to pipeline results for samples not yet in Supabase.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

RESULTS_DIR = "/mnt/projects/debruinz_project/singlify_pipeline/results/2026-04/"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "python" / "singlet" / "data"
CACHE_DIR = Path.home() / ".singlet" / "cache"

SUPABASE_FIELDS = "gsm_id,gse_id,organism,status,protocol,mapping_rate,cells_called,median_genes,median_umis,mt_pct,doublet_rate,wall_time_s,title"


def fetch_supabase() -> pd.DataFrame:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("WARNING: SUPABASE_URL/SUPABASE_SERVICE_KEY not set, using pipeline only")
        return pd.DataFrame()

    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    all_rows = []
    offset = 0
    while True:
        r = requests.get(
            f"{url}/rest/v1/samples?select={SUPABASE_FIELDS}&offset={offset}&limit=1000",
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        offset += 1000
        if len(data) < 1000:
            break
    return pd.DataFrame(all_rows)


def load_pipeline_extras(supabase_gsms: set) -> list:
    """Load pipeline results for samples not in Supabase."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    extras = []
    for f in os.listdir(RESULTS_DIR):
        if not f.endswith(".json"):
            continue
        gsm = f.replace(".json", "")
        if gsm in supabase_gsms:
            continue
        with open(os.path.join(RESULTS_DIR, f)) as fh:
            d = json.load(fh)
        extras.append({
            "gsm_id": d.get("gsm_id", gsm),
            "gse_id": d.get("gse_id", ""),
            "organism": d.get("organism", "") or "",
            "status": d.get("status", ""),
            "protocol": d.get("autodetect_protocol") or d.get("protocol", ""),
            "mapping_rate": d.get("mapping_rate"),
            "cells_called": d.get("cells_called", 0),
            "median_genes": None,
            "median_umis": None,
            "mt_pct": None,
            "doublet_rate": None,
            "wall_time_s": d.get("wall_time_s", 0),
            "title": None,
        })
    return extras


def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    catalog = []
    for gse_id, group in df.groupby("gse_id"):
        success = group[group["status"] == "SUCCESS"]
        orgs = group[group["organism"] != ""]["organism"]
        catalog.append({
            "gse_id": gse_id,
            "organism": orgs.mode().iloc[0] if len(orgs.mode()) > 0 else "",
            "n_samples": len(group),
            "n_success": len(success),
            "n_cells": int(success["cells_called"].sum()),
            "protocol": (
                success["protocol"].mode().iloc[0]
                if len(success) > 0 and len(success["protocol"].mode()) > 0
                else ""
            ),
            "median_mapping_rate": (
                round(success["mapping_rate"].median(), 4)
                if len(success) > 0 else None
            ),
        })
    return pd.DataFrame(catalog)


def main():
    print("Fetching Supabase samples...")
    supabase_df = fetch_supabase()
    print(f"  {len(supabase_df)} from Supabase")

    supabase_gsms = set(supabase_df["gsm_id"]) if len(supabase_df) > 0 else set()

    print("Loading pipeline extras...")
    extras = load_pipeline_extras(supabase_gsms)
    print(f"  {len(extras)} pipeline-only samples")

    # Merge
    if len(supabase_df) > 0:
        supabase_df["organism"] = supabase_df["organism"].fillna("").replace("unknown", "")
        supabase_df["protocol"] = supabase_df["protocol"].fillna("").replace("unknown", "")
        df = supabase_df
    else:
        df = pd.DataFrame()

    if extras:
        extra_df = pd.DataFrame(extras)
        df = pd.concat([df, extra_df], ignore_index=True) if len(df) > 0 else extra_df

    print(f"\nTotal: {len(df)} samples")
    success_count = (df["status"] == "SUCCESS").sum()
    organism_count = (df["organism"] != "").sum()
    print(f"  SUCCESS: {success_count}")
    print(f"  Organism coverage: {organism_count}/{len(df)} ({100*organism_count/len(df):.0f}%)")

    print("\nBuilding catalog...")
    catalog = build_catalog(df)
    print(f"  {len(catalog)} series, {catalog['n_cells'].sum():,.0f} cells")

    print("\nSaving...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DIR / "sample_index.parquet", index=False)
    catalog.to_parquet(OUTPUT_DIR / "catalog_v1.parquet", index=False)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUTPUT_DIR / "sample_index.parquet", CACHE_DIR / "sample_index.parquet")
    shutil.copy(OUTPUT_DIR / "catalog_v1.parquet", CACHE_DIR / "catalog_v1.parquet")

    si_kb = (OUTPUT_DIR / "sample_index.parquet").stat().st_size // 1024
    cat_kb = (OUTPUT_DIR / "catalog_v1.parquet").stat().st_size // 1024
    print(f"  sample_index.parquet: {si_kb} KB")
    print(f"  catalog_v1.parquet: {cat_kb} KB")
    print(f"\nDone! {len(df):,} samples • {success_count:,} SUCCESS • {len(catalog):,} series • {catalog['n_cells'].sum():,.0f} cells")


if __name__ == "__main__":
    main()
