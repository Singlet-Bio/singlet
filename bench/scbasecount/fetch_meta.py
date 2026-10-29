"""Fetch the inputs for the Singlet vs scBaseCount coverage comparison (run on Clipper, in a job).

Writes into OUT (default $WORK/data/scbasecount/meta):
  sbc_samples.parquet   scBaseCount 2026-01-12 sample_metadata for every organism (GeneFull_Ex50pAS tables)
  singlet_gsm.parquet   every sample row of the public singlet.bio /api/gsm listing
  catalog_map.parquet   Singlet's processing catalog: gsm, gse, srx, srr, protocol, organism, status columns
"""
import io
import json
import os
import sys
import time
import urllib.request

import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["WORK"], "data/scbasecount/meta")
os.makedirs(OUT, exist_ok=True)
BUCKET = "arc-institute-virtual-cell-atlas"
REL = "scbasecount/2026-01-12"


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "singlet-bench-scbasecount/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print("retry", url, e, flush=True)
            time.sleep(2 + 3 * i)


def gcs_list(prefix):
    items, prefixes, tok = [], [], None
    while True:
        q = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={prefix}&delimiter=/&maxResults=1000"
        if tok:
            q += f"&pageToken={tok}"
        d = json.loads(get(q))
        items += d.get("items", [])
        prefixes += d.get("prefixes", [])
        tok = d.get("nextPageToken")
        if not tok:
            return items, prefixes


def sbc():
    frames = []
    for feat in ["GeneFull_Ex50pAS", "Gene"]:
        _, species = gcs_list(f"{REL}/metadata/{feat}/")
        for sp in species:
            name = sp.rstrip("/").split("/")[-1]
            raw = get(f"https://storage.googleapis.com/{BUCKET}/{sp}sample_metadata.parquet")
            df = pd.read_parquet(io.BytesIO(raw))
            df["species_dir"] = name
            df["feature"] = feat
            frames.append(df)
            print(feat, name, len(df), flush=True)
    d = pd.concat(frames, ignore_index=True)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype(str)
    d.to_parquet(os.path.join(OUT, "sbc_samples.parquet"))


def singlet():
    rows, page = [], 0
    while True:
        d = json.loads(get(f"https://singlet.bio/api/gsm?page={page}&page_size=500&sort=gsm_id&asc=1"))
        rows += d["data"]
        print("singlet page", page, len(rows), "/", d["total"], flush=True)
        if not d["data"] or len(rows) >= d["total"]:
            break
        page += 1
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["srr_ids"] = df["srr_ids"].map(lambda x: ",".join(x) if isinstance(x, list) else "")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    df.to_parquet(os.path.join(OUT, "singlet_gsm.parquet"))


def catalog():
    cols = ["gsm_id", "gse_id", "organism", "protocol_inferred", "srx_accession", "srr_accessions",
            "library_strategy", "library_source", "submission_date", "processing_status", "target_assay",
            "title_lib_type", "screen_vdj", "exclusion_reason", "read_count"]
    p = "/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet"
    df = pd.read_parquet(p, columns=cols)
    df.to_parquet(os.path.join(OUT, "catalog_map.parquet"))
    print("catalog", len(df), flush=True)


if __name__ == "__main__":
    which = sys.argv[2:] or ["catalog", "sbc", "singlet"]
    for w in which:
        globals()[w]()
