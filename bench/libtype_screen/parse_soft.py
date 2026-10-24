# SPDX-License-Identifier: MIT
"""Parse cached GEO family SOFT files into one row per GSM (all samples of each atlas series, in the atlas or not)."""

import glob
import gzip
import os
import sys

import pandas as pd

# working directory for caches and outputs (default: next to this script)
HERE = os.environ.get("LIBTYPE_WORK", os.path.dirname(os.path.abspath(__file__)))
KEEP = {
    "Sample_title": "title",
    "Sample_source_name_ch1": "source_name",
    "Sample_description": "description",
    "Sample_library_strategy": "library_strategy",
    "Sample_library_source": "library_source",
    "Sample_library_selection": "library_selection",
    "Sample_molecule_ch1": "molecule",
    "Sample_characteristics_ch1": "characteristics",
    "Sample_organism_ch1": "organism",
    "Sample_platform_id": "platform",
    "Sample_instrument_model": "instrument",
    "Sample_extract_protocol_ch1": "extract_protocol",
}


def parse(path):
    gse = os.path.basename(path).split("_")[0].split(".")[0]
    rows, cur = [], None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("^"):
                if cur:
                    rows.append(cur)
                cur = None
                if line.startswith("^SAMPLE"):
                    cur = {
                        "gse_id": gse,
                        "gsm_id": line.split("=", 1)[1].strip(),
                        "supp": [],
                        "sra": [],
                    }
                continue
            if cur is None or not line.startswith("!Sample_"):
                continue
            k, _, v = line[1:].partition(" = ")
            v = v.strip()
            if k.startswith("Sample_supplementary_file"):
                if v and v != "NONE":
                    cur["supp"].append(v.rsplit("/", 1)[-1])
            elif k == "Sample_relation":
                if v.startswith("SRA:"):
                    cur["sra"].append(v.rsplit("=", 1)[-1])
            elif k in KEEP:
                cur.setdefault(KEEP[k], []).append(v)
    if cur:
        rows.append(cur)
    for r in rows:
        for k in list(r):
            if isinstance(r[k], list):
                sep = " | " if k != "extract_protocol" else " "
                r[k] = sep.join(r[k])
        r["extract_protocol"] = r.get("extract_protocol", "")[:2000]
    return rows


if __name__ == "__main__":
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "soft", "*.soft.gz"))):
        try:
            out += parse(p)
        except Exception as e:  # noqa: BLE001
            print("bad", p, e, file=sys.stderr)
    df = pd.DataFrame(out)
    # a GSM listed in several series: keep the first (fields are per-sample anyway)
    df = df.drop_duplicates(["gse_id", "gsm_id"])
    df.to_parquet(os.path.join(HERE, "geo_samples.parquet"))
    print(len(df), df.gse_id.nunique())
