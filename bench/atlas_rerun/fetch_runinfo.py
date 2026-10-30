"""Every SRA run behind every sample of the v2 plan studies, from ENA.

    python fetch_runinfo.py <plan.tsv> <out_dir>

For each BioProject / SRA study linked to a plan GSE in Singlet's processing catalog, ENA's
`filereport` (result=read_run) lists every run with its read and base counts and its download
sizes. Runs are matched to GSMs by experiment/sample alias (GEO puts the GSM there), falling back
to the catalog's SRX. Writes:
  runs.parquet     one row per run (project, run, experiment, gsm, read_count, base_count, bytes)
  samples.parquet  one row per catalog GSM of the plan studies: n_runs, reads, bases, bytes, the
                   catalog's own read_count (what the old campaign saw), status and protocol fields
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

CATALOG = "/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet"
FIELDS = ",".join([
    "run_accession", "experiment_accession", "sample_accession", "study_accession",
    "experiment_alias", "sample_alias", "run_alias", "read_count", "base_count",
    "library_strategy", "library_source", "library_layout", "instrument_model",
    "fastq_bytes", "sra_bytes", "submitted_bytes", "nominal_length",
])


def ena(project: str, cache: str) -> pd.DataFrame:
    path = os.path.join(cache, f"{project}.tsv")
    if not os.path.exists(path):
        url = ("https://www.ebi.ac.uk/ena/portal/api/filereport?"
               + urllib.parse.urlencode({"accession": project, "result": "read_run", "fields": FIELDS,
                                         "format": "tsv", "limit": 0}))
        txt = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    txt = r.read().decode()
                break
            except Exception as e:  # noqa: BLE001
                if any(code in str(e) for code in ("400", "404", "204")):
                    txt = ""
                    break
                time.sleep(2 ** attempt)
        if txt is None:
            return pd.DataFrame()
        with open(path, "w") as f:
            f.write(txt)
    txt = open(path).read()
    if not txt.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
    df["project"] = project
    return df


def total_bytes(s) -> float:
    """ENA lists one size per file, ';'-separated."""
    if not isinstance(s, str) or not s:
        return float("nan")
    return float(sum(int(x) for x in s.split(";") if x.strip().isdigit()))


def gsm_of(row) -> str | None:
    for c in ("experiment_alias", "sample_alias", "run_alias"):
        v = row.get(c)
        if isinstance(v, str) and v.startswith("GSM"):
            return v.split(" ")[0].split(":")[0].split("_")[0]
    return None


def main():
    plan_tsv, out = sys.argv[1], sys.argv[2]
    cache = os.path.join(out, "ena")
    os.makedirs(cache, exist_ok=True)
    plan = pd.read_csv(plan_tsv, sep="\t")
    cat = pd.read_parquet(CATALOG)
    cat = cat[cat["gse_id"].isin(set(plan["gse"]))].copy()
    print("catalog rows in plan studies", len(cat), "GSE", cat["gse_id"].nunique(), flush=True)

    projects = sorted((set(cat["sra_study"].dropna()) | set(cat["bioproject"].dropna())) - {""})
    print("projects", len(projects), flush=True)
    with ThreadPoolExecutor(8) as ex:
        frames = list(ex.map(lambda p: ena(p, cache), projects))
    runs = pd.concat([f for f in frames if len(f)], ignore_index=True)
    runs = runs.drop_duplicates("run_accession")
    for c in ("read_count", "base_count", "nominal_length"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce")
    for c in ("fastq_bytes", "sra_bytes", "submitted_bytes"):
        runs[c + "_total"] = runs[c].map(total_bytes)
    print("runs", len(runs), flush=True)

    runs["gsm"] = runs.apply(gsm_of, axis=1)
    srx2gsm = dict(zip(cat["srx_accession"], cat["gsm_id"]))
    miss = runs["gsm"].isna()
    runs.loc[miss, "gsm"] = runs.loc[miss, "experiment_accession"].map(srx2gsm)
    runs.to_parquet(os.path.join(out, "runs.parquet"))

    g = runs.dropna(subset=["gsm"]).groupby("gsm")
    per = pd.DataFrame({
        "n_runs": g.size(),
        "n_experiments": g["experiment_accession"].nunique(),
        "reads": g["read_count"].sum(min_count=1),
        "bases": g["base_count"].sum(min_count=1),
        "fastq_bytes": g["fastq_bytes_total"].sum(min_count=1),
        "sra_bytes": g["sra_bytes_total"].sum(min_count=1),
        "max_run_reads": g["read_count"].max(),
        "library_layout": g["library_layout"].first(),
    })
    keep = ["gsm_id", "gse_id", "organism", "protocol_inferred", "processing_status", "target_assay",
            "srx_accession", "srr_accessions", "read_count", "base_count", "title_lib_type",
            "screen_vdj", "screen_adt_read_length", "screen_non_gex_title", "avg_read_length"]
    s = cat[keep].rename(columns={"read_count": "catalog_read_count", "base_count": "catalog_base_count"})
    s = s.merge(per, left_on="gsm_id", right_index=True, how="left")
    s["catalog_n_srr"] = s["srr_accessions"].fillna("").map(
        lambda x: len([y for y in x.replace(",", ";").split(";") if y.strip()]))
    s.to_parquet(os.path.join(out, "samples.parquet"))
    summ = {
        "catalog_gsm": int(len(s)),
        "gsm_with_ena_runs": int(s["n_runs"].notna().sum()),
        "runs": int(len(runs)),
        "done_gsm": int((s["processing_status"] == "done").sum()),
        "projects": len(projects),
    }
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
