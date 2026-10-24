# SPDX-License-Identifier: MIT
"""Inventory of the atlas's antibody-capture libraries (ADT / CITE-seq and HTO hashtags) for the ADT modality.

Inputs (read-only, metadata only; no reads are downloaded here):

* the library-type screen (Singlet-Bio/singlet#21, ``bench/libtype_screen/libtype_suspects.tsv``): one row per flagged
  sample with its class, confidence and matching mRNA GSM;
* the screen's parsed GEO sample records (``geo_samples.parquet``) and cached family SOFT files (series-level
  supplementary files, BioProject, PubMed ids, summary and design text);
* the screen's D1 QC export (``d1_qc.parquet``), for the paired GEX sample's protocol and cell counts;
* ENA ``filereport`` (one call per BioProject, cached as JSON) for runs, read counts and FASTQ availability.

Output: ``adt_hto_inventory.tsv``, one row per ADT/HTO library (high/medium confidence), with the paired GEX sample,
chemistry hints, run accessions and the panel evidence found in the metadata. The panel itself is settled from the
reads (``singlet.adt.panel``); this table only records where a feature reference might come from.

    python -P bench/adt/inventory.py --screen DIR --out bench/adt/inventory
"""

import argparse
import gzip
import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

CLASSES = ("ADT", "HTO")
ENA_FIELDS = ("run_accession,experiment_accession,sample_accession,read_count,base_count,fastq_ftp,fastq_bytes,"
              "submitted_ftp,submitted_format,submitted_bytes,sra_bytes,library_layout,instrument_model")

# Text hints. Matched case-insensitively on sample title/description/protocol/characteristics and series text.
HINTS = {
    "totalseq_a": r"total\s*-?\s*seq\s*(?:\W|tm|™)*\s*-?\s*a\b",
    "totalseq_b": r"total\s*-?\s*seq\s*(?:\W|tm|™)*\s*-?\s*b\b",
    "totalseq_c": r"total\s*-?\s*seq\s*(?:\W|tm|™)*\s*-?\s*c\b",
    "totalseq": r"total\s*-?\s*seq",
    "biolegend": r"biolegend",
    "abseq": r"ab\s*-?\s*seq|bd\s+rhapsody|rhapsody",
    "hashtag": r"hash\s*-?\s*tag|cell\s*hashing|\bhto\b",
    "cite_seq": r"cite\s*-?\s*seq",
    "five_prime": r"5['’`]|five\s*prime|5\s*prime|5'\s*v[12]",
    "three_v3": r"3['’`]\s*v3|v3(?:\.1)?\s*chemistry|single\s*cell\s*3['’`]?\s*(?:reagent\s*kits?\s*)?v3",
    "three_v2": r"3['’`]\s*v2|v2\s*chemistry",
    "feature_barcoding": r"feature\s*barcod",
}
HINT_RE = {k: re.compile(v, re.I) for k, v in HINTS.items()}

# Supplementary file names that may carry a feature reference or the authors' own antibody counts.
PANEL_FILE = re.compile(r"feature[_\-. ]?ref|featureref|antibod|totalseq|adt[_\-. ]?(?:panel|barcode|ref|seq)|"
                        r"hto[_\-. ]?(?:panel|barcode|ref|seq)|tag[_\-. ]?(?:list|seq|barcode)|cite[_\-. ]?(?:ref|panel)|"
                        r"barcodes?\.csv|panel", re.I)
AUTHOR_COUNTS = re.compile(r"(?:adt|hto|protein|antibody|cite|hashtag)[^/]*\.(?:csv|tsv|txt|mtx|h5|rds|h5ad|loom)(?:\.gz)?$"
                           r"|feature_bc_matrix|filtered_feature|raw_feature|\.h5$", re.I)


def read_series(soft_dir, gse):
    """Series-level fields from the cached family SOFT file."""
    out = {"series_supp": [], "bioproject": [], "pubmed": [], "series_text": [], "superseries_of": [], "subseries_of": []}
    path = os.path.join(soft_dir, f"{gse}.soft.gz")
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("^SAMPLE"):
                break
            if not line.startswith("!Series_"):
                continue
            k, _, v = line[1:].partition(" = ")
            v = v.strip()
            if k == "Series_supplementary_file" and v and v != "NONE":
                out["series_supp"].append(v.rsplit("/", 1)[-1])
            elif k == "Series_relation":
                if v.startswith("BioProject:"):
                    out["bioproject"].append(v.rsplit("/", 1)[-1])
                elif v.startswith("SuperSeries of:"):
                    out["superseries_of"].append(v.split(":", 1)[1].strip())
                elif v.startswith("SubSeries of:"):
                    out["subseries_of"].append(v.split(":", 1)[1].strip())
            elif k == "Series_pubmed_id":
                out["pubmed"].append(v)
            elif k in ("Series_summary", "Series_overall_design", "Series_title"):
                out["series_text"].append(v)
    return out


def ena_runs(cache_dir, accession, tries=4):
    """All runs of one ENA/SRA study, BioProject or experiment accession (cached)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{accession}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    q = urllib.parse.urlencode({"accession": accession, "result": "read_run", "fields": ENA_FIELDS,
                                "format": "json", "limit": 0})
    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?{q}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read().decode()
            rows = json.loads(body) if body.strip() else []
            break
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                rows = []
                break
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    else:
        return None
    with open(path, "w") as f:
        json.dump(rows, f)
    return rows


def hints(text):
    return ",".join(k for k, rx in HINT_RE.items() if rx.search(text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="dir with libtype_suspects.tsv, geo_samples.parquet, d1_qc.parquet")
    ap.add_argument("--soft", required=True, help="dir of cached <GSE>.soft.gz family files")
    ap.add_argument("--cache", required=True, help="ENA filereport cache dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sus = pd.read_csv(os.path.join(a.screen, "libtype_suspects.tsv"), sep="\t")
    sus = sus[sus.lib_class.isin(CLASSES) & sus.confidence.isin(["high", "medium"])].copy()
    geo = pd.read_parquet(os.path.join(a.screen, "geo_samples.parquet")).set_index("gsm_id")
    qc = pd.read_parquet(os.path.join(a.screen, "d1_qc.parquet")).drop_duplicates("gsm_id").set_index("gsm_id")

    series = {g: read_series(a.soft, g) for g in sorted(sus.gse_id.unique())}

    # ENA: one call per BioProject of each series (falls back to the sample's SRX accessions).
    runs_by_srx = {}
    for i, (gse, s) in enumerate(series.items()):
        for bp in s["bioproject"]:
            rows = ena_runs(a.cache, bp)
            for r in rows or []:
                runs_by_srx.setdefault(r["experiment_accession"], []).append(r)
        if i % 50 == 0:
            print(f"ENA {i}/{len(series)}", flush=True)
    for gsm in sus.gsm_id:
        for srx in str(geo.loc[gsm, "sra"]).split(" | ") if gsm in geo.index else []:
            if srx and srx not in runs_by_srx:
                rows = ena_runs(a.cache, srx)
                runs_by_srx[srx] = rows or []

    out = []
    for r in sus.itertuples():
        g = geo.loc[r.gsm_id] if r.gsm_id in geo.index else None
        s = series[r.gse_id]
        srxs = [x for x in str(g["sra"]).split(" | ") if x] if g is not None else []
        runs = [run for x in srxs for run in runs_by_srx.get(x, [])]
        sample_text = " ".join(str(g[k]) for k in ("title", "description", "extract_protocol", "characteristics",
                                                    "library_strategy")) if g is not None else ""
        series_text = " ".join(s["series_text"])
        supp = [x for x in str(g["supp"]).split(" | ") if x] if g is not None else []
        panel_files = sorted({f for f in supp + s["series_supp"] if PANEL_FILE.search(f)})
        author_files = sorted({f for f in supp + s["series_supp"] if AUTHOR_COUNTS.search(f)})
        mrna = r.mrna_gsm if isinstance(r.mrna_gsm, str) else ""
        gq = qc.loc[mrna] if mrna and mrna in qc.index else None
        n_fastq = [len(str(run.get("fastq_ftp") or "").split(";")) if run.get("fastq_ftp") else 0 for run in runs]
        out.append({
            "gse": r.gse_id, "gsm": r.gsm_id, "lib_class": r.lib_class, "confidence": r.confidence,
            "title": r.title, "organism": g["organism"] if g is not None else "",
            "gex_gsm": mrna, "gex_in_atlas": bool(r.mrna_in_atlas) if mrna else False,
            "gex_protocol": gq["protocol"] if gq is not None else "",
            "gex_n_called": int(gq["n_cells_called"]) if gq is not None and pd.notna(gq["n_cells_called"]) else -1,
            "gex_reference": gq["reference_build"] if gq is not None else "",
            "adt_atlas_protocol": r.protocol, "adt_atlas_n_called": r.n_cells_called,
            "srx": ";".join(srxs), "runs": ";".join(run["run_accession"] for run in runs),
            "n_runs": len(runs),
            "read_count": sum(int(run.get("read_count") or 0) for run in runs),
            "mean_spot_len": round(sum(int(run.get("base_count") or 0) for run in runs) /
                                   max(1, sum(int(run.get("read_count") or 0) for run in runs)), 1),
            "fastq_files_per_run": ",".join(map(str, n_fastq)),
            "fastq_gb": round(sum(sum(int(b) for b in str(run.get("fastq_bytes") or "0").split(";") if b)
                                  for run in runs) / 1e9, 3),
            "submitted_format": ",".join(sorted({str(run.get("submitted_format") or "") for run in runs})),
            "hints_sample": hints(sample_text), "hints_series": hints(series_text),
            "panel_files": ";".join(panel_files), "author_count_files": ";".join(author_files[:20]),
            "bioproject": ";".join(s["bioproject"]), "pubmed": ";".join(s["pubmed"]),
            "superseries": ";".join(s["subseries_of"]),
        })
    df = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "adt_hto_inventory.tsv"), sep="\t", index=False)
    print(df.groupby("lib_class").agg(n=("gsm", "size"), studies=("gse", "nunique"),
                                      gex_in_atlas=("gex_in_atlas", "sum"),
                                      with_runs=("n_runs", lambda x: (x > 0).sum()),
                                      fastq_gb=("fastq_gb", "sum")))


if __name__ == "__main__":
    main()
