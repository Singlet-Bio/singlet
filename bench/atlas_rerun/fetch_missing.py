"""Run info for samples the per-project report did not cover, from accessions.

    python fetch_missing.py <wanted.tsv> <out.parquet>

wanted.tsv: gsm, srx, srr (either srx or srr may be empty; srr may be ';'-joined). Runs are
resolved run -> experiment with batched ENA portal searches, then every run of each experiment
is listed (so a sample with several runs gets all of them, not just the SRR we knew).
"""

from __future__ import annotations

import io
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from fetch_runinfo import FIELDS, total_bytes

API = "https://www.ebi.ac.uk/ena/portal/api/search"


def search(field: str, values: list[str], fields: str) -> pd.DataFrame:
    q = " OR ".join(f'{field}="{v}"' for v in values)
    data = urllib.parse.urlencode({"result": "read_run", "query": q, "fields": fields,
                                   "format": "tsv", "limit": 0}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(API, data=data,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = r.read().decode()
            return pd.read_csv(io.StringIO(txt), sep="\t", dtype=str) if txt.strip() else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            print("retry", field, attempt, e, flush=True)
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def batched(field, values, fields, n=100):
    out = []
    for i in range(0, len(values), n):
        df = search(field, values[i:i + n], fields)
        if len(df):
            out.append(df)
        if i % 2000 == 0:
            print(field, i, len(values), flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    want = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
    srx = set(want["srx"].loc[want["srx"].str.startswith(("SRX", "ERX", "DRX"))])
    srr = sorted({r for v in want.loc[want["srx"] == "", "srr"] for r in v.split(";") if r})
    if srr:
        m = batched("run_accession", srr, "run_accession,experiment_accession")
        srx |= set(m["experiment_accession"].dropna())
    runs = batched("experiment_accession", sorted(srx), FIELDS)
    for c in ("read_count", "base_count", "nominal_length"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce")
    for c in ("fastq_bytes", "sra_bytes", "submitted_bytes"):
        runs[c + "_total"] = runs[c].map(total_bytes)
    # map back to GSM: via SRX given, or via the SRR we knew
    gsm_by_srx = dict(zip(want["srx"], want["gsm"]))
    srr2gsm = {r: g for g, v in zip(want["gsm"], want["srr"]) for r in v.split(";") if r}
    runs["gsm"] = runs["experiment_accession"].map(gsm_by_srx)
    exp2gsm = {e: srr2gsm[r] for r, e in zip(runs["run_accession"], runs["experiment_accession"]) if r in srr2gsm}
    runs["gsm"] = runs["gsm"].fillna(runs["experiment_accession"].map(exp2gsm))
    runs.to_parquet(sys.argv[2])
    print("runs", len(runs), "gsm", runs["gsm"].nunique(), flush=True)


if __name__ == "__main__":
    main()
