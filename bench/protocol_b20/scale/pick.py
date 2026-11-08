#!/usr/bin/env python3
"""B20 at scale: pick ~300 Beta-corpus samples stratified by stated assay, one sample per study, one gene-expression run each.

Reads the corpus tables (singlet-agents corpus/beta, Clipper data/corpus-beta/out) and writes picks.tsv:
sample, study, assay, run, instrument, fastq_urls (ENA, semicolon-joined), odp_url. Detection probe only: no engine run.
"""
import sys
import pandas as pd

STRATA = {  # stated assay -> samples
    "10x_3p_v3": 50, "10x_3p_v2": 50, "10x_5p": 80, "10x_3p_v1": 15, "10x_3p_v4": 15, "10x_multiome": 15,
    "visium": 25, "dropseq": 10, "seqwell": 8, "bd_rhapsody": 10, "indrop": 8, "dnbelab_c4": 8, "singleron": 6,
}


def main(out_dir, dest, seed=20261108):
    s = pd.read_parquet(out_dir + "/samples.parquet", columns=["sample", "study", "assay", "eligible", "lib_types", "n_gex_runs_with_url"])
    r = pd.read_parquet(out_dir + "/runs.parquet", columns=["run_accession", "sample", "lib_type", "instrument_model", "fastq_urls", "odp_url", "read_count"])
    s = s[(s.eligible == True) & s.assay.isin(STRATA) & (s.n_gex_runs_with_url >= 1)]
    r = r[(r.lib_type == "gex") & r.fastq_urls.fillna("").str.count(";").ge(1) & (r.read_count.fillna(0) >= 100000)]
    s = s[s["sample"].isin(set(r["sample"]))]
    rows = []
    for assay, k in STRATA.items():
        a = s[s.assay == assay].sample(frac=1.0, random_state=seed).drop_duplicates("study")   # one sample per study
        for _, x in a.head(k).iterrows():
            run = r[r["sample"] == x["sample"]].sort_values("run_accession").iloc[0]
            rows.append([x["sample"], x["study"], assay, run.run_accession, run.instrument_model, run.fastq_urls, run.odp_url or ""])
    df = pd.DataFrame(rows, columns=["sample", "study", "assay", "run", "instrument", "fastq_urls", "odp_url"])
    df.to_csv(dest, sep="\t", index=False)
    print(df.groupby("assay").size().to_string(), "\ntotal", len(df), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
