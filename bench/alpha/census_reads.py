"""Reads used / reads available for every Research Alpha sample (runs on Clipper, read-only inputs).

    python -P census_reads.py GSM_LIST CAMPAIGN_SAMPLES_CSV CATALOG_PARQUET OUT_TSV_GZ

reads_used      = total_reads from the campaign's per-sample results (pipeline-benchmark/results/campaign/
                  campaign_samples.csv, harvested from pileup_stats/summary of the GRCh38 campaign)
reads_available = read_count (SRA spots) in Singlet's processing catalog for the sample
"""
import sys

import pandas as pd

gl, camp, cat, out = sys.argv[1:5]
g = pd.read_csv(gl, sep="\t")
c = pd.read_csv(camp, usecols=["gsm", "total_reads", "wall_s", "threads", "protocol", "wrong_strand", "exon_hits", "version"])
c = c.drop_duplicates("gsm", keep="last").rename(columns={"gsm": "gsm_id", "protocol": "campaign_protocol"})
k = pd.read_parquet(cat, columns=["gsm_id", "read_count", "srr_accessions"]).drop_duplicates("gsm_id", keep="last")
k["n_srr"] = k.srr_accessions.fillna("").astype(str).str.count(";") + (k.srr_accessions.fillna("").astype(str) != "")
d = g.merge(c, on="gsm_id", how="left").merge(k.drop(columns=["srr_accessions"]), on="gsm_id", how="left")
d["frac_used"] = d.total_reads / d.read_count
d.to_csv(out, sep="\t", index=False, compression="gzip")
print(len(d), "rows; with total_reads", d.total_reads.notna().sum(), "with read_count", d.read_count.notna().sum())
print(d.frac_used.describe())
