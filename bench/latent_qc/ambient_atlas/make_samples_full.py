"""samples_full.tsv = fetched + cached samples that produced a record, with tissue labels. usage: make_samples_full.py sample_meta.csv recs.txt"""
import re
import sys

import pandas as pd

meta = pd.read_csv(sys.argv[1]).fillna("")
recs = set(open(sys.argv[2]).read().split())
s = pd.read_csv("bench/latent_qc/ambient_atlas/samples.tsv", sep="\t").fillna("")
src = open("bench/latent_qc/ambient_atlas/select_samples.py").read()
ns = {"re": re}
exec(src[src.index("RULES"):src.index('d["tissue_coarse"]')], ns)
extra = recs - set(s.gsm)
m = meta[meta.gsm.isin(extra)].copy()
m["tissue_coarse"] = m.tissue_label.map(ns["coarse"])
m["org"] = m.organism.map(lambda x: "human" if "sapiens" in x else "mouse" if "musculus" in x else "")
m = m[(m.org != "") & (m.tissue_coarse != "")]
full = pd.concat([s, m[s.columns]])
f = full[full.gsm.isin(recs)]
f.to_csv("bench/latent_qc/ambient_atlas/samples_full.tsv", sep="\t", index=False, lineterminator="\n")
print(len(extra), len(f), f.org.value_counts().to_dict(), f[f.org == "human"].tissue_label.nunique(), f.gse.nunique())
