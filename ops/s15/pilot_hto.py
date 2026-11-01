#!/usr/bin/env python3
"""S15: Singlet HTO on a pilot sample. Counts the HTO library (python/singlet/adt/count.py, branch claude/adt-modality-0637a6) against the pilot
GEX run's called cells and applies Singlet's hashtag caller. Panel and offset come from the atlas run of the same library.
    pilot_hto.py ATLAS_DIR FASTQ_DIR CELL_CALLS WHITELIST OUT"""
import glob, json, os, sys
import numpy as np
import pandas as pd
atlas, fqd, cc, wlp, out = sys.argv[1:6]
import singlet.adt.count as C
import singlet.adt.hto as H
os.makedirs(out, exist_ok=True)
st = json.load(open(atlas + "/status.json")); off = int(st["offset"]); umi = int((st.get("chemistry") or {}).get("umi_len") or 10)
panel = pd.read_csv(atlas + "/panel.tsv", sep="\t"); seqs = list(panel.sequence); names = list(panel.name)
cells = [l.split("\t")[0].split("-")[0] for l in open(cc).read().splitlines()[1:] if l.split("\t")[1] == "TRUE"]
r1s = sorted(glob.glob(fqd + "/*_2.fastq.gz")); r2s = sorted(glob.glob(fqd + "/*_3.fastq.gz")); assert r1s and len(r1s) == len(r2s), (r1s, r2s)
wl = C.load_barcode_codes(wlp)
nreads = sum(int(os.popen("zcat %s | wc -l" % f).read()) // 4 for f in r1s[:1]) * len(r1s)
shards = int(np.ceil(nreads / 15e6)) if nreads > 60e6 else 0
lc = C.count_library(r1s, r2s, cb_len=16, umi_len=umi, offset=off, panel_sequences=seqs, whitelist_codes=wl, gex_barcodes=cells, shards=shards, tmpdir="/tmp")
M = lc.matrix.toarray().astype(float)
call, lfc_top, lfc_sec = H.call_hashtags(M, lc.ambient)
pd.DataFrame(M.T, index=cells, columns=names).to_csv(out + "/hto_counts.csv.gz")
lab = np.where(call == -2, "doublet", np.where(call < 0, "negative", np.array(names + ["x"])[np.clip(call, 0, len(names))]))
pd.DataFrame({"barcode": cells, "call": call, "label": lab, "lfc_top": lfc_top, "lfc_second": lfc_sec}).to_csv(out + "/singlet_hto_calls.csv", index=False)
json.dump({"offset": off, "umi_len": umi, "n_reads_est": nreads, "shards": shards, "features": names, "stats": lc.stats, "n_cells": len(cells),
           "calls": {k: int(v) for k, v in zip(*np.unique(lab, return_counts=True))}}, open(out + "/summary.json", "w"), default=str, indent=1)
print(open(out + "/summary.json").read()[:1500])
