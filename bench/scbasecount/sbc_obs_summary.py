"""Per-SRX cell counts and median UMIs / genes from scBaseCount's cell table, read by HTTP range (only the needed columns).

Usage (Clipper job): python -P sbc_obs_summary.py OUT.parquet Homo_sapiens Mus_musculus ...
Needs EDROP (a checkout with bench/empty_droplets/fetch.py) for the RangeFile reader.
"""
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.environ["EDROP"], "bench/empty_droplets"))
from fetch import RangeFile  # noqa: E402

OUT = sys.argv[1]
BASE = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/scbasecount/2026-01-12/metadata/GeneFull_Ex50pAS"
frames = []
for org in sys.argv[2:]:
    pf = pq.ParquetFile(RangeFile(f"{BASE}/{org}/obs_metadata.parquet", cache_blocks=16))
    names = pf.schema_arrow.names
    print(org, pf.metadata.num_rows, names, flush=True)
    srx = next(c for c in names if c.lower() == "srx_accession")
    want = [srx] + [c for c in ("umi_count_Unique", "gene_count_Unique", "umi_count", "gene_count") if c in names][:2]
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=want).to_pandas()
        t.columns = ["srx", "umi", "genes"]
        g = t.groupby("srx", observed=True).agg(n=("umi", "size"), umi_med=("umi", "median"), genes_med=("genes", "median"),
                                                 umi_sum=("umi", "sum"))
        g["organism_dir"] = org
        frames.append(g.reset_index())
        if i % 20 == 0:
            print(org, "row group", i, "/", pf.num_row_groups, flush=True)
d = pd.concat(frames)
# an SRX can straddle row groups: medians are recombined weighted by cells (approximate only for straddlers)
d["umi_w"] = d.umi_med * d.n
d["genes_w"] = d.genes_med * d.n
d = d.groupby(["srx", "organism_dir"]).agg(n=("n", "sum"), umi_w=("umi_w", "sum"), genes_w=("genes_w", "sum"),
                                           umi_sum=("umi_sum", "sum"), parts=("n", "size")).reset_index()
d["umi_med"] = d.umi_w / d.n
d["genes_med"] = d.genes_w / d.n
d.drop(columns=["umi_w", "genes_w"]).to_parquet(OUT)
print("done", len(d), "srx;", int((d.parts > 1).sum()), "straddle row groups")
