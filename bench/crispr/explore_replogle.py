# SPDX-License-Identifier: MIT
"""Look at the Replogle 2022 inputs: Table S1 sheets, h5ad obs columns, a few guide reads."""
import gzip
import sys

import pandas as pd

d = sys.argv[1]
x = pd.ExcelFile(f"{d}/replogle2022_tableS1.xlsx")
for s in x.sheet_names:
    df = x.parse(s, nrows=5)
    print("SHEET", s, list(df.columns))
    print(df.head(3).to_string()[:1500])
import anndata as ad  # noqa: E402

a = ad.read_h5ad(f"{d}/K562_essential_raw_singlecell_01.h5ad", backed="r")
print(a)
print(a.obs.head(5).to_string()[:3000])
print(a.var.head(3).to_string())
for c in a.obs.columns:
    if a.obs[c].dtype.name in ("category", "object"):
        print(c, a.obs[c].nunique(), list(a.obs[c].astype(str).unique()[:5]))
print(a.obs.index[:5].tolist(), a.obs.index[-5:].tolist())
with gzip.open(f"{d}/fastqs/SRR19331011_S1_L001_R2_001.fastq.gz", "rt") as f:
    for i, ln in enumerate(f):
        if i % 4 == 1:
            print(ln.strip())
        if i > 60:
            break
with gzip.open(f"{d}/fastqs/SRR19331011_S1_L001_R1_001.fastq.gz", "rt") as f:
    print([next(f) for _ in range(4)][1].strip())
