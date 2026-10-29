"""Describe the processing catalog and the v2 plan study list (columns, SRR format, read counts)."""
import sys
import pandas as pd
cat = pd.read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet")
print(cat.shape); print(cat.dtypes.to_string())
print(cat.head(5).T.to_string()[:4000])
for c in cat.columns:
    if cat[c].dtype == object and cat[c].nunique() < 60:
        print("==", c, cat[c].value_counts(dropna=False).head(30).to_dict())
srr = [c for c in cat.columns if "srr" in c.lower() or "run" in c.lower()]
print("run-like cols", srr)
for c in srr:
    s = cat[c].astype(str)
    print(c, "with ';' or ',':", s.str.contains("[;,]").mean())
plan = pd.read_csv(sys.argv[1], sep="\t")
print("plan", plan.shape, list(plan.columns)[:30]); print(plan.head(3).T.to_string()[:2000])
