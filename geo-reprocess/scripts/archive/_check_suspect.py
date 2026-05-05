import pandas as pd
cat = pd.read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet")
suspect = cat[(cat["protocol_inferred"] == "10x_suspect") & (cat["notes"].str.startswith("non_geo", na=False))]
print("10x_suspect non-geo:", len(suspect))
r1_empty = suspect["ena_fastq_r1"].fillna("").str.len() <= 5
print("R1 empty:", r1_empty.sum())
print("R1 present:", (~r1_empty).sum())
for _, row in suspect[~r1_empty].head(3).iterrows():
    print(row["gsm_id"], row["ena_fastq_r1"][:80])
for _, row in suspect[r1_empty].head(3).iterrows():
    print(row["gsm_id"], "SRR:", row["srr_accessions"][:40])
