#!/usr/bin/env python3
"""Metadata pipeline diagnostic script."""
import pandas as pd, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

cat = pd.read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/metadata_catalog.parquet")

skipped = cat[cat["stage2a_status"] == "skipped"]
has_tab = 0
no_tab = 0
no_any = 0
for _, row in skipped.iterrows():
    try:
        d = json.loads(row["stage0_formats"])
        if d.get("h5ad", 0) > 0 or d.get("tabular", 0) > 0 or d.get("loom", 0) > 0:
            has_tab += 1
        elif d.get("rds", 0) > 0:
            no_tab += 1
        else:
            no_any += 1
    except:
        no_any += 1

print("Stage2a skipped breakdown:")
print(f"  Has h5ad/tabular/loom (tried but no usable metadata): {has_tab}")
print(f"  Has RDS only (stage0 pre-skip, needs stage2b): {no_tab}")
print(f"  No metadata files at all: {no_any}")

# Stage2a done with 0% match
done_zero = cat[(cat["stage2a_status"] == "done") & (cat["stage2a_match_rate"] == 0)]
print(f"\nStage2a done with 0% match: {len(done_zero)}")
for _, row in done_zero.head(10).iterrows():
    print(f"  {row['gse_id']}/{row['gsm_id']}: src={row['stage2a_source']}, fmt={row['stage2a_format']}, ncols={row['stage2a_n_cols']}")

# Stage2a done with >0 but <50% match
done_low = cat[(cat["stage2a_status"] == "done") & (cat["stage2a_match_rate"] > 0) & (cat["stage2a_match_rate"] < 0.5)]
print(f"\nStage2a done with 0-50% match: {len(done_low)}")
for _, row in done_low.head(5).iterrows():
    print(f"  {row['gse_id']}/{row['gsm_id']}: src={row['stage2a_source']}, match={row['stage2a_match_rate']:.2%}")

# Combined stage2b
print(f"\n=== Stage 2b ===")
for status in ["done", "skipped", "failed", "pending"]:
    n = (cat["stage2b_status"] == status).sum()
    if n > 0:
        print(f"  {status}: {n}")
done_2b = cat[cat["stage2b_status"] == "done"]
print(f"  2b match>=0.5: {(done_2b['stage2b_match_rate'] >= 0.5).sum()}")
print(f"  2b match>0: {(done_2b['stage2b_match_rate'] > 0).sum()}")

# Stage2c
print(f"\n=== Stage 2c ===")
for status in ["done", "skipped", "failed", "pending"]:
    n = (cat["stage2c_status"] == status).sum()
    if n > 0:
        print(f"  {status}: {n}")

# Overall: best metadata for each GSM
print("\n=== OVERALL BEST CELL-LEVEL METADATA ===")
has_any_cell = cat[
    ((cat["stage2a_match_rate"] > 0) & (cat["stage2a_status"] == "done")) |
    ((cat["stage2b_match_rate"] > 0) & (cat["stage2b_status"] == "done")) |
    ((cat["stage2c_match_rate"] > 0) & (cat["stage2c_status"] == "done"))
]
print(f"GSMs with ANY cell-level metadata match: {len(has_any_cell)} / {len(cat)} ({100*len(has_any_cell)/len(cat):.1f}%)")

has_good = cat[
    ((cat["stage2a_match_rate"] >= 0.5) & (cat["stage2a_status"] == "done")) |
    ((cat["stage2b_match_rate"] >= 0.5) & (cat["stage2b_status"] == "done")) |
    ((cat["stage2c_match_rate"] >= 0.5) & (cat["stage2c_status"] == "done"))
]
print(f"GSMs with >=50% match: {len(has_good)} / {len(cat)} ({100*len(has_good)/len(cat):.1f}%)")

has_great = cat[
    ((cat["stage2a_match_rate"] >= 0.9) & (cat["stage2a_status"] == "done")) |
    ((cat["stage2b_match_rate"] >= 0.9) & (cat["stage2b_status"] == "done")) |
    ((cat["stage2c_match_rate"] >= 0.9) & (cat["stage2c_status"] == "done"))
]
print(f"GSMs with >=90% match: {len(has_great)} / {len(cat)} ({100*len(has_great)/len(cat):.1f}%)")

# Check quant directory structure
quant_base = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
cat_gses = set(cat["gse_id"].unique())
quant_gses = sorted([d for d in os.listdir(quant_base) if d.startswith("GSE")])
missing_gses = sorted(set(quant_gses) - cat_gses)
print(f"\n=== QUANT VS CATALOG GAP ===")
print(f"Quant GSEs: {len(quant_gses)}")
print(f"Catalog GSEs: {len(cat_gses)}")
print(f"Missing from catalog: {len(missing_gses)}")

# Count GSMs in missing GSEs
missing_gsm_count = 0
for gse in missing_gses:
    gse_path = os.path.join(quant_base, gse)
    try:
        gsms = [d for d in os.listdir(gse_path) if d.startswith("GSM")]
        missing_gsm_count += len(gsms)
    except:
        pass
print(f"Missing GSMs (estimated): {missing_gsm_count}")

# Check structure of quant GSMs
for gse in missing_gses[:3]:
    gse_path = os.path.join(quant_base, gse)
    gsms = sorted([d for d in os.listdir(gse_path) if d.startswith("GSM")])[:1]
    for gsm in gsms:
        gsm_path = os.path.join(gse_path, gsm)
        files = os.listdir(gsm_path)
        print(f"  {gse}/{gsm}: {files}")

# What's the relationship between dataset/ and quant/?
# The metadata pipeline seeds from dataset/ (which has cells.parquet)
# The processing pipeline outputs to quant/ (which has counts.1pz or counts.spz)
# dataset/ was populated by a separate export step
print(f"\nDataset dir has cells.parquet; quant dir has counts.1pz (or legacy counts.spz)")
print(f"Metadata catalog seeded from dataset/ only - {len(missing_gses)} GSEs need export to dataset/ first")
