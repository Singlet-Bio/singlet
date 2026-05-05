#!/usr/bin/env python3
"""Verify the non-GEO catalog merge."""
import pandas as pd

cat = pd.read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet")
print(f"Catalog: {len(cat):,} rows x {cat.shape[1]} cols")
print()

# Non-GEO samples
non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]
print(f"Non-GEO samples: {len(non_geo):,}")
print(f"  HCA BioProject: {(non_geo['notes'] == 'non_geo_hca_bioproject').sum():,}")
print(f"  EMTAB: {(non_geo['notes'] == 'non_geo_emtab').sum():,}")
print(f"  CellxGene SRA: {(non_geo['notes'] == 'non_geo_cellxgene_sra').sum():,}")
print()

# Protocol distribution for non-GEO
print("Non-GEO protocol distribution:")
for proto, cnt in non_geo["protocol_inferred"].value_counts().items():
    print(f"  {proto}: {cnt:,}")
print()

# What's processable via our simpleaf pipeline?  
processable = non_geo[non_geo["protocol_inferred"].isin(["10xv2", "10xv3", "dropseq"])]
print(f"Immediately processable (10x/dropseq): {len(processable):,}")
print(f"  10xv2: {(processable['protocol_inferred'] == '10xv2').sum()}")
print(f"  10xv3: {(processable['protocol_inferred'] == '10xv3').sum()}")
print(f"  dropseq: {(processable['protocol_inferred'] == 'dropseq').sum()}")
print()

# 10x_suspect need protocol resolution
suspect = non_geo[non_geo["protocol_inferred"] == "10x_suspect"]
print(f"10x_suspect (need protocol resolution): {len(suspect):,}")
print()

# Smart-seq2 needs different pipeline
ss2 = non_geo[non_geo["protocol_inferred"] == "smartseq2"]
print(f"Smart-seq2 (needs plate-based pipeline): {len(ss2):,}")
print()

# Species for processable samples
print("Processable by organism:")
for org, cnt in processable["organism"].value_counts().items():
    print(f"  {org}: {cnt}")
print()

# ENA FASTQ availability for processable
print(f"Processable with ENA R1: {(processable['ena_fastq_r1'] != '').sum()}")
print(f"Processable with ENA R2: {(processable['ena_fastq_r2'] != '').sum()}")
print(f"Processable with SRR: {(processable['srr_accessions'] != '').sum()}")

# Check species ref assignment
print()
print("Processable by species_ref_genome:")
for ref, cnt in processable["species_ref_genome"].value_counts().head(10).items():
    print(f"  {repr(ref)}: {cnt}")

# Check unknown_sc samples to understand what they might be
print()
unknown = non_geo[non_geo["protocol_inferred"] == "unknown_sc"]
print(f"unknown_sc by library_layout:")
for layout, cnt in unknown["library_layout"].value_counts().items():
    print(f"  {layout}: {cnt}")

# Show some gse_ids for processable
print()
print("Processable sample gse_id examples:")
for gse in processable["gse_id"].unique()[:10]:
    sub = processable[processable["gse_id"] == gse]
    print(f"  {gse}: {len(sub)} samples, protocols={sub['protocol_inferred'].unique()}, "
          f"organism={sub['organism'].unique()}")
