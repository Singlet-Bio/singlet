#!/usr/bin/env python3
"""Check discovery provenance and coverage gaps."""
import pyarrow.parquet as pq
from collections import Counter
import sys

cat = pq.read_table('/mnt/projects/debruinz_project/cellarium/catalog/stage7_multimodal_catalog.parquet')

# Discovery modalities
if 'discovery_modalities' in cat.column_names:
    dm = cat.column('discovery_modalities').to_pylist()
    dm_counts = Counter(str(d) if d else 'none' for d in dm)
    print("=== Discovery modality provenance (top 15) ===")
    for d, cnt in dm_counts.most_common(15):
        print(f"  {cnt:7,}  {d}")
else:
    print("No discovery_modalities column")

ac = cat.column('assay_class').to_pylist()
gse = cat.column('gse_id').to_pylist()

# Visium check
visium_gses = set(g for a, g in zip(ac, gse) if a == 'visium')
print(f"\n=== Visium: {len(visium_gses):,} GSEs ===")

if 'discovery_modalities' in cat.column_names:
    vis_dm = Counter()
    for a, d, g in zip(ac, dm, gse):
        if a == 'visium':
            vis_dm[str(d)] += 1
    print("Visium by discovery query:")
    for d, cnt in vis_dm.most_common(10):
        print(f"  {cnt:7,}  {d}")

# Multiome check
mm_gses = set(g for a, g in zip(ac, gse) if a == 'multiome')
print(f"\n=== Multiome: {len(mm_gses):,} GSEs ===")

# CITEseq check
cs_gses = set(g for a, g in zip(ac, gse) if a == 'citeseq')
print(f"\n=== CITEseq: {len(cs_gses):,} GSEs ===")

# Perturbseq check
ps_gses = set(g for a, g in zip(ac, gse) if a == 'perturbseq')
print(f"\n=== Perturbseq: {len(ps_gses):,} GSEs ===")

sys.stdout.flush()
