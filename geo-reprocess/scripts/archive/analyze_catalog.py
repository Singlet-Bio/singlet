#!/usr/bin/env python3
"""Build priority-annotated catalog with species and modality breakdowns.

Reads stage7_multimodal_catalog.parquet and produces a priority-annotated view.
"""
import pyarrow.parquet as pq
import pyarrow.compute as pc
from collections import Counter
import os

catalog_dir = '/mnt/projects/debruinz_project/cellarium/catalog'
cat_path = os.path.join(catalog_dir, 'stage7_multimodal_catalog.parquet')

table = pq.read_table(cat_path)
print(f"Total catalog rows: {table.num_rows:,}")
print(f"Unique GSMs: {pc.count_distinct(table.column('gsm_id')).as_py():,}")
print(f"Unique GSEs: {pc.count_distinct(table.column('gse_id')).as_py():,}")

organisms = [str(o).lower().strip() if o else 'unknown' for o in table.column('organism').to_pylist()]
assay_classes = table.column('assay_class').to_pylist()
library_types = table.column('library_type').to_pylist()
gsm_ids = table.column('gsm_id').to_pylist()
gse_ids = table.column('gse_id').to_pylist()
protocols = table.column('protocol_inferred').to_pylist() if 'protocol_inferred' in table.column_names else ['unknown']*len(organisms)

def classify_species(org):
    org = org.lower().strip()
    if 'homo sapiens' in org and 'mus musculus' not in org:
        return 'human'
    elif 'mus musculus' in org and 'homo sapiens' not in org:
        return 'mouse'
    elif 'homo sapiens' in org and 'mus musculus' in org:
        return 'human_mouse_mix'
    elif 'danio rerio' in org:
        return 'zebrafish'
    elif 'drosophila' in org:
        return 'drosophila'
    elif 'rattus' in org:
        return 'rat'
    elif 'macaca' in org:
        return 'macaque'
    elif 'caenorhabditis' in org:
        return 'c_elegans'
    else:
        return 'other'

def assign_priority(species, assay_class, library_type):
    is_human = species in ('human', 'human_mouse_mix')
    if is_human:
        if assay_class == 'visium':
            return 1
        if library_type in ('rna', 'unknown'):
            return 1
        if library_type == 'atac':
            return 2
        if library_type in ('adt', 'guide'):
            return 3
        return 1
    if library_type in ('rna', 'unknown') or assay_class == 'visium':
        return 4
    return 5

species_list = [classify_species(o) for o in organisms]
priorities = [assign_priority(s, a, l) for s, a, l in zip(species_list, assay_classes, library_types)]

print("\n" + "="*80)
print("PRIORITY BREAKDOWN (SRR-level)")
print("="*80)
priority_counts = Counter(priorities)
desc = {1: 'Human transcriptomics (incl Visium/multi-omic GEX)',
        2: 'Human ATAC', 3: 'Human other (ADT/guide)',
        4: 'Other species transcriptomics', 5: 'Other species multi-omics'}
for p in sorted(priority_counts):
    print(f"  P{p}: {priority_counts[p]:>8,}  {desc.get(p, '')}")
print(f"  Total: {sum(priority_counts.values()):>8,}")

print("\n" + "="*80)
print("PRIORITY BREAKDOWN (unique GSMs)")
print("="*80)
priority_gsms = {}
for p, g in zip(priorities, gsm_ids):
    priority_gsms.setdefault(p, set()).add(g)
for p in sorted(priority_gsms):
    print(f"  P{p}: {len(priority_gsms[p]):>8,} GSMs  {desc.get(p, '')}")

print("\n" + "="*80)
print("SPECIES BREAKDOWN")
print("="*80)
species_counts = Counter(species_list)
for sp, cnt in species_counts.most_common():
    gsm_set = set(g for s, g in zip(species_list, gsm_ids) if s == sp)
    gse_set = set(g for s, g in zip(species_list, gse_ids) if s == sp)
    print(f"  {sp:25s}: {cnt:>8,} SRRs, {len(gsm_set):>7,} GSMs, {len(gse_set):>5,} GSEs")

print("\n" + "="*80)
print("MODALITY x SPECIES MATRIX (unique GSMs)")
print("="*80)
sm = {}
for sp, ac, gs in zip(species_list, assay_classes, gsm_ids):
    sm.setdefault((sp, ac), set()).add(gs)

header = f"{'Species':25s} {'rna_only':>10} {'multiome':>10} {'citeseq':>10} {'visium':>10} {'perturbseq':>10} {'TOTAL':>10}"
print(header)
print("-" * len(header))
for sp in ['human', 'mouse', 'human_mouse_mix', 'zebrafish', 'macaque', 'rat', 'drosophila', 'c_elegans', 'other']:
    total = 0
    parts = [f"{sp:25s}"]
    for ac in ['rna_only', 'multiome', 'citeseq', 'visium', 'perturbseq']:
        n = len(sm.get((sp, ac), set()))
        parts.append(f"{n:>10,}")
        total += n
    if total > 0:
        parts.append(f"{total:>10,}")
        print(" ".join(parts))

print("\n" + "="*80)
print("PROTOCOL DISTRIBUTION (top 15)")
print("="*80)
for p, cnt in Counter(protocols).most_common(15):
    print(f"  {p:25s}: {cnt:>8,}")

print("\n" + "="*80)
print("LIBRARY TYPE x SPECIES (unique GSMs)")
print("="*80)
sl = {}
for sp, lt, gs in zip(species_list, library_types, gsm_ids):
    sl.setdefault((sp, lt), set()).add(gs)
header = f"{'Species':25s} {'rna':>10} {'atac':>10} {'adt':>10} {'guide':>10} {'unknown':>10}"
print(header)
print("-" * len(header))
for sp in ['human', 'mouse', 'human_mouse_mix', 'zebrafish', 'macaque', 'rat', 'other']:
    parts = [f"{sp:25s}"]
    for lt in ['rna', 'atac', 'adt', 'guide', 'unknown']:
        n = len(sl.get((sp, lt), set()))
        parts.append(f"{n:>10,}")
    print(" ".join(parts))

# Standalone ATAC check
print("\n" + "="*80)
print("STANDALONE ATAC-seq CHECK")
print("="*80)
atac_mm, atac_other = set(), set()
for ac, lt, gs in zip(assay_classes, library_types, gsm_ids):
    if lt == 'atac':
        (atac_mm if ac == 'multiome' else atac_other).add(gs)
print(f"  ATAC in multiome: {len(atac_mm):,} GSMs")
print(f"  ATAC non-multiome: {len(atac_other):,} GSMs")
print(f"  NOTE: standalone scATAC-seq not in 'expression profiling' DataSet Type may be missing")

# Head of catalog
print("\n" + "="*80)
print("HEAD OF CATALOG (first 10 rows)")
print("="*80)
key_cols = ['gsm_id', 'gse_id', 'organism', 'assay_class', 'library_type', 'protocol_inferred', 'is_droplet']
for i in range(min(10, table.num_rows)):
    row = {c: table.column(c)[i].as_py() for c in key_cols if c in table.column_names}
    print(f"  {row}")
