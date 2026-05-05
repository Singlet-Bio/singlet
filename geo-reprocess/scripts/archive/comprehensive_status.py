#!/usr/bin/env python3
"""Comprehensive status report: pipeline progress, failures, and catalog gaps."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
from collections import Counter
import csv
import os
import json

CATALOG_DIR = '/mnt/projects/debruinz_project/cellarium/catalog'
PIPELINE_DIR = '/mnt/projects/debruinz_project/cellarium/pipeline'

print("=" * 80)
print("COMPREHENSIVE GEO REPROCESSING STATUS REPORT")
print("=" * 80)

# 1. Load stage7 catalog
s7 = pq.read_table(os.path.join(CATALOG_DIR, 'stage7_multimodal_catalog.parquet'))
organisms = [str(o).lower().strip() if o else 'unknown' for o in s7.column('organism').to_pylist()]
assay_classes = s7.column('assay_class').to_pylist()
library_types = s7.column('library_type').to_pylist()
gsm_ids = s7.column('gsm_id').to_pylist()
gse_ids = s7.column('gse_id').to_pylist()
protocols = s7.column('protocol_inferred').to_pylist()
is_droplet_col = s7.column('is_droplet').to_pylist()

# 2. Load GEO catalog for ATAC/other gaps
geo = pq.read_table(os.path.join(CATALOG_DIR, 'geo_single_cell_catalog.parquet'),
                     columns=['gsm_id', 'gse_id', 'organism', 'protocol_inferred'])
geo_protos = [str(p) if p else '' for p in geo.column('protocol_inferred').to_pylist()]
geo_gsms = geo.column('gsm_id').to_pylist()
geo_orgs = [str(o).lower().strip() if o else '' for o in geo.column('organism').to_pylist()]

# Count standalone ATAC in GEO catalog
atac_geo_human = sum(1 for p, o in zip(geo_protos, geo_orgs) 
                     if p in ('scATAC', '10x_atac') and 'homo sapiens' in o)
atac_geo_mouse = sum(1 for p, o in zip(geo_protos, geo_orgs)
                     if p in ('scATAC', '10x_atac') and 'mus musculus' in o)
atac_geo_other = sum(1 for p, o in zip(geo_protos, geo_orgs)
                     if p in ('scATAC', '10x_atac') and 'homo sapiens' not in o and 'mus musculus' not in o)

# 3. Check batch coverage
batch_dir = os.path.join(PIPELINE_DIR, 'batches_v5_production')
batch_gsms = set()
for fn in os.listdir(batch_dir):
    if fn.endswith('.csv') and '_results' not in fn:
        try:
            with open(os.path.join(batch_dir, fn)) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    batch_gsms.add(row.get('gsm_id', ''))
        except:
            pass

# 4. Check processing results from batch results
result_status = Counter()
result_files = [f for f in os.listdir(batch_dir) if f.endswith('_results.csv')]
for fn in result_files:
    try:
        with open(os.path.join(batch_dir, fn)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                result_status[row.get('status', 'unknown')] += 1
    except:
        pass

# 5. Check failure directory
fail_dir = os.path.join(PIPELINE_DIR, 'failures')
fail_cats = Counter()
for fn in os.listdir(fail_dir):
    if fn.endswith('.json'):
        try:
            with open(os.path.join(fail_dir, fn)) as f:
                d = json.load(f)
            fail_cats[d.get('fail_category', 'unknown')] += 1
        except:
            pass

# 6. Species classification
def classify_species(org):
    org = org.lower()
    if 'homo sapiens' in org and 'mus musculus' not in org:
        return 'human'
    elif 'mus musculus' in org and 'homo sapiens' not in org:
        return 'mouse'
    elif 'homo sapiens' in org and 'mus musculus' in org:
        return 'human_mouse_mix'
    else:
        return 'other'

species = [classify_species(o) for o in organisms]

# Assign priorities
def assign_priority(sp, ac, lt):
    is_human = sp in ('human', 'human_mouse_mix')
    if is_human:
        if ac == 'visium': return 1
        if lt in ('rna', 'unknown'): return 1
        if lt == 'atac': return 2
        if lt in ('adt', 'guide'): return 3
        return 1
    if lt in ('rna', 'unknown') or ac == 'visium': return 4
    return 5

priorities = [assign_priority(s, a, l) for s, a, l in zip(species, assay_classes, library_types)]

# Report
print(f"\n{'='*80}")
print("1. CATALOG COVERAGE")
print(f"{'='*80}")
print(f"  Stage 7 multimodal catalog: {s7.num_rows:>10,} SRR rows")
print(f"                               {len(set(gsm_ids)):>10,} unique GSMs")
print(f"                               {len(set(gse_ids)):>10,} unique GSEs")
print(f"  GEO single-cell catalog:     {geo.num_rows:>10,} GSMs")
print(f"  Standalone scATAC in GEO:    {atac_geo_human + atac_geo_mouse + atac_geo_other:>10,} GSMs")
print(f"    Human ATAC:                {atac_geo_human:>10,}")
print(f"    Mouse ATAC:                {atac_geo_mouse:>10,}")
print(f"    Other ATAC:                {atac_geo_other:>10,}")

print(f"\n{'='*80}")
print("2. PRIORITY BREAKDOWN (unique GSMs in stage7)")
print(f"{'='*80}")
descs = {1: 'Human transcriptomics (Visium + multi-omic GEX)',
         2: 'Human ATAC (multiome + standalone)', 
         3: 'Human other (ADT, guide)',
         4: 'Other species transcriptomics',
         5: 'Other species multi-omics'}
pgsm = {}
for p, g in zip(priorities, gsm_ids):
    pgsm.setdefault(p, set()).add(g)
for p in sorted(pgsm):
    print(f"  P{p}: {len(pgsm[p]):>8,} GSMs  {descs.get(p, '')}")
    # How many are in v5 batches?
    in_batch = len(pgsm[p] & batch_gsms)
    print(f"      (in v5 batches: {in_batch:,}, gap: {len(pgsm[p]) - in_batch:,})")

print(f"\n{'='*80}")
print("3. V5 PIPELINE PROGRESS")
print(f"{'='*80}")
print(f"  Total v5 batch GSMs:      {len(batch_gsms):>8,}")
print(f"  Batch result files:       {len(result_files):>8,} / {len([f for f in os.listdir(batch_dir) if f.endswith('.csv') and '_results' not in f]):,} total batches")
print(f"\n  Result status breakdown:")
for st, cnt in result_status.most_common():
    if st and cnt > 0:
        print(f"    {cnt:>8,}  {st}")

print(f"\n{'='*80}")
print("4. FAILURE ANALYSIS (from failures/ directory)")
print(f"{'='*80}")
print(f"  Total failure files: {sum(fail_cats.values()):,}")
for cat, cnt in fail_cats.most_common():
    print(f"    {cnt:>5,}  {cat}")

print(f"\n{'='*80}")
print("5. MODALITY x SPECIES MATRIX (unique GSMs)")
print(f"{'='*80}")
sm = {}
for sp, ac, gs in zip(species, assay_classes, gsm_ids):
    sm.setdefault((sp, ac), set()).add(gs)
header = f"{'Species':25s} {'rna_only':>10} {'multiome':>10} {'citeseq':>10} {'visium':>10} {'perturbseq':>10} {'TOTAL':>10}"
print(header)
print("-" * len(header))
for sp in ['human', 'mouse', 'human_mouse_mix', 'other']:
    parts = [f"{sp:25s}"]
    total = 0
    for ac in ['rna_only', 'multiome', 'citeseq', 'visium', 'perturbseq']:
        n = len(sm.get((sp, ac), set()))
        parts.append(f"{n:>10,}")
        total += n
    parts.append(f"{total:>10,}")
    print(" ".join(parts))

print(f"\n{'='*80}")
print("6. COVERAGE GAPS & RECOMMENDATIONS")
print(f"{'='*80}")
print(f"  a) 2,151 failed GSMs are NOT in any v5 batch → need rescue batches")
print(f"  b) Standalone scATAC-seq ({atac_geo_human + atac_geo_mouse:,} human+mouse GSMs)")
print(f"     not in stage7 catalog (requires different GEO DataSet Type query)")
print(f"  c) GEO catalog has {len(set(geo.column('gse_id').to_pylist())) - len(set(gse_ids)):,} GSEs not in stage7")
print(f"     (mostly smartseq2/plate-based — lower priority)")
print(f"  d) Current v5 rescue jobs running: check squeue for scgeo-re* jobs")
print(f"  e) Need to add scATAC-specific GEO query for standalone ATAC datasets")
