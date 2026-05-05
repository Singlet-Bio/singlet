#!/bin/bash
# Sequential resolution + batch rebuild
# Run on compute node after 10x_suspect finishes
set -e

echo "=== Phase 1: Resolve unknown protocols ==="
python3 -u /mnt/home/debruinz/Singlet-AI/geo-reprocess/scripts/resolve_non_geo_protocols.py \
    --target unknown 2>&1 | tee /mnt/projects/debruinz_project/cellarium/catalog/resolve_unknown_apply.log

echo ""
echo "=== Phase 2: Catalog summary ==="
python3 -c "
import pandas as pd
cat = pd.read_parquet('/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet')
non_geo = cat[cat['notes'].str.startswith('non_geo', na=False)]
print(f'Total catalog: {len(cat):,} rows')
print(f'Non-GEO: {len(non_geo):,} rows')
print()
print('Non-GEO protocol distribution:')
print(non_geo['protocol_inferred'].value_counts().to_string())
print()
droplet = ['10xv2', '10xv3', '10x_other', 'dropseq']
proc = non_geo[non_geo['protocol_inferred'].isin(droplet)]
has_srr = proc['srr_accessions'].notna() & (proc['srr_accessions'] != '')
has_ena = proc['ena_fastq_r1'].notna() & (proc['ena_fastq_r1'] != '')
processable = proc[has_srr | has_ena]
controlled = proc.get('controlled_access', pd.Series(dtype=bool))
if 'controlled_access' in proc.columns:
    processable = processable[processable['controlled_access'] != True]
print(f'Processable droplet samples: {len(processable):,}')
print(f'  By protocol: ')
for proto, cnt in processable['protocol_inferred'].value_counts().items():
    print(f'    {proto}: {cnt:,}')
print(f'  By species (top 10):')
for sp, cnt in processable['organism'].value_counts().head(10).items():
    print(f'    {sp}: {cnt:,}')
"

echo ""
echo "=== Phase 3: Build v11 batches ==="
python3 -c "
import pandas as pd
from pathlib import Path
cat = pd.read_parquet('/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet')
non_geo = cat[cat['notes'].str.startswith('non_geo', na=False)]
droplet = ['10xv2', '10xv3', '10x_other', 'dropseq']
proc = non_geo[non_geo['protocol_inferred'].isin(droplet)]
has_srr = proc['srr_accessions'].notna() & (proc['srr_accessions'] != '')
has_ena = proc['ena_fastq_r1'].notna() & (proc['ena_fastq_r1'] != '')
proc = proc[has_srr | has_ena]
if 'controlled_access' in proc.columns:
    proc = proc[proc['controlled_access'] != True]

batch_dir = Path('/mnt/projects/debruinz_project/cellarium/catalog/batches_v11_resolved')
batch_dir.mkdir(exist_ok=True)
for old in batch_dir.glob('*.csv'):
    old.unlink()

batch_size = 50
batch_num = 0
for start in range(0, len(proc), batch_size):
    batch = proc.iloc[start:start + batch_size]
    batch_num += 1
    rows = []
    for _, row in batch.iterrows():
        rows.append({
            'gsm_id': row.get('gsm_id', ''),
            'gse_id': row.get('gse_id', ''),
            'organism': row['organism'],
            'protocol': row['protocol_inferred'],
            'ena_r1': row.get('ena_fastq_r1', ''),
            'ena_r2': row.get('ena_fastq_r2', ''),
            'srr': row.get('srr_accessions', ''),
        })
    pd.DataFrame(rows).to_csv(batch_dir / f'batch_{batch_num:03d}.csv', index=False)

print(f'Wrote {batch_num} batch files ({len(proc)} samples) to {batch_dir}')
"

echo ""
echo "=== Done ==="
