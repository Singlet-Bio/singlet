#!/usr/bin/env python3
"""
Catalog cleanup: remove samples that will 100% fail from the processing catalog.

Categories removed:
  1. Known plate-based protocols (smartseq2, celseq, marsseq, etc.)
  2. Prescreened smartseq (FASTQ-confirmed by prescreener)
  3. No download path (no SRR accessions AND no ENA R1 URL)
  4. Non-processable assays (chipseq, methylation, Hi-C, etc.)
     NOTE: ATAC, CITE-seq, Visium, spatial are KEPT for future multimodal processing
  5. Unsupported species (no reference in scgeo.config.species.SPECIES_REF)
  6. SINGLE-end droplet protocols (10x/indrop/dropseq with SINGLE layout)

Outputs:
  - geo_single_cell_catalog.parquet (overwritten with cleaned version)
  - geo_single_cell_catalog_removed_YYYYMMDD.parquet (backup of removed rows)
  - catalog_cleanup_report.json (summary stats)
"""

import pandas as pd
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add scgeo to path for species config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scgeo.config.species import SPECIES_REF, ORGANISM_TO_TAXON

CATALOG_DIR = Path('/mnt/projects/debruinz_project/cellarium/catalog')
CATALOG_PATH = CATALOG_DIR / 'geo_single_cell_catalog.parquet'
PRESCREENER = Path('/mnt/projects/debruinz_project/cellarium/pipeline/prescreener/prescreener_results.csv')

# --- Load ---
print("Loading catalog...")
cat = pd.read_parquet(CATALOG_PATH)
pre = pd.read_csv(PRESCREENER)
original_count = len(cat)
print(f"  Original catalog: {original_count:,} rows")

# --- Define removal categories ---
p_lower = cat['protocol_inferred'].str.lower()
org_lower = cat['organism'].str.lower().fillna('')

PLATE_PROTOS = {'smartseq2', 'smartseq3', 'plate_based', 'marsseq', 'celseq', 
                'celseq2', 'strtseq', 'quartzseq', 'icell8'}

# Only remove assays that can never produce gene expression data.
# KEEP: ATAC (10x_atac, atacseq, scatac), CITE-seq (citeseq, cite_seq_adt),
#       spatial (visium, 10x_visium, slideseq, spatial) — needed for
#       future multimodal processing and ADT+RNA / ATAC+RNA bridges.
NON_PROCESSABLE_ASSAYS = {'chipseq', 'methylation',
                          'hi_c', 'mirna_seq', 'rip_seq',
                          'dnase_seq', 'mnase_seq', 'bulkrna'}

DROPLET_PROTOS = {'10xv2', '10xv3', '10xv3_5prime', '10xv4', '10x_multiome',
                  '10x_suspect', 'dropseq', 'bd_rhapsody', 'surecell', 'ddseq', 
                  'indrops', 'indrop', 'parse', 'dnbelab', 'seqwell', 'scifi'}

ss_gsms = set(pre[pre['classification'] == 'smartseq']['gsm_id'])

no_srr = cat['srr_accessions'].isna() | (cat['srr_accessions'].astype(str).isin(['', 'nan']))
no_r1 = cat['ena_fastq_r1'].isna() | (cat['ena_fastq_r1'].astype(str).isin(['', 'nan']))

# --- Build removal masks ---
masks = {}
masks['plate_based'] = p_lower.isin(PLATE_PROTOS)
masks['prescreened_smartseq'] = cat['gsm_id'].isin(ss_gsms)
masks['no_download_path'] = no_srr & no_r1
masks['non_processable_assay'] = p_lower.isin(NON_PROCESSABLE_ASSAYS)

# Build supported species set from scgeo.config.species (all non-skip entries)
_supported_species = set()
for _tid, _info in SPECIES_REF.items():
    if not _info.get('skip', False):
        _supported_species.add(_info['name'].lower())
for _name, _tid in ORGANISM_TO_TAXON.items():
    if not SPECIES_REF.get(_tid, {}).get('skip', False):
        _supported_species.add(_name.lower())

def _is_supported_species(org_str):
    org = str(org_str).lower().strip()
    if org in _supported_species:
        return True
    if ';' in org:
        for part in org.split(';'):
            p = part.strip()
            if p and p in _supported_species:
                return True
    return False

masks['unsupported_species'] = ~cat['organism'].apply(_is_supported_species)
masks['single_end_droplet'] = p_lower.isin(DROPLET_PROTOS) & (cat['library_layout'] == 'SINGLE')

# --- Compute union ---
remove = pd.Series(False, index=cat.index)
category_stats = {}
for name, mask in masks.items():
    net_new = (mask & ~remove).sum()
    total = mask.sum()
    category_stats[name] = {'total': int(total), 'net_new': int(net_new)}
    remove |= mask

# --- Tag each removed row with its primary removal reason ---
cat['removal_reason'] = ''
remaining_to_tag = remove.copy()
for name, mask in masks.items():
    to_tag = mask & remaining_to_tag
    cat.loc[to_tag, 'removal_reason'] = name
    remaining_to_tag &= ~mask

# --- Report ---
removed_count = remove.sum()
remaining_count = (~remove).sum()

print(f"\n=== REMOVAL SUMMARY ===")
running = 0
for name, stats in category_stats.items():
    running += stats['net_new']
    print(f"  {name}: {stats['total']:>9,} total | +{stats['net_new']:>7,} net-new | running: {running:>9,}")
print(f"\n  TOTAL REMOVED: {removed_count:,}")
print(f"  TOTAL REMAINING: {remaining_count:,}")

# --- Backup original ---
datestamp = datetime.now().strftime('%Y%m%d')
backup_path = CATALOG_DIR / f'geo_single_cell_catalog_pre_cleanup_{datestamp}.parquet'
print(f"\nBacking up original catalog to {backup_path}...")
shutil.copy2(CATALOG_PATH, backup_path)

# --- Save removed rows (with reason) ---
removed_path = CATALOG_DIR / f'geo_single_cell_catalog_removed_{datestamp}.parquet'
removed_df = cat[remove].copy()
print(f"Saving {len(removed_df):,} removed rows to {removed_path}...")
removed_df.to_parquet(removed_path, index=False)

# --- Save cleaned catalog ---
cleaned = cat[~remove].drop(columns=['removal_reason']).copy()
print(f"Saving cleaned catalog ({len(cleaned):,} rows) to {CATALOG_PATH}...")
cleaned.to_parquet(CATALOG_PATH, index=False)

# --- Save report ---
report = {
    'timestamp': datetime.now().isoformat(),
    'original_count': int(original_count),
    'removed_count': int(removed_count),
    'remaining_count': int(remaining_count),
    'categories': category_stats,
    'remaining_protocol_distribution': dict(cleaned['protocol_inferred'].value_counts()),
    'remaining_organism_distribution': dict(cleaned['organism'].value_counts().head(10)),
    'backup_path': str(backup_path),
    'removed_path': str(removed_path),
}
report_path = CATALOG_DIR / f'catalog_cleanup_report_{datestamp}.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"Report saved to {report_path}")

print(f"\n=== DONE ===")
print(f"  Before: {original_count:,}")
print(f"  Removed: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
print(f"  After: {remaining_count:,}")
