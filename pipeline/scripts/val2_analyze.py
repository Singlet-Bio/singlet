#!/usr/bin/env python3
"""Aggregate VAL2 results and generate flagged samples report."""

import csv
import sys
import json
from pathlib import Path
from collections import Counter

VAL2_BASE = Path("/mnt/projects/debruinz_project/singlify_validation/val2")
SAMPLES_CSV = Path("/mnt/home/debruinz/Singlet-AI/singlify/scripts/val2_samples.csv")

def load_samples():
    """Load expected sample metadata."""
    samples = {}
    with open(SAMPLES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples[row['srr']] = row
    return samples

def collect_results(samples):
    """Scan VAL2 directories for results."""
    results = []
    for srr, meta in samples.items():
        sample_dir = VAL2_BASE / srr
        result = {
            'srr': srr,
            'species': meta.get('species', 'unknown'),
            'protocol': meta.get('protocol_expected', 'unknown'),
            'modality': meta.get('modality', 'unknown'),
        }
        
        # Check download
        onefq = sample_dir / f"{srr}.1fq"
        result['downloaded'] = onefq.exists()
        result['onefq_size'] = onefq.stat().st_size if onefq.exists() else 0
        
        # Check processing
        output_dir = sample_dir / "output"
        pileup = output_dir / "pileup_stats.json"
        star_log = output_dir / "star_Log.final.out"
        
        result['processed'] = pileup.exists()
        result['mapping_rate'] = None
        result['cells'] = None
        
        if star_log.exists():
            with open(star_log) as f:
                for line in f:
                    if 'Uniquely mapped reads %' in line:
                        try:
                            result['mapping_rate'] = float(line.split('|')[1].strip().rstrip('%'))
                        except (ValueError, IndexError):
                            pass
        
        if pileup.exists():
            try:
                with open(pileup) as f:
                    stats = json.load(f)
                    result['cells'] = stats.get('cells_called', stats.get('n_barcodes'))
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        results.append(result)
    return results

def flag_samples(results):
    """Flag samples that need investigation."""
    flagged = []
    for r in results:
        flags = []
        
        if not r['downloaded']:
            flags.append('download_failed')
        elif not r['processed']:
            flags.append('processing_failed')
        
        if r['mapping_rate'] is not None:
            modality = r.get('modality', 'scRNA')
            if modality in ('scRNA', 'snRNA') and r['mapping_rate'] < 20:
                flags.append(f"low_mapping_{r['mapping_rate']:.1f}%")
            elif modality == 'ATAC' and r['mapping_rate'] < 15:
                flags.append(f"low_mapping_{r['mapping_rate']:.1f}%")
            elif r['mapping_rate'] < 10:
                flags.append(f"very_low_mapping_{r['mapping_rate']:.1f}%")
        
        if r['cells'] is not None and r['cells'] < 50:
            flags.append(f"few_cells_{r['cells']}")
        
        if flags:
            r['flags'] = '; '.join(flags)
            flagged.append(r)
    
    return flagged

def print_summary(results, flagged):
    """Print summary statistics."""
    total = len(results)
    downloaded = sum(1 for r in results if r['downloaded'])
    processed = sum(1 for r in results if r['processed'])
    
    print(f"=== VAL2 Summary ===")
    print(f"Total samples: {total}")
    print(f"Downloaded: {downloaded}/{total} ({100*downloaded/max(total,1):.1f}%)")
    print(f"Processed: {processed}/{total} ({100*processed/max(total,1):.1f}%)")
    
    # Mapping rate stats
    mapping_rates = [r['mapping_rate'] for r in results if r['mapping_rate'] is not None]
    if mapping_rates:
        mapping_rates.sort()
        print(f"Mapping rate: median={mapping_rates[len(mapping_rates)//2]:.1f}%, "
              f"mean={sum(mapping_rates)/len(mapping_rates):.1f}%, "
              f"min={min(mapping_rates):.1f}%, max={max(mapping_rates):.1f}%")
    
    # By species
    species_counts = Counter(r['species'] for r in results if r['processed'])
    print(f"\nProcessed by species: {dict(species_counts)}")
    
    # Flagged
    print(f"\nFlagged: {len(flagged)}/{total}")
    for f in flagged[:20]:
        print(f"  {f['srr']} ({f['species']}/{f['protocol']}): {f['flags']}")
    if len(flagged) > 20:
        print(f"  ... and {len(flagged)-20} more")

def main():
    samples = load_samples()
    results = collect_results(samples)
    flagged = flag_samples(results)
    print_summary(results, flagged)
    
    # Write flagged CSV
    output_csv = VAL2_BASE / "val2_flagged.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['srr', 'species', 'protocol', 'modality', 'mapping_rate', 'cells', 'flags'])
        writer.writeheader()
        for r in flagged:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(f"\nFlagged samples written to: {output_csv}")

if __name__ == '__main__':
    main()
