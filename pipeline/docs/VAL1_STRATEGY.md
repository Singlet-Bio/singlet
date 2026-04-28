# VAL1: Large-Scale SRA Validation Strategy

## Goal
Validate singlify on 200+ diverse SRA samples to ensure zero-config processing works across the single-cell landscape.

## Diversity Axes

### Protocol Families (target ≥3 samples each)
| Family | Examples | Priority |
|--------|----------|----------|
| 10x Chromium v2 | 10xv2, 10xv2-5p | High |
| 10x Chromium v3 | 10xv3, 10xv3-HT | High |
| 10x Chromium v4 | Not yet in lib1fq | Medium |
| Drop-seq | dropseq | High |
| sci-RNA-seq | sci-RNA-seq3 | High |
| ddSEQ | ddseq | Medium |
| inDrop | indrop-v1/v2/v3 | Medium |
| STRT-seq | strt-seq, strt-seq-2i | Low |
| CEL-seq | cel-seq2 | Low |
| Smart-seq2 | plate-based, no barcode | Medium |
| 10x ATAC | sc-atac, arc-atac | Medium |
| 10x Multiome | arc-gex + arc-atac | Medium |
| Visium | visium-v1/v2 | Medium |
| CITE-seq | totalseq-a/b/c | Medium |
| Bulk RNA-seq | standard PE/SE | Low |

### Species (target ≥10 samples each for top 2, ≥3 for others)
- Human (GRCh38): 80+ samples
- Mouse (GRCm39): 60+ samples
- Rat, Zebrafish, Drosophila, C. elegans: 5-10 each
- Other mammals (pig, dog, macaque): 3-5 each

### Quality Tiers
- **Tier 1 – Gold**: High-quality datasets from published papers, >50% mapping expected
- **Tier 2 – Silver**: Standard quality, 30-50% mapping
- **Tier 3 – Challenging**: Low quality, mixed species, novel protocols, <30% mapping expected

### Read Count Bins
- Small: <10M reads (5 samples)
- Medium: 10-100M reads (150 samples)
- Large: 100M-1B reads (30 samples)
- Very large: >1B reads (5 samples)

## Data Sourcing

1. **Primary**: geo-reprocess catalog (`scgeo` package at geo-reprocess/)
   - Contains protocol, species, assay metadata for thousands of GEO series
   - Use `filter_catalog()` API to draw stratified sample
   - Each GEO series may have multiple SRR accessions

2. **Secondary**: Manual curation for underrepresented categories
   - ATAC, CITE-seq, Visium, Smart-seq2 may need manual SRA search
   - Use NCBI SRA search with protocol-specific keywords

3. **Download**: `singlify download SRR... -o <path>.1fq` (built-in SRA streaming)
   - Or `fasterq-dump` for pre-existing corpus entries
   - Target: ~500GB total for 200 samples

## Pass Criteria

| Modality | Metric | Pass | Warn | Fail |
|----------|--------|------|------|------|
| scRNA | Mapping rate | >30% | 10-30% | <10% |
| scRNA | Cells called | >100 | 10-100 | <10 |
| scRNA | Gene r vs STARsolo | >0.99 | 0.95-0.99 | <0.95 |
| scATAC | Mapping rate | >30% | 10-30% | <10% |
| Smart-seq2 | Mapping rate | >50% | 20-50% | <20% |
| Bulk RNA | Mapping rate | >60% | 30-60% | <30% |
| All | Exit code | 0 | - | ≠0 |
| All | Wall time | <10min | 10-60min | >60min |

## Failure Triage Classification

- **A – Bad data**: Empty reads, wrong organism, corrupted files → skip, note in manifest
- **B – Pipeline bug**: Crash, wrong output, regression → create DAG node, fix
- **C – Unsupported protocol**: New protocol not in registry → add support or document limitation
- **D – Edge case**: Unusual but valid data that triggers unexpected behavior → defensive fix

## Scale Estimates

- Download: ~2.5GB/sample average × 200 = ~500GB storage
- .1fq encoding: ~30s/sample average
- Processing: ~120s/sample average at 20T
- Total compute: ~200 × 150s ≈ 8.3 hours single-threaded on c001
- With 4 parallel: ~2 hours
- SLURM array: `sbatch --array=1-200` with per-sample scripts

## Automation Plan

1. **Draw script**: `scripts/val1_draw_samples.py` — reads geo-reprocess catalog, applies stratification, outputs sample list CSV
2. **Download script**: `scripts/val1_download.sh` — SLURM array, downloads + encodes to .1fq
3. **Process script**: `scripts/val1_process.sh` — SLURM array, runs singlify, captures metrics
4. **Triage script**: `scripts/val1_triage.py` — reads metrics, classifies pass/warn/fail, generates report

## Timeline

- Cycle 87-88: Draw sample list, start downloads
- Cycle 89-92: Process in batches (50/cycle)
- Cycle 93: Triage + fix pipeline bugs
- Cycle 94+: Re-run fixed samples
