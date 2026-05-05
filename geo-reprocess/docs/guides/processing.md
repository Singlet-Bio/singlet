# Sample Processing Pipeline

## Overview

The processing pipeline takes individual GEO samples (GSMs) through download → protocol detection → quantification → QC → optional cleanup.

## Stage-by-stage

### Download

FASTQ files are downloaded from ENA (preferred) or SRA (fallback) with parallel segments:

```python
result = scgeo.process_sample(
    gsm_id="GSM3308545",
    gse_id="GSE115978",
    organism="human",
)
```

Configuration controls download behavior:
- **8 concurrent segments** for parallel download
- **MD5 verification** enabled by default
- **SRA fallback** via `fasterq-dump` if ENA unavailable

### Protocol Detection

Chemistry is detected from FASTQ read structure:
- Read lengths, adapter sequences, UMI patterns
- Supports: 10x-v2, 10x-v3, 10x-v4, Drop-seq, plate-based protocols
- Uses catalog metadata hints when available

### Quantification

Runs `simpleaf` with piscem index (36 threads, 1h timeout by default):
- Produces sparse barcode × gene count matrix
- Reports mapping rate = mapped_reads / total_reads

### Quality Control

- Cell calling: filter low-count barcodes
- Gene filtering: minimum counts per gene
- Per-cell metrics: n_genes, n_counts, % unspliced
- Adaptive QC: 3-MAD filtering if batch ≥ 20 cells

### Kraken2 Classification (optional)

Classifies non-host RNA for contamination detection:
- Reports fraction of non-host reads among detected cells

### Cleanup

Optionally deletes FASTQs and intermediate files after QC to free disk space.

## Batch processing

```python
# Process all samples in a series
results = scgeo.process_gse("GSE115978", samples=sample_list, organism="human")

# Process with download prefetching
results = scgeo.process_samples(samples, prefetch=True)
```
