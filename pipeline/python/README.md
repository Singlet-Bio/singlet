# singlify

Streaming BAM pileup engine for single-cell multi-omics feature extraction.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import singlify

result = singlify.pileup(
    bam_path="sample.bam",
    barcode_path="barcodes.tsv",
    exon_gtf_path="genes.gtf.gz",
)

# Exon counts as scipy CSC matrix (features × barcodes)
print(result.exons.shape)

# Convert to AnnData for scanpy
adata = result.to_anndata()
```

## Features

- **Single-pass extraction**: Exons, introns, splice junctions, SNP genotypes, chrM alleles
- **Zero-copy**: C++ engine returns scipy sparse matrices directly
- **Streaming**: Processes unsorted BAM from STAR pipe or sorted BAM from disk
- **UMI deduplication**: Built-in FNV-1a hash-based UMI dedup
