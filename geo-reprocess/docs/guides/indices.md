# Reference Index Building

## Overview

Build splici reference indices from genome FASTA and GTF files for use with simpleaf/piscem.

## Supported organisms

| Key | Species | NCBI Taxon |
|-----|---------|-----------|
| `human` | Homo sapiens | 9606 |
| `mouse` | Mus musculus | 10090 |
| `rat` | Rattus norvegicus | 10116 |
| `zebrafish` | Danio rerio | 7955 |
| `drosophila` | Drosophila melanogaster | 7227 |
| `c_elegans` | Caenorhabditis elegans | 6239 |

## Building indices

```python
import scgeo

# Build for a single organism
scgeo.build_index("human")

# Build for multiple organisms
scgeo.build_indices(["human", "mouse", "zebrafish"])

# Check existing index path
print(scgeo.get_index_path("human"))
```

## What happens during index build

1. **Download** genome FASTA (~3 GB for human) and GTF annotation (~60 MB)
2. **Build splici** reference (transcript + intron sequences)
3. **Build piscem** index for fast pseudoalignment

Index build takes ~10 minutes per organism.

## CLI

```bash
sc-geo index build --organism human
sc-geo index build --organisms human mouse zebrafish
```
