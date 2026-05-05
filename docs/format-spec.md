# .1pz Format Specification

The `.1pz` format is singlet's compressed single-cell data format. It stores:

- **Sparse count matrix** (CSC, zstd-compressed)
- **Cell barcodes** with QC annotations
- **Gene names + IDs** (Ensembl)
- **Sample metadata** (organism, protocol, QC metrics)
- **Optional layers**: spliced/unspliced, ADT, ATAC peaks

## File Structure

```
[header: 64 bytes]
[metadata: msgpack]
[gene_names: zstd]
[cell_barcodes: zstd]
[matrix_indptr: zstd]
[matrix_indices: zstd]
[matrix_data: zstd]
[layers...]
```

## Reading

```python
import singlet
adata = singlet.read("sample.1pz")
```

```r
library(singlet)
sce <- read_1pz("sample.1pz")
```
