# R Integration

## Overview

singlepress provides native R bindings via Rcpp for reading and writing .1pz files.
The R API integrates with the single-cell R ecosystem: **Seurat**, **SingleCellExperiment**,
**Matrix** (dgCMatrix), and standard R data types.

Files written by the Python API are byte-compatible with the R API, and vice versa.

## Installation

```r
# Option 1: Source the files directly
Rcpp::sourceCpp("R/singlepress_rcpp.cpp")
source("R/singlepress.R")

# Option 2: Install as an R package
devtools::install_github("Singlet-AI/singlepress", subdir = "R")
```

**System requirements:** libzstd, C++17 compiler.

## The OnePZ Class

The `OnePZ` S4 class is the R equivalent of Python's `OnePZFile` — a lazy file handle
that reads header metadata on construction.

```r
opz <- OnePZ("counts.1pz")
opz
# OnePZ: counts.1pz
#   Shape:   30,000 features x 10,000 cells
#   NNZ:     15,000,000 (5.0% dense)
#   Size:    45.2 MB
#   Format:  .1pz v3
#   Stored:  names, colsums, obs/var
```

### Dimensions

```r
dim(opz)    # [1] 30000 10000
nrow(opz)   # 30000
ncol(opz)   # 10000
n_obs(opz)  # 10000 (AnnData convention — cells)
n_vars(opz) # 30000 (AnnData convention — features)
```

### Indexing with `[`

```r
# Integer subsetting
sub <- opz[1:100, 1:500]           # dgCMatrix

# Name-based subsetting (requires stored rownames/colnames)
sub <- opz[c("CD3D", "CD3E"), ]    # specific genes × all cells

# Logical subsetting
mask <- c(rep(TRUE, 100), rep(FALSE, 9900))
sub <- opz[, mask]
```

### Transpose

```r
t_mat <- t(opz)   # returns dgCMatrix (cells × genes)
```

### Head / Tail

```r
head(opz, n = 10)   # first 10 cells (dgCMatrix)
tail(opz, n = 10)   # last 10 cells
```

### On-the-Fly Log-Normalization

The OnePZ class supports automatic log-normalization during reads.
The formula is `log1p(x / colsum * scale)`, equivalent to Seurat's `LogNormalize`.

```r
# Open in normalized mode — all reads return log-normalized data
opz <- OnePZ("counts.1pz", normalize = TRUE)
opz[1:100, 1:500]          # log-normalized submatrix
as(opz, "dgCMatrix")       # full log-normalized matrix

# Create a normalized view from an existing handle
opz <- OnePZ("counts.1pz")
npz <- normalized(opz)              # default scale = 10000
npz <- normalized(opz, scale = 1e6) # CPM-style normalization
```

### Type coercion

```r
mat <- as(opz, "dgCMatrix")    # sparse matrix
mat <- as(opz, "matrix")       # dense matrix
df  <- as(opz, "data.frame")   # data.frame

# If Seurat is installed:
obj <- as(opz, "Seurat")

# If SingleCellExperiment is installed:
sce <- as(opz, "SingleCellExperiment")
```

## Core API

### Reading

```r
# Read to dgCMatrix (genes × cells)
mat <- read_1pz("counts.1pz", num_threads = 4)
dim(mat)                    # [1] 30000 10000
rownames(mat)[1:3]          # Gene names
colnames(mat)[1:3]          # Cell barcodes
attr(mat, "colsums")[1:5]   # Column sums
attr(mat, "obs")            # Cell metadata (data.frame)
attr(mat, "var")            # Gene metadata (data.frame)
attr(mat, "uns")            # Study metadata (named vector)

# Read log-normalized (equivalent to Seurat::LogNormalize)
mat_norm <- read_1pz("counts.1pz", normalize = TRUE, scale = 10000)
```

### Writing

```r
# Write from dgCMatrix
write_1pz(mat, "output.1pz",
          rownames = rownames(mat),
          colnames = colnames(mat),
          num_threads = 4, level = 3, chunk_cols = 1024)
```

### Info & Validation

```r
info <- info_1pz("counts.1pz")
info$m        # rows (genes)
info$n        # columns (cells)
info$nnz      # non-zeros
info$has_metadata
info$has_obs_var

result <- validate_1pz("counts.1pz")
result$valid  # TRUE/FALSE
```

## Dataset Operations

### Horizontal concatenation (cbind)

Concatenate cells across multiple files:

```r
cbind_1pz("sample_A.1pz", "sample_B.1pz",
          output = "combined.1pz",
          verify_rownames = TRUE)
```

### Vertical concatenation (rbind)

Concatenate features across files:

```r
rbind_1pz("genes_A.1pz", "genes_B.1pz",
          output = "all_genes.1pz",
          verify_colnames = TRUE)
```

### Subsetting

Write a filtered subset to a new file:

```r
# By integer indices
subset_1pz("counts.1pz", "subset.1pz",
           rows = 1:100, cols = 1:500)

# By name
subset_1pz("counts.1pz", "markers.1pz",
           rows = c("CD3D", "CD3E", "CD4"))

# By logical mask
mask <- info_1pz("counts.1pz")  # example — any logical vector works
subset_1pz("counts.1pz", "subset.1pz",
           cols = c(rep(TRUE, 100), rep(FALSE, 900)))
```

### Random sampling

```r
# Sample 1000 cells
sample_1pz("counts.1pz", "sample.1pz", n = 1000, seed = 42)

# Sample 10% of cells
sample_1pz("counts.1pz", "sample.1pz", fraction = 0.1, seed = 42)
```

## Summary Statistics

```r
# Non-zeros per column (cell)
npc <- nnz_per_col("counts.1pz")

# Non-zeros per row (feature)
npr <- nnz_per_row("counts.1pz")

# Row sums
rs <- rowsums_1pz("counts.1pz")

# Quick file summary
desc <- describe_1pz("counts.1pz")
desc$shape    # c(30000, 10000)
desc$nnz      # 15000000
desc$density  # 0.05
```

## Seurat

### Read .1pz → Seurat Object

```r
library(Seurat)
obj <- read_1pz_seurat("counts.1pz", assay_name = "RNA")

# Cell metadata is merged into obj@meta.data
head(obj@meta.data)

# Total counts from stored colsums
head(obj$total_counts)

# Study metadata stored in Misc
Seurat::Misc(obj)
```

### Write Seurat → .1pz

```r
write_1pz_seurat(obj, "output.1pz", assay_name = "RNA")
```

## SingleCellExperiment

### Read .1pz → SCE

```r
library(SingleCellExperiment)
sce <- read_1pz_sce("counts.1pz")

# Count matrix
counts(sce)[1:5, 1:5]

# Cell metadata
colData(sce)

# Gene metadata
rowData(sce)

# Study metadata
metadata(sce)
```

### Write SCE → .1pz

```r
write_1pz_sce(sce, "output.1pz", assay_name = "counts")
```

## Sparse Format Conversions

```r
# COO (triplet) format
mat_coo <- read_1pz_coo("counts.1pz")  # dgTMatrix

# Dense matrix (warning: memory intensive)
mat_dense <- read_1pz_dense("counts.1pz")

# Data frame
df <- read_1pz_dataframe("counts.1pz")
```

## Log-Normalization

```r
# Using stored column sums
mat <- read_1pz("counts.1pz")
cs <- attr(mat, "colsums")
norm <- lognorm(mat, cs, scale = 10000)
```

## Cross-Language Compatibility

Files written by Python singlepress are directly readable from R, and vice versa:

```python
# Python: write
import singlepress as sp
sp.write_1pz("shared.1pz", mat, rownames=genes, colnames=cells)
```

```r
# R: read the same file
mat <- read_1pz("shared.1pz")
```

Both implementations:
- Use the same header/footer structure (96 + 16 bytes)
- Apply identical VOCSC encoding with the same byte-split and zstd-3 pipeline
- Compute identical CRC32 checksums
- Serialize metadata in the same TLV format
