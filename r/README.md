# singlify (R)

An R reader for [singlify](https://github.com/zdebruine/Singlet-AI) pipeline outputs.

Reads the `.1pz` sparse-matrix files plus the per-cell sidecar TSVs from a singlify pipeline output directory and converts them into:

- `Matrix::dgCMatrix` (low-level reader)
- `SingleCellExperiment` with `altExp` for per-feature matrices
- `Seurat` object with `spliced` / `unspliced` / `ambiguous` assays attached for scvelo-compatible velocity analysis

All decompression and format parsing happens in a **header-only C++ reader** that is byte-identical to the Python wrapper. R is a thin Rcpp binding.

## Install

From GitHub:

```r
install.packages("remotes")
remotes::install_github("zdebruine/Singlet-AI", subdir = "singlify/r")
```

Requires a C++17 compiler and `libzstd` (>= 1.4) available at link time.

## Quickstart

```r
library(singlify)

# Low-level: one file → dgCMatrix
mat <- read_1pz("quant/scrna/GSE174/GSE174399/GSM5293863/gene_counts.1pz")
dim(mat)                             # 38606 × 16079
attr(mat, "user_kv")[["gsm_id"]]     # "GSM5293863"

# Mid-level: a whole pipeline directory → named list of matrices
dd <- read_singlify_dir("quant/scrna/GSE174/GSE174399/GSM5293863")
print(dd)

# High-level: drop into Bioconductor
library(SingleCellExperiment)
sce <- as_sce("quant/scrna/GSE174/GSE174399/GSM5293863")
sce
metadata(sce)$singlify[["gsm_id"]]

# High-level: drop into Seurat (scvelo-ready)
library(Seurat)
obj <- as_seurat("quant/scrna/GSE174/GSE174399/GSM5293863")
obj
obj@misc$singlify[["gsm_id"]]
```

## What's in the embedded metadata

Every `.1pz` file carries the GEO context the pipeline was launched with:

- `gsm_id`, `gse_id`, `srr_ids`
- `organism`, `protocol`
- `singlify_version`, `pipeline_date`
- `read_count`

`read_singlify_dir()` exposes this as `attr(dd, "user_kv")`. `as_sce` puts it in `metadata(sce)$singlify`. `as_seurat` puts it in `obj@misc$singlify`.
