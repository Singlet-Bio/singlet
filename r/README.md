# singlet (R)

An R package for reading and writing `.1pz` sparse-matrix files and working with singlet pipeline outputs.

Provides:

- `Matrix::dgCMatrix` (low-level reader)
- `SingleCellExperiment` with `altExp` for per-feature matrices
- `Seurat` object with `spliced` / `unspliced` / `ambiguous` assays for velocity analysis

All decompression and format parsing happens in a **header-only C++ reader** (namespace `singlet::pz`) that is byte-identical to the Python package. R is a thin Rcpp binding.

## Install

From GitHub:

```r
install.packages("remotes")
remotes::install_github("Singlet-Bio/singlet", subdir = "r")
```

Requires a C++17 compiler and `libzstd` (>= 1.4) available at link time.

## Quickstart

```r
library(singlet)

# Low-level: one file → dgCMatrix
mat <- read_1pz("gene_counts.1pz")
dim(mat)                             # 38606 × 16079
attr(mat, "user_kv")[["gsm_id"]]     # "GSM5293863"

# Mid-level: a whole pipeline directory → named list of matrices
dd <- read_singlify_dir("quant/scrna/GSE174/GSE174399/GSM5293863")
print(dd)

# High-level: drop into Bioconductor
library(SingleCellExperiment)
sce <- as_sce("quant/scrna/GSE174/GSE174399/GSM5293863")
sce

# High-level: drop into Seurat (scvelo-ready)
library(Seurat)
obj <- as_seurat("quant/scrna/GSE174/GSE174399/GSM5293863")
obj
```

## GPU Analysis (optional)

If CUDA 12+ is available:

```r
if (singlet::has_gpu()) {
    sce <- singlet::gpu_pca(sce, n = 50)
    sce <- singlet::gpu_neighbors(sce, k = 15)
    sce <- singlet::gpu_leiden(sce)
}
```

## Embedded Metadata

Every `.1pz` file carries the GEO context from the pipeline:

- `gsm_id`, `gse_id`, `srr_ids`
- `organism`, `protocol`
- `singlet_version`, `pipeline_date`
- `read_count`

## License

MIT
