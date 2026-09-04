# singlet (R)

An R client for the [Singlet](https://singlet.bio) single-cell RNA-seq atlas —
search and download processed public studies as a `SingleCellExperiment` or
`Seurat` object — plus an R package for reading and writing `.1pz`
sparse-matrix files and working with singlet pipeline outputs directly.

All decompression and format parsing happens in a **header-only C++ reader**
(namespace `singlet::pz`) that is byte-identical to the Python package. R is a
thin Rcpp binding.

## Install

From GitHub:

```r
install.packages("remotes")
remotes::install_github("Singlet-Bio/singlet", subdir = "r")
```

Requires a C++17 compiler and `libzstd` (>= 1.4) available at link time.

`load()` returns a `SingleCellExperiment` by default (or a `Seurat` object
with `as = "seurat"`), so also install whichever of those two you plan to
use — they're Bioconductor/CRAN packages, not automatically pulled in by the
`remotes::install_github()` call above:

```r
# Needed for the default `load()` / `read_singlet()` output (SingleCellExperiment)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("SingleCellExperiment", "SummarizedExperiment", "S4Vectors"))

# Only needed if you pass as = "seurat" to load()
install.packages("Seurat")
```

## Quickstart: browse and load the atlas

```r
library(singlet)

# Search the atlas with natural language -> GSE accessions
hits <- find("T cells from pediatric AML")

# Load one or more studies (downloaded + cached automatically) -> SingleCellExperiment
sce <- load("GSE149298")
sce <- load(hits[1:3])                    # combine several studies into one object
obj <- load("GSE149298", as = "seurat")   # ...or as a Seurat object

# Search and load in one step
sce <- find_load("human pancreas islet cells")

# A local .singlet bundle works the same way as an accession
sce <- load("/path/to/data.singlet")
```

## Reading pipeline output directly

For local `.1pz` files and pipeline output directories (rather than atlas
accessions):

```r
library(singlet)

# Low-level: one file → dgCMatrix
mat <- read_1pz("gene_counts.1pz")
dim(mat)                             # 38606 × 16079
attr(mat, "user_kv")[["gsm_id"]]     # "GSM5293863"

# Mid-level: a whole pipeline directory → named list of matrices
dd <- read_singlet_dir("quant/scrna/GSE174/GSE174399/GSM5293863")
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
