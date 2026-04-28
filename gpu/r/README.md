# singletGpu — GPU-native single-cell analysis in R

R interface to the [singlet-gpu](https://github.com/zdebruine/Singlet-AI)
header-only C++20/CUDA library.  Reads singlify `.1pz` count matrices and runs
GPU-native normalization, HVG selection, PCA, and NMF via
[factornet](https://github.com/zdebruine/factornet) GPU backends.  Results
are returned as `SingleCellExperiment` objects.

---

## Requirements

| Requirement | Minimum version |
|---|---|
| R | 4.3.0 |
| CUDA Toolkit | 12.0 |
| GPU compute capability | 7.0 (Volta / Turing / Ampere / Hopper) |
| C++ compiler | GCC ≥ 11 or Clang ≥ 14 (C++20) |
| Rcpp | 1.0.11 |
| SingleCellExperiment | any Bioconductor ≥ 3.18 |

The package is **GitHub-only** — CRAN distribution is not planned because
CRAN does not accept packages requiring CUDA toolchains.

---

## Installation

### 1. Install Bioconductor dependencies

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("SingleCellExperiment", "SummarizedExperiment"))
```

### 2. Install singletGpu

```r
if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes")

remotes::install_github(
    "zdebruine/Singlet-AI",
    subdir   = "singlet-gpu/r",
    ref      = "main",
    build    = TRUE,
    build_opts = c("--no-resave-data", "--no-manual")
)
```

The install step invokes `nvcc` via `R CMD SHLIB` to compile the Rcpp binding.
Make sure `nvcc` is on your `PATH` (e.g. `export PATH=/usr/local/cuda/bin:$PATH`)
before calling `remotes::install_github`.

The `SINGLET_GPU_INCLUDE` environment variable must point to the
`singlet-gpu/include` directory, and `FACTORNET_INCLUDE` must point to
`factornet/include`:

```bash
export SINGLET_GPU_INCLUDE=/path/to/Singlet-AI/singlet-gpu/include
export FACTORNET_INCLUDE=/path/to/factornet/include
```

---

## Usage

```r
library(singletGpu)

# Load a .1pz count matrix and construct a SingleCellExperiment
sce <- read_1pz_sce(
    pz_dir  = "/path/to/GSM4037629/",
    modality = "exon"           # "exon" | "intron" | "gene" | "adt" | "atac"
)

# Log-normalise (result stored in assays(sce)$logcounts)
sce <- lognorm(sce, target_sum = 1e4)

# Select 2000 highly variable genes
sce <- hvg(sce, n_top = 2000, flavor = "seurat_v3")

# PCA (50 components, factornet auto-select SVD backend)
sce <- run_pca(sce, n_pcs = 50)

# NMF (20 factors, MSE loss)
sce <- run_nmf(sce, n_factors = 20, loss = "MSE")

# Access results
head(SingleCellExperiment::reducedDim(sce, "PCA"))
```

### Reticulate bridge (optional)

If `reticulate` and the `singlet_gpu` Python wheel are installed in the active
Python environment, Harmony batch correction and UMAP are available:

```r
sce <- rapids_harmony(sce, batch_key = "sample_id")
sce <- rapids_umap(sce, basis = "X_pca_harmony")
```

Both functions throw an informative error if the Python dependencies are absent.

---

## Architecture

- **Native Rcpp path** (all functions above): calls into the singlet-gpu
  header-only C++ library directly.  No Python required.
- **Reticulate bridge** (`rapids_harmony`, `rapids_umap`): marshals the SCE
  to AnnData via reticulate and calls the Python wrapper; requires the
  `singlet_gpu` Python wheel.

**Host-copy cost**: R does not have a native GPU sparse type.  Every kernel
call incurs an O(nnz) host copy on input (dgCMatrix → pinned host CSC) and
output.  For most scRNA-seq samples this is < 200 ms.  Users who need full
GPU residency between operations should use the Python wrapper instead.

---

## License

GPL-2 (inherits from factornet, which is GPL-2.0).
