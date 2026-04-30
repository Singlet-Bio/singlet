---
feature: r_wrapper_foundation
roadmap_id: 24
module: r/{DESCRIPTION,NAMESPACE,src/Makevars,src/*.cpp,R/*.R,tests/testthat/*.R}
status: design
tolerance: round-trip CSC matrix bit-identical between AnnData (Python) and SingleCellExperiment (R) for the same .1pz file
target_perf: load_1pz_sce wrapper call ≤200ms overhead vs the C++ direct call (Rcpp marshal + Matrix::dgCMatrix construction)
ooc_plan: streaming variant deferred to cycle 25+ (single-sample only in cycle 24)
---

## Cycle 23+ context

Cycle 22 closed with all 17 original kernels having Python wrappers + the binding extension. Cycle 23 (in flight) adds the final Python wrappers for cycles 13-17. Cycle 24 begins the R wrapper sprint.

Per cycle 23 lit-scout: **hybrid Rcpp + reticulate** is the right approach. CRAN forbids native CUDA, so we ship via GitHub-only `remotes::install_github("zdebruine/Singlet-AI", subdir = "singlet-gpu/r")`.

## Architecture

```
singlet-gpu/r/
├── DESCRIPTION                          # Bioconductor + GitHub package metadata
├── NAMESPACE                            # roxygen2-generated
├── README.md
├── R/
│   ├── zzz.R                            # .onLoad — link to compiled C++
│   ├── load_pz.R                        # read_1pz_sce()
│   ├── preprocess.R                     # lognorm(), hvg()
│   ├── reduce.R                         # pca(), nmf()
│   ├── reticulate_bridge.R              # rapids_harmony(), rapids_umap() etc.
│   └── utils.R                          # internal helpers
├── src/
│   ├── Makevars                         # CXX17 + CUDA toolchain flags
│   ├── Makevars.win                     # Windows variant
│   ├── singlet_gpu_r_bindings.cpp       # Rcpp entry — calls into C++20 header
│   ├── _bind_load.hpp                   # cycle-2 loader R binding
│   ├── _bind_preprocess.hpp             # cycle-3 + cycle-4 R bindings
│   ├── _bind_reduce.hpp                 # cycle-4 + cycle-5 R bindings
│   └── _r_to_eigen.hpp                  # Matrix::dgCMatrix → C++ DeviceCSC marshalling
├── inst/
│   └── extdata/
│       └── tiny_test.1pz                # bundled tiny test sample
├── man/                                 # roxygen2 manpages (auto-gen)
├── tests/
│   └── testthat/
│       ├── test-load.R
│       ├── test-preprocess.R
│       └── test-reduce.R
└── vignettes/
    └── singlet_gpu_intro.Rmd
```

## API surface (cycle 24 minimal — to expand in cycle 25+)

```r
# r/R/load_pz.R
read_1pz_sce <- function(pz_dir, modality = "exon", keep_host_pinned = FALSE) {
  # Calls into C++ via Rcpp:
  #   singlet_gpu_r::load_pz_to_sce_cpp(pz_dir, modality, keep_host_pinned)
  # Returns: SingleCellExperiment with assays(sce)$counts as Matrix::dgCMatrix
  # Embeds GEO metadata into metadata(sce)$singlify
}

# r/R/preprocess.R
lognorm <- function(sce, target_sum = NULL, size_factors = "median", layer = "counts") {
  # Calls singlet_gpu_r::lognorm_cpp(sce@assays$counts, target_sum, size_factors)
  # Returns: SCE with assays(sce)$logcounts populated
}

hvg <- function(sce, n_top = 2000, flavor = c("seurat_v3", "pearson_residuals"), layer = "counts") {
  # Calls singlet_gpu_r::hvg_cpp(...)
  # Returns: SCE with rowData(sce)$highly_variable, rowData(sce)$variances_norm
}

# r/R/reduce.R
run_pca <- function(sce, n_pcs = 50, layer = "logcounts", svd_solver = "auto", seed = 0) {
  # Calls singlet_gpu_r::pca_cpp(...)
  # Returns: SCE with reducedDim(sce, "PCA"), metadata(sce)$pca
}

run_nmf <- function(sce, n_factors = 20, loss = "MSE", layer = "logcounts", seed = 0) {
  # Calls singlet_gpu_r::nmf_cpp(...)
  # Returns: SCE with reducedDim(sce, "NMF"), rowData(sce)$NMF_loadings
}

# r/R/reticulate_bridge.R
rapids_harmony <- function(sce, batch_key, ...) {
  # OPTIONAL: only if reticulate + rapids-singlecell installed
  # Falls back to error message: "install reticulate + singlet-gpu Python wheel"
}
```

## Marshalling: Matrix::dgCMatrix ↔ C++ DeviceCSC

R's canonical sparse type is `Matrix::dgCMatrix` (CSC, 0-indexed `i`, `p`, `x` slots). The C++ side wants pinned-host CSC. Marshalling:

```cpp
// _r_to_eigen.hpp (~150 LOC)
namespace singlet_gpu_r {

// dgCMatrix → singlet_gpu::core::HostCsc (pinned)
core::HostCsc r_sparse_to_host_csc(const Rcpp::S4& dgc) {
    Rcpp::IntegerVector i = dgc.slot("i");
    Rcpp::IntegerVector p = dgc.slot("p");
    Rcpp::NumericVector x = dgc.slot("x");
    Rcpp::IntegerVector dim = dgc.slot("Dim");
    // Allocate pinned host CSC and copy.
    // The pinned buffer is then handed to load_pz / kernel as-is.
}

// HostCsc → dgCMatrix (for results that need to leave GPU)
Rcpp::S4 host_csc_to_r_sparse(const core::HostCsc& csc, int n_rows, int n_cols);

}  // namespace
```

The cost: O(nnz) host copy on input (Matrix::dgCMatrix → pinned host CSC) and same on output. Document that R does NOT have native GPU sparse — every kernel call has a host round-trip.

## Constraints

- **CRAN-incompatible by design** — distribute via GitHub only.
- Native CUDA: `Makevars` invokes `nvcc` for any `.cu` files (via `R CMD SHLIB`). Document the build requirement: `R >= 4.3 + CUDA Toolkit >= 12.0`.
- The R wrapper takes the host-copy cost as a documented limitation. R has no GPU sparse type, so this is unavoidable.
- For users who want full GPU residency: use the Python wrapper.
- reticulate bridge is OPT-IN — `if (!requireNamespace("reticulate", quietly=TRUE)) stop("install reticulate to use this function")`.

## Test spec

testthat suite under `r/tests/testthat/`:

1. **`test-load.R`** — load `inst/extdata/tiny_test.1pz`, confirm SCE has expected dims + GEO metadata in `metadata(sce)$singlify`.
2. **`test-preprocess.R`** — run `lognorm` and `hvg` on tiny SCE, confirm `assays(sce)$logcounts` and `rowData(sce)$highly_variable` populated.
3. **`test-reduce.R`** — run `run_pca` on tiny SCE, confirm `reducedDim(sce, "PCA")` populated.
4. **`test-reticulate-bridge.R`** — `expect_error(rapids_harmony(...))` if reticulate unavailable; smoke test if available.

Tests run via `R CMD check r/` (or `devtools::test()`).

## Distribution

- `remotes::install_github("zdebruine/Singlet-AI", subdir = "singlet-gpu/r")`
- Bioconductor submission deferred — Bioconductor allows native CUDA but requires extensive documentation + maintenance.
- `inst/extdata/tiny_test.1pz` bundled (≤1 MB) so `R CMD check` runs offline.

## Implementation notes

- Use roxygen2 for documentation generation.
- Use testthat for tests.
- `DESCRIPTION` lists `Imports: SingleCellExperiment, Matrix, Rcpp` and `LinkingTo: Rcpp`.
- `SystemRequirements: CUDA Toolkit (>= 12.0), GPU with compute capability >= 7.0`.
- `LICENSE: GPL-2` (matches singlet-gpu's GPL-2.0-or-later).

## Risks

1. **CUDA build flags in Makevars** are fragile across distributions (Ubuntu, Fedora, RHEL, macOS-without-CUDA). Provide `Makevars.win` for Windows fallback (or fail gracefully).
2. **anndataR vs native marshalling**: anndataR is the modern path — could simplify our marshalling. But it adds a dependency. Decision: native marshalling for cycle 24, anndataR as a future cycle.
3. **Seurat v5 BPCells format**: Seurat v5 uses BPCells for on-disk sparse matrices. Our `dgCMatrix` returns require an `as.Seurat()` step. Document.
4. **Reticulate version pinning**: rapids-singlecell pins to specific cupy/cuml versions. Reticulate bridge tests must skip if versions don't match.
