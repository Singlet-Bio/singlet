---
feature: r_wrappers_part_2
roadmap_id: 25
module: r/{src/_bind_graph.hpp,_bind_embed.hpp,_bind_de.hpp,_bind_anno.hpp,_bind_streaming.hpp} + r/R/{neighbors,leiden,umap,rank_genes_groups,markers,streaming}.R
status: design
tolerance: per-function output rel_err ≤ 1e-5 vs C++ direct call (modulo R sparse host-copy)
target_perf: Rcpp marshal + dgCMatrix host copy ≤500ms overhead at 100k cells
ooc_plan: streaming wrapper passes a vector of paths through to the C++ streaming pipeline
---

## Cycle 25 scope

R wrappers for cycles 7-12 kernels:
- Cycle 7 streaming → `singletGpu::run_pipeline()`
- Cycle 8 kNN → `singletGpu::neighbors()` (writes to `metadata(sce)$neighbors` since SCE doesn't have a native obsp)
- Cycle 9 leiden → `singletGpu::leiden()`
- Cycle 10 UMAP → `singletGpu::run_umap()`
- Cycle 11 DE → `singletGpu::rank_genes_groups()`
- Cycle 12 annotation → `singletGpu::score_genes()` + `singletGpu::celltypist_predict()`

Extends cycle 24's R package with 6 new `_bind_*.hpp` C++ files and 6 new R wrapper files.

## R API surface

```r
# r/R/neighbors.R
neighbors <- function(sce,
                       n_neighbors = 15,
                       use_dimred = "PCA",
                       k_within = NULL,
                       metric = "euclidean",
                       seed = 0) {
  # Calls singlet_gpu_r::neighbors_cpp(reducedDim(sce, use_dimred), n_neighbors, ...)
  # Stores: metadata(sce)$neighbors$indices, $distances, $connectivities
  # Returns the modified sce.
}

# r/R/leiden.R
leiden <- function(sce,
                    resolution = 1.0,
                    seed = 0,
                    weight_function = "Connectivity",
                    column_name = "leiden") {
  # Reads metadata(sce)$neighbors built by neighbors().
  # Calls singlet_gpu_r::leiden_cpp(...).
  # Stores: colData(sce)[[column_name]] as factor.
}

# r/R/umap.R
run_umap <- function(sce,
                      n_components = 2,
                      min_dist = 0.5,
                      spread = 1.0,
                      init = "random",
                      seed = 0,
                      use_neighbors = TRUE) {
  # Reads metadata(sce)$neighbors. Calls singlet_gpu_r::umap_cpp(...).
  # Stores: reducedDim(sce, "UMAP")
}

# r/R/rank_genes_groups.R
rank_genes_groups <- function(sce,
                                groupby,
                                method = c("wilcoxon", "t-test"),
                                n_genes = 100,
                                layer = "logcounts") {
  # Calls singlet_gpu_r::wilcoxon_de_cpp(...) or ttest_de_cpp(...).
  # Returns: a DataFrame with per-gene per-cluster results (also stored in metadata(sce)$rank_genes_groups).
}

# r/R/markers.R
score_genes <- function(sce,
                          gene_sets,            # list of character vectors keyed by set name
                          method = c("mlm", "ulm", "wsum", "ucell"),
                          score_name = "score") {
  # Converts gene_sets list to a GeneSetDB struct in C++.
  # Stores: colData(sce)[[<set>_score]] for each set.
}

celltypist_predict <- function(sce,
                                model_path,
                                use_dimred = "PCA",
                                key_added = "celltypist") {
  # Loads the .npz model file (the same one Python uses).
  # Calls singlet_gpu_r::celltypist_project_cpp(reducedDim(sce, use_dimred), model_path).
  # Stores: colData(sce)[[key_added]] as factor.
}

# r/R/streaming.R
run_pipeline <- function(input_paths,
                          chunk_cols = 100000L,
                          run_lognorm = TRUE,
                          run_hvg = TRUE,
                          run_pca = FALSE,
                          pca_k = 50L,
                          run_nmf = FALSE,
                          nmf_factors = 20L) {
  # Calls singlet_gpu_r::streaming_pipeline_run_cpp(input_paths, ...).
  # Returns a list with size_factors, hvg_indices, pca_pcs, nmf_W/H, etc.
  # NOT an SCE because the input is multi-file and may exceed in-memory capacity.
}
```

## Bindings in C++

Each cycle 25 R binding is a `RcppExport` C function in a new `_bind_*.hpp` that:
1. Takes Rcpp R types (`S4`, `List`, `IntegerVector`, etc.).
2. Marshals to the singlet-gpu C++ API via the cycle 24 `_r_to_eigen.hpp` helpers + new helpers as needed.
3. Calls into the singlet-gpu header library (cycles 7-12 kernels).
4. Marshals results back to R.

Files:
- `r/src/_bind_graph.hpp` (~250 LOC) — `neighbors_cpp`, `leiden_cpp`
- `r/src/_bind_embed.hpp` (~150 LOC) — `umap_cpp`
- `r/src/_bind_de.hpp` (~250 LOC) — `wilcoxon_de_cpp`, `ttest_de_cpp`
- `r/src/_bind_anno.hpp` (~250 LOC) — `marker_score_cpp`, `celltypist_project_cpp`, `load_celltypist_model_cpp`
- `r/src/_bind_streaming.hpp` (~200 LOC) — `streaming_pipeline_run_cpp`

Plus extending `r/src/singlet_gpu_r_bindings.cpp` (+~200 LOC) to register the new functions.

## Constraints

- Same as cycle 24: native Rcpp + GPL-2.0-or-later. The cycle 24 code-reader's RcppML correction (drop reticulate, gpu_stubs, sparse_from_csc) will be applied in a follow-up cycle (cycle 26.5 or 27).
- All R-side functions follow the SCE-modify-and-return pattern.
- Result locations match scanpy conventions where applicable: leiden → `colData`, umap → `reducedDim("UMAP")`, neighbors → `metadata$neighbors` (since SCE has no native obsp).
- For `run_pipeline`: the result is a plain list, NOT an SCE, because the input is multi-file. Document in roxygen.

## Test spec

Extend the cycle 24 testthat suite with 6 new files: `test-neighbors.R`, `test-leiden.R`, `test-umap.R`, `test-rank-genes-groups.R`, `test-markers.R`, `test-streaming.R`.

Each ~120 LOC, ~3-5 cases. Reuse cycle 24 fixtures.

## Risks

1. **Streaming pipeline R wrapper has no SCE return** — it's a plain list. Document loudly to avoid user confusion.
2. **`metadata(sce)$neighbors` is not a standard SCE slot** — scran uses `colPairs(sce)` for kNN graphs. Consider that as an alternative; for cycle 25 we use `metadata` and document the future migration.
3. **CelltypistModel .npz loading** in R is non-trivial — reuse the cycle 12 NPZ reader from `_bind_kernels.hpp` (which would need to be moved into a shared header).
