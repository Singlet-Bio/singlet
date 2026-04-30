---
feature: r_wrappers_part_3
roadmap_id: 26
module: r/{src/_bind_gsea,_bind_integrate,_bind_velocity,_bind_lineage,_bind_pseudobulk}.hpp + r/R/{enrichment,integrate,velocity,lineage,pseudobulk}.R
status: design
tolerance: per-function output rel_err ≤ 1e-5 vs C++ direct call
target_perf: Rcpp marshal overhead ≤500ms per function
ooc_plan: not applicable for these high-level wrappers
---

## Cycle 26 scope — FINAL wrapper cycle

R wrappers for cycles 13-17 kernels:
- Cycle 13 GSEA + AUCell → `singletGpu::run_gsea()` + `singletGpu::run_aucell()` matching `fgsea::fgsea` + `AUCell::AUCell_run` signatures
- Cycle 14 Harmony + BBKNN → `singletGpu::run_harmony()` + `singletGpu::run_bbknn()` matching `harmony::RunHarmony` + `bbknnR::bbknn`
- Cycle 15 Velocity → `singletGpu::velocity_prep()` analog of `velociraptor::scvelo`
- Cycle 16 MT lineage → `singletGpu::detect_clones()` (NEW — closest is MitoTrace)
- Cycle 17 Donor pseudobulk DE → `singletGpu::pseudobulk_de()` matching `muscat::aggregateData` + `muscat::pbDS`

After cycle 26, ALL 17 original kernels have BOTH Python and R wrappers. Phase A (wrappers) is COMPLETE. Cycle 27+ begins Phase B (new features: Cell2fate, MultiVI, spatial GCN).

## R API surface

```r
# r/R/enrichment.R
run_gsea <- function(stats,                  # named numeric vector of per-gene rank statistics
                       pathways,             # named list of character vectors (gene set members)
                       eps = 0.0,
                       min_size = 15,
                       max_size = 500,
                       seed = 42,
                       n_perm = 1000) {
  # Calls singlet_gpu_r::fgsea_cpp(stats, pathways, eps, min_size, max_size, n_perm, seed)
  # Returns: data.frame with columns: pathway, ES, NES, pval, padj, leadingEdge
  # Drop-in for fgsea::fgsea (modulo eps semantics)
}

run_aucell <- function(sce,                   # SingleCellExperiment
                         gene_sets,           # named list of character vectors
                         layer = "counts",
                         aucMaxRank = NULL,   # default: ceil(0.05 * n_genes)
                         normAUC = TRUE) {
  # Calls singlet_gpu_r::aucell_cpp(...)
  # Stores: colData(sce)[, paste0("AUC_", names(gene_sets))] (one column per set)
  # Drop-in for AUCell::AUCell_run
}

# r/R/integrate.R
run_harmony <- function(sce,
                          batch_key,         # column in colData(sce)
                          use_dimred = "PCA",
                          adjusted_dimred = "Harmony",
                          n_clusters = 20L,
                          max_iter = 10L,
                          tol = 1e-4,
                          seed = 0L) {
  # Calls singlet_gpu_r::harmony_cpp(reducedDim(sce, use_dimred), colData(sce)[[batch_key]], ...)
  # Stores: reducedDim(sce, adjusted_dimred)
  # Mirrors harmony::RunHarmony API
}

run_bbknn <- function(sce,
                        batch_key,
                        use_dimred = "PCA",
                        neighbors_within_batch = 3L,
                        n_pcs = NULL,
                        metric = "euclidean") {
  # Calls singlet_gpu_r::bbknn_cpp(...)
  # Stores: metadata(sce)$neighbors$indices, $distances, $connectivities
  # Mirrors bbknnR::bbknn for Seurat
}

# r/R/velocity.R
velocity_moments <- function(sce,
                                n_neighbors = 30L,
                                use_dimred = "PCA",
                                layer_spliced = "spliced",
                                layer_unspliced = "unspliced") {
  # Calls singlet_gpu_r::velocity_moments_cpp(...)
  # Stores: assays(sce)$Ms, assays(sce)$Mu (smoothed first-order moments)
}

velocity_prep <- function(sce,
                            mode = "steady_state",
                            min_S_count = 10L,
                            min_U_count = 5L,
                            top_n_quantile = 5L,
                            compute_velocity = TRUE) {
  # Calls singlet_gpu_r::velocity_prep_compute_cpp(...)
  # Stores: assays(sce)$velocity, rowData(sce)$velocity_gamma, rowData(sce)$velocity_filter
}

# r/R/lineage.R
detect_clones <- function(sce,
                            alt_assay = "mt_alt",
                            depth_assay = "mt_depth",
                            min_depth = 10L,
                            min_cells_alt = 5L,
                            min_vaf = 0.01,
                            min_K = 2L,
                            max_K = 10L,
                            seed = 0L) {
  # Calls singlet_gpu_r::mt_lineage_call_clones_cpp(...)
  # Stores: colData(sce)$mt_clone_id, metadata(sce)$mt_lineage$informative_sites,
  #         altExp(sce, "mt_heteroplasmy") (or reducedDim)
  # NEW — closest CPU analog is MitoTrace
}

# r/R/pseudobulk.R
aggregate_pseudobulk <- function(sce,
                                   group_by,              # cluster column
                                   sample_by,             # donor column
                                   layer = "counts",
                                   min_cells_per_pseudobulk = 10L) {
  # Calls singlet_gpu_r::pseudobulk_aggregate_cpp(...)
  # Returns: a NEW SingleCellExperiment with cols = (cluster, donor) groups
  #          and assays(pb)$counts as the summed pseudobulk matrix.
  # Mirrors muscat::aggregateData(fun="sum")
}

pseudobulk_de <- function(sce,
                            group_by,               # cluster column
                            sample_by,              # donor column
                            layer = "counts",
                            min_cells_per_pseudobulk = 10L,
                            apeglm_shrinkage = TRUE,
                            seed = 0L) {
  # Wraps aggregate_pseudobulk + singlet_gpu_r::donor_pseudobulk_de_cpp(...)
  # Returns: list of data.frames (one per cluster) with columns: gene, log2FC, pvalue, padj, dispersion
  # Mirrors muscat::pbDS(method="DESeq2") output shape
}
```

## C++ Rcpp bindings (~1100 LOC across 5 files)

- `r/src/_bind_gsea.hpp` (~250 LOC) — fgsea_cpp, aucell_cpp
- `r/src/_bind_integrate.hpp` (~250 LOC) — harmony_cpp, bbknn_cpp
- `r/src/_bind_velocity.hpp` (~200 LOC) — velocity_moments_cpp, velocity_prep_compute_cpp
- `r/src/_bind_lineage.hpp` (~180 LOC) — mt_lineage_call_clones_cpp
- `r/src/_bind_pseudobulk.hpp` (~250 LOC) — pseudobulk_aggregate_cpp, donor_pseudobulk_de_cpp

Plus modify `r/src/singlet_gpu_r_bindings.cpp` (+~150 LOC) to register them.

## Constraints

- Native Rcpp; no reticulate.
- API mirrors the canonical Bioconductor packages exactly per the cycle 25 lit-scout.
- Result locations match the canonical conventions: harmony → `reducedDim`, gsea → data.frame return, aucell → `colData`, velocity → `assays + rowData`, mt_lineage → `colData + metadata`, pseudobulk → list-of-data.frames.

## Test spec

5 new test files in `r/tests/testthat/` matching the cycle 25 testthat pattern. Each ~120 LOC.

After cycle 26, the wrapper sprint is COMPLETE and the post-roadmap-plan transitions to Phase B (Cell2fate, MultiVI, spatial GCN, etc.).

## Risks

1. **velocity uses splice/unsplice from singlify directly** — the R wrapper assumes the SCE has `assays(sce)$spliced` and `assays(sce)$unspliced`. Document and provide a `read_pz_velocity_sce()` helper.
2. **mt_lineage requires alt + depth matrices** — same concern. Provide `read_pz_mt_sce()` helper.
3. **muscat::pbDS output shape is complex** — list of data.frames keyed by cluster. Match it loosely; full parity may need a follow-up.
4. **MitoTrace is barely-active** as a comparison reference — testthat tests should `skip_if_not_installed("MitoTrace")` and accept a smoke test fallback.
