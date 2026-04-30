---
feature: cycle_18_binding_extension_2
roadmap_id: 22
module: extend python/src/_bind_kernels.hpp + add new result classes
status: design
tolerance: each binding round-trip element-wise bit-identical to the C++ direct call
target_perf: ≤1µs marshal overhead per binding call
ooc_plan: not applicable (host-side bindings only)
---

## Why this exists

Cycle 21 finished writing 6 Python wrappers for cycles 7-12 kernels and surfaced 7 missing bindings:
- `streaming_pipeline_run` + `StreamingPipelineConfig` (cycle 7)
- `knn_graph`, `knn_graph_from_indices` (cycle 8)
- `leiden_partition` (cycle 9)
- `umap_embed` (cycle 10)
- `wilcoxon_de`, `ttest_de` (cycle 11)
- `marker_score`, `celltypist_project` (cycle 12)

Cycle 23 will write Python wrappers for cycles 13-17 kernels and will need ~10 more bindings:
- `fgsea`, `aucell` (cycle 13)
- `harmony`, `bbknn` (cycle 14)
- `velocity_prep_compute` (cycle 15)
- `mt_lineage_call_clones` (cycle 16)
- `donor_pseudobulk_de` (cycle 17)

This cycle exposes ALL 17 bindings + ~13 result classes in one pass, unblocking both cycle 21 and the cycle 23 wrapper sprint.

## Architecture

Extend `python/src/_bind_kernels.hpp` (cycle 20) with the new functions. Add a new helper `python/src/_bind_results.hpp` (~400 LOC) for the result classes (cycle 20 had 4; cycle 22 adds ~13 more).

### New bindings (per cycle)

#### Cycle 7 — streaming_pipeline_run
```cpp
m.def("streaming_pipeline_run", &py_streaming_pipeline_run,
      py::arg("input_paths"), py::kw_only(),
      py::arg("chunk_cols") = 100000,
      py::arg("run_lognorm") = true,
      py::arg("run_hvg") = true,
      py::arg("run_pca") = false, py::arg("pca_k") = 50,
      py::arg("run_nmf") = false, py::arg("nmf_factors") = 20,
      py::arg("cache_normalized") = false,
      py::arg("in_memory_pca_threshold") = 2000000);
```
Returns: `StreamingPipelineResult` with the cycle 7 fields.

#### Cycle 8 — knn_graph
```cpp
m.def("knn_graph", &py_knn_graph,
      py::arg("embedding"), py::arg("k") = 15, py::kw_only(),
      py::arg("backend") = std::string("auto"),
      py::arg("metric") = std::string("L2"),
      py::arg("return_squared") = false,
      py::arg("seed") = uint64_t{0},
      py::arg("hnsw_M") = 16, py::arg("hnsw_ef") = 64);
```
Returns: `KnnResult` with `row_offsets`, `neighbors`, `distances`.

#### Cycle 9 — leiden_partition
```cpp
m.def("leiden_partition", &py_leiden_partition,
      py::arg("knn_result"), py::arg("resolution") = 1.0f, py::kw_only(),
      py::arg("max_iter") = 100,
      py::arg("seed") = uint64_t{0},
      py::arg("backend") = std::string("CuGraph"),
      py::arg("weight_function") = std::string("Connectivity"),
      py::arg("gaussian_sigma") = 0.0f);
m.def("leiden_multi_resolution", &py_leiden_multi_resolution, ...);
```
Returns: `LeidenResult` with `labels`, `n_clusters`, `modularity`, `iterations`.

#### Cycle 10 — umap_embed
```cpp
m.def("umap_embed", &py_umap_embed,
      py::arg("knn_result"), py::kw_only(),
      py::arg("n_components") = 2,
      py::arg("n_epochs") = 0,
      py::arg("min_dist") = 0.5f,
      py::arg("spread") = 1.0f,
      py::arg("learning_rate") = 1.0f,
      py::arg("init") = std::string("Random"),
      py::arg("seed") = uint64_t{0},
      py::arg("negative_sample_rate") = 5);
```
Returns: `UmapResult` with the embedding.

#### Cycle 11 — wilcoxon_de + ttest_de
```cpp
m.def("wilcoxon_de", &py_wilcoxon_de,
      py::arg("mat"), py::arg("labels"), py::arg("n_clusters"), py::kw_only(),
      py::arg("n_bins") = 4096,
      py::arg("top_n") = 100,
      py::arg("gene_tile") = 1024,
      py::arg("deterministic") = false);
m.def("ttest_de", &py_ttest_de, /* similar */);
```
Returns: `WilcoxonResult` / `TtestResult` containing `std::vector<ClusterMarkers>`.

#### Cycle 12 — marker_score + celltypist_project
```cpp
m.def("marker_score", &py_marker_score,
      py::arg("mat"), py::arg("gene_sets"), py::kw_only(),
      py::arg("method") = std::string("Mlm"),
      py::arg("min_n_genes_per_set") = 5);
m.def("celltypist_project", &py_celltypist_project,
      py::arg("embedding"), py::arg("model"), /* ... */);
m.def("load_celltypist_model", &py_load_celltypist_model,
      py::arg("npz_path"));
```
Returns: `MarkerScoreResult`, `RefMapResult`, `CelltypistModel`.

#### Cycle 13 — fgsea + aucell
```cpp
m.def("fgsea", &py_fgsea,
      py::arg("stats"), py::arg("gene_sets"), py::kw_only(),
      py::arg("min_perm") = 1000, py::arg("max_perm") = 10000,
      py::arg("adaptive_target_pvalue") = 0.05f,
      py::arg("min_set_size") = 15, py::arg("max_set_size") = 500,
      py::arg("seed") = uint64_t{0});
m.def("aucell", &py_aucell, /* similar */);
```
Returns: `FgseaResult`, `AUCellResult`.

#### Cycle 14 — harmony + bbknn
```cpp
m.def("harmony", &py_harmony,
      py::arg("embedding"), py::arg("batch_labels"), py::arg("n_batches"),
      py::kw_only(),
      py::arg("n_clusters") = 20, py::arg("max_iter") = 10,
      py::arg("tol") = 1e-4f, py::arg("lambda") = 1.0f,
      py::arg("seed") = uint64_t{0});
m.def("bbknn", &py_bbknn,
      py::arg("embedding"), py::arg("batch_labels"), py::arg("n_batches"),
      py::kw_only(),
      py::arg("k_within") = 3, py::arg("approx_threshold") = 100000);
```
Returns: `HarmonyResult`, `KnnResult` (BBKNN reuses cycle 8 result).

#### Cycle 15 — velocity_prep_compute
```cpp
m.def("velocity_prep_compute", &py_velocity_prep_compute,
      py::arg("spliced"), py::arg("unspliced"), py::kw_only(),
      py::arg("knn_for_smoothing") = py::none(),
      py::arg("min_S_count") = 10, py::arg("min_U_count") = 5,
      py::arg("top_n_quantile") = 5,
      py::arg("smooth_moments") = true,
      py::arg("compute_velocity") = true);
```
Returns: `VelocityPrepResult`.

#### Cycle 16 — mt_lineage_call_clones
```cpp
m.def("mt_lineage_call_clones", &py_mt_lineage_call_clones,
      py::arg("alt_counts"), py::arg("depth_counts"), py::kw_only(),
      py::arg("min_depth") = 10,
      py::arg("min_cells_alt") = 5,
      py::arg("min_vaf") = 0.01f,
      py::arg("max_em_iters") = 100,
      py::arg("em_tol") = 1e-5f,
      py::arg("min_K") = 2, py::arg("max_K") = 10,
      py::arg("seed") = uint64_t{0});
```
Returns: `ClonePrediction`.

#### Cycle 17 — donor_pseudobulk_de
```cpp
m.def("donor_pseudobulk_de", &py_donor_pseudobulk_de,
      py::arg("mat"), py::arg("cluster_labels"), py::arg("n_clusters"),
      py::arg("donor_labels"), py::arg("n_donors"), py::kw_only(),
      py::arg("min_cells_per_pseudobulk") = 10,
      py::arg("max_irls_iters") = 50,
      py::arg("irls_tol") = 1e-5f,
      py::arg("max_dispersion_iters") = 10,
      py::arg("apeglm_shrinkage") = true,
      py::arg("top_n") = 100,
      py::arg("seed") = uint64_t{0});
```
Returns: `DonorPseudobulkResult`.

### New result classes (~13 in `_bind_results.hpp`)

`StreamingPipelineResult`, `KnnResult`, `LeidenResult`, `UmapResult`, `WilcoxonResult`, `TtestResult`, `ClusterMarkers`, `MarkerScoreResult`, `RefMapResult`, `CelltypistModel`, `FgseaResult`, `AUCellResult`, `HarmonyResult`, `VelocityPrepResult`, `ClonePrediction`, `DonorPseudobulkResult`.

Each result class follows the cycle 20 pattern: a `pybind11::class_<...>` with `py::shared_ptr` lifetime, plus `_view` properties returning Python dicts via `__cuda_array_interface__` for each `DeviceMemory<T>` member.

## Constraints
- **NO host data copies** in any binding.
- pybind11 ≥ 2.11 idioms.
- Each `py_*` function ≤60 LOC.
- Reuse cycle 18-20 patterns (`from_cupy_csr`, `to_cupy_csr`, `__cuda_array_interface__` view properties).
- For functions that take a `std::vector<DeviceCSC>` (graph factorize) or `std::vector<std::string>` (streaming pipeline input paths), use pybind11's `py::list` → `std::vector<...>` automatic conversion.

## Build / test

- No nvcc. Source-only.

## Return format (≤30 lines, exact)

```
## gpu-kernel-dev — cycle 22 (binding extension #2)
Files written:
  - python/src/_bind_results.hpp ({LOC})
Files modified:
  - python/src/_bind_kernels.hpp (+{LOC})
  - python/src/_singlet_gpu_core.cpp (+{LOC} for the result class registration)
Total new: {LOC}
Build: SKIPPED (no nvcc)
Existing tests: SKIPPED
Workspace budget: pure marshal
Streams used: factornet's
Precision: fp32
Determinism: inherits from C++
Self-check: no host copies; result classes use shared_ptr lifetime: CONFIRMED
Bindings exposed: 17 functions + 13 result classes (full coverage of cycles 7-17)
Notes: {1-3 lines}
```

Nothing else.
